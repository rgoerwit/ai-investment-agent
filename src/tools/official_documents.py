"""Bounded extraction of text from official exchange disclosure documents."""

from __future__ import annotations

import asyncio
import io
import ipaddress
import json
import re
import shutil
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
import structlog
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from src.blocking_io import BlockingCallPolicy, run_blocking_call
from src.config import config
from src.error_safety import summarize_exception
from src.forensic_budget import AuditorBudgetPolicy
from src.runtime_services import get_current_inspection_service
from src.tooling.inspector import InspectionEnvelope, SourceKind

logger = structlog.get_logger(__name__)

_OFFICIAL_HOST_SUFFIXES = (
    "sgx.com",
    "hkexnews.hk",
    "hkex.com.hk",
    "jpx.co.jp",
    "dart.fss.or.kr",
    "edinet-fsa.go.jp",
    "companieshouse.gov.uk",
    "sec.gov",
)
_PDF_POLICY = BlockingCallPolicy("official_document_pdf_extract", 20.0)
_DOCUMENT_TIMEOUT_SECONDS = 20.0
_MIN_USEFUL_TEXT_CHARS = 200
_DEFAULT_KEYWORDS = (
    "statement of financial position",
    "balance sheet",
    "income statement",
    "statement of cash flows",
    "cash flow statement",
    "independent auditor",
    "audit opinion",
    "재무상태표",
    "손익계산서",
    "현금흐름표",
    "貸借対照表",
    "損益計算書",
    "キャッシュフロー",
)


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    backend: str
    pages_total: int
    pages_selected: tuple[int, ...]
    reason: str | None = None


class DocumentExtractionError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _configured_host_suffixes(value: str) -> tuple[str, ...]:
    hosts: list[str] = []
    for raw in value.split(","):
        host = raw.strip().rstrip(".").lower()
        if not host:
            continue
        if (
            "." not in host
            or len(host) > 253
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", host)
        ):
            logger.warning("auditor_document_host_ignored", host=host)
            continue
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            logger.warning("auditor_document_host_ignored", host=host)
            continue
        hosts.append(host)
    return tuple(dict.fromkeys(hosts))


def _official_url(url: str, extra_hosts: tuple[str, ...] = ()) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    try:
        port = parsed.port
    except ValueError:
        return False
    if parsed.username or parsed.password or port not in {None, 443}:
        return False
    host = parsed.hostname.rstrip(".").lower()
    approved = (*_OFFICIAL_HOST_SUFFIXES, *extra_hosts)
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in approved)


def _select_pages(
    page_texts: list[str], keywords: tuple[str, ...], maximum: int
) -> tuple[int, ...]:
    terms = tuple(term.casefold() for term in keywords if term.strip())
    scores: list[tuple[int, int]] = []
    for index, text in enumerate(page_texts):
        folded = text.casefold()
        keyword_score = sum(20 for term in terms if term in folded)
        numeric_density = min(10, len(re.findall(r"\d[\d,.]*", text)) // 20)
        scores.append((keyword_score + numeric_density, index))
    ranked = [
        index for _, index in sorted(scores, key=lambda item: (-item[0], item[1]))
    ]
    selected = sorted({0, *ranked[:maximum]})[:maximum]
    return tuple(selected)


def _extract_with_pypdf(
    payload: bytes, *, max_pages: int, selected_pages: int, keywords: tuple[str, ...]
) -> ExtractedDocument:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(payload), strict=False)
    pages_total = len(reader.pages)
    if pages_total == 0:
        raise DocumentExtractionError("PDF_MALFORMED")
    scan_count = min(pages_total, max_pages)
    page_texts: list[str] = []
    for page in reader.pages[:scan_count]:
        try:
            page_texts.append(page.extract_text() or "")
        except Exception:
            page_texts.append("")
    selected = _select_pages(page_texts, keywords, selected_pages)
    text = "\n\n".join(
        f"[PDF PAGE {index + 1}]\n{page_texts[index]}" for index in selected
    ).strip()
    reason = "DOCUMENT_PAGE_LIMIT" if pages_total > max_pages else None
    return ExtractedDocument(text, "pypdf", pages_total, selected, reason)


async def _extract_with_pdftotext(
    payload: bytes, *, max_pages: int, selected_pages: int, keywords: tuple[str, ...]
) -> ExtractedDocument:
    executable = shutil.which("pdftotext")
    if executable is None:
        raise DocumentExtractionError("PDF_PARSER_UNAVAILABLE")
    process = await asyncio.create_subprocess_exec(
        executable,
        "-f",
        "1",
        "-l",
        str(max_pages),
        "-",
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(payload), timeout=_PDF_POLICY.hard_timeout_seconds
        )
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise DocumentExtractionError("EXTRACTION_TIMEOUT") from exc
    if process.returncode != 0:
        raise DocumentExtractionError("PDF_MALFORMED")
    page_texts = stdout.decode("utf-8", errors="replace").split("\f")
    if page_texts and not page_texts[-1].strip():
        page_texts.pop()
    selected = _select_pages(page_texts, keywords, selected_pages)
    text = "\n\n".join(
        f"[PDF PAGE {index + 1}]\n{page_texts[index]}" for index in selected
    ).strip()
    return ExtractedDocument(text, "pdftotext", len(page_texts), selected)


async def extract_pdf_text(
    payload: bytes, policy: AuditorBudgetPolicy, keywords: str
) -> ExtractedDocument:
    terms = tuple(filter(None, (part.strip() for part in keywords.split(","))))
    search_terms = terms or _DEFAULT_KEYWORDS
    primary_reason = "PDF_PARSER_UNAVAILABLE"
    try:
        extracted = await run_blocking_call(
            _PDF_POLICY,
            lambda: _extract_with_pypdf(
                payload,
                max_pages=policy.max_document_pages,
                selected_pages=policy.max_selected_pages,
                keywords=search_terms,
            ),
        )
        if len(extracted.text) >= _MIN_USEFUL_TEXT_CHARS:
            return extracted
        primary_reason = "PDF_TEXT_UNAVAILABLE"
    except DocumentExtractionError as exc:
        primary_reason = exc.reason
    except ModuleNotFoundError:
        primary_reason = "PDF_PARSER_UNAVAILABLE"
    except TimeoutError:
        primary_reason = "EXTRACTION_TIMEOUT"
    except Exception:
        primary_reason = "PDF_MALFORMED"

    try:
        fallback = await _extract_with_pdftotext(
            payload,
            max_pages=policy.max_document_pages,
            selected_pages=policy.max_selected_pages,
            keywords=search_terms,
        )
        if len(fallback.text) >= _MIN_USEFUL_TEXT_CHARS:
            return fallback
        raise DocumentExtractionError("PDF_TEXT_UNAVAILABLE")
    except DocumentExtractionError as exc:
        reason = (
            exc.reason if primary_reason == "PDF_PARSER_UNAVAILABLE" else primary_reason
        )
        raise DocumentExtractionError(reason) from exc


async def _download_official(
    url: str, max_bytes: int, extra_hosts: tuple[str, ...] = ()
) -> tuple[bytes, str, str]:
    current = url
    async with httpx.AsyncClient(timeout=_DOCUMENT_TIMEOUT_SECONDS) as client:
        for _ in range(4):
            if not _official_url(current, extra_hosts):
                raise DocumentExtractionError("UNAPPROVED_DOCUMENT_HOST")
            async with client.stream(
                "GET", current, follow_redirects=False
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise DocumentExtractionError("DOCUMENT_REDIRECT_INVALID")
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    raise DocumentExtractionError("DOCUMENT_HTTP_ERROR")
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise DocumentExtractionError("DOCUMENT_SIZE_LIMIT")
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise DocumentExtractionError("DOCUMENT_SIZE_LIMIT")
                    chunks.append(chunk)
                return (
                    b"".join(chunks),
                    response.headers.get("content-type", ""),
                    current,
                )
    raise DocumentExtractionError("DOCUMENT_REDIRECT_LIMIT")


@tool
async def get_official_document(
    url: str,
    keywords: str = "",
) -> str:
    """Download and extract bounded evidence from an approved official URL."""
    policy = AuditorBudgetPolicy.from_settings()
    extra_hosts = _configured_host_suffixes(config.auditor_official_document_hosts)
    if not _official_url(url, extra_hosts):
        return "STATUS: INSUFFICIENT_DATA\nREASON: UNAPPROVED_DOCUMENT_HOST"
    try:
        payload, content_type, final_url = await _download_official(
            url, policy.max_document_bytes, extra_hosts
        )
        is_pdf = payload.startswith(b"%PDF") or "application/pdf" in content_type
        if is_pdf:
            extracted = await extract_pdf_text(payload, policy, keywords)
        elif "html" in content_type or payload.lstrip().startswith(b"<"):
            soup = BeautifulSoup(payload, "html.parser")
            text = soup.get_text("\n", strip=True)
            extracted = ExtractedDocument(text, "html", 1, (0,))
        elif content_type.startswith("text/"):
            extracted = ExtractedDocument(
                payload.decode("utf-8", errors="replace"), "text", 1, (0,)
            )
        else:
            raise DocumentExtractionError("UNSUPPORTED_DOCUMENT_TYPE")

        if len(extracted.text) < _MIN_USEFUL_TEXT_CHARS:
            raise DocumentExtractionError(
                "PDF_TEXT_UNAVAILABLE" if is_pdf else "DOCUMENT_TEXT_UNAVAILABLE"
            )
        evidence = extracted.text[: policy.max_evidence_chars]
        reason = extracted.reason or (
            "EVIDENCE_CHAR_LIMIT" if len(extracted.text) > len(evidence) else None
        )
        inspected = await get_current_inspection_service().check(
            InspectionEnvelope(
                content_text=evidence,
                raw_content=evidence,
                source_kind=SourceKind.official_filing,
                source_name="official_document",
                metadata={"url": final_url, "backend": extracted.backend},
            )
        )
        metadata = {
            "status": "PARTIAL_DATA" if reason else "EXTRACTED",
            "reason": reason,
            "source_url": final_url,
            "backend": extracted.backend,
            "bytes_downloaded": len(payload),
            "pages_total": extracted.pages_total,
            "pages_selected": [page + 1 for page in extracted.pages_selected],
            "evidence_chars": len(str(inspected)),
        }
        return (
            f"DOCUMENT_METADATA: {json.dumps(metadata, sort_keys=True)}\n\n{inspected}"
        )
    except DocumentExtractionError as exc:
        return f"STATUS: INSUFFICIENT_DATA\nREASON: {exc.reason}"
    except Exception as exc:
        logger.warning(
            "official_document_failed",
            **summarize_exception(exc, operation="official_document_failed"),
        )
        return "STATUS: INSUFFICIENT_DATA\nREASON: DOCUMENT_RETRIEVAL_FAILED"
