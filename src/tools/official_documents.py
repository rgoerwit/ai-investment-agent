"""Bounded extraction of text from official exchange disclosure documents."""

from __future__ import annotations

import asyncio
import io
import ipaddress
import json
import re
import shutil
import socket
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urljoin, urlparse, urlunsplit

import httpx
import structlog
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from src.blocking_io import BlockingCallPolicy, run_blocking_call
from src.config import config
from src.error_safety import summarize_exception
from src.exchange_metadata import registered_official_document_hosts
from src.forensic_budget import AuditorBudgetPolicy
from src.runtime_services import (
    get_current_inspection_service,
    get_current_issuer_hosts,
)
from src.tooling.inspector import InspectionEnvelope, SourceKind

logger = structlog.get_logger(__name__)

SourceAuthority = Literal[
    "PRIMARY_REGISTRY",
    "PRIMARY_ISSUER",
    "SECONDARY",
    "UNSUPPORTED",
]

_OFFICIAL_HOST_SUFFIXES = (
    "sec.gov",
    *registered_official_document_hosts(),
)
_PDF_POLICY = BlockingCallPolicy("official_document_pdf_extract", 20.0)
_DNS_POLICY = BlockingCallPolicy("official_document_dns_check", 5.0)
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
_DISCLOSURE_LINK_TERMS = (
    "result",
    "earnings",
    "financial",
    "quarter",
    "annual",
    "guidance",
    "presentation",
    "report",
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


def _rank_same_host_document_paths(
    soup: BeautifulSoup,
    base_url: str,
    keywords: str,
    *,
    limit: int = 8,
) -> tuple[str, ...]:
    """Return ranked relative paths without turning child links into evidence URLs."""
    base = urlparse(base_url)
    if not base.hostname:
        return ()
    terms = {
        *(_DISCLOSURE_LINK_TERMS),
        *(term.strip().casefold() for term in keywords.split(",") if term.strip()),
    }
    ranked: dict[str, int] = {}
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        if not href:
            continue
        candidate = urlparse(urljoin(base_url, href))
        try:
            port = candidate.port
        except ValueError:
            continue
        if (
            candidate.scheme != "https"
            or candidate.hostname != base.hostname
            or candidate.username
            or candidate.password
            or port not in {None, 443}
        ):
            continue
        path = urlunsplit(("", "", candidate.path or "/", candidate.query, ""))
        if path == urlunsplit(("", "", base.path or "/", base.query, "")):
            continue
        searchable = f"{anchor.get_text(' ', strip=True)} {candidate.path}".casefold()
        score = sum(3 for term in terms if term in searchable)
        score += 2 if re.search(r"\b20\d{2}\b", searchable) else 0
        score += 2 if candidate.path.casefold().endswith(".pdf") else 0
        if score:
            ranked[path] = max(score, ranked.get(path, 0))
    return tuple(
        path
        for path, _ in sorted(
            ranked.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    )


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


def is_official_document_url(url: str) -> bool:
    """Return whether a URL is allowed by registered or operator-added policy."""
    extra_hosts = (
        *_configured_host_suffixes(config.auditor_official_document_hosts),
        *get_current_issuer_hosts(),
    )
    return _official_url(url, extra_hosts)


def resolve_source_authority(url: str | None) -> SourceAuthority:
    """Classify URL authority from the single run-scoped trust policy."""
    if not url:
        return "UNSUPPORTED"
    configured_hosts = _configured_host_suffixes(config.auditor_official_document_hosts)
    if _official_url(url):
        return "PRIMARY_REGISTRY"
    if _official_url(url, (*configured_hosts, *get_current_issuer_hosts())):
        return "PRIMARY_ISSUER"
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        return "SECONDARY"
    return "UNSUPPORTED"


async def _ensure_public_hostname(url: str) -> None:
    host = urlparse(url).hostname
    if not host:
        raise DocumentExtractionError("DOCUMENT_HOST_INVALID")

    def _resolve() -> set[str]:
        return {
            item[4][0]
            for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        }

    try:
        addresses = await run_blocking_call(_DNS_POLICY, _resolve)
    except Exception as exc:
        raise DocumentExtractionError("DOCUMENT_DNS_FAILED") from exc
    if not addresses:
        raise DocumentExtractionError("DOCUMENT_DNS_FAILED")
    for raw_address in addresses:
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as exc:
            raise DocumentExtractionError("DOCUMENT_DNS_INVALID") from exc
        if not address.is_global:
            raise DocumentExtractionError("DOCUMENT_PRIVATE_ADDRESS")


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
            await _ensure_public_hostname(current)
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
    ticker: str = "",
    company_name: str = "",
) -> str:
    """Download and extract bounded evidence from an approved official URL."""
    policy = AuditorBudgetPolicy.from_settings()
    configured_hosts = _configured_host_suffixes(config.auditor_official_document_hosts)
    extra_hosts = (*configured_hosts, *get_current_issuer_hosts())
    if not is_official_document_url(url):
        return "STATUS: INSUFFICIENT_DATA\nREASON: UNAPPROVED_DOCUMENT_HOST"
    try:
        payload, content_type, final_url = await _download_official(
            url, policy.max_document_bytes, extra_hosts
        )
        is_pdf = payload.startswith(b"%PDF") or "application/pdf" in content_type
        if is_pdf:
            extracted = await extract_pdf_text(payload, policy, keywords)
            candidate_paths: tuple[str, ...] = ()
        elif "html" in content_type or payload.lstrip().startswith(b"<"):
            soup = BeautifulSoup(payload, "html.parser")
            candidate_paths = _rank_same_host_document_paths(
                soup,
                final_url,
                keywords,
            )
            text = soup.get_text("\n", strip=True)
            extracted = ExtractedDocument(text, "html", 1, (0,))
        elif content_type.startswith("text/"):
            candidate_paths = ()
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
            "authority": resolve_source_authority(final_url),
            "reason": reason,
            "source_url": final_url,
            "backend": extracted.backend,
            "bytes_downloaded": len(payload),
            "pages_total": extracted.pages_total,
            "pages_selected": [page + 1 for page in extracted.pages_selected],
            "evidence_chars": len(str(inspected)),
            "candidate_paths": list(candidate_paths),
        }
        return (
            "STATUS: EVIDENCE_FOUND\n"
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
