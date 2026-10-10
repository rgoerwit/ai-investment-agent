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
from typing import Any, Literal
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
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import require_allowance
from src.runtime_services import (
    get_current_inspection_service,
    get_current_issuer_hosts,
)
from src.text_patterns import is_safe_public_host
from src.tooling.inspector import InspectionEnvelope, SourceKind
from src.tooling.runtime import tool_execution_timeout_seconds

logger = structlog.get_logger(__name__)

# Detached reapers of killed extraction subprocesses, held so one that outlives
# its bounded wait is neither garbage-collected nor left with an unobserved
# exception (_reap_finished logs it).
_PENDING_REAPS: set[asyncio.Future[Any]] = set()

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
# Each policy's number is only the shipped default of its setting (pinned by a
# parity test): each call applies OFFICIAL_DOCUMENT_EXTRACT_TIMEOUT_SECONDS or
# OFFICIAL_DOCUMENT_DNS_TIMEOUT_SECONDS with with_timeout(), read at the call
# and shrunk to what is left (see "Allocation" below).
_PDF_POLICY = BlockingCallPolicy("official_document_pdf_extract", 20.0)
_DNS_POLICY = BlockingCallPolicy("official_document_dns_check", 5.0)
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


# --- Allocation --------------------------------------------------------------
#
# The whole sequence (DNS check, download, pypdf, pdftotext, inspection) runs
# inside the tool's allowance: the runner's setting, clamped to the operational
# deadline (the caller's enclosing deadline and the tool runner, kept back from
# the runner's hard cap by a margin). It is shared by fractions of that
# allowance, so the split holds for any runner value:
#
# - the download (DNS checks and redirects included) may use
#   OFFICIAL_DOCUMENT_DOWNLOAD_FRACTION of it;
# - of the rest, OFFICIAL_DOCUMENT_INSPECTION_FRACTION is held back for the
#   content inspection of the evidence; extraction gets the remainder, each
#   backend at most OFFICIAL_DOCUMENT_EXTRACT_TIMEOUT_SECONDS, and one that
#   cannot start is skipped with EXTRACTION_TIMEOUT;
# - inspection gets whatever is then left, and one that cannot finish fails the
#   document closed with INSPECTION_TIMEOUT.
#
# Read, never bound here: binding the LLM enclosing deadline would also cap
# the injection judge, which fails open.


@dataclass(frozen=True)
class _Allocation:
    """One document's time budget: an absolute deadline and how it is shared."""

    deadline: float
    download_seconds: float
    inspection_reserve: float

    def left(self) -> float:
        return self.deadline - llm_timeouts.now()


def _allocate() -> _Allocation:
    """Share the tool's allowance between download, extraction and inspection.

    The allowance is read once, when the tool starts. A spent one leaves every
    share at or below zero, so the download refuses before a client opens.
    """
    allowance = tool_execution_timeout_seconds()
    download = allowance * config.official_document_download_fraction
    return _Allocation(
        deadline=llm_timeouts.now() + allowance,
        download_seconds=download,
        inspection_reserve=(allowance - download)
        * config.official_document_inspection_fraction,
    )


def _extraction_allowance(allocation: _Allocation, *, label: str) -> float:
    """One extractor's bound: its cap, within what is left less the inspection
    reserve. Raises ``TimeoutError`` before any work once nothing is left."""
    return require_allowance(
        min(
            config.official_document_extract_timeout_seconds,
            allocation.left() - allocation.inspection_reserve,
        ),
        label=label,
    )


def _reap_finished(reap: asyncio.Future[Any]) -> None:
    """Done-callback of a reaper: forget it, and log (never leak) its failure."""
    _PENDING_REAPS.discard(reap)
    if reap.cancelled():
        logger.warning("official_document_subprocess_reap_cancelled")
        return
    exc = reap.exception()
    if exc is not None:
        logger.warning(
            "official_document_subprocess_reap_failed",
            **summarize_exception(exc, operation="official_document_subprocess_reap"),
        )


async def _kill_and_reap(process: asyncio.subprocess.Process) -> None:
    """Kill an extraction child and reap it, the wait bounded and shielded.

    Runs from an abnormal exit (timeout, error, cancellation). The reap is a
    separate task behind ``asyncio.shield``, so a second cancellation of the
    caller interrupts only the wait: the reap still finishes and its outcome
    is logged by ``_reap_finished``. Past SUBPROCESS_CLEANUP_TIMEOUT_SECONDS
    the caller stops waiting and the reap continues detached.
    """
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass  # It exited between the check and the kill; reap it anyway.
    reap = asyncio.ensure_future(process.wait())
    _PENDING_REAPS.add(reap)
    reap.add_done_callback(_reap_finished)
    try:
        async with asyncio.timeout(config.subprocess_cleanup_timeout_seconds):
            await asyncio.shield(reap)
    except TimeoutError:
        logger.warning(
            "official_document_subprocess_reap_pending",
            timeout_seconds=config.subprocess_cleanup_timeout_seconds,
        )
    except Exception:
        pass  # The reap failed; _reap_finished has logged it.


async def _communicate_bounded(
    process: asyncio.subprocess.Process, payload: bytes, *, timeout: float
) -> bytes:
    """``communicate()`` within ``timeout``; any abnormal exit kills and reaps.

    The original exception (``TimeoutError``, an error, or the caller's
    cancellation) is re-raised after the cleanup.
    """
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(payload), timeout=timeout
        )
    except BaseException:
        await _kill_and_reap(process)
        raise
    return stdout


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
        if not is_safe_public_host(host):
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
        addresses: set[str] = set()
        for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
            raw_address = item[4][0]
            if not isinstance(raw_address, str):
                raise TypeError("DNS resolver returned a non-string address")
            addresses.add(raw_address)
        return addresses

    try:
        # A spent deadline raises TimeoutError in run_blocking_call, before
        # any lookup, and fails the document exactly as a DNS timeout does.
        policy = _DNS_POLICY.with_timeout(config.official_document_dns_timeout_seconds)
        addresses = await run_blocking_call(policy, _resolve)
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
    payload: bytes,
    *,
    max_pages: int,
    selected_pages: int,
    keywords: tuple[str, ...],
    allocation: _Allocation | None = None,
) -> ExtractedDocument:
    executable = shutil.which("pdftotext")
    if executable is None:
        raise DocumentExtractionError("PDF_PARSER_UNAVAILABLE")
    # Read now (pypdf may have used part of the allowance) and resolved before
    # the subprocess exists, so a spent allowance spawns nothing. With no
    # allocation (a direct caller) the tool's own is taken now.
    try:
        timeout = _extraction_allowance(
            allocation or _allocate(), label="official_document_pdftotext"
        )
    except TimeoutError as exc:
        logger.info(
            "official_document_extraction_skipped",
            backend="pdftotext",
            reason="EXTRACTION_TIMEOUT",
        )
        raise DocumentExtractionError("EXTRACTION_TIMEOUT") from exc
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
        stdout = await _communicate_bounded(process, payload, timeout=timeout)
    except TimeoutError as exc:
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
    payload: bytes,
    policy: AuditorBudgetPolicy,
    keywords: str,
    allocation: _Allocation | None = None,
) -> ExtractedDocument:
    """Extract evidence text: pypdf, then pdftotext, inside *allocation*.

    With no allocation (a direct caller) the tool's own is taken now.
    """
    allocation = allocation or _allocate()
    terms = tuple(filter(None, (part.strip() for part in keywords.split(","))))
    search_terms = terms or _DEFAULT_KEYWORDS
    primary_reason = "PDF_PARSER_UNAVAILABLE"
    try:
        # A spent allowance raises TimeoutError before the thread starts and
        # takes the EXTRACTION_TIMEOUT path below, as a pypdf stall does;
        # pdftotext then re-checks what is left.
        pdf_policy = _PDF_POLICY.with_timeout(
            _extraction_allowance(allocation, label=_PDF_POLICY.label)
        )
        extracted = await run_blocking_call(
            pdf_policy,
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
            allocation=allocation,
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
    url: str,
    max_bytes: int,
    extra_hosts: tuple[str, ...] = (),
    *,
    allocation: _Allocation,
) -> tuple[bytes, str, str]:
    current = url
    # One bound for the whole download (DNS checks and redirects included):
    # its share of the allocation, so a trickling body stops there. httpx gets
    # the same value per phase; a tighter per-phase bound would only fail a
    # stall sooner with nothing to spend the time on. A spent share raises
    # TimeoutError before the client opens, which get_official_document
    # reports as DOCUMENT_RETRIEVAL_FAILED, as it does an httpx timeout.
    total = require_allowance(
        allocation.download_seconds, label="official_document_download"
    )
    async with (
        asyncio.timeout(total),
        httpx.AsyncClient(timeout=total) as client,
    ):
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


def _html_document_text(soup: BeautifulSoup) -> str:
    """Keep the established flattened-cell grammar used by evidence readers."""
    return soup.get_text("\n", strip=True)


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
        approved = sorted({*_OFFICIAL_HOST_SUFFIXES, *extra_hosts})
        rejected_host = urlparse(url).hostname or "unknown"
        return (
            "STATUS: INSUFFICIENT_DATA\n"
            "REASON: UNAPPROVED_DOCUMENT_HOST\n"
            f"REJECTED_HOST: {rejected_host}\n"
            f"APPROVED_HOSTS: {', '.join(approved) if approved else 'none'}"
        )
    allocation = _allocate()
    try:
        payload, content_type, final_url = await _download_official(
            url, policy.max_document_bytes, extra_hosts, allocation=allocation
        )
        is_pdf = payload.startswith(b"%PDF") or "application/pdf" in content_type
        if is_pdf:
            extracted = await extract_pdf_text(
                payload, policy, keywords, allocation=allocation
            )
            candidate_paths: tuple[str, ...] = ()
        elif "html" in content_type or payload.lstrip().startswith(b"<"):
            soup = BeautifulSoup(payload, "html.parser")
            candidate_paths = _rank_same_host_document_paths(
                soup,
                final_url,
                keywords,
            )
            text = _html_document_text(soup)
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
        # Whatever is left, at least the reserve held back from extraction.
        # Fails closed: evidence that could not be inspected in time is
        # never returned.
        inspected = await get_current_inspection_service().check_within(
            InspectionEnvelope(
                content_text=evidence,
                raw_content=evidence,
                source_kind=SourceKind.official_filing,
                source_name="official_document",
                metadata={"url": final_url, "backend": extracted.backend},
            ),
            seconds=allocation.left(),
            label="official_document_inspection",
        )
        if inspected is None:
            logger.warning(
                "official_document_inspection_timeout", backend=extracted.backend
            )
            raise DocumentExtractionError("INSPECTION_TIMEOUT")
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
