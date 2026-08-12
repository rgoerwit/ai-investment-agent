"""Run-scoped ledger of inspected tool evidence available to narrative agents."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from src.tooling.runtime import ToolInvocation, ToolResult

_MAX_CONTENT_CHARS = 20_000
_MAX_RECORDS = 200
_MAX_TOTAL_CONTENT_CHARS = 250_000
_URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
_STATUS_RE = re.compile(r"(?im)^\s*STATUS:\s*([A-Z_]+)\s*$")
_REASON_RE = re.compile(r"(?im)^\s*REASON:\s*([A-Z0-9_]+)\s*$")
_ERROR_ENVELOPE_RE = re.compile(
    r"(?is)(?:"
    r"['\"]error['\"]\s*:"
    r"|(?:error|exception)\s*\("
    r"|(?:status|status_code|http_status)\s*['\"]?\s*:\s*(?:401|403)\b"
    r"|error\s+(?:401|403)\b"
    r"|unauthorized|forbidden"
    r")"
)
_AUTH_ERROR_RE = re.compile(r"(?is)\b(?:401|403|unauthorized|forbidden)\b")

ExecutionStatus = Literal["SUCCEEDED", "FAILED", "BLOCKED", "SKIPPED"]
EvidenceStatus = Literal[
    "EVIDENCE_FOUND",
    "RESULTS_FOUND",
    "COVERAGE_COMPLETE_NO_MATCH",
    "NO_RESULTS",
    "UNAVAILABLE",
    "AUTH_ERROR",
    "INSUFFICIENT",
]
EvidenceAuthority = Literal[
    "PRIMARY_REGISTRY",
    "PRIMARY_ISSUER",
    "SECONDARY",
    "UNSUPPORTED",
]


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    sequence: int
    agent_key: str | None
    tool_name: str
    source: str
    content: str
    content_sha256: str
    requested_urls: tuple[str, ...]
    urls: tuple[str, ...]
    blocked: bool
    findings: tuple[str, ...]
    execution_status: ExecutionStatus
    evidence_status: EvidenceStatus
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvidenceBinding:
    """One fetched source bound to a claim by exact normalized URL."""

    evidence_id: str
    requested_url: str | None
    canonical_url: str
    authority: EvidenceAuthority
    evidence_status: EvidenceStatus
    provider: str | None


def normalize_http_url(value: str) -> str | None:
    """Return a comparable HTTP(S) URL without weakening host validation."""
    candidate = value.strip().rstrip(".,;:!?)]}'")
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    return candidate.rstrip("/")


def _urls_from_value(value: Any) -> tuple[str, ...]:
    text = _bounded_text(value)
    return tuple(
        dict.fromkeys(
            normalized
            for match in _URL_RE.finditer(text)
            if (normalized := normalize_http_url(match.group(0)))
        )
    )


def _requested_urls(args: dict[str, Any]) -> tuple[str, ...]:
    values = [
        value for key, value in args.items() if key.casefold().endswith(("url", "urls"))
    ]
    return tuple(
        dict.fromkeys(url for value in values for url in _urls_from_value(value))
    )


def _canonical_document_url(record: Any) -> str | None:
    content = str(getattr(record, "content", ""))
    urls = tuple(getattr(record, "urls", ()) or ())
    match = re.search(r"(?m)^DOCUMENT_METADATA:\s*(\{.*\})\s*$", content)
    if match:
        try:
            metadata = json.loads(match.group(1))
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        source_url = normalize_http_url(str(metadata.get("source_url") or ""))
        if source_url:
            return source_url
    return urls[0] if len(urls) == 1 else None


def resolve_evidence_authority(
    *,
    tool_name: str,
    evidence_status: EvidenceStatus,
    urls: tuple[str, ...],
) -> EvidenceAuthority:
    """Resolve evidence authority once from source kind and validated URLs."""
    if evidence_status != "EVIDENCE_FOUND":
        return (
            "SECONDARY"
            if urls and evidence_status == "RESULTS_FOUND"
            else "UNSUPPORTED"
        )
    if tool_name == "get_official_filings":
        return "PRIMARY_REGISTRY"

    from src.tools.official_documents import resolve_source_authority

    authorities = tuple(resolve_source_authority(url) for url in urls)
    if "PRIMARY_REGISTRY" in authorities:
        return "PRIMARY_REGISTRY"
    if "PRIMARY_ISSUER" in authorities:
        return "PRIMARY_ISSUER"
    if "SECONDARY" in authorities:
        return "SECONDARY"
    return "UNSUPPORTED"


def find_fetched_evidence_record(
    records: list[Any] | tuple[Any, ...],
    source_url: str | None,
) -> Any | None:
    """Return the inspected fetched record for an exact requested/result URL."""
    normalized = normalize_http_url(source_url or "")
    if not normalized:
        return None
    for record in records:
        if (
            bool(getattr(record, "blocked", False))
            or getattr(record, "evidence_status", None) != "EVIDENCE_FOUND"
        ):
            continue
        requested_urls = tuple(getattr(record, "requested_urls", ()) or ())
        result_urls = tuple(getattr(record, "urls", ()) or ())
        known_urls = {*requested_urls, *result_urls}
        if normalized not in known_urls:
            continue
        return record
    return None


def bind_fetched_evidence(
    records: list[Any] | tuple[Any, ...],
    source_url: str | None,
) -> EvidenceBinding | None:
    """Bind only inspected fetched evidence, never a search-results mention."""
    normalized = normalize_http_url(source_url or "")
    record = find_fetched_evidence_record(records, source_url)
    if not normalized or record is None:
        return None
    requested_urls = tuple(getattr(record, "requested_urls", ()) or ())
    canonical_url = _canonical_document_url(record) or normalized
    authority = resolve_evidence_authority(
        tool_name=str(getattr(record, "tool_name", "")),
        evidence_status="EVIDENCE_FOUND",
        urls=(canonical_url,),
    )
    return EvidenceBinding(
        evidence_id=(
            f"evidence:{getattr(record, 'sequence', 0)}:"
            f"{str(getattr(record, 'content_sha256', ''))[:12]}"
        ),
        requested_url=(normalized if normalized in requested_urls else None),
        canonical_url=canonical_url,
        authority=authority,
        evidence_status="EVIDENCE_FOUND",
        provider=str(getattr(record, "tool_name", "")) or None,
    )


def _bounded_text(value: Any) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict | list | tuple):
        text = json.dumps(
            value,
            ensure_ascii=False,
            default=lambda item: f"<{type(item).__name__}>",
            sort_keys=True,
        )
    elif value is None or isinstance(value, int | float | bool):
        text = str(value)
    else:
        text = f"<{type(value).__name__}>"
    if len(text) <= _MAX_CONTENT_CHARS:
        return text
    return text[:_MAX_CONTENT_CHARS] + "\n...[evidence ledger truncated]"


def classify_evidence_value(
    tool_name: str,
    value: Any,
    *,
    blocked: bool = False,
) -> tuple[ExecutionStatus, EvidenceStatus, str | None, str]:
    """Classify one inspected tool result without trusting its outer status alone."""
    content = _bounded_text(value)
    if blocked:
        return "BLOCKED", "UNAVAILABLE", "TOOL_BLOCKED", content

    status_match = _STATUS_RE.search(content)
    reason_match = _REASON_RE.search(content)
    status = status_match.group(1).upper() if status_match else ""
    reason = reason_match.group(1).upper() if reason_match else None

    # Some adapters return STATUS: EVIDENCE_FOUND around a nested provider error.
    # Error semantics override that optimistic wrapper.
    error_blocks = [
        block
        for block in re.findall(r"(?is)<result\b[^>]*>.*?</result>", content)
        if _ERROR_ENVELOPE_RE.search(block)
    ]
    valid_blocks = [
        block
        for block in re.findall(r"(?is)<result\b[^>]*>.*?</result>", content)
        if not _ERROR_ENVELOPE_RE.search(block)
        and ("<url>" in block.casefold() or "<content>" in block.casefold())
    ]
    if error_blocks and valid_blocks:
        partial_status: EvidenceStatus = (
            "EVIDENCE_FOUND" if status == "EVIDENCE_FOUND" else "RESULTS_FOUND"
        )
        return "SUCCEEDED", partial_status, "PARTIAL_PROVIDER_ERROR", content
    if _ERROR_ENVELOPE_RE.search(content):
        error_status: EvidenceStatus = (
            "AUTH_ERROR" if _AUTH_ERROR_RE.search(content) else "INSUFFICIENT"
        )
        return "SUCCEEDED", error_status, reason or "EMBEDDED_PROVIDER_ERROR", content

    explicit: dict[str, EvidenceStatus] = {
        "EVIDENCE_FOUND": "EVIDENCE_FOUND",
        "RESULTS_FOUND": "RESULTS_FOUND",
        "COVERAGE_COMPLETE_NO_MATCH": "COVERAGE_COMPLETE_NO_MATCH",
        "NO_RESULTS": "NO_RESULTS",
        "UNAVAILABLE": "UNAVAILABLE",
        "AUTH_ERROR": "AUTH_ERROR",
    }
    if status in explicit:
        return "SUCCEEDED", explicit[status], reason, content
    if status == "INSUFFICIENT_DATA":
        if reason and ("AUTH" in reason or reason in {"UNAUTHORIZED", "FORBIDDEN"}):
            evidence_status: EvidenceStatus = "AUTH_ERROR"
        elif reason and ("UNAVAILABLE" in reason or reason == "NO_ADAPTER"):
            evidence_status = "UNAVAILABLE"
        elif reason in {"NO_RESULTS", "NO_CANDIDATE_URLS"}:
            evidence_status = "NO_RESULTS"
        else:
            evidence_status = "INSUFFICIENT"
        return "SUCCEEDED", evidence_status, reason, content
    if "DOCUMENT_METADATA:" in content:
        return "SUCCEEDED", "EVIDENCE_FOUND", reason, content
    if "<search_results" in content or "Foreign Source Search Results" in content:
        return "SUCCEEDED", "RESULTS_FOUND", reason, content
    if tool_name == "get_official_filings" and content.strip():
        return "SUCCEEDED", "EVIDENCE_FOUND", reason, content
    if not content.strip():
        return "SUCCEEDED", "NO_RESULTS", reason, ""
    return "SUCCEEDED", "RESULTS_FOUND", reason, content


class EvidenceRecorder:
    """Record the final post-hook result without mutating tool execution."""

    def __init__(self) -> None:
        self._records: list[EvidenceRecord] = []
        self._dedupe_keys: set[tuple[str | None, str, str, tuple[str, ...]]] = set()
        self._content_chars = 0
        self._overflowed = False

    def _record_overflow(self) -> None:
        """Append a single durable marker when the ledger hits its capacity.

        Without this the recorder silently drops evidence past the cap, leaving
        no trace that the ledger is incomplete. Idempotent — one marker per run.
        """
        if self._overflowed:
            return
        self._overflowed = True
        self._records.append(
            EvidenceRecord(
                sequence=len(self._records) + 1,
                agent_key=None,
                tool_name="__ledger_overflow__",
                source="evidence_recorder",
                content="",
                content_sha256="",
                requested_urls=(),
                urls=(),
                blocked=False,
                findings=("EVIDENCE_LEDGER_CAPACITY_REACHED",),
                execution_status="SKIPPED",
                evidence_status="UNAVAILABLE",
                reason="LEDGER_OVERFLOW",
            )
        )

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        content = _bounded_text(result.value)
        execution_status, evidence_status, reason, _ = classify_evidence_value(
            call.name,
            content,
            blocked=result.blocked,
        )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        requested_urls = _requested_urls(call.args)
        # Include the requested-URL identity in the dedupe key: byte-identical
        # content fetched from two DIFFERENT source URLs is two corroborating
        # sources, not a duplicate, and must keep both identities.
        dedupe_key = (call.agent_key, call.name, digest, requested_urls)
        if dedupe_key in self._dedupe_keys:
            return result
        if (
            len(self._records) >= _MAX_RECORDS
            or self._content_chars + len(content) > _MAX_TOTAL_CONTENT_CHARS
        ):
            self._record_overflow()
            return result
        self._dedupe_keys.add(dedupe_key)
        urls = tuple(
            dict.fromkeys(
                normalized
                for match in _URL_RE.finditer(content)
                if (normalized := normalize_http_url(match.group(0)))
            )
        )
        self._records.append(
            EvidenceRecord(
                sequence=len(self._records) + 1,
                agent_key=call.agent_key,
                tool_name=call.name,
                source=call.source,
                content=content,
                content_sha256=digest,
                requested_urls=requested_urls,
                urls=urls,
                blocked=result.blocked,
                findings=tuple(result.findings or ()),
                execution_status=execution_status,
                evidence_status=evidence_status,
                reason=reason,
            )
        )
        self._content_chars += len(content)
        return result

    def snapshot(self, *, agent_key: str | None = None) -> list[EvidenceRecord]:
        if agent_key is None:
            return list(self._records)
        return [record for record in self._records if record.agent_key == agent_key]

    def serialized_snapshot(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._records]
