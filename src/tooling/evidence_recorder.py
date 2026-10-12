"""Run-scoped ledger of inspected tool evidence available to narrative agents."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from src.text_patterns import RESULT_ENVELOPE_RE, URL_RE
from src.tooling.runtime import ToolInvocation, ToolResult, tool_error_message

_MAX_CONTENT_CHARS = 20_000
_MAX_RECORDS = 200
_MAX_TOTAL_CONTENT_CHARS = 250_000
_URL_RE = URL_RE
# Six variants of this line grammar exist (here, agents/forensic_repair,
# agents/capital_structure, agents/management_guidance, agents/consultant_nodes,
# graph/routing) differing on leading-whitespace tolerance and the `:`/`=` separator.
# Measured inert -- across 2,191 reports containing "STATUS:" the tolerant and strict
# spellings disagree once each way -- so they are left separate rather than merged into
# a reader none of the six quite wants. Byte-identical twin: agents/forensic_repair.
# Widen one and you probably want to widen that twin too.
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

# Tools that retain an attempted document acquisition, including failed/blocked calls.
DOCUMENT_ACQUISITION_TOOLS = frozenset(
    {"get_official_document", "extract_guidance_sources", "get_official_filings"}
)

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
    original_content_chars: int | None = None
    retained_content_chars: int | None = None
    content_truncated: bool = False
    # Resolved by the recorder after inspection, never from document metadata.
    authority_by_url: tuple[tuple[str, EvidenceAuthority], ...] = ()

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


def evidence_record_id(record: Any) -> str:
    """Return the canonical identifier for an object or serialized record."""
    if isinstance(record, Mapping):
        sequence = record.get("sequence", 0)
        content_sha256 = record.get("content_sha256", "")
    else:
        sequence = getattr(record, "sequence", 0)
        content_sha256 = getattr(record, "content_sha256", "")
    return f"evidence:{sequence}:{str(content_sha256)[:12]}"


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
    authority_by_url: tuple[tuple[str, EvidenceAuthority], ...] = (),
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

    recorded = dict(authority_by_url)
    authorities = tuple(
        recorded.get(url) or resolve_source_authority(url) for url in urls
    )
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
        authority_by_url=tuple(getattr(record, "authority_by_url", ()) or ()),
    )
    return EvidenceBinding(
        evidence_id=evidence_record_id(record),
        requested_url=(normalized if normalized in requested_urls else None),
        canonical_url=canonical_url,
        authority=authority,
        evidence_status="EVIDENCE_FOUND",
        provider=str(getattr(record, "tool_name", "")) or None,
    )


def _value_text(value: Any) -> str:
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
    return text


def _bounded_text(value: Any) -> str:
    text = _value_text(value)
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
    source = _value_text(value)
    content = _bounded_text(source)
    if blocked:
        return "BLOCKED", "UNAVAILABLE", "TOOL_BLOCKED", content

    # Classify the full inspected result; ledger retention limits must not hide a
    # useful neighbor or a provider error. Unclosed result envelopes prove neither.
    depth = tag_count = 0
    for tag in re.finditer(r"(?i)</?result\b", source):
        depth += -1 if tag.group().startswith("</") else 1
        tag_count += 1
        if depth not in {0, 1}:
            return "SUCCEEDED", "INSUFFICIENT", "MALFORMED_RESULT_ENVELOPE", content
    if depth:
        return "SUCCEEDED", "INSUFFICIENT", "MALFORMED_RESULT_ENVELOPE", content
    blocks = RESULT_ENVELOPE_RE.findall(source)
    if tag_count != 2 * len(blocks):
        return "SUCCEEDED", "INSUFFICIENT", "MALFORMED_RESULT_ENVELOPE", content

    status_match = _STATUS_RE.search(source)
    reason_match = _REASON_RE.search(source)
    status = status_match.group(1).upper() if status_match else ""
    reason = reason_match.group(1).upper() if reason_match else None

    # Some adapters return STATUS: EVIDENCE_FOUND around a nested provider error.
    # Error semantics override that optimistic wrapper.
    error_blocks = [block for block in blocks if _ERROR_ENVELOPE_RE.search(block)]
    valid_blocks = [
        block
        for block in blocks
        if not _ERROR_ENVELOPE_RE.search(block)
        and ("<url>" in block.casefold() or "<content>" in block.casefold())
    ]
    if error_blocks and valid_blocks:
        partial_status: EvidenceStatus = (
            "EVIDENCE_FOUND" if status == "EVIDENCE_FOUND" else "RESULTS_FOUND"
        )
        return "SUCCEEDED", partial_status, "PARTIAL_PROVIDER_ERROR", content
    if _ERROR_ENVELOPE_RE.search(source):
        error_status: EvidenceStatus = (
            "AUTH_ERROR" if _AUTH_ERROR_RE.search(source) else "INSUFFICIENT"
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
    if "DOCUMENT_METADATA:" in source:
        return "SUCCEEDED", "EVIDENCE_FOUND", reason, content
    if "<search_results" in source or "Foreign Source Search Results" in source:
        return "SUCCEEDED", "RESULTS_FOUND", reason, content
    if tool_name == "get_official_filings" and source.strip():
        return "SUCCEEDED", "EVIDENCE_FOUND", reason, content
    if not source.strip():
        return "SUCCEEDED", "NO_RESULTS", reason, ""
    return "SUCCEEDED", "RESULTS_FOUND", reason, content


class EvidenceRecorder:
    """Record each tool call's one outcome without mutating tool execution.

    A ``ToolCompletionObserver`` and a ``ToolFailureObserver``: the service
    tells it either the result the caller received (``completed``) or why the
    call produced none (``failed``), never both. Its ``after()`` is a
    pass-through, so the record does not depend on its place in the hook
    chain and no result the caller was told timed out is ever recorded.
    """

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

    def _admit(
        self, dedupe_key: tuple[str | None, str, str, tuple[str, ...]], chars: int
    ) -> bool:
        """Whether a record with this identity and size may be appended; if so,
        its identity is remembered. A duplicate is dropped, a full ledger marks
        its overflow once."""
        if dedupe_key in self._dedupe_keys:
            return False
        if (
            len(self._records) >= _MAX_RECORDS
            or self._content_chars + chars > _MAX_TOTAL_CONTENT_CHARS
        ):
            self._record_overflow()
            return False
        self._dedupe_keys.add(dedupe_key)
        return True

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        return result

    def completed(self, call: ToolInvocation, result: ToolResult) -> None:
        """Record the result the caller received, blocked or not."""
        original = _value_text(result.value)
        execution_status, evidence_status, reason, content = classify_evidence_value(
            call.name,
            original,
            blocked=result.blocked,
        )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        requested_urls = _requested_urls(call.args)
        # Include the requested-URL identity in the dedupe key: byte-identical
        # content fetched from two DIFFERENT source URLs is two corroborating
        # sources, not a duplicate, and must keep both identities.
        dedupe_key = (call.agent_key, call.name, digest, requested_urls)
        if not self._admit(dedupe_key, len(content)):
            return
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
                original_content_chars=len(original),
                retained_content_chars=min(len(original), _MAX_CONTENT_CHARS),
                content_truncated=len(original) > _MAX_CONTENT_CHARS,
                authority_by_url=tuple(
                    (
                        url,
                        resolve_evidence_authority(
                            tool_name=call.name,
                            evidence_status=evidence_status,
                            urls=(url,),
                        ),
                    )
                    for url in urls
                )
                if not result.blocked
                else (),
            )
        )
        self._content_chars += len(content)

    def failed(self, call: ToolInvocation, reason: str, scope: str | None) -> None:
        """Keep the attempt of a call that produced no result.

        ``completed()`` never runs for it, so without this the requested URL
        (and that the call timed out, and where) is lost and the model is left
        to guess the cause from prose. Same record, ``FAILED`` and ``UNAVAILABLE``.
        """
        content = tool_error_message(reason) + (
            f"\nTIMEOUT_SCOPE: {scope}" if scope else ""
        )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        requested_urls = _requested_urls(call.args)
        if not self._admit(
            (call.agent_key, call.name, digest, requested_urls), len(content)
        ):
            return
        self._records.append(
            EvidenceRecord(
                sequence=len(self._records) + 1,
                agent_key=call.agent_key,
                tool_name=call.name,
                source=call.source,
                content=content,
                content_sha256=digest,
                requested_urls=requested_urls,
                urls=(),
                blocked=False,
                findings=(f"TIMEOUT_SCOPE:{scope}",) if scope else (),
                execution_status="FAILED",
                evidence_status="UNAVAILABLE",
                reason=reason,
                original_content_chars=len(content),
                retained_content_chars=len(content),
            )
        )
        self._content_chars += len(content)

    def snapshot(self, *, agent_key: str | None = None) -> list[EvidenceRecord]:
        if agent_key is None:
            return list(self._records)
        return [record for record in self._records if record.agent_key == agent_key]

    def serialized_snapshot(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._records]
