"""Deterministic evidence gating for Value Trap acquisition context."""

from __future__ import annotations

import re
from collections.abc import Sequence

import structlog
from langchain_core.messages import BaseMessage

from src.data_block_utils import (
    build_fenced_block,
    extract_last_fenced_block,
    normalize_structured_block_boundaries,
)

from .message_utils import normalize_http_url, tool_evidence_urls

logger = structlog.get_logger(__name__)

_M_AND_A_FIELDS = (
    "M&A_CONTEXT_EVIDENCE",
    "M&A_CONTEXT_SOURCE_URL",
    "M&A_CONTEXT",
)


def _m_and_a_field(body: str, name: str) -> tuple[str, bool]:
    """Read one indented or column-zero field; reject ambiguous duplicates."""
    matches = re.findall(rf"(?m)^[ \t]*{re.escape(name)}:[ \t]*([^\n]*)$", body)
    return (matches[0].strip(), True) if len(matches) == 1 else ("", False)


def _replace_m_and_a_fields(body: str, values: dict[str, str]) -> str:
    """Write one column-zero field set for the strict downstream block reader."""
    lines = body.splitlines()
    field_pattern = re.compile(
        r"^[ \t]*(M&A_CONTEXT_EVIDENCE|M&A_CONTEXT_SOURCE_URL|M&A_CONTEXT):"
    )
    existing = [index for index, line in enumerate(lines) if field_pattern.match(line)]
    if existing:
        insertion = sum(not field_pattern.match(line) for line in lines[: existing[0]])
    else:
        insertion = next(
            (index for index, line in enumerate(lines) if line.strip() == "CATALYSTS:"),
            len(lines),
        )
    lines = [line for line in lines if not field_pattern.match(line)]
    lines[insertion:insertion] = [f"{name}: {values[name]}" for name in _M_AND_A_FIELDS]
    return "\n".join(lines)


def normalize_value_trap_m_and_a_evidence(
    report: str,
    evidence_messages: Sequence[BaseMessage],
    *,
    ticker: str,
) -> str:
    """Retain M&A context only when its cited URL occurred in this agent's tools.

    Agent-scoped message filtering happens before this function is called. This
    function adds a second, deterministic boundary: an asserted citation must be
    an HTTP(S) URL present in one of those ToolMessages. It validates provenance,
    not semantic entailment.
    """
    normalized_report = normalize_structured_block_boundaries(report) or report
    block_with_markers = extract_last_fenced_block(
        normalized_report,
        "VALUE_TRAP_BLOCK",
        include_markers=True,
    )
    block_body = extract_last_fenced_block(normalized_report, "VALUE_TRAP_BLOCK")
    if not block_with_markers or block_body is None:
        return report

    status, unique_status = _m_and_a_field(block_body, "M&A_CONTEXT_EVIDENCE")
    source_url, unique_source = _m_and_a_field(block_body, "M&A_CONTEXT_SOURCE_URL")
    context, unique_context = _m_and_a_field(block_body, "M&A_CONTEXT")
    fields_unambiguous = unique_status and unique_source and unique_context
    status = status.upper()
    normalized_source = normalize_http_url(source_url)
    available_urls = tool_evidence_urls(evidence_messages)

    citation_valid = (
        fields_unambiguous
        and status == "CITED"
        and normalized_source is not None
        and normalized_source in available_urls
        and context.upper() not in {"", "N/A", "NONE", "UNKNOWN"}
    )
    if citation_valid:
        resolved_status = "CITED"
        resolved_url = source_url
        resolved_context = context
    elif fields_unambiguous and status == "NOT_FOUND":
        resolved_status = "NOT_FOUND"
        resolved_url = "N/A"
        resolved_context = "UNKNOWN"
    else:
        resolved_status = "UNKNOWN"
        resolved_url = "N/A"
        resolved_context = "UNKNOWN"

    updated_body = _replace_m_and_a_fields(
        block_body,
        {
            "M&A_CONTEXT_EVIDENCE": resolved_status,
            "M&A_CONTEXT_SOURCE_URL": resolved_url,
            "M&A_CONTEXT": resolved_context,
        },
    )
    if updated_body == block_body:
        return normalized_report

    if not citation_valid and status == "CITED":
        logger.warning(
            "value_trap_m_and_a_citation_rejected",
            ticker=ticker,
            source_url_present=normalized_source is not None,
            source_url_seen_in_agent_tools=normalized_source in available_urls
            if normalized_source
            else False,
        )

    block_index = normalized_report.rfind(block_with_markers)
    if block_index < 0:
        return report
    replacement = build_fenced_block("VALUE_TRAP_BLOCK", updated_body.rstrip())
    return (
        normalized_report[:block_index]
        + replacement
        + normalized_report[block_index + len(block_with_markers) :]
    )
