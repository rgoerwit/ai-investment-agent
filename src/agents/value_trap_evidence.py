"""Deterministic evidence gating for Value Trap acquisition context."""

from __future__ import annotations

from collections.abc import Sequence

import structlog
from langchain_core.messages import BaseMessage

from src.data_block_utils import (
    build_fenced_block,
    extract_block_text_value,
    extract_last_fenced_block,
    normalize_structured_block_boundaries,
    replace_or_append_block_line,
)

from .message_utils import normalize_http_url, tool_evidence_urls

logger = structlog.get_logger(__name__)


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

    status = extract_block_text_value(block_body, "M&A_CONTEXT_EVIDENCE").upper()
    source_url = extract_block_text_value(block_body, "M&A_CONTEXT_SOURCE_URL")
    context = extract_block_text_value(block_body, "M&A_CONTEXT")
    normalized_source = normalize_http_url(source_url)
    available_urls = tool_evidence_urls(evidence_messages)

    citation_valid = (
        status == "CITED"
        and normalized_source is not None
        and normalized_source in available_urls
        and context.upper() not in {"", "N/A", "NONE", "UNKNOWN"}
    )
    if citation_valid:
        resolved_status = "CITED"
        resolved_url = source_url
        resolved_context = context
    elif status == "NOT_FOUND":
        resolved_status = "NOT_FOUND"
        resolved_url = "N/A"
        resolved_context = "UNKNOWN"
    else:
        resolved_status = "UNKNOWN"
        resolved_url = "N/A"
        resolved_context = "UNKNOWN"

    updated_body = replace_or_append_block_line(
        block_body, "M&A_CONTEXT_EVIDENCE", resolved_status
    )
    updated_body = replace_or_append_block_line(
        updated_body, "M&A_CONTEXT_SOURCE_URL", resolved_url
    )
    updated_body = replace_or_append_block_line(
        updated_body, "M&A_CONTEXT", resolved_context
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
