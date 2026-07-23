"""Deterministic retrieval and normalization for management earnings guidance."""

from __future__ import annotations

import re
import time
from datetime import date

import structlog

from src.data_block_utils import (
    build_fenced_block,
    extract_block_text_value,
    extract_last_fenced_block,
    fenced_block_pattern,
    normalize_structured_block_boundaries,
    replace_or_append_block_line,
)
from src.earnings_baseline import GUIDANCE_COVERAGE_STATUSES
from src.guidance_vocabulary import guidance_locale_policy

from .evidence_preflight import run_preflight_calls

logger = structlog.get_logger(__name__)

GUIDANCE_PREFLIGHT_MAX_CHARS = 36_000

GUIDANCE_PROMOTION_FIELDS: dict[str, str] = {
    "COVERAGE_STATUS": "GUIDANCE_COVERAGE_STATUS",
    "SOURCE_DATE": "GUIDANCE_SOURCE_DATE",
    "SOURCE_URL": "GUIDANCE_SOURCE_URL",
    "GUIDANCE_PERIOD": "GUIDANCE_PERIOD",
    "REVENUE_GUIDANCE": "GUIDANCE_REVENUE",
    "OPERATING_PROFIT_GUIDANCE": "GUIDANCE_OPERATING_PROFIT",
    "ORDINARY_OR_PRETAX_PROFIT_GUIDANCE": ("GUIDANCE_ORDINARY_OR_PRETAX_PROFIT"),
    "NET_INCOME_GUIDANCE": "GUIDANCE_NET_INCOME",
    "NET_INCOME_YOY": "GUIDANCE_NET_INCOME_YOY",
    "OPERATING_VS_NET_DIRECTION": "OPERATING_VS_NET_DIRECTION",
    "MATERIAL_NONOPERATING_DRIVER": "MATERIAL_NONOPERATING_DRIVER",
    "DRIVER_TYPE": "DRIVER_TYPE",
    "DRIVER_PERSISTENCE": "DRIVER_PERSISTENCE",
    "DRIVER_MATERIALITY": "DRIVER_MATERIALITY",
    "DRIVER_AFFECTED_PERIOD": "DRIVER_AFFECTED_PERIOD",
    "EARNINGS_BASELINE_STATUS": "EARNINGS_BASELINE_STATUS",
    "NORMALIZED_EARNINGS_AVAILABLE": "NORMALIZED_EARNINGS_AVAILABLE",
    "GUIDANCE_BRIDGE_STATUS": "GUIDANCE_BRIDGE_STATUS",
}


def _management_guidance_queries(
    ticker: str,
    company_name: str,
    *,
    as_of: date | None = None,
) -> tuple[tuple[str, str], ...]:
    """Build locale-aware searches covering figures and their earnings bridge."""
    from src.ticker_policy import get_ticker_suffix

    security_code = ticker.split(".", maxsplit=1)[0]
    subject = f"{security_code} {company_name}".strip()
    suffix = get_ticker_suffix(ticker)
    locale_policy = guidance_locale_policy(ticker)
    current_date = as_of or date.today()
    year_terms = f"{current_date.year} {current_date.year + 1}"
    if suffix == ".T":
        fiscal_end_year = (
            current_date.year + 1 if current_date.month >= 4 else current_date.year
        )
        fiscal_period = f"{fiscal_end_year}年3月期"
        return (
            (
                "results_package",
                f"{subject} {fiscal_period} {locale_policy.results_terms}",
            ),
            (
                "earnings_bridge",
                f"{subject} {fiscal_period} {locale_policy.bridge_terms}",
            ),
        )
    if suffix in {
        ".KS",
        ".KQ",
        ".HK",
        ".TW",
        ".TWO",
        ".SS",
        ".SZ",
        ".WA",
        ".KL",
    }:
        return (
            (
                "results_package",
                f"{subject} {year_terms} {locale_policy.results_terms}",
            ),
            (
                "earnings_bridge",
                f"{subject} {year_terms} {locale_policy.bridge_terms}",
            ),
        )
    return (
        (
            "results_package",
            f"{subject} {locale_policy.results_terms}",
        ),
        (
            "earnings_bridge",
            f"{subject} {locale_policy.bridge_terms}",
        ),
    )


def _discover_local_issuer_name(search_payload: str, ticker: str) -> str | None:
    """Extract a bounded local listing name from an exact ticker-matched title."""
    security_code = ticker.split(".", maxsplit=1)[0]
    identifiers = sorted({ticker, security_code}, key=len, reverse=True)
    identifier_pattern = "|".join(re.escape(value) for value in identifiers)
    title_patterns = (
        re.compile(
            rf"^(?P<name>.{{2,80}}?)\s*[\[(【（]\s*(?:{identifier_pattern})"
            r"\s*[\])】）]",
            re.IGNORECASE,
        ),
        re.compile(
            rf"^(?P<name>.{{2,80}}?)\s*[-–—:：|/]\s*(?:{identifier_pattern})"
            r"(?:\s|/|$)",
            re.IGNORECASE,
        ),
    )
    titles = re.findall(r"(?is)<title>(.*?)</title>", search_payload)
    for raw_title in titles:
        title = re.sub(r"<[^>]+>", "", raw_title)
        title = re.sub(r"\s+", " ", title).strip()
        match = None
        for pattern in title_patterns:
            match = pattern.search(title)
            if match is not None:
                break
        if match is None:
            continue
        candidate = match.group("name").strip(" \t-–—:：|/【[（(")
        for separator in (":", "："):
            if separator not in candidate:
                continue
            _, narrower_candidate = candidate.rsplit(separator, maxsplit=1)
            if sum(character.isalpha() for character in narrower_candidate) >= 2:
                candidate = narrower_candidate.strip()
                break
        candidate = re.sub(r"[（(]?株[)）]?\s*$", "", candidate).strip()
        if not 2 <= len(candidate) <= 80:
            continue
        if sum(character.isalpha() for character in candidate) < 2:
            continue
        return candidate
    return None


async def _preload_management_guidance_evidence(
    ticker: str,
    company_name: str,
    *,
    enable_extraction: bool = True,
) -> str:
    """Run mandatory guidance searches once through the normal tool hook chain."""
    from src.tools.research import (
        extract_guidance_sources,
        get_official_filings,
        search_foreign_sources,
    )

    started_at = time.monotonic()

    queries = dict(_management_guidance_queries(ticker, company_name))
    priority_terms = list(guidance_locale_policy(ticker).excerpt_priority_terms)
    initial_outcomes, call_durations_ms = await run_preflight_calls(
        [
            (
                "results_package",
                search_foreign_sources,
                {
                    "ticker": ticker,
                    "search_query": queries["results_package"],
                    "priority_terms": priority_terms,
                },
            ),
            ("statutory_filing_api", get_official_filings, {"ticker": ticker}),
        ],
        agent_key="foreign_language_analyst",
        source="toolnode",
        ticker=ticker,
        failure_event="management_guidance_preflight_call_failed",
        logger=logger,
    )
    results_payload = initial_outcomes[0][1]
    local_issuer_name = _discover_local_issuer_name(results_payload, ticker)
    bridge_query = queries["earnings_bridge"]
    if local_issuer_name:
        bridge_query = dict(_management_guidance_queries(ticker, local_issuer_name))[
            "earnings_bridge"
        ]
    bridge_outcomes, bridge_durations = await run_preflight_calls(
        [
            (
                "earnings_bridge",
                search_foreign_sources,
                {
                    "ticker": ticker,
                    "search_query": bridge_query,
                    "priority_terms": priority_terms,
                },
            )
        ],
        agent_key="foreign_language_analyst",
        source="toolnode",
        ticker=ticker,
        failure_event="management_guidance_preflight_call_failed",
        logger=logger,
    )
    call_durations_ms.update(bridge_durations)
    outcomes = [initial_outcomes[0], bridge_outcomes[0], initial_outcomes[1]]
    bridge_payload = next(
        (payload for label, payload in outcomes if label == "earnings_bridge"),
        "",
    )
    candidate_urls = list(
        dict.fromkeys(re.findall(r"https?://[^\s<]+", bridge_payload))
    )[:3]
    if enable_extraction and candidate_urls:
        extraction_outcomes, extraction_durations = await run_preflight_calls(
            [
                (
                    "guidance_extract",
                    extract_guidance_sources,
                    {
                        "urls": candidate_urls,
                        "query": bridge_query,
                        "priority_terms": priority_terms,
                    },
                )
            ],
            agent_key="foreign_language_analyst",
            source="toolnode",
            ticker=ticker,
            failure_event="management_guidance_preflight_call_failed",
            logger=logger,
        )
        outcomes.extend(extraction_outcomes)
        call_durations_ms.update(extraction_durations)
    else:
        reason = "QUICK_MODE" if not enable_extraction else "NO_CANDIDATE_URLS"
        outcomes.append(("guidance_extract", f"STATUS: SKIPPED ({reason})"))

    sections = [
        "### CODE-OWNED MANAGEMENT GUIDANCE PREFLIGHT",
        "SOURCE_CLASSES_TARGETED: RESULTS_RELEASE; PRESENTATION; TRANSCRIPT_QA; STATUTORY_FILING",
        "LOCAL_ISSUER_NAME: " + (local_issuer_name or "NOT_RESOLVED"),
        "PROVENANCE_RULE: Only the searches listed below were executed by code. "
        "Do not claim that a source class was found unless a returned URL or filing payload supports it.",
    ]
    for label, payload in outcomes:
        sections.extend((f"\n#### {label}", payload))
    evidence = "\n".join(sections)
    status_by_label: dict[str, str] = {}
    for label, payload in outcomes:
        match = re.search(r"^STATUS:\s+([A-Z_]+)", payload)
        status_by_label[label] = match.group(1) if match else "UNKNOWN"
    logger.info(
        "management_guidance_preflight_complete",
        ticker=ticker,
        elapsed_ms=round((time.monotonic() - started_at) * 1000),
        call_durations_ms=call_durations_ms,
        call_statuses=status_by_label,
        extraction_attempted=enable_extraction and bool(candidate_urls),
        local_issuer_name_resolved=bool(local_issuer_name),
        evidence_chars=len(evidence),
    )
    if len(evidence) <= GUIDANCE_PREFLIGHT_MAX_CHARS:
        return evidence
    head = GUIDANCE_PREFLIGHT_MAX_CHARS - 4_000
    removed = len(evidence) - GUIDANCE_PREFLIGHT_MAX_CHARS
    return (
        evidence[:head]
        + f"\n[...preflight aggregate omitted {removed:,} chars...]\n"
        + evidence[-4_000:]
    )


def normalize_management_guidance_output(
    content: str,
    management_guidance_evidence: str,
) -> str:
    """Attach code-owned provenance and enforce conservative bridge semantics."""
    normalized = normalize_structured_block_boundaries(content) or content
    block_with_markers = extract_last_fenced_block(
        normalized,
        "MANAGEMENT_GUIDANCE",
        include_markers=True,
    )
    block_body = extract_last_fenced_block(normalized, "MANAGEMENT_GUIDANCE")
    statuses = dict(
        re.findall(
            r"(?ms)^####\s+([a-z_]+)\s*$.*?^STATUS:\s+([A-Z_]+)",
            management_guidance_evidence,
        )
    )
    searches_completed = _format_searches_completed(statuses)
    if not block_with_markers or block_body is None:
        if not _has_substantive_report_content(normalized) or not statuses:
            return normalized
        conservative_block = _build_unresolved_guidance_block(
            statuses,
            searches_completed,
        )
        return normalized.rstrip() + "\n\n" + conservative_block + "\n"

    coverage_status = (
        extract_block_text_value(block_body, "COVERAGE_STATUS").strip().upper()
    )
    if coverage_status not in GUIDANCE_COVERAGE_STATUSES:
        conservative_block = _build_unresolved_guidance_block(
            statuses,
            searches_completed,
        )
        block_index = normalized.rfind(block_with_markers)
        return (
            normalized[:block_index]
            + conservative_block
            + normalized[block_index + len(block_with_markers) :]
        )

    for field in (
        "COVERAGE_STATUS",
        "SOURCE_TYPE",
        "OPERATING_VS_NET_DIRECTION",
        "MATERIAL_NONOPERATING_DRIVER",
        "DRIVER_TYPE",
        "DRIVER_PERSISTENCE",
        "DRIVER_MATERIALITY",
        "MANAGEMENT_IDENTIFIED",
        "EARNINGS_BASELINE_STATUS",
        "NORMALIZED_EARNINGS_AVAILABLE",
        "GUIDANCE_BRIDGE_STATUS",
    ):
        value = extract_block_text_value(block_body, field)
        if value:
            block_body = replace_or_append_block_line(
                block_body, field, value.strip().upper()
            )

    if statuses:
        block_body = replace_or_append_block_line(
            block_body,
            "SEARCHES_COMPLETED",
            searches_completed,
        )
        block_body = replace_or_append_block_line(
            block_body,
            "SEARCH_PROVENANCE",
            "CODE_OWNED_PREFLIGHT",
        )

    direction = extract_block_text_value(
        block_body, "OPERATING_VS_NET_DIRECTION"
    ).upper()
    driver_type = extract_block_text_value(block_body, "DRIVER_TYPE").upper()
    material_driver = extract_block_text_value(
        block_body, "MATERIAL_NONOPERATING_DRIVER"
    ).upper()
    persistence = extract_block_text_value(block_body, "DRIVER_PERSISTENCE").upper()
    baseline = extract_block_text_value(block_body, "EARNINGS_BASELINE_STATUS").upper()
    coverage_status = extract_block_text_value(block_body, "COVERAGE_STATUS").upper()
    operating_guidance = extract_block_text_value(
        block_body, "OPERATING_PROFIT_GUIDANCE"
    )
    net_income_guidance = extract_block_text_value(block_body, "NET_INCOME_GUIDANCE")
    source_url = extract_block_text_value(block_body, "SOURCE_URL")

    bridge_status = "NOT_APPLICABLE"
    missing_bridge_values = coverage_status == "FOUND" and (
        _is_missing_guidance_value(operating_guidance)
        or _is_missing_guidance_value(net_income_guidance)
    )
    if missing_bridge_values:
        # A document URL is not proof that the forward earnings bridge was
        # actually extracted. Fail conservatively so trailing EPS cannot be
        # scored as durable merely because a results-release landing page was
        # found.
        bridge_status = "UNRESOLVED"
        block_body = replace_or_append_block_line(
            block_body,
            "OPERATING_VS_NET_DIRECTION",
            "UNKNOWN",
        )
        if material_driver != "YES":
            block_body = replace_or_append_block_line(
                block_body,
                "MATERIAL_NONOPERATING_DRIVER",
                "UNKNOWN",
            )
        if driver_type in {"", "NONE"}:
            block_body = replace_or_append_block_line(
                block_body,
                "DRIVER_TYPE",
                "UNKNOWN",
            )
        if baseline == "DURABLE":
            block_body = replace_or_append_block_line(
                block_body,
                "EARNINGS_BASELINE_STATUS",
                "UNKNOWN",
            )
    elif direction == "OP_UP_NET_DOWN":
        has_sourced_driver = (
            material_driver == "YES"
            and driver_type not in {"", "NONE", "UNKNOWN"}
            and bool(re.fullmatch(r"https?://\S+", source_url, re.IGNORECASE))
        )
        bridge_status = "RECONCILED" if has_sourced_driver else "UNRESOLVED"
        if not has_sourced_driver:
            block_body = replace_or_append_block_line(
                block_body,
                "MATERIAL_NONOPERATING_DRIVER",
                "UNKNOWN",
            )
            block_body = replace_or_append_block_line(
                block_body,
                "DRIVER_TYPE",
                "UNKNOWN",
            )
            if baseline == "DURABLE":
                block_body = replace_or_append_block_line(
                    block_body,
                    "EARNINGS_BASELINE_STATUS",
                    "UNKNOWN",
                )

    if (
        driver_type == "TAX_CREDIT"
        and persistence in {"ONE_TIME", "EXPIRING"}
        and baseline == "DURABLE"
    ):
        block_body = replace_or_append_block_line(
            block_body,
            "EARNINGS_BASELINE_STATUS",
            "TEMPORARILY_BOOSTED",
        )
    block_body = replace_or_append_block_line(
        block_body,
        "GUIDANCE_BRIDGE_STATUS",
        bridge_status,
    )
    updated_block = build_fenced_block("MANAGEMENT_GUIDANCE", block_body.rstrip())
    block_index = normalized.rfind(block_with_markers)
    return (
        normalized[:block_index]
        + updated_block
        + normalized[block_index + len(block_with_markers) :]
    )


def _format_searches_completed(statuses: dict[str, str]) -> str:
    return "; ".join(
        f"{label}={statuses.get(label, 'NOT_RUN')}"
        for label in (
            "results_package",
            "earnings_bridge",
            "statutory_filing_api",
            "guidance_extract",
        )
    )


def _has_substantive_report_content(content: str) -> bool:
    narrative = fenced_block_pattern("MANAGEMENT_GUIDANCE").sub("", content)
    return len(narrative.strip()) >= 160


def _build_unresolved_guidance_block(
    statuses: dict[str, str],
    searches_completed: str,
) -> str:
    required_search_statuses = {
        statuses.get("results_package"),
        statuses.get("earnings_bridge"),
    }
    search_executed = bool(
        required_search_statuses & {"COMPLETED", "INSUFFICIENT_DATA"}
    )
    coverage_status = (
        "UNRESOLVED_AFTER_TARGETED_SEARCH" if search_executed else "SEARCH_FAILED"
    )
    body = "\n".join(
        (
            f"COVERAGE_STATUS: {coverage_status}",
            "SOURCE_TYPE: N/A",
            "SOURCE_DATE: N/A",
            "SOURCE_URL: N/A",
            f"SEARCHES_COMPLETED: {searches_completed}",
            "SEARCH_PROVENANCE: CODE_OWNED_PREFLIGHT",
            "GUIDANCE_PERIOD: N/A",
            "REVENUE_GUIDANCE: N/A",
            "OPERATING_PROFIT_GUIDANCE: N/A",
            "ORDINARY_OR_PRETAX_PROFIT_GUIDANCE: N/A",
            "NET_INCOME_GUIDANCE: N/A",
            "NET_INCOME_YOY: N/A",
            "OPERATING_VS_NET_DIRECTION: UNKNOWN",
            "MATERIAL_NONOPERATING_DRIVER: UNKNOWN",
            "DRIVER_TYPE: UNKNOWN",
            "DRIVER_PERSISTENCE: UNKNOWN",
            "DRIVER_MATERIALITY: UNKNOWN",
            "DRIVER_AFFECTED_PERIOD: UNKNOWN",
            "MANAGEMENT_IDENTIFIED: UNKNOWN",
            "EARNINGS_BASELINE_STATUS: UNKNOWN",
            "NORMALIZED_EARNINGS_AVAILABLE: UNKNOWN",
            "DRIVER_DESCRIPTION: Targeted evidence did not resolve forward guidance.",
            "GUIDANCE_BRIDGE_STATUS: UNRESOLVED",
        )
    )
    return build_fenced_block("MANAGEMENT_GUIDANCE", body)


def _is_missing_guidance_value(value: str) -> bool:
    """Return whether a claimed guidance value is absent or only prose."""
    normalized = value.strip().upper()
    if not normalized:
        return True
    missing_markers = (
        "N/A",
        "NOT AVAILABLE",
        "NOT DISCLOSED",
        "NOT EXPLICITLY",
        "UNKNOWN",
        "記載なし",
        "非開示",
    )
    if any(marker in normalized for marker in missing_markers):
        return True
    return not bool(re.search(r"\d", normalized))


def promote_management_guidance(body: str, foreign_data: str) -> tuple[str, bool]:
    """Copy the validated FLA guidance block into Senior DATA_BLOCK fields."""
    guidance = extract_last_fenced_block(foreign_data, "MANAGEMENT_GUIDANCE")
    if guidance is None:
        return body, False
    updated = body
    promoted = False
    for source_field, target_field in GUIDANCE_PROMOTION_FIELDS.items():
        value = extract_block_text_value(guidance, source_field)
        if not value:
            continue
        updated = replace_or_append_block_line(updated, target_field, value)
        promoted = True
    return updated, promoted
