"""Deterministic retrieval and normalization for management earnings guidance."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urljoin

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

from .evidence_preflight import (
    run_preflight_calls,
    skipped_preflight_outcome,
)

logger = structlog.get_logger(__name__)

GUIDANCE_PREFLIGHT_MAX_CHARS = 36_000

GUIDANCE_PROMOTION_FIELDS: dict[str, str] = {
    "COVERAGE_STATUS": "GUIDANCE_COVERAGE_STATUS",
    "SOURCE_TYPE": "GUIDANCE_SOURCE_TYPE",
    "SOURCE_AUTHORITY": "GUIDANCE_SOURCE_AUTHORITY",
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
    "MANAGEMENT_IDENTIFIED": "GUIDANCE_MANAGEMENT_IDENTIFIED",
    "EARNINGS_BASELINE_STATUS": "EARNINGS_BASELINE_STATUS",
    "NORMALIZED_EARNINGS_AVAILABLE": "NORMALIZED_EARNINGS_AVAILABLE",
    "GUIDANCE_BRIDGE_STATUS": "GUIDANCE_BRIDGE_STATUS",
}
_THIRD_PARTY_SOURCE_MARKERS = (
    "research report",
    "analyst report",
    "sell-side",
    "broker report",
    "securities research",
    "投顧研究報告",
    "券商報告",
)
_NON_NAME_TITLE_MARKERS = (
    "revenue breakdown",
    "stock price",
    "stock quote",
    "company profile",
    "financials",
    "investing.com",
    "yahoo",
    "marketwatch",
)
_GUIDANCE_VALUE_FIELDS = (
    "REVENUE_GUIDANCE",
    "OPERATING_PROFIT_GUIDANCE",
    "ORDINARY_OR_PRETAX_PROFIT_GUIDANCE",
    "NET_INCOME_GUIDANCE",
    "NET_INCOME_YOY",
)
_NUMBER_TOKEN_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _matching_source_evidence(evidence: str, source_url: str) -> str:
    if not source_url:
        return ""
    for block in re.findall(r"(?is)<result\b[^>]*>(.*?)</result>", evidence or ""):
        if source_url in block:
            return block
    return ""


def _guidance_values_supported(block_body: str, evidence: str) -> bool:
    """Require every asserted guidance number to occur in the bound evidence."""
    expected: list[Decimal] = []
    for field in _GUIDANCE_VALUE_FIELDS:
        value = extract_block_text_value(block_body, field)
        if _is_missing_guidance_value(value):
            continue
        for token in _NUMBER_TOKEN_RE.findall(value):
            try:
                expected.append(Decimal(token.replace(",", "")))
            except InvalidOperation:
                return False
    if not expected:
        return False
    observed: set[Decimal] = set()
    for token in _NUMBER_TOKEN_RE.findall(evidence):
        try:
            observed.add(Decimal(token.replace(",", "")))
        except InvalidOperation:
            continue
    return all(value in observed for value in expected)


def _guidance_source_authority(
    block_body: str,
    management_guidance_evidence: str,
    evidence_records: Sequence[Any] = (),
) -> str:
    from src.tooling.evidence_recorder import (
        bind_fetched_evidence,
        find_fetched_evidence_record,
    )

    source_type = extract_block_text_value(block_body, "SOURCE_TYPE").upper()
    source_url = extract_block_text_value(block_body, "SOURCE_URL")
    management_identified = extract_block_text_value(
        block_body, "MANAGEMENT_IDENTIFIED"
    ).upper()
    matching_evidence = _matching_source_evidence(
        management_guidance_evidence,
        source_url,
    ).casefold()
    fetched_record = find_fetched_evidence_record(
        list(evidence_records),
        source_url,
    )
    fetched_evidence = (
        str(getattr(fetched_record, "content", "")) if fetched_record else ""
    )
    if "RESEARCH" in source_type or any(
        marker in f"{matching_evidence}\n{fetched_evidence.casefold()}"
        for marker in _THIRD_PARTY_SOURCE_MARKERS
    ):
        return "THIRD_PARTY"
    binding = bind_fetched_evidence(list(evidence_records), source_url)
    primary_types = {
        "RESULTS_RELEASE",
        "PRESENTATION",
        "TRANSCRIPT",
        "FILING",
        "MULTIPLE",
    }
    if (
        source_type in primary_types
        and management_identified == "YES"
        and binding is not None
        and _guidance_values_supported(block_body, fetched_evidence)
        and binding.authority in {"PRIMARY_REGISTRY", "PRIMARY_ISSUER"}
    ):
        return "PRIMARY"
    return "UNKNOWN"


def _official_child_urls(evidence: str) -> list[str]:
    """Resolve bounded same-host child paths emitted by official HTML extraction."""
    candidates: list[str] = []
    for raw_metadata in re.findall(
        r"(?m)^DOCUMENT_METADATA:\s*(\{.*\})\s*$",
        evidence,
    ):
        try:
            metadata = json.loads(raw_metadata)
        except (TypeError, ValueError):
            continue
        source_url = str(metadata.get("source_url") or "")
        paths = metadata.get("candidate_paths")
        if not source_url or not isinstance(paths, list):
            continue
        for path in paths:
            if isinstance(path, str):
                candidates.append(urljoin(source_url, path))
    return list(dict.fromkeys(candidates))


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
        if any(marker in candidate.casefold() for marker in _NON_NAME_TITLE_MARKERS):
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
    from src.tools.official_documents import (
        get_official_document,
        is_official_document_url,
    )
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
    results_payload = initial_outcomes[0].render()
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
    rendered_outcomes = "\n".join(outcome.render() for outcome in outcomes)
    official_candidate_urls = [
        url.rstrip(".,;:!?)]}")
        for url in dict.fromkeys(
            re.findall(
                r"https?://[^\s<>\"]+",
                rendered_outcomes,
            )
        )
        if is_official_document_url(url.rstrip(".,;:!?)]}"))
    ][:2]
    if enable_extraction and official_candidate_urls:
        labels = ("latest_results_document", "latest_results_document_backup")
        official_outcomes, official_durations = await run_preflight_calls(
            [
                (
                    labels[index],
                    get_official_document,
                    {
                        "url": url,
                        "keywords": ",".join(priority_terms[:12]),
                        "ticker": ticker,
                        "company_name": company_name,
                    },
                )
                for index, url in enumerate(official_candidate_urls)
            ],
            agent_key="foreign_language_analyst",
            source="toolnode",
            ticker=ticker,
            failure_event="management_guidance_preflight_call_failed",
            logger=logger,
        )
        outcomes.extend(official_outcomes)
        call_durations_ms.update(official_durations)
        child_urls = [
            url
            for url in _official_child_urls(
                "\n".join(outcome.render() for outcome in official_outcomes)
            )
            if url not in official_candidate_urls and is_official_document_url(url)
        ][:2]
        if child_urls:
            child_outcomes, child_durations = await run_preflight_calls(
                [
                    (
                        f"official_child_document_{index + 1}",
                        get_official_document,
                        {
                            "url": url,
                            "keywords": ",".join(priority_terms[:12]),
                            "ticker": ticker,
                            "company_name": company_name,
                        },
                    )
                    for index, url in enumerate(child_urls)
                ],
                agent_key="foreign_language_analyst",
                source="toolnode",
                ticker=ticker,
                failure_event="management_guidance_preflight_call_failed",
                logger=logger,
            )
            outcomes.extend(child_outcomes)
            call_durations_ms.update(child_durations)
    else:
        reason = "QUICK_MODE" if not enable_extraction else "NO_OFFICIAL_URLS"
        outcomes.append(skipped_preflight_outcome("latest_results_document", reason))

    bridge_payload = next(
        (
            outcome.render()
            for outcome in outcomes
            if outcome.label == "earnings_bridge"
        ),
        "",
    )
    guidance_candidate_urls = list(
        dict.fromkeys(re.findall(r"https?://[^\s<]+", bridge_payload))
    )[:3]
    if enable_extraction and guidance_candidate_urls:
        extraction_outcomes, extraction_durations = await run_preflight_calls(
            [
                (
                    "guidance_extract",
                    extract_guidance_sources,
                    {
                        "urls": guidance_candidate_urls,
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
        outcomes.append(skipped_preflight_outcome("guidance_extract", reason))

    sections = [
        "### CODE-OWNED MANAGEMENT GUIDANCE PREFLIGHT",
        "SOURCE_CLASSES_TARGETED: RESULTS_RELEASE; PRESENTATION; TRANSCRIPT_QA; STATUTORY_FILING",
        "LOCAL_ISSUER_NAME: " + (local_issuer_name or "NOT_RESOLVED"),
        "PROVENANCE_RULE: Only the searches listed below were executed by code. "
        "Do not claim that a source class was found unless a returned URL or filing payload supports it.",
    ]
    for outcome in outcomes:
        sections.extend((f"\n#### {outcome.label}", outcome.render()))
    evidence = "\n".join(sections)
    status_by_label = {
        outcome.label: (f"{outcome.execution_status}/{outcome.evidence_status}")
        for outcome in outcomes
    }
    logger.info(
        "management_guidance_preflight_complete",
        ticker=ticker,
        elapsed_ms=round((time.monotonic() - started_at) * 1000),
        call_durations_ms=call_durations_ms,
        call_statuses=status_by_label,
        extraction_attempted=enable_extraction and bool(guidance_candidate_urls),
        official_results_extraction_attempted=(
            enable_extraction and bool(official_candidate_urls)
        ),
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
    evidence_records: Sequence[Any] = (),
) -> str:
    """Attach code-owned provenance and enforce conservative bridge semantics."""
    normalized = normalize_structured_block_boundaries(content) or content
    block_with_markers = extract_last_fenced_block(
        normalized,
        "MANAGEMENT_GUIDANCE",
        include_markers=True,
    )
    block_body = extract_last_fenced_block(normalized, "MANAGEMENT_GUIDANCE")
    execution_statuses, evidence_statuses = _parse_preflight_statuses(
        management_guidance_evidence
    )
    searches_completed = _format_searches_completed(
        execution_statuses,
        evidence_statuses,
    )
    if not block_with_markers or block_body is None:
        if not _has_substantive_report_content(normalized) or not execution_statuses:
            return normalized
        conservative_block = _build_unresolved_guidance_block(
            execution_statuses,
            searches_completed,
        )
        return normalized.rstrip() + "\n\n" + conservative_block + "\n"

    coverage_status = (
        extract_block_text_value(block_body, "COVERAGE_STATUS").strip().upper()
    )
    if coverage_status not in GUIDANCE_COVERAGE_STATUSES:
        conservative_block = _build_unresolved_guidance_block(
            execution_statuses,
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

    if execution_statuses:
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

    if (
        coverage_status == "NOT_DISCLOSED_AFTER_TARGETED_SEARCH"
        and "COVERAGE_COMPLETE_NO_MATCH" not in evidence_statuses.values()
    ):
        coverage_status = "UNRESOLVED_AFTER_TARGETED_SEARCH"
        block_body = replace_or_append_block_line(
            block_body,
            "COVERAGE_STATUS",
            coverage_status,
        )

    block_body = replace_or_append_block_line(
        block_body,
        "SOURCE_AUTHORITY",
        _guidance_source_authority(
            block_body,
            management_guidance_evidence,
            evidence_records,
        ),
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
    incomplete_guidance = coverage_status == "FOUND" and (
        _is_missing_guidance_value(operating_guidance)
        or _is_missing_guidance_value(net_income_guidance)
    )
    bridge_required = direction == "OP_UP_NET_DOWN" or material_driver == "YES"
    if incomplete_guidance:
        # A document URL is not proof that the forward earnings bridge was
        # actually extracted. Keep the earnings baseline unknown so trailing EPS
        # cannot be scored as durable. Reserve UNRESOLVED for evidence that
        # actually requires an operating-to-net-income bridge; otherwise a
        # revenue-only source would block BUY more strongly than no causal
        # evidence at all.
        bridge_status = "UNRESOLVED" if bridge_required else "NOT_APPLICABLE"
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


def _parse_preflight_statuses(
    evidence: str,
) -> tuple[dict[str, str], dict[str, str]]:
    execution_statuses: dict[str, str] = {}
    evidence_statuses: dict[str, str] = {}
    blocks = re.findall(
        r"(?ms)^####\s+([a-z0-9_]+)\s*$\n(.*?)(?=^####\s+|\Z)",
        evidence or "",
    )
    for label, payload in blocks:
        execution_match = re.search(
            r"(?m)^EXECUTION_STATUS:\s+([A-Z_]+)",
            payload,
        )
        evidence_match = re.search(
            r"(?m)^EVIDENCE_STATUS:\s+([A-Z_]+)",
            payload,
        )
        legacy_match = re.search(r"(?m)^STATUS:\s+([A-Z_]+)", payload)
        legacy = legacy_match.group(1) if legacy_match else "UNKNOWN"
        execution_statuses[label] = (
            execution_match.group(1)
            if execution_match
            else "SUCCEEDED"
            if legacy in {"COMPLETED", "INSUFFICIENT_DATA"}
            else legacy
        )
        evidence_statuses[label] = (
            evidence_match.group(1)
            if evidence_match
            else "RESULTS_FOUND"
            if legacy == "COMPLETED"
            else "INSUFFICIENT"
        )
    return execution_statuses, evidence_statuses


def _format_searches_completed(
    execution_statuses: dict[str, str],
    evidence_statuses: dict[str, str],
) -> str:
    return "; ".join(
        f"{label}={execution_statuses.get(label, 'NOT_RUN')}/"
        f"{evidence_statuses.get(label, 'NOT_RUN')}"
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
    execution_statuses: dict[str, str],
    searches_completed: str,
) -> str:
    required_search_statuses = {
        execution_statuses.get("results_package"),
        execution_statuses.get("earnings_bridge"),
    }
    search_executed = "SUCCEEDED" in required_search_statuses
    coverage_status = (
        "UNRESOLVED_AFTER_TARGETED_SEARCH" if search_executed else "SEARCH_FAILED"
    )
    body = "\n".join(
        (
            f"COVERAGE_STATUS: {coverage_status}",
            "SOURCE_TYPE: N/A",
            "SOURCE_AUTHORITY: UNKNOWN",
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
