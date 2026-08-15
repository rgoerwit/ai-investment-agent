from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import RunnableConfig

from src.charts.extractors.valuation_signals import VALUATION_CONTEXT_TOKENS
from src.config import config as settings_config
from src.data_block_utils import (
    build_fenced_block,
    detect_legacy_data_block_shape,
    extract_block_text_value,
    extract_last_data_block,
    fenced_end,
    fenced_start,
    has_parseable_data_block,
    has_parseable_fenced_block,
    normalize_legacy_data_block_report,
    normalize_structured_block_boundaries,
    replace_or_append_block_line,
)
from src.error_safety import summarize_exception
from src.runtime_config import get_runtime_config
from src.runtime_diagnostics import failure_artifact, success_artifact
from src.service_tiers import floor_llm_hard_timeout
from src.thesis_constants import DRAWDOWN_52WK_RATIO, DRAWDOWN_SMA200_RATIO
from src.token_tracker import (
    TokenTrackingCallback,
    canonical_display_name,
    get_tracker,
)
from src.tooling.text_boundary import format_untrusted_block

from . import message_utils, support
from . import runtime as agent_runtime
from .capital_structure import promote_capital_structure
from .evidence_constraints import AUTHORITATIVE_CORRECTION_MARKER
from .foreign_language_evidence import (
    normalize_foreign_language_evidence,
    promote_foreign_growth_evidence,
)
from .fundamentals_reconciler import (
    HORIZON_FIELD_RAW_KEYS,
    append_analyst_coverage_data_quality_note,
    extract_raw_metrics_payload,
    reconcile_high_risk_fields,
    reconcile_score_consistency,
    stamp_price_currency,
    statement_mrq_period_lag_note,
    withhold_eps_growth_for_unusable_baseline,
)
from .management_guidance import (
    _preload_management_guidance_evidence,
    backfill_guidance_contract,
    normalize_management_guidance_output,
    promote_management_guidance,
)
from .output_limits import cap_state_value
from .output_validation import (
    log_output_diagnostics,
    log_truncation_diagnostic,
    should_fail_closed,
    validate_required_output,
)
from .state import AgentState
from .value_trap_evidence import normalize_value_trap_m_and_a_evidence

logger = structlog.get_logger(__name__)

_FUNDAMENTALS_RETRY_FORMAT_SUFFIX = """
CRITICAL FORMAT CORRECTION:
Emit the DATA_BLOCK first.
Use exactly these fenced markers:
{start_marker}
...
{end_marker}
Inside DATA_BLOCK, use plain KEY: VALUE lines only.
Do NOT use markdown tables inside DATA_BLOCK.
Promote the Foreign Language Analyst MANAGEMENT_GUIDANCE result into DATA_BLOCK.
At minimum emit GUIDANCE_COVERAGE_STATUS, MATERIAL_NONOPERATING_DRIVER,
EARNINGS_BASELINE_STATUS, NORMALIZED_EARNINGS_AVAILABLE, and GUIDANCE_BRIDGE_STATUS.
If coverage is FOUND, also emit GUIDANCE_SOURCE_URL and OPERATING_VS_NET_DIRECTION.
Use explicit UNKNOWN/N/A values when evidence is absent; never omit these fields.
Two fields have narrower vocabularies than that:
MATERIAL_NONOPERATING_DRIVER must be YES, NO, or UNKNOWN (never N/A), and
GUIDANCE_BRIDGE_STATUS must be RECONCILED, UNRESOLVED, or NOT_APPLICABLE
(never UNKNOWN or N/A) — copy it from the Foreign Language Analyst block, or emit
UNRESOLVED if that block is unavailable.
""".format(
    start_marker=fenced_start("DATA_BLOCK"),
    end_marker=fenced_end("DATA_BLOCK"),
)

_FOREIGN_LANGUAGE_EVIDENCE_RETRY_SUFFIX = """
CRITICAL EVIDENCE CORRECTION:
Your first response omitted or malformed a mandatory evidence block.
Run targeted searches of the latest results release, presentation, transcript, and
statutory filing for management's forward guidance and material tax, subsidy,
regulatory, accounting, or other non-operating earnings drivers. Also locate the
newest official income statement and inspect its document before extracting current
and year-ago comparative revenue and earnings. Then emit one parseable block for
each of these marker pairs:
{guidance_start}
...
{guidance_end}
{results_start}
...
{results_end}
If guidance is not disclosed or a search fails, record that explicit coverage status
and list SEARCHES_COMPLETED. Use NOT_FOUND or SEARCH_FAILED for latest-results
coverage when appropriate. Do not silently omit either block.
""".format(
    guidance_start=fenced_start("MANAGEMENT_GUIDANCE"),
    guidance_end=fenced_end("MANAGEMENT_GUIDANCE"),
    results_start=fenced_start("LATEST_RESULTS"),
    results_end=fenced_end("LATEST_RESULTS"),
)

_QUARANTINED_FORWARD_KEYS = ("PE_RATIO_FORWARD", "PEG_RATIO")
_VALUATION_RELIABILITY_FIELD = "VALUATION_INPUT_RELIABILITY"
_VALID_ADR_EXCHANGES = {
    "NYSE",
    "NASDAQ",
    "AMEX",
    "OTC",
    "OTC-OTCQX",
    "OTC-OTCQB",
    "OTC-OTCPK",
    "PINK",
}
_AUTHORITATIVE_ADR_DOMAINS = (
    "sec.gov",
    "adr.db.com",
    "adrbny.com",
    "depositaryreceipts.citi.com",
    "citiadr.factsetdigitalsolutions.com",
    "adr.com",
    "jpmadr",
    "markitdigital.com/jpmadr",
    "otcmarkets.com",
)
_UNSPONSORED_ADR_MARKERS = (
    "unsponsored adr",
    "unsp/adr",
    "sponsorship level: unsponsored",
    "unsponsored adr program",
    "multi unsponsored",
)

_NARRATIVE_METRIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "NET_DEBT_EBITDA": re.compile(
        r"(?im)^\s*(?:[-*]\s*)?\*{0,2}(?:net\s*debt\s*/\s*ebitda|netdebt/ebitda)"
        r"\*{0,2}\s*:\s*\*{0,2}\s*(-?\d+(?:\.\d+)?)"
    ),
    "CASH_TO_ASSETS": re.compile(
        r"(?im)^\s*(?:[-*]\s*)?\*{0,2}cash\s*(?:to|/)\s*assets\*{0,2}\s*:\s*"
        r"\*{0,2}\s*(-?\d+(?:\.\d+)?)\s*(%)?"
    ),
    "DE_RATIO": re.compile(
        r"(?im)^\s*(?:[-*]\s*)?\*{0,2}(?:debt\s*(?:to|/)\s*equity|d/e)"
        r"\*{0,2}\s*:\s*\*{0,2}\s*(-?\d+(?:\.\d+)?)\s*(%)?"
    ),
}
_NARRATIVE_SCORE_PATTERNS: dict[str, re.Pattern[str]] = {
    "HEALTH": re.compile(
        r"(?ims)^###\s+FINANCIAL HEALTH DETAIL\b.*?"
        r"^\*\*Score\*\*:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)"
        r"\s*\(Adjusted:\s*(\d+(?:\.\d+)?)%\)"
    ),
    "GROWTH": re.compile(
        r"(?ims)^###\s+GROWTH TRANSITION DETAIL\b.*?"
        r"^\*\*Score\*\*:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)"
        r"\s*\(Adjusted:\s*(\d+(?:\.\d+)?)%\)"
    ),
}


def _parse_metric_number(raw: str, *, percent: bool = False) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value / 100 if percent else value


def _authoritative_metric_warning(
    narrative: str,
    updated_body: str,
) -> str:
    """Build a warning for exact, labeled narrative/DATA_BLOCK conflicts.

    Only narrowly formatted metric labels are inspected. Natural-language prose
    is never rewritten, avoiding accidental period or scope substitutions.
    """
    if AUTHORITATIVE_CORRECTION_MARKER in narrative:
        return ""

    conflicts: list[str] = []
    for field, pattern in _NARRATIVE_METRIC_PATTERNS.items():
        narrative_match = pattern.search(narrative)
        authoritative_raw = extract_block_text_value(updated_body, field)
        authoritative_match = re.search(r"-?\d+(?:\.\d+)?", authoritative_raw)
        if not narrative_match or not authoritative_match:
            continue
        narrative_value = _parse_metric_number(
            narrative_match.group(1),
            percent=bool(
                narrative_match.lastindex
                and narrative_match.lastindex > 1
                and narrative_match.group(2)
            ),
        )
        authoritative_value = _parse_metric_number(
            authoritative_match.group(0), percent="%" in authoritative_raw
        )
        if narrative_value is None or authoritative_value is None:
            continue
        tolerance = max(0.02, abs(authoritative_value) * 0.05)
        if abs(narrative_value - authoritative_value) <= tolerance:
            continue
        conflicts.append(
            f"- {field}: preceding labeled narrative={narrative_match.group(1)}"
            f"{'%' if narrative_match.lastindex and narrative_match.lastindex > 1 and narrative_match.group(2) else ''}; "
            f"authoritative DATA_BLOCK={authoritative_raw}."
        )

    for kind, pattern in _NARRATIVE_SCORE_PATTERNS.items():
        narrative_match = pattern.search(narrative)
        raw_score = extract_block_text_value(updated_body, f"RAW_{kind}_SCORE")
        adjusted_score = extract_block_text_value(
            updated_body, f"ADJUSTED_{kind}_SCORE"
        )
        raw_match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", raw_score)
        adjusted_match = re.search(r"(\d+(?:\.\d+)?)\s*%", adjusted_score)
        if not narrative_match or not raw_match or not adjusted_match:
            continue
        narrative_values = tuple(float(narrative_match.group(i)) for i in range(1, 4))
        authoritative_values = (
            float(raw_match.group(1)),
            float(raw_match.group(2)),
            float(adjusted_match.group(1)),
        )
        if all(
            abs(narrative_value - authoritative_value) <= 0.15
            for narrative_value, authoritative_value in zip(
                narrative_values,
                authoritative_values,
                strict=True,
            )
        ):
            continue
        conflicts.append(
            f"- {kind}_SCORE: narrative={narrative_match.group(1)}/"
            f"{narrative_match.group(2)} ({narrative_match.group(3)}%); "
            f"authoritative DATA_BLOCK={raw_score} ({adjusted_score})."
        )

    if not conflicts:
        return ""
    return (
        f"> **{AUTHORITATIVE_CORRECTION_MARKER}:** The labeled narrative values "
        "below conflict with reconciled structured inputs. The DATA_BLOCK values "
        "are authoritative for downstream scoring; the earlier values are superseded.\n"
        + "\n".join(f"> {line}" for line in conflicts)
        + "\n\n"
    )


def _extract_valuation_context(report: str | None) -> str:
    """Return compact valuation-relevant flags that sit outside DATA_BLOCK."""
    if not report:
        return "None"
    lines: list[str] = []
    for raw_line in report.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        upper = line.upper()
        if any(keyword in upper for keyword in VALUATION_CONTEXT_TOKENS):
            lines.append(line[:300])
        if len(lines) >= 8:
            break
    return "\n".join(lines) if lines else "None"


_SPONSORED_ADR_MARKERS = (
    "sponsorship level: sponsored",
    "sponsored level i adr",
    "sponsored level ii adr",
    "sponsored level iii adr",
    "sponsored level 1 adr",
    "sponsored level 2 adr",
    "sponsored level 3 adr",
    "sponsored adr program",
    "sponsored depositary receipt program",
)


def _home_suffix(ticker: str) -> str:
    return ticker[ticker.rfind(".") :].upper() if "." in ticker else ""


def _invalid_adr_routing(body: str, ticker: str) -> bool:
    if extract_block_text_value(body, "ADR_EXISTS").upper() != "YES":
        return False

    adr_ticker = extract_block_text_value(body, "ADR_TICKER").upper()
    adr_exchange = extract_block_text_value(body, "ADR_EXCHANGE").upper()
    suffix = _home_suffix(ticker)
    ticker_root = ticker.upper().split(".", 1)[0]
    ticker_bad = adr_ticker not in {"", "NONE", "N/A"} and (
        adr_ticker == ticker.upper()
        or adr_ticker == ticker_root
        or (suffix and adr_ticker.endswith(suffix))
    )
    exchange_bad = (
        adr_exchange not in {"", "NONE", "N/A"}
        and adr_exchange not in _VALID_ADR_EXCHANGES
    )
    return ticker_bad or exchange_bad


def _classify_otc_adr_evidence(raw_text: str) -> str | None:
    """Classify OTC ADR sponsorship only from explicit authoritative evidence."""
    text = raw_text.lower()
    if any(marker in text for marker in _UNSPONSORED_ADR_MARKERS):
        return "UNSPONSORED"
    has_authoritative_source = any(
        domain in text for domain in _AUTHORITATIVE_ADR_DOMAINS
    )
    if has_authoritative_source and any(
        marker in text for marker in _SPONSORED_ADR_MARKERS
    ):
        return "SPONSORED"
    return None


def _valuation_input_reliability(payload: dict[str, Any]) -> str:
    """Classify whether DATA_BLOCK valuation multiples can be trusted, from markers the
    fetcher/merge layer already set on the raw metrics payload.

    QUARANTINED = a forward OR trailing valuation input was actively distrusted (split
    sensitivity, low-P/E anomaly, unit/decimal/currency error, or forward-P/E quarantine);
    UNAVAILABLE = no forward valuation input present; USABLE otherwise. Conservative on an
    empty/malformed payload (UNAVAILABLE) — never raises.
    """
    if not payload:
        return "UNAVAILABLE"
    if (
        payload.get("_split_sensitive_metrics_quarantined") is True
        or payload.get("_pe_low_anomaly_quarantined") is True
        or payload.get("_pe_unit_error_quarantined") in {"forward", "trailing"}
        or bool(payload.get("_forwardPE_quarantine_reason"))
    ):
        return "QUARANTINED"
    if all(payload.get(key) is None for key in ("forwardPE", "forwardEps", "pegRatio")):
        return "UNAVAILABLE"
    return "USABLE"


def _append_metric_provenance(body: str, payload: dict[str, Any]) -> str:
    field_sources = payload.get("_field_sources")
    if not isinstance(field_sources, dict):
        field_sources = {}

    updated = body
    forward_eps = payload.get("forwardEps")
    if isinstance(forward_eps, int | float) and not isinstance(forward_eps, bool):
        updated = replace_or_append_block_line(
            updated,
            "FORWARD_EPS",
            f"{float(forward_eps):.2f}",
        )
    updated = replace_or_append_block_line(
        updated,
        "FORWARD_EPS_SOURCE",
        str(field_sources.get("forwardEps") or "UNKNOWN"),
    )
    updated = replace_or_append_block_line(
        updated,
        "PE_RATIO_FORWARD_SOURCE",
        str(field_sources.get("forwardPE") or "UNKNOWN"),
    )

    cfo_ni_years = payload.get("moat_cfoToNiYears")
    if isinstance(cfo_ni_years, int | float) and not isinstance(cfo_ni_years, bool):
        updated = replace_or_append_block_line(
            updated,
            "MOAT_CFO_NI_YEARS",
            f"{float(cfo_ni_years):g}",
        )
    return replace_or_append_block_line(
        updated,
        "MOAT_CFO_NI_SOURCE",
        str(field_sources.get("moat_cfoToNiAvg") or "UNKNOWN"),
    )


def _sanitize_fundamentals_output(
    content: str,
    raw_data: str,
    ticker: str,
    foreign_data: str = "",
    legal_data: str = "",
    canonical_snapshot: Mapping[str, Any] | None = None,
) -> str:
    # Score consistency is checked even without raw data (it is DATA_BLOCK-internal),
    # so only an unparseable block short-circuits; payload-dependent steps stay
    # gated on has_structured_payload below.
    if not has_parseable_data_block(content):
        return content

    payload = extract_raw_metrics_payload(raw_data)
    has_structured_payload = bool(payload)

    block_body = extract_last_data_block(content, include_markers=False)
    block_with_markers = extract_last_data_block(content, include_markers=True)
    if block_body is None or block_with_markers is None:
        return content

    updated_body = block_body
    # `corrective` marks a sanitization that *fixed something suspect* (quarantine,
    # invalid ADR routing, ADR sponsorship correction) vs. routine additive
    # annotation (coverage data-quality note, valuation-reliability field, N/A
    # backfills) that happens on essentially every run. The final log fires at
    # WARNING only when corrective, else INFO — so the WARNING keeps signal value.
    corrective = False
    if has_structured_payload:
        updated_body = reconcile_high_risk_fields(updated_body, payload)
        updated_body = _append_metric_provenance(updated_body, payload)
        # Payload-gated like every sibling reconciliation. With no structured
        # payload the denomination is unverifiable, and stamping N/A over the
        # model's line would destroy a possibly-correct transcription while
        # adding a line to every thin-data block. Consumers resolve the unit
        # from the payload-derived snapshot currency, not from this field, so
        # an unstamped block degrades to today's suffix resolution.
        updated_body = stamp_price_currency(updated_body, payload)
        mrq_lag_note = statement_mrq_period_lag_note(payload)
        if mrq_lag_note:
            existing_note = extract_block_text_value(
                updated_body,
                "GROWTH_DATA_QUALITY_NOTE",
            )
            combined_note = (
                f"{existing_note} {mrq_lag_note}".strip()
                if mrq_lag_note not in existing_note
                else existing_note
            )
            updated_body = replace_or_append_block_line(
                updated_body,
                "GROWTH_DATA_QUALITY_NOTE",
                combined_note,
            )

    updated_body, guidance_promoted = promote_management_guidance(
        updated_body,
        foreign_data,
    )
    if guidance_promoted:
        logger.info("management_guidance_promoted", ticker=ticker)

    # A degraded or absent Foreign Language Analyst leaves the guidance contract
    # incomplete, which fails the whole fundamentals artifact closed and skips the
    # Portfolio Manager. State the absence deterministically instead: the analysis
    # survives, and the conservative values withhold EPS-growth credit and block
    # BUY on their own terms.
    updated_body, guidance_backfilled = backfill_guidance_contract(
        updated_body,
        foreign_data,
    )
    if guidance_backfilled:
        logger.warning(
            "management_guidance_backfilled",
            ticker=ticker,
            fields=list(guidance_backfilled),
            reason=(
                "Foreign Language Analyst guidance contract unavailable; "
                "conservative code-owned values applied"
            ),
        )

    updated_body, growth_evidence_promoted = promote_foreign_growth_evidence(
        updated_body,
        foreign_data,
    )
    if growth_evidence_promoted:
        logger.info("foreign_growth_evidence_promoted", ticker=ticker)
    latest_results_authority = extract_block_text_value(
        updated_body,
        "LATEST_RESULTS_SOURCE_AUTHORITY",
    ).upper()
    latest_results_end = extract_block_text_value(
        updated_body,
        "LATEST_RESULTS_PERIOD_END",
    )
    statement_mrq_period = (
        payload.get("latest_quarter_date")
        if payload.get("_latest_quarter_date_source") == "yfinance_quarterly"
        and any(
            payload.get(source_field) == "calculated_from_quarterly"
            for source_field in (
                "_revenueGrowth_MRQ_source",
                "_earningsGrowth_MRQ_source",
            )
        )
        else None
    )
    if isinstance(statement_mrq_period, str) and latest_results_end:
        try:
            newer_results_exist = datetime.fromisoformat(
                latest_results_end
            ) > datetime.fromisoformat(statement_mrq_period)
        except ValueError:
            newer_results_exist = False
        if newer_results_exist:
            latest_period = extract_block_text_value(
                updated_body,
                "LATEST_RESULTS_PERIOD",
            )
            if latest_results_authority == "PRIMARY":
                latest_note = (
                    f"Newer primary results exist for "
                    f"{latest_period or latest_results_end}; statement-derived MRQ "
                    f"growth remains aligned to {statement_mrq_period}."
                )
            else:
                latest_note = (
                    f"A newer-period results candidate exists for "
                    f"{latest_period or latest_results_end} (authority: "
                    f"{latest_results_authority or 'UNKNOWN'}) but was not "
                    "primary-validated; statement-derived MRQ growth remains aligned "
                    f"to {statement_mrq_period}. Do not present that MRQ period as "
                    "the latest reported quarter."
                )
            existing_note = extract_block_text_value(
                updated_body,
                "GROWTH_DATA_QUALITY_NOTE",
            )
            if latest_note not in existing_note:
                updated_body = replace_or_append_block_line(
                    updated_body,
                    "GROWTH_DATA_QUALITY_NOTE",
                    f"{existing_note} {latest_note}".strip(),
                )

    updated_body, capital_structure_promoted = promote_capital_structure(
        updated_body,
        legal_data,
    )
    if capital_structure_promoted:
        logger.info("capital_structure_promoted", ticker=ticker)

    updated_body = append_analyst_coverage_data_quality_note(
        updated_body,
        foreign_data,
    )

    if (
        has_structured_payload
        and payload.get("_split_sensitive_metrics_quarantined") is True
    ):
        corrective = True
        for key in _QUARANTINED_FORWARD_KEYS:
            updated_body = replace_or_append_block_line(updated_body, key, "N/A")

    if has_structured_payload and payload.get("_pe_low_anomaly_quarantined") is True:
        corrective = True
        updated_body = replace_or_append_block_line(updated_body, "PE_RATIO_TTM", "N/A")
        updated_body = replace_or_append_block_line(updated_body, "PEG_RATIO", "N/A")

    if has_structured_payload:
        for datablock_key, raw_key in HORIZON_FIELD_RAW_KEYS:
            if payload.get(raw_key) is None:
                updated_body = replace_or_append_block_line(
                    updated_body,
                    datablock_key,
                    "N/A",
                )

    if has_structured_payload:
        updated_body = replace_or_append_block_line(
            updated_body,
            _VALUATION_RELIABILITY_FIELD,
            _valuation_input_reliability(payload),
        )

    latest_quarter_date = payload.get("latest_quarter_date")
    latest_quarter_date_authoritative = has_structured_payload and payload.get(
        "_latest_quarter_date_source"
    ) in {"yfinance_quarterly", "reconciled_most_recent_quarter"}
    if (
        latest_quarter_date_authoritative
        and isinstance(latest_quarter_date, str)
        and latest_quarter_date
    ):
        updated_body = replace_or_append_block_line(
            updated_body,
            "LATEST_QUARTER_DATE",
            latest_quarter_date,
        )
        mrq_source_fields = {
            "REVENUE_GROWTH_MRQ": "_revenueGrowth_MRQ_source",
            "EARNINGS_GROWTH_MRQ": "_earningsGrowth_MRQ_source",
        }
        statement_period = (
            latest_quarter_date
            if payload.get("_latest_quarter_date_source") == "yfinance_quarterly"
            else None
        )
        for field_name, source_field in mrq_source_fields.items():
            value = extract_block_text_value(updated_body, field_name)
            if not value or value.upper().startswith(("N/A", "NA", "NONE")):
                continue
            if (
                statement_period is None
                or payload.get(source_field) != "calculated_from_quarterly"
            ):
                unbound_value = re.sub(
                    r"\s*\(as of\s*\d{4}-\d{2}-\d{2}\)",
                    "",
                    value,
                    flags=re.IGNORECASE,
                ).strip()
                if unbound_value != value:
                    updated_body = replace_or_append_block_line(
                        updated_body,
                        field_name,
                        unbound_value,
                    )
                continue
            value = re.sub(
                r"\(as of\s*\d{4}-\d{2}-\d{2}\)",
                f"(as of {statement_period})",
                value,
                flags=re.IGNORECASE,
            )
            if "(as of" not in value.lower():
                value = f"{value} (as of {statement_period})"
            updated_body = replace_or_append_block_line(
                updated_body,
                field_name,
                value,
            )

    if _invalid_adr_routing(updated_body, ticker):
        corrective = True
        for key, value in {
            "ADR_TICKER": "None",
            "ADR_EXCHANGE": "None",
            "ADR_THESIS_IMPACT": "UNCERTAIN",
            "ADR_DATA_QUALITY_NOTE": (
                "Invalid ADR routing fields removed; ADR status unresolved."
            ),
        }.items():
            updated_body = replace_or_append_block_line(
                updated_body,
                key,
                value,
            )
        logger.warning("adr_routing_invalidated", ticker=ticker)

    adr_exchange = extract_block_text_value(updated_body, "ADR_EXCHANGE").upper()
    adr_type = extract_block_text_value(updated_body, "ADR_TYPE").upper()
    if adr_exchange.startswith("OTC") and adr_type == "SPONSORED":
        evidence = _classify_otc_adr_evidence(raw_data)
        replacements: dict[str, str] | None = None
        if evidence == "UNSPONSORED":
            replacements = {
                "ADR_TYPE": "UNSPONSORED",
                "ADR_THESIS_IMPACT": "EMERGING_INTEREST",
                "ADR_DATA_QUALITY_NOTE": (
                    "OTC ADR sponsorship corrected from explicit unsponsored evidence."
                ),
            }
            logger.warning("adr_sponsorship_corrected_to_unsponsored", ticker=ticker)
        elif evidence is None:
            replacements = {
                "ADR_TYPE": "UNCERTAIN",
                "ADR_THESIS_IMPACT": "UNCERTAIN",
                "ADR_DATA_QUALITY_NOTE": (
                    "OTC sponsorship claim lacked authoritative evidence; loose "
                    "source language ignored."
                ),
            }
            logger.warning("adr_sponsorship_downgraded_to_uncertain", ticker=ticker)

        if replacements:
            corrective = True
            for key, value in replacements.items():
                updated_body = replace_or_append_block_line(
                    updated_body,
                    key,
                    value,
                )

    # This is the single final writer for registered factual claims. Everything
    # below derives scores or annotations from the reconciled facts.
    from src.analysis_snapshot import reconcile_data_block_projection

    updated_body, projection_conflicts = reconcile_data_block_projection(
        updated_body,
        canonical_snapshot,
    )
    if projection_conflicts:
        corrective = True
        logger.warning(
            "senior_claim_projection_reconciled",
            ticker=ticker,
            conflict_fields=[
                conflict["field"] for conflict in projection_conflicts[:20]
            ],
        )

    updated_body, eps_growth_withheld = withhold_eps_growth_for_unusable_baseline(
        updated_body
    )
    if eps_growth_withheld:
        corrective = True
        logger.warning(
            "eps_growth_credit_withheld_for_unusable_baseline",
            ticker=ticker,
        )

    updated_body, score_corrected, score_suspect = reconcile_score_consistency(
        updated_body
    )
    if score_corrected or score_suspect:
        corrective = True
        logger.warning(
            "score_consistency_reconciled",
            ticker=ticker,
            corrected=score_corrected,
            suspect=score_suspect,
        )

    # Keep the final-writer boundary enforceable as this sanitizer evolves.
    # Reusing the projector avoids maintaining a second list of protected fields.
    integrity_projection, late_projection_conflicts = reconcile_data_block_projection(
        updated_body, canonical_snapshot
    )
    if integrity_projection != updated_body:
        conflict_fields = [
            conflict["field"] for conflict in late_projection_conflicts[:20]
        ]
        logger.error(
            "post_projection_fact_mutation",
            ticker=ticker,
            conflict_fields=conflict_fields or ["REGISTERED_METADATA"],
        )
        raise ValueError(
            "POST_PROJECTION_FACT_MUTATION: "
            + ", ".join(conflict_fields or ["registered claim metadata"])
        )

    # WARNING only when a corrective fix was applied; routine additive annotation
    # (which changes the block on nearly every run) logs at INFO.
    log = logger.warning if corrective else logger.info
    log("fundamentals_datablock_sanitized", ticker=ticker, corrective=corrective)
    block_index = content.rfind(block_with_markers)
    if block_index < 0:
        return content
    content_before_block = content[:block_index]
    content_after_block = content[block_index + len(block_with_markers) :]
    narrative = content_before_block + content_after_block
    warning = _authoritative_metric_warning(narrative, updated_body)
    if updated_body == block_body and not warning:
        return content
    # `extract_last_data_block(include_markers=True)` consumes the newline that
    # follows the END marker, but `build_fenced_block` does not re-emit one — so a
    # naive rebuild deletes it and nudges the block toward the glued-boundary form
    # that _repair_glued_datablock_boundary exists to undo. Restore exactly what was
    # consumed so modifying the block cannot alter anything outside it.
    block_suffix = "\n" if block_with_markers.endswith("\n") else ""
    updated_block = build_fenced_block("DATA_BLOCK", updated_body.rstrip())
    return (
        content_before_block
        + warning
        + updated_block
        + block_suffix
        + content_after_block
    )


def _normalize_structured_output(
    agent_key: str,
    content: str,
    ticker: str,
    *,
    raw_data: str = "",
    foreign_data: str = "",
    legal_data: str = "",
    management_guidance_evidence: str = "",
    evidence_messages: list[BaseMessage] | None = None,
    canonical_snapshot: Mapping[str, Any] | None = None,
) -> str:
    """Apply narrow deterministic output repairs for known model-format drift."""
    if agent_key == "foreign_language_analyst":
        from src.runtime_services import get_current_evidence_records

        evidence_records = [
            record
            for record in get_current_evidence_records(agent_key=agent_key)
            if not record.blocked
        ]
        ledger_records = [
            message_utils.evidence_record_to_tool_evidence(record)
            for record in evidence_records
        ]
        guidance_normalized = normalize_management_guidance_output(
            content,
            management_guidance_evidence,
            evidence_records,
        )
        return normalize_foreign_language_evidence(
            guidance_normalized,
            evidence_messages or [],
            ticker=ticker,
            supplemental_evidence=management_guidance_evidence,
            additional_records=ledger_records,
        )

    if agent_key == "value_trap_detector":
        return normalize_value_trap_m_and_a_evidence(
            content,
            evidence_messages or [],
            ticker=ticker,
        )

    if agent_key != "fundamentals_analyst":
        return content

    repair_kind = detect_legacy_data_block_shape(content)
    normalized = normalize_legacy_data_block_report(content) or content
    if normalized != content:
        event = (
            "fundamentals_markdown_table_datablock_repaired"
            if repair_kind == "table"
            else "fundamentals_legacy_datablock_repaired"
        )
        logger.warning(
            event,
            ticker=ticker,
            repair_kind=repair_kind,
            original_has_datablock=has_parseable_fenced_block(content, "DATA_BLOCK"),
            repaired_has_datablock=has_parseable_data_block(normalized),
        )
    boundary_normalized = normalize_structured_block_boundaries(normalized)
    if boundary_normalized != normalized:
        logger.warning(
            "fundamentals_datablock_boundary_repaired",
            ticker=ticker,
            original_has_datablock=has_parseable_fenced_block(content, "DATA_BLOCK"),
            repaired_has_datablock=has_parseable_data_block(boundary_normalized),
        )
    normalized = boundary_normalized or normalized
    normalized = _sanitize_fundamentals_output(
        normalized,
        raw_data,
        ticker,
        foreign_data=foreign_data,
        legal_data=legal_data,
        canonical_snapshot=canonical_snapshot,
    )
    return normalized


def _should_retry_output(content: str, agent_key: str) -> bool:
    """Return True when the initial output should get one deep-model retry."""
    if support._is_output_insufficient(content, agent_key):
        return True

    if agent_key == "fundamentals_analyst":
        return not validate_required_output(agent_key, content)["ok"]
    if agent_key == "foreign_language_analyst":
        return not validate_required_output(agent_key, content)["ok"]
    return False


def _build_retry_invocation_messages(
    invocation_messages: list[Any], agent_key: str, content: str
) -> list[Any]:
    """Add the owning agent's structured-output correction for a retry."""
    if agent_key == "foreign_language_analyst":
        if validate_required_output(agent_key, content)["ok"]:
            return invocation_messages
        return [
            *invocation_messages,
            HumanMessage(content=_FOREIGN_LANGUAGE_EVIDENCE_RETRY_SUFFIX.strip()),
        ]

    if (
        agent_key != "fundamentals_analyst"
        or validate_required_output(agent_key, content)["ok"]
    ):
        return invocation_messages

    return [
        *invocation_messages,
        HumanMessage(content=_FUNDAMENTALS_RETRY_FORMAT_SUFFIX.strip()),
    ]


def _build_portfolio_macro_event_context(ticker: str) -> str:
    """Return the existing portfolio-detected macro-event block for News Analyst."""
    try:
        from src.memory import create_macro_events_store
        from src.ticker_policy import get_ticker_suffix

        macro_store = create_macro_events_store()
        if not macro_store.available:
            return ""

        region = get_ticker_suffix(ticker)
        events = macro_store.get_active_events(region_filter=region or None)
        if not events:
            return ""

        lines = ["### PORTFOLIO MACRO EVENT"]
        for event in events[:2]:
            lines.append(
                f"- {event.event_date} | {event.impact} | "
                f"{event.scope}: {event.news_headline}"
            )
            if event.news_detail:
                lines.append(f"  {event.news_detail}")
        lines.append(
            "Instruction: Determine if this equity is an "
            "'Innocent Bystander' (dropped due to the macro event, "
            "fundamentals intact -> OPPORTUNITY) or "
            "'Structurally Impaired' (business model affected -> EXIT). "
            "Ignore if event is inapplicable to this region/sector."
        )
        logger.debug("macro_events_injected", ticker=ticker, count=len(events[:2]))
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("macro_events_injection_failed", ticker=ticker, error=str(exc))
        return ""


def _build_regional_macro_context_block(context: Any | None, ticker: str) -> str:
    """Return the cached regional macro brief block for News Analyst."""
    return support.format_macro_context_for_agent(context, audience="news")


def _build_news_macro_extra_context(ticker: str, context: Any | None) -> str:
    """Build deterministic macro context for News Analyst.

    Keep discrete portfolio shocks first and broader regional regime context
    second so the prompt can de-duplicate them rather than treating them as two
    unrelated signals.
    """
    blocks = []
    portfolio_block = _build_portfolio_macro_event_context(ticker)
    portfolio_macro_event_present = bool(portfolio_block)
    if portfolio_block:
        blocks.append(portfolio_block)

    regional_block = _build_regional_macro_context_block(context, ticker)
    regional_macro_context_present = bool(regional_block)
    if regional_block:
        macro_region = getattr(context, "macro_context_region", "GLOBAL") or "GLOBAL"
        macro_status = (
            getattr(context, "macro_context_status", "disabled") or "disabled"
        )
        macro_report = getattr(context, "macro_context_report", "") if context else ""
        logger.debug(
            "macro_context_injected",
            ticker=ticker,
            region=macro_region,
            status=macro_status,
            report_len=len(macro_report),
            agent="news_analyst",
            portfolio_macro_event_present=portfolio_macro_event_present,
            regional_macro_context_present=regional_macro_context_present,
        )
        blocks.append(regional_block)

    if not blocks:
        return ""
    return "\n\n" + "\n\n".join(blocks) + "\n"


def _build_news_price_drawdown_context(ticker: str, context: Any | None) -> str:
    """Return the drawdown-investigation trigger block for the News Analyst.

    News runs parallel to fundamentals, so the trigger uses the pre-graph
    ``price_snapshot`` (advisory; empty string when absent or not triggered).
    """
    snapshot = getattr(context, "price_snapshot", None) or {}
    current = snapshot.get("current")
    high = snapshot.get("high_52w")
    sma200 = snapshot.get("sma200")
    if not current or not high or current <= 0 or high <= 0:
        return ""
    triggered = (current / high <= DRAWDOWN_52WK_RATIO) or (
        sma200 and sma200 > 0 and current < DRAWDOWN_SMA200_RATIO * sma200
    )
    if not triggered:
        return ""
    logger.info(
        "news_drawdown_context_injected",
        ticker=ticker,
        pct_of_52wk_high=round(current / high * 100, 1),
    )
    sma_line = f" 200-day SMA: {sma200:.2f}." if sma200 else ""
    return (
        "### PRICE DRAWDOWN CONTEXT\n"
        f"- Current price {current:.2f} is {(1 - current / high) * 100:.0f}% below "
        f"the 52-week high ({high:.2f}).{sma_line}\n"
        "Instruction: follow the PRICE DRAWDOWN PROTOCOL — run the mandatory "
        "targeted 'share price decline reason' searches (English + native "
        "language) and report DRAWDOWN_EXPLANATION in the SUMMARY."
    )


def create_analyst_node(
    llm,
    agent_key: str,
    tools: list[Any],
    output_field: str,
    retry_llm: Any | None = None,
    allow_retry: bool = False,
) -> Callable:
    """
    Factory function creating data analyst agent nodes.
    """

    async def analyst_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt(agent_key)
        if not agent_prompt:
            logger.error("missing_prompt", agent=agent_key)
            return failure_artifact(
                output_field,
                f"Could not load prompt for {agent_key}.",
                provider="unknown",
            )

        messages_template = [MessagesPlaceholder(variable_name="messages")]
        prompt_template = ChatPromptTemplate.from_messages(messages_template)
        runnable = (
            prompt_template | llm.bind_tools(tools) if tools else prompt_template | llm
        )

        try:
            prompts_used = state.get("prompts_used", {})
            prompts_used[output_field] = {
                "agent_name": agent_prompt.agent_name,
                "version": agent_prompt.version,
            }

            from src.llm_runtime.messages import prepare_messages_for_model

            filtered_messages = prepare_messages_for_model(
                llm, state.get("messages", []), agent_key=agent_key
            )
            msg_types = [type(message).__name__ for message in filtered_messages]
            msg_has_tool_calls = [
                bool(getattr(message, "tool_calls", None))
                for message in filtered_messages
                if hasattr(message, "tool_calls")
            ]
            logger.debug(
                "analyst_filtered_messages",
                agent_key=agent_key,
                total_state_messages=len(state.get("messages", [])),
                filtered_count=len(filtered_messages),
                message_types=msg_types,
                has_tool_calls_list=msg_has_tool_calls,
            )

            context = support.get_context_from_config(config)
            current_date = (
                context.trade_date if context else datetime.now().strftime("%Y-%m-%d")
            )
            ticker = (
                context.ticker
                if context
                else state.get("company_of_interest", "UNKNOWN")
            )
            company_name = state.get("company_name", ticker)
            company_resolved = state.get("company_name_resolved", True)

            extra_context = ""
            trusted_context_instructions = ""
            macro_context_injected_into_news = False
            management_guidance_evidence = ""

            if agent_key == "foreign_language_analyst":
                management_guidance_evidence = state.get(
                    "management_guidance_evidence", ""
                )
                if not management_guidance_evidence:
                    management_guidance_evidence = (
                        await _preload_management_guidance_evidence(
                            ticker,
                            company_name,
                            enable_extraction=not get_runtime_config(
                                settings_config
                            ).quick_mode_active,
                        )
                    )
                trusted_context_instructions += (
                    "\nA code-owned management-guidance preflight is supplied below. "
                    "Use it before optional follow-up searches. SEARCHES_COMPLETED "
                    "must reflect its recorded outcomes plus any tool calls you "
                    "actually make; never infer source-class coverage.\n"
                )
                extra_context += f"\n\n{management_guidance_evidence}\n"

            if agent_key == "junior_fundamentals_analyst":
                news_report = state.get("news_report", "")
                if news_report:
                    extra_context = (
                        "\n\n### NEWS CONTEXT (for ADR/analyst search queries)"
                        f"\n{news_report}\n"
                    )

            if agent_key == "fundamentals_analyst":
                from src.claim_policy import RAW_FINANCIAL_METRICS_INPUT
                from src.runtime_diagnostics import get_valid_artifact_content
                from src.tooling.structured_ingress import (
                    render_structured_ingress_payload,
                )

                raw_data = render_structured_ingress_payload(
                    state,
                    RAW_FINANCIAL_METRICS_INPUT,
                )
                foreign_data = get_valid_artifact_content(
                    state,
                    "foreign_language_report",
                )
                legal_data = get_valid_artifact_content(state, "legal_report")
                news_report = state.get("news_report", "")

                if raw_data:
                    extra_context = (
                        f"\n\n### CODE-OWNED RAW FINANCIAL METRICS\n{raw_data}\n"
                    )
                else:
                    logger.warning(
                        "senior_fundamentals_no_raw_data",
                        ticker=ticker,
                        message="Canonical raw-metrics contract is unavailable",
                    )

                if foreign_data:
                    trusted_context_instructions += (
                        "\nCross-reference foreign or alternative-source data against "
                        "the code-owned raw metrics. Preserve the canonical snapshot "
                        "when both sources report the same registered metric.\n"
                    )
                    extra_context += (
                        "\n\n### FOREIGN/ALTERNATIVE SOURCE DATA (Cross-Reference)"
                        f"{foreign_data}\n"
                    )
                    logger.debug(
                        "senior_fundamentals_has_foreign_data",
                        ticker=ticker,
                        foreign_data_length=len(foreign_data),
                    )
                else:
                    logger.debug(
                        "senior_fundamentals_no_foreign_data",
                        ticker=ticker,
                        message="Foreign Language Analyst data not available - proceeding with canonical raw metrics only",
                    )

                if news_report:
                    news_highlights = support.extract_news_highlights(
                        news_report,
                        max_chars=5000,
                    )
                    extra_context += (
                        "\n\n### NEWS HIGHLIGHTS (for Qualitative Growth Scoring)"
                        f"\n{news_highlights}\n"
                    )
                else:
                    logger.debug(
                        "senior_fundamentals_no_news",
                        ticker=ticker,
                        message="News report not yet available (parallel execution) - proceeding without news context",
                    )

                conflict_report = support.compute_data_conflicts(raw_data, foreign_data)
                if conflict_report:
                    trusted_context_instructions += (
                        "\nReview the computed data-conflict report below and resolve "
                        "discrepancies conservatively.\n"
                    )
                    extra_context += conflict_report
                    logger.debug(
                        "senior_fundamentals_conflicts_detected",
                        ticker=ticker,
                        conflict_count=conflict_report.count("\n- "),
                    )

                if legal_data:
                    trusted_context_instructions += (
                        "\nUse Legal Counsel output to inform PFIC_RISK in DATA_BLOCK. "
                        "If Legal Counsel found PFIC disclosure (pfic_status: PROBABLE), "
                        "set PFIC_RISK to MEDIUM or HIGH. If no disclosure was found in "
                        "a high-risk sector (pfic_status: UNCERTAIN), set PFIC_RISK to "
                        "at least MEDIUM. Preserve its capital-structure assessment and "
                        "do not treat ordinary non-recourse commitments as hidden debt.\n"
                    )
                    extra_context += (
                        "\n\n### LEGAL/TAX RISK ASSESSMENT (From Legal Counsel)"
                        f"{legal_data}\n"
                    )
                    logger.debug(
                        "senior_fundamentals_has_legal_data",
                        ticker=ticker,
                        legal_data_length=len(legal_data),
                    )
                else:
                    logger.debug(
                        "senior_fundamentals_no_legal_data",
                        ticker=ticker,
                        message="Legal Counsel data not yet available - proceeding without legal context",
                    )

                from src.analysis_snapshot import render_analysis_snapshot

                snapshot_context = render_analysis_snapshot(
                    state.get("analysis_snapshot")
                )
                if snapshot_context:
                    trusted_context_instructions += (
                        "\nThe canonical pre-Senior snapshot below owns every "
                        "registered fact. Preserve its value, period, authority, "
                        "and eligibility; do not fill an N/A registered fact from "
                        "memory or narrative.\n"
                    )
                    extra_context += f"\n\n{snapshot_context}\n"

            if agent_key == "news_analyst":
                news_macro_context = _build_news_macro_extra_context(ticker, context)
                extra_context += news_macro_context
                macro_context_injected_into_news = (
                    "### REGIONAL MACRO CONTEXT" in news_macro_context
                )
                drawdown_context = _build_news_price_drawdown_context(ticker, context)
                if drawdown_context:
                    extra_context += "\n\n" + drawdown_context + "\n"

            core_system_instruction = (
                f"{agent_prompt.system_message}\n\n"
                f"Date: {support._format_date_with_fy_hint(current_date)}\n"
                f"Ticker: {ticker}\n"
                f"{support._company_line(company_name, company_resolved)}\n"
                f"{support.get_analysis_context(ticker)}"
                f"{trusted_context_instructions}"
            )
            invocation_messages: list[BaseMessage] = [
                SystemMessage(content=core_system_instruction),
            ]
            if extra_context:
                invocation_messages.append(
                    HumanMessage(
                        content=format_untrusted_block(
                            extra_context,
                            "SUPPLEMENTARY CONTEXT",
                            provenance="prior analysis stages and external data sources",
                        )
                    )
                )
            invocation_messages.extend(filtered_messages)

            response = await agent_runtime.invoke_with_rate_limit_handling(
                runnable,
                {"messages": invocation_messages},
                context=agent_prompt.agent_name,
                provider=support.infer_provider_name(llm),
                model_name=support.get_model_name(llm),
            )
            response.name = agent_key

            new_state: dict[str, Any] = {
                "sender": agent_key,
                "messages": [response],
                "prompts_used": prompts_used,
            }
            if agent_key == "news_analyst":
                new_state["macro_context_injected_into_news"] = (
                    macro_context_injected_into_news
                )
            if agent_key == "foreign_language_analyst":
                new_state["management_guidance_evidence"] = management_guidance_evidence

            tool_calls = getattr(response, "tool_calls", None)
            has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
            logger.debug(
                "analyst_response_details",
                agent_key=agent_key,
                content_type=type(response.content).__name__,
                content_len=len(response.content) if response.content else 0,
                tool_calls_count=len(tool_calls) if isinstance(tool_calls, list) else 0,
                has_tool_calls=has_tool_calls,
            )

            if has_tool_calls:
                return new_state

            content_str = message_utils.extract_string_content(response.content)
            content_str = _normalize_structured_output(
                agent_key,
                content_str,
                ticker,
                raw_data=raw_data if agent_key == "fundamentals_analyst" else "",
                foreign_data=foreign_data
                if agent_key == "fundamentals_analyst"
                else "",
                legal_data=legal_data if agent_key == "fundamentals_analyst" else "",
                management_guidance_evidence=management_guidance_evidence,
                evidence_messages=filtered_messages,
                canonical_snapshot=state.get("analysis_snapshot"),
            )

            if (
                allow_retry
                and retry_llm is not None
                and _should_retry_output(content_str, agent_key)
            ):
                logger.warning(
                    "analyst_retry_with_deep_thinking",
                    agent_key=agent_key,
                    ticker=ticker,
                    original_length=len(content_str),
                    has_datablock=has_parseable_data_block(content_str),
                    message="Insufficient or unparseable output from quick LLM, retrying once with deep thinking",
                )
                retry_messages = _build_retry_invocation_messages(
                    invocation_messages, agent_key, content_str
                )
                _retry_base = (
                    prompt_template | retry_llm.bind_tools(tools)
                    if tools
                    else prompt_template | retry_llm
                )
                # Attribute the deep-model retry cost to the ORIGINATING agent
                # (not a pooled "Retry Agent (Deep)" bucket): retry_llm carries no
                # bound token-tracking callback, so attach one per-call via config.
                retry_runnable = _retry_base.with_config(
                    {
                        "callbacks": [
                            TokenTrackingCallback(
                                canonical_display_name(agent_prompt.agent_name),
                                get_tracker(),
                            )
                        ]
                    }
                )

                # In --quick, base the retry's overall budget on the same
                # per-seat cap the main path uses (larger for gate-critical APEX
                # seats), so a retry can't clip an APEX seat below its 180s
                # allowance. Full mode keeps the standard hard cap.
                _retry_rc = get_runtime_config(settings_config)
                _retry_base_seconds = (
                    agent_runtime.quick_mode_hard_timeout_seconds(
                        agent_prompt.agent_name, settings_config
                    )
                    if _retry_rc.quick_mode_active
                    else float(_retry_rc.llm_call_hard_timeout_seconds)
                )
                try:
                    retry_response = (
                        await agent_runtime.invoke_with_rate_limit_handling(
                            retry_runnable,
                            {"messages": retry_messages},
                            context=f"{agent_prompt.agent_name} (RETRY-HIGH)",
                            canonical_agent=agent_prompt.agent_name,
                            provider=support.infer_provider_name(retry_llm),
                            model_name=support.get_model_name(retry_llm),
                            # Floor for flex: an un-floored overall budget
                            # would clamp the flex-aware hard cap back down.
                            overall_timeout_seconds=floor_llm_hard_timeout(
                                _retry_base_seconds,
                                provider=support.infer_provider_name(retry_llm),
                                label="analyst_retry_overall_timeout",
                            ),
                        )
                    )
                    retry_response.name = agent_key
                    retry_content_str = message_utils.extract_string_content(
                        retry_response.content
                    )
                    retry_content_str = _normalize_structured_output(
                        agent_key,
                        retry_content_str,
                        ticker,
                        raw_data=raw_data
                        if agent_key == "fundamentals_analyst"
                        else "",
                        foreign_data=foreign_data
                        if agent_key == "fundamentals_analyst"
                        else "",
                        legal_data=legal_data
                        if agent_key == "fundamentals_analyst"
                        else "",
                        management_guidance_evidence=management_guidance_evidence,
                        evidence_messages=filtered_messages,
                        canonical_snapshot=state.get("analysis_snapshot"),
                    )
                    retry_tool_calls = getattr(retry_response, "tool_calls", None)
                    retry_has_tool_calls = (
                        isinstance(retry_tool_calls, list) and len(retry_tool_calls) > 0
                    )

                    if retry_has_tool_calls:
                        new_state["messages"] = [retry_response]
                        logger.debug(
                            "analyst_retry_produced_tool_calls",
                            agent_key=agent_key,
                            ticker=ticker,
                        )
                        return new_state

                    logger.debug(
                        "analyst_retry_complete",
                        agent_key=agent_key,
                        ticker=ticker,
                        original_length=len(content_str),
                        retry_length=len(retry_content_str),
                        retry_has_datablock=has_parseable_data_block(retry_content_str),
                        retry_improved=len(retry_content_str) > len(content_str),
                    )
                    content_str = retry_content_str
                    response = retry_response
                except Exception as retry_error:
                    logger.error(
                        "analyst_retry_failed",
                        agent_key=agent_key,
                        ticker=ticker,
                        **summarize_exception(
                            retry_error, operation="analyst_retry_failed"
                        ),
                    )

            from src.utils import detect_truncation

            trunc_info = detect_truncation(content_str, agent=agent_key)
            log_truncation_diagnostic(
                agent_key=agent_key,
                ticker=ticker,
                runnable=llm if response is not None else llm,
                response=response,
                content=content_str,
                trunc_info=trunc_info,
            )

            validation = validate_required_output(agent_key, content_str)
            log_output_diagnostics(
                agent_key=agent_key,
                ticker=ticker,
                runnable=llm if response is not None else llm,
                response=response,
                content=content_str,
                truncated=trunc_info["truncated"],
                validation=validation if validation["checks"] else None,
            )
            if should_fail_closed(
                agent_key,
                validation=validation,
                truncated=trunc_info["truncated"],
                content=content_str,
            ):
                logger.error(
                    "analyst_invalid_structure",
                    agent=agent_key,
                    ticker=ticker,
                    missing_sections=validation["missing"],
                    validation_issues=validation.get("issues", {}),
                )
                issue_text = "; ".join(validation.get("issues", {}).values())
                failure_message = (
                    f"{agent_key} output invalid: {issue_text}"
                    if issue_text
                    else f"{agent_key} output missing required structure"
                )
                result = failure_artifact(
                    output_field,
                    failure_message,
                    provider=support.infer_provider_name(llm),
                    fallback_content=content_str,
                )
                new_state.update(result)
                return new_state

            if agent_key == "fundamentals_analyst":
                from src.analysis_snapshot import (
                    add_validated_derivations,
                    project_analysis_report,
                )

                derived_snapshot = add_validated_derivations(
                    state.get("analysis_snapshot"),
                    content_str,
                )
                content_str = project_analysis_report(
                    content_str,
                    derived_snapshot,
                )
                new_state["analysis_snapshot"] = derived_snapshot
                logger.debug(
                    "fundamentals_output",
                    has_datablock=has_parseable_data_block(content_str),
                    length=len(content_str),
                )
            new_state.update(
                success_artifact(
                    output_field,
                    cap_state_value(content_str, output_field),
                    provider=support.infer_provider_name(llm),
                )
            )
            return new_state
        except Exception as exc:
            logger.error(
                "analyst_node_error",
                output_field=output_field,
                **summarize_exception(exc, operation="analyst_node_error"),
            )
            error_message = AIMessage(content=f"Error: {str(exc)}")
            error_message.name = agent_key
            result = failure_artifact(
                output_field,
                exc,
                provider=support.infer_provider_name(llm),
            )
            result["messages"] = [error_message]
            return result

    return analyst_node


def create_valuation_calculator_node(llm) -> Callable:
    """
    Factory function creating Valuation Calculator node for chart generation.
    """

    async def valuation_calculator_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("valuation_calculator")
        if not agent_prompt:
            logger.error("missing_prompt", agent="valuation_calculator")
            return failure_artifact(
                "valuation_params",
                "Missing valuation_calculator prompt",
                provider="unknown",
            )

        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)
        fundamentals_report = state.get("fundamentals_report", "")

        if not isinstance(fundamentals_report, str):
            fundamentals_report = message_utils.extract_string_content(
                fundamentals_report
            )

        data_block = extract_last_data_block(fundamentals_report, include_markers=True)
        if not data_block:
            logger.warning(
                "valuation_calculator_no_datablock",
                ticker=ticker,
                message="No DATA_BLOCK found - skipping valuation params extraction",
            )
            return failure_artifact(
                "valuation_params",
                "DATA_BLOCK missing",
                provider=support.infer_provider_name(llm),
            )

        valuation_context = _extract_valuation_context(fundamentals_report)

        prompt = f"""{agent_prompt.system_message}

TICKER: {ticker}
COMPANY: {company_name}

VALUATION_CONTEXT:
{valuation_context}

DATA_BLOCK:
{data_block}

Extract valuation parameters and output in the required format."""

        try:
            response = await agent_runtime.invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_prompt.agent_name,
                provider=support.infer_provider_name(llm),
                model_name=support.get_model_name(llm),
            )
            content_str = message_utils.extract_string_content(response.content)
            logger.debug(
                "valuation_calculator_complete",
                ticker=ticker,
                has_params_block=has_parseable_fenced_block(
                    content_str, "VALUATION_PARAMS"
                ),
                content_length=len(content_str),
            )
            return success_artifact(
                "valuation_params",
                cap_state_value(content_str, "valuation_params"),
                provider=support.infer_provider_name(llm),
            )
        except Exception as exc:
            logger.error(
                "valuation_calculator_error",
                ticker=ticker,
                **summarize_exception(exc, operation="valuation_calculator"),
            )
            return failure_artifact(
                "valuation_params",
                exc,
                provider=support.infer_provider_name(llm),
            )

    return valuation_calculator_node
