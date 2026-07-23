"""Deterministic retrieval and normalization for off-balance-sheet exposure."""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any

import structlog

from src.data_block_utils import replace_or_append_block_line

from .evidence_preflight import PreflightOutcome, run_preflight_calls

logger = structlog.get_logger(__name__)

CAPITAL_STRUCTURE_PREFLIGHT_MAX_CHARS = 32_000
MATERIAL_EXPOSURE_TO_DEBT_PCT = 20.0
MATERIAL_EXPOSURE_TO_EQUITY_PCT = 20.0
MATERIAL_EXPOSURE_TO_REVENUE_PCT = 10.0

_VALID_FIELDS: dict[str, frozenset[str]] = {
    "coverage_status": frozenset({"FOUND", "NOT_FOUND", "SEARCH_FAILED", "UNRESOLVED"}),
    "exposure_type": frozenset(
        {
            "NONE",
            "LEASE_COMMITMENT",
            "TAKE_OR_PAY",
            "PURCHASE_COMMITMENT",
            "GUARANTEE_BACKSTOP",
            "UNCONSOLIDATED_ENTITY",
            "MULTIPLE",
            "UNKNOWN",
        }
    ),
    "amount_basis": frozenset(
        {
            "PRESENT_VALUE",
            "UNDISCOUNTED",
            "MAXIMUM_EXPOSURE",
            "PROPORTIONATE",
            "UNKNOWN",
            "N/A",
        }
    ),
    "balance_sheet_status": frozenset(
        {"RECOGNIZED", "UNRECOGNIZED", "PARTLY_RECOGNIZED", "UNKNOWN"}
    ),
    "parent_recourse": frozenset({"NONE", "LIMITED", "FULL", "UNKNOWN"}),
    "consolidation_risk": frozenset(
        {
            "NONE",
            "PRIMARY_BENEFICIARY_AMBIGUITY",
            "CONTROL_AMBIGUITY",
            "RELATED_PARTY_CONCERN",
            "UNKNOWN",
        }
    ),
    "materiality": frozenset({"MATERIAL", "IMMATERIAL", "UNKNOWN"}),
}

_CAPITAL_DEFAULTS: dict[str, Any] = {
    "coverage_status": "UNRESOLVED",
    "exposure_type": "UNKNOWN",
    "entity": "N/A",
    "amount": "N/A",
    "amount_basis": "UNKNOWN",
    "balance_sheet_status": "UNKNOWN",
    "parent_recourse": "UNKNOWN",
    "consolidation_risk": "UNKNOWN",
    "materiality": "UNKNOWN",
    "source_url": "N/A",
    "evidence": "Capital-structure assessment unavailable.",
}

_PRIORITY_TERMS = [
    "off-balance-sheet",
    "unconsolidated",
    "variable interest entity",
    "special purpose entity",
    "guarantee",
    "backstop",
    "keepwell",
    "take-or-pay",
    "purchase commitment",
    "lease commitment",
    "related party",
    "continuing involvement",
    "債務保証",
    "偶発債務",
    "채무보증",
    "우발채무",
    "或有负债",
    "债务担保",
]


def _capital_structure_queries(ticker: str, company_name: str) -> tuple[str, str]:
    subject = f"{ticker} {company_name}".strip()
    structures = (
        f'{subject} filing "off-balance-sheet" unconsolidated VIE SPV SPE '
        "joint venture guarantee backstop keepwell continuing involvement "
        "primary beneficiary related party 債務保証 関連会社 채무보증 특수관계자 "
        "债务担保 关联企业 表外融资 特殊目的实体"
    )
    commitments = (
        f'{subject} filing commitments contingencies "take-or-pay" '
        '"purchase commitments" "uncommenced leases" debt guarantee '
        "maximum exposure 偶発債務 リース契約 우발채무 리스 약정 或有负债 租赁承诺"
    )
    return structures, commitments


def _status_map(outcomes: list[PreflightOutcome]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for label, payload in outcomes:
        match = re.search(r"^STATUS:\s+([A-Z_]+)", payload)
        statuses[label] = match.group(1) if match else "UNKNOWN"
    return statuses


def _fallback_coverage(evidence: str) -> str:
    statuses = re.findall(
        r"(?ms)^####\s+[a-z0-9_]+\s*$.*?^STATUS:\s+([A-Z_]+)", evidence
    )
    if any(status == "COMPLETED" for status in statuses):
        return "UNRESOLVED"
    return "SEARCH_FAILED"


async def preload_capital_structure_evidence(
    ticker: str,
    company_name: str,
    *,
    tools_by_name: dict[str, Any] | None = None,
) -> str:
    """Retrieve mandatory filing and search evidence through inspected tools."""
    if tools_by_name is None:
        from src.tools.official_documents import get_official_document
        from src.tools.research import get_official_filings, search_foreign_sources

        tools_by_name = {
            get_official_document.name: get_official_document,
            get_official_filings.name: get_official_filings,
            search_foreign_sources.name: search_foreign_sources,
        }

    started_at = time.monotonic()
    structures, commitments = _capital_structure_queries(ticker, company_name)
    filing_tool = tools_by_name.get("get_official_filings")
    search_tool = tools_by_name.get("search_foreign_sources")
    document_tool = tools_by_name.get("get_official_document")
    calls = []
    if filing_tool is not None:
        calls.append(("statutory_filing_api", filing_tool, {"ticker": ticker}))
    if search_tool is not None:
        calls.extend(
            [
                (
                    "structures_search",
                    search_tool,
                    {
                        "ticker": ticker,
                        "search_query": structures,
                        "priority_terms": _PRIORITY_TERMS,
                    },
                ),
                (
                    "commitments_search",
                    search_tool,
                    {
                        "ticker": ticker,
                        "search_query": commitments,
                        "priority_terms": _PRIORITY_TERMS,
                    },
                ),
            ]
        )
    outcomes, durations = await run_preflight_calls(
        calls,
        agent_key="legal_counsel",
        source="legal_counsel_preflight",
        ticker=ticker,
        failure_event="capital_structure_preflight_call_failed",
        logger=logger,
    )
    labels = {label for label, _ in outcomes}
    for missing_label in (
        "statutory_filing_api",
        "structures_search",
        "commitments_search",
    ):
        if missing_label not in labels:
            outcomes.append((missing_label, "STATUS: SKIPPED (TOOL_UNAVAILABLE)"))

    search_payload = "\n".join(
        payload for label, payload in outcomes if "search" in label
    )
    candidate_urls = list(
        dict.fromkeys(re.findall(r"https?://[^\s<>'\"]+", search_payload))
    )[:2]
    if candidate_urls and document_tool is not None:
        document_calls = [
            (
                f"official_document_{index}",
                document_tool,
                {"url": url, "keywords": " ".join(_PRIORITY_TERMS)},
            )
            for index, url in enumerate(candidate_urls, start=1)
        ]
        document_outcomes, document_durations = await run_preflight_calls(
            document_calls,
            agent_key="legal_counsel",
            source="legal_counsel_preflight",
            ticker=ticker,
            failure_event="capital_structure_preflight_call_failed",
            logger=logger,
        )
        outcomes.extend(document_outcomes)
        durations.update(document_durations)
    else:
        reason = "TOOL_UNAVAILABLE" if document_tool is None else "NO_CANDIDATE_URLS"
        outcomes.append(("official_document", f"STATUS: SKIPPED ({reason})"))

    sections = [
        "### CODE-OWNED CAPITAL STRUCTURE PREFLIGHT",
        "SOURCE_CLASSES_TARGETED: STATUTORY_FILING; OFFICIAL_IR; EXCHANGE_DISCLOSURE; NEWS_CORROBORATION",
        "PROVENANCE_RULE: Treat only returned filing text or source-linked search evidence as findings. "
        "A query term appearing by itself is not evidence of an exposure.",
    ]
    for label, payload in outcomes:
        sections.extend((f"\n#### {label}", payload))
    evidence = "\n".join(sections)
    logger.info(
        "capital_structure_preflight_complete",
        ticker=ticker,
        elapsed_ms=round((time.monotonic() - started_at) * 1000),
        call_durations_ms=durations,
        call_statuses=_status_map(outcomes),
        evidence_chars=len(evidence),
    )
    if len(evidence) <= CAPITAL_STRUCTURE_PREFLIGHT_MAX_CHARS:
        return evidence
    tail_size = 4_000
    head_size = CAPITAL_STRUCTURE_PREFLIGHT_MAX_CHARS - tail_size
    removed = len(evidence) - CAPITAL_STRUCTURE_PREFLIGHT_MAX_CHARS
    return (
        evidence[:head_size]
        + f"\n[...preflight aggregate omitted {removed:,} chars...]\n"
        + evidence[-tail_size:]
    )


def _extract_json_object(content: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            value, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _as_finite_float(value: Any, *, allow_zero: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    if parsed > 0 or (allow_zero and parsed == 0):
        return parsed
    return None


def _parse_exposure_amount(raw_amount: Any) -> tuple[float, str] | None:
    """Parse an ISO-currency amount while preserving magnitude conservatively."""
    text = str(raw_amount or "").strip().upper().replace(",", "")
    match = re.fullmatch(
        r"(?P<currency>[A-Z]{3})\s+(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?P<magnitude>TRILLION|TN|T|BILLION|BN|B|MILLION|MN|M)?",
        text,
    )
    if match is None:
        return None
    multipliers = {
        None: 1.0,
        "M": 1e6,
        "MN": 1e6,
        "MILLION": 1e6,
        "B": 1e9,
        "BN": 1e9,
        "BILLION": 1e9,
        "T": 1e12,
        "TN": 1e12,
        "TRILLION": 1e12,
    }
    value = float(match.group("value")) * multipliers[match.group("magnitude")]
    if not math.isfinite(value) or value <= 0:
        return None
    return value, match.group("currency")


def assess_capital_structure_scale(
    capital: dict[str, Any],
    raw_metrics: dict[str, Any],
    *,
    leverage_threshold: float | None,
) -> dict[str, Any]:
    """Measure exposure against same-currency debt, implied equity, and revenue."""
    parsed_exposure = _parse_exposure_amount(capital.get("amount"))
    amount_basis = str(capital.get("amount_basis", "UNKNOWN")).upper()
    if parsed_exposure is None or amount_basis not in {
        "PRESENT_VALUE",
        "MAXIMUM_EXPOSURE",
        "PROPORTIONATE",
    }:
        return {"status": "UNRESOLVED", "reason": "AMOUNT_OR_BASIS_UNUSABLE"}

    exposure, exposure_currency = parsed_exposure
    financial_currency = str(
        raw_metrics.get("financialCurrency") or raw_metrics.get("currency") or ""
    ).upper()
    if not financial_currency or financial_currency != exposure_currency:
        return {"status": "UNRESOLVED", "reason": "CURRENCY_MISMATCH"}

    total_debt = _as_finite_float(raw_metrics.get("totalDebt"), allow_zero=True)
    if total_debt is None:
        return {"status": "UNRESOLVED", "reason": "DEBT_UNAVAILABLE"}

    equity = next(
        (
            value
            for key in (
                "stockholdersEquity",
                "totalStockholderEquity",
                "totalEquity",
            )
            if (value := _as_finite_float(raw_metrics.get(key))) is not None
        ),
        None,
    )
    raw_de_ratio = _as_finite_float(raw_metrics.get("debtToEquity"))
    if raw_de_ratio is not None and raw_de_ratio > 10:
        raw_de_ratio /= 100.0
    if equity is None and total_debt > 0 and raw_de_ratio is not None:
        equity = total_debt / raw_de_ratio
    if equity is None or not math.isfinite(equity) or equity <= 0:
        return {"status": "UNRESOLVED", "reason": "EQUITY_UNAVAILABLE"}

    revenue = _as_finite_float(
        raw_metrics.get("revenue_TTM") or raw_metrics.get("totalRevenue")
    )
    exposure_to_debt = exposure / total_debt * 100 if total_debt > 0 else None
    exposure_to_equity = exposure / equity * 100
    exposure_to_revenue = exposure / revenue * 100 if revenue else None
    reported_de = total_debt / equity * 100
    adjusted_de = (total_debt + exposure) / equity * 100
    crosses_red_zone = bool(
        leverage_threshold is not None
        and reported_de <= leverage_threshold < adjusted_de
    )
    decision_material = (
        (
            exposure_to_debt is not None
            and exposure_to_debt >= MATERIAL_EXPOSURE_TO_DEBT_PCT
        )
        or exposure_to_equity >= MATERIAL_EXPOSURE_TO_EQUITY_PCT
        or (
            exposure_to_revenue is not None
            and exposure_to_revenue >= MATERIAL_EXPOSURE_TO_REVENUE_PCT
        )
        or crosses_red_zone
    )
    return {
        "status": "MEASURABLE",
        "currency": exposure_currency,
        "exposure_to_debt_pct": (
            round(exposure_to_debt, 1) if exposure_to_debt is not None else None
        ),
        "exposure_to_equity_pct": round(exposure_to_equity, 1),
        "exposure_to_revenue_pct": (
            round(exposure_to_revenue, 1) if exposure_to_revenue is not None else None
        ),
        "reported_de_pct": round(reported_de, 1),
        "adjusted_de_pct": round(adjusted_de, 1),
        "leverage_threshold_pct": leverage_threshold,
        "crosses_red_zone": crosses_red_zone,
        "decision_material": decision_material,
    }


def classify_capital_structure(capital: dict[str, Any]) -> str:
    """Derive decision treatment from normalized accounting attributes."""
    if capital["coverage_status"] == "SEARCH_FAILED":
        return "UNRESOLVED"
    if capital["exposure_type"] == "NONE" and capital["coverage_status"] in {
        "FOUND",
        "NOT_FOUND",
    }:
        return "CLEAR"
    if capital["materiality"] == "IMMATERIAL":
        return "CLEAR"
    if capital["balance_sheet_status"] == "RECOGNIZED":
        return "CLEAR"

    material = capital["materiality"] != "IMMATERIAL"
    unrecognized = capital["balance_sheet_status"] in {
        "UNRECOGNIZED",
        "PARTLY_RECOGNIZED",
        "UNKNOWN",
    }
    recourse = capital["parent_recourse"] in {"LIMITED", "FULL"}
    consolidation_concern = capital["consolidation_risk"] in {
        "PRIMARY_BENEFICIARY_AMBIGUITY",
        "CONTROL_AMBIGUITY",
        "RELATED_PARTY_CONCERN",
    }
    if material and unrecognized and (recourse or consolidation_concern):
        return "BLOCK_BUY"
    if (
        capital["exposure_type"]
        in {
            "LEASE_COMMITMENT",
            "TAKE_OR_PAY",
            "PURCHASE_COMMITMENT",
        }
        and material
    ):
        return "QUALIFY_RATIOS"
    if (
        capital["exposure_type"] == "UNCONSOLIDATED_ENTITY"
        and capital["parent_recourse"] == "NONE"
        and capital["consolidation_risk"] == "NONE"
        and material
    ):
        return "QUALIFY_RATIOS"
    return "UNRESOLVED"


def normalize_legal_output(
    content: str,
    evidence: str,
) -> tuple[str | None, bool]:
    """Normalize nested capital-structure JSON without treating omission as clean."""
    payload = _extract_json_object(content)
    if payload is None:
        return None, False

    raw_capital = payload.get("capital_structure")
    capital = dict(_CAPITAL_DEFAULTS)
    if isinstance(raw_capital, dict):
        for key in capital:
            value = raw_capital.get(key)
            if value is not None:
                capital[key] = value
    else:
        capital["coverage_status"] = _fallback_coverage(evidence)
        capital["evidence"] = (
            "Legal Counsel omitted the required capital-structure assessment; "
            "code-owned search coverage is preserved as unresolved."
        )

    for field, allowed in _VALID_FIELDS.items():
        value = str(capital.get(field, "")).strip().upper()
        capital[field] = value if value in allowed else _CAPITAL_DEFAULTS[field]
    for field in ("entity", "amount", "source_url", "evidence"):
        value = capital.get(field)
        capital[field] = str(value).strip() if value else str(_CAPITAL_DEFAULTS[field])

    if capital["coverage_status"] == "UNRESOLVED" and raw_capital is None:
        capital["coverage_status"] = _fallback_coverage(evidence)
    capital["classification"] = classify_capital_structure(capital)
    payload["capital_structure"] = capital
    return json.dumps(payload, ensure_ascii=False), isinstance(raw_capital, dict)


def promote_capital_structure(body: str, legal_report: str) -> tuple[str, bool]:
    """Copy the normalized Legal assessment into Senior Fundamentals DATA_BLOCK."""
    payload = _extract_json_object(legal_report)
    capital = payload.get("capital_structure") if payload else None
    if not isinstance(capital, dict):
        return body, False

    classification = str(capital.get("classification", "UNRESOLVED"))
    exposure_type = str(capital.get("exposure_type", "UNKNOWN"))
    amount = str(capital.get("amount", "N/A"))
    basis = str(capital.get("amount_basis", "UNKNOWN"))
    entity = str(capital.get("entity", "N/A"))
    evidence = str(capital.get("evidence", "Assessment unavailable."))
    if classification == "CLEAR":
        qualification = (
            "No material unrecognized debt-like exposure identified in targeted search."
        )
    elif classification == "QUALIFY_RATIOS":
        qualification = (
            f"Reported D/E excludes or may not reflect {exposure_type} exposure "
            f"for {entity}; evaluate separately ({amount}, {basis})."
        )
    elif classification == "BLOCK_BUY":
        qualification = (
            f"Reported D/E may omit parent-recourse or consolidation-risk exposure "
            f"for {entity}; compare it with debt, equity, and revenue before sizing "
            f"or initiating ({amount}, {basis})."
        )
    else:
        qualification = f"Capital-structure exposure remains unresolved: {evidence}"

    updated = replace_or_append_block_line(
        body, "CAPITAL_STRUCTURE_CLASSIFICATION", classification
    )
    updated = replace_or_append_block_line(
        updated, "DE_RATIO_QUALIFICATION", qualification
    )
    updated = replace_or_append_block_line(
        updated,
        "INCREMENTAL_DEBT_LIKE_EXPOSURE",
        f"{amount} ({basis}; {exposure_type})" if amount != "N/A" else "N/A",
    )
    return updated, updated != body
