from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from src.data_block_utils import (
    extract_block_number_from_text,
    extract_block_text_value,
    has_block_field_value,
    has_non_na_block_field_value,
    replace_or_append_block_line,
)
from src.earnings_baseline import requires_eps_growth_withholding
from src.sector_normalization import normalize_sector_label
from src.thesis_constants import (
    FINANCIALS_HEALTH_REMOVED_POINTS,
    GROWTH_RUBRIC_POINTS,
    GROWTH_SCORE_CRITERIA,
    HEALTH_RUBRIC_POINTS,
    HEALTH_SCORE_CRITERIA,
    PE_MAX,
    PEG_MAX,
    SCORE_PCT_TOLERANCE,
)
from src.validators.pfic_constants import (
    PFIC_ASSET_PROXIMITY_THRESHOLD,
    PFIC_ASSET_TEST_THRESHOLD,
)

HORIZON_FIELD_RAW_KEYS = (
    ("REVENUE_GROWTH_FY", "revenueGrowth"),
    ("EARNINGS_GROWTH_FY", "earningsGrowth"),
    ("REVENUE_GROWTH_TTM", "revenueGrowth_TTM"),
    ("REVENUE_GROWTH_MRQ", "revenueGrowth_MRQ"),
    ("EARNINGS_GROWTH_TTM", "earningsGrowth_TTM"),
    ("EARNINGS_GROWTH_MRQ", "earningsGrowth_MRQ"),
    ("GROWTH_TRAJECTORY", "growth_trajectory"),
)

_PERIOD_COMPARABLE_GROWTH_FIELDS = (
    ("REVENUE_GROWTH_TTM", "revenueGrowth_TTM"),
    ("REVENUE_GROWTH_MRQ", "revenueGrowth_MRQ"),
    ("EARNINGS_GROWTH_TTM", "earningsGrowth_TTM"),
    ("EARNINGS_GROWTH_MRQ", "earningsGrowth_MRQ"),
)


def _growth_source_label(
    payload: dict[str, Any],
    field: str,
) -> str:
    field_sources = payload.get("_field_sources")
    if not isinstance(field_sources, dict):
        field_sources = {}
    source = str(
        payload.get(f"_{field}_source") or field_sources.get(field) or ""
    ).lower()
    if field == "revenueGrowth":
        if source == "calculated_from_statements":
            return "ANNUAL_STATEMENTS"
    else:
        source_labels = {
            "calculated_from_statement_diluted_eps": "DILUTED_EPS_STATEMENTS",
            "calculated_from_statement_basic_eps": "BASIC_EPS_STATEMENTS",
            "calculated_from_statement_net_income_proxy": (
                "NET_INCOME_STATEMENT_PROXY"
            ),
        }
        if source in source_labels:
            return source_labels[source]

    statement_overrides = payload.get("_statement_overrides")
    if isinstance(statement_overrides, dict) and field in statement_overrides:
        return "ANNUAL_STATEMENTS" if field == "revenueGrowth" else "STATEMENT_DERIVED"
    if source:
        return "AGGREGATOR"
    return "UNKNOWN"


_RAW_METRICS_MARKER = re.compile(
    r"###\s*TOOL\s*\d+:\s*get_financial_metrics",
    re.IGNORECASE,
)
_HIGH_LOCAL_COVERAGE_PATTERN = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?"
    r"(?:LOCAL_ANALYST_COVERAGE|Estimated Local Analysts)\s*:\s*"
    r"[^\n]*(?:\b(?:HIGH|MODERATE)\b|~?[1-9]\d*)"
)


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "").rstrip("%")
        if not text or text.upper() in {"N/A", "NA", "NONE", "NULL"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def format_ratio(value: float, *, decimals: int = 2) -> str:
    formatted = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return formatted or "0"


def format_percent_from_ratio(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_percent(value: float) -> str:
    """Format an already-percent number (e.g. 36.85 -> '36.9%')."""
    return f"{value:.1f}%"


def material_diff(
    body: str,
    key: str,
    expected: float,
    *,
    threshold: float,
    rel_threshold: float | None = None,
) -> bool:
    current = extract_block_number_from_text(body, key)
    if current is None:
        return True
    if rel_threshold is not None and expected:
        return abs(current - expected) / abs(expected) > rel_threshold
    return abs(current - expected) > threshold


def extract_raw_metrics_payload(raw_data: str) -> dict[str, Any]:
    """Extract get_financial_metrics payload from JSON or production raw text."""
    if not raw_data:
        return {}

    try:
        payload = json.loads(raw_data)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None

    if isinstance(payload, dict):
        return payload

    marker = _RAW_METRICS_MARKER.search(raw_data)
    if marker is None:
        return {}

    search_from = raw_data.find("{", marker.end())
    if search_from < 0:
        return {}

    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(raw_data[search_from:])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def statement_mrq_period_lag_note(payload: dict[str, Any]) -> str | None:
    """Return the fetcher's code-owned warning for period-bound stale MRQ metrics."""
    notes = payload.get("_data_quality_notes")
    candidates = notes if isinstance(notes, list) else [notes]
    for candidate in candidates:
        note = str(candidate or "").strip()
        if (
            "Newer quarter metadata exists" in note
            and "statement-derived MRQ metrics remain aligned" in note
        ):
            return (
                f"{note} Treat these MRQ metrics as period-bound trailing indicators, "
                "not the latest reported quarter."
            )
    return None


def _reconcile_numeric_field(
    body: str,
    key: str,
    value: float | None,
    *,
    threshold: float,
    formatter: Callable[[float], str],
) -> tuple[str, bool]:
    if value is None:
        if has_non_na_block_field_value(body, key):
            return replace_or_append_block_line(body, key, "N/A"), True
        return body, False

    if has_block_field_value(body, key) and material_diff(
        body,
        key,
        value,
        threshold=threshold,
    ):
        return replace_or_append_block_line(body, key, formatter(value)), True
    return body, False


def _reconcile_when_present(
    body: str,
    key: str,
    value: float | None,
    *,
    rel_threshold: float,
    formatter: Callable[[float], str],
) -> tuple[str, bool]:
    """Correct a DATA_BLOCK scalar only when the raw value exists AND diverges.

    Unlike ``_reconcile_numeric_field``, this never erases a field to ``N/A`` when the
    raw payload lacks it — valuation/margin/payout values may be legitimately
    filing-derived by the Senior Fundamentals agent, so a missing raw value is not
    evidence the DATA_BLOCK value is wrong.
    """
    if value is None:
        return body, False
    if has_block_field_value(body, key) and material_diff(
        body,
        key,
        value,
        threshold=0.0,
        rel_threshold=rel_threshold,
    ):
        return replace_or_append_block_line(body, key, formatter(value)), True
    return body, False


def reconcile_high_risk_fields(
    body: str,
    payload: dict[str, Any],
) -> str:
    updated = body
    changed_growth = False
    changed_balance_sheet = False
    changed_valuation = False

    for datablock_key, raw_key in (
        ("REVENUE_GROWTH_FY", "revenueGrowth"),
        ("EARNINGS_GROWTH_FY", "earningsGrowth"),
    ):
        value = as_float(payload.get(raw_key))
        reconciled_value = (
            format_percent_from_ratio(value) if value is not None else "N/A"
        )
        if extract_block_text_value(updated, datablock_key) != reconciled_value:
            updated = replace_or_append_block_line(
                updated, datablock_key, reconciled_value
            )
            changed_growth = True
        updated = replace_or_append_block_line(
            updated,
            f"{datablock_key}_SOURCE",
            _growth_source_label(payload, raw_key) if value is not None else "UNKNOWN",
        )

    for datablock_key, raw_key in (
        ("REVENUE_GROWTH_TTM", "revenueGrowth_TTM"),
        ("EARNINGS_GROWTH_TTM", "earningsGrowth_TTM"),
    ):
        if (
            payload.get(raw_key) is None
            and extract_block_text_value(updated, datablock_key).upper() != "N/A"
        ):
            updated = replace_or_append_block_line(updated, datablock_key, "N/A")
            changed_growth = True

    for datablock_key, raw_key, formatter, threshold in (
        ("SECTOR_MEDIAN_PE", "sectorMedianPE", format_ratio, 0.01),
        ("PE_VS_SECTOR", "peVsSector", format_ratio, 0.01),
        ("REVENUE_CAGR_3Y", "revenue_cagr_3y", format_percent_from_ratio, 0.1),
        ("FCF_CAGR_3Y", "fcf_cagr_3y", format_percent_from_ratio, 0.1),
    ):
        updated, changed = _reconcile_numeric_field(
            updated,
            datablock_key,
            as_float(payload.get(raw_key)),
            threshold=threshold,
            formatter=formatter,
        )
        changed_growth = changed_growth or changed

    reference_type = str(payload.get("sectorPeReferenceType") or "UNKNOWN").upper()
    if reference_type not in {
        "STATIC_POLICY_REFERENCE",
        "LIVE_MARKET_REFERENCE",
        "UNKNOWN",
    }:
        reference_type = "UNKNOWN"
    updated = replace_or_append_block_line(
        updated,
        "SECTOR_PE_REFERENCE_TYPE",
        reference_type,
    )
    reference_as_of = payload.get("sectorPeReferenceAsOf")
    updated = replace_or_append_block_line(
        updated,
        "SECTOR_PE_REFERENCE_AS_OF",
        str(reference_as_of) if reference_as_of else "N/A",
    )

    cycle_position = str(payload.get("cycle_position") or "").upper()
    if cycle_position in {"PEAK", "MID", "TROUGH"} and (
        extract_block_text_value(updated, "CYCLE_POSITION").upper() != cycle_position
    ):
        updated = replace_or_append_block_line(
            updated, "CYCLE_POSITION", cycle_position
        )
        changed_growth = True

    comparison_status = str(
        payload.get("mrq_comparison_base_status") or "UNKNOWN"
    ).upper()
    if comparison_status not in {
        "DEPRESSED",
        "ELEVATED",
        "NORMAL",
        "NONPOSITIVE",
        "UNKNOWN",
    }:
        comparison_status = "UNKNOWN"
    updated = replace_or_append_block_line(
        updated,
        "MRQ_COMPARISON_BASE_STATUS",
        comparison_status,
    )
    comparison_delta = as_float(payload.get("mrq_comparison_base_margin_delta_bps"))
    updated = replace_or_append_block_line(
        updated,
        "MRQ_COMPARISON_BASE_MARGIN_DELTA_BPS",
        f"{comparison_delta:.1f}" if comparison_delta is not None else "N/A",
    )

    # 5-year return averages feed cyclical-peak detection; the LLM can drop or
    # mis-copy them. Promote the computed signals (already percent-scaled) when the
    # payload carries them, so the DATA_BLOCK stays consistent with their sibling
    # CYCLE_POSITION/PROFITABILITY_TREND. ``_reconcile_when_present`` never erases a
    # value the agent may have filing-derived when the raw signal is absent; a 5%
    # relative tolerance avoids churn from float-format differences.
    for datablock_key, raw_key in (
        ("ROA_5Y_AVG", "roa_5y_avg"),
        ("ROE_5Y_AVG", "roe_5y_avg"),
    ):
        updated, changed = _reconcile_when_present(
            updated,
            datablock_key,
            as_float(payload.get(raw_key)),
            rel_threshold=0.05,
            formatter=lambda pct: f"{pct:.2f}%",
        )
        changed_growth = changed_growth or changed

    # Valuation/margin scalars: the Senior Fundamentals LLM can emit a value that
    # contradicts the fetched raw metrics (e.g. a fabricated PE_RATIO_TTM copied from
    # EV/EBITDA). Reconcile against the raw payload when present — never erase a value
    # the agent may have legitimately filing-derived (raw missing -> leave intact).
    # ``scale`` brings ratio-valued raw fields into the DATA_BLOCK's display units so
    # ``material_diff`` compares like-for-like (percent vs percent, ratio vs ratio).
    # NOTE on authority: PE_RATIO_TTM/FORWARD and PB_RATIO are pure market-price ratios
    # the aggregator computes authoritatively, so this is straight integrity hardening.
    # PAYOUT_RATIO and NET_MARGIN are *raw-metric* reconciliations (the policy here is
    # "DATA_BLOCK scalars must match fetched metrics"), NOT filing-authority
    # reconciliations — a wide tolerance (20%) is used so only egregious divergence
    # overrides a possibly period-mismatched filing-derived value.
    valuation_specs: tuple[
        tuple[str, str, float, Callable[[float], str], float, bool], ...
    ] = (
        ("PE_RATIO_TTM", "trailingPE", 1.0, format_ratio, 0.15, True),
        ("PE_RATIO_FORWARD", "forwardPE", 1.0, format_ratio, 0.15, False),
        ("PB_RATIO", "priceToBook", 1.0, format_ratio, 0.15, False),
        ("PAYOUT_RATIO", "payoutRatio", 100.0, format_percent, 0.20, False),
        ("NET_MARGIN", "profitMargins", 100.0, format_percent, 0.20, False),
    )
    pe_quarantined = bool(payload.get("_pe_low_anomaly_quarantined"))
    for datablock_key, raw_key, scale, formatter, rel, is_pe in valuation_specs:
        if is_pe and pe_quarantined:
            # Leave PE_RATIO_TTM to the downstream quarantine -> N/A path.
            continue
        raw_value = as_float(payload.get(raw_key))
        value = raw_value * scale if raw_value is not None else None
        updated, changed = _reconcile_when_present(
            updated,
            datablock_key,
            value,
            rel_threshold=rel,
            formatter=formatter,
        )
        changed_valuation = changed_valuation or changed

    # Honest payout: a zero/absent aggregator ``payoutRatio`` on a name whose
    # provider dividend fields show an actual distribution is a data gap, not a
    # 0% policy. Asserting PAYOUT_RATIO: 0.0% there manufactures a contradiction
    # against the real dividend (the 6831.HK HK$0.52 dispute). Emit N/A + a
    # data-quality note instead, claiming only what the provider fields show.
    payout_raw = as_float(payload.get("payoutRatio"))
    has_provider_dividend = any(
        (as_float(payload.get(key)) or 0) > 0
        for key in ("dividendRate", "lastDividendValue", "trailingAnnualDividendRate")
    )
    if (
        has_provider_dividend
        and (payout_raw is None or payout_raw == 0)
        and extract_block_text_value(updated, "PAYOUT_RATIO") not in ("N/A", "")
    ):
        updated = replace_or_append_block_line(updated, "PAYOUT_RATIO", "N/A")
        updated = replace_or_append_block_line(updated, "DIVIDEND_COVERAGE", "N/A")
        updated = replace_or_append_block_line(
            updated,
            "DIVIDEND_DATA_QUALITY_NOTE",
            "Provider dividend fields present (dividendRate/lastDividendValue) "
            "but payout ratio is unavailable; payout not asserted.",
        )
        changed_valuation = True

    total_debt = as_float(payload.get("totalDebt"))
    cash_and_short_term = as_float(payload.get("cashAndShortTermInvestments"))
    ebitda = as_float(payload.get("ebitda"))
    market_cap = as_float(payload.get("marketCap"))
    total_assets = as_float(payload.get("totalAssets"))
    capital_cash_to_assets = as_float(payload.get("capital_cashToAssets"))

    net_debt_ebitda = (
        (total_debt - cash_and_short_term) / ebitda
        if total_debt is not None and cash_and_short_term is not None and ebitda
        else None
    )
    updated, changed = _reconcile_numeric_field(
        updated,
        "NET_DEBT_EBITDA",
        net_debt_ebitda,
        threshold=0.05,
        formatter=format_ratio,
    )
    changed_balance_sheet = changed_balance_sheet or changed

    net_cash_to_market_cap = (
        (cash_and_short_term - total_debt) / market_cap * 100
        if total_debt is not None and cash_and_short_term is not None and market_cap
        else None
    )
    updated, changed = _reconcile_numeric_field(
        updated,
        "NET_CASH_TO_MARKET_CAP",
        net_cash_to_market_cap,
        threshold=5.0,
        formatter=lambda percent: f"{percent:.1f}%",
    )
    changed_balance_sheet = changed_balance_sheet or changed

    cash_to_assets_ratio = capital_cash_to_assets
    if (
        cash_to_assets_ratio is None
        and cash_and_short_term is not None
        and total_assets
    ):
        cash_to_assets_ratio = cash_and_short_term / total_assets

    cash_to_assets_percent = (
        cash_to_assets_ratio * 100 if cash_to_assets_ratio is not None else None
    )
    for key in ("CASH_TO_ASSETS", "PFIC_ASSET_RATIO"):
        updated, changed = _reconcile_numeric_field(
            updated,
            key,
            cash_to_assets_percent,
            threshold=5.0,
            formatter=lambda percent: f"{percent:.1f}%",
        )
        changed_balance_sheet = changed_balance_sheet or changed

    if cash_to_assets_ratio is not None:
        if (
            PFIC_ASSET_PROXIMITY_THRESHOLD
            <= cash_to_assets_ratio
            < PFIC_ASSET_TEST_THRESHOLD
        ):
            current_pfic_risk = (
                extract_block_text_value(updated, "PFIC_RISK").upper()
                if has_block_field_value(updated, "PFIC_RISK")
                else ""
            )
            if current_pfic_risk in {"", "N/A", "LOW"}:
                updated = replace_or_append_block_line(updated, "PFIC_RISK", "MEDIUM")
                changed_balance_sheet = True

        expected_cash_trap = (
            "YES" if cash_to_assets_ratio >= PFIC_ASSET_TEST_THRESHOLD else "NO"
        )
        if has_block_field_value(updated, "PFIC_CASH_TRAP") and (
            extract_block_text_value(updated, "PFIC_CASH_TRAP").upper()
            != expected_cash_trap
        ):
            updated = replace_or_append_block_line(
                updated, "PFIC_CASH_TRAP", expected_cash_trap
            )
            changed_balance_sheet = True
    elif has_non_na_block_field_value(updated, "PFIC_CASH_TRAP"):
        updated = replace_or_append_block_line(updated, "PFIC_CASH_TRAP", "N/A")
        changed_balance_sheet = True

    if cash_to_assets_ratio is None and changed_balance_sheet:
        updated = replace_or_append_block_line(
            updated,
            "PFIC_ASSET_NOTE",
            "Cash/assets basis unreliable in raw payload; PFIC asset test not asserted.",
        )

    if changed_growth:
        updated = replace_or_append_block_line(
            updated,
            "GROWTH_DATA_QUALITY_NOTE",
            "Growth horizons and provenance reconciled to raw metrics; unavailable "
            "TTM/MRQ values were not backfilled from FY data.",
        )
    missing_growth_fields = [
        label
        for label, raw_key in _PERIOD_COMPARABLE_GROWTH_FIELDS
        if payload.get(raw_key) is None
    ]
    if missing_growth_fields:
        existing_note = extract_block_text_value(
            updated,
            "GROWTH_DATA_QUALITY_NOTE",
        )
        missing_note = (
            "Missing period-comparable growth inputs: "
            f"{', '.join(missing_growth_fields)}. Their absence is a data gap, "
            "not evidence of acceleration, deceleration, or structural contraction."
        )
        combined_note = (
            f"{existing_note} {missing_note}".strip()
            if missing_note not in existing_note
            else existing_note
        )
        updated = replace_or_append_block_line(
            updated,
            "GROWTH_DATA_QUALITY_NOTE",
            combined_note,
        )
    # yfinance can lag a full fiscal year for some ex-US names: the latest annual
    # statements predate the most recent completed FY, so any FY-based growth may be
    # out of date. Surface it deterministically (the data layer sets statements_stale)
    # rather than let a stale FY figure read as current.
    if payload.get("statements_stale"):
        as_of = payload.get("_income_statement_date") or "unknown date"
        updated = replace_or_append_block_line(
            updated,
            "GROWTH_DATA_STALE",
            (
                f"Latest annual statements (as of {as_of}) predate the most recent "
                "completed fiscal year; reported FY growth may not reflect the latest "
                "year — treat the growth read as data-limited."
            ),
        )
    if changed_balance_sheet:
        updated = replace_or_append_block_line(
            updated,
            "BALANCE_SHEET_DATA_QUALITY_NOTE",
            "High-risk balance-sheet fields reconciled to raw get_financial_metrics basis.",
        )
    if changed_valuation:
        updated = replace_or_append_block_line(
            updated,
            "VALUATION_DATA_QUALITY_NOTE",
            "Valuation/margin scalars reconciled to fetched raw metrics.",
        )

    return updated


_SCORE_RUBRIC_TOTALS: tuple[tuple[str, float], ...] = (
    ("HEALTH", HEALTH_RUBRIC_POINTS),
    ("GROWTH", GROWTH_RUBRIC_POINTS),
)
_FRACTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")
# Both observed parentheticals: "(based on 10 available points)" and "(7/10 available)".
_ADJUSTED_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*%"
    r"(?:\s*\(\s*(?:based on\s*)?(?:(\d+(?:\.\d+)?)\s*/\s*)?(\d+(?:\.\d+)?)\s*available[^)]*\))?",
    re.IGNORECASE,
)
_DE_REMOVED_RE = re.compile(r"(?i)D/?E\b[^.\n;]{0,60}(?:remov|not applicable|excluded)")


def _parse_raw_score(body: str, kind: str) -> tuple[float, float] | None:
    match = _FRACTION_RE.match(extract_block_text_value(body, f"RAW_{kind}_SCORE"))
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def _parse_adjusted_score(
    body: str, kind: str
) -> tuple[float, float | None, float | None] | None:
    """Return (percent, parenthetical earned, parenthetical available) or None.

    The ``p% (a/b available)`` form is ambiguous in the wild: the prompt example
    reads it as earned/available ("70% (7/10 available)") while real reports
    also use it as an available-points statement ("79% (12/12 available)" with
    RAW 9.5/12). Trust ``a`` as earned only when that reading is arithmetically
    self-consistent (a != b and a/b matches the stated percent); otherwise only
    ``b`` (available) is taken.
    """
    match = _ADJUSTED_RE.match(extract_block_text_value(body, f"ADJUSTED_{kind}_SCORE"))
    if match is None:
        return None
    pct = float(match.group(1))
    paren_earned = float(match.group(2)) if match.group(2) else None
    paren_available = float(match.group(3)) if match.group(3) else None
    if paren_earned is not None and (
        paren_earned == paren_available
        or not paren_available
        or abs(paren_earned / paren_available * 100.0 - pct) > SCORE_PCT_TOLERANCE
    ):
        paren_earned = None
    return pct, paren_earned, paren_available


_SCORE_CRITERIA: dict[str, dict[str, float]] = {
    "HEALTH": HEALTH_SCORE_CRITERIA,
    "GROWTH": GROWTH_SCORE_CRITERIA,
}
# Each item must fullmatch: a lax scan would mis-read malformed awards
# ("ROE=1.5" as 1, "ROA=0.75" as 0) instead of flagging them.
_BREAKDOWN_ITEM_RE = re.compile(
    r"([A-Z][A-Z0-9_]*)\s*=\s*(0\.5|0|1|N/?A|REMOVED)\.?", re.IGNORECASE
)
# Criteria no sector adjustment ever removes: a REMOVED token on one of these
# claims a sector mandate that does not exist (data gaps must use N/A). The
# sector-removable remainder (D/E, EV/EBITDA, FCF-class, margins, expansion)
# is deliberately permissive — vendors' sector prose varies.
_NEVER_REMOVED_CRITERIA: dict[str, frozenset[str]] = {
    "HEALTH": frozenset(
        {"ROE", "ROA", "CURRENT_RATIO", "OCF_POSITIVE", "PE_OR_PEG", "PB_OR_PS"}
    ),
    "GROWTH": frozenset({"REVENUE_GROWTH", "EPS_GROWTH"}),
}
# Tolerance for objective threshold cross-checks: only flag awards that are
# wrong by a clear margin, never near-threshold judgment calls.
_OBJECTIVE_CHECK_MARGIN = 1.10


def parse_score_breakdown(text: str, kind: str = "HEALTH") -> dict[str, str] | None:
    """Parse a ``*_SCORE_BREAKDOWN`` line value into {criterion: award-token}.

    Returns None when the text is empty, any semicolon-delimited item fails to
    parse cleanly, or keys are duplicated (ambiguous — the caller flags it).
    Tokens are normalized to ``0``/``0.5``/``1``/``N/A``/``REMOVED``. Unknown
    keys are kept so the caller can name them.
    """
    if not text or not text.strip():
        return None
    # Accept a full labeled line ("HEALTH_SCORE_BREAKDOWN: ROE=1; ...") as well
    # as the bare value — the L1 prompt contract feeds whole lines.
    text = re.sub(r"^\s*[A-Z_]*_SCORE_BREAKDOWN\s*:\s*", "", text.strip())
    awards: dict[str, str] = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        match = _BREAKDOWN_ITEM_RE.fullmatch(item)
        if match is None:
            return None  # malformed item (e.g. ROE=1.5, ROA=strong)
        key, token = match.group(1).upper(), match.group(2).upper()
        if token == "NA":
            token = "N/A"
        if key in awards:
            return None  # duplicate key
        awards[key] = token
    return awards or None


def withhold_eps_growth_for_unusable_baseline(body: str) -> tuple[str, bool]:
    """Remove sustained EPS-growth credit when the earnings baseline is unsafe."""
    baseline = extract_block_text_value(body, "EARNINGS_BASELINE_STATUS").upper()
    bridge_status = extract_block_text_value(body, "GUIDANCE_BRIDGE_STATUS").upper()
    if not requires_eps_growth_withholding(baseline, bridge_status):
        return body, False

    breakdown_text = extract_block_text_value(body, "GROWTH_SCORE_BREAKDOWN")
    awards = parse_score_breakdown(breakdown_text, "GROWTH")
    if awards is None or set(awards) != set(GROWTH_SCORE_CRITERIA):
        return body, False
    if awards.get("EPS_GROWTH") == "0":
        return body, False

    awards["EPS_GROWTH"] = "0"
    numeric = {
        key: float(token)
        for key, token in awards.items()
        if token not in {"N/A", "REMOVED"}
    }
    earned = sum(numeric.values())
    available = sum(GROWTH_SCORE_CRITERIA[key] for key in numeric)
    if available <= 0:
        return body, False

    serialized = "; ".join(f"{key}={awards[key]}" for key in GROWTH_SCORE_CRITERIA)
    updated = replace_or_append_block_line(body, "GROWTH_SCORE_BREAKDOWN", serialized)
    updated = replace_or_append_block_line(
        updated,
        "RAW_GROWTH_SCORE",
        f"{earned:g}/{GROWTH_RUBRIC_POINTS:g}",
    )
    updated = replace_or_append_block_line(
        updated,
        "ADJUSTED_GROWTH_SCORE",
        f"{earned / available * 100.0:.1f}% (based on {available:g} available points)",
    )
    updated = replace_or_append_block_line(
        updated,
        "EPS_GROWTH_BASELINE_ADJUSTMENT",
        "WITHHELD — trailing earnings baseline is not durable and no code-reconciled normalized growth rate is available",
    )
    return updated, True


def _is_negative_amount(text: str) -> bool:
    """True when a DATA_BLOCK money value reads as negative (e.g. '-¥1.2B')."""
    return bool(re.search(r"[-−(]\s*[$¥€£]?\s*\d", text))


def _breakdown_objective_reasons(body: str, awards: dict[str, str]) -> list[str]:
    """Flag awards that contradict the (already reconciled) DATA_BLOCK values.

    Deliberately conservative: sign checks (sector-invariant) plus the PE_OR_PEG
    gate for non-IT sectors only (Information Technology has a documented
    P/S-based alternative), with a clear margin — never near-threshold calls.
    The one exception to the margin is the forward/TTM swap signature (145020.KQ
    quick-mode, July 2026): trailing P/E and PEG both fail at the plain
    thresholds while PE_RATIO_FORWARD passes — the passing forward value, not
    generosity on a borderline number, is what earned the point, so the plain
    thresholds apply. Threshold criteria with sector-adjusted bars (D/E, P/B,
    ROE...) are out of scope by design.
    """
    reasons: list[str] = []
    for criterion, fields in (
        ("OCF_POSITIVE", ("OPERATING_CASH_FLOW",)),
        # Canonical field first (v9.31 template); bare FCF tolerated for
        # reports that emit the shorthand.
        ("FCF_POSITIVE", ("FREE_CASH_FLOW", "FCF")),
    ):
        token = awards.get(criterion, "")
        if token not in {"0.5", "1"}:
            continue
        for field in fields:
            value = extract_block_text_value(body, field)
            if value and _is_negative_amount(value):
                reasons.append(
                    f"breakdown awards {criterion}={token} but {field} is negative"
                )
                break

    sector = normalize_sector_label(extract_block_text_value(body, "SECTOR"))
    pe = extract_block_number_from_text(body, "PE_RATIO_TTM")
    peg = extract_block_number_from_text(body, "PEG_RATIO")
    if (
        awards.get("PE_OR_PEG") in {"0.5", "1"}
        and sector != "Information Technology"
        and pe is not None
        and peg is not None
    ):
        forward = extract_block_number_from_text(body, "PE_RATIO_FORWARD")
        if (
            pe > PE_MAX * _OBJECTIVE_CHECK_MARGIN
            and peg > PEG_MAX * _OBJECTIVE_CHECK_MARGIN
        ):
            reasons.append(
                f"breakdown awards PE_OR_PEG but P/E {pe:g} and PEG {peg:g} both "
                f"clearly fail the {PE_MAX:g}/{PEG_MAX:g} thresholds"
            )
        elif (
            pe > PE_MAX and peg > PEG_MAX and forward is not None and forward <= PE_MAX
        ):
            reasons.append(
                f"breakdown awards PE_OR_PEG on the forward P/E basis "
                f"({forward:g} passes) but the criterion is trailing-only — "
                f"P/E (TTM) {pe:g} and PEG {peg:g} fail the "
                f"{PE_MAX:g}/{PEG_MAX:g} thresholds"
            )
    return reasons


def _breakdown_reasons(
    body: str, kind: str, earned: float, available: float
) -> list[str]:
    """Validate the per-criterion breakdown against RAW/available totals.

    Absent line -> no reasons (pre-v9.31 reports degrade to totals-only checks).
    """
    text = extract_block_text_value(body, f"{kind}_SCORE_BREAKDOWN")
    if not text.strip():
        return []
    criteria = _SCORE_CRITERIA[kind]
    awards = parse_score_breakdown(text, kind)
    if awards is None:
        return [f"{kind} breakdown line is unparseable or has duplicate criteria"]
    if set(awards) != set(criteria):
        missing = sorted(set(criteria) - set(awards))
        unknown = sorted(set(awards) - set(criteria))
        detail = "; ".join(
            part
            for part in (
                f"missing {', '.join(missing)}" if missing else "",
                f"unknown {', '.join(unknown)}" if unknown else "",
            )
            if part
        )
        return [f"breakdown keys do not match rubric ({detail})"]

    reasons: list[str] = []
    bogus_removed = sorted(
        key
        for key, token in awards.items()
        if token == "REMOVED" and key in _NEVER_REMOVED_CRITERIA[kind]
    )
    if bogus_removed:
        reasons.append(
            "REMOVED claimed for non-sector-removable criteria: "
            f"{', '.join(bogus_removed)} (data gaps must use N/A)"
        )
    numeric = {
        key: float(token)
        for key, token in awards.items()
        if token not in {"N/A", "REMOVED"}
    }
    for key, points in numeric.items():
        if points > criteria[key]:
            reasons.append(
                f"breakdown award {key}={points:g} exceeds max {criteria[key]:g}"
            )
    awards_sum = sum(numeric.values())
    if abs(awards_sum - earned) > 0.01:
        reasons.append(f"breakdown awards sum {awards_sum:g} != RAW earned {earned:g}")
    expected_available = sum(criteria[key] for key in numeric)
    if abs(expected_available - available) > 0.01:
        reasons.append(
            f"breakdown available points {expected_available:g} != stated "
            f"available {available:g}"
        )
    if not reasons:
        reasons.extend(_breakdown_objective_reasons(body, awards))
    return reasons


def reconcile_score_consistency(body: str) -> tuple[str, bool, bool]:
    """Validate RAW vs ADJUSTED health/growth score lines for internal consistency.

    The scores are LLM arithmetic that feeds the hard quality gates (Adjusted
    Health/Growth < 50% -> SELL), so an inconsistent line is catastrophic. Hybrid
    policy: rewrite ADJUSTED only when the correction is *provable* arithmetic
    (denominators coherent, only the percent diverges); anything template-violating
    or implausible gets a ``*_SCORE_CONSISTENCY: SUSPECT`` line — never fixed by
    inference. N/A criteria legitimately shrink the available denominator below the
    rubric total, so totals are ceilings, not exact requirements. When the prompt's
    (v9.31+) per-criterion ``*_SCORE_BREAKDOWN`` line is present, the numerator is
    audited too: awards must reconcile to RAW/available and pass conservative
    objective cross-checks; pre-v9.31 reports without the line degrade to the
    totals-only checks.

    Returns ``(body, corrected, suspect)``.
    """
    updated, corrected, suspect = body, False, False
    sector = normalize_sector_label(extract_block_text_value(body, "SECTOR"))
    adjustments = extract_block_text_value(body, "SECTOR_ADJUSTMENTS")

    for kind, total in _SCORE_RUBRIC_TOTALS:
        raw = _parse_raw_score(updated, kind)
        if raw is None:
            continue
        adjusted = _parse_adjusted_score(updated, kind)
        adjusted_field = f"ADJUSTED_{kind}_SCORE"
        if adjusted is None:
            # A genuinely absent ADJUSTED line is left alone (nothing to
            # correct against). But a *present* line that's unparseable —
            # most commonly a literal "N/A" — while RAW is fully computable
            # is exactly the provable-arithmetic case this function already
            # handles below for a different trigger; fill it in the same way
            # rather than silently skipping reconciliation entirely.
            if not has_block_field_value(updated, adjusted_field):
                continue
            earned, raw_den = raw
            if raw_den <= 0:
                continue
            corrected = True
            updated = replace_or_append_block_line(
                updated,
                adjusted_field,
                f"{earned / raw_den * 100.0:.1f}% (based on {raw_den:g} available points)",
            )
            updated = replace_or_append_block_line(
                updated,
                f"{kind}_SCORE_DATA_QUALITY_NOTE",
                (
                    f"Adjusted score computed from RAW {earned:g}/{raw_den:g}; "
                    "reported value was unparseable ('N/A' or malformed)."
                ),
            )
            continue
        (earned, raw_den), (pct, paren_earned, paren_available) = raw, adjusted

        available = paren_available if paren_available is not None else raw_den
        denominator_coherent = raw_den in (available, total)

        reasons: list[str] = []
        if not denominator_coherent:
            reasons.append(
                f"raw denominator {raw_den:g} matches neither rubric total "
                f"{total:g} nor available points {available:g}"
            )
        if available > total or available <= 0:
            reasons.append(
                f"available points {available:g} implausible vs rubric total {total:g}"
            )
        if earned > available and denominator_coherent:
            reasons.append(f"earned points {earned:g} exceed available {available:g}")
        if paren_earned is not None and abs(paren_earned - earned) > 0.01:
            reasons.append(
                f"earned points differ between RAW ({earned:g}) and "
                f"ADJUSTED ({paren_earned:g}) lines"
            )
        if (
            kind == "HEALTH"
            and sector == "Financials"
            and _DE_REMOVED_RE.search(adjustments)
            and available > total - FINANCIALS_HEALTH_REMOVED_POINTS
        ):
            reasons.append(
                "SECTOR_ADJUSTMENTS says D/E removed but available points not reduced"
            )

        reasons.extend(_breakdown_reasons(updated, kind, earned, available))

        if reasons:
            suspect = True
            updated = replace_or_append_block_line(
                updated,
                f"{kind}_SCORE_CONSISTENCY",
                "SUSPECT — " + "; ".join(reasons),
            )
            continue

        expected_pct = earned / available * 100.0
        if abs(expected_pct - pct) > SCORE_PCT_TOLERANCE:
            corrected = True
            updated = replace_or_append_block_line(
                updated,
                f"ADJUSTED_{kind}_SCORE",
                f"{expected_pct:.1f}% (based on {available:g} available points)",
            )
            updated = replace_or_append_block_line(
                updated,
                f"{kind}_SCORE_DATA_QUALITY_NOTE",
                (
                    f"Adjusted score recomputed from RAW {earned:g}/{available:g}; "
                    f"reported {pct:.1f}% was arithmetically inconsistent."
                ),
            )

    return updated, corrected, suspect


def append_analyst_coverage_data_quality_note(
    body: str,
    foreign_data: str,
) -> str:
    if not foreign_data or "ANALYST_COVERAGE_DATA_QUALITY_NOTE:" in body:
        return body

    english_coverage = extract_block_number_from_text(body, "ANALYST_COVERAGE_ENGLISH")
    if english_coverage is None or english_coverage >= 5:
        return body

    total_est = extract_block_text_value(body, "ANALYST_COVERAGE_TOTAL_EST").upper()
    has_high_total_est = total_est in {"HIGH", "MODERATE"} or (
        total_est.isdigit() and int(total_est) > english_coverage
    )
    if has_high_total_est or _HIGH_LOCAL_COVERAGE_PATTERN.search(foreign_data):
        return replace_or_append_block_line(
            body,
            "ANALYST_COVERAGE_DATA_QUALITY_NOTE",
            (
                "English aggregator count appears low versus local/total coverage "
                "signals; avoid unqualified hidden/undiscovered framing."
            ),
        )
    return body
