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
from src.sector_normalization import normalize_sector_label
from src.thesis_constants import (
    FINANCIALS_HEALTH_REMOVED_POINTS,
    GROWTH_RUBRIC_POINTS,
    HEALTH_RUBRIC_POINTS,
    SCORE_PCT_TOLERANCE,
)
from src.validators.pfic_constants import (
    PFIC_ASSET_PROXIMITY_THRESHOLD,
    PFIC_ASSET_TEST_THRESHOLD,
)

HORIZON_FIELD_RAW_KEYS = (
    ("REVENUE_GROWTH_TTM", "revenueGrowth_TTM"),
    ("REVENUE_GROWTH_MRQ", "revenueGrowth_MRQ"),
    ("EARNINGS_GROWTH_TTM", "earningsGrowth_TTM"),
    ("EARNINGS_GROWTH_MRQ", "earningsGrowth_MRQ"),
    ("GROWTH_TRAJECTORY", "growth_trajectory"),
)

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

    cycle_position = str(payload.get("cycle_position") or "").upper()
    if cycle_position in {"PEAK", "MID", "TROUGH"} and (
        extract_block_text_value(updated, "CYCLE_POSITION").upper() != cycle_position
    ):
        updated = replace_or_append_block_line(
            updated, "CYCLE_POSITION", cycle_position
        )
        changed_growth = True

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
            "TTM growth unavailable in raw payload; FY/MRQ values were not reused.",
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


def reconcile_score_consistency(body: str) -> tuple[str, bool, bool]:
    """Validate RAW vs ADJUSTED health/growth score lines for internal consistency.

    The scores are LLM arithmetic that feeds the hard quality gates (Adjusted
    Health/Growth < 50% -> SELL), so an inconsistent line is catastrophic. Hybrid
    policy: rewrite ADJUSTED only when the correction is *provable* arithmetic
    (denominators coherent, only the percent diverges); anything template-violating
    or implausible gets a ``*_SCORE_CONSISTENCY: SUSPECT`` line — never fixed by
    inference. N/A criteria legitimately shrink the available denominator below the
    rubric total, so totals are ceilings, not exact requirements; a numerator that
    is merely *wrong* (same data scored differently across runs) is not detectable
    here without a per-criterion breakdown.

    Returns ``(body, corrected, suspect)``.
    """
    updated, corrected, suspect = body, False, False
    sector = normalize_sector_label(extract_block_text_value(body, "SECTOR"))
    adjustments = extract_block_text_value(body, "SECTOR_ADJUSTMENTS")

    for kind, total in _SCORE_RUBRIC_TOTALS:
        raw = _parse_raw_score(updated, kind)
        adjusted = _parse_adjusted_score(updated, kind)
        if raw is None or adjusted is None:
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
