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
from src.validators.pfic_constants import PFIC_ASSET_TEST_THRESHOLD

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


def material_diff(
    body: str,
    key: str,
    expected: float,
    *,
    threshold: float,
) -> bool:
    current = extract_block_number_from_text(body, key)
    return current is None or abs(current - expected) > threshold


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


def reconcile_high_risk_fields(
    body: str,
    payload: dict[str, Any],
) -> str:
    updated = body
    changed_growth = False
    changed_balance_sheet = False

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
    if changed_balance_sheet:
        updated = replace_or_append_block_line(
            updated,
            "BALANCE_SHEET_DATA_QUALITY_NOTE",
            "High-risk balance-sheet fields reconciled to raw get_financial_metrics basis.",
        )

    return updated


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
