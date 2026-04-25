"""Merge-policy helpers for multi-source financial data."""

from __future__ import annotations

import math
from typing import Any

import structlog

from src.fx_normalization import is_near_minor_unit_ratio

logger = structlog.get_logger(__name__)

PERCENT_LIKE_FIELDS = frozenset(
    {
        "dividendYield",
        "trailingAnnualDividendYield",
        "fiveYearAvgDividendYield",
        "regularMarketChangePercent",
    }
)
NON_FINANCIAL_METADATA_FIELDS = frozenset({"maxAge"})
NON_ACTIONABLE_CONFLICT_FIELDS = frozenset(
    {
        "bidSize",
        "askSize",
        "bid",
        "ask",
        "regularMarketBidSize",
        "regularMarketAskSize",
    }
)
CRITICAL_ANALYSIS_FIELDS = (
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "pegRatio",
    "returnOnEquity",
    "returnOnAssets",
    "debtToEquity",
    "currentRatio",
    "operatingMargins",
    "grossMargins",
    "profitMargins",
    "revenueGrowth",
    "earningsGrowth",
    "operatingCashflow",
    "freeCashflow",
    "numberOfAnalystOpinions",
)
ANALYSIS_CRITICAL_CONFLICT_FIELDS = frozenset(CRITICAL_ANALYSIS_FIELDS)
QUOTE_PRICE_FIELDS = (
    "currentPrice",
    "regularMarketPrice",
    "previousClose",
    "regularMarketPreviousClose",
    "open",
    "regularMarketOpen",
    "dayLow",
    "dayHigh",
    "regularMarketDayLow",
    "regularMarketDayHigh",
    "bid",
    "ask",
    "fiftyDayAverage",
    "twoHundredDayAverage",
    "fiftyTwoWeekLow",
    "fiftyTwoWeekHigh",
)
SOURCE_QUALITY = {
    "yfinance_statements": 10,
    "calculated_from_statements": 10,
    "eodhd": 9.5,
    "yfinance": 9,
    "yfinance_info": 9,
    "alpha_vantage": 9,
    "calculated": 8,
    "fmp": 7,
    "fmp_info": 7,
    "yahooquery": 6,
    "yahooquery_info": 6,
    "tavily_extraction": 4,
    "proxy": 2,
}
FORWARD_PE_OUTLIER_THRESHOLD = 200.0
FORWARD_PE_REFERENCE_MAX = 100.0
FORWARD_PE_OUTLIER_RATIO = 5.0


def normalize_percent_pair(old_val: float, new_val: float) -> tuple[float, float]:
    """Normalize decimal-vs-percent representations before comparison."""
    candidates = [
        (old_val, new_val),
        (old_val * 100, new_val),
        (old_val, new_val * 100),
    ]

    def relative_gap(pair: tuple[float, float]) -> float:
        left, right = pair
        baseline = max(abs(left), abs(right), 1e-9)
        return abs(left - right) / baseline

    return min(candidates, key=relative_gap)


def coerce_positive_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def identity_match_from_price(
    price: float | None,
    denominator: Any,
    reported_ratio: Any,
    tolerance: float = 0.15,
) -> bool:
    price_f = safe_float(price)
    denom_f = safe_float(denominator)
    ratio_f = safe_float(reported_ratio)
    if price_f is None or denom_f is None or ratio_f is None:
        return False
    if price_f <= 0 or denom_f <= 0 or ratio_f <= 0:
        return False
    expected = price_f / denom_f
    return abs(expected - ratio_f) / ratio_f <= tolerance


def conflict_field_class(field: str) -> str:
    if field in NON_ACTIONABLE_CONFLICT_FIELDS:
        return "microstructure"
    if field in ANALYSIS_CRITICAL_CONFLICT_FIELDS:
        return "valuation"
    return "other"


def is_actionable_conflict(
    field: str, left_quality: float, right_quality: float
) -> bool:
    if field not in ANALYSIS_CRITICAL_CONFLICT_FIELDS:
        return False
    quality_gap = abs(left_quality - right_quality)
    return quality_gap <= 1.0 or (left_quality >= 9 and right_quality >= 9)


def normalize_scaling_errors(val_a: float, val_b: float) -> float:
    """Detect and correct 100x scaling mismatches like cents vs major units."""
    if not val_a or not val_b:
        return val_a or val_b
    try:
        ratio = val_a / val_b
        if is_near_minor_unit_ratio(ratio):
            return val_b
        if is_near_minor_unit_ratio(1 / ratio):
            return val_a
        return val_b
    except ZeroDivisionError:
        return val_b


def smart_merge_with_quality(
    source_results: dict[str, dict | None],
    symbol: str,
    quarantine_forward_pe_outlier,
    *,
    logger_obj: Any = logger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge source results with quality-aware precedence and conflict tracking."""
    source_results = quarantine_forward_pe_outlier(source_results, symbol)
    merged: dict[str, Any] = {}
    field_sources: dict[str, str] = {}
    field_quality: dict[str, float] = {}
    source_conflicts: dict[str, dict[str, Any]] = {}
    sources_used: set[str] = set()
    gaps_filled = 0

    source_order = ["yahooquery", "fmp", "alpha_vantage", "eodhd", "yfinance"]
    for source_name in source_order:
        source_data = source_results.get(source_name)
        if not source_data:
            continue

        sources_used.add(source_name)
        for key, value in source_data.items():
            if value is None or (key.startswith("_") and key.endswith("_source")):
                continue

            quality = SOURCE_QUALITY.get(
                source_name, SOURCE_QUALITY.get(f"{source_name}_info", 5)
            )
            source_tag_key = f"_{key}_source"
            if (
                source_tag_key in source_data
                and source_data[source_tag_key] in SOURCE_QUALITY
            ):
                quality = SOURCE_QUALITY[source_data[source_tag_key]]

            should_use = False
            if key not in merged:
                should_use = True
            elif merged[key] is None and value is not None:
                should_use = True
                gaps_filled += 1
            elif key in field_quality and quality > field_quality[key]:
                should_use = True
                logger_obj.debug(
                    "replacing_with_higher_quality",
                    symbol=symbol,
                    field=key,
                    old_source=field_sources.get(key),
                    new_source=source_name,
                )

            if should_use:
                if (
                    key in merged
                    and merged[key] is not None
                    and isinstance(value, int | float)
                ):
                    try:
                        if key in [
                            "currentPrice",
                            "regularMarketPrice",
                            "previousClose",
                            "marketCap",
                        ]:
                            corrected_val = normalize_scaling_errors(
                                float(merged[key]), float(value)
                            )
                            if corrected_val != float(value):
                                logger_obj.info(
                                    "scaling_error_corrected",
                                    field=key,
                                    original=merged[key],
                                    candidate=value,
                                    corrected=corrected_val,
                                )
                                value = corrected_val
                    except (ValueError, TypeError):
                        pass

                if key in NON_FINANCIAL_METADATA_FIELDS:
                    merged[key] = value
                    field_sources[key] = source_name
                    field_quality[key] = quality
                    continue

            if (
                key in merged
                and merged[key] is not None
                and value is not None
                and key not in NON_FINANCIAL_METADATA_FIELDS
                and key not in NON_ACTIONABLE_CONFLICT_FIELDS
            ):
                try:
                    old_val = float(merged[key])
                    new_val = float(value)
                    if key in PERCENT_LIKE_FIELDS:
                        old_val, new_val = normalize_percent_pair(old_val, new_val)
                    if old_val != 0 and abs(new_val - old_val) / abs(old_val) > 0.20:
                        old_source = field_sources.get(key, "unknown")
                        old_quality = field_quality.get(key, quality)
                        winner_quality = quality if should_use else old_quality
                        loser_quality = old_quality if should_use else quality
                        source_conflicts[key] = {
                            "old": round(old_val, 4),
                            "old_source": old_source,
                            "new": round(new_val, 4),
                            "new_source": source_name,
                            "variance_pct": round(
                                abs(new_val - old_val) / abs(old_val) * 100, 1
                            ),
                            "field_class": conflict_field_class(key),
                            "resolved_by_quality": not is_actionable_conflict(
                                key, old_quality, quality
                            ),
                            "winner_quality": winner_quality,
                            "loser_quality": loser_quality,
                        }
                except (ValueError, TypeError):
                    pass

            if should_use:
                merged[key] = value
                field_sources[key] = source_name
                field_quality[key] = quality

    metadata = {
        "sources_used": list(sources_used),
        "composite_source": f"composite_{'+'.join(sorted(sources_used))}",
        "gaps_filled": gaps_filled,
        "field_sources": field_sources,
        "field_quality": field_quality,
        "source_conflicts": source_conflicts,
    }

    if source_conflicts:
        formatted_conflicts = {
            key: f"{value['old_source']}={value['old']} vs {value['new_source']}={value['new']} (Δ{value['variance_pct']}%)"
            for key, value in source_conflicts.items()
        }
        actionable_conflicts = {
            key: value
            for key, value in source_conflicts.items()
            if not value["resolved_by_quality"]
        }
        resolved_conflicts = {
            key: value
            for key, value in source_conflicts.items()
            if value["resolved_by_quality"]
        }
        if actionable_conflicts:
            logger_obj.warning(
                "source_data_conflicts",
                symbol=symbol,
                conflicts={
                    key: formatted_conflicts[key] for key in actionable_conflicts
                },
            )
        if resolved_conflicts:
            logger_obj.debug(
                "source_data_conflicts_resolved",
                symbol=symbol,
                conflicts={key: formatted_conflicts[key] for key in resolved_conflicts},
            )

    logger_obj.info(
        "smart_merge_complete",
        symbol=symbol,
        total_fields=len(merged),
        sources=list(sources_used),
        gaps_filled=gaps_filled,
    )
    return merged, metadata


def quarantine_forward_pe_outlier(
    source_results: dict[str, dict | None],
    symbol: str,
    *,
    logger_obj: Any = logger,
) -> dict[str, dict | None]:
    """Quarantine a lone implausible forward P/E outlier against plausible peers."""
    candidates: list[tuple[str, float]] = []
    for source_name, source_data in source_results.items():
        if not source_data:
            continue
        forward_pe = coerce_positive_float(source_data.get("forwardPE"))
        if forward_pe is not None:
            candidates.append((source_name, forward_pe))

    if len(candidates) < 2:
        return source_results

    reference_candidates = [
        (source_name, value)
        for source_name, value in candidates
        if value <= FORWARD_PE_REFERENCE_MAX
    ]
    if not reference_candidates:
        return source_results

    outliers = [
        (source_name, value)
        for source_name, value in candidates
        if value > FORWARD_PE_OUTLIER_THRESHOLD
        and any(
            value / reference_value >= FORWARD_PE_OUTLIER_RATIO
            for _, reference_value in reference_candidates
        )
    ]
    if len(outliers) != 1:
        return source_results

    outlier_source, outlier_value = outliers[0]
    sanitized_results: dict[str, dict | None] = {}
    for source_name, source_data in source_results.items():
        if not source_data:
            sanitized_results[source_name] = source_data
            continue
        updated = source_data.copy()
        if source_name == outlier_source:
            updated["forwardPE"] = None
            updated["_forwardPE_quarantine_reason"] = (
                "single_source_outlier_vs_plausible_peer"
            )
        sanitized_results[source_name] = updated

    logger_obj.info(
        "forward_pe_outlier_quarantined",
        symbol=symbol,
        source=outlier_source,
        forward_pe=round(outlier_value, 4),
        reference_values={
            source_name: round(value, 4) for source_name, value in reference_candidates
        },
    )
    return sanitized_results
