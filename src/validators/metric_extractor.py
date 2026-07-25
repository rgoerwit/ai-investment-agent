"""Metric extraction helpers for red-flag detection."""

from __future__ import annotations

import re
from typing import Any

import structlog

from src.data_block_utils import extract_last_data_block

logger = structlog.get_logger(__name__)


def parse_currency_value(sign: str, value_str: str, multiplier: str | None) -> float:
    """Parse a sign + numeric string + B/M/K multiplier into a float."""
    value = float(value_str.replace(",", ""))
    if sign == "-":
        value = -value
    if multiplier:
        normalized = multiplier.upper()
        if normalized == "B":
            value *= 1_000_000_000
        elif normalized == "M":
            value *= 1_000_000
        elif normalized == "K":
            value *= 1_000
    return value


def parse_ratio_or_percent(raw_value: str) -> float | None:
    """Parse a ratio that may be expressed as either a decimal or percentage."""
    if raw_value is None:
        return None

    text = raw_value.strip()
    if not text or text.upper() == "N/A":
        return None

    has_percent = text.endswith("%")
    if has_percent:
        text = text[:-1].strip()

    try:
        value = float(text)
    except ValueError:
        return None

    if has_percent:
        return value / 100
    if abs(value) >= 2.0:
        return value / 100
    return value


def extract_metrics(
    fundamentals_report: str,
    *,
    ticker: str | None = None,
    source_file: str | None = None,
) -> dict[str, Any]:
    """Extract financial metrics from the Senior Fundamentals DATA_BLOCK and body."""
    metrics: dict[str, Any] = {
        "debt_to_equity": None,
        "net_income": None,
        "fcf": None,
        "interest_coverage": None,
        "pe_ratio": None,
        "sector_median_pe": None,
        "pe_vs_sector": None,
        "pb_ratio": None,
        "adjusted_health_score": None,
        "health_score_consistency": None,
        "growth_score_consistency": None,
        "payout_ratio": None,
        "dividend_coverage": None,
        "net_margin": None,
        "roic_quality": None,
        "profitability_trend": None,
        "roa_current": None,
        "roa_5y_avg": None,
        "roe_5y_avg": None,
        "peg_ratio": None,
        "ocf": None,
        "ocf_source": None,
        "ocf_filing_reason": None,
        "ocf_period": None,
        "segment_flag": None,
        "parent_company": None,
        "listing_role": None,
        "related_listed_tickers": None,
        "metric_scope_payout": None,
        "metric_scope_ocf": None,
        "analyst_coverage_total_est": None,
        "growth_trajectory": None,
        "revenue_cagr_3y": None,
        "fcf_cagr_3y": None,
        "cycle_position": None,
        "revenue_growth_ttm": None,
        "sector_pe_reference_type": None,
        "sector_pe_reference_as_of": None,
        "revenue_backlog_coverage": None,
        "latest_quarter_date": None,
        "net_cash_to_market_cap": None,
        "cash_to_assets": None,
        "net_debt_ebitda": None,
        "capex_to_da": None,
        "capex_to_da_status": None,
        "asset_turnover": None,
        "inventory_turnover_trend": None,
        "capacity_utilization": None,
        "facility_buildout_status": None,
        "capital_plan_status": None,
        "guidance_coverage_status": None,
        "guidance_source_date": None,
        "guidance_source_url": None,
        "guidance_period": None,
        "guidance_revenue": None,
        "guidance_operating_profit": None,
        "guidance_ordinary_or_pretax_profit": None,
        "guidance_net_income": None,
        "guidance_net_income_yoy": None,
        "operating_vs_net_direction": None,
        "material_nonoperating_driver": None,
        "driver_type": None,
        "driver_persistence": None,
        "driver_materiality": None,
        "driver_affected_period": None,
        "earnings_baseline_status": None,
        "normalized_earnings_available": None,
        "guidance_bridge_status": None,
        "sector": None,
        "industry": None,
        "_raw_report": fundamentals_report,
    }

    if not fundamentals_report:
        return metrics

    data_block = extract_last_data_block(fundamentals_report)
    if not data_block:
        log_fields: dict[str, Any] = {}
        if ticker:
            log_fields["ticker"] = ticker
        if source_file:
            log_fields["file"] = source_file
        logger.warning("no_data_block_found_in_fundamentals_report", **log_fields)
        return metrics

    health_match = re.search(r"ADJUSTED_HEALTH_SCORE:\s*(\d+(?:\.\d+)?)%", data_block)
    if health_match:
        metrics["adjusted_health_score"] = float(health_match.group(1))

    for kind in ("HEALTH", "GROWTH"):
        consistency_match = re.search(
            rf"{kind}_SCORE_CONSISTENCY:\s*SUSPECT\b", data_block
        )
        if consistency_match:
            metrics[f"{kind.lower()}_score_consistency"] = "SUSPECT"

    pe_match = re.search(r"PE_RATIO_TTM:\s*(\d+(?:\.\d+)?)", data_block)
    if pe_match:
        metrics["pe_ratio"] = float(pe_match.group(1))

    sector_median_pe_match = re.search(
        r"SECTOR_MEDIAN_PE:\s*(\d+(?:\.\d+)?)", data_block
    )
    if sector_median_pe_match:
        metrics["sector_median_pe"] = float(sector_median_pe_match.group(1))

    pe_vs_sector_match = re.search(r"PE_VS_SECTOR:\s*(\d+(?:\.\d+)?)", data_block)
    if pe_vs_sector_match:
        metrics["pe_vs_sector"] = float(pe_vs_sector_match.group(1))

    sector_pe_reference_type_match = re.search(
        r"SECTOR_PE_REFERENCE_TYPE:\s*"
        r"(STATIC_POLICY_REFERENCE|LIVE_MARKET_REFERENCE|UNKNOWN)",
        data_block,
        re.IGNORECASE,
    )
    if sector_pe_reference_type_match:
        metrics["sector_pe_reference_type"] = sector_pe_reference_type_match.group(
            1
        ).upper()
    sector_pe_reference_as_of_match = re.search(
        r"SECTOR_PE_REFERENCE_AS_OF:\s*(.+?)(?:\n|$)",
        data_block,
        re.IGNORECASE,
    )
    if sector_pe_reference_as_of_match:
        reference_as_of = sector_pe_reference_as_of_match.group(1).strip()
        if reference_as_of.upper() not in {"N/A", "NA", "NONE", "UNKNOWN"}:
            metrics["sector_pe_reference_as_of"] = reference_as_of

    pb_match = re.search(r"PB_RATIO:\s*(\d+(?:\.\d+)?)", data_block)
    if pb_match:
        metrics["pb_ratio"] = float(pb_match.group(1))

    payout_match = re.search(
        r"PAYOUT_RATIO:\s*(\d+(?:\.\d+)?)%", data_block, re.IGNORECASE
    )
    if payout_match:
        metrics["payout_ratio"] = float(payout_match.group(1))

    coverage_match = re.search(
        r"DIVIDEND_COVERAGE:\s*(COVERED|PARTIAL|UNCOVERED|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if coverage_match:
        value = coverage_match.group(1).upper()
        if value != "N/A":
            metrics["dividend_coverage"] = value

    margin_match = re.search(r"NET_MARGIN:\s*(\d+(?:\.\d+)?)%", data_block)
    if margin_match:
        metrics["net_margin"] = float(margin_match.group(1))

    roic_quality_match = re.search(
        r"ROIC_QUALITY:\s*(STRONG|ADEQUATE|WEAK|DESTRUCTIVE|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if roic_quality_match:
        value = roic_quality_match.group(1).upper()
        if value != "N/A":
            metrics["roic_quality"] = value

    trend_match = re.search(
        r"PROFITABILITY_TREND:\s*(IMPROVING|STABLE|DECLINING|UNSTABLE|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if trend_match:
        value = trend_match.group(1).upper()
        if value != "N/A":
            metrics["profitability_trend"] = value

    roa_match = re.search(r"ROA_PERCENT:\s*(\d+(?:\.\d+)?)%?", data_block)
    if roa_match:
        metrics["roa_current"] = float(roa_match.group(1))

    roa_avg_match = re.search(r"ROA_5Y_AVG:\s*(\d+(?:\.\d+)?)%?", data_block)
    if roa_avg_match:
        metrics["roa_5y_avg"] = float(roa_avg_match.group(1))

    roe_avg_match = re.search(r"ROE_5Y_AVG:\s*(\d+(?:\.\d+)?)%?", data_block)
    if roe_avg_match:
        metrics["roe_5y_avg"] = float(roe_avg_match.group(1))

    peg_match = re.search(r"PEG_RATIO:\s*(\d+(?:\.\d+)?)", data_block)
    if peg_match:
        metrics["peg_ratio"] = float(peg_match.group(1))

    ocf_match = re.search(
        r"OPERATING_CASH_FLOW:\s*([+-]?)[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
        data_block,
        re.IGNORECASE,
    )
    if ocf_match:
        metrics["ocf"] = parse_currency_value(
            ocf_match.group(1), ocf_match.group(2), ocf_match.group(3)
        )

    ocf_source_match = re.search(
        r"OPERATING_CASH_FLOW_SOURCE:\s*(JUNIOR|FILING|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if ocf_source_match:
        value = ocf_source_match.group(1).upper()
        if value != "N/A":
            metrics["ocf_source"] = value

    ocf_reason_match = re.search(
        r"OCF_FILING_REASON:\s*(DISCREPANCY|API_UNAVAILABLE|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if ocf_reason_match:
        value = ocf_reason_match.group(1).upper()
        if value != "N/A":
            metrics["ocf_filing_reason"] = value

    ocf_period_match = re.search(
        r"OPERATING_CASH_FLOW_PERIOD:\s*([^\n]+)",
        data_block,
        re.IGNORECASE,
    )
    if ocf_period_match:
        value = ocf_period_match.group(1).strip()
        if value.upper() not in ("N/A", ""):
            metrics["ocf_period"] = value

    segment_flag_match = re.search(
        r"SEGMENT_FLAG:\s*(DETERIORATING|STABLE|N/A)", data_block, re.IGNORECASE
    )
    if segment_flag_match:
        value = segment_flag_match.group(1).upper()
        if value != "N/A":
            metrics["segment_flag"] = value

    parent_match = re.search(r"PARENT_COMPANY:\s*(.+?)(?:\n|$)", data_block)
    if parent_match:
        value = parent_match.group(1).strip()
        if value.upper() not in ("NONE", "N/A"):
            metrics["parent_company"] = value

    listing_role_match = re.search(
        r"LISTING_ROLE:\s*(STANDALONE|PURE_HOLDCO|INTERMEDIATE_HOLDCO|LISTED_SUBSIDIARY|UNKNOWN)",
        data_block,
        re.IGNORECASE,
    )
    if listing_role_match:
        metrics["listing_role"] = listing_role_match.group(1).upper()

    related_listed_match = re.search(
        r"RELATED_LISTED_TICKERS:\s*(.+?)(?:\n|$)", data_block
    )
    if related_listed_match:
        value = related_listed_match.group(1).strip()
        if value.upper() not in ("NONE", "N/A", "UNKNOWN", ""):
            metrics["related_listed_tickers"] = value

    payout_scope_match = re.search(
        r"METRIC_SCOPE_PAYOUT:\s*(CONSOLIDATED|SEPARATE|UNKNOWN)",
        data_block,
        re.IGNORECASE,
    )
    if payout_scope_match:
        metrics["metric_scope_payout"] = payout_scope_match.group(1).upper()

    ocf_scope_match = re.search(
        r"METRIC_SCOPE_OCF:\s*(CONSOLIDATED|SEPARATE|UNKNOWN)",
        data_block,
        re.IGNORECASE,
    )
    if ocf_scope_match:
        metrics["metric_scope_ocf"] = ocf_scope_match.group(1).upper()

    total_est_match = re.search(
        r"ANALYST_COVERAGE_TOTAL_EST:\s*(.+?)(?:\n|$)",
        data_block,
        re.IGNORECASE,
    )
    if total_est_match:
        value = total_est_match.group(1).strip()
        if value.upper() not in ("N/A", "NA", "NONE", "-", "", "UNKNOWN"):
            int_match = re.match(r"^(\d+)", value)
            if int_match:
                metrics["analyst_coverage_total_est"] = int(int_match.group(1))
            else:
                tier = value.upper().split()[0]
                if tier in ("HIGH", "MODERATE", "LOW"):
                    metrics["analyst_coverage_total_est"] = tier

    trajectory_match = re.search(
        r"GROWTH_TRAJECTORY:\s*(ACCELERATING|DECELERATING|STABLE|MIXED|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if trajectory_match:
        value = trajectory_match.group(1).upper()
        if value != "N/A":
            metrics["growth_trajectory"] = value

    revenue_cagr_match = re.search(r"REVENUE_CAGR_3Y:\s*(-?\d+(?:\.\d+)?)%", data_block)
    if revenue_cagr_match:
        metrics["revenue_cagr_3y"] = float(revenue_cagr_match.group(1))

    fcf_cagr_match = re.search(r"FCF_CAGR_3Y:\s*(-?\d+(?:\.\d+)?)%", data_block)
    if fcf_cagr_match:
        metrics["fcf_cagr_3y"] = float(fcf_cagr_match.group(1))

    cycle_position_match = re.search(
        r"CYCLE_POSITION:\s*(PEAK|MID|TROUGH|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if cycle_position_match:
        value = cycle_position_match.group(1).upper()
        if value != "N/A":
            metrics["cycle_position"] = value

    rev_ttm_match = re.search(r"REVENUE_GROWTH_TTM:\s*(-?\d+(?:\.\d+)?)%", data_block)
    if rev_ttm_match:
        metrics["revenue_growth_ttm"] = float(rev_ttm_match.group(1))

    backlog_coverage_match = re.search(
        r"REVENUE_BACKLOG_COVERAGE:\s*([0-9]+(?:\.\d+)?)", data_block
    )
    if backlog_coverage_match:
        metrics["revenue_backlog_coverage"] = float(backlog_coverage_match.group(1))

    quarter_date_match = re.search(
        r"LATEST_QUARTER_DATE:\s*(\d{4}-\d{2}-\d{2})", data_block
    )
    if quarter_date_match:
        metrics["latest_quarter_date"] = quarter_date_match.group(1)

    net_cash_to_mc_match = re.search(
        r"NET_CASH_TO_MARKET_CAP:\s*([^\n]+)", data_block, re.IGNORECASE
    )
    if net_cash_to_mc_match:
        metrics["net_cash_to_market_cap"] = parse_ratio_or_percent(
            net_cash_to_mc_match.group(1)
        )

    cash_to_assets_match = re.search(
        r"CASH_TO_ASSETS:\s*([^\n]+)", data_block, re.IGNORECASE
    )
    if cash_to_assets_match:
        metrics["cash_to_assets"] = parse_ratio_or_percent(
            cash_to_assets_match.group(1)
        )

    net_debt_ebitda_match = re.search(
        r"NET_DEBT_EBITDA:\s*([^\n]+)", data_block, re.IGNORECASE
    )
    if net_debt_ebitda_match:
        raw_value = net_debt_ebitda_match.group(1).strip()
        if raw_value.upper() != "N/A":
            try:
                metrics["net_debt_ebitda"] = float(raw_value.lstrip("~≈"))
            except ValueError:
                pass

    capex_to_da_match = re.search(r"CAPEX_TO_DA:\s*([^\n]+)", data_block, re.IGNORECASE)
    if capex_to_da_match:
        raw_value = capex_to_da_match.group(1).strip()
        if raw_value.upper() != "N/A":
            try:
                metrics["capex_to_da"] = float(raw_value)
            except ValueError:
                pass

    asset_turnover_match = re.search(
        r"ASSET_TURNOVER:\s*([0-9]+(?:\.[0-9]+)?)", data_block
    )
    if asset_turnover_match:
        try:
            metrics["asset_turnover"] = float(asset_turnover_match.group(1))
        except ValueError:
            pass

    inv_trend_match = re.search(
        r"INVENTORY_TURNOVER_TREND:\s*(RISING|STABLE|FALLING|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if inv_trend_match:
        value = inv_trend_match.group(1).upper()
        if value != "N/A":
            metrics["inventory_turnover_trend"] = value

    capacity_util_match = re.search(
        r"CAPACITY_UTILIZATION:\s*([0-9]+(?:\.[0-9]+)?)\s*%", data_block
    )
    if capacity_util_match:
        try:
            metrics["capacity_utilization"] = float(capacity_util_match.group(1))
        except ValueError:
            pass

    facility_status_match = re.search(
        r"FACILITY_BUILDOUT_STATUS:\s*"
        r"(UNDER_CONSTRUCTION|RAMPING|AT_CAPACITY|NONE|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if facility_status_match:
        value = facility_status_match.group(1).upper()
        if value != "N/A":
            metrics["facility_buildout_status"] = value

    capex_status_match = re.search(
        r"CAPEX_TO_DA_STATUS:\s*(UNDERINVESTING|MAINTENANCE|GROWTH_INVESTING|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if capex_status_match:
        value = capex_status_match.group(1).upper()
        if value != "N/A":
            metrics["capex_to_da_status"] = value

    plan_status_match = re.search(
        r"CAPITAL_PLAN_STATUS:\s*(EXPLICIT|NONE|UNKNOWN|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if plan_status_match:
        value = plan_status_match.group(1).upper()
        if value != "N/A":
            metrics["capital_plan_status"] = value

    guidance_fields = {
        "GUIDANCE_COVERAGE_STATUS": "guidance_coverage_status",
        "GUIDANCE_SOURCE_DATE": "guidance_source_date",
        "GUIDANCE_SOURCE_URL": "guidance_source_url",
        "GUIDANCE_PERIOD": "guidance_period",
        "GUIDANCE_REVENUE": "guidance_revenue",
        "GUIDANCE_OPERATING_PROFIT": "guidance_operating_profit",
        "GUIDANCE_ORDINARY_OR_PRETAX_PROFIT": ("guidance_ordinary_or_pretax_profit"),
        "GUIDANCE_NET_INCOME": "guidance_net_income",
        "GUIDANCE_NET_INCOME_YOY": "guidance_net_income_yoy",
        "OPERATING_VS_NET_DIRECTION": "operating_vs_net_direction",
        "MATERIAL_NONOPERATING_DRIVER": "material_nonoperating_driver",
        "DRIVER_TYPE": "driver_type",
        "DRIVER_PERSISTENCE": "driver_persistence",
        "DRIVER_MATERIALITY": "driver_materiality",
        "DRIVER_AFFECTED_PERIOD": "driver_affected_period",
        "EARNINGS_BASELINE_STATUS": "earnings_baseline_status",
        "NORMALIZED_EARNINGS_AVAILABLE": "normalized_earnings_available",
        "GUIDANCE_BRIDGE_STATUS": "guidance_bridge_status",
    }
    for field_name, metric_name in guidance_fields.items():
        match = re.search(
            rf"^{re.escape(field_name)}:\s*([^\n]+)",
            data_block,
            re.IGNORECASE | re.MULTILINE,
        )
        if match:
            value = match.group(1).strip()
            if value.upper() not in {"N/A", "NA", "NONE", ""}:
                metrics[metric_name] = (
                    value.upper() if "_URL" not in field_name else value
                )

    sector_match = re.search(r"SECTOR:\s*(.+?)(?:\n|$)", data_block)
    if sector_match:
        metrics["sector"] = sector_match.group(1).strip().lower()

    industry_match = re.search(r"INDUSTRY:\s*(.+?)(?:\n|$)", data_block)
    if industry_match:
        metrics["industry"] = industry_match.group(1).strip().lower()

    metrics["debt_to_equity"] = extract_debt_to_equity(fundamentals_report)
    metrics["interest_coverage"] = extract_interest_coverage(fundamentals_report)
    metrics["fcf"] = extract_free_cash_flow(fundamentals_report)
    metrics["net_income"] = extract_net_income(fundamentals_report)
    if metrics["ocf"] is None:
        metrics["ocf"] = extract_operating_cash_flow(fundamentals_report)

    return metrics


def extract_debt_to_equity(report: str) -> float | None:
    """Extract D/E ratio, converting a ratio to percentage where needed."""
    patterns = [
        r"(?:^|\n)\s*-?\s*D/E:\s*(\d+(?:\.\d+)?)(%?)",
        r"(?:^|\n)\s*-?\s*Debt/Equity:\s*(\d+(?:\.\d+)?)(%?)",
        r"(?:^|\n)\s*-?\s*Debt-to-Equity:\s*(\d+(?:\.\d+)?)(%?)",
        r"D/E:\s*(\d+(?:\.\d+)?)(%?)",
        r"Debt/Equity:\s*(\d+(?:\.\d+)?)(%?)",
        r"DE_RATIO:\s*(\d+(?:\.\d+)?)(%?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, report, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if match.group(2):
                return value
            return value if value >= 10 else value * 100
    return None


def extract_interest_coverage(report: str) -> float | None:
    """Extract interest coverage ratio."""
    patterns = [
        r"\*\*Interest Coverage\*\*:\s*(\d+(?:\.\d+)?)x?",
        r"Interest Coverage:\s*(\d+(?:\.\d+)?)x?",
        r"Interest Coverage Ratio:\s*(\d+(?:\.\d+)?)x?",
    ]
    for pattern in patterns:
        match = re.search(pattern, report, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def extract_free_cash_flow(report: str) -> float | None:
    """Extract free cash flow with support for signs and B/M/K multipliers."""
    patterns = [
        # Canonical DATA_BLOCK field (fundamentals prompt v9.31)
        r"(?:^|\n)\s*FREE_CASH_FLOW:\s*([+-]?)[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
        r"\*\*Free Cash Flow\*\*:\s*([+-]?)[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
        r"(?:^|\n)\s*Free Cash Flow:\s*([+-]?)[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
        r"(?:^|\n)\s*FCF:\s*([+-]?)[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
        r"(?:Free Cash Flow|FCF):\s*([+-]?)[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
        r"Positive FCF:\s*[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
    ]
    for pattern in patterns:
        match = re.search(pattern, report, re.IGNORECASE | re.MULTILINE)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return parse_currency_value("", groups[0], groups[1])
            return parse_currency_value(groups[0], groups[1], groups[2])
    return None


def extract_net_income(report: str) -> float | None:
    """Extract net income with support for signs and B/M/K multipliers."""
    patterns = [
        r"\*\*Net Income\*\*:\s*([+-]?)[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
        r"(?:^|\n)\s*Net Income:\s*([+-]?)[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
        r"Net Income:\s*([+-]?)[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
    ]
    for pattern in patterns:
        match = re.search(pattern, report, re.IGNORECASE | re.MULTILINE)
        if match:
            groups = match.groups()
            return parse_currency_value(groups[0], groups[1], groups[2])
    return None


def extract_operating_cash_flow(report: str) -> float | None:
    """Extract operating cash flow with support for signs and B/M/K multipliers."""
    patterns = [
        r"\*\*Operating Cash Flow\*\*:\s*([+-]?)[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
        r"(?:^|\n)\s*Operating Cash Flow:\s*([+-]?)[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
        r"(?:^|\n)\s*OCF:\s*([+-]?)[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
        r"(?:Operating Cash Flow|OCF):\s*([+-]?)[$¥€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([BMK])?",
    ]
    for pattern in patterns:
        match = re.search(pattern, report, re.IGNORECASE | re.MULTILINE)
        if match:
            groups = match.groups()
            return parse_currency_value(groups[0], groups[1], groups[2])
    return None
