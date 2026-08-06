"""Metric extraction and derived-financial-signal helpers for SmartMarketDataFetcher."""

from __future__ import annotations

import math
import statistics
from datetime import date, timedelta
from typing import Any

import pandas as pd
import structlog

from src.config import config
from src.error_safety import summarize_exception
from src.sector_normalization import normalize_sector_label
from src.thesis_constants import SECTOR_MEDIAN_PE

logger = structlog.get_logger(__name__)

_MRQ_BASE_ABS_DELTA_BPS = 500.0
_MRQ_BASE_RELATIVE_DELTA = 0.30
_MRQ_BASE_MIN_REFERENCE_QUARTERS = 3


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if pd.notna(numeric) else None


STATEMENT_ROW_ALIASES: dict[str, tuple[str, ...]] = {
    "total_revenue": ("Total Revenue", "Operating Revenue", "Revenue"),
    "gross_profit": ("Gross Profit",),
    "operating_income": ("Operating Income", "Operating Income Loss"),
    "net_income": ("Net Income", "Net Income Common Stockholders"),
    "eps_diluted": ("Diluted EPS",),
    "eps_basic": ("Basic EPS",),
    "operating_cash_flow": (
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
    ),
    "capital_expenditure": ("Capital Expenditure", "Capital Expenditures"),
    "current_assets": ("Current Assets", "Total Current Assets"),
    "current_liabilities": ("Current Liabilities", "Total Current Liabilities"),
    "total_debt": ("Total Debt",),
    "long_term_debt": ("Long Term Debt", "Long Term Debt And Capital Lease Obligation"),
    "current_debt": ("Current Debt", "Current Debt And Capital Lease Obligation"),
    "stockholders_equity": ("Stockholders Equity", "Total Stockholder Equity"),
    "total_assets": ("Total Assets",),
    "cash_only": ("Cash And Cash Equivalents", "Cash"),
    "cash_and_short_term_investments": ("Cash And Short Term Investments",),
    "short_term_investments": ("Short Term Investments",),
    "cost_of_revenue": (
        "Cost Of Revenue",
        "Reconciled Cost Of Revenue",
        "Cost Of Goods Sold",
    ),
    "inventory": ("Inventory",),
}


def statement_value(df: pd.DataFrame, key: str, col: int = 0) -> float | None:
    if df.empty or key not in STATEMENT_ROW_ALIASES or len(df.columns) <= col:
        return None
    for label in STATEMENT_ROW_ALIASES[key]:
        if label in df.index:
            return _safe_float(df.loc[label].iloc[col])
    return None


def _canonical_sector(value: Any) -> str | None:
    normalized = normalize_sector_label(str(value) if value is not None else None)
    return normalized if normalized in SECTOR_MEDIAN_PE else None


def inventory_turnover_trend(
    financials: pd.DataFrame, balance_sheet: pd.DataFrame
) -> tuple[str, float] | None:
    """Inventory-turnover (COGS/inventory) trend over available years, latest-first.

    Distinguishes stocking-ahead-of-capacity from obsolescence for a build-out name:
    RISING turnover = selling faster; FALLING = inventory building faster than sales.
    Returns ``(trend, latest_turnover)`` or ``None`` when <2 comparable periods exist.
    """
    if financials.empty or balance_sheet.empty:
        return None
    years = min(len(financials.columns), len(balance_sheet.columns), 4)
    turnovers: list[float] = []
    for col in range(years):
        cogs = statement_value(financials, "cost_of_revenue", col=col)
        inv = statement_value(balance_sheet, "inventory", col=col)
        if cogs is None or not inv or inv <= 0:
            continue
        turnovers.append(abs(cogs) / inv)
    if len(turnovers) < 2:
        return None
    latest, oldest = turnovers[0], turnovers[-1]
    if oldest <= 0:
        return None
    if latest > oldest * 1.1:
        trend = "RISING"
    elif latest < oldest * 0.9:
        trend = "FALLING"
    else:
        trend = "STABLE"
    return trend, round(latest, 2)


def _statement_series(df: pd.DataFrame, key: str, max_years: int = 4) -> list[float]:
    values: list[float] = []
    if df.empty or key not in STATEMENT_ROW_ALIASES:
        return values
    for col in range(min(len(df.columns), max_years)):
        value = statement_value(df, key, col=col)
        if value is None:
            continue
        values.append(value)
    return values


def _statement_sum_series(
    df: pd.DataFrame,
    left_key: str,
    right_key: str,
    max_years: int = 4,
) -> list[float]:
    values: list[float] = []
    if df.empty:
        return values
    for col in range(min(len(df.columns), max_years)):
        left = statement_value(df, left_key, col=col)
        right = statement_value(df, right_key, col=col)
        if left is None or right is None:
            continue
        values.append(left + right)
    return values


def calculate_cagr_from_latest_series(values_latest_first: list[float]) -> float | None:
    """Return CAGR from annual values ordered newest to oldest."""
    if len(values_latest_first) < 4:
        return None
    latest = values_latest_first[0]
    oldest = values_latest_first[3]
    if latest <= 0 or oldest <= 0:
        return None
    return math.pow(latest / oldest, 1 / 3) - 1


def classify_cycle_position(current_roa: float, average_roa: float) -> str:
    if average_roa <= 0:
        return "MID"
    if current_roa > 1.5 * average_roa:
        return "PEAK"
    if current_roa < 0.6 * average_roa:
        return "TROUGH"
    return "MID"


def _log_statement_field_extraction_failed(
    symbol: str, key: str, exc: Exception
) -> None:
    logger.debug(
        "statement_field_extraction_failed",
        symbol=symbol,
        key=key,
        **summarize_exception(
            exc,
            operation="extracting financial statement field",
            provider="unknown",
        ),
    )


def extract_from_financial_statements(
    fetcher: Any, ticker, symbol: str
) -> dict[str, Any]:
    """Extract high-value metrics from yfinance statements and enrich with derived signals."""
    extracted: dict[str, Any] = {}
    financials = pd.DataFrame()
    cashflow = pd.DataFrame()
    balance_sheet = pd.DataFrame()
    try:
        financials = ticker.financials
        cashflow = ticker.cashflow
        balance_sheet = ticker.balance_sheet
        if financials.empty and cashflow.empty and balance_sheet.empty:
            return extracted

        fetcher.stats["sources"]["statements"] += 1

        if not financials.empty:
            # Annual statements can lag a full fiscal year for some ex-US names
            # (yfinance carries only through the prior FY). Flag when the latest
            # annual column is older than a normal reporting lag, so a stale FY
            # growth figure is surfaced rather than silently trusted (consumed by
            # the GROWTH_DATA over-interpretation guard downstream).
            latest_col = financials.columns[0]
            if hasattr(latest_col, "to_pydatetime"):
                age_days = (date.today() - latest_col.date()).days
                extracted["_income_statement_date"] = latest_col.date().isoformat()
                extracted["_statements_age_days"] = age_days
                if age_days > 457:  # ~15 months: a completed FY is likely missing
                    extracted["statements_stale"] = True
                    extracted["_statements_stale_source"] = "calculated_from_statements"
            if len(financials.columns) >= 2:
                try:
                    current = statement_value(financials, "total_revenue", col=0)
                    previous = statement_value(financials, "total_revenue", col=1)
                    if current is not None and previous:
                        growth = (current - previous) / previous
                        if -0.5 < growth < 5.0:
                            extracted["revenueGrowth"] = growth
                            extracted["_revenueGrowth_source"] = (
                                "calculated_from_statements"
                            )
                except Exception as exc:
                    _log_statement_field_extraction_failed(
                        symbol, "revenue_growth", exc
                    )
                # FY EPS growth from the filed EPS rows (diluted > basic). Net-income
                # growth is only a last-resort proxy (NI != EPS once share count moves
                # via buybacks/dilution/splits), and is tagged as such so its lower
                # reliability is visible to the merge and downstream.
                try:
                    eps_cur = statement_value(financials, "eps_diluted", col=0)
                    eps_prev = statement_value(financials, "eps_diluted", col=1)
                    eps_src = "calculated_from_statement_diluted_eps"
                    if eps_cur is None or eps_prev is None:
                        eps_cur = statement_value(financials, "eps_basic", col=0)
                        eps_prev = statement_value(financials, "eps_basic", col=1)
                        eps_src = "calculated_from_statement_basic_eps"
                    if eps_cur is None or eps_prev is None:
                        eps_cur = statement_value(financials, "net_income", col=0)
                        eps_prev = statement_value(financials, "net_income", col=1)
                        eps_src = "calculated_from_statement_net_income_proxy"
                    # A non-positive prior base makes a YoY % meaningless; leave N/A
                    # rather than fabricate a growth figure from a loss year.
                    if eps_cur is not None and eps_prev is not None and eps_prev > 0:
                        eps_growth = (eps_cur - eps_prev) / eps_prev
                        if -1.0 < eps_growth < 5.0:
                            extracted["earningsGrowth"] = eps_growth
                            extracted["_earningsGrowth_source"] = eps_src
                except Exception as exc:
                    _log_statement_field_extraction_failed(
                        symbol, "earnings_growth", exc
                    )
            try:
                revenue_cagr = calculate_cagr_from_latest_series(
                    _statement_series(financials, "total_revenue")
                )
                if revenue_cagr is not None:
                    extracted["revenue_cagr_3y"] = revenue_cagr
                    extracted["_revenue_cagr_3y_source"] = "calculated_from_statements"
            except Exception as exc:
                _log_statement_field_extraction_failed(symbol, "revenue_cagr_3y", exc)

            try:
                revenue = statement_value(financials, "total_revenue")
                if revenue:
                    gross_profit = statement_value(financials, "gross_profit")
                    if gross_profit is not None:
                        extracted["grossMargins"] = gross_profit / revenue
                        extracted["_grossMargins_source"] = "calculated_from_statements"

                    op_income = statement_value(financials, "operating_income")
                    if op_income is not None:
                        extracted["operatingMargins"] = op_income / revenue
                        extracted["_operatingMargins_source"] = (
                            "calculated_from_statements"
                        )

                    net_income = statement_value(financials, "net_income")
                    if net_income is not None:
                        extracted["profitMargins"] = net_income / revenue
                        extracted["_profitMargins_source"] = (
                            "calculated_from_statements"
                        )
            except Exception as exc:
                _log_statement_field_extraction_failed(symbol, "margins", exc)

        if not cashflow.empty:
            try:
                ocf = statement_value(cashflow, "operating_cash_flow")
                if ocf is not None:
                    extracted["operatingCashflow"] = ocf
                    extracted["_operatingCashflow_source"] = "extracted_from_statements"
                capex = statement_value(cashflow, "capital_expenditure")
                if ocf is not None and capex is not None:
                    extracted["freeCashflow"] = ocf + capex
                    extracted["_freeCashflow_source"] = "calculated_from_statements"
            except Exception as exc:
                _log_statement_field_extraction_failed(symbol, "cashflow", exc)
            try:
                fcf_values = _statement_sum_series(
                    cashflow, "operating_cash_flow", "capital_expenditure"
                )
                fcf_cagr = calculate_cagr_from_latest_series(fcf_values)
                if fcf_cagr is not None:
                    extracted["fcf_cagr_3y"] = fcf_cagr
                    extracted["_fcf_cagr_3y_source"] = "calculated_from_statements"
            except Exception as exc:
                _log_statement_field_extraction_failed(symbol, "fcf_cagr_3y", exc)

        if not balance_sheet.empty:
            statement_date = balance_sheet.columns[0]
            extracted["_statements_date"] = (
                statement_date.strftime("%Y-%m-%d")
                if hasattr(statement_date, "strftime")
                else str(statement_date)
            )
            try:
                current_assets = statement_value(balance_sheet, "current_assets")
                current_liabilities = statement_value(
                    balance_sheet, "current_liabilities"
                )
                if current_assets is not None and current_liabilities:
                    extracted["currentRatio"] = current_assets / current_liabilities
                    extracted["_currentRatio_source"] = "calculated_from_statements"
            except Exception as exc:
                _log_statement_field_extraction_failed(symbol, "current_ratio", exc)

            try:
                inv_trend = inventory_turnover_trend(financials, balance_sheet)
                if inv_trend is not None:
                    extracted["inventoryTurnoverTrend"] = inv_trend[0]
                    extracted["inventoryTurnoverLatest"] = inv_trend[1]
                    extracted["_inventoryTurnoverTrend_source"] = (
                        "calculated_from_statements"
                    )
            except Exception as exc:
                _log_statement_field_extraction_failed(
                    symbol, "inventory_turnover_trend", exc
                )

            try:
                debt = statement_value(balance_sheet, "total_debt")
                if debt is None:
                    long_term = statement_value(balance_sheet, "long_term_debt")
                    short_term = statement_value(balance_sheet, "current_debt") or 0
                    debt = None if long_term is None else long_term + short_term
                equity = statement_value(balance_sheet, "stockholders_equity")

                if debt is not None and equity is not None and equity != 0:
                    extracted["debtToEquity"] = debt / equity
                    extracted["_debtToEquity_source"] = "calculated_from_statements"
            except Exception as exc:
                _log_statement_field_extraction_failed(symbol, "debt_to_equity", exc)

            try:
                total_assets = statement_value(balance_sheet, "total_assets")
                if total_assets is not None:
                    extracted["totalAssets"] = total_assets
                    extracted["_totalAssets_source"] = "calculated_from_statements"

                liquid = statement_value(balance_sheet, "cash_only")
                if liquid is not None:
                    sti = (
                        statement_value(balance_sheet, "short_term_investments") or 0.0
                    )
                    extracted["cashAndShortTermInvestments"] = liquid + sti
                    extracted["_cashAndShortTermInvestments_source"] = (
                        "calculated_from_statements"
                    )
                else:
                    bundled_liquid = statement_value(
                        balance_sheet, "cash_and_short_term_investments"
                    )
                    if bundled_liquid is not None:
                        extracted["cashAndShortTermInvestments"] = bundled_liquid
                        extracted["_cashAndShortTermInvestments_source"] = (
                            "extracted_from_statements"
                        )
            except Exception as exc:
                _log_statement_field_extraction_failed(
                    symbol, "balance_sheet_assets", exc
                )
    except Exception as exc:
        logger.debug(
            "statement_extraction_failed",
            symbol=symbol,
            **summarize_exception(
                exc,
                operation="extracting financial statements",
                provider="unknown",
            ),
        )

    moat_signals = calculate_moat_signals(financials, cashflow, symbol)
    for key, value in moat_signals.items():
        extracted[key] = value
        extracted[f"_{key}_source"] = "calculated_from_statements"

    capital_signals = calculate_capital_efficiency_signals(
        income_stmt=financials,
        balance_sheet=balance_sheet,
        info=extracted,
        symbol=symbol,
        cashflow=cashflow,
    )
    for key, value in capital_signals.items():
        extracted[key] = value

    return_trends = calculate_return_trends(financials, balance_sheet, symbol)
    for key, value in return_trends.items():
        extracted[key] = value

    graham_signals = calculate_graham_earnings_test(financials, symbol)
    for key, value in graham_signals.items():
        extracted[key] = value

    quarterly_horizons = extract_quarterly_horizons(ticker, symbol)
    for key, value in quarterly_horizons.items():
        extracted[key] = value
    return extracted


def extract_quarterly_horizons(ticker, symbol: str) -> dict[str, Any]:
    """Extract TTM and MRQ growth/earnings/cash-flow horizons from quarterly statements."""
    extracted: dict[str, Any] = {}
    diagnostics: list[dict[str, str]] = []

    def add_diagnostic(field: str, reason: str) -> None:
        diagnostics.append(
            {
                "field": field,
                "status": "unavailable",
                "reason": reason,
            }
        )

    try:
        qt_inc = ticker.quarterly_financials
        qt_cf = ticker.quarterly_cashflow
    except Exception as exc:
        logger.debug(
            "quarterly_data_unavailable",
            symbol=symbol,
            **summarize_exception(
                exc,
                operation="extracting quarterly data",
                provider="unknown",
            ),
        )
        add_diagnostic("quarterly_horizons", "quarterly_data_unavailable")
        extracted["_quarterly_diagnostics"] = diagnostics
        return extracted

    if qt_inc is not None and not qt_inc.empty:
        latest_q_date = qt_inc.columns[0]
        extracted["latest_quarter_date"] = str(latest_q_date.date())
        extracted["_latest_quarter_date_source"] = "yfinance_quarterly"

    income_available = qt_inc is not None and not qt_inc.empty
    cashflow_available = qt_cf is not None and not qt_cf.empty

    def _find_yoy_match_idx(
        series_index: pd.DatetimeIndex, latest_date: pd.Timestamp
    ) -> int | None:
        target = latest_date - pd.DateOffset(months=12)
        best_idx = None
        best_delta = timedelta(days=999)
        for i, dt in enumerate(series_index):
            if i == 0:
                continue
            delta = abs(dt - target)
            if delta < best_delta and delta < timedelta(days=45):
                best_delta = delta
                best_idx = i
        return best_idx

    rev_series: pd.Series | None = None
    ni_series: pd.Series | None = None

    if income_available and "Total Revenue" in qt_inc.index:
        rev_raw = qt_inc.loc["Total Revenue"]
        rev_series = rev_raw.dropna()
        if len(rev_series) >= 5:
            match_idx = _find_yoy_match_idx(rev_series.index, rev_series.index[0])
            if match_idx is not None:
                mrq_current = float(rev_series.iloc[0])
                mrq_prior = float(rev_series.iloc[match_idx])
                if mrq_prior > 0:
                    mrq_growth = (mrq_current - mrq_prior) / mrq_prior
                    if -1.0 < mrq_growth < 10.0:
                        extracted["revenueGrowth_MRQ"] = mrq_growth
                        extracted["_revenueGrowth_MRQ_source"] = (
                            "calculated_from_quarterly"
                        )
                else:
                    add_diagnostic("revenueGrowth_MRQ", "nonpositive_prior_year_value")
            else:
                add_diagnostic("revenueGrowth_MRQ", "prior_year_quarter_not_matched")
        else:
            add_diagnostic("revenueGrowth_MRQ", "insufficient_quarterly_history")
        if len(rev_series) >= 8:
            ttm_current = rev_series.iloc[0:4].sum(min_count=4)
            ttm_prior = rev_series.iloc[4:8].sum(min_count=4)
            if pd.notna(ttm_current) and pd.notna(ttm_prior) and ttm_prior > 0:
                ttm_growth = (ttm_current - ttm_prior) / ttm_prior
                if -1.0 < ttm_growth < 10.0:
                    extracted["revenueGrowth_TTM"] = float(ttm_growth)
                    extracted["_revenueGrowth_TTM_source"] = "calculated_from_quarterly"
            if pd.notna(ttm_current):
                extracted["revenue_TTM"] = float(ttm_current)
                extracted["_revenue_TTM_source"] = "calculated_from_quarterly"
            else:
                add_diagnostic("revenue_TTM", "malformed_quarterly_window")
            if "revenueGrowth_TTM" not in extracted:
                add_diagnostic("revenueGrowth_TTM", "malformed_or_invalid_ttm_window")
        else:
            reason = (
                "malformed_quarterly_window"
                if len(rev_raw) >= 8
                else "insufficient_quarterly_history"
            )
            add_diagnostic("revenueGrowth_TTM", reason)
            add_diagnostic("revenue_TTM", reason)
    elif income_available:
        add_diagnostic("revenueGrowth_MRQ", "missing_total_revenue_row")
        add_diagnostic("revenueGrowth_TTM", "missing_total_revenue_row")
        add_diagnostic("revenue_TTM", "missing_total_revenue_row")
    else:
        add_diagnostic("revenueGrowth_MRQ", "quarterly_income_unavailable")
        add_diagnostic("revenueGrowth_TTM", "quarterly_income_unavailable")
        add_diagnostic("revenue_TTM", "quarterly_income_unavailable")

    if income_available and "Net Income" in qt_inc.index:
        ni_raw = qt_inc.loc["Net Income"]
        ni_series = ni_raw.dropna()
        if len(ni_series) >= 5:
            match_idx = _find_yoy_match_idx(ni_series.index, ni_series.index[0])
            if match_idx is not None:
                mrq_ni = float(ni_series.iloc[0])
                mrq_ni_prior = float(ni_series.iloc[match_idx])
                if mrq_ni_prior > 0:
                    mrq_ni_growth = (mrq_ni - mrq_ni_prior) / mrq_ni_prior
                    if -5.0 < mrq_ni_growth < 50.0:
                        extracted["earningsGrowth_MRQ"] = mrq_ni_growth
                        extracted["_earningsGrowth_MRQ_source"] = (
                            "calculated_from_quarterly"
                        )
                else:
                    add_diagnostic("earningsGrowth_MRQ", "nonpositive_prior_year_value")
            else:
                add_diagnostic("earningsGrowth_MRQ", "prior_year_quarter_not_matched")
        else:
            add_diagnostic("earningsGrowth_MRQ", "insufficient_quarterly_history")
        if len(ni_series) >= 4:
            ttm_ni = ni_series.iloc[0:4].sum(min_count=4)
            if pd.notna(ttm_ni):
                extracted["netIncome_TTM"] = float(ttm_ni)
                extracted["_netIncome_TTM_source"] = "calculated_from_quarterly"
            else:
                add_diagnostic("netIncome_TTM", "malformed_quarterly_window")
            if len(ni_series) >= 8:
                ttm_ni_prior = ni_series.iloc[4:8].sum(min_count=4)
                if pd.notna(ttm_ni) and pd.notna(ttm_ni_prior) and ttm_ni_prior > 0:
                    ttm_ni_growth = (ttm_ni - ttm_ni_prior) / ttm_ni_prior
                    if -5.0 < ttm_ni_growth < 50.0:
                        extracted["earningsGrowth_TTM"] = float(ttm_ni_growth)
                        extracted["_earningsGrowth_TTM_source"] = (
                            "calculated_from_quarterly"
                        )
                if "earningsGrowth_TTM" not in extracted:
                    add_diagnostic(
                        "earningsGrowth_TTM", "malformed_or_invalid_ttm_window"
                    )
            else:
                add_diagnostic(
                    "earningsGrowth_TTM",
                    "malformed_quarterly_window"
                    if len(ni_raw) >= 8
                    else "insufficient_quarterly_history",
                )
        else:
            reason = (
                "malformed_quarterly_window"
                if len(ni_raw) >= 4
                else "insufficient_quarterly_history"
            )
            add_diagnostic("netIncome_TTM", reason)
            add_diagnostic("earningsGrowth_TTM", reason)
    elif income_available:
        add_diagnostic("earningsGrowth_MRQ", "missing_net_income_row")
        add_diagnostic("earningsGrowth_TTM", "missing_net_income_row")
        add_diagnostic("netIncome_TTM", "missing_net_income_row")
    else:
        add_diagnostic("earningsGrowth_MRQ", "quarterly_income_unavailable")
        add_diagnostic("earningsGrowth_TTM", "quarterly_income_unavailable")
        add_diagnostic("netIncome_TTM", "quarterly_income_unavailable")

    if rev_series is not None and ni_series is not None:
        common_dates = rev_series.index.intersection(ni_series.index)
        comparison_status = "UNKNOWN"
        comparison_delta_bps: float | None = None
        if len(common_dates) >= 5:
            latest_date = common_dates[0]
            prior_idx = _find_yoy_match_idx(common_dates, latest_date)
            if prior_idx is not None:
                prior_date = common_dates[prior_idx]
                prior_revenue = float(rev_series.loc[prior_date])
                prior_net_income = float(ni_series.loc[prior_date])
                if prior_revenue > 0 and prior_net_income <= 0:
                    comparison_status = "NONPOSITIVE"
                elif prior_revenue > 0:
                    reference_margins = [
                        float(ni_series.loc[date]) / float(rev_series.loc[date])
                        for date in common_dates
                        if date not in {latest_date, prior_date}
                        and float(rev_series.loc[date]) > 0
                    ]
                    if len(reference_margins) >= _MRQ_BASE_MIN_REFERENCE_QUARTERS:
                        reference_margin = float(statistics.median(reference_margins))
                        prior_margin = prior_net_income / prior_revenue
                        if reference_margin > 0:
                            comparison_delta_bps = (
                                prior_margin - reference_margin
                            ) * 10_000
                            relative_delta = (
                                prior_margin - reference_margin
                            ) / reference_margin
                            if (
                                comparison_delta_bps <= -_MRQ_BASE_ABS_DELTA_BPS
                                or relative_delta <= -_MRQ_BASE_RELATIVE_DELTA
                            ):
                                comparison_status = "DEPRESSED"
                            elif (
                                comparison_delta_bps >= _MRQ_BASE_ABS_DELTA_BPS
                                or relative_delta >= _MRQ_BASE_RELATIVE_DELTA
                            ):
                                comparison_status = "ELEVATED"
                            else:
                                comparison_status = "NORMAL"
        extracted["mrq_comparison_base_status"] = comparison_status
        extracted["_mrq_comparison_base_status_source"] = (
            "calculated_from_quarterly_margins"
        )
        if comparison_delta_bps is not None:
            extracted["mrq_comparison_base_margin_delta_bps"] = round(
                comparison_delta_bps, 1
            )
            extracted["_mrq_comparison_base_margin_delta_bps_source"] = (
                "calculated_from_quarterly_margins"
            )

    if cashflow_available and "Operating Cash Flow" in qt_cf.index:
        ocf_raw = qt_cf.loc["Operating Cash Flow"]
        ocf_series = ocf_raw.dropna()
        if len(ocf_series) >= 4:
            ttm_ocf = ocf_series.iloc[0:4].sum(min_count=4)
            if pd.notna(ttm_ocf):
                extracted["operatingCashflow_TTM"] = float(ttm_ocf)
                extracted["_operatingCashflow_TTM_source"] = "calculated_from_quarterly"
            else:
                add_diagnostic("operatingCashflow_TTM", "malformed_quarterly_window")
        else:
            add_diagnostic(
                "operatingCashflow_TTM",
                "malformed_quarterly_window"
                if len(ocf_raw) >= 4
                else "insufficient_quarterly_history",
            )
    elif cashflow_available:
        add_diagnostic("operatingCashflow_TTM", "missing_operating_cash_flow_row")
    else:
        add_diagnostic("operatingCashflow_TTM", "quarterly_cashflow_unavailable")

    if (
        cashflow_available
        and "Operating Cash Flow" in qt_cf.index
        and "Capital Expenditure" in qt_cf.index
    ):
        ocf_s = qt_cf.loc["Operating Cash Flow"].dropna()
        capex_s = qt_cf.loc["Capital Expenditure"].dropna()
        common_dates = ocf_s.index.intersection(capex_s.index)[:4]
        if len(common_dates) >= 4:
            ttm_fcf_ocf = ocf_s[common_dates].sum(min_count=4)
            ttm_fcf_capex = capex_s[common_dates].sum(min_count=4)
            if pd.notna(ttm_fcf_ocf) and pd.notna(ttm_fcf_capex):
                extracted["freeCashflow_TTM"] = float(ttm_fcf_ocf + ttm_fcf_capex)
                extracted["_freeCashflow_TTM_source"] = "calculated_from_quarterly"
            else:
                add_diagnostic("freeCashflow_TTM", "malformed_quarterly_window")
        else:
            add_diagnostic("freeCashflow_TTM", "insufficient_quarterly_history")
    elif cashflow_available:
        add_diagnostic("freeCashflow_TTM", "missing_cashflow_or_capex_row")
    else:
        add_diagnostic("freeCashflow_TTM", "quarterly_cashflow_unavailable")

    extracted_mrq_growth = _safe_float(extracted.get("revenueGrowth_MRQ"))
    ttm_growth = _safe_float(extracted.get("revenueGrowth_TTM"))
    if extracted_mrq_growth is not None and ttm_growth is not None:
        delta = extracted_mrq_growth - ttm_growth
        if delta > 0.10:
            extracted["growth_trajectory"] = "ACCELERATING"
        elif delta < -0.10:
            extracted["growth_trajectory"] = "DECELERATING"
        else:
            extracted["growth_trajectory"] = "STABLE"
        extracted["_growth_trajectory_source"] = "calculated_from_quarterly"

    if (
        extracted.get("growth_trajectory") == "ACCELERATING"
        and extracted.get("earningsGrowth_TTM") is not None
        and extracted["earningsGrowth_TTM"] < -0.05
    ):
        extracted["growth_trajectory"] = "MIXED"
        extracted["_growth_trajectory_source"] = (
            f"{extracted.get('_growth_trajectory_source', 'calculated_from_quarterly')}|eps_divergence"
        )

    if extracted:
        logger.debug(
            "quarterly_horizons_extracted",
            symbol=symbol,
            fields=sorted(key for key in extracted if not key.startswith("_")),
        )
    if diagnostics:
        extracted["_quarterly_diagnostics"] = diagnostics
        logger.debug(
            "quarterly_horizon_diagnostics",
            symbol=symbol,
            diagnostics=diagnostics,
        )
    return extracted


def calculate_moat_signals(
    financials: pd.DataFrame, cashflow: pd.DataFrame, symbol: str
) -> dict[str, Any]:
    """Calculate gross-margin stability and cash-conversion moat signals."""
    signals: dict[str, Any] = {}
    if financials.empty or len(financials.columns) < 3:
        logger.debug("moat_signals_insufficient_data", symbol=symbol, years=0)
        return signals

    try:
        if "Gross Profit" in financials.index and "Total Revenue" in financials.index:
            margins: list[float] = []
            for i in range(min(5, len(financials.columns))):
                try:
                    gross_profit = financials.loc["Gross Profit"].iloc[i]
                    revenue = financials.loc["Total Revenue"].iloc[i]
                    if pd.notna(gross_profit) and pd.notna(revenue) and revenue != 0:
                        margin = float(gross_profit) / float(revenue)
                        if -0.5 < margin < 1.0:
                            margins.append(margin)
                except (ValueError, TypeError, KeyError):
                    continue
            if len(margins) >= 3:
                mean_margin = statistics.mean(margins)
                if mean_margin > 0.05:
                    cv = statistics.stdev(margins) / mean_margin
                    signals["moat_grossMarginCV"] = round(cv, 4)
                    signals["moat_grossMarginAvg"] = round(mean_margin, 4)
                    signals["moat_grossMarginYears"] = len(margins)
                    signals["moat_marginStability"] = (
                        "HIGH" if cv < 0.08 else "MEDIUM" if cv < 0.15 else "LOW"
                    )
    except Exception as exc:
        logger.debug(
            "moat_margin_calc_failed",
            symbol=symbol,
            **summarize_exception(
                exc,
                operation="calculating moat margin stability",
                provider="unknown",
            ),
        )

    try:
        if (
            not cashflow.empty
            and "Operating Cash Flow" in cashflow.index
            and "Net Income" in financials.index
        ):
            ratios: list[float] = []
            for i in range(min(3, len(financials.columns), len(cashflow.columns))):
                try:
                    ocf = cashflow.loc["Operating Cash Flow"].iloc[i]
                    ni = financials.loc["Net Income"].iloc[i]
                    if pd.notna(ocf) and pd.notna(ni) and float(ni) > 0:
                        ratio = float(ocf) / float(ni)
                        if 0.1 < ratio < 3.0:
                            ratios.append(ratio)
                except (ValueError, TypeError, KeyError):
                    continue
            if len(ratios) >= 2:
                avg_ratio = statistics.mean(ratios)
                signals["moat_cfoToNiAvg"] = round(avg_ratio, 4)
                signals["moat_cfoToNiYears"] = len(ratios)
                signals["moat_cashConversion"] = (
                    "STRONG"
                    if avg_ratio > 0.90
                    else "ADEQUATE"
                    if avg_ratio > 0.70
                    else "WEAK"
                )
    except Exception as exc:
        logger.debug(
            "moat_cash_conversion_failed",
            symbol=symbol,
            **summarize_exception(
                exc,
                operation="calculating moat cash conversion",
                provider="unknown",
            ),
        )
    return signals


def calculate_capital_efficiency_signals(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    info: dict[str, Any],
    symbol: str,
    cashflow: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Calculate ROIC, leverage-quality, and idle-cash-supporting signals."""
    signals: dict[str, Any] = {}
    try:
        ebit = None
        tax_rate = None
        invested_capital = None
        total_debt = _safe_float(info.get("totalDebt"))
        cash = _safe_float(
            info.get("cashAndShortTermInvestments") or info.get("totalCash")
        )
        total_assets = _safe_float(info.get("totalAssets"))
        market_cap = _safe_float(info.get("marketCap"))
        capex = None
        d_and_a = None

        if not income_stmt.empty and len(income_stmt.columns) > 0:
            if "EBIT" in income_stmt.index:
                val = income_stmt.loc["EBIT"].iloc[0]
                if pd.notna(val):
                    ebit = float(val)
            if "Tax Rate For Calcs" in income_stmt.index:
                val = income_stmt.loc["Tax Rate For Calcs"].iloc[0]
                if pd.notna(val):
                    tax_rate = float(val)

        if not balance_sheet.empty and len(balance_sheet.columns) > 0:
            if "Invested Capital" in balance_sheet.index:
                val = balance_sheet.loc["Invested Capital"].iloc[0]
                if pd.notna(val) and val > 0:
                    invested_capital = float(val)
            if total_debt is None:
                if "Total Debt" in balance_sheet.index:
                    val = balance_sheet.loc["Total Debt"].iloc[0]
                    if pd.notna(val):
                        total_debt = float(val)
                elif "Long Term Debt" in balance_sheet.index:
                    long_term = balance_sheet.loc["Long Term Debt"].iloc[0]
                    short_term = (
                        balance_sheet.loc["Current Debt"].iloc[0]
                        if "Current Debt" in balance_sheet.index
                        else 0
                    )
                    if pd.notna(long_term) and pd.notna(short_term):
                        total_debt = float(long_term) + float(short_term)
            if total_assets is None and "Total Assets" in balance_sheet.index:
                val = balance_sheet.loc["Total Assets"].iloc[0]
                if pd.notna(val):
                    total_assets = float(val)
            if cash is None:
                used_combined_cash_row = False
                for cash_row in [
                    "Cash And Short Term Investments",
                    "Cash And Cash Equivalents",
                    "Cash",
                ]:
                    if cash_row in balance_sheet.index:
                        val = balance_sheet.loc[cash_row].iloc[0]
                        if pd.notna(val):
                            cash = float(val)
                            used_combined_cash_row = (
                                cash_row == "Cash And Short Term Investments"
                            )
                            break
                if (
                    cash is not None
                    and not used_combined_cash_row
                    and "Short Term Investments" in balance_sheet.index
                ):
                    sti = balance_sheet.loc["Short Term Investments"].iloc[0]
                    if pd.notna(sti):
                        cash += float(sti)

        if cashflow is not None and not cashflow.empty and len(cashflow.columns) > 0:
            if "Capital Expenditure" in cashflow.index:
                val = cashflow.loc["Capital Expenditure"].iloc[0]
                if pd.notna(val):
                    capex = float(val)
            for da_row in (
                "Depreciation And Amortization",
                "Depreciation Amortization Depletion",
                "Depreciation & Amortization",
                "Depreciation",
            ):
                if da_row in cashflow.index:
                    val = cashflow.loc[da_row].iloc[0]
                    if pd.notna(val):
                        d_and_a = float(val)
                        break

        roe = info.get("returnOnEquity")
        roic = None
        if ebit is not None and invested_capital is not None and invested_capital > 0:
            effective_tax = max(
                0.0, min(0.5, tax_rate if tax_rate is not None else 0.21)
            )
            nopat = ebit * (1 - effective_tax)
            roic = nopat / invested_capital
            signals["capital_roic"] = round(roic, 4)
            signals["capital_roic_source"] = "calculated"
            if roic < 0:
                signals["capital_roicQuality"] = "DESTRUCTIVE"
            elif roic < config.roic_hurdle_rate:
                signals["capital_roicQuality"] = "WEAK"
            elif roic < config.roic_strong_threshold:
                signals["capital_roicQuality"] = "ADEQUATE"
            else:
                signals["capital_roicQuality"] = "STRONG"
            signals["capital_hurdleSpread"] = round(roic - config.roic_hurdle_rate, 4)

        if roic is not None and roe is not None:
            if roic <= 0 and roe > 0:
                signals["capital_leverageQuality"] = "VALUE_DESTRUCTION"
            elif roic > 0:
                ratio = roe / roic
                signals["capital_roeRoicRatio"] = round(ratio, 2)
                if ratio > config.leverage_engineered_ratio:
                    signals["capital_leverageQuality"] = "ENGINEERED"
                elif ratio > config.leverage_suspect_ratio:
                    signals["capital_leverageQuality"] = "SUSPECT"
                elif ratio < 1.0:
                    signals["capital_leverageQuality"] = "CONSERVATIVE"
                else:
                    signals["capital_leverageQuality"] = "GENUINE"

        if (
            cash is not None
            and total_debt is not None
            and market_cap
            and market_cap > 0
        ):
            signals["capital_netCashToMarketCap"] = round(
                (cash - total_debt) / market_cap, 4
            )
        if cash is not None and total_assets and total_assets > 0:
            signals["capital_cashToAssets"] = round(cash / total_assets, 4)
        # Asset turnover (revenue / total assets): a structural proxy for the
        # low-margin/high-turnover distribution model, used to gate relaxed margin
        # floors for distributors (see SECTOR_OPERATING_MARGIN_MIN).
        revenue = _safe_float(info.get("totalRevenue"))
        if (
            revenue is None
            and not income_stmt.empty
            and "Total Revenue" in income_stmt.index
        ):
            val = income_stmt.loc["Total Revenue"].iloc[0]
            if pd.notna(val):
                revenue = float(val)
        if revenue is not None and total_assets and total_assets > 0:
            signals["capital_assetTurnover"] = round(revenue / total_assets, 2)
        if capex is not None and d_and_a is not None and d_and_a != 0:
            capex_to_da_ratio = abs(capex) / abs(d_and_a)
            signals["capital_capexToDaRatio"] = round(capex_to_da_ratio, 2)
            if capex_to_da_ratio < config.capex_to_da_underinvesting_threshold:
                signals["capital_capexToDaStatus"] = "UNDERINVESTING"
            elif capex_to_da_ratio > config.capex_to_da_growth_threshold:
                signals["capital_capexToDaStatus"] = "GROWTH_INVESTING"
            else:
                signals["capital_capexToDaStatus"] = "MAINTENANCE"
    except Exception as exc:
        logger.debug(
            "capital_efficiency_calculation_failed",
            symbol=symbol,
            **summarize_exception(
                exc,
                operation="calculating capital efficiency signals",
                provider="unknown",
            ),
        )
    return signals


def compute_trend_regression(values: list[float], mean_val: float) -> str:
    """Determine profitability trend with CV guard and regression slope."""
    n = len(values)
    if n < 3 or mean_val == 0:
        return "N/A"
    try:
        cv = abs(statistics.stdev(values) / mean_val) if mean_val != 0 else 0
    except statistics.StatisticsError:
        cv = 0
    if cv > 0.40:
        return "UNSTABLE"
    x_mean = (n - 1) / 2.0
    numerator = sum((i - x_mean) * (v - mean_val) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return "STABLE"
    slope_pct = (numerator / denominator) / abs(mean_val) if mean_val != 0 else 0
    if slope_pct > 0.005:
        return "IMPROVING"
    if slope_pct < -0.005:
        return "DECLINING"
    return "STABLE"


def calculate_return_trends(
    financials: pd.DataFrame, balance_sheet: pd.DataFrame, symbol: str
) -> dict[str, Any]:
    """Calculate 5-year ROA/ROE averages and profitability trend."""
    signals: dict[str, Any] = {}
    if financials.empty or balance_sheet.empty:
        return signals
    years_available = min(len(financials.columns), len(balance_sheet.columns), 5)
    if years_available < 3:
        logger.debug(
            "return_trends_insufficient_data", symbol=symbol, years=years_available
        )
        return signals

    try:
        if "Net Income" in financials.index and "Total Assets" in balance_sheet.index:
            roas: list[float] = []
            for i in range(years_available):
                try:
                    ni = financials.loc["Net Income"].iloc[i]
                    assets = balance_sheet.loc["Total Assets"].iloc[i]
                    if pd.notna(ni) and pd.notna(assets) and float(assets) > 0:
                        roa = float(ni) / float(assets)
                        if -0.50 < roa < 0.50:
                            roas.append(roa)
                except (ValueError, TypeError, IndexError):
                    continue
            if len(roas) >= 3:
                avg_roa = statistics.mean(roas)
                signals["roa_5y_avg"] = round(avg_roa * 100, 2)
                signals["_roa_5y_years"] = len(roas)
                signals["cycle_position"] = classify_cycle_position(roas[0], avg_roa)
                signals["profitability_trend"] = compute_trend_regression(
                    list(reversed(roas)), avg_roa
                )
    except Exception as exc:
        logger.debug(
            "roa_trend_calc_failed",
            symbol=symbol,
            **summarize_exception(
                exc,
                operation="calculating ROA trend",
                provider="unknown",
            ),
        )

    try:
        equity_key = (
            "Stockholders Equity"
            if "Stockholders Equity" in balance_sheet.index
            else (
                "Total Stockholder Equity"
                if "Total Stockholder Equity" in balance_sheet.index
                else None
            )
        )
        if "Net Income" in financials.index and equity_key:
            roes: list[float] = []
            for i in range(years_available):
                try:
                    ni = financials.loc["Net Income"].iloc[i]
                    equity = balance_sheet.loc[equity_key].iloc[i]
                    if pd.notna(ni) and pd.notna(equity) and float(equity) > 0:
                        roe = float(ni) / float(equity)
                        if -1.0 < roe < 1.0:
                            roes.append(roe)
                except (ValueError, TypeError, IndexError):
                    continue
            if len(roes) >= 3:
                signals["roe_5y_avg"] = round(statistics.mean(roes) * 100, 2)
                signals["_roe_5y_years"] = len(roes)
    except Exception as exc:
        logger.debug(
            "roe_trend_calc_failed",
            symbol=symbol,
            **summarize_exception(
                exc,
                operation="calculating ROE trend",
                provider="unknown",
            ),
        )
    return signals


def calculate_graham_earnings_test(
    financials: pd.DataFrame, symbol: str
) -> dict[str, Any]:
    """Run a Graham-style consecutive positive earnings test."""
    signals: dict[str, Any] = {}
    try:
        if (
            financials.empty
            or len(financials.columns) == 0
            or "Net Income" not in financials.index
        ):
            signals["graham_consecutive_positive_years"] = None
            signals["graham_test"] = "INSUFFICIENT_DATA"
            return signals
        net_incomes = financials.loc["Net Income"]
        consecutive_positive = 0
        for ni in net_incomes:
            if pd.notna(ni) and float(ni) > 0:
                consecutive_positive += 1
            else:
                break
        years_available = len(net_incomes.dropna())
        signals["graham_consecutive_positive_years"] = consecutive_positive
        signals["_graham_years_available"] = years_available
        if years_available >= 5 and consecutive_positive >= years_available:
            signals["graham_test"] = "PASS"
        elif consecutive_positive >= 4:
            signals["graham_test"] = "PASS"
        elif years_available >= 3 and consecutive_positive < years_available:
            signals["graham_test"] = "FAIL"
        else:
            signals["graham_test"] = "INSUFFICIENT_DATA"
    except Exception as exc:
        logger.warning(
            "graham_test_error",
            symbol=symbol,
            **summarize_exception(
                exc,
                operation="calculating Graham earnings test",
                provider="unknown",
            ),
        )
        signals["graham_consecutive_positive_years"] = None
        signals["graham_test"] = "ERROR"
    return signals


def calculate_derived_metrics(data: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Calculate simple derived metrics that depend on already-merged fields."""
    calculated: dict[str, Any] = {}
    try:
        if data.get("returnOnEquity") is None:
            roa = data.get("returnOnAssets")
            de = data.get("debtToEquity")
            if roa is not None and de is not None:
                calculated["returnOnEquity"] = roa * (1 + de)
                calculated["_returnOnEquity_source"] = "calculated_from_roa_de"

        if data.get("pegRatio") is None:
            pe = data.get("trailingPE")
            ttm_eg = data.get("earningsGrowth_TTM")
            if pe and ttm_eg and ttm_eg > 0.01:
                calculated_peg = pe / (ttm_eg * 100)
                if 0 < calculated_peg < 10:
                    calculated["pegRatio"] = calculated_peg
                    calculated["_pegRatio_source"] = "calculated_from_ttm_aligned"

        sector = _canonical_sector(data.get("sector"))
        pe = _safe_float(data.get("trailingPE"))
        sector_median_pe = SECTOR_MEDIAN_PE[sector] if sector else None
        if pe and sector_median_pe:
            calculated["sectorMedianPE"] = sector_median_pe
            calculated["peVsSector"] = round(pe / sector_median_pe, 2)
            calculated["sectorPeReferenceType"] = "STATIC_POLICY_REFERENCE"
            calculated["sectorPeReferenceAsOf"] = "N/A"
            calculated["_sectorMedianPE_source"] = "static_gics_sector_median"
            calculated["_peVsSector_source"] = "static_gics_sector_median"
            calculated["_sectorPeReferenceType_source"] = "static_gics_sector_median"
            calculated["_sectorPeReferenceAsOf_source"] = "static_gics_sector_median"

        if data.get("growth_trajectory") is None:
            mrq = data.get("revenueGrowth_MRQ")
            fy = data.get("revenueGrowth")
            if mrq is not None and fy is not None:
                delta = mrq - fy
                calculated["growth_trajectory"] = (
                    "ACCELERATING"
                    if delta > 0.10
                    else "DECELERATING"
                    if delta < -0.10
                    else "STABLE"
                )
                calculated["_growth_trajectory_source"] = "calculated_mrq_vs_fy"

        if (
            calculated.get("growth_trajectory") == "ACCELERATING"
            and data.get("earningsGrowth_TTM") is not None
            and data["earningsGrowth_TTM"] < -0.05
        ):
            calculated["growth_trajectory"] = "MIXED"
            calculated["_growth_trajectory_source"] = (
                f"{calculated.get('_growth_trajectory_source', 'calculated_mrq_vs_fy')}|eps_divergence"
            )

        if data.get("marketCap") is None:
            price = data.get("currentPrice") or data.get("regularMarketPrice")
            shares = data.get("sharesOutstanding")
            if price and shares:
                calculated["marketCap"] = price * shares
                calculated["_marketCap_source"] = "calculated_from_price_shares"
    except Exception:
        pass
    return calculated
