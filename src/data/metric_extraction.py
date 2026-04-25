"""Metric extraction and derived-financial-signal helpers for SmartMarketDataFetcher."""

from __future__ import annotations

import statistics
from datetime import timedelta
from typing import Any

import pandas as pd
import structlog

from src.config import config
from src.error_safety import summarize_exception

logger = structlog.get_logger(__name__)


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if pd.notna(numeric) else None


def extract_from_financial_statements(
    fetcher: Any, ticker, symbol: str
) -> dict[str, Any]:
    """Extract high-value metrics from yfinance statements and enrich with derived signals."""
    extracted: dict[str, Any] = {}
    try:
        financials = ticker.financials
        cashflow = ticker.cashflow
        balance_sheet = ticker.balance_sheet
        if financials.empty and cashflow.empty and balance_sheet.empty:
            return extracted

        fetcher.stats["sources"]["statements"] += 1

        if not financials.empty:
            if "Total Revenue" in financials.index and len(financials.columns) >= 2:
                try:
                    revenue_series = financials.loc["Total Revenue"]
                    current = float(revenue_series.iloc[0])
                    previous = float(revenue_series.iloc[1])
                    if previous and previous != 0:
                        growth = (current - previous) / previous
                        if -0.5 < growth < 5.0:
                            extracted["revenueGrowth"] = growth
                            extracted["_revenueGrowth_source"] = (
                                "calculated_from_statements"
                            )
                except Exception:
                    pass

            try:
                if (
                    "Gross Profit" in financials.index
                    and "Total Revenue" in financials.index
                ):
                    gross_profit = float(financials.loc["Gross Profit"].iloc[0])
                    revenue = float(financials.loc["Total Revenue"].iloc[0])
                    if revenue:
                        extracted["grossMargins"] = gross_profit / revenue
                        extracted["_grossMargins_source"] = "calculated_from_statements"
                if (
                    "Operating Income" in financials.index
                    and "Total Revenue" in financials.index
                ):
                    op_income = float(financials.loc["Operating Income"].iloc[0])
                    revenue = float(financials.loc["Total Revenue"].iloc[0])
                    if revenue:
                        extracted["operatingMargins"] = op_income / revenue
                        extracted["_operatingMargins_source"] = (
                            "calculated_from_statements"
                        )
                if (
                    "Net Income" in financials.index
                    and "Total Revenue" in financials.index
                ):
                    net_income = float(financials.loc["Net Income"].iloc[0])
                    revenue = float(financials.loc["Total Revenue"].iloc[0])
                    if revenue:
                        extracted["profitMargins"] = net_income / revenue
                        extracted["_profitMargins_source"] = (
                            "calculated_from_statements"
                        )
            except Exception:
                pass

        if not cashflow.empty:
            if "Operating Cash Flow" in cashflow.index:
                try:
                    ocf = float(cashflow.loc["Operating Cash Flow"].iloc[0])
                    extracted["operatingCashflow"] = ocf
                    extracted["_operatingCashflow_source"] = "extracted_from_statements"
                except Exception:
                    pass
            try:
                if (
                    "Operating Cash Flow" in cashflow.index
                    and "Capital Expenditure" in cashflow.index
                ):
                    ocf = float(cashflow.loc["Operating Cash Flow"].iloc[0])
                    capex = float(cashflow.loc["Capital Expenditure"].iloc[0])
                    extracted["freeCashflow"] = ocf + capex
                    extracted["_freeCashflow_source"] = "calculated_from_statements"
            except Exception:
                pass

        if not balance_sheet.empty:
            extracted["_statements_date"] = balance_sheet.columns[0].strftime(
                "%Y-%m-%d"
            )
            try:
                if (
                    "Current Assets" in balance_sheet.index
                    and "Current Liabilities" in balance_sheet.index
                ):
                    current_assets = float(balance_sheet.loc["Current Assets"].iloc[0])
                    current_liabilities = float(
                        balance_sheet.loc["Current Liabilities"].iloc[0]
                    )
                    if current_liabilities:
                        extracted["currentRatio"] = current_assets / current_liabilities
                        extracted["_currentRatio_source"] = "calculated_from_statements"
            except Exception:
                pass

            try:
                debt = None
                equity = None
                if "Total Debt" in balance_sheet.index:
                    debt = float(balance_sheet.loc["Total Debt"].iloc[0])
                elif "Long Term Debt" in balance_sheet.index:
                    long_term = float(balance_sheet.loc["Long Term Debt"].iloc[0])
                    short_term = (
                        float(balance_sheet.loc["Current Debt"].iloc[0])
                        if "Current Debt" in balance_sheet.index
                        else 0
                    )
                    debt = long_term + short_term

                if "Stockholders Equity" in balance_sheet.index:
                    equity = float(balance_sheet.loc["Stockholders Equity"].iloc[0])
                elif "Total Stockholder Equity" in balance_sheet.index:
                    equity = float(
                        balance_sheet.loc["Total Stockholder Equity"].iloc[0]
                    )

                if debt is not None and equity is not None and equity != 0:
                    extracted["debtToEquity"] = debt / equity
                    extracted["_debtToEquity_source"] = "calculated_from_statements"
            except Exception:
                pass

            try:
                if "Total Assets" in balance_sheet.index:
                    extracted["totalAssets"] = float(
                        balance_sheet.loc["Total Assets"].iloc[0]
                    )
                    extracted["_totalAssets_source"] = "calculated_from_statements"

                liquid = None
                for cash_row in [
                    "Cash And Cash Equivalents",
                    "Cash",
                    "Cash And Short Term Investments",
                ]:
                    if cash_row in balance_sheet.index:
                        liquid = float(balance_sheet.loc[cash_row].iloc[0])
                        break
                if liquid is not None:
                    sti = (
                        float(balance_sheet.loc["Short Term Investments"].iloc[0])
                        if "Short Term Investments" in balance_sheet.index
                        else 0.0
                    )
                    extracted["cashAndShortTermInvestments"] = liquid + sti
                    extracted["_cashAndShortTermInvestments_source"] = (
                        "calculated_from_statements"
                    )
            except Exception:
                pass
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
        return extracted

    if qt_inc is not None and not qt_inc.empty:
        latest_q_date = qt_inc.columns[0]
        extracted["latest_quarter_date"] = str(latest_q_date.date())
        extracted["_latest_quarter_date_source"] = "yfinance_quarterly"

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

    if qt_inc is not None and not qt_inc.empty and "Total Revenue" in qt_inc.index:
        rev_series = qt_inc.loc["Total Revenue"].dropna()
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

    if qt_inc is not None and not qt_inc.empty and "Net Income" in qt_inc.index:
        ni_series = qt_inc.loc["Net Income"].dropna()
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
        if len(ni_series) >= 4:
            ttm_ni = ni_series.iloc[0:4].sum(min_count=4)
            if pd.notna(ttm_ni):
                extracted["netIncome_TTM"] = float(ttm_ni)
                extracted["_netIncome_TTM_source"] = "calculated_from_quarterly"
            if len(ni_series) >= 8:
                ttm_ni_prior = ni_series.iloc[4:8].sum(min_count=4)
                if pd.notna(ttm_ni) and pd.notna(ttm_ni_prior) and ttm_ni_prior > 0:
                    ttm_ni_growth = (ttm_ni - ttm_ni_prior) / ttm_ni_prior
                    if -5.0 < ttm_ni_growth < 50.0:
                        extracted["earningsGrowth_TTM"] = float(ttm_ni_growth)
                        extracted["_earningsGrowth_TTM_source"] = (
                            "calculated_from_quarterly"
                        )

    if qt_cf is not None and not qt_cf.empty and "Operating Cash Flow" in qt_cf.index:
        ocf_series = qt_cf.loc["Operating Cash Flow"].dropna()
        if len(ocf_series) >= 4:
            ttm_ocf = ocf_series.iloc[0:4].sum(min_count=4)
            if pd.notna(ttm_ocf):
                extracted["operatingCashflow_TTM"] = float(ttm_ocf)
                extracted["_operatingCashflow_TTM_source"] = "calculated_from_quarterly"

    if (
        qt_cf is not None
        and not qt_cf.empty
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

    mrq_growth = extracted.get("revenueGrowth_MRQ")
    ttm_growth = extracted.get("revenueGrowth_TTM")
    if mrq_growth is not None and ttm_growth is not None:
        delta = mrq_growth - ttm_growth
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
        if capex is not None and d_and_a not in (None, 0):
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
            else "Total Stockholder Equity"
            if "Total Stockholder Equity" in balance_sheet.index
            else None
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
