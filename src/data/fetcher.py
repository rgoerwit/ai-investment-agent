"""
Smart Multi-Source Data Fetcher with Unified Parallel Approach
UPDATED: Integrated Alpha Vantage with circuit breaker for rate limit handling.
UPDATED: Integrated EOD Historical Data (EODHD) for international coverage.
FIXED: Smart Merge logic now correctly respects field-specific quality tags.

Strategy:
1. Launch ALL sources in parallel (yfinance, yahooquery, FMP, EODHD, Alpha Vantage)
2. Enhance yfinance with financial statement extraction
3. Smart merge with quality scoring (Statements > EODHD > Alpha Vantage/yfinance > FMP > Yahoo Info)
4. Mandatory Tavily gap-fill if coverage <70%
"""

import asyncio
import re
import statistics
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import structlog
import yfinance as yf

from src.config import config
from src.data.interfaces import FinancialFetcher
from src.ticker_utils import generate_strict_search_query

logger = structlog.get_logger(__name__)

# --- Optional Dependencies ---
try:
    from yahooquery import Ticker as YQTicker

    YAHOOQUERY_AVAILABLE = True
except ImportError:
    YAHOOQUERY_AVAILABLE = False
    logger.warning("yahooquery_not_available")

try:
    from src.data.fmp_fetcher import get_fmp_fetcher

    FMP_AVAILABLE = True
except ImportError:
    FMP_AVAILABLE = False
    logger.warning("fmp_not_available")

try:
    from src.data.eodhd_fetcher import get_eodhd_fetcher

    EODHD_AVAILABLE = True
except ImportError:
    EODHD_AVAILABLE = False
    logger.warning("eodhd_not_available")

try:
    from src.data.alpha_vantage_fetcher import get_av_fetcher

    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    logger.warning("alpha_vantage_not_available")

try:
    from tavily import TavilyClient

    TAVILY_LIB_AVAILABLE = True
except ImportError:
    TAVILY_LIB_AVAILABLE = False
    logger.warning("tavily_python_not_available")


# Constants
MIN_INFO_FIELDS = 3
ROE_PERCENTAGE_THRESHOLD = 1.0
DEBT_EQUITY_PERCENTAGE_THRESHOLD = 100.0
PRICE_TO_BOOK_CURRENCY_MISMATCH_THRESHOLD = 5.0
FX_CACHE_TTL_SECONDS = 3600
PER_SOURCE_TIMEOUT = 15

# Source quality rankings (higher = more reliable)
SOURCE_QUALITY = {
    "yfinance_statements": 10,  # Calculated directly from filings (Highest trust)
    "calculated_from_statements": 10,  # Tag used by extraction logic
    "eodhd": 9.5,  # Professional paid feed (High trust for Int'l)
    "yfinance": 9,  # Standard feed
    "yfinance_info": 9,  # Standard feed
    "alpha_vantage": 9,  # High-quality fundamentals (Int'l)
    "calculated": 8,  # Derived metrics
    "fmp": 7,  # Good backup
    "fmp_info": 7,
    "yahooquery": 6,  # Scraped backup
    "yahooquery_info": 6,
    "tavily_extraction": 4,  # Web NLP extraction
    "proxy": 2,  # Estimates
}

MergeResult = namedtuple("MergeResult", ["data", "gaps_filled"])


@dataclass
class DataQuality:
    """Track data quality and sources."""

    basics_ok: bool = False
    basics_missing: list[str] = None
    coverage_pct: float = 0.0
    sources_used: list[str] = None
    gaps_filled: int = 0
    suspicious_fields: list[str] = None

    def __post_init__(self):
        if self.basics_missing is None:
            self.basics_missing = []
        if self.sources_used is None:
            self.sources_used = []
        if self.suspicious_fields is None:
            self.suspicious_fields = []


class FinancialPatternExtractor:
    """Handles regex-based extraction of financial metrics from text."""

    def __init__(self):
        self.patterns = {
            "trailingPE": [
                re.compile(
                    r"(?:Trailing P/E|P/E \(TTM\)|P/E Ratio \(TTM\))(?:.*?)\s*[:=]?\s*(\d+[\.,]\d+)",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"(?:P/E|est|trading at|valuation).*?\s+(\d+[\.,]\d+)x",
                    re.IGNORECASE,
                ),
                re.compile(r"P/E\s+(?:of|is|around)\s+(\d+[\.,]\d+)", re.IGNORECASE),
                re.compile(
                    r"(?<!Forward\s)(?<!Fwd\s)(?:P/E|Price[- ]to[- ]Earnings)(?:.*?)(?:Ratio)?\s*[:=]?\s*(\d+[\.,]\d+)",
                    re.IGNORECASE,
                ),
                re.compile(r"\btrades?\s+at\s+(\d+[\.,]\d+)x", re.IGNORECASE),
                re.compile(r"\bvalued\s+at\s+(\d+[\.,]\d+)x", re.IGNORECASE),
                re.compile(
                    r"\btrading\s+at\s+(\d+(?:[\.,]\d+)?)\s+times", re.IGNORECASE
                ),
            ],
            "forwardPE": [
                re.compile(
                    r"(?:Forward P/E|Fwd P/E)(?:.*?)\s*[:=]?\s*(\d+[\.,]\d+)",
                    re.IGNORECASE,
                ),
                re.compile(r"(?:Forward P/E|Fwd P/E).*?(\d+[\.,]\d+)x", re.IGNORECASE),
                re.compile(r"est.*?P/E.*?(\d+[\.,]\d+)x", re.IGNORECASE),
            ],
            "priceToBook": [
                re.compile(
                    r"(?:P/B|Price[- ]to[- ]Book)(?:.*?)(?:Ratio)?\s*[:=]?\s*(\d+[\.,]\d+)",
                    re.IGNORECASE,
                ),
                re.compile(r"PB\s*Ratio\s*[:=]?\s*(\d+[\.,]\d+)", re.IGNORECASE),
                re.compile(r"Price\s*/\s*Book\s*[:=]?\s*(\d+[\.,]\d+)", re.IGNORECASE),
                re.compile(r"trading at\s+(\d+[\.,]\d+)x\s+book", re.IGNORECASE),
            ],
            "returnOnEquity": [
                re.compile(r"(?:ROE|Return on Equity).*?(\d+[\.,]\d+)%?", re.IGNORECASE)
            ],
            "marketCap": [
                re.compile(
                    r"(?:Market Cap|Valuation).*?(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d+)?)\s*([TBM])",
                    re.IGNORECASE,
                )
            ],
            "enterpriseToEbitda": [
                re.compile(
                    r"(?:EV/EBITDA|Enterprise Value/EBITDA)(?:.*?)\s*[:=]?\s*(\d+[\.,]\d+)",
                    re.IGNORECASE,
                ),
                re.compile(r"EV/EBITDA.*?(\d+[\.,]\d+)x", re.IGNORECASE),
            ],
            "numberOfAnalystOpinions": [
                re.compile(r"(\d+)\s+analyst(?:s)?\s+cover", re.IGNORECASE),
                re.compile(r"covered\s+by\s+(\d+)\s+analyst", re.IGNORECASE),
                re.compile(r"(\d+)\s+analyst(?:s)?\s+rating", re.IGNORECASE),
                re.compile(r"analyst\s+coverage:\s*(\d+)", re.IGNORECASE),
                re.compile(r"based\s+on\s+(\d+)\s+analyst", re.IGNORECASE),
                re.compile(r"consensus.*?(\d+)\s+analyst", re.IGNORECASE),
                re.compile(r"(\d+)\s+wall\s+street\s+analyst", re.IGNORECASE),
            ],
            "us_revenue_pct": [
                re.compile(r"US\s+revenue\s+.*?\s+(\d+(?:\.\d+)?)%", re.IGNORECASE),
                re.compile(
                    r"North\s+America\s+revenue\s+.*?\s+(\d+(?:\.\d+)?)%", re.IGNORECASE
                ),
                re.compile(
                    r"revenue\s+from\s+.*?Americas.*?\s+(\d+(?:\.\d+)?)%", re.IGNORECASE
                ),
            ],
        }

        self.multipliers = {"T": 1e12, "B": 1e9, "M": 1e6}

    def _normalize_number(self, val_str: str) -> float:
        try:
            val_str = val_str.strip()
            val_str = re.sub(r"[xX%]$", "", val_str).strip()

            # Robust International Format Handling
            if "," in val_str and "." in val_str:
                if val_str.rfind(",") < val_str.rfind("."):
                    clean_str = val_str.replace(",", "")  # US: 1,234.56
                else:
                    clean_str = val_str.replace(".", "").replace(
                        ",", "."
                    )  # EU: 1.234,56
            elif "," in val_str:
                # Ambiguous: 1,234 vs 12,34. Assume comma as decimal if not xxx,xxx format
                if re.match(r"^\d{1,3},\d{3}$", val_str):
                    clean_str = val_str.replace(",", "")
                else:
                    clean_str = val_str.replace(",", ".")
            else:
                clean_str = val_str

            return float(clean_str)
        except ValueError:
            return 0.0

    def extract_from_text(
        self, content: str, skip_fields: set = None
    ) -> dict[str, Any]:
        skip_fields = skip_fields or set()
        extracted = {}

        for field, pattern_list in self.patterns.items():
            if field != "forwardPE" and field in skip_fields:
                continue

            for pattern in pattern_list:
                match = pattern.search(content)
                if match:
                    try:
                        val_str = match.group(1)
                        val = self._normalize_number(val_str)

                        if field == "returnOnEquity" and val > ROE_PERCENTAGE_THRESHOLD:
                            val = val / 100.0
                        elif field == "marketCap":
                            suffix = match.group(2).upper()
                            multiplier = self.multipliers.get(suffix, 1)
                            val = val * multiplier
                        elif field == "numberOfAnalystOpinions":
                            val = int(val)
                            if val < 0 or val > 200:
                                continue  # Sanity check

                        extracted[field] = val
                        extracted[f"_{field}_source"] = "web_search_extraction"
                        break
                    except (ValueError, IndexError):
                        continue

        # Proxy fill
        if (
            "trailingPE" not in skip_fields
            and "trailingPE" not in extracted
            and "forwardPE" in extracted
        ):
            extracted["trailingPE"] = extracted["forwardPE"]
            extracted["_trailingPE_source"] = "proxy_from_forward_pe"

        return extracted


class SmartMarketDataFetcher(FinancialFetcher):
    """Intelligent multi-source fetcher with unified parallel approach."""

    REQUIRED_BASICS = ["symbol", "currentPrice", "currency"]

    IMPORTANT_FIELDS = [
        "marketCap",
        "trailingPE",
        "priceToBook",
        "returnOnEquity",
        "revenueGrowth",
        "profitMargins",
        "operatingMargins",
        "grossMargins",
        "debtToEquity",
        "currentRatio",
        "freeCashflow",
        "operatingCashflow",
        "numberOfAnalystOpinions",
        "pegRatio",
        "forwardPE",
    ]

    def is_available(self) -> bool:
        """
        SmartMarketDataFetcher is always available as it aggregates from multiple sources.
        yfinance (the primary source) requires no API key.
        """
        return True

    def __init__(self):
        self.fx_cache = {}
        self.fx_cache_expiry_time = {}

        self.fmp_fetcher = get_fmp_fetcher() if FMP_AVAILABLE else None
        self.eodhd_fetcher = get_eodhd_fetcher() if EODHD_AVAILABLE else None
        self.av_fetcher = get_av_fetcher() if ALPHA_VANTAGE_AVAILABLE else None
        self.pattern_extractor = FinancialPatternExtractor()

        api_key = config.get_tavily_api_key()
        self.tavily_client = (
            TavilyClient(api_key=api_key) if TAVILY_LIB_AVAILABLE and api_key else None
        )

        self.stats = {
            "fetches": 0,
            "basics_ok": 0,
            "basics_failed": 0,
            "avg_coverage": 0.0,
            "sources": {
                "yfinance": 0,
                "statements": 0,
                "yahooquery": 0,
                "fmp": 0,
                "eodhd": 0,
                "alpha_vantage": 0,
                "web_search": 0,
                "calculated": 0,
            },
            "gaps_filled": 0,
        }

    def get_currency_rate(self, from_curr: str, to_curr: str) -> float:
        """Get FX rate with caching."""
        if not from_curr or not to_curr or from_curr == to_curr:
            return 1.0

        from_curr = from_curr.upper()
        to_curr = to_curr.upper()
        cache_key = f"{from_curr}_{to_curr}"

        now = datetime.now()
        expiry_time = self.fx_cache_expiry_time.get(cache_key)

        if expiry_time and now < expiry_time:
            return self.fx_cache.get(cache_key, 1.0)

        try:
            pair_symbol = f"{from_curr}{to_curr}=X"
            ticker = yf.Ticker(pair_symbol)
            hist = ticker.history(period="1d")

            if not hist.empty:
                rate = float(hist["Close"].iloc[-1])
                self.fx_cache[cache_key] = rate
                self.fx_cache_expiry_time[cache_key] = now + timedelta(
                    seconds=FX_CACHE_TTL_SECONDS
                )
                return rate
        except Exception as e:
            logger.debug(
                "fx_rate_fetch_failed", pair=f"{from_curr}/{to_curr}", error=str(e)
            )

        return 1.0

    def _extract_from_financial_statements(
        self, ticker: yf.Ticker, symbol: str
    ) -> dict[str, Any]:
        """Extract metrics from yfinance financial statements."""
        extracted = {}

        try:
            financials = ticker.financials
            cashflow = ticker.cashflow
            balance_sheet = ticker.balance_sheet

            if financials.empty and cashflow.empty and balance_sheet.empty:
                return extracted

            self.stats["sources"]["statements"] += 1

            # INCOME STATEMENT
            if not financials.empty:
                # Revenue Growth
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

                # Margins
                try:
                    if (
                        "Gross Profit" in financials.index
                        and "Total Revenue" in financials.index
                    ):
                        gross_profit = float(financials.loc["Gross Profit"].iloc[0])
                        revenue = float(financials.loc["Total Revenue"].iloc[0])
                        if revenue:
                            extracted["grossMargins"] = gross_profit / revenue
                            extracted["_grossMargins_source"] = (
                                "calculated_from_statements"
                            )

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

            # CASH FLOW STATEMENT
            if not cashflow.empty:
                if "Operating Cash Flow" in cashflow.index:
                    try:
                        ocf = float(cashflow.loc["Operating Cash Flow"].iloc[0])
                        extracted["operatingCashflow"] = ocf
                        extracted["_operatingCashflow_source"] = (
                            "extracted_from_statements"
                        )
                    except Exception:
                        pass

                try:
                    if (
                        "Operating Cash Flow" in cashflow.index
                        and "Capital Expenditure" in cashflow.index
                    ):
                        ocf = float(cashflow.loc["Operating Cash Flow"].iloc[0])
                        capex = float(cashflow.loc["Capital Expenditure"].iloc[0])
                        fcf = ocf + capex  # Capex is usually negative
                        extracted["freeCashflow"] = fcf
                        extracted["_freeCashflow_source"] = "calculated_from_statements"
                except Exception:
                    pass

            # BALANCE SHEET
            if not balance_sheet.empty:
                try:
                    if (
                        "Current Assets" in balance_sheet.index
                        and "Current Liabilities" in balance_sheet.index
                    ):
                        current_assets = float(
                            balance_sheet.loc["Current Assets"].iloc[0]
                        )
                        current_liabilities = float(
                            balance_sheet.loc["Current Liabilities"].iloc[0]
                        )
                        if current_liabilities:
                            extracted["currentRatio"] = (
                                current_assets / current_liabilities
                            )
                            extracted["_currentRatio_source"] = (
                                "calculated_from_statements"
                            )
                except Exception:
                    pass

                try:
                    debt = None
                    equity = None

                    if "Total Debt" in balance_sheet.index:
                        debt = float(balance_sheet.loc["Total Debt"].iloc[0])
                    elif "Long Term Debt" in balance_sheet.index:
                        long_term = float(balance_sheet.loc["Long Term Debt"].iloc[0])
                        short_term = 0
                        if "Current Debt" in balance_sheet.index:
                            short_term = float(
                                balance_sheet.loc["Current Debt"].iloc[0]
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

        except Exception as e:
            logger.debug("statement_extraction_failed", symbol=symbol, error=str(e))

        # --- MOAT SIGNALS ---
        # Calculate multi-year moat indicators from statements
        moat_signals = self._calculate_moat_signals(financials, cashflow, symbol)
        for key, value in moat_signals.items():
            extracted[key] = value
            extracted[f"_{key}_source"] = "calculated_from_statements"

        # --- CAPITAL EFFICIENCY SIGNALS ---
        # Calculate ROIC and leverage quality from statements
        capital_signals = self._calculate_capital_efficiency_signals(
            financials, balance_sheet, extracted, symbol
        )
        for key, value in capital_signals.items():
            extracted[key] = value

        # --- HISTORICAL RETURN TRENDS ---
        # Calculate 5Y average ROA/ROE for sustainability assessment
        return_trends = self._calculate_return_trends(financials, balance_sheet, symbol)
        for key, value in return_trends.items():
            extracted[key] = value

        return extracted

    def _calculate_moat_signals(
        self, financials: "pd.DataFrame", cashflow: "pd.DataFrame", symbol: str
    ) -> dict[str, Any]:
        """
        Calculate economic moat signal metrics from multi-year financial statements.

        Uses Coefficient of Variation (CV) for stability measurement - statistically
        superior to raw standard deviation as it's scale-independent.

        Metrics calculated:
        1. Gross Margin CV (5-year): Low CV (<8%) indicates pricing power
        2. CFO/Net Income Ratio (3-year avg): High ratio (>90%) indicates quality earnings

        Args:
            financials: yfinance financials DataFrame (columns are years, newest first)
            cashflow: yfinance cashflow DataFrame (columns are years, newest first)
            symbol: Ticker symbol for logging

        Returns:
            Dict with moat signal metrics and human-readable assessments.
            Empty dict if insufficient data.

        Note:
            Uses sample standard deviation (N-1) via statistics.stdev() for proper
            small-sample statistics (3-5 data points).
        """
        signals: dict[str, Any] = {}

        # Require minimum 3 years of data for meaningful statistics
        if financials.empty or len(financials.columns) < 3:
            logger.debug("moat_signals_insufficient_data", symbol=symbol, years=0)
            return signals

        # --- 1. GROSS MARGIN STABILITY (5-year CV) ---
        try:
            if (
                "Gross Profit" in financials.index
                and "Total Revenue" in financials.index
            ):
                margins = []
                years_available = min(5, len(financials.columns))

                for i in range(years_available):
                    try:
                        gross_profit = financials.loc["Gross Profit"].iloc[i]
                        revenue = financials.loc["Total Revenue"].iloc[i]

                        # Skip if either is None, NaN, or zero
                        if (
                            pd.notna(gross_profit)
                            and pd.notna(revenue)
                            and revenue != 0
                        ):
                            margin = float(gross_profit) / float(revenue)
                            # Sanity check: margin should be between -0.5 and 1.0
                            if -0.5 < margin < 1.0:
                                margins.append(margin)
                    except (ValueError, TypeError, KeyError):
                        continue

                # Require minimum 3 valid data points for meaningful statistics
                if len(margins) >= 3:
                    mean_margin = statistics.mean(margins)

                    # CV = StdDev / Mean (only valid if mean > 0)
                    # Using sample stdev (N-1) for small-sample correctness
                    if mean_margin > 0.05:  # Require minimum 5% margin
                        std_margin = statistics.stdev(margins)  # N-1 denominator
                        cv = std_margin / mean_margin

                        signals["moat_grossMarginCV"] = round(cv, 4)
                        signals["moat_grossMarginAvg"] = round(mean_margin, 4)
                        signals["moat_grossMarginYears"] = len(margins)

                        # Human-readable signal (used for threshold logic downstream)
                        if cv < 0.08:
                            signals["moat_marginStability"] = "HIGH"
                        elif cv < 0.15:
                            signals["moat_marginStability"] = "MEDIUM"
                        else:
                            signals["moat_marginStability"] = "LOW"

                        logger.debug(
                            "moat_margin_calculated",
                            symbol=symbol,
                            cv=cv,
                            avg=mean_margin,
                            years=len(margins),
                            signal=signals.get("moat_marginStability"),
                        )
        except Exception as e:
            logger.debug("moat_margin_calc_failed", symbol=symbol, error=str(e))

        # --- 2. CASH CONVERSION QUALITY (3-year CFO/NI ratio) ---
        try:
            if (
                not cashflow.empty
                and "Operating Cash Flow" in cashflow.index
                and "Net Income" in financials.index
            ):
                ratios = []
                years_available = min(3, len(financials.columns), len(cashflow.columns))

                for i in range(years_available):
                    try:
                        ocf = cashflow.loc["Operating Cash Flow"].iloc[i]
                        ni = financials.loc["Net Income"].iloc[i]

                        # Only calculate for profitable years (NI > 0)
                        if pd.notna(ocf) and pd.notna(ni) and float(ni) > 0:
                            ratio = float(ocf) / float(ni)
                            # Sanity check: ratio typically 0.3 to 2.5
                            # (can exceed 1.0 when D&A high relative to capex)
                            if 0.1 < ratio < 3.0:
                                ratios.append(ratio)
                    except (ValueError, TypeError, KeyError):
                        continue

                if len(ratios) >= 2:
                    avg_ratio = statistics.mean(ratios)
                    signals["moat_cfoToNiAvg"] = round(avg_ratio, 4)
                    signals["moat_cfoToNiYears"] = len(ratios)

                    # Human-readable signal (used for threshold logic downstream)
                    # Note: ratio > 1.0 is valid (strong cash generation)
                    if avg_ratio > 0.90:
                        signals["moat_cashConversion"] = "STRONG"
                    elif avg_ratio > 0.70:
                        signals["moat_cashConversion"] = "ADEQUATE"
                    else:
                        signals["moat_cashConversion"] = "WEAK"

                    logger.debug(
                        "moat_cash_conversion_calculated",
                        symbol=symbol,
                        avg_ratio=avg_ratio,
                        years=len(ratios),
                        signal=signals.get("moat_cashConversion"),
                    )
        except Exception as e:
            logger.debug("moat_cash_conversion_failed", symbol=symbol, error=str(e))

        return signals

    def _calculate_capital_efficiency_signals(
        self,
        income_stmt: "pd.DataFrame",
        balance_sheet: "pd.DataFrame",
        info: dict[str, Any],
        symbol: str,
    ) -> dict[str, Any]:
        """
        Calculate capital efficiency metrics: ROIC and leverage quality.

        Detects value destruction (ROIC < 0 with positive ROE) and financial
        engineering (ROE >> ROIC). Uses hurdle rate as WACC proxy to avoid
        precision theater of CAPM calculations.

        Args:
            income_stmt: yfinance income statement DataFrame
            balance_sheet: yfinance balance sheet DataFrame
            info: Extracted data dict (contains ROA, ROE from info)
            symbol: Ticker symbol for logging

        Returns:
            Dict with capital efficiency signals. Empty dict if insufficient data.
        """
        from src.config import config

        signals: dict[str, Any] = {}

        try:
            # --- Extract components ---
            ebit: float | None = None
            tax_rate: float | None = None
            invested_capital: float | None = None

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

            # ROA and ROE from info dict
            # roa = info.get("returnOnAssets")
            roe = info.get("returnOnEquity")

            # --- Calculate ROIC ---
            roic: float | None = None
            if (
                ebit is not None
                and invested_capital is not None
                and invested_capital > 0
            ):
                # Default tax rate to 21% if not available
                effective_tax = tax_rate if tax_rate is not None else 0.21
                # Clamp tax rate to reasonable range
                effective_tax = max(0.0, min(0.5, effective_tax))
                nopat = ebit * (1 - effective_tax)
                roic = nopat / invested_capital

                signals["capital_roic"] = round(roic, 4)
                signals["capital_roic_source"] = "calculated"

                # --- ROIC Quality Classification ---
                hurdle = config.roic_hurdle_rate
                strong = config.roic_strong_threshold

                if roic < 0:
                    signals["capital_roicQuality"] = "DESTRUCTIVE"
                elif roic < hurdle:
                    signals["capital_roicQuality"] = "WEAK"
                elif roic < strong:
                    signals["capital_roicQuality"] = "ADEQUATE"
                else:
                    signals["capital_roicQuality"] = "STRONG"

                # Spread vs hurdle rate (proxy for ROIC - WACC)
                signals["capital_hurdleSpread"] = round(roic - hurdle, 4)

            # --- Leverage Quality (ROE/ROIC relationship) ---
            if roic is not None and roe is not None:
                # Handle edge cases
                if roic <= 0 and roe > 0:
                    # Value destruction: negative operating returns, positive equity returns
                    signals["capital_leverageQuality"] = "VALUE_DESTRUCTION"
                elif roic > 0:
                    ratio = roe / roic
                    signals["capital_roeRoicRatio"] = round(ratio, 2)

                    suspect = config.leverage_suspect_ratio
                    engineered = config.leverage_engineered_ratio

                    if ratio > engineered:
                        signals["capital_leverageQuality"] = "ENGINEERED"
                    elif ratio > suspect:
                        signals["capital_leverageQuality"] = "SUSPECT"
                    elif ratio < 1.0:
                        # ROIC > ROE: under-leveraged or conservative
                        signals["capital_leverageQuality"] = "CONSERVATIVE"
                    else:
                        signals["capital_leverageQuality"] = "GENUINE"

            if signals:
                logger.debug(
                    "capital_efficiency_calculated",
                    symbol=symbol,
                    roic=signals.get("capital_roic"),
                    roic_quality=signals.get("capital_roicQuality"),
                    leverage_quality=signals.get("capital_leverageQuality"),
                )

        except Exception as e:
            logger.debug(
                "capital_efficiency_calculation_failed",
                symbol=symbol,
                error=str(e),
            )

        return signals

    @staticmethod
    def _compute_trend_regression(values: list[float], mean_val: float) -> str:
        """
        Determine trend using linear regression slope and coefficient of variation.

        Args:
            values: Time series (oldest first, newest last)
            mean_val: Pre-computed mean for CV calculation

        Returns:
            UNSTABLE: CV > 0.40 (high variance masks any trend)
            IMPROVING: Slope > 0.5% of mean per year
            DECLINING: Slope < -0.5% of mean per year
            STABLE: Slope within ±0.5% of mean per year
        """
        n = len(values)
        if n < 3 or mean_val == 0:
            return "N/A"

        # Coefficient of variation (CV) - measures volatility
        try:
            stdev = statistics.stdev(values)
            cv = abs(stdev / mean_val) if mean_val != 0 else 0
        except statistics.StatisticsError:
            cv = 0

        # High variance = UNSTABLE (cyclical or erratic)
        if cv > 0.40:
            return "UNSTABLE"

        # Linear regression: slope = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
        x_mean = (n - 1) / 2.0
        numerator = sum((i - x_mean) * (v - mean_val) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "STABLE"

        slope = numerator / denominator

        # Normalize slope as % of mean per year
        slope_pct = (slope / abs(mean_val)) if mean_val != 0 else 0

        # Thresholds: ±0.5% annual change relative to mean
        if slope_pct > 0.005:
            return "IMPROVING"
        elif slope_pct < -0.005:
            return "DECLINING"
        else:
            return "STABLE"

    def _calculate_return_trends(
        self,
        financials: "pd.DataFrame",
        balance_sheet: "pd.DataFrame",
        symbol: str,
    ) -> dict[str, Any]:
        """
        Calculate 5-year historical average ROA/ROE and trend direction.

        Used to detect cyclical peaks vs structural quality improvement.
        Requires minimum 3 years of data; uses up to 5 years if available.

        Args:
            financials: yfinance financials DataFrame (columns are years, newest first)
            balance_sheet: yfinance balance sheet DataFrame
            symbol: Ticker symbol for logging

        Returns:
            Dict with roa_5y_avg, roe_5y_avg, and profitability_trend.
            Empty dict if insufficient data.
        """
        signals: dict[str, Any] = {}

        if financials.empty or balance_sheet.empty:
            return signals

        years_available = min(len(financials.columns), len(balance_sheet.columns), 5)
        if years_available < 3:
            logger.debug(
                "return_trends_insufficient_data", symbol=symbol, years=years_available
            )
            return signals

        # --- ROA: Net Income / Total Assets ---
        try:
            if (
                "Net Income" in financials.index
                and "Total Assets" in balance_sheet.index
            ):
                roas = []
                for i in range(years_available):
                    try:
                        ni = financials.loc["Net Income"].iloc[i]
                        assets = balance_sheet.loc["Total Assets"].iloc[i]
                        if pd.notna(ni) and pd.notna(assets) and float(assets) > 0:
                            roa = float(ni) / float(assets)
                            # Sanity: exclude extreme outliers
                            if -0.50 < roa < 0.50:
                                roas.append(roa)
                    except (ValueError, TypeError, IndexError):
                        continue

                if len(roas) >= 3:
                    avg_roa = statistics.mean(roas)
                    signals["roa_5y_avg"] = round(avg_roa * 100, 2)
                    signals["_roa_5y_years"] = len(roas)

                    # Trend via regression + variance analysis
                    # roas[0] = newest, roas[-1] = oldest; invert for time series
                    signals["profitability_trend"] = self._compute_trend_regression(
                        list(reversed(roas)), avg_roa
                    )

                    logger.debug(
                        "roa_trend_calculated",
                        symbol=symbol,
                        roa_5y_avg=signals.get("roa_5y_avg"),
                        trend=signals.get("profitability_trend"),
                        years=len(roas),
                    )
        except Exception as e:
            logger.debug("roa_trend_calc_failed", symbol=symbol, error=str(e))

        # --- ROE: Net Income / Stockholders Equity ---
        try:
            equity_key = (
                "Stockholders Equity"
                if "Stockholders Equity" in balance_sheet.index
                else "Total Stockholder Equity"
                if "Total Stockholder Equity" in balance_sheet.index
                else None
            )

            if "Net Income" in financials.index and equity_key:
                roes = []
                for i in range(years_available):
                    try:
                        ni = financials.loc["Net Income"].iloc[i]
                        equity = balance_sheet.loc[equity_key].iloc[i]
                        # Require positive equity (negative = insolvent, skip)
                        if pd.notna(ni) and pd.notna(equity) and float(equity) > 0:
                            roe = float(ni) / float(equity)
                            if -1.0 < roe < 1.0:
                                roes.append(roe)
                    except (ValueError, TypeError, IndexError):
                        continue

                if len(roes) >= 3:
                    signals["roe_5y_avg"] = round(statistics.mean(roes) * 100, 2)
                    signals["_roe_5y_years"] = len(roes)

                    logger.debug(
                        "roe_trend_calculated",
                        symbol=symbol,
                        roe_5y_avg=signals.get("roe_5y_avg"),
                        years=len(roes),
                    )
        except Exception as e:
            logger.debug("roe_trend_calc_failed", symbol=symbol, error=str(e))

        return signals

    async def _fetch_yfinance_enhanced(self, symbol: str) -> dict | None:
        """Fetch yfinance data including statement calculation."""
        try:
            ticker = yf.Ticker(symbol)
            info = {}
            try:
                info = ticker.info
            except Exception:
                info = {}

            has_price = False
            price_fields = ["currentPrice", "regularMarketPrice", "previousClose"]

            if info:
                for field in price_fields:
                    if field in info and info[field] is not None:
                        has_price = True
                        break

            if not has_price and hasattr(ticker, "fast_info"):
                try:
                    fast_price = ticker.fast_info.get("lastPrice")
                    if fast_price:
                        info["currentPrice"] = fast_price
                        has_price = True
                except (AttributeError, KeyError):
                    pass

            if not has_price:
                logger.warning("yfinance_no_price", symbol=symbol)
                info = info or {}

            # ALWAYS extract from statements (for gap-filling and divergence detection)
            statement_data = self._extract_from_financial_statements(ticker, symbol)

            # Flag significant TTM vs statement divergence for data quality awareness
            fcf_ttm = info.get("freeCashflow")
            fcf_stmt = statement_data.get("freeCashflow")
            if fcf_ttm and fcf_stmt and fcf_ttm != 0 and fcf_stmt != 0:
                ratio = abs(fcf_ttm / fcf_stmt)
                if ratio > 1.5 or ratio < 0.67:
                    info["fcf_data_note"] = (
                        f"FCF DATA QUALITY UNCERTAIN: TTM ({fcf_ttm/1e9:.2f}B) vs "
                        f"statement ({fcf_stmt/1e9:.2f}B) = {ratio:.1f}x divergence"
                    )

            for key, value in statement_data.items():
                if key.startswith("_"):
                    # Always copy source tags
                    info[key] = value
                elif key not in info or info.get(key) is None:
                    # Only use statement data to fill gaps (TTM is more current)
                    if value is not None:
                        info[key] = value

            if not info or (not has_price and len(info) < 5):
                return None

            if "symbol" not in info:
                info["symbol"] = symbol

            self.stats["sources"]["yfinance"] += 1
            return info

        except Exception as e:
            logger.error("yfinance_enhanced_failed", symbol=symbol, error=str(e))
            return None

    def _fetch_yahooquery_fallback(self, symbol: str) -> dict | None:
        """Fallback: yahooquery."""
        if not YAHOOQUERY_AVAILABLE:
            return None

        try:
            yq = YQTicker(symbol)
            combined = {}
            modules = [
                yq.summary_profile,
                yq.summary_detail,
                yq.key_stats,
                yq.financial_data,
                yq.price,
            ]

            for module in modules:
                if isinstance(module, dict) and symbol in module:
                    data = module[symbol]
                    if isinstance(data, dict):
                        combined.update(data)

            if not combined or len(combined) < MIN_INFO_FIELDS:
                return None

            if "currentPrice" not in combined and "regularMarketPrice" in combined:
                combined["currentPrice"] = combined["regularMarketPrice"]

            self.stats["sources"]["yahooquery"] += 1
            return combined
        except Exception:
            return None

    async def _fetch_fmp_fallback(self, symbol: str) -> dict | None:
        """Fallback: FMP."""
        if (
            not FMP_AVAILABLE
            or not self.fmp_fetcher
            or not self.fmp_fetcher.is_available()
        ):
            return None

        try:
            fmp_data = await self.fmp_fetcher.get_financial_metrics(symbol)

            # Check if we got valid data (keys other than _source)
            if fmp_data and any(
                v is not None for k, v in fmp_data.items() if k != "_source"
            ):
                self.stats["sources"]["fmp"] += 1
                return fmp_data

        except Exception:
            return None

        return None

    async def _fetch_eodhd_fallback(self, symbol: str) -> dict | None:
        """
        Fallback: EOD Historical Data.
        High quality source for fundamentals.
        Gracefully handles API Limits/Errors by returning None.
        """
        if not EODHD_AVAILABLE or not self.eodhd_fetcher:
            return None

        try:
            # Check circuit breaker before attempting
            if not self.eodhd_fetcher.is_available():
                return None

            data = await self.eodhd_fetcher.get_financial_metrics(symbol)

            # If successful and contains data
            if data and any(v is not None for k, v in data.items() if k != "_source"):
                self.stats["sources"]["eodhd"] += 1
                return data

            # If data is None, fetcher might have hit rate limit (logged inside fetcher)
            return None

        except Exception as e:
            logger.warning("eodhd_fetch_error", symbol=symbol, error=str(e))
            return None

    async def _fetch_av_fallback(self, symbol: str) -> dict | None:
        """
        Fallback: Alpha Vantage.
        High-quality fundamentals with circuit breaker for rate limit handling.
        Free tier: 25 requests/day, 5 requests/minute.
        """
        if not ALPHA_VANTAGE_AVAILABLE or not self.av_fetcher:
            return None

        try:
            # Check circuit breaker before attempting
            if not self.av_fetcher.is_available():
                return None

            data = await self.av_fetcher.get_financial_metrics(symbol)

            # If successful and contains data
            if data and any(
                v is not None for k, v in data.items() if not k.startswith("_")
            ):
                self.stats["sources"]["alpha_vantage"] += 1
                return data

            # If data is None, fetcher might have hit rate limit (logged inside fetcher)
            return None

        except Exception as e:
            logger.warning("alpha_vantage_fetch_error", symbol=symbol, error=str(e))
            return None

    async def _fetch_all_sources_parallel(self, symbol: str) -> dict[str, dict | None]:
        """PHASE 1: Launch all data sources in parallel."""
        logger.info("launching_parallel_sources", symbol=symbol)

        tasks = {
            "yfinance": self._fetch_yfinance_enhanced(symbol),
            "yahooquery": asyncio.to_thread(self._fetch_yahooquery_fallback, symbol),
            "fmp": self._fetch_fmp_fallback(symbol),
            "eodhd": self._fetch_eodhd_fallback(symbol),
            "alpha_vantage": self._fetch_av_fallback(symbol),
        }

        results = {}
        for source_name, coro in tasks.items():
            try:
                # Wait for each task with timeout
                result = await asyncio.wait_for(coro, timeout=PER_SOURCE_TIMEOUT)
                results[source_name] = result
                if result:
                    logger.info(
                        f"{source_name}_success", symbol=symbol, fields=len(result)
                    )
                else:
                    logger.warning(f"{source_name}_returned_none", symbol=symbol)
            except asyncio.TimeoutError:
                logger.warning(f"{source_name}_timeout", symbol=symbol)
                results[source_name] = None
            except Exception as e:
                logger.warning(f"{source_name}_error", symbol=symbol, error=str(e))
                results[source_name] = None

        return results

    def _smart_merge_with_quality(
        self, source_results: dict[str, dict | None], symbol: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        PHASE 3: Intelligent merge with quality scoring.
        Updated to include EODHD and Alpha Vantage in the priority logic and field-specific override checks.
        """
        merged = {}
        field_sources = {}
        field_quality = {}
        sources_used = set()
        gaps_filled = 0

        # Order of processing (lowest priority to highest priority if quality scores match)
        # Note: Actual precedence is determined by SOURCE_QUALITY dict
        source_order = ["yahooquery", "fmp", "alpha_vantage", "eodhd", "yfinance"]

        for source_name in source_order:
            source_data = source_results.get(source_name)
            if not source_data:
                continue

            sources_used.add(source_name)

            for key, value in source_data.items():
                if value is None:
                    continue

                if key.startswith("_") and key.endswith("_source"):
                    continue

                # 1. Determine Base Quality for this source
                # Check for explicit keys in SOURCE_QUALITY to handle fallbacks correctly
                if source_name in SOURCE_QUALITY:
                    quality = SOURCE_QUALITY[source_name]
                else:
                    quality = SOURCE_QUALITY.get(f"{source_name}_info", 5)

                # 2. Check for Field-Specific Override (e.g. calculated_from_statements)
                source_tag_key = f"_{key}_source"
                if source_tag_key in source_data:
                    tag = source_data[source_tag_key]
                    if tag in SOURCE_QUALITY:
                        quality = SOURCE_QUALITY[tag]

                should_use = False

                if key not in merged:
                    should_use = True
                elif merged[key] is None and value is not None:
                    should_use = True
                    gaps_filled += 1
                elif key in field_quality:
                    if quality > field_quality[key]:
                        should_use = True
                        logger.debug(
                            "replacing_with_higher_quality",
                            symbol=symbol,
                            field=key,
                            old_source=field_sources.get(key),
                            new_source=source_name,
                        )

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
        }

        logger.info(
            "smart_merge_complete",
            symbol=symbol,
            total_fields=len(merged),
            sources=list(sources_used),
            gaps_filled=gaps_filled,
        )

        return merged, metadata

    def _calculate_coverage(self, data: dict) -> float:
        """Calculate percentage of IMPORTANT_FIELDS present."""
        if not data:
            return 0.0
        present = sum(
            1 for field in self.IMPORTANT_FIELDS if data.get(field) is not None
        )
        return present / len(self.IMPORTANT_FIELDS) if self.IMPORTANT_FIELDS else 0.0

    def _identify_critical_gaps(self, data: dict) -> list[str]:
        """Identify which critical fields are missing."""
        critical = [
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
        ]
        return [f for f in critical if f not in data or data[f] is None]

    async def _fetch_tavily_gaps(
        self, symbol: str, missing_fields: list[str]
    ) -> dict[str, Any]:
        """PHASE 5: Tavily gap-filling."""
        DANGEROUS_FIELDS = [
            "trailingPE",
            "forwardPE",
            "pegRatio",
            "currentPrice",
            "marketCap",
        ]
        safe_missing_fields = [f for f in missing_fields if f not in DANGEROUS_FIELDS]

        if "us_revenue_pct" in missing_fields or "geographic_revenue" in missing_fields:
            safe_missing_fields.append("us_revenue_pct")

        if not self.tavily_client or not safe_missing_fields:
            return {}

        try:
            import yfinance as yf

            ticker_obj = yf.Ticker(symbol)
            company_name = (
                ticker_obj.info.get("longName")
                or ticker_obj.info.get("shortName")
                or symbol
            )
        except Exception:
            company_name = symbol

        fields_to_search = safe_missing_fields[:5]
        search_results = {}

        for field in fields_to_search:
            # Map internal field names to search terms
            field_terms = {
                "trailingPE": "trailing P/E ratio price earnings",
                "forwardPE": "forward P/E ratio estimate",
                "priceToBook": "price to book ratio P/B",
                "returnOnEquity": "ROE return on equity",
                "debtToEquity": "debt to equity ratio leverage",
                "numberOfAnalystOpinions": "analyst coverage count",
                "revenueGrowth": "revenue growth year over year",
            }

            if field == "us_revenue_pct":
                query = f'"{company_name}" annual report revenue by geography North America United States'
            else:
                term = field_terms.get(field, field)
                query = generate_strict_search_query(symbol, company_name, term)

            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self.tavily_client.search, query, max_results=3),
                    timeout=5,
                )
                if result and "results" in result:
                    combined = "\n".join(
                        [i.get("content", "") for i in result["results"]]
                    )
                    search_results[field] = combined
            except (TimeoutError, asyncio.TimeoutError, Exception):
                pass

        if not search_results:
            return {}

        all_text = "\n\n".join(search_results.values())
        return self.pattern_extractor.extract_from_text(all_text, skip_fields=set())

    def _merge_gap_fill_data(
        self,
        merged: dict[str, Any],
        gap_fill_data: dict[str, Any],
        merge_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge Tavily data."""
        tavily_quality = SOURCE_QUALITY["tavily_extraction"]
        added = 0
        for key, value in gap_fill_data.items():
            if value is None:
                continue
            should_use = False

            if key not in merged:
                should_use = True
            elif merged[key] is None:
                should_use = True
            elif (
                key in merge_metadata["field_quality"]
                and tavily_quality > merge_metadata["field_quality"][key]
            ):
                should_use = True

            if should_use:
                merged[key] = value
                merge_metadata["field_sources"][key] = "tavily"
                merge_metadata["field_quality"][key] = tavily_quality
                added += 1

        merge_metadata["gaps_filled"] += added
        return merged

    def _calculate_derived_metrics(self, data: dict, symbol: str) -> dict:
        """Calculate metrics."""
        calculated = {}
        try:
            if data.get("returnOnEquity") is None:
                roa = data.get("returnOnAssets")
                de = data.get("debtToEquity")
                if roa is not None and de is not None:
                    calculated["returnOnEquity"] = roa * (1 + de)
                    calculated["_returnOnEquity_source"] = "calculated_from_roa_de"

            if data.get("pegRatio") is None:
                pe = data.get("trailingPE")
                growth = data.get("earningsGrowth")
                if pe and growth and growth > 0:
                    calculated["pegRatio"] = pe / (growth * 100)
                    calculated["_pegRatio_source"] = "calculated_from_pe_growth"

            # FIX: Ensure marketCap is calculated if missing
            if data.get("marketCap") is None:
                price = data.get("currentPrice") or data.get("regularMarketPrice")
                shares = data.get("sharesOutstanding")
                if price and shares:
                    calculated["marketCap"] = price * shares
                    calculated["_marketCap_source"] = "calculated_from_price_shares"
        except Exception:
            pass
        return calculated

    def _merge_data(self, primary: dict, *fallbacks: dict) -> MergeResult:
        """Merge simple dictionaries."""
        merged = primary.copy() if primary else {}
        gaps = 0
        for fb in fallbacks:
            if not fb:
                continue
            for k, v in fb.items():
                if k in merged and merged[k] is not None:
                    continue
                if v is not None:
                    merged[k] = v
                    if not k.startswith("_"):
                        gaps += 1
        return MergeResult(merged, gaps)

    def _fix_currency_mismatch(self, info: dict, symbol: str) -> dict:
        """Fix currency mismatch."""
        trading_curr = info.get("currency", "USD").upper()
        financial_curr = info.get("financialCurrency", trading_curr).upper()
        price = info.get("currentPrice")
        book = info.get("bookValue")

        if not (book and price):
            return info

        if trading_curr != financial_curr:
            fx = self.get_currency_rate(financial_curr, trading_curr)
            if abs(fx - 1.0) > 0.1:
                info["bookValue"] = book * fx
                info["priceToBook"] = price / info["bookValue"]
        return info

    def _fix_debt_equity_scaling(self, info: dict, symbol: str) -> dict:
        de = info.get("debtToEquity")
        if de is not None and de > DEBT_EQUITY_PERCENTAGE_THRESHOLD:
            info["debtToEquity"] = de / 100.0
        return info

    def _normalize_data_integrity(self, info: dict, symbol: str) -> dict:
        info = self._fix_currency_mismatch(info, symbol)
        info = self._fix_debt_equity_scaling(info, symbol)

        # P/E Normalization with sanity checks
        # Only replace trailing with forward if BOTH values are reasonable
        trailing = info.get("trailingPE")
        forward = info.get("forwardPE")

        if trailing and forward and trailing > 0 and forward > 0:
            # Sanity thresholds based on realistic P/E distributions:
            # - P/E < 5: Almost always suspect (distress, data error, stock split issue)
            # - P/E > 50: Likely temporary (one-time charges depressing earnings)
            # - Divergence > 3x: One value is almost certainly wrong
            MIN_REASONABLE_PE = 5
            MAX_DIVERGENCE_RATIO = 3.0
            HIGH_PE_THRESHOLD = 50

            trailing_reasonable = trailing >= MIN_REASONABLE_PE
            forward_reasonable = forward >= MIN_REASONABLE_PE
            divergence_ratio = max(trailing / forward, forward / trailing)
            ratio_reasonable = divergence_ratio <= MAX_DIVERGENCE_RATIO

            # Only replace trailing with forward if:
            # 1. Trailing is unusually high (suggesting one-time earnings hit)
            # 2. Forward is in a reasonable range (>= 5, not suspiciously low)
            # 3. The divergence isn't extreme (which suggests data error, not real)
            # 4. Trailing is significantly higher than forward (original condition)
            if (
                trailing > HIGH_PE_THRESHOLD
                and forward_reasonable
                and ratio_reasonable
                and trailing > (forward * 1.4)
            ):
                logger.info(
                    "pe_normalized",
                    symbol=symbol,
                    original_trailing=trailing,
                    forward_used=forward,
                    reason="trailing_inflated",
                )
                info["trailingPE"] = forward
                info["_trailingPE_source"] = "normalized_forward_proxy"
            elif not ratio_reasonable:
                # Log suspicious divergence - don't replace, keep trailing
                logger.warning(
                    "pe_divergence_suspicious",
                    symbol=symbol,
                    trailing=trailing,
                    forward=forward,
                    ratio=f"{divergence_ratio:.1f}x",
                    action="keeping_trailing_pe",
                    hint="extreme divergence suggests stale/incorrect forward estimate",
                )
            elif not forward_reasonable and trailing_reasonable:
                # Forward is too low to be trusted, keep trailing
                logger.debug(
                    "pe_forward_suspect",
                    symbol=symbol,
                    trailing=trailing,
                    forward=forward,
                    action="keeping_trailing_pe",
                    hint="forward P/E < 5 suggests data error or stale estimate",
                )

        return info

    def _validate_basics(self, data: dict, symbol: str) -> DataQuality:
        quality = DataQuality()
        quality.sources_used = [k for k, v in self.stats["sources"].items() if v > 0]

        missing = []
        for field in self.REQUIRED_BASICS:
            if field == "currentPrice":
                if not any(
                    k in data
                    for k in ["currentPrice", "regularMarketPrice", "previousClose"]
                ):
                    missing.append("price")
            elif data.get(field) is None:
                missing.append(field)

        quality.basics_missing = missing
        quality.basics_ok = len(missing) == 0

        present = sum(
            1 for field in self.IMPORTANT_FIELDS if data.get(field) is not None
        )
        quality.coverage_pct = (present / len(self.IMPORTANT_FIELDS)) * 100

        return quality

    async def get_financial_metrics(
        self, ticker: str, timeout: int = 30
    ) -> dict[str, Any]:
        """
        UNIFIED APPROACH: Main entry point with parallel sources and mandatory gap-filling.
        Includes EODHD fallback.
        """
        self.stats["fetches"] += 1

        try:
            # PHASE 1: Parallel source execution
            source_results = await self._fetch_all_sources_parallel(ticker)

            # PHASE 3: Smart merge with quality scoring
            merged, merge_metadata = self._smart_merge_with_quality(
                source_results, ticker
            )

            # Panic Mode for Asian tickers
            basics_failed = not all(k in merged for k in self.REQUIRED_BASICS)
            is_asian = ticker.endswith((".HK", ".TW", ".KS", ".T"))

            if not merged or (is_asian and basics_failed):
                logger.warning(
                    "data_vacuum_detected",
                    symbol=ticker,
                    msg="Triggering Panic Mode for Asian ticker",
                )
                all_critical = self.IMPORTANT_FIELDS + self.REQUIRED_BASICS
                tavily_rescue = await self._fetch_tavily_gaps(ticker, all_critical)
                if tavily_rescue:
                    merged = self._merge_gap_fill_data(
                        merged, tavily_rescue, merge_metadata
                    )
                    if "currentPrice" not in merged and "price" in tavily_rescue:
                        merged["currentPrice"] = tavily_rescue["price"]

            if not merged:
                return {"error": "No data available", "symbol": ticker}

            # PHASE 4: Calculate coverage
            coverage = self._calculate_coverage(merged)
            gaps = self._identify_critical_gaps(merged)

            # PHASE 5: Mandatory Tavily gap-filling if needed
            if coverage < 0.70 and gaps:
                tavily_data = await self._fetch_tavily_gaps(ticker, gaps)
                if tavily_data:
                    merged = self._merge_gap_fill_data(
                        merged, tavily_data, merge_metadata
                    )

            # Derived & Normalize
            calculated = self._calculate_derived_metrics(merged, ticker)
            if calculated:
                result = self._merge_data(merged, calculated)
                merged = result.data
                merge_metadata["gaps_filled"] += result.gaps_filled

            merged = self._normalize_data_integrity(merged, ticker)

            # --- DATA HYGIENE PIPELINE ---
            # Run comprehensive validation including new integrity checks
            from src.data.validator import validator as data_validator

            validation = data_validator.validate_comprehensive(merged, ticker)

            # If triangle validation failed and EODHD available, arbitrate
            triangle_result = next(
                (r for r in validation.results if r.category == "triangle"), None
            )

            if triangle_result and not triangle_result.passed:
                logger.warning(
                    "triangle_validation_failed",
                    ticker=ticker,
                    issues=triangle_result.issues,
                )

                # Only call EODHD if we haven't already used it in primary fetch
                sources_used = merge_metadata.get("sources_used", [])
                if self.eodhd_fetcher and "eodhd" not in sources_used:
                    logger.info("attempting_eodhd_arbitration", ticker=ticker)

                    anchor_data = await self.eodhd_fetcher.verify_anchor_metrics(ticker)

                    if anchor_data:
                        # EODHD succeeded - patch the data
                        merged["marketCap"] = anchor_data.get(
                            "marketCap"
                        ) or merged.get("marketCap")
                        merged["trailingPE"] = anchor_data.get(
                            "trailingPE"
                        ) or merged.get("trailingPE")
                        merged["_eodhd_arbitration"] = "success"
                        logger.info("eodhd_arbitration_success", ticker=ticker)
                    else:
                        # EODHD failed (402/error) - flag data as suspect
                        merged["_data_quality_flag"] = "SUSPECT_VALUATION"
                        logger.warning("eodhd_arbitration_failed", ticker=ticker)

            # Continue with existing validation flow
            # Validate
            quality = self._validate_basics(merged, ticker)
            if quality.basics_ok:
                self.stats["basics_ok"] += 1
            else:
                self.stats["basics_failed"] += 1

            # PHASE 6: Metadata
            merged.update(
                {
                    "_coverage_pct": coverage,
                    "_data_source": merge_metadata["composite_source"],
                    "_sources_used": merge_metadata["sources_used"],
                    "_field_sources": merge_metadata.get("field_sources", {}),
                    "_gaps_filled": merge_metadata["gaps_filled"],
                    "_quality": {
                        "basics_ok": quality.basics_ok,
                        "coverage_pct": quality.coverage_pct,
                        "sources_used": quality.sources_used,
                    },
                }
            )

            return merged

        except Exception as e:
            logger.error("unexpected_fetch_error", ticker=ticker, error=str(e))
            return {"error": str(e), "symbol": ticker}

    async def get_price_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetch historical price data."""
        try:
            stock = yf.Ticker(ticker)
            hist = await asyncio.to_thread(stock.history, period=period)
            return hist
        except Exception as e:
            logger.error("history_fetch_failed", ticker=ticker, error=str(e))
            return pd.DataFrame()

    # Alias for backward compatibility and interface compliance
    async def get_historical_prices(
        self, ticker: str, period: str = "1y"
    ) -> pd.DataFrame:
        return await self.get_price_history(ticker, period)

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics on fetcher performance."""
        return self.stats.copy()

    def clear_fx_cache(self):
        """Clear FX rate cache."""
        self.fx_cache = {}
        self.fx_cache_expiry_time = {}


# Singleton instance
fetcher = SmartMarketDataFetcher()


# Backward compatibility
async def fetch_ticker_data(ticker: str) -> dict[str, Any]:
    return await fetcher.get_financial_metrics(ticker)
