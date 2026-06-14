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

## Currency Convention for Returned Data
----------------------------------------------------------------------
get_financial_metrics() returns a merged dict where:

  PRICE / ABSOLUTE fields (currentPrice, 52w_high, 52w_low, market_cap,
  revenue, operating_cash_flow, free_cash_flow, moving averages, …)
      → LOCAL TRADING CURRENCY of the stock (JPY for .T, HKD for .HK, …).
        The 'currency' key in the dict identifies which local currency.

  RATIO / PERCENTAGE fields (pe_ratio, pb_ratio, peg_ratio, profit_margin,
  roa, roe, debt_to_equity, revenue_growth, eps_growth, …)
      → CURRENCY-NEUTRAL (dimensionless; safe to compare across markets).

This dict is consumed by LLM agents for analysis.  The IBKR order-sizing
pipeline (src/ibkr/reconciler.py) does NOT use prices from this dict —
it reads fx_rate_to_usd from analysis snapshots and src/fx_normalization.py.
"""

import asyncio
import copy
import json
import math
import re
import time
from collections import namedtuple
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import structlog
import yfinance as yf

from src.async_utils import run_with_hard_timeout
from src.config import config
from src.data.gap_fill import (
    calculate_coverage as calculate_coverage_impl,
)
from src.data.gap_fill import (
    fetch_tavily_gaps as fetch_tavily_gaps_impl,
)
from src.data.gap_fill import (
    identify_critical_gaps as identify_critical_gaps_impl,
)
from src.data.gap_fill import (
    merge_gap_fill_data as merge_gap_fill_data_impl,
)
from src.data.interfaces import FinancialFetcher
from src.data.merge_policy import QUOTE_PRICE_FIELDS
from src.data.merge_policy import (
    quarantine_forward_pe_outlier as quarantine_forward_pe_outlier_impl,
)
from src.data.merge_policy import (
    smart_merge_with_quality as smart_merge_with_quality_impl,
)
from src.data.metric_extraction import (
    calculate_capital_efficiency_signals as calculate_capital_efficiency_signals_impl,
)
from src.data.metric_extraction import (
    calculate_derived_metrics as calculate_derived_metrics_impl,
)
from src.data.metric_extraction import (
    calculate_graham_earnings_test as calculate_graham_earnings_test_impl,
)
from src.data.metric_extraction import (
    calculate_moat_signals as calculate_moat_signals_impl,
)
from src.data.metric_extraction import (
    calculate_return_trends as calculate_return_trends_impl,
)
from src.data.metric_extraction import (
    compute_trend_regression as compute_trend_regression_impl,
)
from src.data.metric_extraction import (
    extract_from_financial_statements as extract_from_financial_statements_impl,
)
from src.data.metric_extraction import (
    extract_quarterly_horizons as extract_quarterly_horizons_impl,
)
from src.data.pattern_extraction import FinancialPatternExtractor
from src.data.source_fetchers import (
    classify_aggregate_source_failure as classify_aggregate_source_failure_impl,
)
from src.data.source_fetchers import (
    fetch_all_sources_parallel as fetch_all_sources_parallel_impl,
)
from src.data.source_fetchers import (
    fetch_av_fallback as fetch_av_fallback_impl,
)
from src.data.source_fetchers import (
    fetch_eodhd_fallback as fetch_eodhd_fallback_impl,
)
from src.data.source_fetchers import (
    fetch_fmp_fallback as fetch_fmp_fallback_impl,
)
from src.data.source_fetchers import (
    fetch_yahooquery_fallback as fetch_yahooquery_fallback_impl,
)
from src.data.source_fetchers import (
    fetch_yfinance_enhanced as fetch_yfinance_enhanced_impl,
)
from src.error_safety import safe_error_payload, summarize_exception
from src.fx_normalization import (
    is_near_minor_unit_ratio,
    normalize_minor_unit_amount,
)
from src.runtime_services import get_current_market_data_fetcher
from src.tavily_utils import search_tavily_inspected
from src.ticker_policy import (
    FRAGILE_EXCHANGE_SUFFIXES,
    allows_search_resolution,
    get_ticker_suffix,
    normalize_exchange_specific_base,
    sibling_ticker_candidates,
)
from src.ticker_utils import generate_strict_search_query
from src.yfinance_runtime import configure_yfinance_defaults

logger = structlog.get_logger(__name__)
configure_yfinance_defaults()

# --- Optional Dependencies ---
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


# Constants (merge-policy field sets and source rankings live in
# src/data/merge_policy.py — the canonical home; do not redefine here)
# D/E > 10 (1000%) is extremely rare; values like 14.77 are percentages (14.77%)
DEBT_EQUITY_PERCENTAGE_THRESHOLD = 10.0
FX_CACHE_TTL_SECONDS = 3600
FETCH_RESULT_CACHE_TTL_SECONDS = 30
PRICE_HISTORY_CACHE_TTL_SECONDS = 30
PER_SOURCE_TIMEOUT = 15

RECENT_SPLIT_WINDOW_DAYS = 180
SPLIT_RATIO_MATCH_TOLERANCE = 0.25
QUARTER_DATE_RECONCILE_WINDOW_DAYS = 45
MAX_LOW_OVER_PRICE = 1.10
MIN_HIGH_UNDER_PRICE = 0.90
LOW_PE_QUARANTINE_THRESHOLD = 3.0
LOW_PE_FLAG_THRESHOLD = 5.0
PE_IDENTITY_TOLERANCE = 0.20

MergeResult = namedtuple("MergeResult", ["data", "gaps_filled"])


def _quality_notes(info: dict[str, Any]) -> list[str]:
    notes = info.get("_data_quality_notes")
    if not isinstance(notes, list):
        notes = [] if notes in (None, "") else [str(notes)]
        info["_data_quality_notes"] = notes
    return notes


def _coerce_positive_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _identity_match_from_price(
    price: float | None,
    denominator: Any,
    reported_ratio: Any,
    tolerance: float = 0.15,
) -> bool:
    price_f = _safe_float(price)
    denom_f = _safe_float(denominator)
    ratio_f = _safe_float(reported_ratio)
    if price_f is None or denom_f is None or ratio_f is None:
        return False
    if price_f <= 0 or denom_f <= 0 or ratio_f <= 0:
        return False
    expected = price_f / denom_f
    return abs(expected - ratio_f) / ratio_f <= tolerance


def _normalize_history_bound(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _coerce_epoch_date(value: Any) -> datetime | None:
    """Convert common epoch/date representations to a naive UTC datetime."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            if stripped.isdigit():
                return datetime.fromtimestamp(int(stripped), UTC).replace(tzinfo=None)
            return datetime.strptime(stripped, "%Y-%m-%d")
        if isinstance(value, int | float):
            return datetime.fromtimestamp(int(value), UTC).replace(tzinfo=None)
    except (OverflowError, OSError, TypeError, ValueError):
        return None
    return None


def _parse_split_factor(value: Any) -> float | None:
    """Parse split factors like '2:1' into a numeric factor (2.0)."""
    if value in (None, ""):
        return None
    if isinstance(value, int | float):
        return float(value) if float(value) > 0 else None

    if not isinstance(value, str):
        return None

    stripped = value.strip()
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$", stripped)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator > 0:
            return numerator / denominator

    try:
        numeric = float(stripped)
        return numeric if numeric > 0 else None
    except ValueError:
        return None


@dataclass
class DataQuality:
    """Track data quality and sources."""

    basics_ok: bool = False
    basics_missing: list[str] | None = None
    coverage_pct: float = 0.0
    sources_used: list[str] | None = None
    gaps_filled: int = 0
    suspicious_fields: list[str] | None = None

    def __post_init__(self):
        if self.basics_missing is None:
            self.basics_missing = []
        if self.sources_used is None:
            self.sources_used = []
        if self.suspicious_fields is None:
            self.suspicious_fields = []


class SmartMarketDataFetcher(FinancialFetcher):
    """Intelligent multi-source fetcher with unified parallel approach."""

    REQUIRED_BASICS = ["symbol", "currentPrice", "currency"]

    # Exchanges where IBKR mnemonics ≠ yfinance numeric codes (confirmed cases only)
    _MNEMONIC_EXCHANGES = frozenset({".KL"})
    _MNEMONIC_CACHE_FILE = Path("scratch/ticker_mnemonic_map.json")
    _MNEMONIC_CACHE_TTL = 30 * 24 * 3600  # 30 days

    IMPORTANT_FIELDS = [
        "sector",  # Prevents sector hallucination (e.g., industrial classified as tech)
        "industry",  # Sub-classification for sector-specific thresholds
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
        configure_yfinance_defaults()
        self.fx_cache: dict[str, float] = {}
        self.fx_cache_expiry_time: dict[str, datetime] = {}
        self._mnemonic_cache: dict[str, str] = self._load_mnemonic_cache()
        self._metrics_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._metrics_inflight: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._history_failure_logged: set[str] = set()
        self._history_cache: dict[
            tuple[str, str, str | None, str | None], tuple[float, pd.DataFrame]
        ] = {}
        self._history_inflight: dict[
            tuple[str, str, str | None, str | None], asyncio.Task[pd.DataFrame]
        ] = {}
        self._ibkr_security_service = None

        self.fmp_fetcher = get_fmp_fetcher() if FMP_AVAILABLE else None
        self.eodhd_fetcher = get_eodhd_fetcher() if EODHD_AVAILABLE else None
        self.av_fetcher = get_av_fetcher() if ALPHA_VANTAGE_AVAILABLE else None
        self.pattern_extractor = FinancialPatternExtractor()

        api_key = config.get_tavily_api_key()
        self.tavily_client = (
            TavilyClient(api_key=api_key) if TAVILY_LIB_AVAILABLE and api_key else None
        )

        self.stats: dict[str, Any] = {
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

    def _get_cached_metrics(self, ticker: str) -> dict[str, Any] | None:
        cached = self._metrics_cache.get(ticker)
        if not cached:
            return None
        expires_at, payload = cached
        if time.monotonic() >= expires_at:
            self._metrics_cache.pop(ticker, None)
            return None
        logger.debug("financial_metrics_cache_hit", symbol=ticker)
        return copy.deepcopy(payload)

    def _set_cached_metrics(self, ticker: str, payload: dict[str, Any]) -> None:
        if not payload or payload.get("error"):
            return
        self._metrics_cache[ticker] = (
            time.monotonic() + FETCH_RESULT_CACHE_TTL_SECONDS,
            copy.deepcopy(payload),
        )

    def _get_ibkr_security_service(self):
        if self._ibkr_security_service is None:
            from src.ibkr.security_data_service import IbkrSecurityDataService

            self._ibkr_security_service = IbkrSecurityDataService()
        return self._ibkr_security_service

    async def _probe_ibkr_security(self, ticker: str):
        try:
            return await self._get_ibkr_security_service().probe_security(ticker)
        except Exception as exc:
            logger.debug(
                "ibkr_security_probe_failed",
                ticker=ticker,
                **summarize_exception(
                    exc,
                    operation="probing IBKR security data",
                    provider="unknown",
                ),
            )
            return None

    @staticmethod
    def _has_safe_identity_anchor(data: dict[str, Any]) -> bool:
        if not data:
            return False
        return any(data.get(field) for field in ("longName", "shortName", "industry"))

    @staticmethod
    def _has_price_anchor(data: dict[str, Any]) -> bool:
        if not data:
            return False
        return any(
            data.get(field) is not None
            for field in ("currentPrice", "regularMarketPrice", "previousClose")
        )

    def _has_vacuum_anchor(self, data: dict[str, Any]) -> bool:
        """Return True when data has enough quote/entity anchor to avoid panic mode."""
        if not data:
            return False
        return (
            self._has_price_anchor(data)
            or bool(data.get("currency"))
            or bool(data.get("longName"))
            or bool(data.get("shortName"))
        )

    def _has_required_quote_identity(self, data: dict[str, Any]) -> bool:
        return (
            bool(data.get("currency"))
            and self._has_price_anchor(data)
            and self._has_safe_identity_anchor(data)
        )

    async def _ibkr_probe_allows_sibling_rescue(
        self, original: str, candidate: str, candidate_merged: dict[str, Any]
    ) -> bool:
        """Use IBKR as a contract-level guardrail when it is configured."""
        probe = await self._probe_ibkr_security(candidate)
        if probe is None:
            logger.info(
                "ticker_sibling_rescue_rejected",
                original=original,
                candidate=candidate,
                reason="ibkr_probe_missing",
            )
            return False

        confidence = getattr(probe, "identity_confidence", None)
        error_kind = getattr(probe, "error_kind", None)
        resolved = getattr(probe, "resolved_yf_ticker", None)
        candidate_merged["_ticker_rescue_ibkr_identity_confidence"] = confidence
        if error_kind:
            candidate_merged["_ticker_rescue_ibkr_probe_error_kind"] = error_kind

        if (
            getattr(probe, "configured", True) is False
            or error_kind == "NOT_CONFIGURED"
        ):
            logger.info(
                "ticker_sibling_rescue_ibkr_probe_unavailable",
                original=original,
                candidate=candidate,
                reason=error_kind or "NOT_CONFIGURED",
            )
            return True

        if confidence == "VERIFIED" and (not resolved or resolved == candidate):
            candidate_merged["_ibkr_identity_confidence"] = confidence
            if resolved:
                candidate_merged["_ticker_rescue_ibkr_resolved"] = resolved
            if getattr(probe, "company_name", None):
                candidate_merged.setdefault("_ibkr_company_name", probe.company_name)
            if getattr(probe, "market_data_availability", None):
                candidate_merged["_ibkr_market_data_availability"] = (
                    probe.market_data_availability
                )
            return True

        logger.info(
            "ticker_sibling_rescue_rejected",
            original=original,
            candidate=candidate,
            reason="ibkr_probe_not_verified",
            identity_confidence=confidence,
            resolved=resolved,
            error_kind=error_kind,
        )
        # The candidate had full market data but IBKR could not verify identity
        # — the classic signature of a listing migration (e.g. TWSE→TPEx).
        # Never auto-apply (the strict probe prevents cross-matching the wrong
        # security); surface it for the operator instead.
        logger.warning(
            "ticker_migration_suspected",
            original=original,
            candidate=candidate,
            action=(
                "verify listing change; if confirmed, add to "
                "config/ticker_overrides.json"
            ),
        )
        return False

    async def _try_sibling_suffix_rescue(
        self, ticker: str
    ) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
        """Return a verified sibling ticker payload for exchange-suffix vacuums."""
        for candidate in sibling_ticker_candidates(ticker):
            if candidate == ticker:
                continue
            logger.info(
                "ticker_sibling_rescue_attempt",
                original=ticker,
                candidate=candidate,
            )
            source_results = await self._fetch_all_sources_parallel(candidate)
            candidate_merged, candidate_metadata = self._smart_merge_with_quality(
                source_results, candidate
            )
            if self._has_required_quote_identity(
                candidate_merged
            ) and await self._ibkr_probe_allows_sibling_rescue(
                ticker, candidate, candidate_merged
            ):
                candidate_merged["_ticker_rescue_original"] = ticker
                candidate_merged["_ticker_rescue_resolved"] = candidate
                candidate_merged["_ticker_rescue_reason"] = "sibling_suffix_data_vacuum"
                logger.info(
                    "ticker_sibling_rescue_accepted",
                    original=ticker,
                    resolved=candidate,
                )
                return candidate, candidate_merged, candidate_metadata
            logger.info(
                "ticker_sibling_rescue_candidate_incomplete",
                original=ticker,
                candidate=candidate,
                has_price=self._has_price_anchor(candidate_merged),
                has_currency=bool(candidate_merged.get("currency")),
                has_identity=self._has_safe_identity_anchor(candidate_merged),
            )
        return None

    def _history_cache_key(
        self,
        ticker: str,
        period: str,
        start: str | None = None,
        end: str | None = None,
    ) -> tuple[str, str, str | None, str | None]:
        return (
            ticker,
            period,
            _normalize_history_bound(start),
            _normalize_history_bound(end),
        )

    def _get_cached_history(
        self,
        ticker: str,
        period: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame | None:
        cache_key = self._history_cache_key(ticker, period, start, end)
        cached = self._history_cache.get(cache_key)
        if not cached:
            return None
        expires_at, frame = cached
        if time.monotonic() >= expires_at:
            self._history_cache.pop(cache_key, None)
            return None
        logger.debug(
            "price_history_cache_hit",
            symbol=ticker,
            period=period,
            start=start,
            end=end,
        )
        return frame.copy(deep=True)

    def _set_cached_history(
        self,
        ticker: str,
        period: str,
        hist: pd.DataFrame,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        if hist is None or hist.empty:
            return
        self._history_cache[self._history_cache_key(ticker, period, start, end)] = (
            time.monotonic() + PRICE_HISTORY_CACHE_TTL_SECONDS,
            hist.copy(deep=True),
        )

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
            return float(self.fx_cache.get(cache_key, 1.0))

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
                "fx_rate_fetch_failed",
                pair=f"{from_curr}/{to_curr}",
                **summarize_exception(
                    e,
                    operation="fetching FX rate",
                    provider="unknown",
                ),
            )

        return 1.0

    def _extract_from_financial_statements(
        self, ticker: yf.Ticker, symbol: str
    ) -> dict[str, Any]:
        """Extract metrics from yfinance financial statements."""
        return extract_from_financial_statements_impl(self, ticker, symbol)

    def _extract_quarterly_horizons(
        self, ticker: yf.Ticker, symbol: str
    ) -> dict[str, Any]:
        """Extract TTM and MRQ metrics from quarterly financial statements."""
        return extract_quarterly_horizons_impl(ticker, symbol)

    def _calculate_moat_signals(
        self, financials: "pd.DataFrame", cashflow: "pd.DataFrame", symbol: str
    ) -> dict[str, Any]:
        """Calculate economic moat signal metrics from multi-year financial statements."""
        return calculate_moat_signals_impl(financials, cashflow, symbol)

    def _calculate_capital_efficiency_signals(
        self,
        income_stmt: "pd.DataFrame",
        balance_sheet: "pd.DataFrame",
        info: dict[str, Any],
        symbol: str,
        cashflow: "pd.DataFrame | None" = None,
    ) -> dict[str, Any]:
        """Calculate capital efficiency metrics: ROIC, leverage quality, and idle-cash context."""
        return calculate_capital_efficiency_signals_impl(
            income_stmt=income_stmt,
            balance_sheet=balance_sheet,
            info=info,
            symbol=symbol,
            cashflow=cashflow,
        )

    @staticmethod
    def _compute_trend_regression(values: list[float], mean_val: float) -> str:
        """Determine trend using linear regression slope and coefficient of variation."""
        return compute_trend_regression_impl(values, mean_val)

    def _calculate_return_trends(
        self,
        financials: "pd.DataFrame",
        balance_sheet: "pd.DataFrame",
        symbol: str,
    ) -> dict[str, Any]:
        """Calculate 5-year historical average ROA/ROE and trend direction."""
        return calculate_return_trends_impl(financials, balance_sheet, symbol)

    def _calculate_graham_earnings_test(
        self,
        financials: "pd.DataFrame",
        symbol: str,
    ) -> dict[str, Any]:
        """Run Graham's consecutive positive earnings test."""
        return calculate_graham_earnings_test_impl(financials, symbol)

    async def _fetch_yfinance_enhanced(self, symbol: str) -> dict | None:
        """Fetch yfinance data including statement calculation."""
        return await fetch_yfinance_enhanced_impl(self, symbol)

    def _fetch_yahooquery_fallback(self, symbol: str) -> dict | None:
        """Fallback: yahooquery."""
        return fetch_yahooquery_fallback_impl(self, symbol)

    async def _fetch_fmp_fallback(self, symbol: str) -> dict | None:
        """Fallback: FMP."""
        return await fetch_fmp_fallback_impl(self, symbol)

    async def _fetch_eodhd_fallback(self, symbol: str) -> dict | None:
        """Fallback: EOD Historical Data."""
        return await fetch_eodhd_fallback_impl(self, symbol)

    async def _fetch_av_fallback(self, symbol: str) -> dict | None:
        """Fallback: Alpha Vantage."""
        return await fetch_av_fallback_impl(self, symbol)

    async def _fetch_all_sources_parallel(self, symbol: str) -> dict[str, dict | None]:
        """PHASE 1: Launch all data sources in parallel."""
        return await fetch_all_sources_parallel_impl(
            self,
            symbol,
            PER_SOURCE_TIMEOUT,
            logger_obj=logger,
            asyncio_module=asyncio,
        )

    def _classify_aggregate_source_failure(
        self, source_outcomes: dict[str, str]
    ) -> str:
        """Classify aggregate source failure conservatively from per-source outcomes."""
        return classify_aggregate_source_failure_impl(source_outcomes)

    def _normalize_scaling_errors(self, val_a: float, val_b: float) -> float:
        """
        Detects and fixes 100x scaling errors (e.g. Sen vs Ringgit, Pence vs Pound).
        Returns the value normalized to the lower magnitude (base currency).

        This adjudicates conflicts between data providers where one might report
        in sub-units (e.g. cents) while another reports in whole units.
        """
        if not val_a or not val_b:
            return val_a or val_b

        try:
            ratio = val_a / val_b
            # Case 1: val_a is ~100x val_b (e.g. 201 Sen vs 2.01 Ringgit)
            # Use val_b (the base currency value)
            if is_near_minor_unit_ratio(ratio):
                return val_b

            # Case 2: val_b is ~100x val_a (e.g. 2.01 Ringgit vs 201 Sen)
            # Use val_a (the base currency value)
            if is_near_minor_unit_ratio(1 / ratio):
                return val_a

            # No obvious scaling error -> return val_b (the new candidate value)
            return val_b
        except ZeroDivisionError:
            return val_b

    def _smart_merge_with_quality(
        self, source_results: dict[str, dict | None], symbol: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """PHASE 3: Intelligent merge with quality scoring."""
        return smart_merge_with_quality_impl(
            source_results,
            symbol,
            self._quarantine_forward_pe_outlier,
            logger_obj=logger,
        )

    def _quarantine_forward_pe_outlier(
        self, source_results: dict[str, dict | None], symbol: str
    ) -> dict[str, dict | None]:
        return quarantine_forward_pe_outlier_impl(
            source_results,
            symbol,
            logger_obj=logger,
        )

    def _calculate_coverage(self, data: dict) -> float:
        """Calculate percentage of IMPORTANT_FIELDS present."""
        return calculate_coverage_impl(data, self.IMPORTANT_FIELDS)

    def _identify_critical_gaps(self, data: dict) -> list[str]:
        """Identify which critical fields are missing."""
        return identify_critical_gaps_impl(data)

    async def _fetch_tavily_gaps(
        self, symbol: str, missing_fields: list[str]
    ) -> dict[str, Any]:
        """PHASE 5: Tavily gap-filling."""
        return await fetch_tavily_gaps_impl(
            self,
            symbol,
            missing_fields,
            yf_module=yf,
            asyncio_module=asyncio,
            search_fn=search_tavily_inspected,
            query_builder=generate_strict_search_query,
        )

    def _merge_gap_fill_data(
        self,
        merged: dict[str, Any],
        gap_fill_data: dict[str, Any],
        merge_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge Tavily data."""
        return merge_gap_fill_data_impl(merged, gap_fill_data, merge_metadata)

    def _calculate_derived_metrics(self, data: dict, symbol: str) -> dict:
        """Calculate metrics."""
        return calculate_derived_metrics_impl(data, symbol)

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

    def _recompute_quote_derived_metrics(self, info: dict[str, Any]) -> None:
        price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if price is None or price <= 0:
            return

        ratio_denominators = {
            "trailingPE": info.get("trailingEps"),
            "forwardPE": info.get("forwardEps"),
            "priceEpsCurrentYear": info.get("epsCurrentYear"),
            "priceToBook": info.get("bookValue"),
        }
        for field, denominator in ratio_denominators.items():
            denom = _safe_float(denominator)
            if denom is not None and denom > 0:
                info[field] = price / denom

    def _normalize_quote_unit_mismatch(
        self, info: dict[str, Any], symbol: str
    ) -> dict[str, Any]:
        price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        shares = _safe_float(info.get("sharesOutstanding"))
        market_cap = _safe_float(info.get("marketCap"))
        raw_currency = info.get("currency")
        financial_currency = info.get("financialCurrency")

        if price is None or shares is None or market_cap is None or market_cap == 0:
            return info

        corrected_price, major_currency, scale = normalize_minor_unit_amount(
            price, raw_currency
        )
        if scale != 0.01 or major_currency is None or corrected_price is None:
            return info

        raw_ratio = (price * shares) / market_cap
        if not is_near_minor_unit_ratio(raw_ratio):
            return info

        corrected_ratio = (corrected_price * shares) / market_cap
        if not (0.85 < corrected_ratio < 1.15):
            return info

        evidence: list[str] = []
        normalized_financial_currency = (
            financial_currency.strip().upper()
            if isinstance(financial_currency, str)
            else None
        )
        has_alias_support = (
            normalized_financial_currency is None
            or normalized_financial_currency == major_currency
        )
        if has_alias_support:
            evidence.append(f"currency_alias:{raw_currency}")
        if normalized_financial_currency == major_currency:
            evidence.append("financial_currency_match")

        identity_checks = {
            "trailing_pe_identity": _identity_match_from_price(
                corrected_price, info.get("trailingEps"), info.get("trailingPE")
            ),
            "forward_pe_identity": _identity_match_from_price(
                corrected_price, info.get("forwardEps"), info.get("forwardPE")
            ),
            "current_year_pe_identity": _identity_match_from_price(
                corrected_price,
                info.get("epsCurrentYear"),
                info.get("priceEpsCurrentYear"),
            ),
            "price_to_book_identity": _identity_match_from_price(
                corrected_price, info.get("bookValue"), info.get("priceToBook")
            ),
        }
        evidence.extend(name for name, matched in identity_checks.items() if matched)
        identity_count = sum(identity_checks.values())

        if not has_alias_support and identity_count < 2:
            return info

        for field in QUOTE_PRICE_FIELDS:
            value = _safe_float(info.get(field))
            if value is not None:
                info[field] = value * scale

        info["currency"] = major_currency
        self._recompute_quote_derived_metrics(info)
        info["_unit_normalization"] = {
            "kind": "quote_minor_to_major",
            "scale_factor": scale,
            "from_currency": raw_currency,
            "to_currency": major_currency,
            "reason": "triangle_verified_quote_unit_mismatch",
            "evidence": evidence,
        }
        logger.info(
            "quote_unit_normalized",
            symbol=symbol,
            from_currency=raw_currency,
            to_currency=major_currency,
            scale_factor=scale,
            raw_ratio=round(raw_ratio, 4),
            corrected_ratio=round(corrected_ratio, 4),
            evidence=evidence,
        )
        return info

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
        if de is None:
            return info
        # Some sources (yfinance, yahooquery) can return numeric fields as strings
        # for certain tickers/exchanges; convert before any numeric comparison.
        try:
            de = float(de)
        except (TypeError, ValueError):
            return info
        # Statement-calculated values are already ratios — no correction needed
        if info.get("_debtToEquity_source") == "calculated_from_statements":
            return info
        # Large values are clearly percentage format (e.g., 793.0 → divide → 7.93)
        if de > DEBT_EQUITY_PERCENTAGE_THRESHOLD:
            info["debtToEquity"] = de / 100.0
        # Values in 1.5–10 range from API sources may also be percentage format
        # (e.g., yfinance returns 7.93 for a company with 7.93% D/E).
        # Cross-validate: EV < Market Cap confirms net-cash position, making
        # a genuine 1.5×–10× D/E ratio essentially impossible.
        elif de > 1.5:
            ev = info.get("enterpriseValue")
            mc = info.get("marketCap")
            if ev is not None and mc is not None and ev < mc:
                info["debtToEquity"] = de / 100.0
        return info

    def _quarantine_recent_split_forward_metrics(
        self, info: dict[str, Any], symbol: str
    ) -> dict[str, Any]:
        split_date = _coerce_epoch_date(info.get("lastSplitDate"))
        split_factor = _parse_split_factor(info.get("lastSplitFactor"))
        if not split_date or not split_factor or split_factor <= 1.0:
            return info

        reference_dt = (
            _coerce_epoch_date(info.get("regularMarketTime"))
            or _coerce_epoch_date(info.get("currentDate"))
            or datetime.now(UTC).replace(tzinfo=None)
        )
        if (reference_dt - split_date).days > RECENT_SPLIT_WINDOW_DAYS:
            return info

        try:
            trailing_pe = float(info["trailingPE"])
            forward_pe = float(info["forwardPE"])
            trailing_eps = float(info["epsTrailingTwelveMonths"])
            forward_eps = float(info["forwardEps"])
        except (KeyError, TypeError, ValueError):
            return info

        if min(trailing_pe, forward_pe, trailing_eps, forward_eps) <= 0:
            return info

        pe_ratio = trailing_pe / forward_pe
        eps_ratio = forward_eps / trailing_eps
        lower = split_factor * (1 - SPLIT_RATIO_MATCH_TOLERANCE)
        upper = split_factor * (1 + SPLIT_RATIO_MATCH_TOLERANCE)
        if not (lower <= pe_ratio <= upper and lower <= eps_ratio <= upper):
            return info

        notes = info.get("_data_quality_notes")
        if not isinstance(notes, list):
            notes = [] if notes in (None, "") else [str(notes)]
            info["_data_quality_notes"] = notes
        notes.append(
            "Recent split detected: forward EPS / forward P/E appear pre-split; "
            "quarantined forward valuation metrics."
        )
        info["_split_sensitive_metrics_quarantined"] = True
        info["_split_quarantine_reason"] = "recent_split_share_basis_mismatch"
        info["forwardPE"] = None
        info["forwardEps"] = None
        info["pegRatio"] = None
        logger.warning(
            "split_sensitive_metrics_quarantined",
            symbol=symbol,
            split_factor=split_factor,
            split_date=split_date.strftime("%Y-%m-%d"),
            pe_ratio=round(pe_ratio, 3),
            eps_ratio=round(eps_ratio, 3),
        )
        return info

    def _flag_low_pe_anomaly(self, info: dict[str, Any], symbol: str) -> None:
        """Flag or quarantine abnormally low trailing P/E values.

        This runs after the existing trailing-vs-forward reconciliation. If that
        path already quarantined trailing P/E, this helper short-circuits.
        """
        pe = _safe_float(info.get("trailingPE"))
        if pe is None or pe <= 0 or pe >= LOW_PE_FLAG_THRESHOLD:
            return

        price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        eps = _safe_float(
            info.get("trailingEps") or info.get("epsTrailingTwelveMonths")
        )
        identity_ok = (
            price is not None
            and price > 0
            and eps is not None
            and eps > 0
            and abs((price / eps) - pe) / pe <= PE_IDENTITY_TOLERANCE
        )

        notes = _quality_notes(info)

        if pe < LOW_PE_QUARANTINE_THRESHOLD and not identity_ok:
            info["trailingPE"] = None
            info["pegRatio"] = None
            info["_pe_low_anomaly_quarantined"] = True
            notes.append(
                "Trailing P/E below 3 failed price/EPS identity check; quarantined."
            )
            logger.warning(
                "pe_low_anomaly_quarantined",
                symbol=symbol,
                reported_pe=pe,
            )
            return

        stress: list[str] = []
        earnings_growth = _safe_float(info.get("earningsGrowth_TTM"))
        revenue_growth = _safe_float(
            info.get("revenueGrowth_TTM") or info.get("revenueGrowth")
        )
        profit_margins = _safe_float(info.get("profitMargins"))
        if earnings_growth is not None and earnings_growth < -0.20:
            stress.append("earnings_collapse")
        if revenue_growth is not None and revenue_growth < -0.05:
            stress.append("revenue_decline")
        if profit_margins is not None and profit_margins < 0.03:
            stress.append("thin_margins")

        info["_pe_low_anomaly_flag"] = "LOW_PE_REQUIRES_INVESTIGATION"
        info["_pe_low_anomaly_context"] = stress or [
            "low_multiple_confirmed_but_unexplained"
        ]
        logger.info("pe_low_anomaly_flagged", symbol=symbol, reported_pe=pe)

    def _reconcile_latest_quarter_date(
        self, info: dict[str, Any], symbol: str
    ) -> dict[str, Any]:
        latest_dt = _coerce_epoch_date(info.get("latest_quarter_date"))
        most_recent_dt = _coerce_epoch_date(info.get("mostRecentQuarter"))
        if not latest_dt or not most_recent_dt:
            return info

        delta_days = abs((latest_dt - most_recent_dt).days)
        if delta_days <= QUARTER_DATE_RECONCILE_WINDOW_DAYS:
            return info

        newer_dt = max(latest_dt, most_recent_dt)
        if newer_dt == latest_dt:
            return info

        info["latest_quarter_date"] = newer_dt.strftime("%Y-%m-%d")
        info["_latest_quarter_date_source"] = "reconciled_most_recent_quarter"
        logger.info(
            "latest_quarter_date_reconciled",
            symbol=symbol,
            previous=latest_dt.strftime("%Y-%m-%d"),
            reconciled=info["latest_quarter_date"],
            source="mostRecentQuarter",
        )
        return info

    def _normalize_data_integrity(self, info: dict, symbol: str) -> dict:
        info = self._normalize_quote_unit_mismatch(info, symbol)
        info = self._fix_currency_mismatch(info, symbol)
        info = self._fix_debt_equity_scaling(info, symbol)

        # P/E Normalization with sanity checks
        # Only replace trailing with forward if BOTH values are reasonable
        # Convert to float defensively — some sources return numeric fields as strings.
        try:
            trailing = (
                float(info["trailingPE"])
                if info.get("trailingPE") is not None
                else None
            )
            forward = (
                float(info["forwardPE"]) if info.get("forwardPE") is not None else None
            )
        except (TypeError, ValueError):
            trailing = None
            forward = None

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
                # Distinguish unit/decimal/currency error from genuine
                # staleness. Pattern: ratio ≈ 10/100/1000×, with one value in
                # the plausible band [MIN_REASONABLE_PE..PLAUSIBLE_PE_HIGH]
                # and the other extreme. Staleness produces drifty ratios,
                # not clean powers of ten — a near-exact ×100 with one side
                # in [10, 25] and the other ≪ 1 is almost certainly an EPS
                # unit mismatch (cents vs dollars, GBp vs GBP, …).
                PLAUSIBLE_PE_HIGH = 30.0
                EXTREME_LOW = 1.0
                EXTREME_HIGH = 98.0
                POWER_OF_TEN_TOLERANCE = 0.05

                log_ratio = math.log10(divergence_ratio)
                power_magnitude = round(log_ratio)
                near_power_of_ten = (
                    1 <= power_magnitude <= 3
                    and abs(log_ratio - power_magnitude) < POWER_OF_TEN_TOLERANCE
                )
                trailing_plausible = MIN_REASONABLE_PE <= trailing <= PLAUSIBLE_PE_HIGH
                forward_plausible = MIN_REASONABLE_PE <= forward <= PLAUSIBLE_PE_HIGH
                trailing_extreme = trailing < EXTREME_LOW or trailing > EXTREME_HIGH
                forward_extreme = forward < EXTREME_LOW or forward > EXTREME_HIGH

                unit_error = (
                    near_power_of_ten
                    and (trailing_plausible ^ forward_plausible)
                    and (trailing_extreme ^ forward_extreme)
                )

                if unit_error:
                    suspect = "forward" if forward_extreme else "trailing"
                    # Quarantine the suspect — same pattern as
                    # _quarantine_recent_split_forward_metrics.
                    notes = info.get("_data_quality_notes")
                    if not isinstance(notes, list):
                        notes = [] if notes in (None, "") else [str(notes)]
                        info["_data_quality_notes"] = notes
                    notes.append(
                        f"Suspected unit/decimal/currency error in {suspect} "
                        f"P/E (ratio ≈ {10**power_magnitude}× to plausible "
                        f"side); quarantined {suspect} valuation metrics."
                    )
                    if suspect == "forward":
                        info["forwardPE"] = None
                        info["forwardEps"] = None
                        info["pegRatio"] = None
                    else:
                        info["trailingPE"] = None
                        info["epsTrailingTwelveMonths"] = None
                    info["_pe_unit_error_quarantined"] = suspect
                    logger.warning(
                        "pe_divergence_unit_error_suspect",
                        symbol=symbol,
                        trailing=trailing,
                        forward=forward,
                        ratio=f"{divergence_ratio:.1f}x",
                        power_of_ten=power_magnitude,
                        suspect=suspect,
                        action=f"quarantined_{suspect}_metrics",
                        hint=(
                            "ratio within tolerance of 10/100/1000×; one "
                            "value plausible, the other out of range — "
                            "likely unit/decimal/currency mismatch (e.g. "
                            "EPS in minor units, GBp vs GBP) in the "
                            "suspect source"
                        ),
                    )
                else:
                    # Genuine staleness/incorrectness — keep trailing as before.
                    logger.warning(
                        "pe_divergence_suspicious",
                        symbol=symbol,
                        trailing=trailing,
                        forward=forward,
                        ratio=f"{divergence_ratio:.1f}x",
                        action="keeping_trailing_pe",
                        hint=(
                            "extreme divergence suggests stale/incorrect "
                            "forward estimate"
                        ),
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

        info = self._quarantine_recent_split_forward_metrics(info, symbol)
        info = self._reconcile_latest_quarter_date(info, symbol)
        self._flag_low_pe_anomaly(info, symbol)

        return info

    async def _repair_quote_range_from_history(
        self, info: dict[str, Any], symbol: str
    ) -> dict[str, Any]:
        price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        low = _safe_float(info.get("fiftyTwoWeekLow"))
        high = _safe_float(info.get("fiftyTwoWeekHigh"))

        repair_low = (
            low is None
            or low <= 0
            or (price is not None and low > price * MAX_LOW_OVER_PRICE)
        )
        repair_high = (
            high is None
            or high <= 0
            or (price is not None and high < price * MIN_HIGH_UNDER_PRICE)
        )
        if not (repair_low or repair_high):
            return info

        notes = _quality_notes(info)
        try:
            hist = await self.get_price_history(symbol, period="1y")
        except Exception as exc:
            hist = None
            logger.warning(
                "quote_range_history_fetch_failed",
                symbol=symbol,
                **summarize_exception(
                    exc,
                    operation="get_price_history",
                    provider="yfinance",
                ),
            )

        if hist is not None and not hist.empty and {"Low", "High"} <= set(hist.columns):
            if repair_low:
                info["fiftyTwoWeekLow"] = float(hist["Low"].dropna().min())
                info["_fiftyTwoWeekLow_source"] = "calculated_from_1y_history"
            if repair_high:
                info["fiftyTwoWeekHigh"] = float(hist["High"].dropna().max())
                info["_fiftyTwoWeekHigh_source"] = "calculated_from_1y_history"
            notes.append("52-week range repaired from 1y price history.")
            logger.info("quote_range_repaired", symbol=symbol)
        else:
            if repair_low:
                info["fiftyTwoWeekLow"] = None
            if repair_high:
                info["fiftyTwoWeekHigh"] = None
            notes.append(
                "Invalid or missing 52-week range blanked; history unavailable."
            )
            logger.warning("quote_range_blanked", symbol=symbol)
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

    def _load_mnemonic_cache(self) -> dict[str, str]:
        """Load the mnemonic→numeric cache, discarding entries older than TTL."""
        if not self._MNEMONIC_CACHE_FILE.exists():
            return {}
        try:
            with open(self._MNEMONIC_CACHE_FILE) as f:
                raw = json.load(f)
            now = time.time()
            return {
                k: v["resolved"]
                for k, v in raw.items()
                if now - v.get("ts", 0) < self._MNEMONIC_CACHE_TTL
            }
        except (json.JSONDecodeError, OSError, KeyError):
            return {}

    def _save_mnemonic_cache(self, original: str, resolved: str) -> None:
        """Persist a new mnemonic→numeric mapping to the file cache."""
        self._mnemonic_cache[original] = resolved
        raw = {
            k: {"resolved": v, "ts": time.time()}
            for k, v in self._mnemonic_cache.items()
        }
        self._MNEMONIC_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._MNEMONIC_CACHE_FILE, "w") as f:
                json.dump(raw, f, indent=2)
        except OSError as e:
            logger.warning(
                "mnemonic_cache_save_failed",
                **summarize_exception(
                    e,
                    operation="saving mnemonic cache",
                    provider="unknown",
                ),
            )

    async def _pre_resolve_ticker(self, ticker: str) -> str:
        """
        Map mnemonic tickers (e.g. SCIENTX.KL) to numeric yfinance equivalents
        (e.g. 4731.KL) before any data source is called.

        Only fires for exchanges in _MNEMONIC_EXCHANGES when the base symbol
        contains alphabetic characters. Uses yahooquery.search() with a 30-day
        file-backed cache. Falls through silently on any error — the existing
        Tavily fallback in _resolve_ticker_via_search() still active downstream.
        """
        suffix = next((s for s in self._MNEMONIC_EXCHANGES if ticker.endswith(s)), None)
        if not suffix:
            return ticker

        base = ticker[: -len(suffix)]
        if base.isdigit():  # already numeric — nothing to do
            return ticker

        if ticker in self._mnemonic_cache:
            resolved = self._mnemonic_cache[ticker]
            logger.debug("mnemonic_cache_hit", original=ticker, resolved=resolved)
            return resolved

        try:
            from yahooquery import search as yq_search

            result = await run_with_hard_timeout(
                asyncio.to_thread(yq_search, base),
                timeout=10,
                label=f"yahooquery_search:{base}",
            )
            quotes = result.get("quotes", []) if isinstance(result, dict) else []
            for q in quotes:
                if not isinstance(q, dict):
                    continue
                sym = q.get("symbol", "")
                if not isinstance(sym, str):
                    continue
                # Accept only results on the same exchange whose base is all-digits
                if (
                    sym.endswith(suffix)
                    and sym[: -len(suffix)].isdigit()
                    and sym != ticker
                ):
                    logger.info("mnemonic_pre_resolved", original=ticker, resolved=sym)
                    self._save_mnemonic_cache(ticker, sym)
                    return sym
        except Exception as e:
            logger.warning(
                "mnemonic_pre_resolve_failed",
                ticker=ticker,
                **summarize_exception(
                    e,
                    operation="pre-resolving ticker mnemonic",
                    provider="unknown",
                ),
            )

        return ticker  # original ticker; Tavily fallback still active

    async def _resolve_ticker_via_search(self, symbol: str) -> str | None:
        """
        Recover from a failed yfinance lookup by searching for the numeric ticker.
        Used primarily for Asian markets where users provide Alpha codes (e.g., PADINI.KL)
        but yfinance expects Numeric codes (e.g., 7052.KL).

        This prevents 'Frankenstein Analysis' where tools might otherwise drift to
        the wrong entity due to symbol ambiguity.
        """
        if not self.tavily_client:
            return None

        if not allows_search_resolution(symbol):
            return None

        # ONLY alpha mnemonics may be search-resolved (PADINI.KL → 7052.KL):
        # the mnemonic itself names the company, so the numeric result is the
        # same entity by construction. A NUMERIC input (e.g. delisted 1264.TW)
        # must never be substituted with whichever 4-digit ticker happens to
        # appear in a search result — that produced a wrong-company analysis
        # (1264.TW → 8341.TW, June 2026). Mirrors _pre_resolve_ticker's guard.
        base, _, _suffix = symbol.rpartition(".")
        if not base or base.isdigit():
            return None

        try:
            # Construct a surgical query to find the numeric code
            query = f"{symbol} yahoo finance ticker numeric code"

            # Use Tavily to find the mapping
            result = await run_with_hard_timeout(
                asyncio.to_thread(self.tavily_client.search, query, max_results=1),
                timeout=10,
                label=f"tavily_ticker_resolve:{symbol}",
            )

            if not result or "results" not in result or not result["results"]:
                return None

            content = (
                result["results"][0].get("content", "")
                + " "
                + result["results"][0].get("title", "")
            )

            # Look for patterns like "7052.KL" or just "7052" near the company name
            # Only rescue when the result stays on the exact same exchange suffix.
            _, suffix = symbol.rsplit(".", 1)
            match = re.search(
                rf"\b(\d{{4}})\.{re.escape(suffix)}\b", content, re.IGNORECASE
            )

            if match:
                resolved_base = normalize_exchange_specific_base(
                    match.group(1), f".{suffix}"
                )
                resolved = f"{resolved_base}.{suffix}".upper()
                # Sanity check: Ensure it's different from input
                if resolved != symbol.upper():
                    return resolved

            return None

        except Exception as e:
            logger.warning(
                "ticker_resolution_failed",
                symbol=symbol,
                **summarize_exception(
                    e,
                    operation="resolving ticker via search",
                    provider="unknown",
                ),
            )
            return None

    async def _get_financial_metrics_uncached(
        self, ticker: str, timeout: int = 30
    ) -> dict[str, Any]:
        """Hard wall-clock limit on the entire metrics fetch cycle."""
        try:
            return await asyncio.wait_for(
                self._fetch_metrics_inner(ticker),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "get_financial_metrics_total_timeout", ticker=ticker, timeout=timeout
            )
            return {}

    async def _fetch_metrics_inner(self, ticker: str) -> dict[str, Any]:
        """
        UNIFIED APPROACH: Main entry point with parallel sources and mandatory gap-filling.
        Includes EODHD fallback and Smart Ticker Resolution.
        """
        self.stats["fetches"] += 1

        try:
            # PHASE 0: Proactive mnemonic→numeric pre-resolution (e.g. SCIENTX.KL → 4731.KL)
            ticker = await self._pre_resolve_ticker(ticker)

            # PHASE 1: Parallel source execution
            source_results = await self._fetch_all_sources_parallel(ticker)

            # CHECK: Did yfinance fail completely?
            # If so, and it's a target market, try to resolve the ticker
            if not source_results.get("yfinance"):
                resolved_ticker = await self._resolve_ticker_via_search(ticker)
                if resolved_ticker:
                    logger.info(
                        "ticker_resolved_retrying",
                        original=ticker,
                        resolved=resolved_ticker,
                    )
                    # Retry with resolved ticker
                    source_results = await self._fetch_all_sources_parallel(
                        resolved_ticker
                    )
                    # Update ticker for downstream consistency
                    ticker = resolved_ticker

            # PHASE 3: Smart merge with quality scoring
            merged, merge_metadata = self._smart_merge_with_quality(
                source_results, ticker
            )

            # Panic Mode: full vacuum (any market) OR basics missing for fragile exchanges
            basics_failed = not all(k in merged for k in self.REQUIRED_BASICS)
            suffix = get_ticker_suffix(ticker)
            is_fragile_exchange = suffix in FRAGILE_EXCHANGE_SUFFIXES
            identity_weak = (
                bool(suffix)
                and not self._has_safe_identity_anchor(merged)
                and not merged.get("currentPrice")
            )
            identity_vacuum = bool(merged) and not self._has_vacuum_anchor(merged)

            if (
                not merged
                or (is_fragile_exchange and basics_failed)
                or identity_weak
                or identity_vacuum
            ):
                if not merged:
                    panic_reason = "total data vacuum"
                elif identity_weak:
                    panic_reason = "identity weak"
                elif identity_vacuum:
                    panic_reason = "identity vacuum"
                else:
                    panic_reason = "basics missing"
                logger.warning(
                    "data_vacuum_detected",
                    symbol=ticker,
                    msg=f"Triggering Panic Mode ({panic_reason})",
                )
                all_critical = self.IMPORTANT_FIELDS + self.REQUIRED_BASICS
                sibling_rescue = await self._try_sibling_suffix_rescue(ticker)
                if sibling_rescue is not None:
                    ticker, merged, merge_metadata = sibling_rescue

                probe = (
                    None
                    if sibling_rescue is not None
                    else await self._probe_ibkr_security(ticker)
                )
                if probe is not None:
                    merged.setdefault("_ibkr_probe_used", True)
                    merged["_ibkr_identity_confidence"] = probe.identity_confidence
                    if probe.company_name:
                        merged["_ibkr_company_name"] = probe.company_name
                    if probe.market_data_availability:
                        merged["_ibkr_market_data_availability"] = (
                            probe.market_data_availability
                        )
                    if probe.error_kind:
                        merged["_ibkr_probe_error_kind"] = probe.error_kind

                    if probe.identity_confidence == "VERIFIED":
                        resolved_ticker = probe.resolved_yf_ticker or ticker
                        if not merged.get("symbol"):
                            merged["symbol"] = resolved_ticker
                        if probe.company_name and not merged.get("longName"):
                            merged["longName"] = probe.company_name
                        if probe.currency and not merged.get("currency"):
                            merged["currency"] = probe.currency
                        if probe.last_price is not None and not merged.get(
                            "currentPrice"
                        ):
                            merged["currentPrice"] = probe.last_price

                        if resolved_ticker and resolved_ticker != ticker:
                            original_ticker = ticker
                            logger.info(
                                "ibkr_ticker_retry",
                                original=original_ticker,
                                resolved=resolved_ticker,
                            )
                            retry_results = await self._fetch_all_sources_parallel(
                                resolved_ticker
                            )
                            retry_merged, retry_metadata = (
                                self._smart_merge_with_quality(
                                    retry_results, resolved_ticker
                                )
                            )
                            if retry_merged:
                                retry_merge = (
                                    self._merge_data(merged, retry_merged)
                                    if merged
                                    else MergeResult(retry_merged, len(retry_merged))
                                )
                                merged = retry_merge.data
                                merge_metadata["gaps_filled"] += retry_merge.gaps_filled
                                merge_metadata["sources_used"] = sorted(
                                    set(merge_metadata["sources_used"])
                                    | set(retry_metadata.get("sources_used", []))
                                )
                                merge_metadata["field_sources"].update(
                                    {
                                        key: value
                                        for key, value in retry_metadata.get(
                                            "field_sources", {}
                                        ).items()
                                        if key not in merge_metadata["field_sources"]
                                    }
                                )
                                merged["symbol"] = resolved_ticker
                                merged["_ticker_rescue_original"] = original_ticker
                                merged["_ticker_rescue_resolved"] = resolved_ticker
                                merged["_ticker_rescue_reason"] = (
                                    "ibkr_original_probe_identity_vacuum"
                                )
                                ticker = resolved_ticker

                if sibling_rescue is None:
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
                existing_fields = set(merged)
                result = self._merge_data(merged, calculated)
                merged = result.data
                merge_metadata["gaps_filled"] += result.gaps_filled
                for key in calculated:
                    if not key.startswith("_") and key not in existing_fields:
                        merge_metadata["field_sources"][key] = calculated.get(
                            f"_{key}_source", "calculated"
                        )

            merged = self._normalize_data_integrity(merged, ticker)
            merged = await self._repair_quote_range_from_history(merged, ticker)

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
                    "_source_conflicts": merge_metadata.get("source_conflicts", {}),
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
            logger.error(
                "unexpected_fetch_error",
                ticker=ticker,
                **summarize_exception(
                    e,
                    operation="get_financial_metrics",
                    provider="unknown",
                ),
            )
            return safe_error_payload(
                e,
                operation="get_financial_metrics",
                provider="unknown",
                extra={"symbol": ticker},
            )

    async def get_financial_metrics(
        self, ticker: str, timeout: int = 30
    ) -> dict[str, Any]:
        """Return unified metrics with short-lived per-process caching and inflight dedupe."""
        cached = self._get_cached_metrics(ticker)
        if cached is not None:
            return cached

        inflight = self._metrics_inflight.get(ticker)
        if inflight is not None:
            logger.debug("financial_metrics_inflight_wait", symbol=ticker)
            return copy.deepcopy(await inflight)

        task = asyncio.create_task(
            self._get_financial_metrics_uncached(ticker, timeout)
        )
        self._metrics_inflight[ticker] = task

        # Clear the dedup slot the moment the task ends (success, exception, or
        # cancellation) — independent of whether anyone is still awaiting it.
        # A try/finally around ``await task`` would only run if the awaiter
        # itself returns; if a hung blocking thread keeps the task pending,
        # try/finally would leak the slot indefinitely.
        def _clear_metrics_slot(_t: asyncio.Task[Any], key: str = ticker) -> None:
            self._metrics_inflight.pop(key, None)

        task.add_done_callback(_clear_metrics_slot)
        result = await task

        self._set_cached_metrics(ticker, result)
        return copy.deepcopy(result)

    async def _get_price_history_uncached(
        self,
        ticker: str,
        period: str = "1y",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical price data."""
        try:
            stock = yf.Ticker(ticker)
            history_kwargs: dict[str, str] = {}
            normalized_start = _normalize_history_bound(start)
            normalized_end = _normalize_history_bound(end)
            if normalized_start is not None:
                history_kwargs["start"] = normalized_start
            if normalized_end is not None:
                history_kwargs["end"] = normalized_end
            if not history_kwargs:
                history_kwargs["period"] = period

            hist = await run_with_hard_timeout(
                asyncio.to_thread(stock.history, **history_kwargs),
                timeout=15,
                label=f"yfinance.history:{ticker}",
            )
            if hist.empty:
                resolved = await self._resolve_ticker_via_search(ticker)
                if resolved:
                    logger.info(
                        "history_ticker_resolved", original=ticker, resolved=resolved
                    )
                    stock = yf.Ticker(resolved)
                    hist = await run_with_hard_timeout(
                        asyncio.to_thread(stock.history, **history_kwargs),
                        timeout=15,
                        label=f"yfinance.history:{resolved}",
                    )
            return hist
        except Exception as e:
            summary = summarize_exception(
                e, operation="get_historical_prices", provider="unknown"
            )
            repeated = ticker in self._history_failure_logged
            self._history_failure_logged.add(ticker)
            if repeated:
                log = logger.debug  # one operator-visible record per ticker per run
            elif summary["failure_kind"] == "data_unavailable":
                log = (
                    logger.warning
                )  # expected absence (delisted/migrated), not a fault
            else:
                log = logger.error
            log("history_fetch_failed", ticker=ticker, repeated=repeated, **summary)
            return pd.DataFrame()

    async def get_price_history(
        self,
        ticker: str,
        period: str = "1y",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical price data with short-lived caching and inflight dedupe."""
        normalized_start = _normalize_history_bound(start)
        normalized_end = _normalize_history_bound(end)
        cached = self._get_cached_history(
            ticker,
            period,
            start=normalized_start,
            end=normalized_end,
        )
        if cached is not None:
            return cached

        key = self._history_cache_key(
            ticker,
            period,
            start=normalized_start,
            end=normalized_end,
        )
        inflight = self._history_inflight.get(key)
        if inflight is not None:
            logger.debug(
                "price_history_inflight_wait",
                symbol=ticker,
                period=period,
                start=normalized_start,
                end=normalized_end,
            )
            return (await inflight).copy(deep=True)

        task = asyncio.create_task(
            self._get_price_history_uncached(
                ticker,
                period,
                start=normalized_start,
                end=normalized_end,
            )
        )
        self._history_inflight[key] = task
        history_key = key  # bind for the closure below

        # See _metrics_inflight cleanup note: clear via done-callback so a
        # hung underlying fetch cannot leak the dedup slot.
        def _clear_history_slot(_t: asyncio.Task[Any]) -> None:
            self._history_inflight.pop(history_key, None)

        task.add_done_callback(_clear_history_slot)
        hist = await task

        self._set_cached_history(
            ticker,
            period,
            hist,
            start=normalized_start,
            end=normalized_end,
        )
        return hist.copy(deep=True)

    # Alias for backward compatibility and interface compliance
    async def get_historical_prices(
        self,
        ticker: str,
        period: str = "1y",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        return await self.get_price_history(
            ticker,
            period=period,
            start=start,
            end=end,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics on fetcher performance."""
        return cast(dict[str, Any], self.stats.copy())

    def clear_fx_cache(self):
        """Clear FX rate cache."""
        self.fx_cache = {}
        self.fx_cache_expiry_time = {}


_fetcher_instance: SmartMarketDataFetcher | None = None


def get_fetcher() -> SmartMarketDataFetcher:
    """Return the process-wide market data fetcher, initializing lazily."""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = SmartMarketDataFetcher()
    return _fetcher_instance


class _LazyFetcherProxy:
    """Proxy that defers SmartMarketDataFetcher construction until first use."""

    async def get_financial_metrics(self, *args, **kwargs):
        return await get_current_market_data_fetcher().get_financial_metrics(
            *args, **kwargs
        )

    async def get_historical_prices(self, *args, **kwargs):
        return await get_current_market_data_fetcher().get_historical_prices(
            *args, **kwargs
        )

    async def get_price_history(self, *args, **kwargs):
        return await get_current_market_data_fetcher().get_price_history(
            *args, **kwargs
        )

    def get_stats(self, *args, **kwargs):
        return get_current_market_data_fetcher().get_stats(*args, **kwargs)

    def clear_fx_cache(self, *args, **kwargs):
        return get_current_market_data_fetcher().clear_fx_cache(*args, **kwargs)

    def is_available(self, *args, **kwargs):
        return get_current_market_data_fetcher().is_available(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(get_current_market_data_fetcher(), name)

    def __repr__(self) -> str:
        status = "initialized" if _fetcher_instance is not None else "lazy"
        return f"<_LazyFetcherProxy {status}>"


# Singleton proxy
fetcher = _LazyFetcherProxy()


# Backward compatibility
async def fetch_ticker_data(ticker: str) -> dict[str, Any]:
    return await get_fetcher().get_financial_metrics(ticker)
