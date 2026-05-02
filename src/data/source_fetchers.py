"""Individual source-fetch helpers for SmartMarketDataFetcher."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
import yfinance as yf

from src.error_safety import summarize_exception
from src.yfinance_runtime import YFRateLimitError

logger = structlog.get_logger(__name__)

MIN_INFO_FIELDS = 3

try:
    from yahooquery import Ticker as YQTicker

    YAHOOQUERY_AVAILABLE = True
except ImportError:
    YAHOOQUERY_AVAILABLE = False
    YQTicker = None


async def fetch_yfinance_enhanced(fetcher: Any, symbol: str) -> dict | None:
    """Fetch yfinance data including statement-derived enrichment."""
    try:
        ticker = yf.Ticker(symbol)
        info: dict[str, Any] = {}
        try:
            info = await asyncio.to_thread(lambda: ticker.info)
        except YFRateLimitError as exc:
            logger.warning(
                "yfinance_rate_limited",
                symbol=symbol,
                **summarize_exception(
                    exc,
                    operation="fetching yfinance enhanced data",
                    provider="unknown",
                ),
            )
            return None
        except Exception:
            info = {}

        has_price = any(
            info.get(field) is not None
            for field in ["currentPrice", "regularMarketPrice", "previousClose"]
        )
        if not has_price and hasattr(ticker, "fast_info"):
            try:
                fast_info = await asyncio.to_thread(lambda: ticker.fast_info)
                fast_price = fast_info.get("lastPrice")
                if fast_price:
                    info["currentPrice"] = fast_price
                    has_price = True
            except (AttributeError, KeyError):
                pass

        if not has_price:
            logger.warning("yfinance_no_price", symbol=symbol)
            info = info or {}

        statement_data = await asyncio.to_thread(
            fetcher._extract_from_financial_statements, ticker, symbol
        )
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
                info[key] = value
            elif key not in info or info.get(key) is None:
                if value is not None:
                    info[key] = value

        if not info or (not has_price and len(info) < 5):
            return None
        if "symbol" not in info:
            info["symbol"] = symbol

        trading_curr = info.get("currency", "").upper()
        financial_curr = info.get("financialCurrency", "").upper()
        if trading_curr and financial_curr and trading_curr != financial_curr:
            info["cross_listing_note"] = (
                f"Price data is in {trading_curr} ({info.get('exchange', 'unknown exchange')}). "
                f"Financial statements are reported in {financial_curr}. "
                f"This is a cross-listing — the company's primary exchange may use {financial_curr}. "
                f"All price and liquidity data refers to the {trading_curr} listing only."
            )

        fetcher.stats["sources"]["yfinance"] += 1
        return info
    except YFRateLimitError as exc:
        logger.warning(
            "yfinance_rate_limited",
            symbol=symbol,
            **summarize_exception(
                exc,
                operation="fetching yfinance enhanced data",
                provider="unknown",
            ),
        )
        return None
    except Exception as exc:
        logger.error(
            "yfinance_enhanced_failed",
            symbol=symbol,
            **summarize_exception(
                exc,
                operation="fetching yfinance enhanced data",
                provider="unknown",
            ),
        )
        return None


def fetch_yahooquery_fallback(fetcher: Any, symbol: str) -> dict | None:
    """Fallback fetch via yahooquery."""
    if not YAHOOQUERY_AVAILABLE or YQTicker is None:
        return None
    try:
        yq = YQTicker(symbol)
        combined: dict[str, Any] = {}
        for module in [
            yq.summary_profile,
            yq.summary_detail,
            yq.key_stats,
            yq.financial_data,
            yq.price,
        ]:
            if isinstance(module, dict) and symbol in module:
                data = module[symbol]
                if isinstance(data, dict):
                    combined.update(data)
        if not combined or len(combined) < MIN_INFO_FIELDS:
            return None
        if "currentPrice" not in combined and "regularMarketPrice" in combined:
            combined["currentPrice"] = combined["regularMarketPrice"]
        fetcher.stats["sources"]["yahooquery"] += 1
        return combined
    except Exception:
        return None


async def fetch_fmp_fallback(fetcher: Any, symbol: str) -> dict | None:
    """Fallback fetch via FMP."""
    if not fetcher.fmp_fetcher or not fetcher.fmp_fetcher.is_available():
        return None
    try:
        fmp_data = await fetcher.fmp_fetcher.get_financial_metrics(symbol)
        if fmp_data and any(
            v is not None for k, v in fmp_data.items() if k != "_source"
        ):
            fetcher.stats["sources"]["fmp"] += 1
            return fmp_data
    except Exception:
        return None
    return None


async def fetch_eodhd_fallback(fetcher: Any, symbol: str) -> dict | None:
    """Fallback fetch via EODHD."""
    if not fetcher.eodhd_fetcher or not fetcher.eodhd_fetcher.is_available():
        return None
    try:
        data = await fetcher.eodhd_fetcher.get_financial_metrics(symbol)
        if data and any(v is not None for k, v in data.items() if k != "_source"):
            fetcher.stats["sources"]["eodhd"] += 1
            return data
        return None
    except Exception as exc:
        logger.warning(
            "eodhd_fetch_error",
            symbol=symbol,
            **summarize_exception(
                exc,
                operation="fetching EODHD data",
                provider="unknown",
            ),
        )
        return None


async def fetch_av_fallback(fetcher: Any, symbol: str) -> dict | None:
    """Fallback fetch via Alpha Vantage."""
    if not fetcher.av_fetcher or not fetcher.av_fetcher.is_available():
        return None
    try:
        data = await fetcher.av_fetcher.get_financial_metrics(symbol)
        if data and any(
            v is not None for k, v in data.items() if not k.startswith("_")
        ):
            fetcher.stats["sources"]["alpha_vantage"] += 1
            return data
        return None
    except Exception as exc:
        logger.warning(
            "alpha_vantage_fetch_error",
            symbol=symbol,
            **summarize_exception(
                exc,
                operation="fetching Alpha Vantage data",
                provider="unknown",
            ),
        )
        return None


def classify_aggregate_source_failure(source_outcomes: dict[str, str]) -> str:
    """Classify aggregate multi-source failure conservatively."""
    transient_markers = ("timeout", "connect", "proxy", "ssl", "dns", "socket")
    non_success = [
        outcome for outcome in source_outcomes.values() if outcome != "success"
    ]
    if non_success and all(
        any(marker in outcome.lower() for marker in transient_markers)
        for outcome in non_success
    ):
        return "connectivity_or_provider_transient"
    return "no_data_or_provider_unavailable"


async def fetch_all_sources_parallel(
    fetcher: Any,
    symbol: str,
    per_source_timeout: int,
    *,
    logger_obj: Any = logger,
    asyncio_module: Any = asyncio,
) -> dict[str, dict | None]:
    """Launch all configured sources concurrently and collect outcomes.

    Each source has a hard per-source timeout that orphans the underlying
    task if it exceeds the deadline (see ``run_with_hard_timeout``); slow or
    hung providers cannot block sibling providers or the caller's wall clock
    beyond ``per_source_timeout``.
    """
    from src.async_utils import run_with_hard_timeout

    logger_obj.info("launching_parallel_sources", symbol=symbol)
    builders = {
        "yfinance": lambda: fetcher._fetch_yfinance_enhanced(symbol),
        "yahooquery": lambda: asyncio_module.to_thread(
            fetcher._fetch_yahooquery_fallback, symbol
        ),
        "fmp": lambda: fetcher._fetch_fmp_fallback(symbol),
        "eodhd": lambda: fetcher._fetch_eodhd_fallback(symbol),
        "alpha_vantage": lambda: fetcher._fetch_av_fallback(symbol),
    }

    async def _run_one(name: str) -> tuple[str, dict | None, str]:
        try:
            result = await run_with_hard_timeout(
                builders[name](),
                timeout=per_source_timeout,
                label=f"data_source:{name}:{symbol}",
            )
        except asyncio_module.TimeoutError:
            logger_obj.warning(
                f"{name}_timeout",
                symbol=symbol,
                timeout_seconds=per_source_timeout,
            )
            return name, None, "timeout"
        except Exception as exc:
            logger_obj.warning(
                f"{name}_error",
                symbol=symbol,
                **summarize_exception(
                    exc,
                    operation=f"fetching {name} data",
                    provider="unknown",
                ),
            )
            return name, None, f"error:{type(exc).__name__}"

        if result:
            logger_obj.info(f"{name}_success", symbol=symbol, fields=len(result))
            return name, result, "success"
        logger_obj.debug(f"{name}_returned_none", symbol=symbol)
        return name, None, "empty"

    completed = await asyncio_module.gather(
        *(_run_one(name) for name in builders),
        return_exceptions=False,
    )

    results: dict[str, dict | None] = {}
    source_outcomes: dict[str, str] = {}
    for name, result, outcome in completed:
        results[name] = result
        source_outcomes[name] = outcome

    if not [source for source, result in results.items() if result is not None]:
        logger_obj.warning(
            "all_data_sources_failed",
            symbol=symbol,
            sources_attempted=list(builders.keys()),
            source_outcomes=source_outcomes,
            suspected_cause=classify_aggregate_source_failure(source_outcomes),
        )

    return results
