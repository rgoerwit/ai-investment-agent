"""
Lightweight verification tools for the External Consultant agent.

Design: "Dumb Tool" pattern - tools do pure I/O only.
The Consultant handles all reasoning about whether data conflicts matter.

These tools give the consultant independent access to raw market data,
breaking the circular dependency where all agents rely on the same
Fundamentals Analyst interpretation.

Two spot-check tools:
- spot_check_metric: Fetches from yfinance (same source as DATA_BLOCK pipeline)
- spot_check_metric_alt: Fetches from FMP (independent source for cross-validation)
"""

import asyncio
import json
from typing import Annotated

import structlog
import yfinance as yf
from langchain_core.tools import tool

from src.error_safety import safe_error_payload, summarize_exception
from src.config import config
from src.runtime_diagnostics import classify_failure
from src.runtime_services import get_current_runtime_services

logger = structlog.get_logger(__name__)

SPOT_CHECK_TIMEOUT_SECONDS = 8.0

# Fields the consultant is allowed to spot-check (decision-critical metrics only)
ALLOWED_FIELDS = frozenset(
    {
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "debtToEquity",
        "returnOnEquity",
        "returnOnAssets",
        "operatingMargins",
        "freeCashflow",
        "operatingCashflow",
        "totalRevenue",
        "netIncomeToCommon",
        "currentPrice",
        "marketCap",
        "dividendYield",
        "payoutRatio",
        "currentRatio",
        "earningsGrowth",
        "revenueGrowth",
    }
)

# Map yfinance field names → FMP API endpoints/fields
FMP_FIELD_MAP: dict[str, tuple[str, str]] = {
    "operatingCashflow": ("cash-flow-statement", "operatingCashFlow"),
    "freeCashflow": ("cash-flow-statement", "freeCashFlow"),
    "netIncomeToCommon": ("income-statement", "netIncome"),
    "totalRevenue": ("income-statement", "revenue"),
    "returnOnEquity": ("ratios", "returnOnEquity"),
    "returnOnAssets": ("ratios", "returnOnAssets"),
    "debtToEquity": ("ratios", "debtEquityRatio"),
    "trailingPE": ("ratios", "priceEarningsRatio"),
    "priceToBook": ("ratios", "priceToBookRatio"),
    "payoutRatio": ("ratios", "payoutRatio"),
    "currentRatio": ("ratios", "currentRatio"),
    "dividendYield": ("ratios", "dividendYield"),
}


def _build_fmp_access_failure(
    *,
    ticker: str,
    metric: str,
    error: str,
    suggestion: str,
    retryable: bool,
    cooldown_until: str | None = None,
) -> str:
    payload = {
        "error": error,
        "suggestion": suggestion,
        "ticker": ticker,
        "metric": metric,
        "provider": "fmp",
        "failure_kind": "auth_error" if not retryable else "rate_limit",
        "retryable": retryable,
    }
    if cooldown_until is not None:
        payload["cooldown_until"] = cooldown_until
    return json.dumps(payload)


@tool("spot_check_metric")
async def spot_check_metric(
    ticker: Annotated[str, "Stock ticker (e.g., 7203.T, 0005.HK)"],
    metric: Annotated[str, "Metric name to verify (e.g., trailingPE, debtToEquity)"],
) -> str:
    """
    Fetch a single financial metric directly from yfinance to verify
    a claim in the analyst reports. Use sparingly — only when you suspect
    a specific number is wrong or when DATA_BLOCK and narrative disagree.

    NOTE: This uses the SAME data source (yfinance) as the main pipeline.
    For independent cross-validation, use spot_check_metric_alt (FMP source).

    Returns: JSON with {ticker, metric, value, source} or error.
    """
    if metric not in ALLOWED_FIELDS:
        return json.dumps(
            {
                "error": f"Unknown metric '{metric}'",
                "allowed": sorted(ALLOWED_FIELDS),
            }
        )

    try:
        stock = yf.Ticker(ticker)
        info = await asyncio.wait_for(
            asyncio.to_thread(lambda: stock.info),
            timeout=SPOT_CHECK_TIMEOUT_SECONDS,
        )
        value = info.get(metric)

        return json.dumps(
            {
                "ticker": ticker,
                "metric": metric,
                "value": value,
                "source": "yfinance_direct",
            }
        )
    except asyncio.TimeoutError:
        logger.warning(
            "spot_check_timeout",
            ticker=ticker,
            metric=metric,
            timeout_seconds=SPOT_CHECK_TIMEOUT_SECONDS,
        )
        return json.dumps(
            {
                "error": "Timed out loading yfinance info",
                "ticker": ticker,
                "metric": metric,
            }
        )
    except Exception as e:
        summary = summarize_exception(
            e,
            operation="spot_check_metric",
            provider="unknown",
        )
        logger.warning("spot_check_failed", ticker=ticker, metric=metric, **summary)
        return json.dumps(
            safe_error_payload(
                e,
                operation="spot_check_metric",
                provider="unknown",
                extra={"ticker": ticker, "metric": metric},
            )
        )


@tool("spot_check_metric_alt")
async def spot_check_metric_alt(
    ticker: Annotated[str, "Stock ticker (e.g., 7203.T, 0005.HK)"],
    metric: Annotated[
        str, "Metric name to verify (e.g., operatingCashflow, netIncomeToCommon)"
    ],
) -> str:
    """
    Fetch a single financial metric from Financial Modeling Prep (FMP) as an
    INDEPENDENT alternative source. Use this to cross-validate suspicious values
    from the DATA_BLOCK, which uses yfinance as its primary source.

    Priority metrics for cross-validation: operatingCashflow, freeCashflow,
    netIncomeToCommon (most prone to data source errors for ex-US stocks).

    Returns: JSON with {ticker, metric, value, source: "fmp_direct", fmp_field} or error.
    """
    if metric not in FMP_FIELD_MAP:
        return json.dumps(
            {
                "error": f"Metric '{metric}' not available via FMP alt-source",
                "available_metrics": sorted(FMP_FIELD_MAP.keys()),
                "suggestion": "Use spot_check_metric (yfinance) for this metric instead",
            }
        )

    try:
        from src.data.fmp_fetcher import (
            FMPSubscriptionUnavailableError,
            get_fmp_fetcher,
        )

        fmp = get_fmp_fetcher()
        if not fmp.is_available():
            cooldown_until = getattr(fmp, "_cooldown_until", None)
            if not getattr(fmp, "api_key", None):
                return _build_fmp_access_failure(
                    ticker=ticker,
                    metric=metric,
                    error="FMP alt-source unavailable (no API key configured)",
                    suggestion="spot_check_metric uses yfinance as primary — same source as DATA_BLOCK pipeline",
                    retryable=False,
                )
            return _build_fmp_access_failure(
                ticker=ticker,
                metric=metric,
                error="FMP alt-source temporarily unavailable (cooldown active after quota/rate-limit response)",
                suggestion="Retry later or rely on official filings / primary data until FMP cooldown expires",
                retryable=True,
                cooldown_until=cooldown_until.isoformat() if cooldown_until else None,
            )

        endpoint, fmp_field = FMP_FIELD_MAP[metric]

        # FMP uses plain ticker symbols for most exchanges
        # Exchange suffix mapping for FMP compatibility
        fmp_ticker = ticker
        # FMP uses .T for Tokyo but some need no suffix changes
        # Most international tickers work as-is with FMP

        # FMPFetcher manages its own aiohttp sessions per _get() call —
        # no async context manager needed (or implemented) at the fetcher level.
        data = await fmp._get(endpoint, {"symbol": fmp_ticker, "limit": 1})

        if not data or not isinstance(data, list) or len(data) == 0:
            return json.dumps(
                {
                    "ticker": ticker,
                    "metric": metric,
                    "value": None,
                    "source": "fmp_direct",
                    "fmp_field": fmp_field,
                    "note": "No data returned by FMP for this ticker/metric",
                }
            )

        value = data[0].get(fmp_field)

        return json.dumps(
            {
                "ticker": ticker,
                "metric": metric,
                "value": value,
                "source": "fmp_direct",
                "fmp_field": fmp_field,
                "fmp_endpoint": endpoint,
                "period": data[0].get("period", "Annual"),
            }
        )

    except FMPSubscriptionUnavailableError as e:
        summary = summarize_exception(
            e,
            operation="spot_check_metric_alt",
            provider="unknown",
        )
        logger.debug(
            "spot_check_alt_subscription_unavailable",
            ticker=ticker,
            metric=metric,
            **summary,
        )
        return _build_fmp_access_failure(
            ticker=ticker,
            metric=metric,
            error="FMP alt-source unavailable for this ticker or endpoint",
            suggestion="The current FMP plan does not cover this ticker or endpoint. Use official filings or another primary source instead.",
            retryable=False,
        )
    except ValueError as e:
        # FMP API key validation error
        summary = summarize_exception(
            e,
            operation="spot_check_metric_alt",
            provider="unknown",
        )
        logger.warning(
            "spot_check_alt_key_error",
            ticker=ticker,
            metric=metric,
            **summary,
        )
        return _build_fmp_access_failure(
            ticker=ticker,
            metric=metric,
            error="FMP API key issue",
            suggestion="Check FMP API credentials or use official filings if independent cross-validation is still needed.",
            retryable=False,
        )
    except Exception as e:
        details = classify_failure(e, provider="unknown", model_name="fmp_alt_source")
        safe_payload = safe_error_payload(
            e,
            operation="spot_check_metric_alt",
            provider="unknown",
            extra={
                "ticker": ticker,
                "metric": metric,
                "provider": "fmp",
                "fmp_endpoint": FMP_FIELD_MAP[metric][0],
            },
        )
        logger.warning(
            "spot_check_alt_failed",
            ticker=ticker,
            metric=metric,
            failure_kind=details.kind,
            retryable=details.retryable,
            error_type=details.error_type,
            message_preview=safe_payload.get("message_preview"),
        )
        return json.dumps(safe_payload)


@tool("spot_check_metric_mcp_fmp")
async def spot_check_metric_mcp_fmp(
    ticker: Annotated[str, "Stock ticker (e.g., 7203.T, 0005.HK)"],
    metric: Annotated[str, "Metric name (e.g., trailingPE, debtToEquity)"],
) -> str:
    """Fetch a single financial metric from the **official FMP MCP** server.

    This is an independent, MCP‑based cross‑check that does not rely on
    the main pipeline's yfinance‑driven DATA_BLOCK.  Returns compact JSON.

    Use when you suspect a specific number in the analyst reports is wrong
    or when DATA_BLOCK and narrative claims diverge.
    """
    _METRIC_TO_MCP_TOOL = {
        "trailingPE": "ratios",
        "forwardPE": "ratios",
        "priceToBook": "ratios",
        "debtToEquity": "ratios",
        "returnOnEquity": "ratios",
        "returnOnAssets": "ratios",
        "operatingMargins": "ratios",
        "dividendYield": "ratios",
        "payoutRatio": "ratios",
        "currentRatio": "ratios",
        "freeCashflow": "cash-flow-statement",
        "operatingCashflow": "cash-flow-statement",
        "totalRevenue": "income-statement",
        "netIncomeToCommon": "income-statement",
        "currentPrice": "quote",
        "marketCap": "quote",
    }

    tool_name = _METRIC_TO_MCP_TOOL.get(metric, "ratios")
    services = get_current_runtime_services()
    if services is None or services.mcp_runtime is None:
        return json.dumps({"error": "mcp_not_available", "ticker": ticker, "metric": metric})

    try:
        result_raw = await services.mcp_runtime.call_tool(
            "fmp_remote", tool_name, {"symbol": ticker}, agent_key="consultant"
        )
    except Exception as exc:
        return json.dumps({"error": str(exc), "ticker": ticker, "metric": metric, "source": "fmp_mcp"})

    try:
        data = json.loads(result_raw)
        payload = data.get("result") if isinstance(data, dict) else data
        if isinstance(payload, dict):
            value = payload.get(metric)
            return json.dumps(
                {"ticker": ticker, "metric": metric, "value": value, "source": "fmp_mcp"}
            )
    except Exception:
        pass
    return result_raw


@tool("spot_check_price_or_indicator_mcp_twelvedata")
async def spot_check_price_or_indicator_mcp_twelvedata(
    ticker: Annotated[str, "Stock ticker (e.g., 7203.T, 0005.HK)"],
    query: Annotated[
        str, "One of 'price', 'volume', 'rsi', 'macd', or 'quote'"
    ],
) -> str:
    """Fetch price / volume / technical indicator from **Twelve Data MCP**.

    Use only to verify a specific quote or indicator when the main pipeline
    and FMP disagree.  Never enable the “u‑tool” for consultant checks.

    Returns compact JSON.
    """
    services = get_current_runtime_services()
    if services is None or services.mcp_runtime is None:
        return json.dumps(
            {"error": "mcp_not_available", "ticker": ticker, "query": query}
        )

    allowed_tools = {"price", "quote"}
    tool_name = query if query in allowed_tools else "quote"
    try:
        result_raw = await services.mcp_runtime.call_tool(
            "twelvedata_remote", tool_name, {"symbol": ticker}, agent_key="consultant"
        )
    except Exception as exc:
        return json.dumps(
            {"error": str(exc), "ticker": ticker, "query": query, "source": "twelvedata_mcp"}
        )
    return result_raw


def get_consultant_tools() -> list:
    """Get the list of tools available to the External Consultant.

    DELIBERATELY excludes spot_check_metric (yfinance) because the main
    pipeline already uses yfinance — verifying yfinance against yfinance is
    circular validation. The consultant gets only independent sources:
    - spot_check_metric_alt: FMP REST (independent of pipeline)
    - get_official_filings: Official filing APIs (EDINET/DART) for ground-truth
    - spot_check_metric_mcp_fmp: FMP via MCP (broader tool surface, same vendor)
    - spot_check_price_or_indicator_mcp_twelvedata: Twelve Data via MCP (pinned endpoints)
    """
    from src.tools.research import get_official_filings

    tools: list = [
        spot_check_metric_alt,
        get_official_filings,
    ]

    try:
        services = get_current_runtime_services()
    except Exception:
        services = None

    if (
        services is not None
        and services.mcp_runtime is not None
        and config.consultant_mcp_enabled
    ):
        tools.append(spot_check_metric_mcp_fmp)
        tools.append(spot_check_price_or_indicator_mcp_twelvedata)

    return tools
