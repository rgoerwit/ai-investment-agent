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

logger = structlog.get_logger(__name__)

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
        info = await asyncio.to_thread(lambda: stock.info)
        value = info.get(metric)

        return json.dumps(
            {
                "ticker": ticker,
                "metric": metric,
                "value": value,
                "source": "yfinance_direct",
            }
        )
    except Exception as e:
        logger.warning("spot_check_failed", ticker=ticker, metric=metric, error=str(e))
        return json.dumps({"error": str(e), "ticker": ticker, "metric": metric})


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
        from src.data.fmp_fetcher import get_fmp_fetcher

        fmp = get_fmp_fetcher()
        if not fmp.is_available():
            return json.dumps(
                {
                    "error": "FMP alt-source unavailable (no API key configured)",
                    "suggestion": "spot_check_metric uses yfinance as primary — same source as DATA_BLOCK pipeline",
                    "ticker": ticker,
                    "metric": metric,
                }
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

    except ValueError as e:
        # FMP API key validation error
        logger.warning(
            "spot_check_alt_key_error",
            ticker=ticker,
            metric=metric,
            error=str(e),
        )
        return json.dumps(
            {
                "error": f"FMP API key issue: {e}",
                "ticker": ticker,
                "metric": metric,
            }
        )
    except Exception as e:
        logger.warning(
            "spot_check_alt_failed",
            ticker=ticker,
            metric=metric,
            error=str(e),
        )
        return json.dumps({"error": str(e), "ticker": ticker, "metric": metric})


def get_consultant_tools() -> list:
    """Get the list of tools available to the External Consultant.

    DELIBERATELY excludes spot_check_metric (yfinance) because the main
    pipeline already uses yfinance — verifying yfinance against yfinance is
    circular validation. The consultant gets only independent sources:
    - spot_check_metric_alt: FMP (independent of pipeline)
    - get_official_filings: Official filing APIs (EDINET/DART) for ground-truth
    """
    from src.toolkit import get_official_filings

    return [spot_check_metric_alt, get_official_filings]
