"""
Lightweight verification tools for the External Consultant agent.

Design: "Dumb Tool" pattern - tools do pure I/O only.
The Consultant handles all reasoning about whether data conflicts matter.

These tools give the consultant independent access to raw market data,
breaking the circular dependency where all agents rely on the same
Fundamentals Analyst interpretation.
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


@tool("spot_check_metric")
async def spot_check_metric(
    ticker: Annotated[str, "Stock ticker (e.g., 7203.T, 0005.HK)"],
    metric: Annotated[str, "Metric name to verify (e.g., trailingPE, debtToEquity)"],
) -> str:
    """
    Fetch a single financial metric directly from yfinance to verify
    a claim in the analyst reports. Use sparingly — only when you suspect
    a specific number is wrong or when DATA_BLOCK and narrative disagree.

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


def get_consultant_tools() -> list:
    """Get the list of tools available to the External Consultant."""
    return [spot_check_metric]
