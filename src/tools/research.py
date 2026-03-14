"""Foreign-source research tool implementations."""

import asyncio
from typing import Annotated

import structlog
import yfinance as yf
from langchain_core.tools import tool

from src.ticker_utils import normalize_ticker
from src.tools import shared

logger = structlog.get_logger(__name__)


@tool
async def search_foreign_sources(
    ticker: Annotated[str, "Stock ticker symbol"],
    search_query: Annotated[str, "Search query (can include native language terms)"],
) -> str:
    """
    Search for financial data from foreign-language and premium English sources.

    Use this tool to find official filings, IR pages, and premium source data
    that may not be available through standard English-language APIs.
    """
    try:
        normalized_symbol = normalize_ticker(ticker)
        ticker_obj = yf.Ticker(normalized_symbol)
        company_name = await shared.extract_company_name_async(ticker_obj)
        full_query = f"{search_query} {company_name} {ticker}"

        logger.info("foreign_source_search", ticker=ticker, query=full_query[:100])

        async def _noop():
            return None

        tavily_coro = (
            shared._tavily_search_with_timeout({"query": full_query})
            if shared.tavily_tool
            else _noop()
        )
        ddg_coro = shared._ddg_search(full_query, max_results=5)

        tavily_results, ddg_results = await asyncio.gather(
            tavily_coro, ddg_coro, return_exceptions=True
        )

        if isinstance(tavily_results, Exception):
            logger.warning("tavily_gather_error", error=str(tavily_results))
            tavily_results = None
        if isinstance(ddg_results, Exception):
            logger.debug("ddg_gather_error", error=str(ddg_results))
            ddg_results = []

        merged = shared._merge_search_results(tavily_results, ddg_results)
        if not merged:
            return f"No results found for foreign source search: {search_query}"

        results_str = shared._format_and_truncate_tavily_result(merged)

        sources_used = []
        if tavily_results and not isinstance(tavily_results, Exception):
            sources_used.append("Tavily")
        if ddg_results and isinstance(ddg_results, list) and len(ddg_results) > 0:
            sources_used.append("DuckDuckGo")
        source_note = f"Sources: {', '.join(sources_used)}" if sources_used else ""

        return f"""### Foreign Source Search Results
Query: {search_query}
Ticker: {ticker} ({company_name})
{source_note}

{results_str}

Note: Verify dates and currencies in the source data."""
    except Exception as exc:
        logger.error(f"Foreign source search error: {exc}")
        return f"Error searching foreign sources: {exc}"


@tool
async def get_official_filings(
    ticker: Annotated[str, "Stock ticker symbol (e.g., 2767.T, 005930.KS)"],
) -> str:
    """
    Fetch structured data from official filing APIs (EDINET for Japan,
    DART for Korea, Companies House for UK, etc.).
    """
    from src.data.filings import registry

    normalized = normalize_ticker(ticker)
    logger.info("official_filings_lookup", ticker=normalized)

    result = await registry.fetch(normalized)
    if result is None:
        return (
            f"No official filing API available for {normalized}. "
            "Use search_foreign_sources instead."
        )
    return result.to_report_string()
