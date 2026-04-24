"""Foreign-source research tool implementations."""

import asyncio
from typing import Annotated

import structlog
from langchain_core.tools import tool

from src.runtime_services import get_current_inspection_service
from src.ticker_utils import normalize_ticker
from src.tooling.inspector import InspectionEnvelope, SourceKind
from src.tools import shared

logger = structlog.get_logger(__name__)

OFFICIAL_FILINGS_TIMEOUT_SECONDS = 20.0


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
        company_name = await shared.extract_company_name_async(normalized_symbol)
        company_resolved = company_name != normalized_symbol
        full_query = (
            f"{search_query} {company_name} {ticker}"
            if company_resolved
            else f"{search_query} {ticker}"
        )

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

        # Inspect merged foreign-search output after DDG+Tavily merge.
        results_str = await get_current_inspection_service().check(
            InspectionEnvelope(
                content_text=results_str,
                raw_content=results_str,
                source_kind=SourceKind.web_search,
                source_name="foreign_search_merged",
                metadata={"ticker": ticker, "query": search_query[:100]},
            )
        )

        sources_used = []
        if tavily_results and not isinstance(tavily_results, Exception):
            sources_used.append("Tavily")
        if ddg_results and isinstance(ddg_results, list) and len(ddg_results) > 0:
            sources_used.append("DuckDuckGo")
        source_note = f"Sources: {', '.join(sources_used)}" if sources_used else ""

        return f"""### Foreign Source Search Results
Query: {search_query}
Ticker: {ticker} ({company_name if company_resolved else 'UNVERIFIED COMPANY'})
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

    try:
        result = await asyncio.wait_for(
            registry.fetch(normalized),
            timeout=OFFICIAL_FILINGS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "official_filings_timeout",
            ticker=normalized,
            timeout_seconds=OFFICIAL_FILINGS_TIMEOUT_SECONDS,
        )
        return (
            f"Official filing lookup timed out for {normalized}. "
            "Use search_foreign_sources instead."
        )
    if result is None:
        return (
            f"No official filing API available for {normalized}. "
            "Use search_foreign_sources instead."
        )
    report = result.to_report_string()
    # Inspect official filing text (lighter treatment via SourceKind).
    return await get_current_inspection_service().check(
        InspectionEnvelope(
            content_text=report,
            raw_content=report,
            source_kind=SourceKind.official_filing,
            source_name="official_filings",
            metadata={"ticker": normalized},
        )
    )
