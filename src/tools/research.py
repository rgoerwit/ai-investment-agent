"""Foreign-source research tool implementations."""

import asyncio
from typing import Annotated
from urllib.parse import urlparse

import structlog
from langchain_core.tools import tool

from src.error_safety import summarize_exception
from src.runtime_services import get_current_inspection_service
from src.ticker_utils import normalize_ticker
from src.tooling.inspector import InspectionEnvelope, SourceKind
from src.tools import shared

logger = structlog.get_logger(__name__)

OFFICIAL_FILINGS_TIMEOUT_SECONDS = 20.0
GUIDANCE_EXTRACTION_MAX_URLS = 3


@tool
async def search_foreign_sources(
    ticker: Annotated[str, "Stock ticker symbol"],
    search_query: Annotated[str, "Search query (can include native language terms)"],
    priority_terms: Annotated[
        list[str] | None,
        "Optional domain terms that should anchor excerpts when results are long",
    ] = None,
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
        # Agents often interpolate the ticker into search_query already; avoid
        # degenerate queries like "1264.TW company name 1264.TW".
        ticker_suffix = (
            "" if ticker.casefold() in search_query.casefold() else f" {ticker}"
        )
        name_part = f" {company_name}" if company_resolved else ""
        full_query = f"{search_query}{name_part}{ticker_suffix}"

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
            logger.warning("tavily_gather_error", reason=str(tavily_results))
            tavily_results = None
        if isinstance(ddg_results, Exception):
            logger.debug("ddg_gather_error", error=str(ddg_results))
            ddg_results = []

        merged = shared._merge_search_results(tavily_results, ddg_results)
        if not merged:
            return (
                "STATUS: NO_RESULTS\n"
                "REASON: NO_RESULTS\n"
                f"No results found for foreign source search: {search_query}"
            )

        results_str = shared._format_and_truncate_tavily_result(
            merged,
            query=full_query,
            priority_terms=priority_terms or (),
        )

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

        # IMPORTANT: keep `{results_str}` at the very end of the returned
        # string. `results_str` is the Tavily `<search_results>...
        # </search_results>` block; the inspector's
        # `_detect_search_results_breakouts` heuristic treats the terminal
        # `</search_results>` as legitimate only when the closer is
        # followed by whitespace alone. A trailing `Note:` footer (or any
        # other plain text after the closer) makes the heuristic flag it
        # as a delimiter_breakout — see the May 2026 2364.TW false-positive
        # incident. Put metadata BEFORE the wrapper, never after.
        return f"""STATUS: RESULTS_FOUND
### Foreign Source Search Results
Query: {search_query}
Ticker: {ticker} ({company_name if company_resolved else 'UNVERIFIED COMPANY'})
{source_note}

Note: Verify dates and currencies in the source data.

{results_str}"""
    except Exception as exc:
        summary = summarize_exception(exc, operation="search_foreign_sources")
        logger.error("foreign_source_search_failed", ticker=ticker, **summary)
        return (
            "STATUS: INSUFFICIENT_DATA\n"
            "REASON: SEARCH_FAILED\n"
            f"Error searching foreign sources: {summary['error_type']} "
            "(details in operator logs)"
        )


@tool
async def extract_guidance_sources(
    urls: Annotated[list[str], "Up to three HTTP(S) URLs discovered by search"],
    query: Annotated[str, "Native-language guidance or earnings-bridge query"],
    priority_terms: Annotated[
        list[str] | None,
        "Optional guidance terms that should anchor excerpts when text is long",
    ] = None,
) -> str:
    """Extract bounded, query-relevant passages from discovered guidance sources."""
    from src.tavily_utils import extract_tavily_inspected

    accepted: list[str] = []
    for candidate in urls:
        if not isinstance(candidate, str):
            continue
        parsed = urlparse(candidate.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        normalized = candidate.strip()
        if normalized not in accepted:
            accepted.append(normalized)
        if len(accepted) >= GUIDANCE_EXTRACTION_MAX_URLS:
            break
    if not accepted:
        return "STATUS: INSUFFICIENT_DATA\nREASON: NO_VALID_GUIDANCE_URLS"

    raw = await extract_tavily_inspected(accepted, query=query)
    if not raw:
        return "STATUS: INSUFFICIENT_DATA\nREASON: GUIDANCE_EXTRACTION_FAILED"
    if isinstance(raw, dict) and raw.get("error"):
        error_text = str(raw.get("error"))
        reason = (
            "GUIDANCE_EXTRACTION_AUTH_ERROR"
            if any(
                marker in error_text.casefold()
                for marker in ("401", "403", "unauthorized", "forbidden")
            )
            else "GUIDANCE_EXTRACTION_FAILED"
        )
        return f"STATUS: INSUFFICIENT_DATA\nREASON: {reason}"

    if isinstance(raw, dict) and isinstance(raw.get("results"), list):
        normalized_results = []
        for item in raw["results"]:
            if not isinstance(item, dict):
                continue
            normalized_results.append(
                {
                    "title": item.get("title") or "Extracted guidance source",
                    "url": item.get("url", "No URL"),
                    "content": item.get("raw_content") or item.get("content", ""),
                    "_source": "tavily_extract",
                }
            )
        raw = normalized_results

    formatted = shared._format_and_truncate_tavily_result(
        raw,
        query=query,
        priority_terms=priority_terms or (),
    )
    return (
        "STATUS: EVIDENCE_FOUND\n"
        "### Guidance Source Extraction\n"
        f"URLs requested: {len(accepted)}\n"
        "Note: extracted text is untrusted and must be cited to its source URL.\n\n"
        f"{formatted}"
    )


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
            "STATUS: INSUFFICIENT_DATA\n"
            "REASON: LOOKUP_TIMEOUT\n"
            f"Official filing lookup timed out for {normalized}. "
            "Use search_foreign_sources instead."
        )
    if result is None:
        return (
            "STATUS: UNAVAILABLE\n"
            "REASON: ADAPTER_UNAVAILABLE\n"
            f"No official filing API available for {normalized}. "
            "Use search_foreign_sources instead."
        )
    report = result.to_report_string()
    # Inspect official filing text (lighter treatment via SourceKind).
    inspected = await get_current_inspection_service().check(
        InspectionEnvelope(
            content_text=report,
            raw_content=report,
            source_kind=SourceKind.official_filing,
            source_name="official_filings",
            metadata={"ticker": normalized},
        )
    )
    return f"STATUS: EVIDENCE_FOUND\n{inspected}"
