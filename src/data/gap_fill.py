"""Gap-fill helpers for missing market and fundamentals data."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
import yfinance as yf

from src.data.merge_policy import CRITICAL_ANALYSIS_FIELDS, SOURCE_QUALITY
from src.runtime_services import get_current_inspection_service
from src.tavily_utils import search_tavily_inspected
from src.ticker_utils import generate_strict_search_query
from src.tooling.inspector import InspectionEnvelope, SourceKind

logger = structlog.get_logger(__name__)


def calculate_coverage(data: dict[str, Any], important_fields: list[str]) -> float:
    """Calculate percentage of important fields present."""
    if not data:
        return 0.0
    present = sum(1 for field in important_fields if data.get(field) is not None)
    return present / len(important_fields) if important_fields else 0.0


def identify_critical_gaps(data: dict[str, Any]) -> list[str]:
    """Return the critical analysis fields still missing from the merged payload."""
    return [
        field
        for field in CRITICAL_ANALYSIS_FIELDS
        if field not in data or data[field] is None
    ]


async def fetch_tavily_gaps(
    fetcher: Any,
    symbol: str,
    missing_fields: list[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Fetch a limited set of missing safe fields from Tavily-inspected search results."""
    return await _fetch_tavily_gaps_impl(fetcher, symbol, missing_fields, **kwargs)


async def _fetch_tavily_gaps_impl(
    fetcher: Any,
    symbol: str,
    missing_fields: list[str],
    *,
    yf_module: Any = yf,
    asyncio_module: Any = asyncio,
    search_fn=search_tavily_inspected,
    query_builder=generate_strict_search_query,
) -> dict[str, Any]:
    """Fetch a limited set of missing safe fields from Tavily-inspected search results."""
    dangerous_fields = [
        "trailingPE",
        "forwardPE",
        "pegRatio",
        "currentPrice",
        "marketCap",
    ]
    safe_missing_fields = [
        field for field in missing_fields if field not in dangerous_fields
    ]
    if "us_revenue_pct" in missing_fields or "geographic_revenue" in missing_fields:
        safe_missing_fields.append("us_revenue_pct")

    if not fetcher.tavily_client or not safe_missing_fields:
        return {}

    try:
        ticker_obj = yf_module.Ticker(symbol)
        info = await asyncio_module.wait_for(
            asyncio_module.to_thread(lambda: ticker_obj.info),
            timeout=5,
        )
        company_name = info.get("longName") or info.get("shortName") or symbol
    except Exception:
        company_name = symbol

    fields_to_search = safe_missing_fields[:5]
    search_results: dict[str, str] = {}
    field_terms = {
        "trailingPE": "trailing P/E ratio price earnings",
        "forwardPE": "forward P/E ratio estimate",
        "priceToBook": "price to book ratio P/B",
        "returnOnEquity": "ROE return on equity",
        "debtToEquity": "debt to equity ratio leverage",
        "numberOfAnalystOpinions": "analyst coverage count",
        "revenueGrowth": "revenue growth year over year",
    }

    for field in fields_to_search:
        if field == "us_revenue_pct":
            query = f'"{company_name}" annual report revenue by geography North America United States'
        else:
            query = query_builder(symbol, company_name, field_terms.get(field, field))
        try:
            result = await search_fn(query, profile="finance_deep", timeout=5)
            if isinstance(result, dict) and "results" in result:
                search_results[field] = "\n".join(
                    item.get("content", "") for item in result["results"]
                )
        except (TimeoutError, asyncio_module.TimeoutError, Exception):
            pass

    if not search_results:
        return {}

    all_text = "\n\n".join(search_results.values())
    envelope = InspectionEnvelope(
        content_text=all_text,
        source_kind=SourceKind.web_search,
        source_name="tavily",
        metadata={"symbol": symbol, "fields": list(search_results.keys())},
    )
    inspected_text = await get_current_inspection_service().check(envelope)
    return fetcher.pattern_extractor.extract_from_text(
        inspected_text, skip_fields=set()
    )


def merge_gap_fill_data(
    merged: dict[str, Any],
    gap_fill_data: dict[str, Any],
    merge_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Merge gap-fill fields conservatively behind existing higher-quality data."""
    tavily_quality = SOURCE_QUALITY["tavily_extraction"]
    added = 0
    for key, value in gap_fill_data.items():
        if value is None:
            continue
        should_use = False
        if key not in merged or merged[key] is None:
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
