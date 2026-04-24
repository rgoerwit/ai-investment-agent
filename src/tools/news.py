"""News and sentiment tool implementations."""

import asyncio
from typing import Annotated

import structlog
from langchain_core.tools import tool

from src.error_safety import format_error_message, summarize_exception
from src.runtime_services import get_current_inspection_service
from src.stocktwits_api import StockTwitsAPI
from src.ticker_utils import normalize_ticker
from src.tooling.inspector import InspectionEnvelope, SourceKind
from src.tools import shared

logger = structlog.get_logger(__name__)
stocktwits_api = StockTwitsAPI()


@tool
async def get_news(
    ticker: Annotated[
        str,
        "Exact ticker with exchange suffix — e.g. '3217.TWO' (Taiwan OTC/TPEx),"
        " '7203.T' (Japan), '0005.HK' (HK), '2330.TW' (Taiwan TWSE)."
        " Use exactly as provided; never alter or drop the suffix.",
    ],
    search_query: Annotated[str, "Specific query"] = None,
) -> str:
    """
    Get recent news using Tavily with ENHANCED multi-query strategy.
    Structures output for News Analyst prompt ingestion.
    """
    if not shared.tavily_tool:
        return "News tool unavailable."

    try:
        normalized_symbol = normalize_ticker(ticker)
        company_name = await shared.extract_company_name_async(normalized_symbol)
        company_resolved = company_name != normalized_symbol
        query_anchor = f'"{company_name}"' if company_resolved else ticker
        display_name = company_name if company_resolved else ticker

        local_source_hints = {
            ".KS": "site:pulsenews.co.kr OR site:koreatimes.co.kr OR site:koreaherald.com OR site:mk.co.kr",
            ".HK": "site:scmp.com OR site:thestandard.com.hk OR site:ejinsight.com OR site:aastocks.com OR site:etnet.com.hk",
            ".T": "site:japantimes.co.jp OR site:nikkei.com OR site:minkabu.jp OR site:kabutan.jp",
            ".L": "site:ft.com OR site:bbc.co.uk/news/business",
            ".PA": "site:france24.com OR site:lemonde.fr",
            ".DE": "site:dw.com OR site:handelsblatt.com",
        }

        suffix = ""
        if "." in normalized_symbol:
            suffix = "." + normalized_symbol.split(".")[-1]
        local_hint = local_source_hints.get(suffix, "")

        results = []

        general_query = (
            f"{query_anchor} {search_query}"
            if search_query
            else f"{query_anchor} (earnings OR merger OR acquisition OR regulatory)"
        )
        general_result = await shared._tavily_search_with_timeout(
            {"query": general_query}
        )
        if general_result:
            formatted = shared._format_and_truncate_tavily_result(general_result)
            if formatted.strip():
                results.append(f"=== GENERAL NEWS ===\n{formatted}\n")

        if local_hint and not search_query:
            local_query = (
                f"{query_anchor} {local_hint} (earnings OR guidance OR strategy)"
            )
            local_result = await shared._tavily_search_with_timeout(
                {"query": local_query}
            )
            if local_result:
                formatted_local = shared._format_and_truncate_tavily_result(
                    local_result
                )
                if formatted_local.strip():
                    results.append(
                        f"=== LOCAL/REGIONAL NEWS SOURCES ===\n{formatted_local}\n"
                    )

        if not results:
            return f"No news found for {display_name}."

        return f"News Results for {display_name}:\n\n" + "\n".join(results)
    except Exception as exc:
        summary = summarize_exception(
            exc,
            operation="get_news",
            provider="unknown",
        )
        logger.error(
            "news_fetch_failed",
            ticker=ticker,
            **summary,
        )
        return format_error_message(
            operation="get_news",
            error_type=summary["error_type"],
            message_preview=summary["message_preview"],
        )


@tool
async def get_social_media_sentiment(ticker: str) -> str:
    """Get sentiment from StockTwits."""
    try:
        data = await stocktwits_api.get_sentiment(ticker)

        if "error" in data:
            return f"StockTwits Sentiment Error: {data.get('error')}"

        # Inspect user-generated message content before including in summary.
        raw_messages = data.get("messages", [])
        inspection_tasks = []
        for msg in raw_messages:
            msg_text = str(msg)
            envelope = InspectionEnvelope(
                content_text=msg_text,
                source_kind=SourceKind.social_feed,
                source_name="stocktwits",
                metadata={"ticker": ticker},
            )
            inspection_tasks.append(get_current_inspection_service().check(envelope))

        inspected_messages = (
            await asyncio.gather(*inspection_tasks) if inspection_tasks else []
        )

        summary = (
            f"StockTwits Sentiment for {data.get('ticker')}:\n"
            f"- Bullish: {data.get('bullish_pct')}% ({data.get('bullish_count')} msgs)\n"
            f"- Bearish: {data.get('bearish_pct')}% ({data.get('bearish_count')} msgs)\n"
            f"- Total Volume (last 30): {data.get('total_messages_last_30')}\n\n"
            "Sample Messages:\n"
        )

        if inspected_messages:
            for msg in inspected_messages:
                summary += f"- {msg}\n"
        else:
            summary += "- No recent messages found.\n"

        return summary
    except Exception as exc:
        summary = summarize_exception(
            exc,
            operation="get_social_media_sentiment",
            provider="unknown",
        )
        logger.warning(
            "social_media_sentiment_failed",
            ticker=ticker,
            **summary,
        )
        return format_error_message(
            operation="get_social_media_sentiment",
            error_type=summary["error_type"],
            message_preview=summary["message_preview"],
        )


@tool
async def get_macroeconomic_news(trade_date: str, region: str = "") -> str:
    """Get raw macroeconomic news context for a date and optional region bucket."""
    if not shared.tavily_tool:
        return "Tool unavailable"

    query = (
        f"global macro market conditions {trade_date} "
        "central bank rates inflation CPI PMI GDP FX currency "
        "bond yields liquidity credit spreads risk appetite equity market"
    )
    if region:
        from src.macro_regions import query_hint_for_macro_region

        hint = query_hint_for_macro_region(region)
        query = (
            f"{hint} market conditions {trade_date} "
            "central bank rates inflation CPI PMI GDP FX currency "
            "bond yields liquidity credit spreads risk appetite equity market"
        )

    result = await shared._tavily_search_with_timeout({"query": query})
    if not result:
        return "Macroeconomic news search timed out or failed."
    return shared._format_and_truncate_tavily_result(result)
