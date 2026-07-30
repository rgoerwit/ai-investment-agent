"""Tool registry and concrete tool surface smoke tests."""

import inspect
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.legal import search_legal_tax_disclosures
from src.tools.market import (
    get_financial_metrics,
    get_fundamental_analysis,
    get_technical_indicators,
    get_yfinance_data,
)
from src.tools.news import (
    get_macroeconomic_news,
    get_news,
    get_social_media_sentiment,
)
from src.tools.ownership import get_ownership_structure
from src.tools.registry import Toolkit, toolkit
from src.tools.research import (
    extract_guidance_sources,
    get_official_filings,
    search_foreign_sources,
)


def test_public_tool_exports_support_ainvoke():
    public_tools = [
        get_financial_metrics,
        get_fundamental_analysis,
        get_macroeconomic_news,
        get_news,
        extract_guidance_sources,
        get_official_filings,
        get_ownership_structure,
        get_social_media_sentiment,
        get_technical_indicators,
        get_yfinance_data,
        search_foreign_sources,
        search_legal_tax_disclosures,
    ]

    for tool in public_tools:
        assert hasattr(tool, "ainvoke"), f"{tool} is missing .ainvoke()"


def test_toolkit_group_accessors_return_expected_tools():
    assert isinstance(toolkit, Toolkit)

    market_tool_names = {tool.name for tool in toolkit.get_market_tools()}
    assert {"get_yfinance_data", "get_technical_indicators"} <= market_tool_names

    news_tool_names = {tool.name for tool in toolkit.get_news_tools()}
    assert news_tool_names == {
        "get_news",
        "get_macroeconomic_news",
        "search_foreign_sources",
    }

    foreign_tool_names = {tool.name for tool in toolkit.get_foreign_language_tools()}
    assert foreign_tool_names == {
        "search_foreign_sources",
        "extract_guidance_sources",
        "get_official_filings",
        "get_official_document",
    }

    legal_tool_names = {tool.name for tool in toolkit.get_legal_tools()}
    assert legal_tool_names == {
        "search_legal_tax_disclosures",
        "search_foreign_sources",
        "get_official_filings",
        "get_official_document",
    }

    auditor_tool_names = [tool.name for tool in toolkit.get_auditor_tools()]
    assert auditor_tool_names.count("get_official_document") == 1


def test_get_macroeconomic_news_accepts_optional_region_param():
    signature = inspect.signature(get_macroeconomic_news.coroutine)
    assert "region" in signature.parameters
    assert signature.parameters["region"].default == ""


@pytest.mark.asyncio
async def test_get_macroeconomic_news_region_hint_reaches_tavily_query():
    with patch(
        "src.tools.shared._tavily_search_with_timeout",
        new=AsyncMock(return_value=None),
    ) as tavily_search:
        result = await get_macroeconomic_news.ainvoke(
            {"trade_date": "2026-04-18", "region": "JAPAN"}
        )

    assert "timed out or failed" in result
    called_query = tavily_search.await_args.args[0]["query"]
    assert "Japan" in called_query


def _tool_names(tools):
    return [getattr(t, "name", None) or getattr(t, "__name__", None) for t in tools]


def test_auditor_tools_have_no_duplicate_names():
    # get_auditor_tools splats overlapping groups; a duplicate function name in
    # one bound payload is rejected by strict OpenAI-compatible providers
    # (Moonshot/Kimi 400 "function name ... is duplicated"). Dedup keeps one.
    names = _tool_names(toolkit.get_auditor_tools())
    assert len(names) == len(set(names)), f"duplicate auditor tool names: {names}"
    assert names.count("search_foreign_sources") == 1


def test_no_grouped_tool_set_binds_duplicate_names():
    tk = Toolkit()
    for getter in (
        tk.get_auditor_tools,
        tk.get_news_tools,
        tk.get_foreign_language_tools,
        tk.get_legal_tools,
        tk.get_technical_tools,
        tk.get_junior_fundamental_tools,
    ):
        names = _tool_names(getter())
        assert len(names) == len(set(names)), f"{getter.__name__} duplicates: {names}"
