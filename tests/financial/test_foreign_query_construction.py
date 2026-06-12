"""search_foreign_sources query construction.

Regression for degenerate queries like "1264.TW company name 1264.TW": agents
often interpolate the ticker into search_query themselves, and the tool used
to unconditionally append it again when the company name was unresolved.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.tools.research import search_foreign_sources


async def _captured_query(ticker: str, search_query: str, resolved_name: str) -> str:
    """Invoke the tool with mocked search backends; return the query used."""
    ddg = AsyncMock(return_value=[])
    with (
        patch(
            "src.tools.shared.extract_company_name_async",
            new=AsyncMock(return_value=resolved_name),
        ),
        patch("src.tools.shared._ddg_search", new=ddg),
        patch("src.tools.shared.tavily_tool", None),
    ):
        await search_foreign_sources.ainvoke(
            {"ticker": ticker, "search_query": search_query}
        )
    return ddg.call_args.args[0]


@pytest.mark.asyncio
async def test_ticker_not_appended_when_already_in_query():
    # Unresolved name (resolver echoes the ticker back)
    query = await _captured_query("1264.TW", "1264.TW company name", "1264.TW")
    assert query == "1264.TW company name"


@pytest.mark.asyncio
async def test_ticker_appended_when_absent():
    query = await _captured_query("7203.T", "決算短信 業績", "7203.T")
    assert query == "決算短信 業績 7203.T"


@pytest.mark.asyncio
async def test_resolved_name_included_without_duplicate_ticker():
    query = await _captured_query("7203.T", "7203.T 決算短信", "Toyota Motor")
    assert query == "7203.T 決算短信 Toyota Motor"


@pytest.mark.asyncio
async def test_lowercase_ticker_in_query_not_double_suffixed():
    """LLM-written queries may contain the ticker in lowercase."""
    query = await _captured_query("1264.TW", "1264.tw company filings", "1264.TW")
    assert query == "1264.tw company filings"
