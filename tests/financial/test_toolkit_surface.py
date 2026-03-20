"""Public toolkit surface smoke tests."""

from src.toolkit import (
    Toolkit,
    get_financial_metrics,
    get_fundamental_analysis,
    get_macroeconomic_news,
    get_news,
    get_official_filings,
    get_ownership_structure,
    get_social_media_sentiment,
    get_technical_indicators,
    get_yfinance_data,
    search_foreign_sources,
    search_legal_tax_disclosures,
    toolkit,
)


def test_public_tool_exports_support_ainvoke():
    public_tools = [
        get_financial_metrics,
        get_fundamental_analysis,
        get_macroeconomic_news,
        get_news,
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
    assert news_tool_names == {"get_news", "get_macroeconomic_news"}

    foreign_tool_names = {tool.name for tool in toolkit.get_foreign_language_tools()}
    assert foreign_tool_names == {"search_foreign_sources", "get_official_filings"}

    legal_tool_names = {tool.name for tool in toolkit.get_legal_tools()}
    assert legal_tool_names == {"search_legal_tax_disclosures"}
