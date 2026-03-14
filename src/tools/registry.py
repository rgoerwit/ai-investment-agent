"""Toolkit registry and grouped tool accessors."""

from src.data.fetcher import fetcher as market_data_fetcher
from src.enhanced_sentiment_toolkit import get_multilingual_sentiment_search
from src.liquidity_calculation_tool import calculate_liquidity_metrics
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
from src.tools.research import get_official_filings, search_foreign_sources


class Toolkit:
    def __init__(self):
        self.market_data_fetcher = market_data_fetcher

    def get_core_tools(self):
        return [get_yfinance_data, get_technical_indicators]

    def get_technical_tools(self):
        return [
            get_yfinance_data,
            get_technical_indicators,
            calculate_liquidity_metrics,
        ]

    def get_market_tools(self):
        return self.get_technical_tools()

    def get_junior_fundamental_tools(self):
        """Tools for Junior Fundamentals Analyst (data gathering)."""
        return [get_financial_metrics, get_fundamental_analysis]

    def get_senior_fundamental_tools(self):
        """Senior Fundamentals Analyst has NO tools - receives data from Junior."""
        return []

    def get_fundamental_tools(self):
        return self.get_junior_fundamental_tools()

    def get_sentiment_tools(self):
        return [get_social_media_sentiment, get_multilingual_sentiment_search]

    def get_news_tools(self):
        return [get_news, get_macroeconomic_news]

    def get_foreign_language_tools(self):
        """Tools for Foreign Language Analyst (supplemental data from native sources)."""
        return [search_foreign_sources, get_official_filings]

    def get_legal_tools(self):
        """Tools for Legal Counsel (PFIC/VIE detection for US investors)."""
        return [search_legal_tax_disclosures]

    def get_value_trap_tools(self):
        """Tools for Value Trap Detector (governance & capital allocation analysis)."""
        return [
            get_ownership_structure,
            get_news,
            search_foreign_sources,
            get_official_filings,
        ]

    def get_all_tools(self):
        return [
            get_yfinance_data,
            get_technical_indicators,
            get_financial_metrics,
            get_news,
            get_social_media_sentiment,
            get_multilingual_sentiment_search,
            calculate_liquidity_metrics,
            get_macroeconomic_news,
            get_fundamental_analysis,
            search_foreign_sources,
            search_legal_tax_disclosures,
            get_ownership_structure,
            get_official_filings,
        ]


toolkit = Toolkit()
