"""Toolkit registry and grouped tool accessors."""


class Toolkit:
    def __init__(self):
        pass

    def get_core_tools(self):
        from src.tools.market import get_technical_indicators, get_yfinance_data

        return [get_yfinance_data, get_technical_indicators]

    def get_technical_tools(self):
        from src.liquidity_calculation_tool import calculate_liquidity_metrics
        from src.tools.market import get_technical_indicators, get_yfinance_data

        return [
            get_yfinance_data,
            get_technical_indicators,
            calculate_liquidity_metrics,
        ]

    def get_market_tools(self):
        return self.get_technical_tools()

    def get_junior_fundamental_tools(self):
        """Tools for Junior Fundamentals Analyst (data gathering)."""
        from src.tools.market import get_financial_metrics, get_fundamental_analysis

        return [get_financial_metrics, get_fundamental_analysis]

    def get_senior_fundamental_tools(self):
        """Senior Fundamentals Analyst has NO tools - receives data from Junior."""
        return []

    def get_fundamental_tools(self):
        return self.get_junior_fundamental_tools()

    def get_sentiment_tools(self):
        from src.enhanced_sentiment_toolkit import get_multilingual_sentiment_search
        from src.tools.news import get_social_media_sentiment

        return [get_social_media_sentiment, get_multilingual_sentiment_search]

    def get_news_tools(self):
        from src.tools.news import get_macroeconomic_news, get_news
        from src.tools.research import search_foreign_sources

        # search_foreign_sources powers the PRICE DRAWDOWN PROTOCOL's
        # native-language "share price decline reason" pass (news prompt v5.4).
        return [get_news, get_macroeconomic_news, search_foreign_sources]

    def get_foreign_language_tools(self):
        """Tools for Foreign Language Analyst (supplemental data from native sources)."""
        from src.tools.official_documents import get_official_document
        from src.tools.research import (
            extract_guidance_sources,
            get_official_filings,
            search_foreign_sources,
        )

        return [
            search_foreign_sources,
            extract_guidance_sources,
            get_official_filings,
            get_official_document,
        ]

    def get_auditor_tools(self):
        """Bounded retrieval and deterministic calculations for the Auditor."""
        from src.tools.forensic import (
            calculate_forensic_ratios,
            validate_forensic_evidence,
        )

        return [
            *self.get_foreign_language_tools(),
            *self.get_junior_fundamental_tools(),
            *self.get_news_tools(),
            validate_forensic_evidence,
            calculate_forensic_ratios,
        ]

    def get_legal_tools(self):
        """Tools for Legal Counsel tax, structural, and disclosure review."""
        from src.tools.legal import search_legal_tax_disclosures
        from src.tools.official_documents import get_official_document
        from src.tools.research import get_official_filings, search_foreign_sources

        return [
            search_legal_tax_disclosures,
            search_foreign_sources,
            get_official_filings,
            get_official_document,
        ]

    def get_portfolio_tools(self):
        """Read-only IBKR/account tools for portfolio-aware workflows."""
        from src.tools.portfolio import (
            get_ibkr_account_status,
            get_ibkr_cash_summary,
            get_ibkr_holdings,
            get_ibkr_live_orders,
            get_ibkr_portfolio_snapshot,
            get_ibkr_watchlist,
        )

        return [
            get_ibkr_holdings,
            get_ibkr_watchlist,
            get_ibkr_live_orders,
            get_ibkr_cash_summary,
            get_ibkr_account_status,
            get_ibkr_portfolio_snapshot,
        ]

    def get_value_trap_tools(self):
        """Tools for Value Trap Detector (governance & capital allocation analysis)."""
        from src.tools.news import get_news
        from src.tools.ownership import get_ownership_structure
        from src.tools.research import get_official_filings, search_foreign_sources

        return [
            get_ownership_structure,
            get_news,
            search_foreign_sources,
            get_official_filings,
        ]

    def get_all_tools(self):
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
