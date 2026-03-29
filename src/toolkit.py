"""Compatibility facade for the public toolkit surface.

Implementation now lives in :mod:`src.tools`.
"""

from importlib import import_module

_EXPORT_MODULES = {
    "Toolkit": "src.tools.registry",
    "toolkit": "src.tools.registry",
    "WITHHOLDING_TAX_RATES": "src.tools.legal",
    "get_financial_metrics": "src.tools.market",
    "get_fundamental_analysis": "src.tools.market",
    "get_ibkr_account_status": "src.tools.portfolio",
    "get_ibkr_cash_summary": "src.tools.portfolio",
    "get_ibkr_holdings": "src.tools.portfolio",
    "get_ibkr_live_orders": "src.tools.portfolio",
    "get_ibkr_portfolio_snapshot": "src.tools.portfolio",
    "get_ibkr_watchlist": "src.tools.portfolio",
    "get_macroeconomic_news": "src.tools.news",
    "get_news": "src.tools.news",
    "get_official_filings": "src.tools.research",
    "get_ownership_structure": "src.tools.ownership",
    "get_social_media_sentiment": "src.tools.news",
    "get_technical_indicators": "src.tools.market",
    "get_yfinance_data": "src.tools.market",
    "search_foreign_sources": "src.tools.research",
    "search_legal_tax_disclosures": "src.tools.legal",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
