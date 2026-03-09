"""
Tests for watchlist support in reconciler.py (Phase 1.5).

All tests are unit tests — no IBKR connection required.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.ibkr.models import AnalysisRecord, NormalizedPosition, PortfolioSummary
from src.ibkr.reconciler import reconcile
from src.ibkr.ticker import Ticker

# ── Helpers ──────────────────────────────────────────────────────────────────


def _fresh_analysis(
    ticker: str,
    verdict: str,
    age_days: int = 2,
) -> AnalysisRecord:
    """Create a fresh (non-stale) AnalysisRecord with the given verdict."""
    analysis_date = (date.today() - timedelta(days=age_days)).isoformat()
    return AnalysisRecord(
        ticker=ticker,
        analysis_date=analysis_date,
        verdict=verdict,
    )


def _stale_analysis(ticker: str, verdict: str = "BUY") -> AnalysisRecord:
    """Create a stale AnalysisRecord (32 days old, beyond default 14-day limit)."""
    analysis_date = (date.today() - timedelta(days=32)).isoformat()
    return AnalysisRecord(
        ticker=ticker,
        analysis_date=analysis_date,
        verdict=verdict,
    )


def _empty_portfolio() -> PortfolioSummary:
    return PortfolioSummary(
        portfolio_value_usd=0.0,
        cash_balance_usd=0.0,
        available_cash_usd=0.0,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


def test_watchlist_buy_verdict_surfaces_as_buy():
    """Fresh BUY analysis on a watchlist ticker → action=BUY, is_watchlist=True."""
    ticker = "7203.T"
    analyses = {ticker: _fresh_analysis(ticker, "BUY")}
    items = reconcile(
        positions=[],
        analyses=analyses,
        portfolio=_empty_portfolio(),
        watchlist_tickers={ticker},
    )
    assert len(items) == 1
    item = items[0]
    assert item.ticker.yf == ticker
    assert item.action == "BUY"
    assert item.is_watchlist is True


def test_watchlist_dni_verdict_surfaces_as_remove():
    """Fresh DO_NOT_INITIATE analysis → action=REMOVE, is_watchlist=True."""
    ticker = "ZQM.SI"
    analyses = {ticker: _fresh_analysis(ticker, "DO_NOT_INITIATE")}
    items = reconcile(
        positions=[],
        analyses=analyses,
        portfolio=_empty_portfolio(),
        watchlist_tickers={ticker},
    )
    assert len(items) == 1
    item = items[0]
    assert item.ticker.yf == ticker
    assert item.action == "REMOVE"
    assert item.is_watchlist is True
    assert "DO_NOT_INITIATE" in item.reason


def test_watchlist_sell_verdict_surfaces_as_remove():
    """Fresh SELL analysis → action=REMOVE."""
    ticker = "1810.HK"
    analyses = {ticker: _fresh_analysis(ticker, "SELL")}
    items = reconcile(
        positions=[],
        analyses=analyses,
        portfolio=_empty_portfolio(),
        watchlist_tickers={ticker},
    )
    assert len(items) == 1
    item = items[0]
    assert item.action == "REMOVE"
    assert item.is_watchlist is True


def test_watchlist_reject_verdict_surfaces_as_remove():
    """Fresh REJECT analysis → action=REMOVE."""
    ticker = "0941.HK"
    analyses = {ticker: _fresh_analysis(ticker, "REJECT")}
    items = reconcile(
        positions=[],
        analyses=analyses,
        portfolio=_empty_portfolio(),
        watchlist_tickers={ticker},
    )
    assert len(items) == 1
    assert items[0].action == "REMOVE"


def test_watchlist_hold_verdict_surfaces_as_monitoring():
    """Fresh HOLD analysis → action=HOLD, is_watchlist=True (monitoring, not a position)."""
    ticker = "0005.HK"
    analyses = {ticker: _fresh_analysis(ticker, "HOLD")}
    items = reconcile(
        positions=[],
        analyses=analyses,
        portfolio=_empty_portfolio(),
        watchlist_tickers={ticker},
    )
    assert len(items) == 1
    item = items[0]
    assert item.ticker.yf == ticker
    assert item.action == "HOLD"
    assert item.is_watchlist is True
    assert item.ibkr_position is None
    assert "monitoring" in item.reason.lower()


def test_watchlist_no_analysis_surfaces_as_review():
    """Watchlist ticker with no analysis on disk → action=REVIEW."""
    ticker = "9984.T"
    items = reconcile(
        positions=[],
        analyses={},  # no analyses at all
        portfolio=_empty_portfolio(),
        watchlist_tickers={ticker},
    )
    assert len(items) == 1
    item = items[0]
    assert item.ticker.yf == ticker
    assert item.action == "REVIEW"
    assert item.is_watchlist is True
    assert "no analysis" in item.reason.lower()


def test_watchlist_stale_analysis_surfaces_as_review():
    """Watchlist ticker with stale analysis → action=REVIEW with staleness note."""
    ticker = "2330.TW"
    analyses = {ticker: _stale_analysis(ticker, "BUY")}
    items = reconcile(
        positions=[],
        analyses=analyses,
        portfolio=_empty_portfolio(),
        watchlist_tickers={ticker},
        max_age_days=14,
    )
    assert len(items) == 1
    item = items[0]
    assert item.ticker.yf == ticker
    assert item.action == "REVIEW"
    assert item.is_watchlist is True
    assert "stale" in item.reason.lower()


def test_watchlist_ticker_also_held_uses_phase1():
    """
    When a watchlist ticker is also held as a live IBKR position,
    Phase 1 handles it (normal position logic) — Phase 1.5 skips it.
    The resulting HOLD item must NOT have is_watchlist=True.
    """
    ticker = "0005.HK"
    pos = NormalizedPosition(
        conid=12345,
        ticker=Ticker.from_yf(ticker, currency="HKD"),
        quantity=1000.0,
        market_value_usd=12000.0,
        currency="HKD",
        current_price_local=9.50,
    )
    analyses = {ticker: _fresh_analysis(ticker, "BUY")}
    items = reconcile(
        positions=[pos],
        analyses=analyses,
        portfolio=PortfolioSummary(
            portfolio_value_usd=100_000.0,
            cash_balance_usd=5000.0,
            available_cash_usd=4750.0,
        ),
        watchlist_tickers={ticker},
    )
    # Phase 1 should handle it; there should be exactly one item
    assert len(items) == 1
    item = items[0]
    assert item.ticker.yf == ticker
    assert item.is_watchlist is False  # Phase 1 item, not Phase 1.5


def test_watchlist_empty_set_no_change():
    """
    An empty watchlist set → behaviour identical to passing no watchlist.
    Only Phase 2 new-BUY logic runs as before.
    """
    ticker = "ASML.AS"
    analyses = {ticker: _fresh_analysis(ticker, "BUY")}

    items_no_watchlist = reconcile(
        positions=[],
        analyses=analyses,
        portfolio=_empty_portfolio(),
        watchlist_tickers=None,
    )
    items_empty_watchlist = reconcile(
        positions=[],
        analyses=analyses,
        portfolio=_empty_portfolio(),
        watchlist_tickers=set(),
    )

    assert len(items_no_watchlist) == len(items_empty_watchlist)
    assert items_no_watchlist[0].action == items_empty_watchlist[0].action
    # Neither should have is_watchlist set (Phase 2 logic)
    assert items_no_watchlist[0].is_watchlist is False
    assert items_empty_watchlist[0].is_watchlist is False


def test_watchlist_none_no_change():
    """
    watchlist_tickers=None → identical to today's behaviour (no Phase 1.5).
    """
    ticker = "SAP.DE"
    analyses = {ticker: _fresh_analysis(ticker, "BUY")}
    items = reconcile(
        positions=[],
        analyses=analyses,
        portfolio=_empty_portfolio(),
        watchlist_tickers=None,
    )
    # Phase 2 surfaces the BUY
    assert len(items) == 1
    assert items[0].action == "BUY"
    assert items[0].is_watchlist is False


def test_watchlist_buy_not_duplicated_in_phase2():
    """
    A watchlist BUY ticker must appear exactly once — Phase 1.5 handles it
    and Phase 2 must skip it (no duplicate item).
    """
    ticker = "6758.T"
    analyses = {ticker: _fresh_analysis(ticker, "BUY")}
    items = reconcile(
        positions=[],
        analyses=analyses,
        portfolio=_empty_portfolio(),
        watchlist_tickers={ticker},
    )
    buy_items = [i for i in items if i.ticker.yf == ticker]
    assert len(buy_items) == 1
    assert buy_items[0].is_watchlist is True


def test_watchlist_remove_reason_contains_verdict_and_date():
    """REMOVE item reason should include the verdict name and analysis date."""
    ticker = "3690.HK"
    analysis = _fresh_analysis(ticker, "DO_NOT_INITIATE", age_days=3)
    analyses = {ticker: analysis}
    items = reconcile(
        positions=[],
        analyses=analyses,
        portfolio=_empty_portfolio(),
        watchlist_tickers={ticker},
    )
    assert items[0].action == "REMOVE"
    assert "DO_NOT_INITIATE" in items[0].reason
    assert analysis.analysis_date in items[0].reason
