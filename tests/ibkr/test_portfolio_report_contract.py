"""Byte-stable contracts for the complete portfolio-manager text report."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.portfolio_manager import format_report
from src.ibkr.models import (
    AnalysisRecord,
    PortfolioSummary,
    ReconciliationItem,
    TradeBlockData,
)
from tests.ibkr.format_report_cases import (
    CORRELATED_SELL_EVENT_FLAG,
    _make_offwatch_buy,
    _panic_items,
)
from tests.ibkr.reconciler_cases import _make_position

_SNAPSHOTS = Path(__file__).with_name("report_snapshots")
_NOW = datetime(2026, 7, 17, 12, 0)


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return _NOW if tz is None else _NOW.astimezone(tz)


class _FrozenDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 7, 17)


@contextmanager
def _fixed_report_clock():
    with (
        patch("scripts.portfolio_manager.datetime", _FrozenDateTime),
        patch("scripts.portfolio_manager.date", _FrozenDate),
        patch("src.ibkr.models.datetime", _FrozenDateTime),
    ):
        yield


def _portfolio() -> PortfolioSummary:
    return PortfolioSummary(
        account_id="U-GOLDEN",
        portfolio_value_usd=100_000,
        cash_balance_usd=15_000,
        settled_cash_usd=12_000,
        cash_pct=15.0,
        position_count=3,
        available_cash_usd=7_000,
        sector_weights={"Industrials": 24.0, "Information Technology": 18.0},
        exchange_weights={"T": 31.0, "US": 27.0},
        currency_weights={"JPY": 31.0, "USD": 27.0},
    )


def _analysis(
    ticker: str,
    *,
    verdict: str,
    conviction: str = "High",
    sector: str = "Industrials",
) -> AnalysisRecord:
    return AnalysisRecord(
        ticker=ticker,
        analysis_date="2026-07-10",
        verdict=verdict,
        health_adj=72.0,
        growth_adj=64.0,
        zone="MODERATE",
        current_price=2100.0,
        entry_price=2000.0,
        stop_price=1850.0,
        target_1_price=2500.0,
        target_2_price=2800.0,
        conviction=conviction,
        currency="JPY",
        sector=sector,
        exchange="T",
        trade_block=TradeBlockData(
            action=verdict,
            size_pct=4.0,
            conviction=conviction,
            entry_price=2000.0,
            stop_price=1850.0,
            target_1_price=2500.0,
            target_2_price=2800.0,
        ),
    )


def _mixed_report() -> str:
    stop = ReconciliationItem(
        ticker="7203.T",
        action="SELL",
        urgency="HIGH",
        reason="Stop breached: price 1800.00 < stop 1850.00",
        ibkr_position=_make_position(
            ticker="7203.T", current_price=1800.0, market_value_usd=1200.0
        ),
        analysis=_analysis("7203.T", verdict="BUY"),
        sell_type="STOP_BREACH",
        suggested_quantity=100,
        suggested_price=1800.0,
        cash_impact_usd=1200.0,
        settlement_date="2026-07-20",
    )
    add = ReconciliationItem(
        ticker="6758.T",
        action="ADD",
        urgency="MEDIUM",
        reason="Underweight high-conviction position",
        ibkr_position=_make_position(
            ticker="6758.T", quantity=40, current_price=3500.0, market_value_usd=950
        ),
        analysis=_analysis("6758.T", verdict="BUY", sector="Information Technology"),
        suggested_quantity=10,
        suggested_price=3500.0,
        cash_impact_usd=-240.0,
    )
    hold = ReconciliationItem(
        ticker="9432.T",
        action="HOLD",
        urgency="LOW",
        reason="Position remains within thesis",
        ibkr_position=_make_position(
            ticker="9432.T", quantity=80, current_price=160.0, market_value_usd=860
        ),
        analysis=_analysis("9432.T", verdict="HOLD", conviction="Medium"),
    )
    watchlist_buy = ReconciliationItem(
        ticker="9984.T",
        action="BUY",
        urgency="MEDIUM",
        reason="Watchlist BUY (2026-07-10) — High conviction, target 4.0%",
        analysis=_analysis("9984.T", verdict="BUY"),
        suggested_quantity=20,
        suggested_price=9000.0,
        cash_impact_usd=-1200.0,
        is_watchlist=True,
    )
    with _fixed_report_clock():
        return format_report(
            [stop, add, hold, watchlist_buy],
            _portfolio(),
            show_recommendations=True,
            watchlist_name="Core Candidates",
            watchlist_total=1,
            watchlist_tickers={"9984.T"},
        )


def _macro_degraded_report() -> str:
    candidate = _make_offwatch_buy("WDO.TO")
    orders = [
        {
            "ticker": "WDO",
            "side": "B",
            "remainingSize": 100,
            "price": 15.0,
            "orderType": "LMT",
            "status": "Filled",
        },
        {
            "ticker": "WDO",
            "side": "B",
            "remainingSize": 50,
            "price": 14.5,
            "orderType": "LMT",
            "status": "Submitted",
        },
    ]
    with _fixed_report_clock():
        return format_report(
            [*_panic_items(), candidate],
            _portfolio(),
            show_recommendations=True,
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
            live_orders=orders,
            errors={"watchlist": "brokerage session unavailable"},
            portfolio_data_loaded=True,
        )


def _read_only_report() -> str:
    candidate = _make_offwatch_buy("WDO.TO")
    malformed_orders = [
        {},
        {"ticker": "WDO", "side": None, "remainingSize": "not-a-number"},
    ]
    with _fixed_report_clock():
        return format_report(
            [candidate],
            PortfolioSummary(),
            show_recommendations=True,
            live_orders=malformed_orders,
            errors={"live_orders": "order endpoint unavailable"},
            portfolio_data_loaded=False,
        )


@pytest.mark.parametrize(
    ("snapshot_name", "render"),
    (
        ("mixed_portfolio.txt", _mixed_report),
        ("macro_watchlist_unavailable.txt", _macro_degraded_report),
        ("read_only_malformed_orders.txt", _read_only_report),
    ),
)
def test_complete_report_contract(snapshot_name, render):
    assert f"{render()}\n" == (_SNAPSHOTS / snapshot_name).read_text()


# ── Ambient-date independence guard ──────────────────────────────────────────
# _fixed_report_clock freezes only RENDER-time clocks. Fixture builders (e.g.
# reconciler_cases._make_analysis with age_days=…) import datetime at call time
# and derive dates from the real ambient clock — if any such date reaches the
# rendered bytes, the golden silently drifts at midnight (2026-07-19 incident:
# a relative analysis_date rendered into macro_watchlist_unavailable.txt).
# These classes shift the ambient clock by a large, non-round offset; rendering
# must be byte-identical under the shift, which fails the day a leak is
# introduced instead of the day after the golden is regenerated.

_AMBIENT_SHIFT = timedelta(days=37)


class _ShiftedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.now(tz) + _AMBIENT_SHIFT

    @classmethod
    def today(cls):
        return datetime.now() + _AMBIENT_SHIFT


class _ShiftedDate(date):
    @classmethod
    def today(cls):
        real = date.today()
        shifted = real + _AMBIENT_SHIFT
        return cls(shifted.year, shifted.month, shifted.day)


@pytest.mark.parametrize(
    ("snapshot_name", "render"),
    (
        ("mixed_portfolio.txt", _mixed_report),
        ("macro_watchlist_unavailable.txt", _macro_degraded_report),
        ("read_only_malformed_orders.txt", _read_only_report),
    ),
)
def test_report_contract_is_ambient_date_independent(snapshot_name, render):
    baseline = render()
    with (
        patch("datetime.datetime", _ShiftedDateTime),
        patch("datetime.date", _ShiftedDate),
    ):
        shifted = render()
    assert shifted == baseline, (
        "Rendered report depends on the ambient clock — a fixture is deriving "
        "a rendered date from real now()/today(). Pin absolute dates in the "
        "fixture (see format_report_cases._panic_items)."
    )
