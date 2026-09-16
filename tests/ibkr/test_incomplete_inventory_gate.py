"""Portfolio-increasing actions require a complete broker inventory identity."""

import pytest

from src.fx_normalization import set_fx_rate_cache
from src.ibkr.models import UnresolvedBrokerPosition
from src.ibkr.reconciler import reconcile
from tests.ibkr.reconciler_cases import (
    _FakeFxRateCache,
    _make_analysis,
    _make_portfolio,
    _make_position,
)


@pytest.fixture(autouse=True)
def _deterministic_fx_cache():
    set_fx_rate_cache(_FakeFxRateCache())
    yield
    set_fx_rate_cache(None)


def _incomplete_portfolio():
    portfolio = _make_portfolio(cash=15_000)
    portfolio.unresolved_positions = [
        UnresolvedBrokerPosition(
            conid=637_692_266,
            broker_token="IBCID637692266",
            quantity=100,
            currency="TWD",
            market_value_usd=5_000,
            reason="canonical security identity unavailable",
        )
    ]
    return portfolio


def test_unresolved_inventory_blocks_held_position_add() -> None:
    position = _make_position(market_value_usd=1_000)
    analysis = _make_analysis(size_pct=5.0)

    items = reconcile(
        [position],
        {"7203.T": analysis},
        _incomplete_portfolio(),
        underweight_threshold_pct=1.0,
    )
    item = next(item for item in items if item.ticker.yf == "7203.T")

    assert item.action == "REVIEW"
    assert item.action_basis == "CAPITAL_ALLOCATION"
    assert item.suggested_quantity is None
    assert "unresolved broker identities" in item.reason


def test_unresolved_inventory_blocks_watchlist_buy() -> None:
    analysis = _make_analysis(ticker="6758.T")

    items = reconcile(
        [],
        {"6758.T": analysis},
        _incomplete_portfolio(),
        watchlist_tickers={"6758.T"},
    )
    item = next(item for item in items if item.ticker.yf == "6758.T")

    assert item.action == "REVIEW"
    assert item.action_basis == "ENTRY_CONSTRAINT"
    assert item.suggested_quantity is None
    assert "unresolved broker identities" in item.reason


def test_unresolved_inventory_blocks_off_watchlist_buy() -> None:
    analysis = _make_analysis(ticker="6758.T")

    items = reconcile([], {"6758.T": analysis}, _incomplete_portfolio())
    item = next(item for item in items if item.ticker.yf == "6758.T")

    assert item.action == "REVIEW"
    assert item.action_basis == "ENTRY_CONSTRAINT"
    assert item.suggested_quantity is None
    assert "unresolved broker identities" in item.reason


def test_complete_inventory_allows_held_position_add() -> None:
    position = _make_position(market_value_usd=1_000)
    analysis = _make_analysis(size_pct=5.0)

    items = reconcile(
        [position],
        {"7203.T": analysis},
        _make_portfolio(cash=15_000),
        underweight_threshold_pct=1.0,
    )
    item = next(item for item in items if item.ticker.yf == "7203.T")

    assert item.action == "ADD"
    assert item.suggested_quantity is not None


def test_complete_inventory_allows_watchlist_buy() -> None:
    analysis = _make_analysis(ticker="6758.T")

    items = reconcile(
        [],
        {"6758.T": analysis},
        _make_portfolio(cash=15_000),
        watchlist_tickers={"6758.T"},
    )
    item = next(item for item in items if item.ticker.yf == "6758.T")

    assert item.action == "BUY"
    assert item.is_watchlist is True
    assert item.suggested_quantity is not None and item.suggested_quantity > 0


def test_complete_inventory_allows_off_watchlist_buy() -> None:
    analysis = _make_analysis(ticker="6758.T")

    items = reconcile([], {"6758.T": analysis}, _make_portfolio(cash=15_000))
    item = next(item for item in items if item.ticker.yf == "6758.T")

    assert item.action == "BUY"
    assert item.is_watchlist is False
    assert item.suggested_quantity is not None and item.suggested_quantity > 0
