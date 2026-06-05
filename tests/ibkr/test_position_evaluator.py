"""Collected held-position evaluation tests extracted from reconciler cases."""

from src.ibkr.reconciler import reconcile
from tests.ibkr.reconciler_cases import (
    TestConcentration,
    TestReconcile,
    _make_analysis,
    _make_portfolio,
    _make_position,
)


def _first_item_for_ticker(items, ticker: str):
    return next(item for item in items if item.ticker.yf == ticker)


def test_low_zone_dni_held_position_routes_to_review_not_sell():
    """4396.T regression: LOW-zone DNI is a review, not an executable sell."""
    pos = _make_position(
        ticker="4396.T",
        quantity=400,
        avg_cost=1091.87,
        current_price=1009.30,
        market_value_usd=2705.0,
    )
    analysis = _make_analysis(
        ticker="4396.T",
        verdict="DO_NOT_INITIATE",
        stop_price=None,
    )
    analysis.zone = "LOW"
    analysis.health_adj = 96.0
    analysis.growth_adj = 33.0

    items = reconcile([pos], {"4396.T": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "4396.T")

    assert item.action == "REVIEW"
    assert item.sell_type == "SCREEN_REJECT"
    assert item.suggested_quantity is None
    assert item.suggested_price is None
    assert item.cash_impact_usd == 0.0
    assert item.settlement_date is None


def test_moderate_zone_dni_held_position_routes_to_review_not_sell():
    """MODERATE-zone DNI follows the same screen-review path as LOW-zone DNI."""
    pos = _make_position(ticker="4396.T", current_price=1009.30)
    analysis = _make_analysis(
        ticker="4396.T",
        verdict="DO_NOT_INITIATE",
        stop_price=None,
    )
    analysis.zone = "MODERATE"
    analysis.health_adj = 70.0
    analysis.growth_adj = 45.0

    items = reconcile([pos], {"4396.T": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "4396.T")

    assert item.action == "REVIEW"
    assert item.sell_type == "SCREEN_REJECT"


def test_high_zone_dni_remains_executable_sell():
    """5285.T-class HIGH-zone DNI still converts to SELL."""
    pos = _make_position(
        ticker="5285.T",
        quantity=100,
        avg_cost=1909.53,
        current_price=1224.11,
        market_value_usd=820.0,
    )
    analysis = _make_analysis(
        ticker="5285.T",
        verdict="DO_NOT_INITIATE",
        stop_price=None,
    )
    analysis.zone = "HIGH"
    analysis.health_adj = 75.0
    analysis.growth_adj = 33.0

    items = reconcile([pos], {"5285.T": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "5285.T")

    assert item.action == "SELL"
    assert item.sell_type == "HARD_REJECT"
    assert item.suggested_quantity == 100
    assert item.cash_impact_usd == 820.0


def test_explicit_sell_verdict_remains_sell_even_if_low_zone():
    """A PM SELL verdict remains executable regardless of zone."""
    pos = _make_position(ticker="4396.T", current_price=1009.30)
    analysis = _make_analysis(
        ticker="4396.T",
        verdict="SELL",
        stop_price=None,
    )
    analysis.zone = "LOW"
    analysis.health_adj = 96.0
    analysis.growth_adj = 33.0

    items = reconcile([pos], {"4396.T": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "4396.T")

    assert item.action == "SELL"
    assert item.sell_type == "HARD_REJECT"


def test_active_tender_emits_special_situation_exit_label():
    """2371.T regression: held position with M_AND_A_STATUS=ACTIVE_TENDER
    routes to executable SELL labelled M&A EXIT, not FUNDAMENTAL FAILURE.

    The PM still produces a HIGH-zone DNI verdict here (so it does not
    fall through the SCREEN_REJECT REVIEW path), but `_classify_sell_type`
    sees the active tender and prefers SPECIAL_SITUATION_EXIT over the
    default HARD_REJECT classifier output.
    """
    from src.ibkr.portfolio_presentation import get_sell_type_label

    pos = _make_position(
        ticker="2371.T",
        quantity=100,
        avg_cost=2604.0,
        current_price=3338.98,
        market_value_usd=2237.0,
    )
    analysis = _make_analysis(
        ticker="2371.T",
        verdict="DO_NOT_INITIATE",
        stop_price=None,
    )
    analysis.zone = "HIGH"
    analysis.health_adj = 67.0
    analysis.growth_adj = 33.0
    analysis.m_and_a_status = "ACTIVE_TENDER"

    items = reconcile([pos], {"2371.T": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "2371.T")

    assert item.action == "SELL"
    assert item.sell_type == "SPECIAL_SITUATION_EXIT"
    assert get_sell_type_label(item.sell_type) == "M&A EXIT"
    # Still an executable sell — the operator gets quantity + cash impact —
    # only the displayed reason changes.
    assert item.suggested_quantity == 100
    assert item.cash_impact_usd == 2237.0
