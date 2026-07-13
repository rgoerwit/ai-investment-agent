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


def test_data_vacuum_high_zone_dni_routes_to_review_not_sell():
    """Bad suffix/data-vacuum DNI must not become an executable held SELL."""
    pos = _make_position(
        ticker="1264.TW",
        quantity=50,
        avg_cost=270.0,
        current_price=271.0,
        market_value_usd=433.0,
        currency="TWD",
    )
    bad_analysis = _make_analysis(
        ticker="1264.TW",
        verdict="DO_NOT_INITIATE",
        stop_price=None,
        current_price=None,
    )
    bad_analysis.zone = "HIGH"
    bad_analysis.health_adj = 20.0
    bad_analysis.growth_adj = 0.0
    bad_analysis.data_quality = {"data_vacuum": True}
    sibling_analysis = _make_analysis(
        ticker="1264.TWO",
        verdict="HOLD",
        stop_price=245.0,
        current_price=270.5,
    )
    sibling_analysis.zone = "LOW"
    unrelated_same_base = _make_analysis(
        ticker="1264.KS",
        verdict="BUY",
        current_price=10.0,
    )

    items = reconcile(
        [pos],
        {
            "1264.TW": bad_analysis,
            "1264.TWO": sibling_analysis,
            "1264.KS": unrelated_same_base,
        },
        _make_portfolio(),
    )
    item = _first_item_for_ticker(items, "1264.TW")

    assert item.action == "REVIEW"
    assert item.urgency == "HIGH"
    assert item.sell_type == "DATA_QUALITY_REVIEW"
    assert item.analysis is bad_analysis
    assert "1264.TW" in item.reason
    assert "1264.TWO" in item.reason
    assert "1264.KS" not in item.reason
    assert item.suggested_quantity is None
    assert item.cash_impact_usd == 0.0


def test_data_vacuum_dni_stop_breach_still_routes_to_sell():
    """Stop-loss protection remains executable even if analysis quality was poor."""
    pos = _make_position(
        ticker="1264.TW",
        quantity=50,
        avg_cost=270.0,
        current_price=240.0,
        market_value_usd=383.0,
        currency="TWD",
    )
    analysis = _make_analysis(
        ticker="1264.TW",
        verdict="DO_NOT_INITIATE",
        stop_price=245.0,
        current_price=None,
    )
    analysis.zone = "HIGH"
    analysis.data_quality = {"data_vacuum": True}

    items = reconcile([pos], {"1264.TW": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "1264.TW")

    assert item.action == "SELL"
    assert item.sell_type == "STOP_BREACH"


def test_zero_score_no_price_dni_routes_to_review_not_sell():
    """AGS-shaped all-N/A analysis must not become an executable hard-reject SELL."""
    pos = _make_position(
        ticker="AGS",
        quantity=5,
        avg_cost=66.7,
        current_price=66.7,
        market_value_usd=364.0,
        currency="USD",
    )
    analysis = _make_analysis(
        ticker="AGS",
        verdict="DO_NOT_INITIATE",
        stop_price=None,
        current_price=None,
    )
    analysis.zone = "HIGH"
    analysis.health_adj = 0.0
    analysis.growth_adj = 0.0
    analysis.data_quality = {"data_vacuum": False}

    items = reconcile([pos], {"AGS": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "AGS")

    assert item.action == "REVIEW"
    assert item.urgency == "HIGH"
    assert item.sell_type == "DATA_QUALITY_REVIEW"
    assert item.suggested_quantity is None
    assert item.cash_impact_usd == 0.0


def test_zero_score_no_price_dni_stop_breach_still_routes_to_sell():
    """The zero-score guard must not suppress a genuine stop breach."""
    pos = _make_position(
        ticker="AGS",
        quantity=5,
        avg_cost=66.7,
        current_price=60.0,
        market_value_usd=327.0,
        currency="USD",
    )
    analysis = _make_analysis(
        ticker="AGS",
        verdict="DO_NOT_INITIATE",
        stop_price=65.0,
        current_price=None,
    )
    analysis.zone = "HIGH"
    analysis.health_adj = 0.0
    analysis.growth_adj = 0.0
    analysis.data_quality = {"data_vacuum": False}

    items = reconcile([pos], {"AGS": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "AGS")

    assert item.action == "SELL"
    assert item.sell_type == "STOP_BREACH"


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
