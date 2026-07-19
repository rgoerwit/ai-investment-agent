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


def test_invalid_position_valuation_blocks_all_trade_actions():
    pos = _make_position(ticker="7203.T").model_copy(
        update={
            "market_value_usd": 0.0,
            "unrealized_pnl_usd": 0.0,
            "valuation_valid": False,
            "valuation_issue": "Broker value units could not be verified",
        }
    )
    analysis = _make_analysis(ticker="7203.T", verdict="DO_NOT_INITIATE")
    analysis.zone = "HIGH"

    items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "7203.T")

    assert item.action == "REVIEW"
    assert item.action_basis == "DATA_QUALITY"
    assert item.sell_type == "DATA_QUALITY_REVIEW"
    assert item.suggested_quantity is None
    assert item.cash_impact_usd == 0.0


def test_invalid_position_with_unparseable_quantity_still_surfaces_for_review():
    pos = _make_position(ticker="7203.T", quantity=0).model_copy(
        update={
            "valuation_valid": False,
            "valuation_issue": "Malformed broker numeric field(s): quantity",
        }
    )
    analysis = _make_analysis(ticker="7203.T", verdict="BUY")

    items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "7203.T")

    assert item.action == "REVIEW"
    assert item.action_basis == "DATA_QUALITY"
    assert "quantity" in item.reason


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


def test_high_zone_dni_reviews_unconfirmed_and_sells_confirmed():
    """5285.T-class HIGH-zone DNI: REVIEW on a single reject, executable SELL
    once a prior full-mode reject confirms the thesis failure."""
    from datetime import datetime, timedelta
    from unittest.mock import patch

    from src.ibkr.buy_stability import PriorVerdict

    def _scenario():
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
        return pos, analysis

    pos, analysis = _scenario()
    items = reconcile([pos], {"5285.T": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "5285.T")
    assert item.action == "REVIEW"
    assert item.sell_type == "HARD_REJECT"
    assert item.action_basis == "THESIS_REASSESSMENT"

    pos, analysis = _scenario()
    confirming = [
        PriorVerdict(
            verdict="DO_NOT_INITIATE",
            analysis_dt=datetime.now() - timedelta(days=20),
            is_quick_mode=False,
            file_path="prior.json",
        )
    ]
    with patch(
        "src.ibkr.position_evaluator._load_prior_history", return_value=confirming
    ):
        items = reconcile([pos], {"5285.T": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "5285.T")
    assert item.action == "SELL"
    assert item.sell_type == "HARD_REJECT"
    assert item.action_basis == "CONFIRMED_THESIS_FAILURE"
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


def test_data_vacuum_dni_stop_breach_routes_to_review():
    """A price break on a data-vacuum DNI is an urgent REVIEW — doubly so:
    price alone never sells (July 2026), and the analysis quality is suspect."""
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

    assert item.action == "REVIEW"
    assert item.urgency == "HIGH"
    assert item.suggested_quantity is None


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


def test_zero_score_no_price_dni_stop_breach_routes_to_review():
    """A price break on a zero-score no-price DNI routes to the data-quality
    review — data quality outranks the price signal, and neither sells."""
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

    assert item.action == "REVIEW"
    assert item.urgency == "HIGH"
    assert item.sell_type == "DATA_QUALITY_REVIEW"


def test_explicit_sell_verdict_reviews_until_confirmed_even_if_low_zone():
    """A PM SELL verdict — any zone — reviews when unconfirmed and becomes
    executable once a prior full-mode reject confirms it."""
    from datetime import datetime, timedelta
    from unittest.mock import patch

    from src.ibkr.buy_stability import PriorVerdict

    def _scenario():
        pos = _make_position(ticker="4396.T", current_price=1009.30)
        analysis = _make_analysis(
            ticker="4396.T",
            verdict="SELL",
            stop_price=None,
        )
        analysis.zone = "LOW"
        analysis.health_adj = 96.0
        analysis.growth_adj = 33.0
        return pos, analysis

    pos, analysis = _scenario()
    items = reconcile([pos], {"4396.T": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "4396.T")
    assert item.action == "REVIEW"
    assert item.sell_type == "HARD_REJECT"
    assert item.action_basis == "THESIS_REASSESSMENT"

    pos, analysis = _scenario()
    confirming = [
        PriorVerdict(
            verdict="SELL",
            analysis_dt=datetime.now() - timedelta(days=20),
            is_quick_mode=False,
            file_path="prior.json",
        )
    ]
    with patch(
        "src.ibkr.position_evaluator._load_prior_history", return_value=confirming
    ):
        items = reconcile([pos], {"4396.T": analysis}, _make_portfolio())
    item = _first_item_for_ticker(items, "4396.T")
    assert item.action == "SELL"
    assert item.action_basis == "CONFIRMED_THESIS_FAILURE"


def test_active_tender_routes_to_special_situation_review():
    """2371.T-class: held position with M_AND_A_STATUS=ACTIVE_TENDER routes to
    a SPECIAL_SITUATION_REVIEW, not an auto-SELL — whether to tender, sell in
    market, or hold for a bump depends on deal mechanics and premium, which the
    verdict does not encode. (Deliberate widening vs the legacy auto-exit.)
    """
    from src.ibkr.portfolio_presentation import get_action_label

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

    assert item.action == "REVIEW"
    assert item.action_basis == "SPECIAL_SITUATION_REVIEW"
    # sell_type still carries the M&A tag for grouping/conditional-cash logic
    assert item.sell_type == "SPECIAL_SITUATION_EXIT"
    assert get_action_label(item) == "M&A TENDER REVIEW"
