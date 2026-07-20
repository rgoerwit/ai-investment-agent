from __future__ import annotations

import json

import pytest

import scripts.portfolio_manager as portfolio_manager
from scripts.portfolio_manager import format_json
from src.ibkr.models import PortfolioSummary, ReconciliationItem
from src.ibkr.portfolio_action_plan import (
    build_action_plan_counts,
    build_portfolio_action_plan,
)
from src.ibkr.portfolio_presentation import build_cash_summary
from src.ibkr.recommendation_service import PortfolioRecommendationBundle
from src.web.ibkr_dashboard.serializers import serialize_dashboard_snapshot
from tests.factories.ibkr import make_analysis, make_position


def _portfolio(*, exchange_weights: dict[str, float] | None = None) -> PortfolioSummary:
    return PortfolioSummary(
        portfolio_value_usd=100_000,
        cash_balance_usd=15_000,
        settled_cash_usd=10_000,
        available_cash_usd=8_000,
        exchange_weights=exchange_weights or {},
    )


def _buy(
    ticker: str,
    *,
    is_watchlist: bool = False,
    cost: float = 1_500,
) -> ReconciliationItem:
    analysis = make_analysis(ticker=ticker, conviction="High", size_pct=4.0)
    analysis.health_adj = 60.0
    analysis.growth_adj = 60.0
    return ReconciliationItem(
        ticker=ticker,
        action="BUY",
        reason="BUY",
        urgency="MEDIUM",
        analysis=analysis,
        is_watchlist=is_watchlist,
        suggested_quantity=10,
        suggested_price=100.0,
        cash_impact_usd=-cost,
    )


def test_action_plan_excludes_only_genuinely_open_offwatch_buy_orders():
    item = _buy("WDO.TO")
    open_plan = build_portfolio_action_plan(
        [item],
        _portfolio(),
        watchlist_tickers=set(),
        watchlist_supplied=True,
        watchlist_unavailable=False,
        live_orders=[{"ticker": "WDO", "side": "B", "status": "Submitted"}],
    )
    filled_plan = build_portfolio_action_plan(
        [item],
        _portfolio(),
        watchlist_tickers=set(),
        watchlist_supplied=True,
        watchlist_unavailable=False,
        live_orders=[{"ticker": "WDO", "side": "B", "status": "Filled"}],
    )

    assert open_plan.in_flight_buys == (item,)
    assert open_plan.optimization.add == ()
    assert filled_plan.in_flight_buys == ()
    assert filled_plan.optimization.add == (item,)


def test_action_plan_macro_state_enables_macro_review_dip():
    analysis = make_analysis(
        ticker="7203.T",
        verdict="DO_NOT_INITIATE",
        zone="HIGH",
        health_adj=60.0,
        growth_adj=60.0,
        entry_price=2_100,
        current_price=1_800,
    )
    item = ReconciliationItem(
        ticker="7203.T",
        action="REVIEW",
        reason="Macro review",
        urgency="MEDIUM",
        sell_type="SOFT_REJECT",
        analysis=analysis,
        ibkr_position=make_position(current_price=1_800),
    )

    inactive = build_portfolio_action_plan(
        [item],
        _portfolio(),
        watchlist_tickers=None,
        watchlist_supplied=False,
        watchlist_unavailable=False,
    )
    active = build_portfolio_action_plan(
        [item],
        _portfolio(),
        watchlist_tickers=None,
        watchlist_supplied=False,
        watchlist_unavailable=False,
        macro_event_active=True,
    )

    assert inactive.groups.dip_candidates == ()
    assert active.groups.dip_candidates == (item,)


def test_json_and_dashboard_share_optimized_counts_and_cash():
    screened = _buy("7203.T", is_watchlist=True, cost=2_000)
    kept = _buy("MEGP.L", is_watchlist=True, cost=1_500)
    portfolio = _portfolio(exchange_weights={"T": 45.0})
    bundle = PortfolioRecommendationBundle(
        portfolio=portfolio,
        items=[screened, kept],
        watchlist_tickers={"7203.T", "MEGP.L"},
        watchlist_total=2,
    )

    dashboard = serialize_dashboard_snapshot(bundle)
    machine = json.loads(
        format_json(
            bundle.items,
            bundle.portfolio,
            watchlist_total=bundle.watchlist_total,
            watchlist_tickers=bundle.watchlist_tickers,
        )
    )

    assert len(machine["items"]) == 2
    assert (
        machine["recommendation_plan"]["summary_counts"] == dashboard["summary_counts"]
    )
    assert machine["cash_summary"]["recommended_buy_cost_usd"] == 1_500
    assert dashboard["cash_summary"]["recommended_buy_cost_usd"] == 1_500
    assert [
        row["ticker_yf"] for row in machine["recommendation_plan"]["watchlist"]["keep"]
    ] == ["MEGP.L"]


def test_cash_summary_rejects_two_execution_authorities():
    item = _buy("7203.T", is_watchlist=True)
    plan = build_portfolio_action_plan(
        [item],
        _portfolio(),
        watchlist_tickers={"7203.T"},
        watchlist_supplied=True,
        watchlist_unavailable=False,
    )

    with pytest.raises(ValueError, match="not both"):
        build_cash_summary(
            [item],
            _portfolio(),
            watchlist_optimization=plan.optimization,
            executable_buy_ids=plan.executable_buy_ids,
        )


def test_unavailable_watchlist_keeps_in_flight_membership_unknown():
    item = _buy("WDO.TO")
    bundle = PortfolioRecommendationBundle(
        portfolio=_portfolio(),
        items=[item],
        watchlist_unavailable=True,
        live_orders=[{"ticker": "WDO", "side": "B", "status": "Submitted"}],
    )

    payload = serialize_dashboard_snapshot(bundle)

    assert payload["watchlist"]["status"] == "unavailable"
    assert payload["actions"]["watchlist_candidate"] == []
    assert (
        payload["actions"]["watchlist_in_flight"][0]["watchlist_membership"]
        == "unknown"
    )
    plan = build_portfolio_action_plan(
        bundle.items,
        bundle.portfolio,
        watchlist_tickers=None,
        watchlist_supplied=False,
        watchlist_unavailable=True,
        live_orders=bundle.live_orders,
    )
    assert build_action_plan_counts(plan, bundle.items)["watchlist_in_flight"] == 1


def test_report_concentration_bars_use_configured_limits(monkeypatch):
    limits: list[float] = []

    def capture_limit(_pct: float, limit: float, width: int = 14) -> str:
        del width
        limits.append(limit)
        return "bar"

    monkeypatch.setattr("src.ibkr.portfolio_report_positions.bar_chart", capture_limit)
    portfolio = _portfolio(exchange_weights={"T": 45.0})
    portfolio.sector_weights = {"Industrials": 35.0}

    portfolio_manager.format_report(
        [],
        portfolio,
        exchange_limit_pct=60.0,
        sector_limit_pct=50.0,
    )

    assert limits == [50.0, 60.0]
