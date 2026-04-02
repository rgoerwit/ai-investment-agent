from __future__ import annotations

from pathlib import Path

import pytest

from src.ibkr.models import ReconciliationItem
from src.ibkr.recommendation_service import PortfolioRecommendationBundle
from src.ibkr.refresh_service import (
    AnalysisFreshnessRow,
    AnalysisFreshnessSummary,
    RefreshActivity,
)
from tests.factories.ibkr import make_analysis, make_portfolio, make_position


@pytest.fixture
def sample_bundle(tmp_path: Path) -> PortfolioRecommendationBundle:
    hard_sell_analysis = make_analysis(ticker="7203.T", verdict="DO_NOT_INITIATE")
    review_analysis = make_analysis(ticker="5285.T", verdict="DO_NOT_INITIATE")
    review_analysis.health_adj = 87.0
    review_analysis.growth_adj = 75.0
    hold_analysis = make_analysis(ticker="MEGP.L", verdict="BUY")
    hold_analysis.health_adj = 92.0
    hold_analysis.growth_adj = 67.0
    hold_analysis.entry_price = 146.5
    hold_analysis.stop_price = 125.0
    hold_analysis.target_1_price = 175.0
    hold_analysis.file_path = str(tmp_path / "MEGP.L_20260328_000000_analysis.json")

    hard_sell = ReconciliationItem(
        ticker="7203.T",
        action="SELL",
        urgency="HIGH",
        reason="Verdict → DO_NOT_INITIATE  (2026-03-28)",
        ibkr_position=make_position(ticker="7203.T", current_price=1950),
        analysis=hard_sell_analysis,
        suggested_quantity=100,
        suggested_price=1950.0,
        cash_impact_usd=1300.0,
        settlement_date="2026-03-31",
        sell_type="HARD_REJECT",
    )
    review = ReconciliationItem(
        ticker="5285.T",
        action="REVIEW",
        urgency="MEDIUM",
        reason=(
            "Verdict → DO_NOT_INITIATE  (2026-03-28)"
            "  [MACRO_WATCH: demoted from SELL — correlated event detected]"
        ),
        ibkr_position=make_position(
            ticker="5285.T",
            current_price=1587,
            avg_cost=1909.53,
        ),
        analysis=review_analysis,
        suggested_quantity=100,
        suggested_price=1587.0,
        cash_impact_usd=1063.0,
        settlement_date="2026-03-31",
        sell_type="SOFT_REJECT",
    )
    hold = ReconciliationItem(
        ticker="MEGP.L",
        action="HOLD",
        urgency="LOW",
        reason="No action",
        ibkr_position=make_position(
            ticker="MEGP.L",
            current_price=129.87,
            avg_cost=138.75,
            currency="GBX",
        ),
        analysis=hold_analysis,
    )
    watchlist = ReconciliationItem(
        ticker="ASML.AS",
        action="BUY",
        urgency="MEDIUM",
        reason="Watchlist candidate",
        analysis=make_analysis(ticker="ASML.AS", verdict="BUY"),
        is_watchlist=True,
    )
    buy = ReconciliationItem(
        ticker="BMW.DE",
        action="BUY",
        urgency="MEDIUM",
        reason="Plain buy candidate",
        analysis=make_analysis(ticker="BMW.DE", verdict="BUY"),
    )
    freshness = AnalysisFreshnessSummary(
        blocking_now=[
            AnalysisFreshnessRow(
                display_ticker="5285",
                run_ticker="5285.T",
                bucket="blocking_now",
                reason_family="review",
                reason_text="Needs refresh",
                action="REVIEW",
                age_days=23,
                expires_date="2026-03-29",
                days_until_due=0,
            )
        ],
        due_soon=[
            AnalysisFreshnessRow(
                display_ticker="MEGP",
                run_ticker="MEGP.L",
                bucket="due_soon",
                reason_family="hold",
                reason_text="Expires soon",
                action="HOLD",
                age_days=7,
                expires_date="2026-04-04",
                days_until_due=7,
            )
        ],
    )
    return PortfolioRecommendationBundle(
        analyses={
            "7203.T": hard_sell_analysis,
            "5285.T": review_analysis,
            "MEGP.L": hold_analysis,
        },
        positions=[
            hard_sell.ibkr_position,
            review.ibkr_position,
            hold.ibkr_position,
        ],
        portfolio=make_portfolio(),
        watchlist_tickers={"ASML.AS"},
        watchlist_name="watchlist-2026",
        watchlist_total=1,
        live_orders=[{"ticker": "7203", "side": "SELL", "status": "Submitted"}],
        items=[hard_sell, review, hold, watchlist, buy],
        health_flags=[
            "CORRELATED_SELL_EVENT: 8 positions changed verdict within 7d of 2026-03-20"
            " (45% of held positions) — probable macro event.",
            "STALE_ANALYSIS_RATIO: 18/53 positions (34%) have analyses older than 14d",
        ],
        freshness_summary=freshness,
        refresh_activity=RefreshActivity(policy="off", limit=10),
    )
