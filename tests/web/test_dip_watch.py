from __future__ import annotations

from src.ibkr.dip_watch import (
    build_dip_watch_candidates,
    compute_dip_score,
    risk_reward_ratio,
    select_dip_watch_candidates,
)
from src.ibkr.models import ReconciliationItem
from tests.factories.ibkr import make_analysis, make_position


def test_compute_dip_score_returns_zero_without_analysis():
    item = ReconciliationItem(
        ticker="7203.T",
        action="REVIEW",
        urgency="MEDIUM",
        reason="No analysis",
        ibkr_position=make_position(),
    )
    assert compute_dip_score(item) == 0.0


def test_select_dip_watch_candidates_filters_and_ranks(sample_bundle):
    review_items = [item for item in sample_bundle.items if item.action == "REVIEW"]
    selected = select_dip_watch_candidates(review_items)
    assert [item.ticker.yf for item in selected] == ["5285.T"]


def test_build_dip_watch_candidates_includes_star_thresholds(sample_bundle):
    review_items = [item for item in sample_bundle.items if item.action == "REVIEW"]
    rows = build_dip_watch_candidates(review_items)
    assert rows[0].ticker_yf == "5285.T"
    assert rows[0].stars in {"★★★", "★★", "★"}


def test_risk_reward_ratio_returns_none_when_stop_or_target_missing():
    analysis = make_analysis()
    analysis.target_1_price = None
    item = ReconciliationItem(
        ticker="7203.T",
        action="REVIEW",
        urgency="MEDIUM",
        reason="Missing target",
        ibkr_position=make_position(),
        analysis=analysis,
    )
    assert risk_reward_ratio(item) is None


def test_compute_dip_score_prefers_better_dip():
    strong = ReconciliationItem(
        ticker="7203.T",
        action="REVIEW",
        urgency="MEDIUM",
        reason="Better dip",
        ibkr_position=make_position(current_price=1800),
        analysis=make_analysis(entry_price=2100, current_price=1800),
    )
    weak = ReconciliationItem(
        ticker="7203.T",
        action="REVIEW",
        urgency="MEDIUM",
        reason="At entry",
        ibkr_position=make_position(current_price=2100),
        analysis=make_analysis(entry_price=2100, current_price=2100),
    )
    assert compute_dip_score(strong) > compute_dip_score(weak)
