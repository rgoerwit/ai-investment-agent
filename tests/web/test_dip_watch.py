from __future__ import annotations

from src.ibkr.dip_watch import (
    build_dip_watch_candidates,
    compute_dip_score,
    is_dip_watch_eligible,
    risk_reward_ratio,
    select_dip_watch_candidates,
)
from src.ibkr.models import ReconciliationItem
from tests.factories.ibkr import make_analysis, make_position


def _dip_item(
    *,
    ticker: str = "7203.T",
    verdict: str = "BUY",
    zone: str = "MODERATE",
    age_days: int = 5,
    health_adj: float | None = 75.0,
    growth_adj: float | None = 72.0,
    entry_price: float | None = 2100.0,
    current_price: float = 1800.0,
    stop_price: float | None = 1700.0,
    target_1: float | None = 2600.0,
) -> ReconciliationItem:
    analysis = make_analysis(
        ticker=ticker,
        verdict=verdict,
        zone=zone,
        age_days=age_days,
        health_adj=health_adj,
        growth_adj=growth_adj,
        entry_price=entry_price or 0.0,
        current_price=current_price,
        stop_price=stop_price or 0.0,
        target_1=target_1 or 0.0,
    )
    analysis.entry_price = entry_price
    analysis.stop_price = stop_price
    analysis.target_1_price = target_1
    return ReconciliationItem(
        ticker=ticker,
        action="REVIEW",
        urgency="MEDIUM",
        reason="Macro review",
        ibkr_position=make_position(ticker=ticker, current_price=current_price),
        analysis=analysis,
        sell_type="SOFT_REJECT",
    )


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
    assert selected == []


def test_build_dip_watch_candidates_includes_star_thresholds(sample_bundle):
    rows = build_dip_watch_candidates([_dip_item()])
    assert rows[0].ticker_yf == "7203.T"
    assert rows[0].stars in {"★★★", "★★", "★"}
    assert rows[0].run_ticker == "7203.T"


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


def test_select_dip_watch_candidates_includes_fresh_buy():
    selected = select_dip_watch_candidates([_dip_item()])
    assert [item.ticker.yf for item in selected] == ["7203.T"]


def test_select_dip_watch_candidates_filters_and_limits_after_eligibility():
    low_score = _dip_item(
        ticker="LOW.T",
        health_adj=58.0,
        growth_adj=56.0,
        entry_price=2000.0,
        current_price=1980.0,
        stop_price=1900.0,
        target_1=2100.0,
    )
    high_score = _dip_item(
        ticker="HIGH.T",
        health_adj=80.0,
        growth_adj=78.0,
        entry_price=2000.0,
        current_price=1800.0,
        stop_price=1700.0,
        target_1=2600.0,
    )
    rejected = _dip_item(ticker="DNI.T", verdict="DO_NOT_INITIATE")

    selected = select_dip_watch_candidates(
        [low_score, high_score, rejected],
        limit=1,
    )

    assert [item.ticker.yf for item in selected] == ["HIGH.T"]


def test_dip_watch_excludes_do_not_initiate_verdict():
    item = _dip_item(verdict="DO_NOT_INITIATE", zone="HIGH")
    assert is_dip_watch_eligible(item) is False
    assert select_dip_watch_candidates([item]) == []


def test_dip_watch_excludes_non_buy_verdicts():
    for verdict in ("HOLD", "SELL", "REJECT", "", "DO NOT INITIATE"):
        assert is_dip_watch_eligible(_dip_item(verdict=verdict)) is False


def test_dip_watch_excludes_high_zone_even_when_buy():
    assert is_dip_watch_eligible(_dip_item(verdict="BUY", zone="HIGH")) is False


def test_dip_watch_freshness_boundary_is_30_days():
    assert is_dip_watch_eligible(_dip_item(age_days=30)) is True
    assert is_dip_watch_eligible(_dip_item(age_days=31)) is False


def test_dip_watch_excludes_invalid_or_missing_analysis_date():
    invalid = _dip_item()
    invalid.analysis.analysis_date = "not-a-date"
    missing = _dip_item()
    missing.analysis.analysis_date = ""

    assert is_dip_watch_eligible(invalid) is False
    assert is_dip_watch_eligible(missing) is False


def test_dip_watch_missing_inputs_do_not_raise():
    item = _dip_item(
        health_adj=None,
        growth_adj=None,
        entry_price=None,
        stop_price=None,
        target_1=None,
    )
    assert is_dip_watch_eligible(item) is False
    assert compute_dip_score(item) == 0.0
