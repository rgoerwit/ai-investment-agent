"""A saved quick-mode BUY must surface as REVIEW, never an investable BUY action."""

from __future__ import annotations

from src.ibkr.models import AnalysisRecord, TradeBlockData
from src.ibkr.opportunity_finder import find_opportunities


def _buy_record(*, is_quick_mode: bool) -> AnalysisRecord:
    return AnalysisRecord(
        ticker="7203.T",
        analysis_date="2026-07-09",
        verdict="BUY",
        health_adj=88.0,
        growth_adj=83.0,
        zone="LOW",
        current_price=2100.0,
        currency="JPY",
        entry_price=2100.0,
        conviction="Medium",
        is_quick_mode=is_quick_mode,
        trade_block=TradeBlockData(
            action="BUY", size_pct=5.0, conviction="Medium", entry_price=2100.0
        ),
    )


def _find(record: AnalysisRecord, portfolio):
    return find_opportunities(
        {"7203.T": record},
        set(),
        portfolio,
        diagnostics=None,
        structural_macro_events=[],
        max_age_days=100000,
        drift_threshold_pct=100.0,
        sector_limit_pct=100.0,
        exchange_limit_pct=100.0,
        sector_weights={},
        exchange_weights={},
        remaining_cash=50000.0,
    )


def test_quick_mode_buy_becomes_review(sample_portfolio):
    items, _ = _find(_buy_record(is_quick_mode=True), sample_portfolio)
    assert len(items) == 1
    assert items[0].action == "REVIEW"
    assert "quick-mode" in items[0].reason.lower()


def test_full_mode_buy_stays_buy(sample_portfolio):
    items, _ = _find(_buy_record(is_quick_mode=False), sample_portfolio)
    assert len(items) == 1
    assert items[0].action == "BUY"
