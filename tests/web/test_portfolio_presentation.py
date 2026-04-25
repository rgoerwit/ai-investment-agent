from __future__ import annotations

from src.ibkr.portfolio_presentation import (
    build_action_summary_counts,
    group_portfolio_actions,
)
from src.sector_normalization import aggregate_sector_weights


def test_group_portfolio_actions_matches_cli_buckets(sample_bundle):
    groups = group_portfolio_actions(
        sample_bundle.items,
        watchlist_tickers=sample_bundle.watchlist_tickers,
    )

    assert [item.ticker.yf for item in groups.hard_sells] == ["7203.T"]
    assert [item.ticker.yf for item in groups.macro_reviews] == ["5285.T"]
    assert [item.ticker.yf for item in groups.holds_real] == ["MEGP.L"]
    assert [item.ticker.yf for item in groups.new_buys] == ["ASML.AS"]
    assert [item.ticker.yf for item in groups.watchlist_candidates] == ["BMW.DE"]
    assert [item.ticker.yf for item in groups.dip_candidates] == ["5285.T"]


def test_build_action_summary_counts_separates_buys_from_candidates(sample_bundle):
    groups = group_portfolio_actions(
        sample_bundle.items,
        watchlist_tickers=sample_bundle.watchlist_tickers,
    )
    counts = build_action_summary_counts(groups)

    assert counts["SELL"] == 1
    assert counts["BUY"] == 1
    assert counts["CANDIDATES"] == 1
    assert counts["HOLD"] == 1
    assert counts["MACRO_WATCH"] == 1


def test_aggregate_sector_weights_normalizes_equivalent_labels():
    weights = aggregate_sector_weights({"Healthcare": 12.5, "Health Care": 7.5})
    assert weights == {"Health Care": 20.0}
