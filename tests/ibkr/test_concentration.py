from __future__ import annotations

from src.ibkr.concentration import (
    canonical_exchange_bucket,
    canonical_sector_bucket,
    project_concentration_breaches,
)
from src.ibkr.models import PortfolioSummary
from src.ibkr.portfolio_presentation import group_portfolio_actions
from src.ibkr.watchlist_evaluator import evaluate_watchlist
from src.ibkr.watchlist_optimization import resolve_watchlist_optimization
from tests.factories.ibkr import make_analysis


def test_canonical_buckets_use_suffix_space_and_gics_labels():
    assert canonical_exchange_bucket("7203.T", analysis_exchange="TSEJ") == "T"
    assert canonical_exchange_bucket("7203", analysis_exchange="TSEJ") == "T"
    assert canonical_exchange_bucket("AAPL", analysis_exchange="NASDAQ") == "US"
    assert canonical_sector_bucket("Technology") == "Information Technology"
    assert canonical_sector_bucket(None) is None


def test_projected_breach_is_strictly_over_limit():
    exact = project_concentration_breaches(
        exchange_key="T",
        sector_key=None,
        candidate_pct=5.0,
        exchange_weights={"T": 35.0},
        sector_weights={},
        exchange_limit_pct=40.0,
        sector_limit_pct=30.0,
    )
    over = project_concentration_breaches(
        exchange_key="T",
        sector_key="Information Technology",
        candidate_pct=5.1,
        exchange_weights={"T": 35.0},
        sector_weights={"Information Technology": 26.0},
        exchange_limit_pct=40.0,
        sector_limit_pct=30.0,
    )

    assert exact == ()
    assert [(row.dimension, row.key) for row in over] == [
        ("exchange", "T"),
        ("sector", "Information Technology"),
    ]


def test_watchlist_warning_and_optimizer_use_identical_canonical_keys():
    analysis = make_analysis(
        ticker="7203.T",
        conviction="Medium",
        health_adj=60.0,
        growth_adj=60.0,
        size_pct=4.0,
    )
    analysis.exchange = "TSEJ"
    analysis.sector = "Technology"
    analysis.fx_rate_to_usd = 0.0067
    portfolio = PortfolioSummary(
        portfolio_value_usd=100_000,
        cash_balance_usd=15_000,
        settled_cash_usd=15_000,
        available_cash_usd=15_000,
    )
    items, _, _ = evaluate_watchlist(
        {"7203.T"},
        set(),
        {"7203.T": analysis},
        portfolio,
        alpha_base_lookup={},
        alpha_base_to_key={},
        structural_macro_events=[],
        max_age_days=14,
        drift_threshold_pct=20.0,
        sector_limit_pct=30.0,
        exchange_limit_pct=40.0,
        sector_weights={"Information Technology": 35.0},
        exchange_weights={"T": 45.0},
        remaining_cash=15_000,
    )
    item = items[0]
    optimization = resolve_watchlist_optimization(
        items,
        group_portfolio_actions(items, watchlist_tickers={"7203.T"}),
        watchlist_tickers={"7203.T"},
        watchlist_supplied=True,
        watchlist_unavailable=False,
        exchange_weights={"T": 45.0},
        sector_weights={"Information Technology": 35.0},
    )

    assert "⚠ T →" in item.reason
    assert "⚠ Information Technology sector →" in item.reason
    assert "TSEJ" not in item.reason
    move = optimization.retained_for_watchlist_floor[0]
    assert move.note is not None
    assert [(row.dimension, row.key) for row in move.note.breaches] == [
        ("exchange", "T"),
        ("sector", "Information Technology"),
    ]
