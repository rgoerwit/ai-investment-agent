from __future__ import annotations

from pathlib import Path

import pytest

from src.ibkr.models import PortfolioSummary, ReconciliationItem
from src.ibkr.portfolio_data_service import PortfolioSnapshot, WatchlistSnapshot
from src.ibkr.recommendation_service import (
    PortfolioRecommendationRequest,
    PortfolioRecommendationService,
)
from tests.ibkr.test_reconciler import _make_analysis, _make_position


class FakePortfolioDataService:
    def __init__(self, snapshot: PortfolioSnapshot):
        self.snapshot = snapshot
        self.calls: list[dict] = []

    async def fetch_snapshot(self, **kwargs) -> PortfolioSnapshot:
        self.calls.append(kwargs)
        return self.snapshot


def _make_request(**overrides) -> PortfolioRecommendationRequest:
    values = {
        "results_dir": Path("results"),
        "account_id": "U123456",
        "watchlist_name": "watchlist-2026",
        "cash_buffer": 0.05,
        "max_age_days": 14,
        "drift_pct": 15.0,
        "sector_limit_pct": 30.0,
        "exchange_limit_pct": 40.0,
        "overweight_pct": 20.0,
        "underweight_pct": 20.0,
        "recommend": False,
        "read_only": False,
        "quick_mode": False,
        "refresh_policy": "off",
        "refresh_limit": 10,
    }
    values.update(overrides)
    return PortfolioRecommendationRequest(**values)


@pytest.mark.asyncio
async def test_build_bundle_read_only_skips_portfolio_fetch():
    service = PortfolioRecommendationService(
        load_analyses_fn=lambda path: {"7203.T": _make_analysis(ticker="7203.T")},
        reconcile_fn=lambda **kwargs: [],
        compute_portfolio_health_fn=lambda **kwargs: [],
    )

    bundle = await service.build_bundle(_make_request(read_only=True))

    assert bundle.portfolio.portfolio_value_usd == 0
    assert bundle.live_orders == []
    assert bundle.watchlist_tickers == set()


@pytest.mark.asyncio
async def test_recommend_mode_includes_live_orders_from_snapshot():
    snapshot = PortfolioSnapshot(
        positions=[_make_position(ticker="7203.T")],
        portfolio=PortfolioSummary(portfolio_value_usd=1000),
        watchlist=WatchlistSnapshot(
            tickers={"7203.T"},
            loaded_name="watchlist-2026",
            total=1,
            found=True,
            explicitly_requested=True,
        ),
        live_orders=[{"ticker": "7203", "side": "BUY"}],
    )
    portfolio_service = FakePortfolioDataService(snapshot)
    service = PortfolioRecommendationService(
        portfolio_data_service=portfolio_service,
        load_analyses_fn=lambda path: {"7203.T": _make_analysis(ticker="7203.T")},
        reconcile_fn=lambda **kwargs: [],
        compute_portfolio_health_fn=lambda **kwargs: [],
    )

    bundle = await service.build_bundle(_make_request(recommend=True))

    assert portfolio_service.calls[0]["include_live_orders"] is True
    assert bundle.live_orders == [{"ticker": "7203", "side": "BUY"}]
    assert bundle.watchlist_name == "watchlist-2026"


@pytest.mark.asyncio
async def test_build_bundle_preserves_cash_blocked_watchlist_candidate_count():
    snapshot = PortfolioSnapshot(
        portfolio=PortfolioSummary(portfolio_value_usd=1000),
        watchlist=WatchlistSnapshot(found=True, explicitly_requested=False),
    )
    portfolio_service = FakePortfolioDataService(snapshot)

    def fake_reconcile(**kwargs):
        kwargs["diagnostics"].cash_blocked_offwatch_buy_count = 3
        return []

    service = PortfolioRecommendationService(
        portfolio_data_service=portfolio_service,
        load_analyses_fn=lambda path: {"7203.T": _make_analysis(ticker="7203.T")},
        reconcile_fn=fake_reconcile,
        compute_portfolio_health_fn=lambda **kwargs: [],
    )

    bundle = await service.build_bundle(_make_request())

    assert bundle.watchlist_candidates_blocked_by_cash == 3


@pytest.mark.asyncio
async def test_refresh_runs_and_rereconciles():
    snapshot = PortfolioSnapshot(
        positions=[_make_position(ticker="7203.T")],
        portfolio=PortfolioSummary(portfolio_value_usd=1000),
        watchlist=WatchlistSnapshot(found=True, explicitly_requested=False),
    )
    portfolio_service = FakePortfolioDataService(snapshot)

    analyses_calls = [
        {"7203.T": _make_analysis(ticker="7203.T", age_days=20)},
        {"7203.T": _make_analysis(ticker="7203.T", age_days=0)},
    ]

    def fake_load_analyses(path: Path):
        return analyses_calls.pop(0)

    stale_item = ReconciliationItem(
        ticker="7203.T",
        action="REVIEW",
        reason="Stale analysis: age 20d > max_age_days 14",
        urgency="MEDIUM",
        ibkr_position=_make_position(ticker="7203.T"),
        analysis=_make_analysis(ticker="7203.T", age_days=20),
    )
    reconcile_calls: list[dict] = []

    def fake_reconcile(**kwargs):
        reconcile_calls.append(kwargs)
        return [stale_item] if len(reconcile_calls) == 1 else []

    health_calls: list[dict] = []

    def fake_health(**kwargs):
        health_calls.append(kwargs)
        return []

    refresh_calls: list[tuple[str, bool, bool]] = []
    saved: list[tuple[str, bool]] = []

    async def fake_run_analysis(*, ticker: str, quick_mode: bool, skip_charts: bool):
        refresh_calls.append((ticker, quick_mode, skip_charts))
        return {"ticker": ticker}

    def fake_save(result, ticker: str, *, quick_mode: bool) -> Path:
        saved.append((ticker, quick_mode))
        return Path(f"/tmp/{ticker}.json")

    service = PortfolioRecommendationService(
        portfolio_data_service=portfolio_service,
        load_analyses_fn=fake_load_analyses,
        reconcile_fn=fake_reconcile,
        compute_portfolio_health_fn=fake_health,
        run_analysis_fn=fake_run_analysis,
        save_results_fn=fake_save,
    )

    bundle = await service.build_bundle(
        _make_request(recommend=True, refresh_policy="blocking")
    )

    assert refresh_calls == [("7203.T", False, True)]
    assert saved == [("7203.T", False)]
    assert len(reconcile_calls) == 2
    assert len(health_calls) == 2
    assert bundle.refresh_activity.refreshed == ["7203.T"]
    assert bundle.items == []


@pytest.mark.asyncio
async def test_missing_explicit_watchlist_raises_value_error():
    snapshot = PortfolioSnapshot(
        positions=[],
        portfolio=PortfolioSummary(),
        watchlist=WatchlistSnapshot(
            tickers=set(),
            loaded_name="watchlist-2026",
            total=None,
            found=False,
            explicitly_requested=True,
        ),
    )
    service = PortfolioRecommendationService(
        portfolio_data_service=FakePortfolioDataService(snapshot),
        load_analyses_fn=lambda path: {"7203.T": _make_analysis(ticker="7203.T")},
        reconcile_fn=lambda **kwargs: [],
        compute_portfolio_health_fn=lambda **kwargs: [],
    )

    with pytest.raises(
        ValueError, match="watchlist 'watchlist-2026' not found in IBKR"
    ):
        await service.build_bundle(_make_request())
