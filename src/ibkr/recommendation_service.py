from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
)
from src.ibkr.portfolio_data_service import IbkrPortfolioDataService, PortfolioSnapshot
from src.ibkr.reconciler import (
    ReconciliationDiagnostics,
    compute_portfolio_health,
    load_latest_analyses,
    reconcile,
)
from src.ibkr.refresh_service import (
    AnalysisFreshnessSummary,
    AnalysisRefreshService,
    RefreshActivity,
    RefreshExecutionOptions,
    RefreshPlanOptions,
    RefreshPolicy,
)
from src.ibkr.screening_freshness import (
    ScreeningFreshnessSummary,
    load_screening_freshness,
)
from src.ibkr.types import ProgressCallback


@dataclass(frozen=True)
class PortfolioRecommendationRequest:
    results_dir: Path
    account_id: str | None
    watchlist_name: str | None
    cash_buffer: float
    max_age_days: int
    drift_pct: float
    sector_limit_pct: float
    exchange_limit_pct: float
    overweight_pct: float = 20.0
    underweight_pct: float = 20.0
    recommend: bool = False
    read_only: bool = False
    quick_mode: bool = False
    refresh_policy: RefreshPolicy = "off"
    refresh_limit: int = 10


@dataclass
class PortfolioRecommendationBundle:
    analyses: dict[str, AnalysisRecord] = field(default_factory=dict)
    positions: list[NormalizedPosition] = field(default_factory=list)
    portfolio: PortfolioSummary = field(default_factory=PortfolioSummary)
    watchlist_tickers: set[str] = field(default_factory=set)
    watchlist_name: str | None = None
    watchlist_total: int | None = None
    watchlist_candidates_blocked_by_cash: int = 0
    live_orders: list[dict] = field(default_factory=list)
    items: list[ReconciliationItem] = field(default_factory=list)
    health_flags: list[str] = field(default_factory=list)
    freshness_summary: AnalysisFreshnessSummary = field(
        default_factory=AnalysisFreshnessSummary
    )
    refresh_activity: RefreshActivity = field(
        default_factory=lambda: RefreshActivity(policy="off", limit=0)
    )
    screening_freshness: ScreeningFreshnessSummary = field(
        default_factory=lambda: ScreeningFreshnessSummary(status="missing")
    )


class PortfolioRecommendationService:
    def __init__(
        self,
        *,
        portfolio_data_service: IbkrPortfolioDataService | None = None,
        refresh_service: AnalysisRefreshService | None = None,
        load_analyses_fn: Callable[[Path], dict[str, AnalysisRecord]] | None = None,
        reconcile_fn: Callable[..., list[ReconciliationItem]] | None = None,
        compute_portfolio_health_fn: Callable[..., list[str]] | None = None,
        run_analysis_fn: Callable[..., Awaitable[dict | None]] | None = None,
        save_results_fn: Callable[..., Path] | None = None,
    ) -> None:
        self._portfolio_data_service = portfolio_data_service
        self._refresh_service = refresh_service or AnalysisRefreshService()
        self._load_analyses_fn = load_analyses_fn or load_latest_analyses
        self._reconcile_fn = reconcile_fn or reconcile
        self._compute_portfolio_health_fn = (
            compute_portfolio_health_fn or compute_portfolio_health
        )
        self._run_analysis_fn = run_analysis_fn
        self._save_results_fn = save_results_fn

    async def build_bundle(
        self,
        request: PortfolioRecommendationRequest,
        *,
        progress: ProgressCallback | None = None,
    ) -> PortfolioRecommendationBundle:
        analyses = self._load_analyses_fn(request.results_dir)
        if not analyses:
            raise ValueError(f"No analysis JSONs found in {request.results_dir}/")

        positions: list[NormalizedPosition] = []
        portfolio = PortfolioSummary(
            portfolio_value_usd=0,
            cash_balance_usd=0,
            available_cash_usd=0,
        )
        watchlist_tickers: set[str] = set()
        watchlist_name: str | None = None
        watchlist_total: int | None = None
        live_orders: list[dict] = []

        if not request.read_only:
            if self._portfolio_data_service is None:
                raise ValueError(
                    "portfolio_data_service is required when read_only=False"
                )
            snapshot = await self._portfolio_data_service.fetch_snapshot(
                account_id=request.account_id,
                watchlist_name=request.watchlist_name,
                explicitly_requested=request.watchlist_name is not None,
                cash_buffer_pct=request.cash_buffer,
                include_live_orders=request.recommend,
                progress=progress,
            )
            self._validate_watchlist_snapshot(snapshot, request.watchlist_name)
            positions = snapshot.positions
            portfolio = snapshot.portfolio
            watchlist_tickers = snapshot.watchlist.tickers
            watchlist_name = snapshot.watchlist.loaded_name
            watchlist_total = snapshot.watchlist.total
            live_orders = snapshot.live_orders

        (
            items,
            health_flags,
            freshness_summary,
            watchlist_candidates_blocked_by_cash,
        ) = self._reconcile_and_classify(
            request=request,
            analyses=analyses,
            positions=positions,
            portfolio=portfolio,
            watchlist_tickers=watchlist_tickers,
        )
        refresh_activity = self._refresh_service.plan(
            freshness_summary,
            options=RefreshPlanOptions(
                policy=request.refresh_policy,
                limit=request.refresh_limit,
                show_recommendations=request.recommend,
                read_only=request.read_only,
                max_age_days=request.max_age_days,
            ),
        )

        if refresh_activity.queued:
            if self._run_analysis_fn is None or self._save_results_fn is None:
                raise ValueError(
                    "run_analysis_fn and save_results_fn are required when refreshes are queued"
                )
            if progress is not None:
                progress(
                    f"Refreshing {len(refresh_activity.queued)} analyses ({request.refresh_policy})..."
                )
            refresh_activity = await self._refresh_service.execute(
                refresh_activity,
                execution=RefreshExecutionOptions(quick_mode=request.quick_mode),
                run_analysis_fn=self._run_analysis_fn,
                save_results_fn=self._save_results_fn,
                progress=progress,
            )
            analyses = self._load_analyses_fn(request.results_dir)
            (
                items,
                health_flags,
                freshness_summary,
                watchlist_candidates_blocked_by_cash,
            ) = self._reconcile_and_classify(
                request=request,
                analyses=analyses,
                positions=positions,
                portfolio=portfolio,
                watchlist_tickers=watchlist_tickers,
            )

        return PortfolioRecommendationBundle(
            analyses=analyses,
            positions=positions,
            portfolio=portfolio,
            watchlist_tickers=watchlist_tickers,
            watchlist_name=watchlist_name,
            watchlist_total=watchlist_total,
            watchlist_candidates_blocked_by_cash=watchlist_candidates_blocked_by_cash,
            live_orders=live_orders,
            items=items,
            health_flags=health_flags,
            freshness_summary=freshness_summary,
            refresh_activity=refresh_activity,
            screening_freshness=load_screening_freshness(request.results_dir),
        )

    @staticmethod
    def _validate_watchlist_snapshot(
        snapshot: PortfolioSnapshot,
        watchlist_name: str | None,
    ) -> None:
        watchlist = snapshot.watchlist
        if not watchlist.found and watchlist.explicitly_requested:
            raise ValueError(f"watchlist '{watchlist_name or ''}' not found in IBKR")

    def _reconcile_and_classify(
        self,
        *,
        request: PortfolioRecommendationRequest,
        analyses: dict[str, AnalysisRecord],
        positions: list[NormalizedPosition],
        portfolio: PortfolioSummary,
        watchlist_tickers: set[str],
    ) -> tuple[list[ReconciliationItem], list[str], AnalysisFreshnessSummary, int]:
        diagnostics = ReconciliationDiagnostics()
        items = self._reconcile_fn(
            positions=positions,
            analyses=analyses,
            portfolio=portfolio,
            max_age_days=request.max_age_days,
            drift_threshold_pct=request.drift_pct,
            overweight_threshold_pct=request.overweight_pct,
            underweight_threshold_pct=request.underweight_pct,
            sector_limit_pct=request.sector_limit_pct,
            exchange_limit_pct=request.exchange_limit_pct,
            watchlist_tickers=watchlist_tickers or None,
            diagnostics=diagnostics,
        )
        health_flags = self._compute_portfolio_health_fn(
            positions=positions,
            analyses=analyses,
            portfolio=portfolio,
            max_age_days=request.max_age_days,
            reconciliation_items=items,
        )
        freshness_summary = self._refresh_service.classify(
            items,
            max_age_days=request.max_age_days,
        )
        return (
            items,
            health_flags,
            freshness_summary,
            diagnostics.cash_blocked_offwatch_buy_count,
        )
