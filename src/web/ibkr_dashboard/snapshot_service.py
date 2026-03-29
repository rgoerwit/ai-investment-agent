from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from src.ibkr.portfolio_data_service import IbkrPortfolioDataService
from src.ibkr.recommendation_service import (
    PortfolioRecommendationBundle,
    PortfolioRecommendationRequest,
    PortfolioRecommendationService,
)
from src.web.ibkr_dashboard.settings import DashboardPreferences, DashboardSettings

SnapshotStatus = Literal["idle", "loading", "ready", "error"]


@dataclass(frozen=True)
class SnapshotMetadata:
    status: SnapshotStatus
    fetched_at: str | None
    cache_hit: bool
    refreshing: bool
    last_error: str | None


class DashboardSnapshotService:
    def __init__(
        self,
        settings: DashboardSettings,
        *,
        preferences: DashboardPreferences | None = None,
    ) -> None:
        self._settings = settings
        self._preferences = preferences or DashboardPreferences(
            account_id=settings.account_id,
            read_only=settings.read_only,
            watchlist_name=settings.watchlist_name,
            max_age_days=settings.max_age_days,
            quick_mode_default=True,
            refresh_limit=settings.default_refresh_limit,
        )
        self._bundle: PortfolioRecommendationBundle | None = None
        self._fetched_at: datetime | None = None
        self._last_error: str | None = None
        self._results_mtime_ns: int | None = None
        self._loading = False
        self._preferences_version = 0
        self._load_done = threading.Event()
        self._load_done.set()
        self._lock = threading.Lock()

    def get_cached_snapshot(self) -> PortfolioRecommendationBundle | None:
        with self._lock:
            self._invalidate_if_stale_locked()
            return self._bundle

    def current_preferences(self) -> DashboardPreferences:
        with self._lock:
            return self._preferences.model_copy()

    def apply_preferences(self, preferences: DashboardPreferences) -> bool:
        with self._lock:
            invalidates_snapshot = any(
                (
                    preferences.account_id != self._preferences.account_id,
                    preferences.read_only != self._preferences.read_only,
                    preferences.watchlist_name != self._preferences.watchlist_name,
                    preferences.max_age_days != self._preferences.max_age_days,
                )
            )
            self._preferences = preferences
            if invalidates_snapshot:
                self._preferences_version += 1
                self._bundle = None
                self._fetched_at = None
                self._last_error = None
                self._results_mtime_ns = None
            return invalidates_snapshot

    def load_snapshot_sync(
        self,
        *,
        force: bool = False,
    ) -> tuple[PortfolioRecommendationBundle | None, SnapshotMetadata]:
        with self._lock:
            self._invalidate_if_stale_locked()

            if self._loading:
                if self._bundle is not None:
                    return self._bundle, self._meta_locked(
                        "ready",
                        cache_hit=True,
                        refreshing=True,
                    )
                return None, self._meta_locked(
                    "loading",
                    cache_hit=False,
                    refreshing=True,
                )

            if self._bundle is not None and not force:
                return self._bundle, self._meta_locked(
                    "ready",
                    cache_hit=True,
                    refreshing=False,
                )

            if self._bundle is None and self._last_error is not None and not force:
                return None, self._meta_locked(
                    "error",
                    cache_hit=False,
                    refreshing=False,
                )

            self._start_background_load_locked()
            if self._bundle is not None:
                return self._bundle, self._meta_locked(
                    "ready",
                    cache_hit=True,
                    refreshing=True,
                )
            return None, self._meta_locked(
                "loading",
                cache_hit=False,
                refreshing=True,
            )

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        return self._load_done.wait(timeout)

    async def _load_snapshot(self) -> PortfolioRecommendationBundle:
        preferences = self.current_preferences()
        read_only = preferences.read_only
        service = PortfolioRecommendationService(
            portfolio_data_service=(None if read_only else IbkrPortfolioDataService()),
        )
        request = PortfolioRecommendationRequest(
            results_dir=self._settings.results_dir,
            account_id=preferences.account_id,
            watchlist_name=preferences.watchlist_name,
            cash_buffer=self._settings.cash_buffer,
            max_age_days=preferences.max_age_days,
            drift_pct=self._settings.drift_pct,
            sector_limit_pct=self._settings.sector_limit_pct,
            exchange_limit_pct=self._settings.exchange_limit_pct,
            recommend=not read_only,
            read_only=read_only,
            quick_mode=False,
            refresh_policy="off",
            refresh_limit=preferences.refresh_limit,
        )
        return await service.build_bundle(request)

    def _start_background_load_locked(self) -> None:
        self._loading = True
        self._last_error = None
        self._load_done.clear()
        load_version = self._preferences_version
        thread = threading.Thread(
            target=self._load_in_background,
            args=(load_version,),
            daemon=True,
        )
        thread.start()

    def _load_in_background(self, load_version: int) -> None:
        bundle: PortfolioRecommendationBundle | None = None
        error_message: str | None = None
        results_mtime_ns: int | None = None
        try:
            bundle = asyncio.run(self._load_snapshot())
            results_mtime_ns = self._results_dir_mtime_ns()
        except Exception as exc:
            error_message = str(exc)
        finally:
            with self._lock:
                stale_load = load_version != self._preferences_version
                if not stale_load and bundle is not None:
                    self._bundle = bundle
                    self._fetched_at = datetime.now(UTC)
                    self._results_mtime_ns = results_mtime_ns
                    self._last_error = None
                elif not stale_load and error_message is not None:
                    self._last_error = error_message
                self._loading = False
                self._load_done.set()

    def _invalidate_if_stale_locked(self) -> None:
        if self._loading or self._bundle is None:
            return
        current_mtime = self._results_dir_mtime_ns()
        if (
            current_mtime is None
            or self._results_mtime_ns is None
            or current_mtime <= self._results_mtime_ns
        ):
            return
        self._bundle = None
        self._fetched_at = None
        self._results_mtime_ns = current_mtime
        self._last_error = None

    def _results_dir_mtime_ns(self) -> int | None:
        if not self._settings.results_dir.exists():
            return None
        return self._settings.results_dir.stat().st_mtime_ns

    def _meta_locked(
        self,
        status: SnapshotStatus,
        *,
        cache_hit: bool,
        refreshing: bool,
    ) -> SnapshotMetadata:
        return SnapshotMetadata(
            status=status,
            fetched_at=self._fetched_at.isoformat() if self._fetched_at else None,
            cache_hit=cache_hit,
            refreshing=refreshing,
            last_error=self._last_error,
        )
