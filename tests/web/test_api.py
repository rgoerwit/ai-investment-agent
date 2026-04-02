from __future__ import annotations

from pathlib import Path

import pytest

from src.ibkr.refresh_service import AnalysisFreshnessSummary
from src.web.ibkr_dashboard.app import create_app
from src.web.ibkr_dashboard.job_store import RefreshJobStore
from src.web.ibkr_dashboard.settings import (
    DashboardPreferences,
    DashboardPreferencesStore,
    DashboardSettings,
)
from src.web.ibkr_dashboard.snapshot_service import SnapshotMetadata


class _FakeSnapshotService:
    def __init__(
        self,
        bundle,
        metadata: SnapshotMetadata,
        *,
        preferences: DashboardPreferences | None = None,
    ):
        self._bundle = bundle
        self._metadata = metadata
        self.force_calls: list[bool] = []
        self.preferences = preferences or DashboardPreferences()

    def load_snapshot_sync(self, *, force: bool = False):
        self.force_calls.append(force)
        return self._bundle, self._metadata

    def get_cached_snapshot(self):
        return self._bundle

    def current_preferences(self):
        return self.preferences.model_copy()

    def apply_preferences(self, preferences: DashboardPreferences):
        invalidates = any(
            (
                preferences.account_id != self.preferences.account_id,
                preferences.read_only != self.preferences.read_only,
                preferences.watchlist_name != self.preferences.watchlist_name,
                preferences.max_age_days != self.preferences.max_age_days,
            )
        )
        self.preferences = preferences.model_copy(deep=True)
        return invalidates


class _FakeMacroAlertService:
    def __init__(self, payload=None):
        self.payload = payload or {"detected": True, "headline": "macro headline"}

    def build_alert(self, _health_flags):
        return self.payload


def _make_client(
    tmp_path: Path,
    *,
    bundle,
    metadata: SnapshotMetadata,
    preferences_override: dict | None = None,
):
    settings = DashboardSettings(runtime_dir=tmp_path / "runtime")
    app = create_app(settings, preferences_override=preferences_override)
    app.config["TESTING"] = True
    app.config["SNAPSHOT_SERVICE"] = _FakeSnapshotService(
        bundle,
        metadata,
        preferences=(
            DashboardPreferences.model_validate(preferences_override)
            if preferences_override
            else None
        ),
    )
    app.config["JOB_STORE"] = RefreshJobStore(tmp_path / "runtime" / "jobs.sqlite")
    app.config["PREFERENCES_STORE"] = DashboardPreferencesStore(
        tmp_path / "runtime" / "settings.json"
    )
    app.config["MACRO_ALERT_SERVICE"] = _FakeMacroAlertService()
    return app.test_client(), app.config["SNAPSHOT_SERVICE"]


@pytest.fixture
def client(tmp_path: Path, sample_bundle):
    return _make_client(
        tmp_path,
        bundle=sample_bundle,
        metadata=SnapshotMetadata(
            status="ready",
            fetched_at="2026-03-28T12:00:00Z",
            cache_hit=True,
            refreshing=False,
            last_error=None,
        ),
    )[0]


def test_get_portfolio_returns_payload(tmp_path: Path, sample_bundle):
    client, snapshot_service = _make_client(
        tmp_path,
        bundle=sample_bundle,
        metadata=SnapshotMetadata(
            status="ready",
            fetched_at="2026-03-28T12:00:00Z",
            cache_hit=True,
            refreshing=True,
            last_error=None,
        ),
    )
    response = client.get("/api/portfolio?refresh=1")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["portfolio"]["account_id"] == sample_bundle.portfolio.account_id
    assert payload["refreshing"] is True
    assert snapshot_service.force_calls[-1] is True


def test_index_busts_static_asset_cache(client):
    response = client.get("/")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "dashboard.css?v=" in body
    assert "dashboard.js?v=" in body
    assert "Reload Data" in body
    assert "Analysis Refresh" in body
    assert "Reload Snapshot" not in body


def test_get_portfolio_returns_loading_response(tmp_path: Path):
    client, _snapshot_service = _make_client(
        tmp_path,
        bundle=None,
        metadata=SnapshotMetadata(
            status="loading",
            fetched_at=None,
            cache_hit=False,
            refreshing=True,
            last_error=None,
        ),
    )
    response = client.get("/api/portfolio")
    assert response.status_code == 202
    assert response.get_json()["status"] == "loading"


def test_get_orders_returns_loading_response(tmp_path: Path):
    client, _snapshot_service = _make_client(
        tmp_path,
        bundle=None,
        metadata=SnapshotMetadata(
            status="loading",
            fetched_at=None,
            cache_hit=False,
            refreshing=True,
            last_error=None,
        ),
    )
    response = client.get("/api/orders")
    assert response.status_code == 202
    assert response.get_json()["status"] == "loading"


def test_get_watchlist_returns_structured_error(tmp_path: Path):
    client, _snapshot_service = _make_client(
        tmp_path,
        bundle=None,
        metadata=SnapshotMetadata(
            status="error",
            fetched_at=None,
            cache_hit=False,
            refreshing=False,
            last_error="boom",
        ),
    )
    response = client.get("/api/watchlist")
    assert response.status_code == 503
    assert response.get_json()["error"] == "snapshot_load_failed"


def test_get_equity_drilldown_returns_404_for_unknown_ticker(client):
    response = client.get("/api/equities/UNKNOWN")
    assert response.status_code == 404


def test_create_refresh_job_enqueues_ticker_list(client):
    response = client.post(
        "/api/refresh/jobs",
        json={"scope": "ticker_list", "tickers": ["7203.T"], "quick_mode": True},
    )
    assert response.status_code == 202
    payload = response.get_json()
    assert payload["accepted"] is True
    jobs = client.get("/api/refresh/jobs").get_json()["jobs"]
    assert jobs[0]["scope"] == "ticker_list"


def test_create_refresh_job_requires_snapshot_for_scope(tmp_path: Path):
    client, _snapshot_service = _make_client(
        tmp_path,
        bundle=None,
        metadata=SnapshotMetadata(
            status="loading",
            fetched_at=None,
            cache_hit=False,
            refreshing=True,
            last_error=None,
        ),
    )
    response = client.post("/api/refresh/jobs", json={"scope": "stale_positions"})
    assert response.status_code == 409


def test_create_refresh_job_rejects_empty_scope_result(client, sample_bundle):
    sample_bundle.freshness_summary = AnalysisFreshnessSummary()
    response = client.post("/api/refresh/jobs", json={"scope": "stale_positions"})
    assert response.status_code == 409
    assert response.get_json()["error"] == "no_refresh_candidates"


def test_create_refresh_job_requires_ticker_list_entries(client):
    response = client.post(
        "/api/refresh/jobs", json={"scope": "ticker_list", "tickers": []}
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_request"


def test_settings_round_trip(client):
    response = client.post(
        "/api/settings",
        json={
            "account_id": "U20958465",
            "watchlist_name": "alpha",
            "read_only": True,
            "max_age_days": 21,
        },
    )
    assert response.status_code == 200
    payload = client.get("/api/settings").get_json()
    assert payload["account_id"] == "U20958465"
    assert payload["watchlist_name"] == "alpha"
    assert payload["read_only"] is True
    assert payload["max_age_days"] == 21


def test_settings_save_reports_snapshot_reload_requirement(client):
    response = client.post(
        "/api/settings",
        json={"notes": "operator note"},
    )
    assert response.status_code == 200
    assert response.get_json()["snapshot_reload_required"] is False

    response = client.post(
        "/api/settings",
        json={"watchlist_name": "alpha"},
    )
    assert response.status_code == 200
    assert response.get_json()["snapshot_reload_required"] is True


def test_settings_save_preserves_startup_overrides_on_partial_update(
    tmp_path: Path, sample_bundle
):
    client, _snapshot_service = _make_client(
        tmp_path,
        bundle=sample_bundle,
        metadata=SnapshotMetadata(
            status="ready",
            fetched_at="2026-03-28T12:00:00Z",
            cache_hit=True,
            refreshing=False,
            last_error=None,
        ),
        preferences_override={
            "account_id": "U20958465",
            "read_only": True,
            "watchlist_name": "default watchlist",
        },
    )

    response = client.post("/api/settings", json={"notes": "operator note"})
    assert response.status_code == 200
    payload = response.get_json()

    assert payload["account_id"] == "U20958465"
    assert payload["read_only"] is True
    assert payload["watchlist_name"] == "default watchlist"
    assert payload["notes"] == "operator note"
    assert payload["snapshot_reload_required"] is False
