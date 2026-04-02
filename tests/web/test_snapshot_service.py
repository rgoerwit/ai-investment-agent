from __future__ import annotations

import threading
import time
from pathlib import Path

from src.web.ibkr_dashboard.settings import DashboardPreferences, DashboardSettings
from src.web.ibkr_dashboard.snapshot_service import DashboardSnapshotService


def test_first_request_returns_loading_then_ready(
    tmp_path: Path,
    sample_bundle,
    monkeypatch,
):
    service = DashboardSnapshotService(
        DashboardSettings(
            results_dir=tmp_path / "results", runtime_dir=tmp_path / "runtime"
        )
    )
    started = threading.Event()
    release = threading.Event()

    async def fake_load():
        started.set()
        release.wait(1.0)
        return sample_bundle

    monkeypatch.setattr(service, "_load_snapshot", fake_load)

    bundle, meta = service.load_snapshot_sync()
    assert bundle is None
    assert meta.status == "loading"
    assert started.wait(1.0)

    release.set()
    assert service.wait_until_idle(1.0)

    bundle, meta = service.load_snapshot_sync()
    assert bundle is sample_bundle
    assert meta.status == "ready"
    assert meta.cache_hit is True


def test_inflight_load_is_started_once(
    tmp_path: Path,
    sample_bundle,
    monkeypatch,
):
    service = DashboardSnapshotService(
        DashboardSettings(
            results_dir=tmp_path / "results", runtime_dir=tmp_path / "runtime"
        )
    )
    started = threading.Event()
    release = threading.Event()
    calls = 0

    async def fake_load():
        nonlocal calls
        calls += 1
        started.set()
        release.wait(1.0)
        return sample_bundle

    monkeypatch.setattr(service, "_load_snapshot", fake_load)

    first_bundle, first_meta = service.load_snapshot_sync()
    assert started.wait(1.0)
    second_bundle, second_meta = service.load_snapshot_sync()

    assert first_bundle is None
    assert first_meta.status == "loading"
    assert second_bundle is None
    assert second_meta.status == "loading"
    assert calls == 1

    release.set()
    assert service.wait_until_idle(1.0)


def test_failed_refresh_preserves_cached_bundle(
    tmp_path: Path,
    sample_bundle,
    monkeypatch,
):
    service = DashboardSnapshotService(
        DashboardSettings(
            results_dir=tmp_path / "results", runtime_dir=tmp_path / "runtime"
        )
    )

    async def load_ok():
        return sample_bundle

    monkeypatch.setattr(service, "_load_snapshot", load_ok)
    service.load_snapshot_sync()
    assert service.wait_until_idle(1.0)

    async def load_fail():
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "_load_snapshot", load_fail)

    bundle, meta = service.load_snapshot_sync(force=True)
    assert bundle is sample_bundle
    assert meta.refreshing is True
    assert service.wait_until_idle(1.0)

    bundle, meta = service.load_snapshot_sync()
    assert bundle is sample_bundle
    assert meta.status == "ready"
    assert meta.last_error == "boom"


def test_results_dir_mtime_invalidates_cache(
    tmp_path: Path,
    sample_bundle,
    monkeypatch,
):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "MEGP.L_20260328_000000_analysis.json").write_text(
        "{}",
        encoding="utf-8",
    )
    service = DashboardSnapshotService(
        DashboardSettings(results_dir=results_dir, runtime_dir=tmp_path / "runtime")
    )

    async def load_ok():
        return sample_bundle

    monkeypatch.setattr(service, "_load_snapshot", load_ok)
    service.load_snapshot_sync()
    assert service.wait_until_idle(1.0)
    assert service.get_cached_snapshot() is sample_bundle

    time.sleep(0.01)
    (results_dir / "7203.T_20260329_000000_analysis.json").write_text(
        "{}",
        encoding="utf-8",
    )

    assert service.get_cached_snapshot() is None


def test_apply_preferences_clears_cached_snapshot(
    tmp_path: Path,
    sample_bundle,
    monkeypatch,
):
    service = DashboardSnapshotService(
        DashboardSettings(
            results_dir=tmp_path / "results", runtime_dir=tmp_path / "runtime"
        )
    )

    async def load_ok():
        return sample_bundle

    monkeypatch.setattr(service, "_load_snapshot", load_ok)
    service.load_snapshot_sync()
    assert service.wait_until_idle(1.0)
    assert service.get_cached_snapshot() is sample_bundle

    invalidated = service.apply_preferences(
        DashboardPreferences(
            account_id="U20958465",
            read_only=True,
            watchlist_name="default watchlist",
            max_age_days=21,
            quick_mode_default=True,
            refresh_limit=5,
        )
    )

    assert invalidated is True
    assert service.get_cached_snapshot() is None
    preferences = service.current_preferences()
    assert preferences.account_id == "U20958465"
    assert preferences.read_only is True
    assert preferences.watchlist_name == "default watchlist"


def test_apply_preferences_preserves_cached_snapshot_for_non_bundle_fields(
    tmp_path: Path,
    sample_bundle,
    monkeypatch,
):
    service = DashboardSnapshotService(
        DashboardSettings(
            results_dir=tmp_path / "results", runtime_dir=tmp_path / "runtime"
        )
    )

    async def load_ok():
        return sample_bundle

    monkeypatch.setattr(service, "_load_snapshot", load_ok)
    service.load_snapshot_sync()
    assert service.wait_until_idle(1.0)
    assert service.get_cached_snapshot() is sample_bundle

    invalidated = service.apply_preferences(
        DashboardPreferences(
            account_id=None,
            read_only=False,
            watchlist_name=None,
            max_age_days=14,
            quick_mode_default=False,
            refresh_limit=25,
            notes="operator note",
        )
    )

    assert invalidated is False
    assert service.get_cached_snapshot() is sample_bundle


def test_failed_initial_load_returns_error_after_background_completion(
    tmp_path: Path,
    monkeypatch,
):
    service = DashboardSnapshotService(
        DashboardSettings(
            results_dir=tmp_path / "results", runtime_dir=tmp_path / "runtime"
        )
    )

    async def load_fail():
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "_load_snapshot", load_fail)

    bundle, meta = service.load_snapshot_sync()
    assert bundle is None
    assert meta.status == "loading"
    assert service.wait_until_idle(1.0)

    bundle, meta = service.load_snapshot_sync()
    assert bundle is None
    assert meta.status == "error"
    assert meta.last_error == "boom"


def test_inflight_invalidating_preference_change_discards_stale_loaded_bundle(
    tmp_path: Path,
    monkeypatch,
):
    service = DashboardSnapshotService(
        DashboardSettings(
            results_dir=tmp_path / "results", runtime_dir=tmp_path / "runtime"
        )
    )
    started = threading.Event()
    release = threading.Event()
    calls: list[str | None] = []

    async def fake_load():
        preferences = service.current_preferences()
        calls.append(preferences.account_id)
        if len(calls) == 1:
            started.set()
            release.wait(1.0)
            return "stale-bundle"
        return "fresh-bundle"

    monkeypatch.setattr(service, "_load_snapshot", fake_load)

    bundle, meta = service.load_snapshot_sync()
    assert bundle is None
    assert meta.status == "loading"
    assert started.wait(1.0)

    invalidated = service.apply_preferences(
        DashboardPreferences(
            account_id="U20958465",
            read_only=True,
            watchlist_name="default watchlist",
            max_age_days=21,
            quick_mode_default=True,
            refresh_limit=5,
        )
    )
    assert invalidated is True

    release.set()
    assert service.wait_until_idle(1.0)
    assert service.get_cached_snapshot() is None

    bundle, meta = service.load_snapshot_sync()
    assert bundle is None
    assert meta.status == "loading"
    assert service.wait_until_idle(1.0)

    bundle, meta = service.load_snapshot_sync()
    assert bundle == "fresh-bundle"
    assert meta.status == "ready"
    assert calls == [None, "U20958465"]
