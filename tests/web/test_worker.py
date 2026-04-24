from __future__ import annotations

from pathlib import Path

from src.web.ibkr_dashboard.job_store import RefreshJobRequest, RefreshJobStore
from src.web.ibkr_dashboard.settings import DashboardSettings
from src.web.ibkr_dashboard.worker import run_once


def test_worker_completes_job(tmp_path: Path, monkeypatch):
    store = RefreshJobStore(tmp_path / "jobs.sqlite")
    settings = DashboardSettings(runtime_dir=tmp_path / "runtime")
    store.enqueue(
        RefreshJobRequest(
            scope="ticker_list",
            tickers=("7203.T",),
            results_dir="results-a",
            watchlist_name=None,
            quick_mode=True,
            refresh_limit=5,
            max_age_days=14,
        )
    )
    monkeypatch.setattr(
        "src.web.ibkr_dashboard.worker._run_analysis_sync",
        lambda ticker, quick_mode, *, runtime_services: {"ticker": ticker},
    )
    monkeypatch.setattr(
        "src.web.ibkr_dashboard.worker._save_result_sync",
        lambda result, ticker, quick_mode, *, results_dir: Path(
            f"{results_dir}/{ticker}.json"
        ),
    )
    assert run_once(store, settings) is True
    job = store.list_jobs()[0]
    assert job["status"] == "completed"
    assert (
        store.get_job(job["job_id"])["tickers"][0]["output_path"]
        == "results-a/7203.T.json"
    )


def test_worker_marks_partial_when_one_ticker_fails(tmp_path: Path, monkeypatch):
    store = RefreshJobStore(tmp_path / "jobs.sqlite")
    settings = DashboardSettings(runtime_dir=tmp_path / "runtime")
    store.enqueue(
        RefreshJobRequest(
            scope="ticker_list",
            tickers=("7203.T", "MEGP.L"),
            results_dir="results-b",
            watchlist_name=None,
            quick_mode=True,
            refresh_limit=5,
            max_age_days=14,
        )
    )

    def fake_run(ticker, quick_mode, *, runtime_services):
        if ticker == "MEGP.L":
            raise RuntimeError("boom")
        return {"ticker": ticker}

    monkeypatch.setattr("src.web.ibkr_dashboard.worker._run_analysis_sync", fake_run)
    monkeypatch.setattr(
        "src.web.ibkr_dashboard.worker._save_result_sync",
        lambda result, ticker, quick_mode, *, results_dir: Path(
            f"{results_dir}/{ticker}.json"
        ),
    )
    run_once(store, settings)
    job = store.list_jobs()[0]
    assert job["status"] == "partial"
    failure = store.get_job(job["job_id"])["tickers"][1]
    assert failure["status"] == "failed"
    assert (
        failure["error"]
        == "Error in dashboard refresh job: RuntimeError (preview: boom)"
    )


def test_worker_passes_explicit_runtime_services(tmp_path: Path, monkeypatch):
    store = RefreshJobStore(tmp_path / "jobs.sqlite")
    settings = DashboardSettings(runtime_dir=tmp_path / "runtime")
    store.enqueue(
        RefreshJobRequest(
            scope="ticker_list",
            tickers=("7203.T",),
            results_dir="results-c",
            watchlist_name=None,
            quick_mode=True,
            refresh_limit=5,
            max_age_days=14,
        )
    )

    seen = {}
    sentinel_runtime = object()

    def fake_run(ticker, quick_mode, *, runtime_services):
        seen["runtime_services"] = runtime_services
        return {"ticker": ticker}

    monkeypatch.setattr("src.web.ibkr_dashboard.worker._run_analysis_sync", fake_run)
    monkeypatch.setattr(
        "src.web.ibkr_dashboard.worker._save_result_sync",
        lambda result, ticker, quick_mode, *, results_dir: Path(
            f"{results_dir}/{ticker}.json"
        ),
    )

    assert run_once(store, settings, runtime_services=sentinel_runtime) is True
    assert seen["runtime_services"] is sentinel_runtime
