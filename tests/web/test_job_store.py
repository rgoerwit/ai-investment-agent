from __future__ import annotations

import sqlite3
from pathlib import Path

from src.web.ibkr_dashboard.job_store import RefreshJobRequest, RefreshJobStore


def test_enqueue_and_list_jobs(tmp_path: Path):
    store = RefreshJobStore(tmp_path / "jobs.sqlite")
    job_id = store.enqueue(
        RefreshJobRequest(
            scope="ticker_list",
            tickers=("7203.T", "MEGP.L"),
            results_dir="results",
            watchlist_name="watchlist-2026",
            quick_mode=True,
            refresh_limit=5,
            max_age_days=14,
        )
    )
    jobs = store.list_jobs()
    assert jobs[0]["job_id"] == job_id
    assert jobs[0]["status"] == "queued"
    assert jobs[0]["results_dir"] == "results"


def test_claim_next_transitions_to_running(tmp_path: Path):
    store = RefreshJobStore(tmp_path / "jobs.sqlite")
    job_id = store.enqueue(
        RefreshJobRequest(
            scope="ticker_list",
            tickers=("7203.T",),
            results_dir="results",
            watchlist_name=None,
            quick_mode=True,
            refresh_limit=5,
            max_age_days=14,
        )
    )
    claimed = store.claim_next()
    assert claimed is not None
    assert claimed.job_id == job_id
    job = store.get_job(job_id)
    assert job["status"] == "running"


def test_complete_job_and_ticker_status_persist(tmp_path: Path):
    store = RefreshJobStore(tmp_path / "jobs.sqlite")
    job_id = store.enqueue(
        RefreshJobRequest(
            scope="ticker_list",
            tickers=("7203.T",),
            results_dir="results",
            watchlist_name=None,
            quick_mode=True,
            refresh_limit=5,
            max_age_days=14,
        )
    )
    store.claim_next()
    store.update_ticker_status(
        job_id, "7203.T", "completed", output_path="results/7203.T.json"
    )
    store.complete_job(job_id, status="completed")
    job = store.get_job(job_id)
    assert job["finished_at"] is not None
    assert job["tickers"][0]["output_path"] == "results/7203.T.json"


def test_claim_next_returns_none_for_empty_queue(tmp_path: Path):
    store = RefreshJobStore(tmp_path / "jobs.sqlite")
    assert store.claim_next() is None


def test_existing_job_store_is_migrated_to_add_results_dir(tmp_path: Path):
    db_path = tmp_path / "jobs.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                scope TEXT NOT NULL,
                watchlist_name TEXT,
                quick_mode INTEGER NOT NULL DEFAULT 1,
                refresh_limit INTEGER NOT NULL,
                max_age_days INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                error TEXT
            );
            CREATE TABLE job_tickers (
                job_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                error TEXT,
                output_path TEXT,
                PRIMARY KEY (job_id, ticker)
            );
            """
        )

    store = RefreshJobStore(db_path)
    job_id = store.enqueue(
        RefreshJobRequest(
            scope="ticker_list",
            tickers=("7203.T",),
            results_dir="custom-results",
            watchlist_name=None,
            quick_mode=True,
            refresh_limit=5,
            max_age_days=14,
        )
    )

    job = store.get_job(job_id)
    assert job is not None
    assert job["results_dir"] == "custom-results"
    claimed = store.claim_next()
    assert claimed is not None
    assert claimed.request.results_dir == "custom-results"
