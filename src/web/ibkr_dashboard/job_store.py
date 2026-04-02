from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

JobStatus = Literal["queued", "running", "completed", "partial", "failed", "cancelled"]
TickerStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    scope TEXT NOT NULL,
    results_dir TEXT NOT NULL DEFAULT 'results',
    watchlist_name TEXT,
    quick_mode INTEGER NOT NULL DEFAULT 1,
    refresh_limit INTEGER NOT NULL,
    max_age_days INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT
);
CREATE TABLE IF NOT EXISTS job_tickers (
    job_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    output_path TEXT,
    PRIMARY KEY (job_id, ticker),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);
"""


@dataclass(frozen=True)
class RefreshJobRequest:
    scope: str
    tickers: tuple[str, ...]
    results_dir: str
    watchlist_name: str | None
    quick_mode: bool
    refresh_limit: int
    max_age_days: int


@dataclass(frozen=True)
class QueuedRefreshJob:
    job_id: str
    request: RefreshJobRequest


class RefreshJobStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate_schema(conn)

    def enqueue(self, request: RefreshJobRequest) -> str:
        job_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        deduped_tickers = tuple(dict.fromkeys(request.tickers))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, created_at, scope, results_dir, watchlist_name,
                    quick_mode, refresh_limit, max_age_days, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued')
                """,
                (
                    job_id,
                    created_at,
                    request.scope,
                    request.results_dir,
                    request.watchlist_name,
                    1 if request.quick_mode else 0,
                    request.refresh_limit,
                    request.max_age_days,
                ),
            )
            conn.executemany(
                """
                INSERT INTO job_tickers (job_id, ticker, status)
                VALUES (?, ?, 'pending')
                """,
                [(job_id, ticker) for ticker in deduped_tickers],
            )
        return job_id

    def list_jobs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            tickers = conn.execute(
                """
                SELECT ticker, status, error, output_path
                FROM job_tickers
                WHERE job_id = ?
                ORDER BY ticker
                """,
                (job_id,),
            ).fetchall()
            payload = dict(row)
            payload["tickers"] = [dict(ticker_row) for ticker_row in tickers]
            return payload

    def claim_next(self) -> QueuedRefreshJob | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT *
                FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """,
            ).fetchone()
            if row is None:
                return None

            claimed = conn.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = ?
                WHERE job_id = ? AND status = 'queued'
                """,
                (datetime.now(UTC).isoformat(), row["job_id"]),
            )
            if claimed.rowcount != 1:
                return None

            ticker_rows = conn.execute(
                """
                SELECT ticker
                FROM job_tickers
                WHERE job_id = ?
                ORDER BY ticker
                """,
                (row["job_id"],),
            ).fetchall()
            return QueuedRefreshJob(
                job_id=row["job_id"],
                request=RefreshJobRequest(
                    scope=row["scope"],
                    tickers=tuple(ticker_row["ticker"] for ticker_row in ticker_rows),
                    results_dir=row["results_dir"],
                    watchlist_name=row["watchlist_name"],
                    quick_mode=bool(row["quick_mode"]),
                    refresh_limit=row["refresh_limit"],
                    max_age_days=row["max_age_days"],
                ),
            )

    def update_ticker_status(
        self,
        job_id: str,
        ticker: str,
        status: TickerStatus,
        error: str | None = None,
        output_path: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE job_tickers
                SET status = ?, error = ?, output_path = ?
                WHERE job_id = ? AND ticker = ?
                """,
                (status, error, output_path, job_id, ticker),
            )

    def complete_job(
        self,
        job_id: str,
        *,
        status: JobStatus,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, finished_at = ?, error = ?
                WHERE job_id = ?
                """,
                (status, datetime.now(UTC).isoformat(), error, job_id),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        if "results_dir" not in columns:
            conn.execute(
                "ALTER TABLE jobs ADD COLUMN results_dir TEXT NOT NULL DEFAULT 'results'"
            )
