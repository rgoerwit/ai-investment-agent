from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

from src.mcp.config import MCPServerSpec


class BudgetTracker:
    """Daily and per-run call limits backed by a local SQLite database.

    ``MCPRuntime`` is its only client: it admits every attempt with
    ``can_call`` and records every tools/call as it is sent.
    """

    def __init__(self, db_path: str, *, retention_days: int = 7) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._retention_days = retention_days
        self._run_calls: dict[str, int] = {}
        self._init_db()
        self._cleanup_old_rows()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mcp_usage (
                    usage_day TEXT NOT NULL,
                    server_id TEXT NOT NULL,
                    call_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (usage_day, server_id)
                )
                """
            )

    def _cleanup_old_rows(self) -> None:
        cutoff = (dt.date.today() - dt.timedelta(days=self._retention_days)).isoformat()
        with self._connect() as conn:
            conn.execute("DELETE FROM mcp_usage WHERE usage_day < ?", (cutoff,))

    def can_call(self, server_id: str, spec: MCPServerSpec) -> bool:
        today = dt.date.today().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT call_count FROM mcp_usage WHERE usage_day=? AND server_id=?",
                (today, server_id),
            ).fetchone()

        daily = row[0] if row else 0
        run = self._run_calls.get(server_id, 0)

        if spec.daily_call_limit > 0 and daily >= spec.daily_call_limit:
            return False
        if spec.per_run_limit > 0 and run >= spec.per_run_limit:
            return False
        return True

    def record_upstream_consumption(self, server_id: str) -> None:
        today = dt.date.today().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO mcp_usage (usage_day, server_id, call_count)
                VALUES (?, ?, 1)
                ON CONFLICT(usage_day, server_id)
                DO UPDATE SET call_count = call_count + 1
                """,
                (today, server_id),
            )
        self._run_calls[server_id] = self._run_calls.get(server_id, 0) + 1
