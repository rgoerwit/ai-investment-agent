from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path

from src.mcp.config import MCPServerSpec


class BudgetTracker:
    """Daily and per‑run call limits using a local SQLite database."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_db()
        self._run_calls: dict[str, int] = {}

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
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
        conn.commit()
        conn.close()

    def can_call(self, server_id: str, spec: MCPServerSpec) -> bool:
        """Return True if the daily and per‑run budgets are not yet exhausted."""
        today = datetime.date.today().isoformat()
        conn = sqlite3.connect(self._db_path)
        cur = conn.execute(
            "SELECT call_count FROM mcp_usage WHERE usage_day=? AND server_id=?",
            (today, server_id),
        )
        row = cur.fetchone()
        conn.close()
        daily = row[0] if row else 0

        run = self._run_calls.get(server_id, 0)

        if spec.daily_call_limit > 0 and daily >= spec.daily_call_limit:
            return False
        if spec.per_run_limit > 0 and run >= spec.per_run_limit:
            return False
        return True

    def record_call(self, server_id: str) -> None:
        today = datetime.date.today().isoformat()
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            """
            INSERT INTO mcp_usage (usage_day, server_id, call_count)
            VALUES (?, ?, 1)
            ON CONFLICT(usage_day, server_id)
            DO UPDATE SET call_count = call_count + 1
            """,
            (today, server_id),
        )
        conn.commit()
        conn.close()
        self._run_calls[server_id] = self._run_calls.get(server_id, 0) + 1
