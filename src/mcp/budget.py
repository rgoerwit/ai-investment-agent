from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.mcp.config import MCPServerSpec
from src.mcp.errors import parse_mcp_tool_name
from src.tooling.runtime import ToolCallBlocked, ToolInvocation, ToolResult

if TYPE_CHECKING:
    from src.mcp.client import MCPRuntime

logger = structlog.get_logger(__name__)


class BudgetTracker:
    """Daily and per-run call limits backed by a local SQLite database."""

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


class MCPBudgetHook:
    """Tool-execution-service hook that enforces MCP budgets uniformly.

    Recognizes invocations whose name matches ``mcp__<server>__<tool>``.
    ``before()`` blocks the call when the per-run or per-day limit is
    exhausted; ``after()`` increments the counter for any non-blocked result
    (including upstream ``isError=true`` payloads, which still consume
    upstream quota).

    Wire alongside :class:`ContentInspectionHook` on the shared
    :class:`ToolExecutionService` so audit, argument policy, content
    inspection, and budget all observe MCP calls from a single chokepoint.
    """

    def __init__(self, runtime: MCPRuntime) -> None:
        self._runtime = runtime

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        parsed = parse_mcp_tool_name(call.name)
        if parsed is None:
            return call
        server_id, _ = parsed
        spec = self._runtime.specs.get(server_id)
        if spec is None:
            return call
        if not self._runtime.budget.can_call(server_id, spec):
            logger.warning(
                "mcp_budget_exhausted",
                server_id=server_id,
                tool=call.name,
                agent_key=call.agent_key,
            )
            raise ToolCallBlocked(f"MCP budget exhausted for {server_id}")
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        parsed = parse_mcp_tool_name(call.name)
        if parsed is None:
            return result
        if result.blocked:
            return result
        server_id, _ = parsed
        if server_id not in self._runtime.specs:
            return result
        self._runtime.budget.record_upstream_consumption(server_id)
        return result
