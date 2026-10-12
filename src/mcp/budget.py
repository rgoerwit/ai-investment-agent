from __future__ import annotations

import datetime as dt
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import TypeVar

from src.blocking_io import (
    DEFAULT_WORKER_SLOTS,
    BlockingCallPolicy,
    InFlightLimit,
    run_blocking_call,
)
from src.config import config as settings_config
from src.llm_runtime.timeouts import deadline_margin, require_allowance
from src.mcp.config import MCPServerSpec

T = TypeVar("T")

# A budget transaction touches one row of a local file; more unfinished workers
# than the default slots means the store is stuck, and further admissions fail
# fast. Its own pool, so a stuck store never starves document extraction.
_BUDGET_DB_IN_FLIGHT = InFlightLimit(DEFAULT_WORKER_SLOTS)


def admission_bounds() -> tuple[float, float]:
    """``(hard bound, SQLite busy timeout)`` in seconds for one budget transaction.

    An admission step must fit inside one MCP request's allowance, so the hard
    bound is ``MCP_HTTP_TIMEOUT_SECONDS`` clamped to the operational time left.
    The busy timeout stops ``deadline_margin`` short of it, so SQLite's own
    "database is locked" ends the worker before the hard bound abandons it.
    Raises ``TimeoutError`` when the time left is already spent.
    """
    bound = require_allowance(
        settings_config.mcp_http_timeout_seconds, label="mcp_budget"
    )
    return bound, bound - deadline_margin(bound)


class BudgetTracker:
    """Daily and per-run call limits backed by a local SQLite database.

    The counts are **admitted reservations**. ``try_reserve`` checks both
    limits and takes one call from each in a single ``BEGIN IMMEDIATE``
    transaction, so concurrent callers -- in this process or another sharing
    the file -- cannot both take the last call. A reservation is never
    refunded: one whose call is then not sent (cancelled, out of time, or
    committed by a worker after its caller gave up) still counts. Trackers on
    one file share the daily limit; each keeps its own per-run count.

    ``MCPRuntime`` is the only client. It awaits ``reserve`` immediately before
    each tools/call and ``has_room`` as an advisory check before opening a
    session; both run the transaction off the event loop.
    """

    def __init__(self, db_path: str, *, retention_days: int = 7) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._retention_days = retention_days
        self._run_calls: dict[str, int] = {}
        # Schema and retention are applied by the first transaction, on a
        # worker, so constructing a tracker does no database I/O.
        self._prepared = False

    async def reserve(self, server_id: str, spec: MCPServerSpec) -> bool:
        """``try_reserve`` on a worker thread, bounded by ``admission_bounds``."""
        return await self._in_worker(
            "mcp_budget.reserve",
            lambda busy: self.try_reserve(server_id, spec, busy_timeout=busy),
        )

    async def has_room(self, server_id: str, spec: MCPServerSpec) -> bool:
        """``can_call`` on a worker thread, bounded by ``admission_bounds``."""
        return await self._in_worker(
            "mcp_budget.check",
            lambda busy: self.can_call(server_id, spec, busy_timeout=busy),
        )

    def try_reserve(
        self, server_id: str, spec: MCPServerSpec, *, busy_timeout: float
    ) -> bool:
        """Take one call from both limits if both have room; atomically."""
        return self._admit(server_id, spec, busy_timeout=busy_timeout, reserve=True)

    def can_call(
        self, server_id: str, spec: MCPServerSpec, *, busy_timeout: float
    ) -> bool:
        """Whether both limits have room now. Advisory: it reserves nothing."""
        return self._admit(server_id, spec, busy_timeout=busy_timeout, reserve=False)

    async def _in_worker(self, label: str, work: Callable[[float], T]) -> T:
        bound, busy_timeout = admission_bounds()
        policy = BlockingCallPolicy(label, bound, in_flight=_BUDGET_DB_IN_FLIGHT)
        return await run_blocking_call(policy, lambda: work(busy_timeout))

    def _admit(
        self,
        server_id: str,
        spec: MCPServerSpec,
        *,
        busy_timeout: float,
        reserve: bool,
    ) -> bool:
        today = dt.date.today().isoformat()
        # Closing without COMMIT rolls back, so a failure moves no count.
        with closing(self._connect(busy_timeout)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._prepare(conn)
            row = conn.execute(
                "SELECT call_count FROM mcp_usage WHERE usage_day=? AND server_id=?",
                (today, server_id),
            ).fetchone()
            daily = row[0] if row else 0
            # The write lock BEGIN IMMEDIATE holds also serializes this
            # instance's per-run count: every worker of one tracker uses one file.
            run = self._run_calls.get(server_id, 0)
            admitted = _has_room(spec, daily=daily, run=run)
            if admitted and reserve:
                conn.execute(
                    """
                    INSERT INTO mcp_usage (usage_day, server_id, call_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(usage_day, server_id)
                    DO UPDATE SET call_count = call_count + 1
                    """,
                    (today, server_id),
                )
                self._run_calls[server_id] = run + 1
            try:
                conn.execute("COMMIT")
            except BaseException:
                if admitted and reserve:
                    self._run_calls[server_id] = run
                raise
            self._prepared = True
            return admitted

    def _connect(self, busy_timeout: float) -> sqlite3.Connection:
        # isolation_level=None: the module opens no implicit transaction, so
        # BEGIN IMMEDIATE and COMMIT above are the only ones.
        return sqlite3.connect(
            self._db_path, timeout=busy_timeout, isolation_level=None
        )

    def _prepare(self, conn: sqlite3.Connection) -> None:
        """Create the table and prune expired days, once per tracker."""
        if self._prepared:
            return
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
        cutoff = (dt.date.today() - dt.timedelta(days=self._retention_days)).isoformat()
        conn.execute("DELETE FROM mcp_usage WHERE usage_day < ?", (cutoff,))


def _has_room(spec: MCPServerSpec, *, daily: int, run: int) -> bool:
    """Both limits have room; a limit of 0 is unlimited."""
    if spec.daily_call_limit > 0 and daily >= spec.daily_call_limit:
        return False
    return not (spec.per_run_limit > 0 and run >= spec.per_run_limit)
