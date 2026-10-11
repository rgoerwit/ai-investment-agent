"""BudgetTracker: one atomic admission per tools/call, shared through SQLite.

The tracker used to check (``can_call``) and record in two transactions with
awaits between them, so concurrent callers all passed the check and every one
was counted past the limit. ``try_reserve`` now checks and increments the daily
and the per-run count inside one ``BEGIN IMMEDIATE`` transaction: the counter
counts admitted reservations. These tests drive the synchronous core directly,
from real threads; the runtime-level behaviour (off the loop, refusals, late
completion) lives in ``test_mcp_budget_accounting.py``.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from src.config import config
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline
from src.mcp import budget as budget_module
from src.mcp.budget import BudgetTracker
from src.mcp.config import MCPServerSpec

_SERVER = "fmp_remote"
_BUSY = 5.0  # generous: only the locked-store test waits on it, with its own value


def _spec(*, daily: int = 0, per_run: int = 0) -> MCPServerSpec:
    return MCPServerSpec(
        id=_SERVER,
        description="FMP",
        transport="streamable_http",
        base_url="https://example.test/mcp",
        daily_call_limit=daily,
        per_run_limit=per_run,
    )


def _daily(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT call_count FROM mcp_usage WHERE server_id=?", (_SERVER,)
        ).fetchone()
    conn.close()
    return row[0] if row else 0


def _reserve(tracker: BudgetTracker, spec: MCPServerSpec) -> bool:
    return tracker.try_reserve(_SERVER, spec, busy_timeout=_BUSY)


# --- Atomicity ------------------------------------------------------------------


@pytest.mark.parametrize("limit", ["daily", "per_run"])
def test_quota_one_admits_exactly_one_of_many_concurrent_threads(tmp_path, limit):
    db = tmp_path / "mcp.db"
    tracker = BudgetTracker(str(db))
    spec = _spec(**{limit: 1})
    start = threading.Barrier(8)
    results: list[bool] = []

    def contend() -> None:
        start.wait()
        results.append(_reserve(tracker, spec))

    threads = [threading.Thread(target=contend) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [False] * 7 + [True]
    assert _daily(db) == tracker._run_calls[_SERVER] == 1


def test_two_trackers_share_the_daily_limit_and_keep_their_own_run_counts(tmp_path):
    db = tmp_path / "mcp.db"
    first, second = BudgetTracker(str(db)), BudgetTracker(str(db))
    spec = _spec(daily=3, per_run=2)

    assert [_reserve(first, spec) for _ in range(3)] == [True, True, False]
    assert [_reserve(second, spec) for _ in range(2)] == [True, False]

    assert _daily(db) == 3
    assert first._run_calls[_SERVER] == 2
    assert second._run_calls[_SERVER] == 1


@pytest.mark.parametrize(
    ("daily", "per_run"), [(5, 1), (1, 5)], ids=["per_run_spent", "daily_spent"]
)
def test_a_refusal_moves_neither_count(tmp_path, daily, per_run):
    db = tmp_path / "mcp.db"
    tracker = BudgetTracker(str(db))
    spec = _spec(daily=daily, per_run=per_run)

    assert _reserve(tracker, spec) is True
    assert _reserve(tracker, spec) is False

    assert _daily(db) == tracker._run_calls[_SERVER] == 1


def test_exactly_at_the_limit_is_admitted_and_one_past_is_not(tmp_path):
    tracker = BudgetTracker(str(tmp_path / "mcp.db"))
    spec = _spec(daily=2, per_run=2)

    assert [_reserve(tracker, spec) for _ in range(3)] == [True, True, False]


def test_zero_limits_mean_unlimited(tmp_path):
    db = tmp_path / "mcp.db"
    tracker = BudgetTracker(str(db))

    assert all(_reserve(tracker, _spec()) for _ in range(5))
    assert _daily(db) == tracker._run_calls[_SERVER] == 5


def test_the_advisory_check_never_reserves(tmp_path):
    db = tmp_path / "mcp.db"
    tracker = BudgetTracker(str(db))
    spec = _spec(daily=1)

    for _ in range(3):
        assert tracker.can_call(_SERVER, spec, busy_timeout=_BUSY) is True
    assert _reserve(tracker, spec) is True
    assert tracker.can_call(_SERVER, spec, busy_timeout=_BUSY) is False
    assert _daily(db) == 1


def test_the_daily_count_survives_a_new_tracker(tmp_path):
    db = tmp_path / "mcp.db"
    spec = _spec(daily=1)
    assert _reserve(BudgetTracker(str(db)), spec) is True

    assert _reserve(BudgetTracker(str(db)), spec) is False


# --- Store lifecycle ------------------------------------------------------------


def test_constructing_a_tracker_touches_no_database(tmp_path):
    db = tmp_path / "nested" / "mcp.db"

    BudgetTracker(str(db))

    assert db.parent.is_dir()
    assert not db.exists()


def test_the_first_transaction_prunes_rows_past_retention(tmp_path):
    db = tmp_path / "mcp.db"
    assert _reserve(BudgetTracker(str(db)), _spec()) is True
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO mcp_usage (usage_day, server_id, call_count) "
            "VALUES ('2000-01-01', 'old', 9)"
        )
    conn.close()

    assert _reserve(BudgetTracker(str(db), retention_days=7), _spec()) is True

    with sqlite3.connect(db) as conn:
        servers = {row[0] for row in conn.execute("SELECT server_id FROM mcp_usage")}
    conn.close()
    assert servers == {_SERVER}


# --- Failures ---------------------------------------------------------------------


def test_a_locked_store_gives_up_after_the_busy_timeout_and_counts_nothing(tmp_path):
    db = tmp_path / "mcp.db"
    tracker = BudgetTracker(str(db))
    spec = _spec()
    assert _reserve(tracker, spec) is True
    holder = sqlite3.connect(db, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            tracker.try_reserve(_SERVER, spec, busy_timeout=0.01)
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    assert _daily(db) == tracker._run_calls[_SERVER] == 1


def test_a_corrupt_store_raises_and_counts_nothing(tmp_path):
    db = tmp_path / "mcp.db"
    db.write_bytes(b"this is not a sqlite database" * 64)
    tracker = BudgetTracker(str(db))

    with pytest.raises(sqlite3.DatabaseError):
        _reserve(tracker, _spec())

    assert tracker._run_calls == {}


class _CommitFails:
    """A connection whose COMMIT raises, as a full disk or I/O error would."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, *args: Any) -> Any:
        if sql == "COMMIT":
            raise sqlite3.OperationalError("disk I/O error")
        return self._conn.execute(sql, *args)

    def close(self) -> None:
        self._conn.close()


def test_a_failed_commit_moves_neither_count(tmp_path, monkeypatch):
    db = tmp_path / "mcp.db"
    tracker = BudgetTracker(str(db))
    spec = _spec(per_run=1)
    real_connect = tracker._connect
    monkeypatch.setattr(
        tracker, "_connect", lambda busy: _CommitFails(real_connect(busy))
    )

    with pytest.raises(sqlite3.OperationalError, match="disk"):
        _reserve(tracker, spec)

    monkeypatch.undo()
    assert tracker._run_calls.get(_SERVER, 0) == 0
    assert _reserve(tracker, spec) is True  # nothing was spent
    assert _daily(db) == 1


# --- The admission bound ----------------------------------------------------------


def test_the_admission_bound_is_one_mcp_request_allowance_less_its_margin(
    monkeypatch,
):
    monkeypatch.setattr(config, "mcp_http_timeout_seconds", 4.0)
    monkeypatch.setattr(config, "deadline_margin_fraction", 0.25)
    monkeypatch.setattr(config, "llm_dispatch_margin_seconds", 100.0)

    assert budget_module.admission_bounds() == (4.0, 3.0)


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    clock = [1_000.0]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


def test_the_admission_bound_shrinks_to_the_time_left(monkeypatch, fake_clock):
    monkeypatch.setattr(config, "mcp_http_timeout_seconds", 4.0)
    monkeypatch.setattr(config, "deadline_margin_fraction", 0.25)
    monkeypatch.setattr(config, "llm_dispatch_margin_seconds", 100.0)

    with enclosing_deadline(10.0):
        fake_clock[0] += 9.0
        assert budget_module.admission_bounds() == (1.0, 0.75)


@pytest.mark.parametrize("overrun", [0.0, 1.0], ids=["exactly_spent", "past"])
def test_a_spent_allowance_admits_nothing(fake_clock, overrun):
    with enclosing_deadline(10.0):
        fake_clock[0] += 10.0 + overrun
        with pytest.raises(TimeoutError):
            budget_module.admission_bounds()
