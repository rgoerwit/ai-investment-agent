from __future__ import annotations

import asyncio
import contextvars
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from src.blocking_io import (
    GAP_FILL_YFINANCE_INFO_POLICY,
    YFINANCE_INFO_POLICY,
    BlockingCallPolicy,
    BlockingCallSaturatedError,
    BlockingCallTimeoutError,
    InFlightLimit,
    run_blocking_call,
    run_blocking_call_sync,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEADLINE = 0.2
# Generous enough for a loaded CI box; still far below any stalled call.
TOLERANCE = 0.5


def _limited(label: str, max_in_flight: int = 2) -> BlockingCallPolicy:
    return BlockingCallPolicy(label, DEADLINE, in_flight=InFlightLimit(max_in_flight))


def _join_workers(label: str, timeout: float = 2.0) -> list[threading.Thread]:
    """Join every worker thread for *label*; return any still alive."""
    workers = [t for t in threading.enumerate() if t.name == f"blocking:{label}"]
    for worker in workers:
        worker.join(timeout)
    return [t for t in workers if t.is_alive()]


@pytest.mark.asyncio
async def test_run_blocking_call_returns_value() -> None:
    result = await run_blocking_call(
        BlockingCallPolicy("test_blocking_success", 1.0),
        lambda: "ok",
    )

    assert result == "ok"


@pytest.mark.asyncio
async def test_run_blocking_call_times_out() -> None:
    with pytest.raises(TimeoutError):
        await run_blocking_call(
            BlockingCallPolicy("test_blocking_timeout", 0.01),
            lambda: time.sleep(0.1),
        )


@pytest.mark.asyncio
async def test_run_blocking_call_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError):
        await run_blocking_call(
            BlockingCallPolicy("test_blocking_bad_timeout", 0),
            lambda: "never scheduled",
        )


# ---------------------------------------------------------------------------
# InFlightLimit and the policy plumbing
# ---------------------------------------------------------------------------


def test_in_flight_limit_rejects_non_positive_and_over_release() -> None:
    with pytest.raises(ValueError):
        InFlightLimit(0)
    limit = InFlightLimit(1)
    assert limit.try_acquire()
    assert not limit.try_acquire()
    limit.release()
    with pytest.raises(RuntimeError):
        limit.release()


def test_with_label_and_with_timeout_share_one_limit() -> None:
    policy = _limited("shared")
    derived = policy.with_label("shared:7203.T").with_timeout(9.0)
    assert derived.in_flight is policy.in_flight
    assert derived.hard_timeout_seconds == 9.0
    # Unlimited policies stay unlimited (existing call sites are unchanged).
    assert YFINANCE_INFO_POLICY.in_flight is None


def test_gap_fill_policy_keeps_the_five_second_budget() -> None:
    assert GAP_FILL_YFINANCE_INFO_POLICY.hard_timeout_seconds == 5.0
    assert GAP_FILL_YFINANCE_INFO_POLICY.in_flight is not None


# ---------------------------------------------------------------------------
# run_blocking_call_sync
# ---------------------------------------------------------------------------


def test_sync_fast_worker_returns_value() -> None:
    assert run_blocking_call_sync(_limited("sync_fast"), lambda: 42) == 42
    assert _join_workers("sync_fast") == []


def test_sync_requires_an_in_flight_limit() -> None:
    with pytest.raises(ValueError):
        run_blocking_call_sync(BlockingCallPolicy("sync_unlimited", 1.0), lambda: 1)
    with pytest.raises(ValueError):
        run_blocking_call_sync(
            BlockingCallPolicy("sync_zero", 0, in_flight=InFlightLimit(1)), lambda: 1
        )


def test_sync_stalled_worker_caller_returns_by_deadline_then_releases() -> None:
    """(a) the caller returns at the deadline; releasing the stall leaves no thread."""
    policy = _limited("sync_stall")
    release = threading.Event()
    started = time.monotonic()
    with pytest.raises(BlockingCallTimeoutError):
        run_blocking_call_sync(policy, release.wait)
    elapsed = time.monotonic() - started

    assert DEADLINE <= elapsed < DEADLINE + TOLERANCE
    assert policy.in_flight is not None
    assert policy.in_flight.in_flight == 1  # the orphan still holds its slot

    release.set()
    assert _join_workers("sync_stall") == []
    assert policy.in_flight.in_flight == 0


def test_sync_repeated_stalls_stay_within_the_bound() -> None:
    """(b) stalls past the limit fail fast instead of spawning more threads."""
    policy = _limited("sync_repeat", max_in_flight=2)
    release = threading.Event()
    before = threading.active_count()
    try:
        for _ in range(2):
            with pytest.raises(BlockingCallTimeoutError):
                run_blocking_call_sync(policy, release.wait)
        for _ in range(5):
            started = time.monotonic()
            with pytest.raises(BlockingCallSaturatedError):
                run_blocking_call_sync(policy, release.wait)
            assert time.monotonic() - started < TOLERANCE
        assert policy.in_flight is not None
        assert policy.in_flight.in_flight == 2
        assert threading.active_count() - before <= 2
    finally:
        release.set()
    assert _join_workers("sync_repeat") == []
    # Slots come back once the stalls clear.
    assert run_blocking_call_sync(policy, lambda: "recovered") == "recovered"


def test_sync_worker_exception_propagates_unchanged() -> None:
    def _boom() -> None:
        raise ValueError("provider said no")

    with pytest.raises(ValueError, match="provider said no"):
        run_blocking_call_sync(_limited("sync_error"), _boom)

    # A TimeoutError raised by the call itself is not relabelled as ours.
    def _own_timeout() -> None:
        raise TimeoutError("socket timeout inside the library")

    with pytest.raises(TimeoutError, match="inside the library") as excinfo:
        run_blocking_call_sync(_limited("sync_own_timeout"), _own_timeout)
    assert not isinstance(excinfo.value, BlockingCallTimeoutError)


def test_sync_worker_sees_the_callers_context() -> None:
    var: contextvars.ContextVar[str] = contextvars.ContextVar("scoped", default="unset")
    token = var.set("scoped-services")
    try:
        assert (
            run_blocking_call_sync(_limited("sync_ctx"), var.get) == "scoped-services"
        )
    finally:
        var.reset(token)


# ---------------------------------------------------------------------------
# run_blocking_call with an InFlightLimit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_limited_fast_worker_returns_value() -> None:
    assert await run_blocking_call(_limited("async_fast"), lambda: "ok") == "ok"


@pytest.mark.asyncio
async def test_async_limited_stall_returns_by_deadline_and_bounds_orphans() -> None:
    policy = _limited("async_stall", max_in_flight=2)
    release = threading.Event()
    try:
        for _ in range(2):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                await run_blocking_call(policy, release.wait)
            assert time.monotonic() - started < DEADLINE + TOLERANCE
        with pytest.raises(BlockingCallSaturatedError):
            await run_blocking_call(policy, release.wait)
        assert policy.in_flight is not None
        assert policy.in_flight.in_flight == 2
    finally:
        release.set()
    assert _join_workers("async_stall") == []
    assert policy.in_flight.in_flight == 0


@pytest.mark.asyncio
async def test_async_limited_exception_propagates() -> None:
    def _boom() -> None:
        raise KeyError("missing")

    with pytest.raises(KeyError):
        await run_blocking_call(_limited("async_error"), _boom)
    # The loop is still healthy afterwards.
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# (d) Interpreter exit with a stuck worker
# ---------------------------------------------------------------------------

_EXIT_CHILD = """
import json, sys, threading, time
from src.blocking_io import BlockingCallPolicy, InFlightLimit, run_blocking_call_sync

stuck = threading.Event()  # never set: the worker is stuck for good
policy = BlockingCallPolicy("exit_probe", 0.2, in_flight=InFlightLimit(1))
started = time.monotonic()
try:
    run_blocking_call_sync(policy, stuck.wait)
except TimeoutError:
    pass
print(json.dumps({"caller_seconds": time.monotonic() - started}), flush=True)
"""

# The pattern this change replaced: a ThreadPoolExecutor worker. The stall
# releases itself after 1.5 s so this contrast test stays bounded.
_EXECUTOR_CHILD = """
import json, threading, time
from concurrent.futures import ThreadPoolExecutor

stuck = threading.Event()
executor = ThreadPoolExecutor(max_workers=1)
future = executor.submit(stuck.wait, 1.5)
started = time.monotonic()
try:
    future.result(timeout=0.2)
except TimeoutError:
    pass
executor.shutdown(wait=False)
print(json.dumps({"caller_seconds": time.monotonic() - started}), flush=True)
"""


def _run_child(code: str, timeout: float) -> tuple[float, float]:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    started = time.monotonic()
    # posix_spawn path: close_fds=False and no cwd= (see .claude/rules/tests.md).
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        close_fds=False,
        check=False,
    )
    wall = time.monotonic() - started
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])["caller_seconds"], wall


def test_stuck_daemon_worker_does_not_hold_the_interpreter_open() -> None:
    """The caller returns at its deadline and the process then exits.

    The worker is a daemon thread, so finalization abandons it rather than
    joining it. The stalled work itself is never stopped — it just dies with
    the process.
    """
    caller_seconds, wall = _run_child(_EXIT_CHILD, timeout=30)
    assert caller_seconds < DEADLINE + TOLERANCE
    # Import cost dominates `wall`; the point is that it is not unbounded.
    assert wall < 30


def test_executor_worker_holds_interpreter_exit_until_it_finishes() -> None:
    """Contrast: why the sync path no longer uses ThreadPoolExecutor.

    Even with ``shutdown(wait=False)`` and a caller that returned at 0.2 s,
    ``concurrent.futures`` joins its workers at interpreter exit, so the
    process lives as long as the stalled call. A truly stuck worker would
    hang exit forever.
    """
    caller_seconds, wall = _run_child(_EXECUTOR_CHILD, timeout=30)
    assert caller_seconds < DEADLINE + TOLERANCE
    assert wall >= 1.5
