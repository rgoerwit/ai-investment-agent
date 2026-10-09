"""Bounds on the synchronous Tavily entrypoint (``search_tavily_sync_inspected``).

The async entrypoints are covered by the inspection and toolkit suites; this file
owns the sync path, which runs on its own worker threads and so needs its own
proof that a stalled search cannot hold the caller, pile up threads, or keep the
interpreter alive.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import src.tavily_utils as tavily_utils
from src.blocking_io import BlockingCallPolicy, InFlightLimit

REPO_ROOT = Path(__file__).resolve().parents[1]
DEADLINE = 0.2
TOLERANCE = 0.5
LABEL = "tavily-test"


class _FakeTool:
    def __init__(self, behaviour: Any) -> None:
        self._behaviour = behaviour
        self.calls = 0

    def invoke(self, payload: dict[str, str]) -> Any:
        self.calls += 1
        return self._behaviour(payload)


@pytest.fixture
def limit(monkeypatch: pytest.MonkeyPatch) -> InFlightLimit:
    """A fresh two-slot limit so no test inherits another's orphans."""
    fresh = InFlightLimit(2)
    monkeypatch.setattr(tavily_utils, "_TAVILY_SYNC_IN_FLIGHT", fresh)
    monkeypatch.setattr(
        tavily_utils,
        "_TAVILY_SYNC_POLICY",
        BlockingCallPolicy(
            LABEL,
            tavily_utils._TAVILY_SYNC_POLICY.hard_timeout_seconds,
            in_flight=fresh,
        ),
    )
    return fresh


def _use_tool(monkeypatch: pytest.MonkeyPatch, tool: _FakeTool) -> None:
    monkeypatch.setattr(
        tavily_utils, "_build_tavily_tool", lambda profile, include_answer=False: tool
    )


def _join_workers(timeout: float = 2.0) -> list[threading.Thread]:
    workers = [
        t for t in threading.enumerate() if t.name.startswith(f"blocking:{LABEL}")
    ]
    for worker in workers:
        worker.join(timeout)
    return [t for t in workers if t.is_alive()]


def test_timeout_default_is_unchanged() -> None:
    from src.config import Settings

    assert Settings.model_fields["tavily_search_timeout_seconds"].default == 30
    assert tavily_utils._TAVILY_SYNC_POLICY.hard_timeout_seconds == 30


def test_fast_search_returns_inspected_payload(
    monkeypatch: pytest.MonkeyPatch, limit: InFlightLimit
) -> None:
    payload = {"results": [{"title": "t", "content": "c"}], "answer": "a"}
    _use_tool(monkeypatch, _FakeTool(lambda _q: payload))

    result = tavily_utils.search_tavily_sync_inspected(
        "acme", profile="news_basic", timeout=DEADLINE
    )

    assert result == payload
    assert _join_workers() == []
    assert limit.in_flight == 0


def test_no_tool_returns_none_without_spawning(
    monkeypatch: pytest.MonkeyPatch, limit: InFlightLimit
) -> None:
    monkeypatch.setattr(
        tavily_utils, "_build_tavily_tool", lambda profile, include_answer=False: None
    )
    assert tavily_utils.search_tavily_sync_inspected("q", profile="news_basic") is None
    assert limit.in_flight == 0


def test_stalled_search_caller_returns_by_deadline_then_no_leak(
    monkeypatch: pytest.MonkeyPatch, limit: InFlightLimit
) -> None:
    """(a) caller returns within its deadline; released worker leaves no thread."""
    release = threading.Event()
    _use_tool(monkeypatch, _FakeTool(lambda _q: release.wait()))

    started = time.monotonic()
    result = tavily_utils.search_tavily_sync_inspected(
        "stall", profile="news_basic", timeout=DEADLINE
    )
    elapsed = time.monotonic() - started

    assert result is None
    assert DEADLINE <= elapsed < DEADLINE + TOLERANCE
    assert limit.in_flight == 1

    release.set()
    assert _join_workers() == []
    assert limit.in_flight == 0


def test_repeated_stalls_stay_within_the_concurrency_bound(
    monkeypatch: pytest.MonkeyPatch, limit: InFlightLimit
) -> None:
    """(b) only max_in_flight workers ever exist; extra calls fail fast to None."""
    release = threading.Event()
    tool = _FakeTool(lambda _q: release.wait())
    _use_tool(monkeypatch, tool)
    before = threading.active_count()
    try:
        for _ in range(2):
            assert (
                tavily_utils.search_tavily_sync_inspected(
                    "stall", profile="finance_deep", timeout=DEADLINE
                )
                is None
            )
        for _ in range(5):
            started = time.monotonic()
            assert (
                tavily_utils.search_tavily_sync_inspected(
                    "stall", profile="finance_deep", timeout=DEADLINE
                )
                is None
            )
            # Saturated: no wait for the deadline at all.
            assert time.monotonic() - started < DEADLINE
        assert tool.calls == 2
        assert limit.in_flight == 2
        assert threading.active_count() - before <= 2
    finally:
        release.set()
    assert _join_workers() == []
    assert limit.in_flight == 0


def test_search_error_returns_none_as_before(
    monkeypatch: pytest.MonkeyPatch, limit: InFlightLimit
) -> None:
    def _boom(_q: dict[str, str]) -> Any:
        raise ConnectionError("tavily 502")

    _use_tool(monkeypatch, _FakeTool(_boom))
    assert tavily_utils.search_tavily_sync_inspected("q", profile="news_basic") is None
    assert _join_workers() == []
    assert limit.in_flight == 0


def test_inspection_error_still_propagates(
    monkeypatch: pytest.MonkeyPatch, limit: InFlightLimit
) -> None:
    async def _inspect_fails(raw: Any, query_text: str) -> Any:
        raise RuntimeError("inspector exploded")

    _use_tool(monkeypatch, _FakeTool(lambda _q: {"results": []}))
    monkeypatch.setattr(tavily_utils, "_inspect_tavily_result", _inspect_fails)

    with pytest.raises(RuntimeError, match="inspector exploded"):
        tavily_utils.search_tavily_sync_inspected("q", profile="news_basic")
    # The worker publishes its exception before releasing its slot in
    # ``finally``; join it before reading the count.
    assert _join_workers() == []
    assert limit.in_flight == 0


def test_late_search_result_is_discarded_before_inspection(
    monkeypatch: pytest.MonkeyPatch, limit: InFlightLimit
) -> None:
    """A search finishing after the caller's deadline never reaches inspection."""
    release = threading.Event()
    inspected: list[Any] = []

    async def _record(raw: Any, query_text: str) -> Any:
        inspected.append(raw)
        return raw

    _use_tool(monkeypatch, _FakeTool(lambda _q: (release.wait(), {"r": 1})[1]))
    monkeypatch.setattr(tavily_utils, "_inspect_tavily_result", _record)

    assert (
        tavily_utils.search_tavily_sync_inspected(
            "late", profile="news_basic", timeout=DEADLINE
        )
        is None
    )
    release.set()
    assert _join_workers() == []
    assert inspected == []
    assert limit.in_flight == 0


def test_stalled_inspection_is_bounded_by_the_remaining_allowance(
    monkeypatch: pytest.MonkeyPatch, limit: InFlightLimit
) -> None:
    """Inspection is cancelled at the shared deadline and its slot comes back."""
    import asyncio

    cancelled = threading.Event()

    async def _stall(raw: Any, query_text: str) -> Any:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return raw

    _use_tool(monkeypatch, _FakeTool(lambda _q: {"results": []}))
    monkeypatch.setattr(tavily_utils, "_inspect_tavily_result", _stall)

    started = time.monotonic()
    result = tavily_utils.search_tavily_sync_inspected(
        "slow inspect", profile="news_basic", timeout=DEADLINE
    )
    assert result is None
    assert time.monotonic() - started < DEADLINE + TOLERANCE
    # The worker's own wait_for ends it near the deadline, not after 60 s.
    assert _join_workers() == []
    assert cancelled.is_set()
    assert limit.in_flight == 0


@pytest.mark.asyncio
async def test_sync_entrypoint_is_safe_inside_a_running_loop(
    monkeypatch: pytest.MonkeyPatch, limit: InFlightLimit
) -> None:
    """Inspection uses asyncio.run on the worker thread, never the caller's loop."""
    _use_tool(monkeypatch, _FakeTool(lambda _q: {"results": [{"title": "x"}]}))
    result = tavily_utils.search_tavily_sync_inspected(
        "q", profile="news_basic", timeout=1.0
    )
    assert result == {"results": [{"title": "x"}]}


_EXIT_CHILD = """
import json, threading, time
import src.tavily_utils as tavily_utils

stuck = threading.Event()  # never set


class _Stuck:
    def invoke(self, payload):
        stuck.wait()


tavily_utils._build_tavily_tool = lambda profile, include_answer=False: _Stuck()
started = time.monotonic()
result = tavily_utils.search_tavily_sync_inspected(
    "exit probe", profile="news_basic", timeout=0.2
)
print(json.dumps({"caller_seconds": time.monotonic() - started,
                  "result": result}), flush=True)
"""


def test_stuck_search_does_not_hold_interpreter_exit() -> None:
    """(d) With a never-released worker, the child still returns and exits.

    The worker is a daemon thread; finalization abandons it. Nothing stops the
    stalled request — it simply dies with the process. Under the previous
    ``with ThreadPoolExecutor(...)`` block the caller itself would have hung
    in ``__exit__`` (``shutdown(wait=True)``), and the interpreter again at
    exit.
    """
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    # posix_spawn path: close_fds=False and no cwd= (see .claude/rules/tests.md).
    proc = subprocess.run(
        [sys.executable, "-c", _EXIT_CHILD],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        close_fds=False,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["result"] is None
    assert report["caller_seconds"] < DEADLINE + TOLERANCE
