"""Bounds on the gap-fill company-name lookup in ``_fetch_tavily_gaps_impl``.

The lookup used to be ``asyncio_module.wait_for(asyncio_module.to_thread(...))``,
which cancels and then waits on a thread a stalled Yahoo socket never releases.
These tests pin the replacement: the caller returns at the deadline, orphaned
workers are capped, failures fall back to the bare symbol as before, and a stuck
worker does not keep the interpreter alive.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.blocking_io import BlockingCallPolicy, InFlightLimit
from src.data.gap_fill import _fetch_tavily_gaps_impl

REPO_ROOT = Path(__file__).resolve().parents[2]
DEADLINE = 0.2
TOLERANCE = 0.5
LABEL = "gap-fill-test"


class _InfoTicker:
    def __init__(self, info_fn: Any) -> None:
        self._info_fn = info_fn

    @property
    def info(self) -> Any:
        return self._info_fn()


def _yf(info_fn: Any) -> SimpleNamespace:
    return SimpleNamespace(Ticker=lambda symbol: _InfoTicker(info_fn))


def _policy(max_in_flight: int = 2) -> BlockingCallPolicy:
    return BlockingCallPolicy(LABEL, DEADLINE, in_flight=InFlightLimit(max_in_flight))


def _fetcher() -> SimpleNamespace:
    return SimpleNamespace(tavily_client=object(), pattern_extractor=None)


async def _company_name_seen(
    info_fn: Any, policy: BlockingCallPolicy, symbol: str = "7203.T"
) -> tuple[str, float]:
    """Run the impl; return the company name it searched with and the elapsed time."""
    seen: list[str] = []

    def _query_builder(sym: str, company_name: str, terms: str) -> str:
        seen.append(company_name)
        return f"{company_name} {terms}"

    async def _search(query: str, **_kwargs: Any) -> dict[str, Any]:
        return {}  # no results -> the impl returns {} before inspection

    started = time.monotonic()
    result = await _fetch_tavily_gaps_impl(
        _fetcher(),
        symbol,
        ["priceToBook"],
        yf_module=_yf(info_fn),
        search_fn=_search,
        query_builder=_query_builder,
        info_policy=policy,
    )
    elapsed = time.monotonic() - started
    assert result == {}
    assert len(seen) == 1
    return seen[0], elapsed


def _join_workers(timeout: float = 2.0) -> list[threading.Thread]:
    workers = [
        t for t in threading.enumerate() if t.name.startswith(f"blocking:{LABEL}")
    ]
    for worker in workers:
        worker.join(timeout)
    return [t for t in workers if t.is_alive()]


@pytest.mark.asyncio
async def test_fast_info_supplies_company_name() -> None:
    name, _elapsed = await _company_name_seen(
        lambda: {"longName": "Toyota Motor Corporation"}, _policy()
    )
    assert name == "Toyota Motor Corporation"
    assert _join_workers() == []


@pytest.mark.asyncio
async def test_stalled_info_returns_by_deadline_and_falls_back_to_symbol() -> None:
    """(a) the caller returns at the deadline; releasing the stall leaks nothing."""
    policy = _policy()
    release = threading.Event()

    name, elapsed = await _company_name_seen(release.wait, policy)

    assert name == "7203.T"
    assert DEADLINE <= elapsed < DEADLINE + TOLERANCE
    assert policy.in_flight is not None
    assert policy.in_flight.in_flight == 1
    release.set()
    assert _join_workers() == []
    assert policy.in_flight.in_flight == 0


@pytest.mark.asyncio
async def test_repeated_stalls_stay_within_the_concurrency_bound() -> None:
    """(b) a batch of stalled tickers cannot park more than max_in_flight threads."""
    policy = _policy(max_in_flight=2)
    release = threading.Event()
    before = threading.active_count()
    try:
        for index in range(6):
            name, elapsed = await _company_name_seen(
                release.wait, policy, symbol=f"000{index}.HK"
            )
            assert name == f"000{index}.HK"
            if index >= 2:
                # Saturated: falls back immediately, without waiting the deadline.
                assert elapsed < DEADLINE
        assert policy.in_flight is not None
        assert policy.in_flight.in_flight == 2
        assert threading.active_count() - before <= 2
    finally:
        release.set()
    assert _join_workers() == []


@pytest.mark.asyncio
async def test_info_error_falls_back_to_symbol_as_before() -> None:
    def _boom() -> Any:
        raise ConnectionError("yahoo 429")

    name, _elapsed = await _company_name_seen(_boom, _policy())
    assert name == "7203.T"


@pytest.mark.asyncio
async def test_info_without_names_falls_back_to_symbol() -> None:
    name, _elapsed = await _company_name_seen(lambda: {}, _policy())
    assert name == "7203.T"


_EXIT_CHILD = """
import asyncio, json, threading, time
from types import SimpleNamespace
from src.blocking_io import BlockingCallPolicy, InFlightLimit
from src.data.gap_fill import _fetch_tavily_gaps_impl

stuck = threading.Event()  # never set


class _Ticker:
    @property
    def info(self):
        stuck.wait()


async def _search(query, **kwargs):
    return {}


async def main():
    started = time.monotonic()
    await _fetch_tavily_gaps_impl(
        SimpleNamespace(tavily_client=object(), pattern_extractor=None),
        "7203.T",
        ["priceToBook"],
        yf_module=SimpleNamespace(Ticker=lambda symbol: _Ticker()),
        search_fn=_search,
        info_policy=BlockingCallPolicy("exit_probe", 0.2, in_flight=InFlightLimit(1)),
    )
    return time.monotonic() - started


caller_seconds = asyncio.run(main())
print(json.dumps({"caller_seconds": caller_seconds}), flush=True)
"""


def test_stuck_info_does_not_hold_interpreter_exit() -> None:
    """(d) Caller returns at its deadline and the process exits afterwards.

    The worker is a daemon thread outside the loop's default executor, so
    neither ``asyncio.run``'s executor shutdown nor the ``concurrent.futures``
    exit hook waits on it. The stalled request is never interrupted; it dies
    with the process.
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
    assert report["caller_seconds"] < DEADLINE + TOLERANCE
