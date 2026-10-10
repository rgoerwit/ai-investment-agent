"""One operational deadline reader, and the margin that keeps inner work first.

Operational work -- web searches, market-data fetches, FX, filings, blocking
calls, MCP -- reads ``operational_deadline``: the earlier of the caller's LLM
enclosing deadline and the tool call it runs under. Before October 2026 only
MCP and the official-document tool read the tool call's deadline; everything
else, Tavily included, sized itself against the enclosing deadline alone and
was cut off by the 45 s runner as a bare ``TOOL_TIMEOUT``.

The tool call's deadline is bound kept back from the scope's hard cap by
``deadline_margin``, so the inner work's own (typed) timeout fires first:
asyncio orders timers due at the same instant arbitrarily.

The deadline clock is faked throughout; no test waits on a real timer.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import AsyncMock

import pytest

import src.async_utils as async_utils
import src.tavily_utils as tavily_utils
from src.config import config
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import (
    clamp_to_operational_deadline,
    current_enclosing_deadline,
    deadline_margin,
    enclosing_deadline,
    operational_deadline,
    operational_time_left,
    require_allowance,
    tool_call_deadline,
)
from src.tooling.runtime import ToolExecutionService, ToolInvocation

_NOW = 1_000.0


@pytest.fixture
def clock(monkeypatch) -> list[float]:
    """A frozen deadline clock; advance it through ``clock[0]``."""
    now = [_NOW]
    monkeypatch.setattr(llm_timeouts, "now", lambda: now[0])
    return now


# --- deadline_margin ---------------------------------------------------------


@pytest.mark.parametrize(
    ("cap", "expected"),
    [
        (45.0, 2.0),  # the fraction (2.25 s) is above the 2 s ceiling
        (40.0, 2.0),  # exactly at the ceiling
        (10.0, 0.5),  # the fraction wins below it
        (0.02, 0.001),  # a tiny cap still keeps a positive, smaller margin
        (1e6, 2.0),  # very large: the ceiling holds
    ],
)
def test_margin_is_a_fraction_of_the_cap_up_to_the_ceiling(cap, expected):
    assert deadline_margin(cap) == pytest.approx(expected)
    assert 0 < deadline_margin(cap) < cap


def test_margin_follows_both_settings(monkeypatch):
    monkeypatch.setattr(config, "deadline_margin_fraction", 0.2)
    monkeypatch.setattr(config, "llm_dispatch_margin_seconds", 3.0)
    assert deadline_margin(10.0) == pytest.approx(2.0)
    assert deadline_margin(100.0) == pytest.approx(3.0)


def test_margin_of_a_zero_ceiling_is_zero(monkeypatch):
    monkeypatch.setattr(config, "llm_dispatch_margin_seconds", 0.0)
    assert deadline_margin(45.0) == 0.0


# --- The reader ----------------------------------------------------------------


def test_nothing_bound_means_no_deadline(clock):
    del clock
    assert operational_deadline() is None
    assert operational_time_left() == math.inf
    assert clamp_to_operational_deadline(15) == 15.0
    assert require_allowance(15.0, label="probe") == 15.0


def test_enclosing_deadline_alone(clock):
    del clock
    with enclosing_deadline(4.0):
        assert operational_deadline() == _NOW + 4.0
        assert clamp_to_operational_deadline(15.0) == pytest.approx(4.0)
        assert clamp_to_operational_deadline(2.0) == pytest.approx(2.0)


def test_tool_call_deadline_alone_is_kept_back_by_its_margin(clock):
    del clock
    with tool_call_deadline(45.0) as bound:
        assert bound == _NOW + 45.0 - deadline_margin(45.0)
        assert operational_deadline() == bound
        assert require_allowance(30.0, label="tavily") == 30.0
        assert require_allowance(60.0, label="tavily") == pytest.approx(43.0)
        # The tool's deadline is never the LLM one: the judge never sees it.
        assert current_enclosing_deadline() is None


@pytest.mark.parametrize(("enclosing", "expected_left"), [(10.0, 10.0), (60.0, 43.0)])
def test_the_earlier_of_both_wins(clock, enclosing, expected_left):
    del clock
    with enclosing_deadline(enclosing), tool_call_deadline(45.0):
        assert operational_time_left() == pytest.approx(expected_left)


def test_nested_tool_bindings_only_shorten(clock):
    with tool_call_deadline(10.0) as outer:
        clock[0] += 1.0
        with tool_call_deadline(500.0) as inner:
            assert inner == outer
        with tool_call_deadline(2.0) as tighter:
            assert tighter == clock[0] + 2.0 - deadline_margin(2.0)
    assert operational_deadline() is None


@pytest.mark.parametrize("past", [0.0, 1.0, 1e9])
def test_spent_or_exactly_reached_raises_before_any_work(clock, past):
    with tool_call_deadline(10.0) as bound:
        clock[0] = bound + past
        assert clamp_to_operational_deadline(5.0) <= 0
        with pytest.raises(TimeoutError, match="'probe': deadline already passed"):
            require_allowance(5.0, label="probe")


def test_negative_default_is_returned_unclamped_for_the_caller_to_refuse(clock):
    del clock
    with tool_call_deadline(10.0):
        assert clamp_to_operational_deadline(-1.0) == -1.0
        with pytest.raises(TimeoutError):
            require_allowance(-1.0, label="probe")


def test_binding_is_reset_after_an_error(clock):
    del clock
    with pytest.raises(RuntimeError), tool_call_deadline(5.0):
        raise RuntimeError("boom")
    assert operational_deadline() is None


# --- The fix: operational work under a tool runner sees the runner ------------


class _RecordingSearch:
    def __init__(self) -> None:
        self.requests = 0

    async def ainvoke(self, query: dict[str, str]) -> str:
        del query
        self.requests += 1
        return "result"


def _spy_tavily_bounds(monkeypatch) -> list[float]:
    seen: list[float] = []
    real = async_utils.run_with_hard_timeout

    async def spy(coro: Any, *, timeout: float, label: str) -> Any:
        seen.append(timeout)
        return await real(coro, timeout=timeout, label=label)

    monkeypatch.setattr(async_utils, "run_with_hard_timeout", spy)
    return seen


@pytest.fixture
def search(monkeypatch) -> _RecordingSearch:
    tool = _RecordingSearch()
    monkeypatch.setattr(tavily_utils, "_tavily_tool", tool)

    async def _passthrough(raw: Any, query_text: str) -> Any:
        del query_text
        return raw

    monkeypatch.setattr(tavily_utils, "_inspect_tavily_result", _passthrough)
    return tool


@pytest.mark.asyncio
async def test_a_search_inside_a_tool_runner_gets_only_what_the_runner_left(
    monkeypatch, clock, search
):
    seen = _spy_tavily_bounds(monkeypatch)
    monkeypatch.setattr(config, "tool_execution_timeout_seconds", 12.0)

    async def runner(args: dict[str, Any]) -> Any:
        del args
        clock[0] += 5.0  # a first step used part of the runner's allowance
        return await tavily_utils.tavily_search_with_timeout({"query": "q"})

    result = await ToolExecutionService().execute(
        ToolInvocation(name="get_news", args={}, source="consultant"), runner=runner
    )

    assert result.value == "result"
    assert search.requests == 1
    # 30 s is the search setting; the runner (12 s, kept back by its margin)
    # had 12 - margin - 5 left, so that is all the search was given.
    expected = 12.0 - deadline_margin(12.0) - 5.0
    assert seen[0] == pytest.approx(expected)
    assert seen[0] < 12.0 - 5.0  # strictly before the runner's own bound


@pytest.mark.asyncio
async def test_a_search_after_the_runner_is_spent_sends_nothing(
    monkeypatch, clock, search
):
    with tool_call_deadline(3.0) as bound:
        clock[0] = bound
        assert await tavily_utils.tavily_search_with_timeout({"query": "q"}) is None
    assert search.requests == 0


@pytest.mark.asyncio
async def test_inner_and_runner_bounds_never_coincide(monkeypatch, clock):
    """The equal-timer race: an inner clamp is always below the runner's bound."""
    seen_runner: list[float] = []
    inner: list[float] = []
    import src.tooling.runtime as tool_runtime

    real = tool_runtime.run_with_hard_timeout

    async def spy(coro: Any, *, timeout: float, label: str) -> Any:
        if label.startswith("tool:"):
            seen_runner.append(timeout)
        return await real(coro, timeout=timeout, label=label)

    monkeypatch.setattr(tool_runtime, "run_with_hard_timeout", spy)

    async def runner(args: dict[str, Any]) -> None:
        del args
        # A phase whose own ceiling exceeds the runner: it is clamped.
        inner.append(clamp_to_operational_deadline(1e6))

    await ToolExecutionService().execute(
        ToolInvocation(name="t", args={}, source="consultant"),
        runner=AsyncMock(side_effect=runner),
    )

    assert inner[0] == pytest.approx(seen_runner[0] - deadline_margin(seen_runner[0]))
    assert inner[0] < seen_runner[0]
