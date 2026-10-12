"""No new hook work after the call's allowance is spent.

``ToolExecutionService`` checks the operational deadline before every
``before()`` hook, after the last of them, before every ``after()`` hook and
after the last of them. A hook that overruns, even one that swallowed the
combined bound's cancellation and finished in an orphaned task, is therefore
never followed by a later hook, the runner or a returned result. The deadline
clock is faked where sizes are asserted; stalls are releasable events.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

import src.tooling.runtime as tool_runtime
from src.async_utils import run_with_hard_timeout
from src.llm_runtime import timeouts as llm_timeouts
from src.tooling.runtime import ToolExecutionService, ToolInvocation, ToolResult

SAFETY_NET_SECONDS = 2.0
SMALL_BOUND = 0.05
PAST_ALLOWANCE = 10_000.0


def _call() -> ToolInvocation:
    return ToolInvocation(name="get_news", args={"ticker": "7203.T"}, source="test")


@pytest.fixture
def clock(monkeypatch) -> list[float]:
    ticks = [1_000.0]
    monkeypatch.setattr(llm_timeouts, "now", lambda: ticks[0])
    return ticks


@pytest.fixture
def small_bound(monkeypatch) -> None:
    monkeypatch.setattr(
        tool_runtime.settings_config, "tool_execution_timeout_seconds", SMALL_BOUND / 2
    )
    monkeypatch.setattr(
        tool_runtime.settings_config, "tool_hooks_timeout_seconds", SMALL_BOUND / 2
    )


class _Hook:
    """Records each phase it is started in; can spend the allowance or stall."""

    def __init__(
        self,
        name: str,
        log: list[str],
        *,
        spend_before: list[float] | None = None,
        spend_after: list[float] | None = None,
    ) -> None:
        self.name = name
        self.log = log
        self.spend_before = spend_before
        self.spend_after = spend_after
        self.failures: list[tuple[str, Any]] = []

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        self.log.append(f"before:{self.name}")
        if self.spend_before is not None:
            self.spend_before[0] += PAST_ALLOWANCE
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        self.log.append(f"after:{self.name}")
        if self.spend_after is not None:
            self.spend_after[0] += PAST_ALLOWANCE
        return result

    def failed(self, call: ToolInvocation, reason: str, scope: Any) -> None:
        self.failures.append((reason, scope))


class _StubbornBefore(_Hook):
    """A before() hook that swallows cancellation and waits to be released."""

    def __init__(self, name: str, log: list[str]) -> None:
        super().__init__(name, log)
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        self.log.append(f"before:{self.name}")
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            await self.release.wait()
        self.finished.set()
        return call


class _StubbornAfter(_Hook):
    """An after() hook that swallows cancellation and waits to be released."""

    def __init__(self, name: str, log: list[str]) -> None:
        super().__init__(name, log)
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        self.log.append(f"after:{self.name}")
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            await self.release.wait()
        self.finished.set()
        return result


async def _execute(service: ToolExecutionService, runner: Any) -> Any:
    return await run_with_hard_timeout(
        service.execute(_call(), runner=runner),
        timeout=SAFETY_NET_SECONDS,
        label="test_safety_net",
    )


@pytest.mark.asyncio
async def test_an_expired_chain_starts_no_later_before_hook(clock):
    log: list[str] = []
    first = _Hook("first", log, spend_before=clock)
    second = _Hook("second", log)
    runner = AsyncMock(return_value="payload")

    with pytest.raises(TimeoutError, match="no allowance left") as caught:
        await _execute(ToolExecutionService([first, second]), runner)

    assert log == ["before:first"]
    runner.assert_not_called()
    assert caught.value.scope == "hooks_allowance_spent"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_last_before_hook_that_spends_the_allowance_starts_no_runner(clock):
    log: list[str] = []
    only = _Hook("only", log, spend_before=clock)
    runner = AsyncMock(return_value="payload")

    with pytest.raises(TimeoutError, match="no allowance left") as caught:
        await _execute(ToolExecutionService([only]), runner)

    runner.assert_not_called()
    assert log == ["before:only"]
    assert caught.value.scope == "hooks_allowance_spent"  # type: ignore[attr-defined]
    assert only.failures == [("TOOL_TIMEOUT", "hooks_allowance_spent")]


@pytest.mark.asyncio
async def test_a_last_after_hook_that_spends_the_allowance_returns_no_result(clock):
    log: list[str] = []
    only = _Hook("only", log, spend_after=clock)

    with pytest.raises(TimeoutError, match="no allowance left") as caught:
        await _execute(ToolExecutionService([only]), AsyncMock(return_value="x"))

    assert log == ["before:only", "after:only"]
    assert caught.value.scope == "hooks_allowance_spent"  # type: ignore[attr-defined]
    assert only.failures == [("TOOL_TIMEOUT", "hooks_allowance_spent")]


@pytest.mark.asyncio
async def test_an_after_hook_that_spends_the_allowance_starts_no_later_one(clock):
    log: list[str] = []
    later = _Hook("later", log)
    spender = _Hook("spender", log, spend_after=clock)

    # after() runs in reverse: spender, then later.
    with pytest.raises(TimeoutError, match="no allowance left"):
        await _execute(
            ToolExecutionService([later, spender]), AsyncMock(return_value="x")
        )

    assert "after:later" not in log


@pytest.mark.asyncio
async def test_time_left_runs_the_whole_chain(clock):
    log: list[str] = []
    a, b = _Hook("a", log), _Hook("b", log)

    result = await _execute(ToolExecutionService([a, b]), AsyncMock(return_value="ok"))

    assert result.value == "ok"
    assert log == ["before:a", "before:b", "after:b", "after:a"]
    assert a.failures == [] and b.failures == []


@pytest.mark.asyncio
async def test_an_overrunning_before_hook_leaves_later_hooks_and_runner_unstarted(
    small_bound,
):
    log: list[str] = []
    stubborn = _StubbornBefore("stubborn", log)
    later = _Hook("later", log)
    runner = AsyncMock(return_value="payload")

    with pytest.raises(TimeoutError, match="with its hooks exceeded"):
        await _execute(ToolExecutionService([stubborn, later]), runner)

    stubborn.release.set()
    await asyncio.wait_for(stubborn.finished.wait(), timeout=SAFETY_NET_SECONDS)
    for _ in range(20):
        await asyncio.sleep(0)

    assert log == ["before:stubborn"]
    runner.assert_not_called()
    # Exactly one notification, from the combined bound; the orphan adds none.
    assert stubborn.failures == [("TOOL_TIMEOUT", "with_inspection")]
    assert later.failures == [("TOOL_TIMEOUT", "with_inspection")]


@pytest.mark.asyncio
async def test_an_overrunning_after_hook_starts_no_later_hook_and_notifies_once(
    small_bound,
):
    log: list[str] = []
    recorder = _Hook("recorder", log)
    stubborn = _StubbornAfter("stubborn", log)

    with pytest.raises(TimeoutError, match="with its hooks exceeded"):
        await _execute(
            ToolExecutionService([recorder, stubborn]), AsyncMock(return_value="x")
        )

    stubborn.release.set()
    await asyncio.wait_for(stubborn.finished.wait(), timeout=SAFETY_NET_SECONDS)
    for _ in range(20):
        await asyncio.sleep(0)

    assert "after:recorder" not in log
    assert recorder.failures == [("TOOL_TIMEOUT", "with_inspection")]
    assert stubborn.failures == [("TOOL_TIMEOUT", "with_inspection")]
