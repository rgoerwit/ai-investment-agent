"""One combined bound in ``ToolExecutionService``: hooks and runner together.

Every caller of the service -- the graph tool node, and the direct callers
(Consultant, Auditor, Legal Counsel, article editor, evidence preflight) -- gets
``before()``, the runner and ``after()`` under one combined scope
(``TOOL_EXECUTION_TIMEOUT_SECONDS`` plus ``TOOL_HOOKS_TIMEOUT_SECONDS``),
clamped to the caller and to any tool call it is nested inside. The bound is deadline-only, so a hook
that stalls or ignores cancellation cannot hold the caller, and the late
result is never returned or recorded.

Stalls are releasable events and the deadline clock is faked where sizes are
asserted. The few real timeouts are hundredths of a second; the outer safety
net is itself deadline-only, so a regression fails the test instead of hanging.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import src.graph.tool_nodes as tool_nodes
import src.tooling.runtime as tool_runtime
from src.async_utils import run_with_hard_timeout
from src.graph.tool_nodes import create_agent_tool_node
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import deadline_margin, tool_call_deadline
from src.runtime_services import RuntimeServices, use_runtime_services
from src.tooling.inspection_service import InspectionService
from src.tooling.runtime import (
    ToolCallBlocked,
    ToolExecutionService,
    ToolInvocation,
    ToolResult,
)

HOOKS_FIELD = "tool_hooks_timeout_seconds"
EXECUTION_FIELD = "tool_execution_timeout_seconds"
# Only a safety net: a correct bound returns long before it.
SAFETY_NET_SECONDS = 2.0
SMALL_BOUND = 0.05


def _call(name: str = "get_news") -> ToolInvocation:
    return ToolInvocation(name=name, args={"ticker": "7203.T"}, source="consultant")


async def _within_safety_net(awaitable: Any) -> Any:
    """Await *awaitable*, failing (not hanging) if it outlives the safety net."""
    return await run_with_hard_timeout(
        awaitable, timeout=SAFETY_NET_SECONDS, label="test_safety_net"
    )


async def _drain_loop(turns: int = 5) -> None:
    for _ in range(turns):
        await asyncio.sleep(0)


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    """A frozen deadline clock; advance it through ``clock[0]``."""
    clock = [1_000.0]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


@pytest.fixture
def small_bound(monkeypatch) -> None:
    """A tiny combined bound: runner and hooks share it equally."""
    monkeypatch.setattr(tool_runtime.settings_config, EXECUTION_FIELD, SMALL_BOUND / 2)
    monkeypatch.setattr(tool_runtime.settings_config, HOOKS_FIELD, SMALL_BOUND / 2)


def _spy_hard_timeout(monkeypatch) -> list[tuple[str, float]]:
    """Record (label kind, timeout) for each service-level hard timeout."""
    seen: list[tuple[str, float]] = []
    real = tool_runtime.run_with_hard_timeout

    async def spy(coro, *, timeout, label):
        seen.append((label.split(":", 1)[0], timeout))
        return await real(coro, timeout=timeout, label=label)

    monkeypatch.setattr(tool_runtime, "run_with_hard_timeout", spy)
    return seen


class _Recorder:
    """A pass-through hook that records what reaches it."""

    def __init__(self) -> None:
        self.before_calls = 0
        self.after_values: list[Any] = []

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        self.before_calls += 1
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        self.after_values.append(result.value)
        return result


class _StubbornHook:
    """A hook that stalls in one phase and swallows the first cancellation.

    After swallowing it, it keeps waiting until the test releases it, as a
    hook awaiting an uncooperative client can. ``finished`` is set only once
    it has returned.
    """

    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.cancellations = 0

    async def _stall(self) -> None:
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancellations += 1
            await self.release.wait()
        self.finished.set()

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        if self.phase == "before":
            await self._stall()
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        if self.phase == "after":
            await self._stall()
            return ToolResult("LATE_UNINSPECTED_PAYLOAD")
        return result


# --- Direct callers: stalled hooks return within the bound -----------------


@pytest.mark.asyncio
async def test_stalled_before_hook_returns_within_bound(small_bound):
    hook = _StubbornHook("before")
    runner = AsyncMock(return_value="payload")
    service = ToolExecutionService([hook])

    with pytest.raises(TimeoutError, match="with its hooks exceeded"):
        await _within_safety_net(service.execute(_call(), runner=runner))

    # Returned while the hook was still stalled: the bound, not the hook, ended it.
    assert hook.entered.is_set()
    assert not hook.finished.is_set()

    # Released late, the orphan must not start the tool: its allowance is gone.
    hook.release.set()
    await asyncio.wait_for(hook.finished.wait(), timeout=SAFETY_NET_SECONDS)
    await _drain_loop()
    runner.assert_not_called()


@pytest.mark.asyncio
async def test_after_hook_ignoring_cancellation_returns_within_bound(small_bound):
    # Chain order [recorder, stubborn]: after() runs stubborn first, then the
    # recorder, as the real EvidenceRecorder runs last.
    recorder = _Recorder()
    hook = _StubbornHook("after")
    runner = AsyncMock(return_value="payload")
    service = ToolExecutionService([recorder, hook])

    with pytest.raises(TimeoutError, match="with its hooks exceeded"):
        await _within_safety_net(service.execute(_call(), runner=runner))

    runner.assert_awaited_once()
    assert hook.entered.is_set()
    assert not hook.finished.is_set()

    # The hook swallows the cancellation and finishes late. Its late result
    # must not reach a later hook (the evidence ledger, in production).
    hook.release.set()
    await asyncio.wait_for(hook.finished.wait(), timeout=SAFETY_NET_SECONDS)
    await _drain_loop()
    assert hook.cancellations == 1
    assert recorder.after_values == []


@pytest.mark.asyncio
async def test_spent_allowance_before_inspection_discards_result(fake_clock):
    recorder = _Recorder()

    async def slow_on_the_clock(args: dict[str, Any]) -> str:
        del args
        fake_clock[0] += 121.0  # past the 120 s combined allowance
        return "payload"

    service = ToolExecutionService([recorder])
    with pytest.raises(TimeoutError, match="get_news"):
        await _within_safety_net(service.execute(_call(), runner=slow_on_the_clock))

    assert recorder.before_calls == 1
    assert recorder.after_values == []


@pytest.mark.asyncio
async def test_allowance_spent_mid_chain_stops_later_hooks(fake_clock):
    later = _Recorder()

    class _ClockEater:
        async def before(self, call):
            return call

        async def after(self, call, result):
            fake_clock[0] += 200.0
            return result

    # after() order is reversed: the clock eater runs first, then `later`.
    service = ToolExecutionService([later, _ClockEater()])
    with tool_call_deadline(120.0), pytest.raises(TimeoutError, match="no allowance"):
        await service._execute_hooked(_call(), runner=AsyncMock(return_value="x"))
    # Through execute() the refusal keeps its own classification.
    with pytest.raises(TimeoutError, match="no allowance left") as caught:
        await service.execute(_call(), runner=AsyncMock(return_value="x"))
    assert caught.value.scope == "hooks_allowance_spent"
    assert later.after_values == []


@pytest.mark.asyncio
async def test_spent_caller_allowance_runs_no_hook(fake_clock):
    recorder = _Recorder()
    runner = AsyncMock(return_value="payload")

    with llm_timeouts.enclosing_deadline(1.0):
        fake_clock[0] += 2.0
        with pytest.raises(TimeoutError, match="allowance already spent"):
            await ToolExecutionService([recorder]).execute(_call(), runner=runner)

    assert recorder.before_calls == 0
    runner.assert_not_called()


# --- Unchanged semantics under the bound -----------------------------------


@pytest.mark.asyncio
async def test_runner_failure_skips_after_hooks_and_keeps_the_error():
    recorder = _Recorder()
    error = ValueError("upstream rejected the symbol")
    runner = AsyncMock(side_effect=error)

    with pytest.raises(ValueError) as caught:
        await ToolExecutionService([recorder]).execute(_call(), runner=runner)

    assert caught.value is error
    assert recorder.before_calls == 1
    assert recorder.after_values == []


@pytest.mark.asyncio
async def test_runner_timeout_keeps_its_own_message_and_skips_after(monkeypatch):
    monkeypatch.setattr(tool_runtime.settings_config, EXECUTION_FIELD, SMALL_BOUND)
    recorder = _Recorder()
    release = asyncio.Event()

    async def stalled(args):
        del args
        await release.wait()

    with pytest.raises(TimeoutError, match="Tool call 'get_news' exceeded") as caught:
        await _within_safety_net(
            ToolExecutionService([recorder]).execute(_call(), runner=stalled)
        )
    assert "with its hooks" not in str(caught.value)
    assert recorder.after_values == []
    release.set()


@pytest.mark.asyncio
async def test_blocked_call_is_unchanged():
    class _Blocker:
        async def before(self, call):
            raise ToolCallBlocked("ARGUMENT_POLICY")

        async def after(self, call, result):
            raise AssertionError("after() must not run for a blocked call")

    runner = AsyncMock(return_value="payload")
    result = await ToolExecutionService([_Blocker()]).execute(_call(), runner=runner)

    assert result == ToolResult(
        value="TOOL_BLOCKED: ARGUMENT_POLICY",
        blocked=True,
        findings=["ARGUMENT_POLICY"],
    )
    runner.assert_not_called()


@pytest.mark.asyncio
async def test_hook_order_is_unchanged():
    events: list[str] = []

    class _Named:
        def __init__(self, name):
            self.name = name

        async def before(self, call):
            events.append(f"before:{self.name}")
            return call

        async def after(self, call, result):
            events.append(f"after:{self.name}")
            return result

    await ToolExecutionService([_Named("a"), _Named("b")]).execute(
        _call(), runner=AsyncMock(return_value="x")
    )
    assert events == ["before:a", "before:b", "after:b", "after:a"]


@pytest.mark.asyncio
async def test_hooks_never_see_a_bound_llm_deadline():
    """The tool deadline is not the LLM enclosing deadline.

    Binding the LLM one would cap the injection judge inside after(), and the
    judge fails open when admission refuses it.
    """
    seen: list[Any] = []

    class _Probe:
        async def before(self, call):
            seen.append(llm_timeouts.current_enclosing_deadline())
            return call

        async def after(self, call, result):
            seen.append(llm_timeouts.current_enclosing_deadline())
            return result

    await ToolExecutionService([_Probe()]).execute(
        _call(), runner=AsyncMock(return_value="x")
    )
    assert seen == [None, None]


# --- Nesting: an inner execute cannot renew the outer allowance -------------


@pytest.mark.asyncio
async def test_nested_execute_clamps_to_the_outer_runner(monkeypatch, fake_clock):
    seen = _spy_hard_timeout(monkeypatch)
    inner_service = ToolExecutionService()
    inner_runner = AsyncMock(return_value={"metric": 1.0})

    async def outer_runner(args):
        del args
        # 30 s of the outer runner's 45 s pass before the nested MCP call.
        fake_clock[0] += 30.0
        inner = await inner_service.execute(
            ToolInvocation(
                name="mcp__fmp_remote__statements",
                args={"symbol": "7203.T"},
                source="consultant",
            ),
            runner=inner_runner,
        )
        return inner.value

    result = await ToolExecutionService().execute(
        _call("spot_check_metric_mcp_fmp"), runner=outer_runner
    )

    assert result.value == {"metric": 1.0}
    # The outer runner's binding is kept back from its 45 s cap by the
    # margin; 30 s later the inner combined scope gets what is left of it.
    inner_combined = 45.0 - deadline_margin(45.0) - 30.0
    assert seen == [
        ("tool_with_hooks", 120.0),  # outer combined
        ("tool", 45.0),  # outer runner
        ("tool_with_hooks", pytest.approx(inner_combined)),  # inner combined
        (  # inner runner: kept back from the inner combined scope
            "tool",
            pytest.approx(inner_combined - deadline_margin(inner_combined)),
        ),
    ]


@pytest.mark.asyncio
async def test_nested_execute_with_spent_outer_allowance_never_starts(fake_clock):
    inner_runner = AsyncMock(return_value="unreachable")

    async def outer_runner(args):
        del args
        fake_clock[0] += 50.0  # past the outer runner's 45 s
        await ToolExecutionService().execute(_call("inner"), runner=inner_runner)

    with pytest.raises(TimeoutError):
        await ToolExecutionService().execute(_call("outer"), runner=outer_runner)
    inner_runner.assert_not_called()


@pytest.mark.asyncio
async def test_tool_call_deadline_only_shortens(fake_clock):
    left = 10.0 - deadline_margin(10.0)
    with tool_call_deadline(10.0) as outer:
        with tool_call_deadline(500.0) as inner:
            assert inner == outer == 1_000.0 + left
            assert tool_runtime.tool_execution_with_inspection_timeout_seconds() == left
            assert tool_runtime.tool_execution_timeout_seconds() == left
    assert tool_runtime.tool_execution_with_inspection_timeout_seconds() == 120.0


# --- Graph path: the service's bound is the only one ------------------------


def test_graph_node_adds_no_bound_of_its_own():
    """The node neither wraps the service nor binds a deadline (AST, not text)."""
    tree = ast.parse(inspect.getsource(tool_nodes))
    names = {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Name | ast.Attribute)
    }
    forbidden = {
        "run_with_hard_timeout",
        "tool_call_deadline",
        "tool_execution_with_inspection_timeout_seconds",
        "settings_config",
    }
    assert names & forbidden == set()


def _state(agent_key: str, tool_name: str) -> dict[str, Any]:
    return {
        "messages": [
            HumanMessage(content="Analyze"),
            AIMessage(
                name=agent_key,
                content="",
                tool_calls=[
                    {"name": tool_name, "args": {}, "id": "t1", "type": "tool_call"}
                ],
            ),
        ]
    }


@pytest.mark.asyncio
async def test_graph_call_gets_one_combined_scope_inside_an_outer_tool_call(
    monkeypatch, fake_clock
):
    service_seen = _spy_hard_timeout(monkeypatch)
    tool = MagicMock()
    tool.name = "fast_tool"
    tool.ainvoke = AsyncMock(return_value="tool output")
    node = create_agent_tool_node([tool], "test_agent")

    with tool_call_deadline(7.0):
        result = await node(_state("test_agent", "fast_tool"), {"configurable": {}})

    combined = 7.0 - deadline_margin(7.0)
    assert result["messages"][0].content == "tool output"
    assert service_seen == [
        ("tool_with_hooks", pytest.approx(combined)),
        ("tool", pytest.approx(combined - deadline_margin(combined))),
    ]


@pytest.mark.asyncio
async def test_graph_stalled_after_hook_still_converts_to_tool_timeout(small_bound):
    hook = _StubbornHook("after")
    tool = MagicMock()
    tool.name = "slow_inspect"
    tool.ainvoke = AsyncMock(return_value="tool output")
    node = create_agent_tool_node([tool], "test_agent")

    with use_runtime_services(
        RuntimeServices(
            tool_service=ToolExecutionService([hook]),
            inspection_service=InspectionService(),
        )
    ):
        result = await _within_safety_net(
            node(_state("test_agent", "slow_inspect"), {"configurable": {}})
        )

    assert not hook.finished.is_set()
    message = result["messages"][0]
    assert message.status == "error"
    assert message.content == "TOOL_ERROR: TOOL_TIMEOUT"
    hook.release.set()
    await asyncio.wait_for(hook.finished.wait(), timeout=SAFETY_NET_SECONDS)
    await _drain_loop()
    assert "LATE_UNINSPECTED_PAYLOAD" not in repr(result)


# --- Classification: each timeout keeps the scope that produced it -----------


class _RaisingHook:
    """A hook whose own call raises *error* in one phase (a stalled lookup)."""

    def __init__(self, phase: str, error: BaseException) -> None:
        self.phase = phase
        self.error = error

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        if self.phase == "before":
            raise self.error
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        if self.phase == "after":
            raise self.error
        return result


def _timeout_scopes(monkeypatch) -> list[Any]:
    """The ``scope`` of every ``tool_call_timeout`` the service logs, in order."""
    scopes: list[Any] = []

    class _Recorder:
        def __getattr__(self, level: str) -> Any:
            def record(event: str, **fields: Any) -> None:
                if event == "tool_call_timeout":
                    scopes.append(fields.get("scope"))

            return record

    monkeypatch.setattr(tool_runtime, "logger", _Recorder())
    return scopes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "scope", "detail", "runner_awaits"),
    [
        ("before", "before_hooks", "timed out in a before() hook", 0),
        ("after", "after_hooks", "timed out in an after() hook", 1),
    ],
)
async def test_a_hooks_own_timeout_keeps_its_scope(
    monkeypatch, phase, scope, detail, runner_awaits
):
    scopes = _timeout_scopes(monkeypatch)
    runner = AsyncMock(return_value="payload")
    cause = TimeoutError("policy lookup stalled")

    with pytest.raises(TimeoutError) as caught:
        await _within_safety_net(
            ToolExecutionService([_RaisingHook(phase, cause)]).execute(
                _call(), runner=runner
            )
        )

    assert caught.value.scope == scope
    assert str(caught.value) == f"Tool call 'get_news' {detail}"
    assert caught.value.__cause__ is cause
    assert scopes == [scope]  # logged once, never relabelled as the combined scope
    assert runner.await_count == runner_awaits


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before", "after"])
async def test_a_hooks_other_error_is_not_a_timeout(monkeypatch, phase):
    scopes = _timeout_scopes(monkeypatch)
    error = ValueError("bad policy row")

    with pytest.raises(ValueError) as caught:
        await ToolExecutionService([_RaisingHook(phase, error)]).execute(
            _call(), runner=AsyncMock(return_value="payload")
        )

    assert caught.value is error
    assert scopes == []


@pytest.mark.asyncio
async def test_the_runner_timeout_is_logged_once_as_the_runner(monkeypatch):
    monkeypatch.setattr(tool_runtime.settings_config, EXECUTION_FIELD, SMALL_BOUND)
    scopes = _timeout_scopes(monkeypatch)
    release = asyncio.Event()

    async def stalled(args: dict[str, Any]) -> None:
        del args
        await release.wait()

    with pytest.raises(TimeoutError) as caught:
        await _within_safety_net(
            ToolExecutionService().execute(_call(), runner=stalled)
        )

    assert caught.value.scope == "runner"
    assert scopes == ["runner"]
    release.set()


@pytest.mark.asyncio
async def test_the_combined_bound_firing_is_labelled_with_inspection(
    monkeypatch, small_bound
):
    scopes = _timeout_scopes(monkeypatch)
    hook = _StubbornHook("before")

    with pytest.raises(TimeoutError, match="with its hooks exceeded") as caught:
        await _within_safety_net(
            ToolExecutionService([hook]).execute(
                _call(), runner=AsyncMock(return_value="x")
            )
        )

    # The service's own bound carries its scope out, like every other timeout.
    assert caught.value.scope == "with_inspection"
    assert scopes == ["with_inspection"]
    hook.release.set()
    await asyncio.wait_for(hook.finished.wait(), timeout=SAFETY_NET_SECONDS)


@pytest.mark.asyncio
async def test_a_spent_allowance_is_labelled_not_started(monkeypatch, fake_clock):
    scopes = _timeout_scopes(monkeypatch)
    with llm_timeouts.enclosing_deadline(1.0):
        fake_clock[0] += 1.0  # exactly at the deadline: nothing left
        with pytest.raises(TimeoutError, match="not started"):
            await ToolExecutionService().execute(
                _call(), runner=AsyncMock(return_value="x")
            )
    assert scopes == ["not_started"]


@pytest.mark.asyncio
async def test_the_hooks_refusal_is_labelled_as_spent_not_exceeded(
    monkeypatch, fake_clock
):
    scopes = _timeout_scopes(monkeypatch)
    recorder = _Recorder()

    async def slow_on_the_clock(args: dict[str, Any]) -> str:
        del args
        fake_clock[0] += 121.0
        return "payload"

    with pytest.raises(TimeoutError, match="no allowance left") as caught:
        await ToolExecutionService([recorder]).execute(
            _call(), runner=slow_on_the_clock
        )

    assert caught.value.scope == "hooks_allowance_spent"
    assert scopes == ["hooks_allowance_spent"]
    assert recorder.after_values == []
