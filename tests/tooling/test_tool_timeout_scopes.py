"""The two tool deadline scopes: the runner's, and the combined hooks scope.

``ToolExecutionService`` bounds the tool runner (``TOOL_EXECUTION_TIMEOUT_SECONDS``)
and, around it, the before()/after() hooks and the runner together. The combined
scope is derived -- runner plus ``TOOL_HOOKS_TIMEOUT_SECONDS`` -- so it can never
be the narrower of the two. The graph tool node adds no bound of its own: the
service's is the only one. Both scopes clamp to a caller's enclosing deadline
when one is bound. The direct-caller side of the combined scope is covered in
``test_tool_service_combined_scope.py``.

Fake clock and tiny timeouts only; nothing here sleeps for real beyond a few
hundredths of a second.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

import src.tooling.runtime as tool_runtime
from src.config import Settings
from src.graph.tool_nodes import create_agent_tool_node
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import deadline_margin, enclosing_deadline
from src.runtime_services import RuntimeServices, use_runtime_services
from src.tooling.inspection_service import InspectionService
from src.tooling.runtime import ToolExecutionService, ToolInvocation, ToolResult

EXECUTION_FIELD = "tool_execution_timeout_seconds"
HOOKS_FIELD = "tool_hooks_timeout_seconds"


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


def _tool(name: str, value: Any = "tool output") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value=value)
    return tool


def _spy_hard_timeout(monkeypatch) -> list[tuple[str, float]]:
    """Record the (label kind, timeout) each service ``run_with_hard_timeout`` gets.

    The label kind is the label up to its first colon: ``tool`` for the runner,
    ``tool_with_hooks`` for the service's combined scope.
    """
    seen: list[tuple[str, float]] = []
    real = tool_runtime.run_with_hard_timeout

    async def spy(coro, *, timeout, label):
        seen.append((label.split(":", 1)[0], timeout))
        return await real(coro, timeout=timeout, label=label)

    monkeypatch.setattr(tool_runtime, "run_with_hard_timeout", spy)
    return seen


@pytest.fixture
def fake_clock(monkeypatch):
    """Freeze the deadline clock so enclosing deadlines are exact."""
    monkeypatch.setattr(llm_timeouts, "now", lambda: 1_000.0)


# --- Defaults and settings ---------------------------------------------------


def test_defaults_are_unchanged():
    assert Settings.model_fields[EXECUTION_FIELD].default == 45.0
    assert Settings.model_fields[HOOKS_FIELD].default == 75.0


def test_combined_scope_is_runner_plus_hooks_at_the_defaults():
    # 120 s, the combined scope's former default.
    assert tool_runtime.tool_execution_with_inspection_timeout_seconds() == 120.0


@pytest.mark.parametrize(("runner", "hooks"), [(150.0, 1.0), (0.01, 0.01), (45, 1e6)])
def test_combined_scope_follows_both_settings(monkeypatch, runner, hooks):
    monkeypatch.setattr(tool_runtime.settings_config, EXECUTION_FIELD, runner)
    monkeypatch.setattr(tool_runtime.settings_config, HOOKS_FIELD, hooks)
    assert tool_runtime.tool_execution_with_inspection_timeout_seconds() == (
        runner + hooks
    )


def test_a_runner_larger_than_the_old_combined_default_is_accepted():
    # Raising the runner for a slow vendor once failed startup until the
    # combined scope was raised too; derived, it cannot be the narrower one.
    settings = Settings(**{EXECUTION_FIELD: 300.0})
    assert settings.tool_execution_timeout_seconds == 300.0


@pytest.mark.parametrize("field", [EXECUTION_FIELD, HOOKS_FIELD])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_non_positive_scope_is_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})


# --- Happy path: an override reaches the real call ---------------------------


@pytest.mark.asyncio
async def test_execution_override_reaches_runner_timeout(monkeypatch):
    monkeypatch.setattr(tool_runtime.settings_config, EXECUTION_FIELD, 7.5)
    seen = _spy_hard_timeout(monkeypatch)
    runner = AsyncMock(return_value="ok")

    result = await ToolExecutionService().execute(
        ToolInvocation(name="get_news", args={}, source="editor"), runner=runner
    )

    assert result.value == "ok"
    assert seen == [("tool_with_hooks", 82.5), ("tool", 7.5)]


@pytest.mark.asyncio
async def test_hooks_override_reaches_the_service_from_the_graph(monkeypatch):
    monkeypatch.setattr(tool_runtime.settings_config, HOOKS_FIELD, 54.0)
    seen = _spy_hard_timeout(monkeypatch)

    node = create_agent_tool_node([_tool("fast_tool")], "test_agent")
    result = await node(_state("test_agent", "fast_tool"), {"configurable": {}})

    assert result["messages"][0].content == "tool output"
    # One combined scope (the service's), the runner's narrower one inside it.
    assert seen == [("tool_with_hooks", 99.0), ("tool", 45.0)]


# --- Edge: a nested caller deadline wins -------------------------------------


@pytest.mark.asyncio
async def test_caller_deadline_clamps_both_scopes(monkeypatch, fake_clock):
    seen = _spy_hard_timeout(monkeypatch)
    node = create_agent_tool_node([_tool("fast_tool")], "test_agent")

    with enclosing_deadline(5.0):
        result = await node(_state("test_agent", "fast_tool"), {"configurable": {}})

    assert result["messages"][0].content == "tool output"
    # The runner sizes against the combined scope's binding, which is kept
    # back from the 5 s hard cap by its margin: the runner fires first.
    assert seen == [("tool_with_hooks", 5.0), ("tool", 5.0 - deadline_margin(5.0))]


@pytest.mark.asyncio
async def test_looser_caller_deadline_leaves_defaults(monkeypatch, fake_clock):
    seen = _spy_hard_timeout(monkeypatch)
    node = create_agent_tool_node([_tool("fast_tool")], "test_agent")

    with enclosing_deadline(10_000.0):
        await node(_state("test_agent", "fast_tool"), {"configurable": {}})

    assert seen == [("tool_with_hooks", 120.0), ("tool", 45.0)]


@pytest.mark.asyncio
async def test_spent_caller_deadline_never_starts_runner(monkeypatch):
    clock = [1_000.0]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    runner = AsyncMock(return_value="unreachable")

    with enclosing_deadline(5.0):
        clock[0] += 6.0
        with pytest.raises(TimeoutError, match="get_news"):
            await ToolExecutionService().execute(
                ToolInvocation(name="get_news", args={}, source="editor"),
                runner=runner,
            )

    runner.assert_not_called()


@pytest.mark.asyncio
async def test_spent_caller_deadline_fails_node_closed(monkeypatch):
    clock = [1_000.0]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    tool = _tool("fast_tool")
    node = create_agent_tool_node([tool], "test_agent")

    with enclosing_deadline(5.0):
        clock[0] += 6.0
        result = await node(_state("test_agent", "fast_tool"), {"configurable": {}})

    tool.ainvoke.assert_not_awaited()
    message = result["messages"][0]
    assert message.status == "error"
    assert message.content == "TOOL_ERROR: TOOL_TIMEOUT"


# --- Error: a stalled inspection hook is bounded by the combined scope --------


class _StalledInspectionHook:
    """An after() hook that stalls, swallows cancellation, and finishes late.

    The late value stands in for content that never completed inspection: it
    must never reach the agent.
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.finished_late = asyncio.Event()

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Deliberately ignore cancellation, as a hook awaiting an
            # uncooperative client can, and keep going for a little while.
            await asyncio.sleep(0.05)
        self.finished_late.set()
        return ToolResult("LATE_UNINSPECTED_PAYLOAD")


@pytest.mark.asyncio
async def test_stalled_inspection_hook_fails_closed_and_late_result_is_dropped(
    monkeypatch,
):
    # The tool itself returns at once, so only the combined scope (runner plus
    # hooks, 0.05 s here) can end this call: that is what proves it bounds
    # the hooks.
    monkeypatch.setattr(tool_runtime.settings_config, EXECUTION_FIELD, 0.025)
    monkeypatch.setattr(tool_runtime.settings_config, HOOKS_FIELD, 0.025)
    hook = _StalledInspectionHook()
    node = create_agent_tool_node([_tool("slow_inspect")], "test_agent")

    with use_runtime_services(
        RuntimeServices(
            tool_service=ToolExecutionService([hook]),
            inspection_service=InspectionService(),
        )
    ):
        result = await asyncio.wait_for(
            node(_state("test_agent", "slow_inspect"), {"configurable": {}}),
            timeout=2,
        )

    assert hook.entered.is_set()
    # The node answered without waiting for the cancelled call to wind down.
    assert not hook.finished_late.is_set()
    message = result["messages"][0]
    assert message.status == "error"
    assert message.content == "TOOL_ERROR: TOOL_TIMEOUT"

    # Let the orphan finish; its result must not have been published anywhere.
    await asyncio.wait_for(hook.finished_late.wait(), timeout=2)
    await asyncio.sleep(0)
    assert "LATE_UNINSPECTED_PAYLOAD" not in repr(result)
