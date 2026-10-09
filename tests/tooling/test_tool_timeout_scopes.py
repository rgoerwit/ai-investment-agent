"""The two tool deadline scopes: the runner's, and the graph node's inspection scope.

``ToolExecutionService`` bounds the tool runner (``TOOL_EXECUTION_TIMEOUT_SECONDS``);
the graph tool node bounds execution *plus* the after() inspection hooks, which
run outside the runner's timeout (``TOOL_EXECUTION_WITH_INSPECTION_TIMEOUT_SECONDS``).
Both clamp to a caller's enclosing deadline when one is bound.

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

import src.graph.tool_nodes as tool_nodes
import src.tooling.runtime as tool_runtime
from src.config import Settings
from src.graph.tool_nodes import create_agent_tool_node
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline
from src.runtime_services import RuntimeServices, use_runtime_services
from src.tooling.inspection_service import InspectionService
from src.tooling.runtime import ToolExecutionService, ToolInvocation, ToolResult

EXECUTION_FIELD = "tool_execution_timeout_seconds"
INSPECTION_FIELD = "tool_execution_with_inspection_timeout_seconds"


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


def _spy_hard_timeout(monkeypatch, module) -> list[float]:
    """Record the timeout each real ``run_with_hard_timeout`` call receives."""
    seen: list[float] = []
    real = module.run_with_hard_timeout

    async def spy(coro, *, timeout, label):
        seen.append(timeout)
        return await real(coro, timeout=timeout, label=label)

    monkeypatch.setattr(module, "run_with_hard_timeout", spy)
    return seen


@pytest.fixture
def fake_clock(monkeypatch):
    """Freeze the deadline clock so enclosing deadlines are exact."""
    monkeypatch.setattr(llm_timeouts, "now", lambda: 1_000.0)


# --- Defaults and settings ---------------------------------------------------


def test_defaults_are_unchanged():
    assert Settings.model_fields[EXECUTION_FIELD].default == 45.0
    assert Settings.model_fields[INSPECTION_FIELD].default == 120.0


def test_inspection_scope_narrower_than_execution_is_rejected():
    with pytest.raises(ValidationError, match="must be at least"):
        Settings(**{EXECUTION_FIELD: 60.0, INSPECTION_FIELD: 30.0})


def test_equal_scopes_are_accepted():
    settings = Settings(**{EXECUTION_FIELD: 60.0, INSPECTION_FIELD: 60.0})
    assert settings.tool_execution_with_inspection_timeout_seconds == 60.0


@pytest.mark.parametrize("field", [EXECUTION_FIELD, INSPECTION_FIELD])
def test_non_positive_scope_is_rejected(field):
    with pytest.raises(ValidationError):
        Settings(**{field: 0.0})


# --- Happy path: an override reaches the real call ---------------------------


@pytest.mark.asyncio
async def test_execution_override_reaches_runner_timeout(monkeypatch):
    monkeypatch.setattr(tool_runtime.settings_config, EXECUTION_FIELD, 7.5)
    seen = _spy_hard_timeout(monkeypatch, tool_runtime)
    runner = AsyncMock(return_value="ok")

    result = await ToolExecutionService().execute(
        ToolInvocation(name="get_news", args={}, source="editor"), runner=runner
    )

    assert result.value == "ok"
    assert seen == [7.5]


@pytest.mark.asyncio
async def test_inspection_override_reaches_node_wrapper(monkeypatch):
    monkeypatch.setattr(tool_nodes.settings_config, INSPECTION_FIELD, 99.0)
    node_seen = _spy_hard_timeout(monkeypatch, tool_nodes)
    runner_seen = _spy_hard_timeout(monkeypatch, tool_runtime)

    node = create_agent_tool_node([_tool("fast_tool")], "test_agent")
    result = await node(_state("test_agent", "fast_tool"), {"configurable": {}})

    assert result["messages"][0].content == "tool output"
    assert node_seen == [99.0]
    # The runner keeps its own, narrower scope inside the wrapper.
    assert runner_seen == [45.0]


# --- Edge: a nested caller deadline wins -------------------------------------


@pytest.mark.asyncio
async def test_caller_deadline_clamps_both_scopes(monkeypatch, fake_clock):
    node_seen = _spy_hard_timeout(monkeypatch, tool_nodes)
    runner_seen = _spy_hard_timeout(monkeypatch, tool_runtime)
    node = create_agent_tool_node([_tool("fast_tool")], "test_agent")

    with enclosing_deadline(5.0):
        result = await node(_state("test_agent", "fast_tool"), {"configurable": {}})

    assert result["messages"][0].content == "tool output"
    assert node_seen == [5.0]
    assert runner_seen == [5.0]


@pytest.mark.asyncio
async def test_looser_caller_deadline_leaves_defaults(monkeypatch, fake_clock):
    node_seen = _spy_hard_timeout(monkeypatch, tool_nodes)
    runner_seen = _spy_hard_timeout(monkeypatch, tool_runtime)
    node = create_agent_tool_node([_tool("fast_tool")], "test_agent")

    with enclosing_deadline(10_000.0):
        await node(_state("test_agent", "fast_tool"), {"configurable": {}})

    assert node_seen == [120.0]
    assert runner_seen == [45.0]


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


# --- Error: a stalled inspection hook is bounded by the inspection scope ------


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
    # The runner's own scope stays at its default: only the inspection scope
    # can end this call, which is what proves it bounds the hooks.
    monkeypatch.setattr(tool_nodes.settings_config, INSPECTION_FIELD, 0.05)
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
