from unittest.mock import AsyncMock

import pytest

from src.tooling.runtime import (
    ToolCallBlocked,
    ToolExecutionService,
    ToolInvocation,
    ToolResult,
)


class RecordingHook:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        self.events.append(f"before:{call.name}")
        return ToolInvocation(
            name=call.name,
            args={**call.args, "seen": True},
            source=call.source,
            agent_key=call.agent_key,
        )

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        self.events.append(f"after:{call.name}")
        return ToolResult(
            value=f"{result.value}:post",
            blocked=result.blocked,
            findings=result.findings,
        )


class BlockingHook:
    async def before(self, call: ToolInvocation) -> ToolInvocation:
        raise ToolCallBlocked("policy rejected input")

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        return result


@pytest.mark.asyncio
async def test_execute_runs_hooks_in_order_and_mutates_args():
    events: list[str] = []
    service = ToolExecutionService(hooks=[RecordingHook(events)])
    runner = AsyncMock(return_value="runner-output")

    result = await service.execute(
        ToolInvocation(
            name="search_web",
            args={"query": "abc"},
            source="toolnode",
            agent_key="news_analyst",
        ),
        runner=runner,
    )

    runner.assert_awaited_once_with({"query": "abc", "seen": True})
    assert events == ["before:search_web", "after:search_web"]
    assert result.value == "runner-output:post"
    assert result.blocked is False


@pytest.mark.asyncio
async def test_execute_is_noop_when_no_hooks():
    service = ToolExecutionService()
    runner = AsyncMock(return_value={"ok": True})

    result = await service.execute(
        ToolInvocation(name="get_news", args={"ticker": "AAPL"}, source="editor"),
        runner=runner,
    )

    runner.assert_awaited_once_with({"ticker": "AAPL"})
    assert result == ToolResult(value={"ok": True}, blocked=False, findings=None)


@pytest.mark.asyncio
async def test_execute_returns_blocked_result_without_running_runner():
    service = ToolExecutionService(hooks=[BlockingHook()])
    runner = AsyncMock(return_value="should-not-run")

    result = await service.execute(
        ToolInvocation(name="get_news", args={"ticker": "AAPL"}, source="consultant"),
        runner=runner,
    )

    runner.assert_not_awaited()
    assert result.blocked is True
    assert result.value == "TOOL_BLOCKED: policy rejected input"
    assert result.findings == ["policy rejected input"]
