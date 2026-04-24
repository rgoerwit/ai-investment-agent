import asyncio
from contextlib import nullcontext
from unittest.mock import AsyncMock, patch

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


class PassThroughHook:
    def __init__(self, label: str, events: list[str]) -> None:
        self.label = label
        self.events = events

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        self.events.append(f"before:{self.label}")
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        self.events.append(f"after:{self.label}")
        return result


class ExplodingBeforeHook:
    async def before(self, call: ToolInvocation) -> ToolInvocation:
        raise ValueError("unexpected before failure")

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


@pytest.mark.asyncio
async def test_execute_runs_before_hooks_forward_and_after_hooks_reverse():
    events: list[str] = []
    service = ToolExecutionService(
        hooks=[PassThroughHook("first", events), PassThroughHook("second", events)]
    )
    runner = AsyncMock(return_value="ok")

    result = await service.execute(
        ToolInvocation(name="get_news", args={"ticker": "AAPL"}, source="toolnode"),
        runner=runner,
    )

    runner.assert_awaited_once_with({"ticker": "AAPL"})
    assert events == [
        "before:first",
        "before:second",
        "after:second",
        "after:first",
    ]
    assert result == ToolResult(value="ok", blocked=False, findings=None)


@pytest.mark.asyncio
async def test_execute_propagates_unexpected_before_hook_failures():
    service = ToolExecutionService(hooks=[ExplodingBeforeHook()])
    runner = AsyncMock(return_value="should-not-run")

    with pytest.raises(ValueError, match="unexpected before failure"):
        await service.execute(
            ToolInvocation(name="get_news", args={"ticker": "AAPL"}, source="editor"),
            runner=runner,
        )

    runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_does_not_run_after_hooks_when_runner_fails():
    events: list[str] = []
    service = ToolExecutionService(hooks=[PassThroughHook("audit", events)])
    runner = AsyncMock(side_effect=RuntimeError("runner boom"))

    with pytest.raises(RuntimeError, match="runner boom"):
        await service.execute(
            ToolInvocation(name="get_news", args={"ticker": "AAPL"}, source="editor"),
            runner=runner,
        )

    runner.assert_awaited_once_with({"ticker": "AAPL"})
    assert events == ["before:audit"]


@pytest.mark.asyncio
async def test_execute_logs_structured_failure_details_when_runner_fails():
    service = ToolExecutionService()
    runner = AsyncMock(side_effect=RuntimeError("429 Too Many Requests"))

    with patch("src.tooling.runtime.logger") as mock_logger:
        with pytest.raises(RuntimeError, match="429 Too Many Requests"):
            await service.execute(
                ToolInvocation(
                    name="get_news", args={"ticker": "AAPL"}, source="editor"
                ),
                runner=runner,
            )

    mock_logger.error.assert_called_once()
    kwargs = mock_logger.error.call_args.kwargs
    assert kwargs["failure_kind"] == "rate_limit"
    assert kwargs["retryable"] is True
    assert kwargs["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_execute_wraps_runner_in_tool_observation():
    service = ToolExecutionService()
    runner = AsyncMock(return_value={"ok": True})

    with patch(
        "src.tooling.runtime.start_tool_observation",
        return_value=nullcontext(),
    ) as mock_observation:
        result = await service.execute(
            ToolInvocation(
                name="get_news",
                args={"ticker": "AAPL"},
                source="editor",
                agent_key="news_analyst",
            ),
            runner=runner,
        )

    mock_observation.assert_called_once_with(
        tool_name="get_news",
        input_payload={"arg_keys": ["ticker"], "ticker": "AAPL"},
        metadata={
            "tool_name": "get_news",
            "tool_source": "editor",
            "agent_key": "news_analyst",
        },
    )
    runner.assert_awaited_once_with({"ticker": "AAPL"})
    assert result == ToolResult(value={"ok": True}, blocked=False, findings=None)


@pytest.mark.asyncio
async def test_execute_times_out_runner_and_skips_after_hooks():
    events: list[str] = []
    service = ToolExecutionService(hooks=[PassThroughHook("audit", events)])

    async def slow_runner(_args):
        await asyncio.sleep(1)

    with patch("src.tooling.runtime.TOOL_CALL_TIMEOUT_SECONDS", 0.01):
        with patch("src.tooling.runtime.logger") as mock_logger:
            with pytest.raises(TimeoutError, match="get_news"):
                await service.execute(
                    ToolInvocation(
                        name="get_news", args={"ticker": "AAPL"}, source="editor"
                    ),
                    runner=slow_runner,
                )

    assert events == ["before:audit"]
    mock_logger.warning.assert_called_once()
    kwargs = mock_logger.warning.call_args.kwargs
    assert kwargs["tool"] == "get_news"
    assert kwargs["timeout_seconds"] == 0.01
