"""A tool call that produced no result still leaves a record of what was asked.

``after()`` hooks run only for a result that exists, so a timed-out
``get_official_document`` used to leave the evidence ledger nothing: the URL was
lost and the model relabelled the failure ``EXTRACTION_TIMEOUT`` in prose. The
service now tells each observing hook of the failure and its classified timeout
scope, at its one chokepoint, so the graph tool node, the consultant, the auditor
and the preflight all leave the same trace. Tiny real timeouts and the suite's
fake clock; no network.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import src.tooling.evidence_recorder as recorder_module
import src.tooling.runtime as tool_runtime
from src.agents.consultant_tool_loop import _execute_tool_call
from src.claim_policy import RAW_FINANCIAL_METRICS_INPUT
from src.graph.tool_nodes import create_agent_tool_node
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline
from src.runtime_services import RuntimeServices, use_runtime_services
from src.tooling.evidence_recorder import EvidenceRecorder
from src.tooling.inspection_service import InspectionService
from src.tooling.runtime import (
    ToolCallBlocked,
    ToolCallFailed,
    ToolExecutionService,
    ToolFailureObserver,
    ToolInvocation,
    ToolResult,
    tool_error_text,
)

_URL = "https://www.sec.gov/Archives/edgar/data/1/report.pdf"
_GUARD_SECONDS = 5.0
EXECUTION = "tool_execution_timeout_seconds"
HOOKS = "tool_hooks_timeout_seconds"


def _call(url: str = _URL, name: str = "get_official_document") -> ToolInvocation:
    return ToolInvocation(
        name=name, args={"url": url}, source="consultant", agent_key="consultant"
    )


async def _stall(args: dict[str, Any]) -> str:
    del args
    await asyncio.Event().wait()
    return "unreachable"


@pytest.fixture
def small_runner(monkeypatch) -> None:
    """A 0.02 s runner inside a 0.52 s combined scope: the runner fires first."""
    monkeypatch.setattr(tool_runtime.settings_config, EXECUTION, 0.02)
    monkeypatch.setattr(tool_runtime.settings_config, HOOKS, 0.5)


@pytest.fixture
def small_bound(monkeypatch) -> None:
    """A 0.1 s combined scope: only a stalled hook reaches it."""
    monkeypatch.setattr(tool_runtime.settings_config, EXECUTION, 0.05)
    monkeypatch.setattr(tool_runtime.settings_config, HOOKS, 0.05)


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    clock = [1_000.0]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


async def _guarded(awaitable: Any) -> Any:
    return await asyncio.wait_for(awaitable, _GUARD_SECONDS)


# --- Each timeout scope leaves the URL and the scope -----------------------------------


@pytest.mark.asyncio
async def test_a_runner_timeout_keeps_the_url_and_its_scope(small_runner):
    recorder = EvidenceRecorder()

    with pytest.raises(TimeoutError):
        await _guarded(ToolExecutionService([recorder]).execute(_call(), runner=_stall))

    (record,) = recorder.snapshot()
    assert record.tool_name == "get_official_document"
    assert record.agent_key == "consultant"
    assert record.requested_urls == (_URL,)
    assert record.execution_status == "FAILED"
    assert record.evidence_status == "UNAVAILABLE"
    assert record.reason == "TOOL_TIMEOUT"
    assert record.findings == ("TIMEOUT_SCOPE:runner",)
    assert record.content == "TOOL_ERROR: TOOL_TIMEOUT\nTIMEOUT_SCOPE: runner"
    assert record.urls == ()
    assert record.blocked is False


@pytest.mark.asyncio
async def test_a_spent_allowance_is_recorded_as_not_started(fake_clock):
    recorder = EvidenceRecorder()
    runner = AsyncMock(return_value="unreachable")

    with enclosing_deadline(1.0):
        fake_clock[0] += 1.0
        with pytest.raises(TimeoutError, match="not started"):
            await ToolExecutionService([recorder]).execute(_call(), runner=runner)

    runner.assert_not_called()
    (record,) = recorder.snapshot()
    assert record.findings == ("TIMEOUT_SCOPE:not_started",)
    assert record.requested_urls == (_URL,)


class _StalledAfterHook:
    async def before(self, call: ToolInvocation) -> ToolInvocation:
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        await asyncio.Event().wait()
        return result


class _TimingOutAfterHook(_StalledAfterHook):
    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        raise TimeoutError("hook's own timeout")


@pytest.mark.asyncio
async def test_the_combined_bound_is_recorded_as_with_inspection(small_bound):
    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder, _StalledAfterHook()])

    with pytest.raises(TimeoutError, match="with its hooks exceeded"):
        await _guarded(service.execute(_call(), runner=AsyncMock(return_value="x")))

    (record,) = recorder.snapshot()
    assert record.findings == ("TIMEOUT_SCOPE:with_inspection",)


@pytest.mark.asyncio
async def test_a_hooks_own_timeout_is_recorded_with_its_scope(monkeypatch):
    monkeypatch.setattr(tool_runtime.settings_config, EXECUTION, 5.0)
    monkeypatch.setattr(tool_runtime.settings_config, HOOKS, 5.0)
    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder, _TimingOutAfterHook()])

    with pytest.raises(TimeoutError):
        await _guarded(service.execute(_call(), runner=AsyncMock(return_value="x")))

    (record,) = recorder.snapshot()
    assert record.findings == ("TIMEOUT_SCOPE:after_hooks",)


# --- Other failures, and what is not a failure ---------------------------------------------


@pytest.mark.asyncio
async def test_a_runner_exception_is_recorded_as_an_execution_failure():
    recorder = EvidenceRecorder()
    runner = AsyncMock(side_effect=ValueError("boom"))

    with pytest.raises(ValueError, match="boom"):
        await ToolExecutionService([recorder]).execute(_call(), runner=runner)

    (record,) = recorder.snapshot()
    assert record.execution_status == "FAILED"
    assert record.reason == "TOOL_EXECUTION_FAILED"
    assert record.findings == ()
    assert record.requested_urls == (_URL,)
    # Exception prose is not stored: it can carry a URL with a credential.
    assert "boom" not in record.content


@pytest.mark.asyncio
async def test_a_typed_runner_failure_is_recorded_with_its_reason_code():
    recorder = EvidenceRecorder()
    runner = AsyncMock(side_effect=ToolCallFailed("INSPECTION_FAILED"))

    with pytest.raises(ToolCallFailed):
        await ToolExecutionService([recorder]).execute(_call(), runner=runner)

    (record,) = recorder.snapshot()
    assert (record.execution_status, record.evidence_status) == (
        "FAILED",
        "UNAVAILABLE",
    )
    assert record.reason == "INSPECTION_FAILED"
    assert record.content == "TOOL_ERROR: INSPECTION_FAILED"
    assert record.requested_urls == (_URL,)


class _SeesAfter:
    def __init__(self) -> None:
        self.seen: list[ToolResult] = []

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        self.seen.append(result)
        return result


@pytest.mark.asyncio
async def test_a_runner_block_is_a_blocked_result_the_after_hooks_record():
    """One shape for a block, wherever it was raised: the ledger records it."""
    recorder = EvidenceRecorder()
    observer = _SeesAfter()
    runner = AsyncMock(side_effect=ToolCallBlocked("INSPECTION_BLOCKED"))

    result = await ToolExecutionService([recorder, observer]).execute(
        _call(), runner=runner
    )

    assert result.value == "TOOL_BLOCKED: INSPECTION_BLOCKED"
    assert result.blocked is True
    assert result.findings == ["INSPECTION_BLOCKED"]
    assert observer.seen == [result]
    (record,) = recorder.snapshot()
    assert (record.execution_status, record.evidence_status) == (
        "BLOCKED",
        "UNAVAILABLE",
    )
    assert record.blocked is True
    assert record.authority_by_url == ()
    assert record.requested_urls == (_URL,)


@pytest.mark.asyncio
async def test_a_success_leaves_only_its_own_record():
    recorder = EvidenceRecorder()

    await ToolExecutionService([recorder]).execute(
        _call(), runner=AsyncMock(return_value="STATUS: INSUFFICIENT_DATA\nREASON: X")
    )

    (record,) = recorder.snapshot()
    assert record.execution_status == "SUCCEEDED"


@pytest.mark.asyncio
async def test_a_cancelled_call_is_not_a_failure():
    recorder = EvidenceRecorder()
    started = asyncio.Event()

    async def runner(args: dict[str, Any]) -> str:
        del args
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    task = asyncio.create_task(
        ToolExecutionService([recorder]).execute(_call(), runner=runner)
    )
    # Cancelled from outside, well before either bound could fire.
    await asyncio.wait_for(started.wait(), _GUARD_SECONDS)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert recorder.snapshot() == []


@pytest.mark.asyncio
async def test_a_repeated_failure_is_one_record_and_a_new_url_is_another(small_runner):
    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder])

    for url in (_URL, _URL, _URL + "?other=1"):
        with pytest.raises(TimeoutError):
            await _guarded(service.execute(_call(url), runner=_stall))

    assert [r.requested_urls for r in recorder.snapshot()] == [
        (_URL,),
        (_URL + "?other=1",),
    ]
    assert [r.sequence for r in recorder.snapshot()] == [1, 2]


@pytest.mark.asyncio
async def test_a_full_ledger_marks_its_overflow_once(monkeypatch):
    monkeypatch.setattr(recorder_module, "_MAX_RECORDS", 2)
    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder])

    for index in range(4):
        with pytest.raises(ValueError):
            await service.execute(
                _call(f"https://www.sec.gov/{index}"),
                runner=AsyncMock(side_effect=ValueError("x")),
            )

    tools = [r.tool_name for r in recorder.snapshot()]
    assert tools == [
        "get_official_document",
        "get_official_document",
        "__ledger_overflow__",
    ]


@pytest.mark.asyncio
async def test_an_observer_that_raises_cannot_replace_the_failure():
    class Broken:
        async def before(self, call: ToolInvocation) -> ToolInvocation:
            return call

        async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
            return result

        def failed(self, call: ToolInvocation, reason: str, scope: Any) -> None:
            raise RuntimeError("observer fault")

    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder, Broken()])

    with pytest.raises(ValueError, match="the tool's own failure"):
        await service.execute(
            _call(), runner=AsyncMock(side_effect=ValueError("the tool's own failure"))
        )

    # The healthy observer after the broken one still recorded.
    assert len(recorder.snapshot()) == 1


def test_a_hook_without_failed_is_not_an_observer_and_is_skipped():
    class Plain:
        async def before(self, call: ToolInvocation) -> ToolInvocation:
            return call

        async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
            return result

    assert not isinstance(Plain(), ToolFailureObserver)
    assert isinstance(EvidenceRecorder(), ToolFailureObserver)


# --- The model sees the typed code -----------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("exceeded"), "TOOL_ERROR: TOOL_TIMEOUT"),
        (
            tool_runtime._ScopedToolTimeout("x", scope="runner"),
            "TOOL_ERROR: TOOL_TIMEOUT",
        ),
        (asyncio.TimeoutError(), "TOOL_ERROR: TOOL_TIMEOUT"),
        (RuntimeError("secret https://x?token=1"), "TOOL_ERROR: TOOL_EXECUTION_FAILED"),
        (ToolCallFailed("INSPECTION_FAILED"), "TOOL_ERROR: INSPECTION_FAILED"),
        (ValueError(), "TOOL_ERROR: TOOL_EXECUTION_FAILED"),
    ],
)
def test_tool_error_text_types_a_timeout_and_never_echoes_prose(error, expected):
    assert tool_error_text(error) == expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            tool_runtime._ScopedToolTimeout("x", scope="after_hooks"),
            ("TOOL_TIMEOUT", "after_hooks"),
        ),
        (TimeoutError(), ("TOOL_TIMEOUT", None)),
        (ToolCallFailed("INSPECTION_TIMEOUT"), ("INSPECTION_TIMEOUT", None)),
        (ToolCallBlocked("INSPECTION_BLOCKED"), ("TOOL_EXECUTION_FAILED", None)),
        (asyncio.CancelledError(), ("TOOL_EXECUTION_FAILED", None)),
    ],
)
def test_tool_failure_reason_is_the_one_classification(error, expected):
    """What observers record and the model reads; a block that escaped the
    service (never expected) is a failure, not a timeout or a block."""
    assert tool_runtime.tool_failure_reason(error) == expected


@pytest.mark.asyncio
async def test_the_consultant_loop_gets_the_typed_timeout_and_the_ledger_the_url(
    small_runner,
):
    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder])
    tool = SimpleNamespace(name="get_official_document", ainvoke=_stall)

    message, outcome = await _guarded(
        _execute_tool_call(
            {"name": "get_official_document", "args": {"url": _URL}, "id": "t1"},
            tools_by_name={"get_official_document": tool},
            fmp_alt_disabled_kind=None,
            tool_service_getter=lambda: service,
            agent_key="consultant",
            ticker="7203.T",
        )
    )

    assert message.content == "TOOL_ERROR: TOOL_TIMEOUT"
    assert outcome.result_failed is True
    (record,) = recorder.snapshot()
    assert record.requested_urls == (_URL,)
    assert record.findings == ("TIMEOUT_SCOPE:runner",)


@pytest.mark.asyncio
async def test_the_consultant_loop_gets_a_typed_failure_reason():
    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder])
    tool = SimpleNamespace(
        name="get_official_document",
        ainvoke=AsyncMock(side_effect=ToolCallFailed("INSPECTION_TIMEOUT")),
    )

    message, outcome = await _guarded(
        _execute_tool_call(
            {"name": "get_official_document", "args": {"url": _URL}, "id": "t1"},
            tools_by_name={"get_official_document": tool},
            fmp_alt_disabled_kind=None,
            tool_service_getter=lambda: service,
            agent_key="consultant",
            ticker="7203.T",
        )
    )

    assert message.content == "TOOL_ERROR: INSPECTION_TIMEOUT"
    assert outcome.result_failed is True
    (record,) = recorder.snapshot()
    assert record.reason == "INSPECTION_TIMEOUT"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side_effect", "failed"),
    [
        pytest.param(ToolCallBlocked("INSPECTION_BLOCKED"), True, id="raised-block"),
        pytest.param(None, False, id="text-that-reads-like-a-block"),
    ],
)
async def test_the_consultant_loop_reads_a_block_from_the_flag(side_effect, failed):
    tool = SimpleNamespace(
        name="get_official_document",
        ainvoke=AsyncMock(
            side_effect=side_effect, return_value="TOOL_BLOCKED: quoted in a filing"
        ),
    )

    message, outcome = await _guarded(
        _execute_tool_call(
            {"name": "get_official_document", "args": {"url": _URL}, "id": "t1"},
            tools_by_name={"get_official_document": tool},
            fmp_alt_disabled_kind=None,
            tool_service_getter=lambda: ToolExecutionService([]),
            agent_key="consultant",
            ticker="7203.T",
        )
    )

    assert message.content.startswith("TOOL_BLOCKED:")
    assert outcome.result_failed is failed


@pytest.mark.asyncio
async def test_the_graph_tool_node_leaves_the_same_record(small_runner):
    recorder = EvidenceRecorder()
    tool = MagicMock()
    tool.name = "get_official_document"
    tool.ainvoke = _stall
    node = create_agent_tool_node([tool], "test_agent")
    state = {
        "messages": [
            HumanMessage(content="Analyze"),
            AIMessage(
                name="test_agent",
                content="",
                tool_calls=[
                    {
                        "name": "get_official_document",
                        "args": {"url": _URL},
                        "id": "t1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }

    with use_runtime_services(
        RuntimeServices(
            tool_service=ToolExecutionService([recorder]),
            inspection_service=InspectionService(),
        )
    ):
        result = await _guarded(node(state, {"configurable": {}}))

    assert result["messages"][0].content == "TOOL_ERROR: TOOL_TIMEOUT"
    (record,) = recorder.snapshot()
    assert record.requested_urls == (_URL,)
    assert record.findings == ("TIMEOUT_SCOPE:runner",)
    assert record.agent_key == "test_agent"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (ToolCallFailed("INSPECTION_FAILED"), "INSPECTION_FAILED"),
        (ToolCallFailed("INSPECTION_TIMEOUT"), "INSPECTION_TIMEOUT"),
        (RuntimeError("secret https://x?token=1"), "TOOL_EXECUTION_FAILED"),
    ],
)
async def test_the_graph_tool_node_tells_the_model_the_ledgers_reason(error, reason):
    """The model text, the ledger record and the structured ingress record
    carry one classification of the failure, never exception prose."""
    recorder = EvidenceRecorder()
    tool = MagicMock()
    tool.name = "get_financial_metrics"
    tool.ainvoke = AsyncMock(side_effect=error)
    agent_key = "junior_fundamentals_analyst"
    node = create_agent_tool_node([tool], agent_key)
    state = {
        "company_of_interest": "7203.T",
        "messages": [
            HumanMessage(content="Analyze"),
            AIMessage(
                name=agent_key,
                content="",
                tool_calls=[
                    {
                        "name": "get_financial_metrics",
                        "args": {"ticker": "7203.T"},
                        "id": "t1",
                        "type": "tool_call",
                    }
                ],
            ),
        ],
    }

    with use_runtime_services(
        RuntimeServices(
            tool_service=ToolExecutionService([recorder]),
            inspection_service=InspectionService(),
        )
    ):
        result = await _guarded(node(state, {"configurable": {}}))

    (message,) = result["messages"]
    assert message.content == f"TOOL_ERROR: {reason}"
    assert message.status == "error"
    (record,) = recorder.snapshot()
    assert record.execution_status == "FAILED"
    assert record.reason == reason
    ingress = result["structured_inputs"][RAW_FINANCIAL_METRICS_INPUT]
    assert (ingress["status"], ingress["reason"]) == ("INVALID", reason)
    assert "secret" not in message.content
