"""Every tool call has exactly one outcome, decided where the caller gets it.

``ToolExecutionService.execute`` tells failure observers a call failed and
completion observers a call returned its result, never both and never
neither. The evidence ledger used to record inside ``after()``; when the
allowance ran out during or just after that hook, the trailing admission check
failed the call and the ledger then also recorded the failure, so one call was
both SUCCEEDED and FAILED. The deadline clock is faked; no network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.tooling.runtime as tool_runtime
from src.llm_runtime import timeouts as llm_timeouts
from src.tooling.evidence_recorder import EvidenceRecorder
from src.tooling.runtime import (
    ToolCallBlocked,
    ToolCompletionObserver,
    ToolExecutionService,
    ToolFailureObserver,
    ToolInvocation,
    ToolResult,
)
from src.tooling.tool_argument_policy import ToolArgumentPolicyHook

_URL = "https://www.sec.gov/Archives/edgar/data/1/report.pdf"
PAST_ALLOWANCE = 10_000.0


def _call(url: str = _URL) -> ToolInvocation:
    return ToolInvocation(
        name="get_official_document",
        args={"url": url},
        source="consultant",
        agent_key="consultant",
    )


@pytest.fixture
def clock(monkeypatch) -> list[float]:
    ticks = [1_000.0]
    monkeypatch.setattr(llm_timeouts, "now", lambda: ticks[0])
    return ticks


class _Spender:
    """An after() hook that spends the whole allowance."""

    def __init__(self, clock: list[float]) -> None:
        self.clock = clock

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        self.clock[0] += PAST_ALLOWANCE
        return result


class _SpendingRecorder(EvidenceRecorder):
    """The ledger itself overruns: the allowance expires inside its after()."""

    def __init__(self, clock: list[float]) -> None:
        super().__init__()
        self.clock = clock

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        returned = await super().after(call, result)
        self.clock[0] += PAST_ALLOWANCE
        return returned


class _Outcomes:
    """Counts the outcomes the service reports for each call."""

    def __init__(self) -> None:
        self.completed_with: list[ToolResult] = []
        self.failed_with: list[tuple[str, Any]] = []

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        return result

    def completed(self, call: ToolInvocation, result: ToolResult) -> None:
        self.completed_with.append(result)

    def failed(self, call: ToolInvocation, reason: str, scope: Any) -> None:
        self.failed_with.append((reason, scope))


class _Sanitizing:
    async def before(self, call: ToolInvocation) -> ToolInvocation:
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        return ToolResult(
            value="sanitized https://official.example/result", findings=["scrubbed"]
        )


def _assert_one_timeout_record(recorder: EvidenceRecorder) -> None:
    (record,) = recorder.snapshot()
    assert (record.execution_status, record.evidence_status, record.reason) == (
        "FAILED",
        "UNAVAILABLE",
        "TOOL_TIMEOUT",
    )
    assert record.findings == ("TIMEOUT_SCOPE:hooks_allowance_spent",)
    assert record.requested_urls == (_URL,)


# --- The regression: one call, one ledger outcome -----------------------------------------


@pytest.mark.asyncio
async def test_a_later_after_hook_that_spends_the_allowance_leaves_one_failed_record(
    clock,
):
    """after() runs in reverse: the recorder, then the spender, then the
    trailing admission check refuses the result the recorder already saw."""
    recorder = EvidenceRecorder()
    service = ToolExecutionService([_Spender(clock), recorder])

    with pytest.raises(TimeoutError, match="no allowance left"):
        await service.execute(_call(), runner=AsyncMock(return_value="STATUS: OK"))

    _assert_one_timeout_record(recorder)


@pytest.mark.asyncio
async def test_an_allowance_spent_inside_the_ledgers_own_after_leaves_one_failed_record(
    clock,
):
    """Production order: the recorder is first, so its after() runs last and
    only the trailing admission check follows it."""
    recorder = _SpendingRecorder(clock)
    service = ToolExecutionService([recorder, _Outcomes()])

    with pytest.raises(TimeoutError, match="no allowance left"):
        await service.execute(_call(), runner=AsyncMock(return_value="STATUS: OK"))

    _assert_one_timeout_record(recorder)


# --- Exactly one of completed() and failed() per call ------------------------------------


async def _raise_blocked(args: dict[str, Any]) -> str:
    raise ToolCallBlocked("INSPECTION_BLOCKED")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runner", "spend_after", "completed", "failed"),
    [
        pytest.param(AsyncMock(return_value="ok"), False, 1, 0, id="success"),
        pytest.param(_raise_blocked, False, 1, 0, id="runner-block"),
        pytest.param(AsyncMock(side_effect=ValueError("x")), False, 0, 1, id="raised"),
        pytest.param(AsyncMock(return_value="ok"), True, 0, 1, id="admission-refused"),
    ],
)
async def test_each_call_reports_exactly_one_outcome(
    clock, runner, spend_after, completed, failed
):
    outcomes = _Outcomes()
    hooks: list[Any] = [_Spender(clock), outcomes] if spend_after else [outcomes]

    try:
        await ToolExecutionService(hooks).execute(_call(), runner=runner)
    except (TimeoutError, ValueError):
        pass

    assert (len(outcomes.completed_with), len(outcomes.failed_with)) == (
        completed,
        failed,
    )


@pytest.mark.asyncio
async def test_a_spent_allowance_before_the_call_is_only_a_failure(clock):
    outcomes = _Outcomes()
    with llm_timeouts.enclosing_deadline(1.0):
        clock[0] += 1.0
        with pytest.raises(TimeoutError, match="not started"):
            await ToolExecutionService([outcomes]).execute(
                _call(), runner=AsyncMock(return_value="ok")
            )

    assert outcomes.completed_with == []
    assert outcomes.failed_with == [("TOOL_TIMEOUT", "not_started")]


@pytest.mark.asyncio
async def test_completion_sees_the_result_the_caller_receives():
    outcomes = _Outcomes()

    result = await ToolExecutionService([outcomes, _Sanitizing()]).execute(
        _call(), runner=AsyncMock(return_value="raw")
    )

    assert outcomes.completed_with == [result]
    assert result.value.startswith("sanitized")


# --- The ledger's success and block records ----------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("recorder_first", [True, False])
async def test_a_success_is_one_record_of_the_final_result_wherever_the_ledger_sits(
    recorder_first,
):
    """Same expectations as test_recorder_captures_final_post_hook_evidence,
    now also with the recorder last in the chain (its after() runs first)."""
    recorder = EvidenceRecorder()
    hooks: list[Any] = [recorder, _Sanitizing()]
    if not recorder_first:
        hooks.reverse()

    result = await ToolExecutionService(hooks).execute(
        ToolInvocation(
            name="get_official_document",
            args={"url": "https://official.example/result"},
            source="toolnode",
            agent_key="foreign_language_analyst",
        ),
        AsyncMock(return_value="raw untrusted content"),
    )

    (record,) = recorder.snapshot()
    assert record.content == result.value
    assert record.findings == ("scrubbed",)
    assert record.urls == ("https://official.example/result",)
    assert record.requested_urls == ("https://official.example/result",)
    assert (record.execution_status, record.evidence_status) == (
        "SUCCEEDED",
        "RESULTS_FOUND",
    )
    assert record.sequence == 1


@pytest.mark.asyncio
async def test_a_before_hook_block_is_one_blocked_record():
    """A policy refusal returns a blocked result without running the tool or
    any after() hook; the ledger keeps it as BLOCKED, like a runner's block."""
    recorder = EvidenceRecorder()
    outcomes = _Outcomes()
    runner = AsyncMock(return_value="unreachable")
    url = "http://127.0.0.1/admin"
    call = ToolInvocation(
        name="fetch_reference_content",
        args={"url": url},
        source="editor",
        agent_key="editor",
    )

    result = await ToolExecutionService(
        [recorder, ToolArgumentPolicyHook(mode="block"), outcomes]
    ).execute(call, runner=runner)

    runner.assert_not_called()
    assert result.blocked is True
    assert result.value.startswith("TOOL_BLOCKED: editor reference URL")
    assert outcomes.completed_with == [result]
    assert outcomes.failed_with == []
    (record,) = recorder.snapshot()
    assert (record.execution_status, record.evidence_status, record.reason) == (
        "BLOCKED",
        "UNAVAILABLE",
        "TOOL_BLOCKED",
    )
    assert record.blocked is True
    assert record.authority_by_url == ()
    assert record.requested_urls == (url,)


# --- An observer's own fault -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_raising_completion_observer_is_logged_and_the_result_returned(
    monkeypatch,
):
    class Broken(_Outcomes):
        def completed(self, call: ToolInvocation, result: ToolResult) -> None:
            raise RuntimeError("observer fault https://x?token=secret")

    logger = MagicMock()
    monkeypatch.setattr(tool_runtime, "logger", logger)
    broken, healthy = Broken(), _Outcomes()

    result = await ToolExecutionService([broken, healthy]).execute(
        _call(), runner=AsyncMock(return_value="tool output")
    )

    assert result.value == "tool output"
    assert healthy.completed_with == [result]
    assert broken.failed_with == [] and healthy.failed_with == []
    (warning,) = logger.warning.call_args_list
    assert warning.args == ("tool_completion_observer_failed",)
    assert warning.kwargs["tool"] == "get_official_document"
    assert "token=secret" not in repr(warning.kwargs)


def test_the_ledger_observes_both_outcomes_and_a_plain_hook_neither():
    class Plain:
        async def before(self, call: ToolInvocation) -> ToolInvocation:
            return call

        async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
            return result

    assert isinstance(EvidenceRecorder(), ToolCompletionObserver)
    assert isinstance(EvidenceRecorder(), ToolFailureObserver)
    assert not isinstance(Plain(), ToolCompletionObserver)


@pytest.mark.asyncio
async def test_the_ledgers_after_hook_is_a_pass_through():
    recorder = EvidenceRecorder()
    result = ToolResult(value="anything")

    assert await recorder.after(_call(), result) is result
    assert recorder.snapshot() == []
