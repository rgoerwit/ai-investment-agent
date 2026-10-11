"""The Consultant's deadline lives on the shared deadline clock.

It used to be a private ``time.monotonic() + total`` instant, invisible to the
run deadline and to ``llm_timeouts.now`` (the one patchable clock). Now the
instant is measured on ``llm_timeouts.now``, shortened by the run deadline, and
bound as the LLM enclosing deadline for the LLM request alone -- never across
the tool fan-out, whose hooks include the fail-open injection judge.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.agents import consultant_nodes
from src.agents.consultant_tool_loop import (
    ConsultantToolLoopPolicy,
    remaining_consultant_budget,
    run_bounded_consultant_loop,
)
from src.llm_runtime import timeouts as llm_timeouts


class _Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(llm_timeouts, "now", fake)
    return fake


class TestRemainingBudget:
    def test_counts_down_on_the_shared_clock(self, clock):
        deadline = clock.t + 100
        assert remaining_consultant_budget(deadline) == 100
        clock.t += 30
        assert remaining_consultant_budget(deadline) == 70

    @pytest.mark.parametrize("overshoot", [0.0, 0.001, 500.0])
    def test_never_negative_at_or_past_the_deadline(self, clock, overshoot):
        deadline = clock.t + 10
        clock.t += 10 + overshoot
        assert remaining_consultant_budget(deadline) == 0.0

    def test_a_tighter_run_deadline_shortens_it(self, clock):
        with llm_timeouts.enclosing_deadline_at(clock.t + 10):
            assert remaining_consultant_budget(clock.t + 100) == 10

    def test_a_looser_run_deadline_does_not_extend_it(self, clock):
        with llm_timeouts.enclosing_deadline_at(clock.t + 500):
            assert remaining_consultant_budget(clock.t + 100) == 100


def _patch_invoke(monkeypatch, behaviour):
    calls: list[float | None] = []

    async def fake(*_args, **_kwargs):
        calls.append(llm_timeouts.current_enclosing_deadline())
        return await behaviour()

    monkeypatch.setattr(
        consultant_nodes.agent_runtime, "invoke_with_rate_limit_handling", fake
    )
    return calls


async def _invoke(deadline):
    return await consultant_nodes._invoke_consultant_with_deadline(
        "runnable",
        [],
        context="External Consultant",
        provider="openai",
        model_name="m",
        ticker="7740.T",
        deadline=deadline,
        total_timeout=100,
    )


class TestInvokeBindsTheDeadline:
    @pytest.mark.asyncio
    async def test_the_request_sees_the_node_deadline_and_it_is_released(
        self, clock, monkeypatch
    ):
        async def ok():
            return "answer"

        calls = _patch_invoke(monkeypatch, ok)
        deadline = clock.t + 100
        assert await _invoke(deadline) == "answer"
        assert calls == [deadline]
        assert llm_timeouts.current_enclosing_deadline() is None

    @pytest.mark.asyncio
    async def test_a_tighter_outer_deadline_still_wins(self, clock, monkeypatch):
        async def ok():
            return "answer"

        calls = _patch_invoke(monkeypatch, ok)
        with llm_timeouts.enclosing_deadline_at(clock.t + 20):
            await _invoke(clock.t + 100)
        assert calls == [clock.t + 20]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", [RuntimeError("boom"), asyncio.CancelledError()])
    async def test_released_after_failure_and_cancellation(
        self, clock, monkeypatch, exc
    ):
        async def fail():
            raise exc

        _patch_invoke(monkeypatch, fail)
        with pytest.raises(type(exc)):
            await _invoke(clock.t + 100)
        assert llm_timeouts.current_enclosing_deadline() is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("past", [0.0, 1.0])
    async def test_a_spent_deadline_stops_before_any_request(
        self, clock, monkeypatch, past
    ):
        async def ok():
            return "answer"

        calls = _patch_invoke(monkeypatch, ok)
        with pytest.raises(TimeoutError, match="total wall-clock timeout"):
            await _invoke(clock.t - past)
        assert calls == []

    @pytest.mark.asyncio
    async def test_a_spent_run_deadline_stops_before_any_request(
        self, clock, monkeypatch
    ):
        async def ok():
            return "answer"

        calls = _patch_invoke(monkeypatch, ok)
        with llm_timeouts.enclosing_deadline_at(clock.t - 1):
            with pytest.raises(TimeoutError):
                await _invoke(clock.t + 100)
        assert calls == []


class TestLoopBoundary:
    @staticmethod
    def _tool_service(_runner=None):
        from src.tooling.runtime import ToolResult

        class _Service:
            async def execute(self, invocation, runner=None):
                return ToolResult(value=await runner(invocation.args))

        return lambda: _Service()

    @staticmethod
    def _turn():
        return SimpleNamespace(
            content="",
            tool_calls=[{"name": "t", "args": {}, "id": "c0"}],
            response_metadata={"finish_reason": "tool_calls"},
        )

    @staticmethod
    def _final():
        return SimpleNamespace(
            content="FINAL", tool_calls=[], response_metadata={"finish_reason": "stop"}
        )

    @pytest.mark.asyncio
    async def test_tools_run_outside_the_llm_deadline_but_requests_inside(
        self, clock, monkeypatch
    ):
        responses = [self._turn(), self._final()]

        async def next_response():
            return responses.pop(0)

        llm_seen = _patch_invoke(monkeypatch, next_response)
        tool_seen: list[float | None] = []

        async def ainvoke(_args):
            tool_seen.append(llm_timeouts.current_enclosing_deadline())
            return "ok"

        deadline = clock.t + 100
        result = await run_bounded_consultant_loop(
            active_llm="a",
            fallback_llm="f",
            messages=[],
            tools_by_name={"t": SimpleNamespace(ainvoke=ainvoke)},
            policy=ConsultantToolLoopPolicy(
                max_tool_iterations=1,
                max_tool_calls_per_turn=4,
                deadline=deadline,
                total_timeout=100,
            ),
            invoke_with_deadline=lambda llm, msgs: _invoke(deadline),
            tool_service_getter=self._tool_service(),
            agent_name="External Consultant",
            agent_key="consultant",
            ticker="7740.T",
        )
        assert result.content == "FINAL"
        assert llm_seen == [deadline, deadline]
        assert tool_seen == [None]

    @pytest.mark.asyncio
    async def test_a_deadline_spent_during_tools_stops_the_next_request(
        self, clock, monkeypatch
    ):
        async def never():
            raise AssertionError("no request may be sent past the deadline")

        responses = [self._turn()]

        async def first():
            return responses.pop(0) if responses else await never()

        llm_seen = _patch_invoke(monkeypatch, first)

        async def ainvoke(_args):
            clock.t += 200  # the tool batch outlives the whole budget
            return "ok"

        deadline = clock.t + 100
        with pytest.raises(TimeoutError, match="total wall-clock timeout"):
            await run_bounded_consultant_loop(
                active_llm="a",
                fallback_llm="f",
                messages=[],
                tools_by_name={"t": SimpleNamespace(ainvoke=ainvoke)},
                policy=ConsultantToolLoopPolicy(
                    max_tool_iterations=2,
                    max_tool_calls_per_turn=4,
                    deadline=deadline,
                    total_timeout=100,
                ),
                invoke_with_deadline=lambda llm, msgs: _invoke(deadline),
                tool_service_getter=self._tool_service(),
                agent_name="External Consultant",
                agent_key="consultant",
                ticker="7740.T",
            )
        # The tool batch ran, then the next request was refused, not sent.
        assert len(llm_seen) == 1
