"""The Consultant node is hard-bounded, not merely deadline-checked.

The workflow's cooperative deadline is checked *once before the tool fan-out* —
by design, since the prompt mandates plan-then-batch and a turn is one batch.
That bounds the turn, not the node: a batch starting just under the deadline can
run a full per-tool timeout past it, and the metrics preload, re-planning rounds,
forced synthesis and the truncated-response re-ask all sit outside any single
check.

The Auditor already carries a node-level `run_with_hard_timeout` for exactly this
reason, with two documented failed attempts behind it (a loop-scoped deadline
handed the fallback a fresh budget; a node-scoped *checkpoint* bounded nothing
that followed it). This pins the same shape onto the Consultant.
"""

from __future__ import annotations

import ast
import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from src.agents import consultant_nodes, create_consultant_node


def _state() -> dict:
    return {
        "company_of_interest": "TEST.L",
        "company_name": "Test Co",
        "market_report": "m",
        "sentiment_report": "s",
        "news_report": "n",
        "fundamentals_report": "f",
        "investment_debate_state": {},
        "investment_plan": "plan",
        "red_flags": [],
        "pre_screening_result": "PASS",
    }


def _config() -> RunnableConfig:
    return RunnableConfig(configurable={"context": Mock(trade_date="2026-08-16")})


def _prompt_patch():
    prompt = Mock()
    prompt.system_message = "You are a consultant."
    prompt.agent_name = "External Consultant"
    return patch("src.prompts.get_prompt", return_value=prompt)


class TestTheBoundIsTheRightPrimitive:
    """`run_with_hard_timeout`, never `asyncio.wait_for`.

    Guarded **statically**, and the reason is worth recording so the next reader
    does not try to replace it with a behavioural test. Two attempts were made:
    an `asyncio.sleep` payload, and a thread parked in a blocking `Event.wait`.
    Neither discriminates — swapping the implementation to `asyncio.wait_for`
    leaves both passing, because `wait_for` cancels the awaiting coroutine and
    `asyncio.to_thread` propagates that cancellation immediately, abandoning the
    thread exactly as the hard timeout does.

    The divergence appears only when the awaited coroutine delays or swallows
    cancellation, which a unit test can only simulate by writing code the
    production path does not contain. So this follows the repo's existing
    precedent for the same problem — `test_wrap_is_present_in_invoke_path`
    guards the LLM wrap by AST for identical reasons.
    """

    def test_the_node_wraps_the_workflow_in_a_hard_timeout(self):
        source = Path(consultant_nodes.__file__).read_text()
        tree = ast.parse(source)

        wrapper = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.AsyncFunctionDef)
                and node.name == "consultant_node"
            ),
            None,
        )
        assert wrapper is not None, "consultant_node wrapper is gone"

        calls = {
            getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            for node in ast.walk(wrapper)
            if isinstance(node, ast.Call)
        }
        assert "run_with_hard_timeout" in calls, (
            "the Consultant node must be bounded by run_with_hard_timeout"
        )
        assert "wait_for" not in calls, (
            "asyncio.wait_for cancels-then-awaits; the async timeout standard "
            "forbids it around a provider call"
        )

    def test_the_workflow_is_not_reachable_as_the_node_itself(self):
        """The factory must return the wrapper, not the unbounded workflow."""
        source = Path(consultant_nodes.__file__).read_text()
        assert "return consultant_node" in source
        assert "return _consultant_workflow" not in source


@pytest.mark.asyncio
class TestTheBoundSurvivesBlockingWork:
    async def test_a_thread_blocked_in_sync_io_does_not_hold_the_graph(self):
        mock_llm = Mock()
        released = threading.Event()
        entered = threading.Event()

        def _uncancellable_blocking_read() -> str:
            entered.set()
            # Deliberately NOT asyncio.sleep: a thread inside a sync call is
            # exactly what `wait_for` cannot interrupt.
            released.wait(timeout=30)
            return "too late"

        async def _blocking_invoke(*args, **kwargs):
            return await asyncio.to_thread(_uncancellable_blocking_read)

        node = create_consultant_node(mock_llm, "consultant")
        try:
            with (
                _prompt_patch(),
                patch(
                    "src.agents.runtime.invoke_with_rate_limit_handling",
                    new=_blocking_invoke,
                ),
                patch(
                    "src.agents.consultant_nodes.CONSULTANT_HARD_TIMEOUT_GRACE_SECONDS",
                    0.05,
                ),
                patch(
                    "src.agents.consultant_nodes.floor_llm_total_timeout",
                    return_value=0.01,
                ),
            ):
                started = time.monotonic()
                result = await node(_state(), _config())
                elapsed = time.monotonic() - started

            assert entered.wait(timeout=5), "the blocking work never started"
            assert elapsed < 5.0, (
                f"node waited {elapsed:.1f}s on work that cannot be cancelled — "
                "this is the wait_for failure mode"
            )
            status = result.get("artifact_statuses", {}).get("consultant_review")
            assert status is not None and status.get("ok") is False
        finally:
            # Let the orphaned worker finish so it cannot outlive the test.
            released.set()


@pytest.mark.asyncio
class TestNodeReturnsRatherThanAwaitingOrphanedWork:
    async def test_a_wedged_workflow_does_not_hold_the_graph(self):
        """The property that matters: the node RETURNS, promptly.

        A budget-of-0 test is not sufficient — it only proves nothing *starts*.
        The failure mode is work that starts inside the budget and crosses it,
        so this drives a 10 s inner stall against a tiny budget and asserts the
        node comes back in well under that.
        """
        mock_llm = Mock()

        async def _never_returns(*args, **kwargs):
            await asyncio.sleep(10)

        node = create_consultant_node(mock_llm, "consultant")
        with (
            _prompt_patch(),
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling",
                new=_never_returns,
            ),
            patch(
                "src.agents.consultant_nodes.CONSULTANT_HARD_TIMEOUT_GRACE_SECONDS",
                0.05,
            ),
            patch(
                "src.agents.consultant_nodes.floor_llm_total_timeout",
                return_value=0.01,
            ),
        ):
            started = time.monotonic()
            result = await node(_state(), _config())
            elapsed = time.monotonic() - started

        assert elapsed < 5.0, f"node blocked on orphaned work for {elapsed:.1f}s"
        assert "consultant_review" in result

    async def test_expiry_degrades_the_artifact_and_never_raises(self):
        """An optional seat must cost its own output, never the ticker."""
        mock_llm = Mock()

        async def _never_returns(*args, **kwargs):
            await asyncio.sleep(10)

        node = create_consultant_node(mock_llm, "consultant")
        with (
            _prompt_patch(),
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling",
                new=_never_returns,
            ),
            patch(
                "src.agents.consultant_nodes.CONSULTANT_HARD_TIMEOUT_GRACE_SECONDS",
                0.05,
            ),
            patch(
                "src.agents.consultant_nodes.floor_llm_total_timeout",
                return_value=0.01,
            ),
        ):
            result = await node(_state(), _config())

        status = result.get("artifact_statuses", {}).get("consultant_review")
        assert status is not None, "a degraded run must still record a status"
        assert status.get("ok") is False


@pytest.mark.asyncio
class TestTheHappyPathIsUnchanged:
    async def test_a_normal_review_is_not_truncated_by_the_wrapper(self):
        """The wrapper is a ceiling, not a wait — it must be invisible here."""
        mock_llm = Mock()
        response = Mock()
        response.content = (
            "## CONSULTANT REVIEW\n\n"
            "FINAL CONSULTANT VERDICT\nMANDATE_BREACH: NONE\nHARD_STOP: NONE\n"
            + "x"
            * 400
        )
        response.tool_calls = []
        response.response_metadata = {"finish_reason": "stop"}

        async def _ok(*args, **kwargs):
            return response

        node = create_consultant_node(mock_llm, "consultant")
        with (
            _prompt_patch(),
            patch("src.agents.runtime.invoke_with_rate_limit_handling", new=_ok),
        ):
            result = await node(_state(), _config())

        status = result.get("artifact_statuses", {}).get("consultant_review")
        assert status is not None
        assert status.get("ok") is True
