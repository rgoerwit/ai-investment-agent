"""Callers that used to invoke a model directly now run under the budget policy.

Step 3 of the October 2026 deadline work sized every graph-seat request from an
absolute invocation deadline. The analyst structural retry, the retrospective
lesson, the semantic judge and the Article Editor's review still computed their
own caps (or, for the editor, had none) and sent requests outside any
application invocation. These tests pin that each now obtains its cap and
deadline from the same resolver, keeps its failure behaviour, and that two
resolver gaps found in review are closed:

- A: inside an invocation, a registered seat's explicit client timeout (the
  Health Check's 10 s) is the request ceiling, as it is outside one.
- B: a dispatch keeps the invocation's deadline but takes its request ceiling
  from the vendor that actually receives it.

A fake clock and fake transports throughout; nothing reaches the network.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI

import src.config as config_module
import src.llms as llms_mod
from src.agents import runtime as agent_runtime
from src.agents.circuit_breaker import reset_circuit_breaker_for_tests
from src.agents.network_breaker import reset_network_breaker_singleton
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.adapters import transports
from src.llm_runtime.seats import SEATS, SeatId
from src.llm_runtime.timeouts import (
    TimeoutClass,
    current_dispatch_scope,
    invocation_scope,
    policy_for_provider,
    resolve_budget_policy,
    seat_for_agent,
)
from src.runtime_config import RuntimeConfig, use_runtime_config

QUICK_CAP = 60.0
APEX_CAP = 180.0
CROSS_CHECK_CAP = 180.0
FULL_CAP = 120.0
MARGIN = 2.0
MIN_USEFUL_FRACTION = 0.04
HEALTH_CHECK_TIMEOUT = SEATS[
    SeatId.HEALTH_CHECK
].execution_policy.client_timeout_seconds


class FakeClock:
    def __init__(self, start: float = 5_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(llm_timeouts, "now", fake)
    return fake


@pytest.fixture(autouse=True)
def budget_config(monkeypatch):
    """Pin every input the policy reads, on every live config binding."""
    reset_circuit_breaker_for_tests()
    reset_network_breaker_singleton()
    values = {
        "quick_llm_call_hard_timeout_seconds": QUICK_CAP,
        "apex_quick_llm_call_hard_timeout_seconds": APEX_CAP,
        "cross_check_quick_llm_call_hard_timeout_seconds": CROSS_CHECK_CAP,
        "llm_call_hard_timeout_seconds": FULL_CAP,
        "min_useful_attempt_fraction": MIN_USEFUL_FRACTION,
        "llm_dispatch_margin_seconds": MARGIN,
        "flex_quick_attempt_fraction": 0.4,
        "api_timeout": 300,
        "quick_llm_api_timeout_seconds": 120,
        "flex_llm_timeout_seconds": 900,
        # Standard tier: no full-mode flex floor stretches the caps below.
        "gemini_service_tier": "standard",
        "openai_service_tier": "auto",
        "llm_attempts": 2,
    }
    targets = {id(config_module.config): config_module.config}
    targets[id(llms_mod.resolve_settings())] = llms_mod.resolve_settings()
    for target in targets.values():
        for key, value in values.items():
            monkeypatch.setattr(target, key, value, raising=False)
    yield
    reset_circuit_breaker_for_tests()
    reset_network_breaker_singleton()


def _run(*, quick: bool, **overrides: Any):
    return use_runtime_config(
        RuntimeConfig.from_config(config_module.config).with_overrides(
            quick_mode_active=quick, **overrides
        )
    )


class _ScopeRecordingLLM:
    """A chat model stand-in that records the invocation scope it ran under."""

    def __init__(self, content: str = "ok", *, provider: str = "google") -> None:
        self.model_name = "fake-model"
        self._llm_runtime_provider = provider
        self.content = content
        self.scopes: list[Any] = []
        self.configs: list[Any] = []

    async def ainvoke(self, _input, config=None):
        self.scopes.append(current_dispatch_scope())
        self.configs.append(config)
        return AIMessage(content=self.content)


def _gemini_result(model: str) -> ChatResult:
    message = AIMessage(content="ok")
    message.response_metadata = {"model_name": model, "finish_reason": "STOP"}
    return ChatResult(generations=[ChatGeneration(message=message)])


# ------------------------------------------------- follow-up A: seat override


class TestSeatClientTimeoutInsideAnInvocation:
    def test_registered_seat_resolves_with_its_client_timeout(self):
        assert HEALTH_CHECK_TIMEOUT == 10.0
        for quick in (False, True):
            with _run(quick=quick):
                policy = agent_runtime.resolve_invocation_budget(
                    "Health Check", provider="google"
                )
            assert policy.timeout_class is TimeoutClass.OPERATIONAL
            assert policy.request_cap == HEALTH_CHECK_TIMEOUT
            assert policy.request_cap_override == HEALTH_CHECK_TIMEOUT

    def test_seat_without_a_client_timeout_keeps_the_vendor_ceiling(self):
        with _run(quick=False):
            policy = agent_runtime.resolve_invocation_budget(
                "News Analyst R2", provider="google"
            )
        assert policy.timeout_class is TimeoutClass.STANDARD
        assert policy.request_cap == 300.0
        assert policy.request_cap_override is None

    def test_unregistered_context_gets_auxiliary_without_override(self):
        with _run(quick=True):
            policy = agent_runtime.resolve_invocation_budget(
                "some_free_form_label", provider="google"
            )
        assert policy.timeout_class is TimeoutClass.AUXILIARY
        assert policy.request_cap_override is None

    def test_explicit_class_keeps_the_seat_client_timeout(self):
        with _run(quick=True):
            policy = agent_runtime.resolve_invocation_budget(
                "Health Check",
                provider="google",
                timeout_class=TimeoutClass.STANDARD,
            )
        assert policy.timeout_class is TimeoutClass.STANDARD
        assert policy.request_cap == HEALTH_CHECK_TIMEOUT

    @pytest.mark.asyncio
    async def test_scoped_health_check_request_carries_its_10s(self, clock):
        """End to end: the runtime loop sizes the request from the seat's 10 s."""
        sent: list[dict] = []

        async def fake_agenerate(llm, messages, *args, **kwargs):
            sent.append(kwargs)
            return _gemini_result(llm.model)

        llm = llms_mod._TieredChatGoogleGenerativeAI(
            model="gemini-3.8-flash", api_key="k", timeout=300
        )
        with (
            _run(quick=False),
            patch.object(ChatGoogleGenerativeAI, "_agenerate", fake_agenerate),
        ):
            await agent_runtime.invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content="ping")],
                context="Health Check",
                provider="google",
                model_name=llm.model,
            )
        (call,) = sent
        assert call["timeout"] == pytest.approx(HEALTH_CHECK_TIMEOUT)
        # One SDK attempt is the client's own; a dispatch never sends a count.
        assert "max_retries" not in call
        assert llm.max_retries == 1

    def test_shared_display_name_with_conflicting_client_timeouts_fails_closed(
        self, monkeypatch
    ):
        import src.llm_runtime.seats as seats_mod

        writer = seats_mod.SEATS[SeatId.ARTICLE_WRITER_REVIEW_FALLBACK]
        twin = seats_mod.SEATS[SeatId.ARTICLE_WRITER_BASE_FALLBACK]
        assert writer.callback_name == twin.callback_name
        conflicting = replace(
            twin,
            execution_policy=replace(twin.execution_policy, client_timeout_seconds=7.0),
        )
        monkeypatch.setattr(
            seats_mod,
            "SEATS",
            {
                SeatId.ARTICLE_WRITER_REVIEW_FALLBACK: writer,
                SeatId.ARTICLE_WRITER_BASE_FALLBACK: conflicting,
            },
        )
        with pytest.raises(ValueError, match="two client timeouts"):
            seat_for_agent(writer.callback_name)


# ----------------------------------------- follow-up B: the dispatching vendor


class TestRequestCeilingFollowsTheDispatchingVendor:
    def test_same_family_returns_the_scope_policy_unchanged(self):
        with _run(quick=False):
            policy = resolve_budget_policy(TimeoutClass.STANDARD, provider="google")
        assert policy_for_provider(policy, "google") is policy
        assert policy_for_provider(policy, "gemini") is policy

    def test_other_vendor_gets_its_own_ceiling_and_keeps_the_allowance(self):
        with _run(quick=False):
            google = resolve_budget_policy(TimeoutClass.STANDARD, provider="google")
            openai = policy_for_provider(google, "openai")
        assert google.request_cap == 300.0
        assert openai.request_cap == 120.0
        assert openai.provider == "openai"
        for name in ("invocation_cap", "margin", "min_useful_attempt", "attempts"):
            assert getattr(openai, name) == getattr(google, name)

    def test_seat_client_timeout_survives_a_vendor_change(self):
        with _run(quick=False):
            health = agent_runtime.resolve_invocation_budget(
                "Health Check", provider="google"
            )
        assert policy_for_provider(health, "openai").request_cap == (
            HEALTH_CHECK_TIMEOUT
        )

    @pytest.mark.asyncio
    async def test_openai_dispatch_under_a_google_scope_sends_the_openai_ceiling(
        self, clock
    ):
        """The deadline is the scope's; the ceiling is the receiving vendor's."""
        import openai

        seen: dict = {}

        async def fake_request(self, cast_to, options, *args, **kwargs):
            seen["timeout"] = options.timeout
            raise RuntimeError("captured")

        cls = transports.deadline_aware_openai_cls()
        llm = cls(
            model="gpt-5.4",
            api_key="k",
            timeout=999,
            max_retries=0,
            dispatch_provider="openai",
        )
        with _run(quick=False):
            google = resolve_budget_policy(TimeoutClass.STANDARD, provider="google")
            assert google.request_cap == 300.0
            with (
                patch.object(openai.AsyncOpenAI, "request", fake_request),
                invocation_scope(google, clock() + 1_000.0) as scope,
                pytest.raises(RuntimeError, match="captured"),
            ):
                await llm._agenerate([HumanMessage(content="hi")])
        assert seen["timeout"] == pytest.approx(120.0)
        assert scope.dispatches[0]["timeout_s"] == pytest.approx(120.0)

    @pytest.mark.asyncio
    async def test_the_scope_deadline_still_bounds_the_other_vendor(self, clock):
        import openai

        seen: dict = {}

        async def fake_request(self, cast_to, options, *args, **kwargs):
            seen["timeout"] = options.timeout
            raise RuntimeError("captured")

        cls = transports.deadline_aware_openai_cls()
        llm = cls(
            model="gpt-5.4",
            api_key="k",
            timeout=999,
            max_retries=0,
            dispatch_provider="openai",
        )
        with _run(quick=False):
            google = resolve_budget_policy(TimeoutClass.STANDARD, provider="google")
            with (
                patch.object(openai.AsyncOpenAI, "request", fake_request),
                invocation_scope(google, clock() + 40.0),
                pytest.raises(RuntimeError, match="captured"),
            ):
                await llm._agenerate([HumanMessage(content="hi")])
        assert seen["timeout"] == pytest.approx(40.0 - MARGIN)


# ------------------------------------------------------------ editor review


def _editor(llm: Any, *, review_llm: Any = None):
    from src.article_writer import ArticleEditor

    editor = ArticleEditor.__new__(ArticleEditor)
    editor.llm = llm
    editor.review_llm = review_llm
    editor.llm_with_tools = None
    editor.tools = []
    editor._tools_by_name = {}
    editor.prompt_config = {"system_message": "You are an editor."}
    editor._tracing_metadata = {"ticker": "TEST"}
    editor._callbacks = []
    editor._url_cache = {}
    editor._failed_urls = set()
    return editor


_APPROVED = '{"verdict": "APPROVED", "confidence": 0.9}'


class TestEditorReview:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("quick", "cap"), [(True, CROSS_CHECK_CAP), (False, FULL_CAP)]
    )
    async def test_final_review_runs_under_the_review_plane_deadline(
        self, clock, quick, cap
    ):
        llm = _ScopeRecordingLLM(_APPROVED, provider="openai")
        with _run(quick=quick):
            result = await _editor(llm).review("draft", "context")
        assert result["verdict"] == "APPROVED"
        (scope,) = llm.scopes
        assert scope is not None, "editor review ran outside any invocation"
        assert scope.policy.timeout_class is TimeoutClass.CROSS_CHECK
        assert scope.deadline - clock() == pytest.approx(cap)
        # Tracing metadata still reaches the call.
        assert llm.configs[0]["metadata"]["workflow"] == "editor_final_review"
        assert llm.configs[0]["metadata"]["ticker"] == "TEST"

    @pytest.mark.asyncio
    async def test_structured_review_is_capped_too(self, clock):
        llm = _ScopeRecordingLLM(_APPROVED, provider="openai")
        review_llm = _ScopeRecordingLLM(provider="openai")

        async def structured(_input, config=None):
            review_llm.scopes.append(current_dispatch_scope())
            return {"verdict": "APPROVED", "confidence": 0.8}

        review_llm.ainvoke = structured  # type: ignore[method-assign]
        with _run(quick=True):
            result = await _editor(llm, review_llm=review_llm).review("d", "c")
        assert result == {"verdict": "APPROVED", "confidence": 0.8}
        assert review_llm.scopes[0].deadline - clock() == pytest.approx(CROSS_CHECK_CAP)
        assert llm.scopes == []  # the structured result short-circuits

    @pytest.mark.asyncio
    async def test_a_stalled_editor_times_out_at_its_cap_and_degrades(
        self, monkeypatch
    ):
        """Uncapped before: a hung review blocked article publication forever."""
        monkeypatch.setattr(
            config_module.config,
            "cross_check_quick_llm_call_hard_timeout_seconds",
            0.05,
        )
        stalled = asyncio.Event()

        class _Hangs(_ScopeRecordingLLM):
            async def ainvoke(self, _input, config=None):
                stalled.set()
                await asyncio.Event().wait()

        llm = _Hangs(provider="openai")
        with (
            _run(quick=True),
            patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await asyncio.wait_for(
                _editor(llm).review("draft", "context"), timeout=5.0
            )
        assert stalled.is_set()
        assert result == {"verdict": "REVISE", "confidence": 0.0, "review_error": True}

    @pytest.mark.asyncio
    async def test_editor_failure_still_degrades_to_revise(self):
        class _Explodes(_ScopeRecordingLLM):
            async def ainvoke(self, _input, config=None):
                raise ValueError("bad request")

        with _run(quick=False):
            result = await _editor(_Explodes(provider="openai")).review("d", "c")
        assert result["verdict"] == "REVISE"
        assert result["review_error"] is True


# ------------------------------------------------------------- retrospective


_COMPARISON = {
    "ticker": "TEST.L",
    "analysis_date": "2026-06-16",
    "verdict": "DO_NOT_INITIATE",
    "days_elapsed": 60,
    "price_return_pct": 5.9,
    "benchmark_return_pct": -20.0,
    "excess_return_pct": 25.9,
}


class TestRetrospectiveLesson:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("quick", "cap"), [(True, QUICK_CAP), (False, FULL_CAP)])
    async def test_lesson_runs_under_the_operational_policy(
        self, clock, monkeypatch, quick, cap
    ):
        import src.retrospective as retro

        llm = _ScopeRecordingLLM(
            "LESSON: check it\nTYPE: missed_risk\nFAILURE_MODE: MACRO_REGIME"
        )
        monkeypatch.setattr(
            "src.llm_runtime.construction.build_required_model_for_seat",
            lambda *a, **k: llm,
        )
        # A full-mode cap far above the quick one: the removed --quick clamp
        # used to hand the lesson min(this, 120) instead of its class cap.
        with _run(quick=quick, llm_call_hard_timeout_seconds=FULL_CAP):
            lesson = await retro.generate_lesson(dict(_COMPARISON))
        assert lesson is not None and lesson[0] == "check it"
        (scope,) = llm.scopes
        assert scope.policy.timeout_class is TimeoutClass.OPERATIONAL
        assert scope.deadline - clock() == pytest.approx(cap)

    @pytest.mark.asyncio
    async def test_quick_lesson_cap_ignores_the_full_mode_value(
        self, clock, monkeypatch
    ):
        import src.retrospective as retro

        llm = _ScopeRecordingLLM("LESSON: x\nTYPE: missed_risk")
        monkeypatch.setattr(
            "src.llm_runtime.construction.build_required_model_for_seat",
            lambda *a, **k: llm,
        )
        with _run(quick=True, llm_call_hard_timeout_seconds=600.0):
            await retro.generate_lesson(dict(_COMPARISON))
        assert llm.scopes[0].deadline - clock() == pytest.approx(QUICK_CAP)

    @pytest.mark.asyncio
    async def test_a_failed_lesson_call_still_returns_none(self, monkeypatch):
        import src.retrospective as retro

        class _Explodes(_ScopeRecordingLLM):
            async def ainvoke(self, _input, config=None):
                raise ValueError("bad request")

        monkeypatch.setattr(
            "src.llm_runtime.construction.build_required_model_for_seat",
            lambda *a, **k: _Explodes(),
        )
        with _run(quick=False):
            assert await retro.generate_lesson(dict(_COMPARISON)) is None


# ------------------------------------------------------------ semantic judge


class TestSemanticJudge:
    def _judge(self, monkeypatch, llm):
        from src.eval.semantic_judge import SemanticJudge

        monkeypatch.setattr(SemanticJudge, "_create_llm", lambda self, model: llm)
        return SemanticJudge()

    @pytest.mark.asyncio
    async def test_judge_runs_under_the_operational_policy(self, clock, monkeypatch):
        llm = _ScopeRecordingLLM('{"verdict": "PASS", "score": 0.9, "signals": []}')
        with _run(quick=False):
            result = await self._judge(monkeypatch, llm).judge_artifact(
                prompt_key="news_analyst",
                rubric_family="analysis_report",
                baseline_text="b",
                current_text="c",
            )
        assert result.verdict == "PASS"
        (scope,) = llm.scopes
        assert scope.policy.timeout_class is TimeoutClass.OPERATIONAL
        assert scope.deadline - clock() == pytest.approx(FULL_CAP)

    @pytest.mark.asyncio
    async def test_a_refused_judgement_is_still_a_hard_fail(self, monkeypatch):
        llm = _ScopeRecordingLLM("")

        async def refused(_input, config=None):
            message = AIMessage(content="")
            message.response_metadata = {"finish_reason": "SAFETY"}
            return message

        llm.ainvoke = refused  # type: ignore[method-assign]
        with _run(quick=False):
            result = await self._judge(monkeypatch, llm).judge_artifact(
                prompt_key="news_analyst",
                rubric_family="analysis_report",
                baseline_text="b",
                current_text="c",
            )
        assert result.verdict == "HARD_FAIL"
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_a_stalled_judge_raises_timeout_within_its_cap(self, monkeypatch):
        monkeypatch.setattr(config_module.config, "llm_call_hard_timeout_seconds", 0.05)

        class _Hangs(_ScopeRecordingLLM):
            async def ainvoke(self, _input, config=None):
                await asyncio.Event().wait()

        started = time.monotonic()
        with (
            _run(quick=False, llm_call_hard_timeout_seconds=0.05),
            patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(TimeoutError),
        ):
            await asyncio.wait_for(
                self._judge(monkeypatch, _Hangs()).judge_artifact(
                    prompt_key="news_analyst",
                    rubric_family="analysis_report",
                    baseline_text="b",
                    current_text="c",
                ),
                timeout=5.0,
            )
        # The judge's own cap fired, not the test's 5 s safety net.
        assert time.monotonic() - started < 2.0


# ------------------------------------------------- analyst structural retry


_UNPARSEABLE = SimpleNamespace(
    content=(
        "### FINANCIAL ANALYSIS\n\n### DATA_BLOCK\n\n"
        "| Metric | Value |\n| :--- | :--- |\n| Only | Two |\n\n"
        "### FINANCIAL HEALTH DETAIL\nLong enough body here."
    ),
    tool_calls=None,
)
_REPAIRED = SimpleNamespace(
    content=(
        "### --- START DATA_BLOCK ---\n"
        "SECTOR: Energy\n"
        "RAW_HEALTH_SCORE: 9.5/12\n"
        "ADJUSTED_HEALTH_SCORE: 79% (12/12 available)\n"
        "RAW_GROWTH_SCORE: 3/6\n"
        "ADJUSTED_GROWTH_SCORE: 50% (6/6 available)\n"
        "US_REVENUE_PERCENT: Not disclosed\n"
        "ANALYST_COVERAGE_ENGLISH: 2\n"
        "PE_RATIO_TTM: 12.35\n"
        "ADR_EXISTS: YES\n"
        "IBKR_ACCESSIBILITY: Direct\n"
        "PFIC_RISK: LOW\n"
        "### --- END DATA_BLOCK ---\n"
    ),
    tool_calls=None,
)


class TestAnalystStructuralRetry:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("quick", "cap"), [(True, APEX_CAP), (False, FULL_CAP)])
    async def test_retry_budget_is_the_resolvers_invocation_cap(self, quick, cap):
        """One source: the retry's overall budget is the seat's invocation cap."""
        from src.agents import create_analyst_node

        calls: list[dict] = []

        async def fake_invoke(_runnable, input_data, **kwargs):
            calls.append(kwargs)
            return _UNPARSEABLE if len(calls) == 1 else _REPAIRED

        with (
            _run(quick=quick),
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling", new=fake_invoke
            ),
        ):
            node = create_analyst_node(
                MagicMock(),
                "fundamentals_analyst",
                [],
                "fundamentals_report",
                retry_llm=MagicMock(),
                allow_retry=True,
            )
            await node(
                {
                    "messages": [],
                    "company_of_interest": "ALV.V",
                    "trade_date": "2026-03-18",
                },
                {
                    "configurable": {
                        "context": MagicMock(ticker="ALV.V", trade_date="2026-03-18")
                    }
                },
            )
            expected = agent_runtime.resolve_invocation_budget(
                "Fundamentals Analyst (RETRY-HIGH)",
                "Fundamentals Analyst",
                provider=calls[1]["provider"],
            ).invocation_cap
        retry = calls[1]
        assert retry["context"] == "Fundamentals Analyst (RETRY-HIGH)"
        assert retry["canonical_agent"] == "Fundamentals Analyst"
        assert retry["overall_timeout_seconds"] == pytest.approx(expected)
        assert retry["overall_timeout_seconds"] == pytest.approx(cap)

    @staticmethod
    async def _recover(failure_kind: str, *, cap_retry: bool) -> tuple[list, dict]:
        from src.agents import create_analyst_node

        calls: list[dict] = []

        async def fake_invoke(_runnable, input_data, **kwargs):
            calls.append(kwargs)
            return _UNPARSEABLE if len(calls) == 1 else _REPAIRED

        high = MagicMock(model_name="retry-model", thinking_level="high")
        low = MagicMock(model_name="retry-model", thinking_level="low")
        with (
            _run(quick=True),
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling", new=fake_invoke
            ),
            patch(
                "src.agents.analyst_nodes.classify_output_contract_failure",
                return_value=failure_kind,
            ),
        ):
            node = create_analyst_node(
                MagicMock(),
                "fundamentals_analyst",
                [],
                "fundamentals_report",
                retry_llm=high,
                cap_retry_llm=low if cap_retry else None,
                allow_retry=True,
            )
            result = await node(
                {
                    "messages": [],
                    "company_of_interest": "0995.HK",
                    "trade_date": "2026-10-03",
                },
                {
                    "configurable": {
                        "context": MagicMock(ticker="0995.HK", trade_date="2026-10-03")
                    }
                },
            )
        return calls, result["structural_recovery_events"][0]

    @pytest.mark.asyncio
    async def test_output_cap_failure_recovers_at_lowered_reasoning(self):
        """Raising effort after thinking consumed the cap fails the same way."""
        calls, event = await self._recover("output_cap_exhausted", cap_retry=True)

        assert calls[1]["context"] == "Fundamentals Analyst (RETRY-CAP)"
        assert calls[1]["canonical_agent"] == "Fundamentals Analyst"
        assert event["recovery_reasoning_setting"] == "low"
        assert event["outcome"] == "accepted_text"

    @pytest.mark.asyncio
    async def test_other_contract_failures_keep_the_high_reasoning_retry(self):
        calls, event = await self._recover("output_contract_violation", cap_retry=True)

        assert calls[1]["context"] == "Fundamentals Analyst (RETRY-HIGH)"
        assert event["recovery_reasoning_setting"] == "high"

    @pytest.mark.asyncio
    async def test_output_cap_failure_without_lowered_binding_keeps_retry_high(
        self,
    ):
        with patch("src.agents.analyst_nodes.logger") as mock_logger:
            calls, event = await self._recover("output_cap_exhausted", cap_retry=False)

        assert calls[1]["context"] == "Fundamentals Analyst (RETRY-HIGH)"
        assert event["recovery_reasoning_setting"] == "high"
        events = [call.args[0] for call in mock_logger.warning.call_args_list]
        assert "structural_recovery_cap_binding_unavailable" in events


# ---------------------------------------------------------------- consultant


def _consultant_state() -> dict:
    return {
        "company_of_interest": "TST",
        "company_name": "Test Co",
        "market_report": "x",
        "sentiment_report": "x",
        "news_report": "x",
        "fundamentals_report": "x",
        "investment_debate_state": {"history": "x"},
        "investment_plan": "BUY",
    }


async def _consultant_timeouts(node) -> list[float]:
    """Run a consultant node whose LLM call fails; return the budgets it passed."""
    from langgraph.types import RunnableConfig

    seen: list[float] = []

    async def fake_invoke(*_args, **kwargs):
        seen.append(float(kwargs["overall_timeout_seconds"]))
        raise TimeoutError("stubbed")

    prompt = SimpleNamespace(system_message="x", agent_name="External Consultant")
    with (
        patch("src.agents.runtime.invoke_with_rate_limit_handling", fake_invoke),
        patch("src.prompts.get_prompt", return_value=prompt),
    ):
        await node(
            _consultant_state(),
            RunnableConfig(
                configurable={"context": MagicMock(trade_date="2026-05-15")}
            ),
        )
    return seen


class TestConsultantBudgetsAreReadAtCallTime:
    @pytest.mark.asyncio
    async def test_a_config_change_after_import_and_build_is_honoured(
        self, monkeypatch
    ):
        from src.agents import consultant_nodes as cn_mod
        from src.agents import create_consultant_node

        node = create_consultant_node(MagicMock(), "consultant")
        # Changed after the module was imported and the node was built.
        monkeypatch.setattr(
            cn_mod.settings_config, "consultant_call_timeout_seconds", 999.0
        )
        monkeypatch.setattr(
            cn_mod.settings_config, "consultant_total_timeout_seconds", 7.0
        )
        with _run(quick=False):
            seen = await _consultant_timeouts(node)
        assert seen and 0.0 < seen[0] <= 7.0

        monkeypatch.setattr(
            cn_mod.settings_config, "consultant_call_timeout_seconds", 3.0
        )
        monkeypatch.setattr(
            cn_mod.settings_config, "consultant_total_timeout_seconds", 50.0
        )
        with _run(quick=False):
            seen = await _consultant_timeouts(node)
        assert seen[0] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_quick_budget_follows_the_run_config_not_the_build_flag(
        self, monkeypatch
    ):
        from src.agents import consultant_nodes as cn_mod
        from src.agents import create_consultant_node

        monkeypatch.setattr(
            cn_mod.settings_config, "consultant_call_timeout_seconds", 999.0
        )
        monkeypatch.setattr(
            cn_mod.settings_config, "consultant_quick_total_timeout_seconds", 9.0
        )
        monkeypatch.setattr(
            cn_mod.settings_config, "consultant_total_timeout_seconds", 400.0
        )
        # Built without the flag, run in --quick: the run config wins.
        node = create_consultant_node(MagicMock(), "consultant", quick_mode=False)
        with _run(quick=True):
            quick_seen = await _consultant_timeouts(node)
        assert 0.0 < quick_seen[0] <= 9.0
        with _run(quick=False):
            full_seen = await _consultant_timeouts(node)
        assert full_seen[0] > 9.0

    def test_total_budget_accessor_reads_the_run_config(self, monkeypatch):
        from src.agents import consultant_nodes as cn_mod

        monkeypatch.setattr(
            cn_mod.settings_config, "consultant_quick_total_timeout_seconds", 11.0
        )
        monkeypatch.setattr(
            cn_mod.settings_config, "consultant_total_timeout_seconds", 222.0
        )
        with _run(quick=True):
            assert cn_mod.consultant_total_timeout_seconds() == 11.0
        with _run(quick=False):
            assert cn_mod.consultant_total_timeout_seconds() == 222.0
        assert cn_mod.consultant_total_timeout_seconds(quick=True) == 11.0
