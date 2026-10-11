"""Per-dispatch budgets: one policy, an absolute deadline, application-owned retries.

Incident (October 2026): quick-mode Portfolio Manager calls ended in Google 504
DEADLINE_EXCEEDED at ~74 s. A flex-built Gemini client carried a 24 s
flex-attempt timeout fixed at construction; google-genai sends the client timeout
as ``X-Server-Timeout`` and Google enforces it server-side, so the same 24 s
deadline rode along on the standard re-issue, on calls sent at standard after
flex degraded, and on the model-failover copy -- three SDK attempts of it.

These tests pin the replacement (``src/llm_runtime/timeouts.py``): every request
is sized at dispatch from the invocation's absolute deadline, refused when too
little remains, sent with one SDK attempt, and recorded. A fake clock and fake
transports throughout; nothing sleeps and nothing reaches the network.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from google.genai import errors as genai_errors
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI

import src.config as config_module
import src.llms as llms_mod
from src.agents import invoke_with_rate_limit_handling
from src.agents.circuit_breaker import reset_circuit_breaker_for_tests
from src.agents.network_breaker import reset_network_breaker_singleton
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.adapters import transports
from src.llm_runtime.failover import failover_attempt
from src.llm_runtime.seats import SEATS, SeatId
from src.llm_runtime.timeouts import (
    BudgetExhausted,
    TimeoutClass,
    current_dispatch_scope,
    enclosing_deadline,
    enclosing_deadline_at,
    invocation_scope,
    request_timeout,
    resolve_budget_policy,
)
from src.runtime_config import RuntimeConfig, use_runtime_config
from src.token_tracker import get_tracker

PRIMARY = "gemini-3.8-flash"
FAILOVER = "gemini-3.7-flash"

# The numbers every expectation below is derived from.
QUICK_CAP = 60.0
APEX_CAP = 180.0
MARGIN = 2.0
MIN_USEFUL = 5.0
FRACTION = 0.4


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
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
        "cross_check_quick_llm_call_hard_timeout_seconds": APEX_CAP,
        "llm_call_hard_timeout_seconds": 120.0,
        "llm_min_useful_attempt_seconds": MIN_USEFUL,
        "llm_dispatch_margin_seconds": MARGIN,
        "flex_quick_attempt_fraction": FRACTION,
        "api_timeout": 300,
        "quick_llm_api_timeout_seconds": 120,
        "flex_llm_timeout_seconds": 900,
        "gemini_service_tier": "flex",
        "llm_attempts": 2,
    }
    targets = {id(config_module.config): config_module.config}
    targets[id(llms_mod.resolve_settings())] = llms_mod.resolve_settings()
    for target in targets.values():
        for key, value in values.items():
            monkeypatch.setattr(target, key, value, raising=False)
    # Bind the run-scoped view of exactly these values: a RuntimeConfig left
    # bound by another fixture would otherwise decide llm_attempts and quick
    # mode, and every expectation here is derived from the table above.
    with use_runtime_config(RuntimeConfig.from_config(config_module.config)):
        yield
    reset_circuit_breaker_for_tests()
    reset_network_breaker_singleton()


def _quick():
    return use_runtime_config(
        RuntimeConfig.from_config(config_module.config).with_overrides(
            quick_mode_active=True
        )
    )


def _result(model: str = PRIMARY) -> ChatResult:
    message = AIMessage(content=f"answer from {model}")
    message.response_metadata = {"model_name": model, "finish_reason": "STOP"}
    return ChatResult(generations=[ChatGeneration(message=message)])


def _server_error(code: int = 503) -> Exception:
    status = "UNAVAILABLE" if code == 503 else "DEADLINE_EXCEEDED"
    return genai_errors.ServerError(
        code, {"error": {"code": code, "message": status, "status": status}}
    )


def _gemini(**overrides) -> llms_mod._TieredChatGoogleGenerativeAI:
    kwargs = {"model": PRIMARY, "api_key": "k", "service_tier": "flex"}
    kwargs.update(overrides)
    return llms_mod._TieredChatGoogleGenerativeAI(**kwargs)


class _Transport:
    """Stands in for the SDK call: records the options each request carried.

    ``script`` maps a model to a queue of ``(seconds_spent, error_or_None)``.
    The recorded timeout is what the request would actually be sent with: the
    per-request option, or the instance's own value when none was passed.
    """

    def __init__(self, clock: FakeClock | None = None, script=None) -> None:
        self.clock = clock
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls: list[dict] = []

    def _record(self, llm, kwargs) -> None:
        self.calls.append(
            {
                "model": llm.model,
                "tier": llm._effective_tier(kwargs),
                "timeout": kwargs.get("timeout", llm.timeout),
                "max_retries": kwargs.get("max_retries", llm.max_retries),
            }
        )
        queue = self.script.get(llm.model) or []
        if queue:
            spent, error = queue.pop(0)
            if self.clock is not None:
                self.clock.advance(spent)
            if error is not None:
                raise error

    def patch_async(self):
        transport = self

        async def fake_agenerate(llm, messages, *args, **kwargs):
            transport._record(llm, kwargs)
            return _result(llm.model)

        return patch.object(ChatGoogleGenerativeAI, "_agenerate", fake_agenerate)

    def patch_sync(self):
        transport = self

        def fake_generate(llm, messages, *args, **kwargs):
            transport._record(llm, kwargs)
            return _result(llm.model)

        return patch.object(ChatGoogleGenerativeAI, "_generate", fake_generate)


def _policy(timeout_class=TimeoutClass.STANDARD, provider="google"):
    return resolve_budget_policy(timeout_class, provider=provider)


# --------------------------------------------------------------------- policy


class TestPolicy:
    def test_quick_standard_policy_numbers_come_from_config(self):
        with _quick():
            policy = _policy()
        assert policy.invocation_cap == QUICK_CAP
        assert policy.flex_request_cap == pytest.approx(QUICK_CAP * FRACTION)
        assert policy.request_cap == 120.0
        assert (policy.margin, policy.min_useful_attempt) == (MARGIN, MIN_USEFUL)
        assert policy.attempts == 2

    def test_quick_apex_policy_uses_its_own_cap_and_full_request_ceiling(self):
        with _quick():
            policy = _policy(TimeoutClass.APEX)
        assert policy.invocation_cap == APEX_CAP
        assert policy.request_cap == 300.0

    def test_full_mode_flex_cap_is_floored_and_flex_request_uses_the_floor(self):
        policy = _policy()
        assert policy.invocation_cap == 1350.0  # 1.5 x the 900 s flex floor
        assert policy.flex_request_cap == 900.0
        assert policy.request_cap == 300.0

    def test_every_registered_seat_carries_a_real_class(self):
        assert all(
            spec.timeout_class is not TimeoutClass.AUXILIARY for spec in SEATS.values()
        )
        assert SEATS[SeatId.PORTFOLIO_MANAGER].timeout_class is TimeoutClass.APEX
        assert SEATS[SeatId.CONSULTANT].timeout_class is TimeoutClass.CROSS_CHECK
        assert SEATS[SeatId.APAC].timeout_class is TimeoutClass.REGIONAL

    def test_a_seat_cannot_be_registered_as_auxiliary(self):
        spec = SEATS[SeatId.MARKET]
        with pytest.raises(ValueError, match="AUXILIARY"):
            type(spec)(**{**spec.__dict__, "timeout_class": TimeoutClass.AUXILIARY})

    def test_a_seat_needs_a_timeout_class_value(self):
        spec = SEATS[SeatId.MARKET]
        with pytest.raises(TypeError):
            type(spec)(**{**spec.__dict__, "timeout_class": "apex"})

    def test_unregistered_context_gets_the_bounded_auxiliary_policy(self):
        from src.agents.runtime import resolve_invocation_timeout_class

        assert (
            resolve_invocation_timeout_class("pm_verdict_recovery")
            is TimeoutClass.AUXILIARY
        )
        assert (
            resolve_invocation_timeout_class(
                "pm_verdict_recovery", canonical_agent="Portfolio Manager"
            )
            is TimeoutClass.APEX
        )


class TestRequestTimeout:
    def test_flex_and_standard_are_sized_from_the_deadline(self, clock):
        with _quick():
            policy = _policy()
        deadline = clock() + QUICK_CAP
        assert request_timeout(policy, "flex", deadline=deadline) == pytest.approx(24)
        assert request_timeout(policy, "standard", deadline=deadline) == (
            QUICK_CAP - MARGIN
        )

    def test_boundary_remaining_equal_to_min_useful_is_admitted(self, clock):
        with _quick():
            policy = _policy()
        deadline = clock() + MIN_USEFUL + MARGIN
        assert request_timeout(policy, "standard", deadline=deadline) == MIN_USEFUL

    def test_just_below_min_useful_is_refused(self, clock):
        with _quick():
            policy = _policy()
        deadline = clock() + MIN_USEFUL + MARGIN - 0.001
        with pytest.raises(BudgetExhausted):
            request_timeout(policy, "standard", deadline=deadline)

    def test_direct_caller_gets_the_tier_ceiling(self):
        with _quick():
            policy = _policy()
        assert request_timeout(policy, "flex", deadline=None) == pytest.approx(24)
        assert request_timeout(policy, "standard", deadline=None) == 120.0


# ------------------------------------------------------------------ dispatch


class TestDispatchSizing:
    @pytest.mark.asyncio
    async def test_quick_flex_503_then_standard_each_sized_and_one_attempt(self, clock):
        transport = _Transport(clock, {PRIMARY: [(3.0, _server_error(503))]})
        with _quick():
            policy = _policy()
            with (
                transport.patch_async(),
                invocation_scope(policy, clock() + QUICK_CAP) as scope,
            ):
                await _gemini()._agenerate([HumanMessage(content="hi")])
        flex, standard = transport.calls
        assert (flex["tier"], flex["timeout"], flex["max_retries"]) == (
            "flex",
            pytest.approx(24),
            1,
        )
        assert standard["tier"] == "standard"
        assert standard["timeout"] == pytest.approx(QUICK_CAP - 3.0 - MARGIN)
        assert standard["max_retries"] == 1
        assert [d["outcome"] for d in scope.dispatches] == ["server_error", "ok"]

    @pytest.mark.asyncio
    async def test_quick_pm_pinned_standard_one_request_sized_from_180(self, clock):
        transport = _Transport(clock)
        with _quick():
            policy = _policy(TimeoutClass.APEX)
            with (
                transport.patch_async(),
                invocation_scope(policy, clock() + APEX_CAP),
            ):
                await _gemini(service_tier=None, timeout=300)._agenerate(
                    [HumanMessage(content="decide")]
                )
        assert transport.calls == [
            {
                "model": PRIMARY,
                "tier": "standard",
                "timeout": APEX_CAP - MARGIN,
                "max_retries": 1,
            }
        ]

    @pytest.mark.asyncio
    async def test_degraded_flex_seat_is_sized_from_remaining_not_24(
        self, clock, monkeypatch
    ):
        """Today's regression: flex degraded, client flex-built -> not 24 s."""
        monkeypatch.setattr(llms_mod, "flex_degraded", lambda provider: True)
        transport = _Transport(clock)
        with _quick():
            # Built through the real factory, so a construction-time flex
            # rewrite (if it ever came back) would show up as llm.timeout.
            llm = llms_mod.create_gemini_model(PRIMARY, 0.3, 120)
            assert llm.service_tier == "flex"
            policy = _policy()
            with (
                transport.patch_async(),
                invocation_scope(policy, clock() + QUICK_CAP),
            ):
                await llm._agenerate([HumanMessage(content="hi")])
        (call,) = transport.calls
        assert call["tier"] == "standard"
        assert call["timeout"] == pytest.approx(QUICK_CAP - MARGIN)
        assert call["timeout"] != pytest.approx(QUICK_CAP * FRACTION)

    @pytest.mark.asyncio
    async def test_failover_delegate_reads_the_same_deadline(self, clock):
        """3.8 -> 3.7 after 100 s of a 180 s allowance: ~80 s left, no fresh clock."""
        transport = _Transport(clock)
        with _quick():
            policy = _policy(TimeoutClass.APEX)
            deadline = clock() + APEX_CAP
            clock.advance(100.0)
            with (
                transport.patch_async(),
                invocation_scope(policy, deadline) as scope,
                failover_attempt(),
            ):
                await _gemini(failover_model=FAILOVER)._agenerate(
                    [HumanMessage(content="hi")]
                )
        (call,) = transport.calls
        assert call["model"] == FAILOVER
        assert call["timeout"] == pytest.approx(APEX_CAP - 100.0 - MARGIN)
        assert scope.dispatches[0]["phase"] == "failover"

    @pytest.mark.asyncio
    async def test_too_little_remaining_refuses_with_zero_transport_calls(self, clock):
        transport = _Transport(clock)
        with _quick():
            policy = _policy()
            with (
                transport.patch_async(),
                invocation_scope(policy, clock() + MIN_USEFUL) as scope,
                pytest.raises(BudgetExhausted),
            ):
                await _gemini()._agenerate([HumanMessage(content="hi")])
        assert transport.calls == []
        assert scope.dispatches[0]["outcome"] == "budget_exhausted"

    @pytest.mark.asyncio
    async def test_budget_refusal_is_not_a_flex_health_event(self, clock):
        from src.service_tiers import flex_degradation_snapshot

        with _quick():
            policy = _policy()
            with (
                _Transport(clock).patch_async(),
                invocation_scope(policy, clock() + 1.0),
                pytest.raises(BudgetExhausted),
            ):
                await _gemini()._agenerate([HumanMessage(content="hi")])
        assert flex_degradation_snapshot() == {}

    @pytest.mark.asyncio
    async def test_small_cap_skips_flex_straight_to_standard(self, clock, monkeypatch):
        monkeypatch.setattr(
            config_module.config, "quick_llm_call_hard_timeout_seconds", 10.0
        )
        transport = _Transport(clock)
        with _quick():
            policy = _policy()  # flex share 4 s < 5 s minimum useful attempt
            with (
                transport.patch_async(),
                invocation_scope(policy, clock() + 10.0) as scope,
            ):
                await _gemini()._agenerate([HumanMessage(content="hi")])
        (call,) = transport.calls
        assert call["tier"] == "standard"
        assert call["timeout"] == pytest.approx(10.0 - policy.margin)
        assert scope.dispatches[0]["flex_skipped"] is True

    def test_sync_twin_sizes_flex_and_reissue_the_same_way(self, clock):
        transport = _Transport(clock, {PRIMARY: [(3.0, _server_error(503))]})
        with _quick():
            policy = _policy()
            with (
                transport.patch_sync(),
                invocation_scope(policy, clock() + QUICK_CAP),
            ):
                _gemini()._generate([HumanMessage(content="hi")])
        assert [(c["tier"], c["max_retries"]) for c in transport.calls] == [
            ("flex", 1),
            ("standard", 1),
        ]
        assert transport.calls[0]["timeout"] == pytest.approx(24)
        assert transport.calls[1]["timeout"] == pytest.approx(QUICK_CAP - 3 - MARGIN)

    @pytest.mark.asyncio
    async def test_full_mode_flex_timeout_reaches_the_standard_fallback(self, clock):
        """Full mode: the 900 s flex request leaves room for a standard re-issue."""
        transport = _Transport(clock, {PRIMARY: [(900.0, TimeoutError("queued"))]})
        policy = _policy()
        with (
            transport.patch_async(),
            invocation_scope(policy, clock() + policy.invocation_cap),
        ):
            await _gemini()._agenerate([HumanMessage(content="hi")])
        flex, standard = transport.calls
        assert flex["timeout"] == 900.0
        assert standard["tier"] == "standard"
        assert standard["timeout"] == 300.0  # min(request cap, 448 s remaining)

    @pytest.mark.asyncio
    async def test_full_mode_fast_503_reaches_the_standard_fallback(self, clock):
        transport = _Transport(clock, {PRIMARY: [(1.0, _server_error(503))]})
        policy = _policy()
        with (
            transport.patch_async(),
            invocation_scope(policy, clock() + policy.invocation_cap),
        ):
            await _gemini()._agenerate([HumanMessage(content="hi")])
        assert [c["tier"] for c in transport.calls] == ["flex", "standard"]

    @pytest.mark.asyncio
    async def test_direct_caller_keeps_constructed_standard_options(self):
        """Outside an application invocation, only a flex request is sized."""
        transport = _Transport()
        with transport.patch_async():
            await _gemini(service_tier=None, timeout=10)._agenerate(
                [HumanMessage(content="ok?")]
            )
        assert transport.calls[0]["timeout"] == 10
        assert transport.calls[0]["max_retries"] == 1


class TestContextIsolation:
    @pytest.mark.asyncio
    async def test_concurrent_calls_on_one_model_keep_their_own_deadlines(self, clock):
        llm = _gemini(service_tier=None)
        seen: dict[str, float] = {}

        async def fake_agenerate(model, messages, *args, **kwargs):
            await asyncio.sleep(0)
            seen[messages[0].content] = kwargs["timeout"]
            return _result()

        with _quick():
            policy = _policy()

            async def call(label: str, seconds: float) -> None:
                with invocation_scope(policy, clock() + seconds):
                    await llm._agenerate([HumanMessage(content=label)])

            with patch.object(ChatGoogleGenerativeAI, "_agenerate", fake_agenerate):
                await asyncio.gather(call("short", 20.0), call("long", 50.0))
        assert seen == {"short": 18.0, "long": 48.0}

    @pytest.mark.asyncio
    async def test_cancellation_mid_dispatch_leaks_no_deadline(self, clock):
        started = asyncio.Event()

        async def hangs(model, messages, *args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        with _quick():
            policy = _policy()

            async def call() -> None:
                with invocation_scope(policy, clock() + QUICK_CAP):
                    await _gemini()._agenerate([HumanMessage(content="hi")])

            with patch.object(ChatGoogleGenerativeAI, "_agenerate", hangs):
                task = asyncio.create_task(call())
                await started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert current_dispatch_scope() is None
        transport = _Transport()
        with transport.patch_async():
            await _gemini(service_tier=None, timeout=77)._agenerate(
                [HumanMessage(content="next")]
            )
        assert transport.calls[0]["timeout"] == 77


# ------------------------------------------------------------- runtime loop


async def _invoke(llm, context: str, **kwargs):
    return await invoke_with_rate_limit_handling(
        llm,
        [HumanMessage(content="decide")],
        context=context,
        provider="google",
        model_name=llm.model if isinstance(llm, ChatGoogleGenerativeAI) else "m",
        **kwargs,
    )


def _attempts(context: str):
    return [a for a in get_tracker().call_attempts if a.agent_name == context]


class TestRuntimeLoop:
    @pytest.mark.asyncio
    async def test_enclosing_deadline_shorter_than_cap_sizes_the_request(self, clock):
        transport = _Transport(clock)
        with _quick(), transport.patch_async(), enclosing_deadline(30.0):
            await _invoke(_gemini(service_tier=None), "Portfolio Manager")
        assert transport.calls[0]["timeout"] == pytest.approx(30.0 - MARGIN)

    @pytest.mark.asyncio
    async def test_correction_gets_its_allowance_but_not_past_the_node_deadline(
        self, clock
    ):
        transport = _Transport(clock, {PRIMARY: [(170.0, None)]})
        llm = _gemini(service_tier=None)
        with _quick(), transport.patch_async(), enclosing_deadline(200.0):
            await _invoke(llm, "Portfolio Manager")
            await _invoke(
                llm,
                "Portfolio Manager structure correction",
                canonical_agent="Portfolio Manager",
            )
        first, correction = transport.calls
        assert first["timeout"] == pytest.approx(APEX_CAP - MARGIN)
        assert correction["timeout"] == pytest.approx(30.0 - MARGIN)

    @pytest.mark.asyncio
    async def test_transient_503_retry_is_admitted_and_backoff_clamped(self, clock):
        runnable = AsyncMock()
        calls = {"n": 0}

        async def fails_once(*_a, **_kw):
            calls["n"] += 1
            if calls["n"] == 1:
                clock.advance(10.0)
                raise _server_error(503)
            return AIMessage(content="ok")

        runnable.ainvoke = AsyncMock(side_effect=fails_once)
        with (
            patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock) as sleep,
            patch("src.agents.runtime.random.uniform", return_value=3.0),
        ):
            # 20 s overall: at t=10 only 10 - 2 - 5 = 3 s may be slept.
            await _invoke(runnable, "TransientClamp", overall_timeout_seconds=20.0)
        assert runnable.ainvoke.await_count == 2
        sleep.assert_awaited_once()
        assert sleep.await_args.args[0] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_transient_retry_keeps_full_backoff_when_it_fits(self, clock):
        runnable = AsyncMock()
        calls = {"n": 0}

        async def fails_once(*_a, **_kw):
            calls["n"] += 1
            if calls["n"] == 1:
                clock.advance(10.0)
                raise _server_error(503)
            return AIMessage(content="ok")

        runnable.ainvoke = AsyncMock(side_effect=fails_once)
        with (
            patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock) as sleep,
            patch("src.agents.runtime.random.uniform", return_value=3.0),
        ):
            await _invoke(runnable, "TransientFits", overall_timeout_seconds=60.0)
        assert sleep.await_args.args[0] == pytest.approx(8.0)  # 5 x 1 + 3

    @pytest.mark.asyncio
    async def test_rate_limit_without_room_for_wait_and_attempt_raises(self, clock):
        runnable = AsyncMock()

        async def rate_limited(*_a, **_kw):
            clock.advance(10.0)
            raise genai_errors.ClientError(
                429,
                {
                    "error": {
                        "code": 429,
                        "message": "RESOURCE_EXHAUSTED",
                        "status": "RESOURCE_EXHAUSTED",
                    }
                },
            )

        runnable.ainvoke = AsyncMock(side_effect=rate_limited)
        with (
            patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock) as sleep,
            pytest.raises(genai_errors.ClientError),
        ):
            await _invoke(runnable, "RateLimitRefused", overall_timeout_seconds=60.0)
        assert runnable.ainvoke.await_count == 1
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_transient_budget_defaults_to_llm_attempts(self):
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=_server_error(503))
        with (
            use_runtime_config(
                RuntimeConfig.from_config(config_module.config).with_overrides(
                    llm_attempts=3
                )
            ),
            patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(genai_errors.ServerError),
        ):
            await _invoke(runnable, "AttemptsDefault")
        assert runnable.ainvoke.await_count == 3

    @pytest.mark.asyncio
    async def test_flex_then_standard_records_two_dispatches(self, clock):
        get_tracker().reset()
        transport = _Transport(clock, {PRIMARY: [(3.0, _server_error(503))]})
        with _quick(), transport.patch_async():
            await _invoke(_gemini(), "News Analyst")
        (attempt,) = _attempts("News Analyst")
        flex, standard = attempt.dispatches
        assert (flex["tier"], flex["outcome"]) == ("flex", "server_error")
        assert (standard["tier"], standard["outcome"]) == ("standard", "ok")
        assert flex["timeout_s"] == pytest.approx(24)
        assert standard["timeout_s"] == pytest.approx(QUICK_CAP - 3 - MARGIN)
        assert set(flex) >= {"model", "remaining_s", "elapsed_s", "phase"}
        assert not any(isinstance(v, dict | list) for v in flex.values())

    @pytest.mark.asyncio
    async def test_budget_refusal_is_recorded_and_spares_the_breaker(
        self, clock, monkeypatch
    ):
        from src.agents.circuit_breaker import get_circuit_breaker

        get_tracker().reset()
        breaker = get_circuit_breaker()
        outcomes: list = []
        monkeypatch.setattr(
            breaker, "record_outcome", lambda **kw: outcomes.append(kw.get("ok"))
        )
        transport = _Transport(clock)
        with (
            _quick(),
            transport.patch_async(),
            patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(BudgetExhausted),
        ):
            await _invoke(_gemini(), "Refused", overall_timeout_seconds=3.0)
        assert transport.calls == []
        (attempt,) = _attempts("Refused")
        assert attempt.failure_origin == "budget_exhausted"
        assert attempt.dispatches[0]["outcome"] == "budget_exhausted"
        assert False not in outcomes


class TestRunDeadlineCeiling:
    """The ticker run deadline (src/llm_runtime/run_deadline.py) is a hard ceiling.

    Bound with enclosing_deadline_at around the analysis, it caps every node
    deadline inside it, so admission truncates the retry ladder near the
    pipeline watchdog instead of the watchdog killing the process.
    """

    @pytest.mark.asyncio
    async def test_spent_run_deadline_refuses_the_first_call_without_a_request(
        self, clock
    ):
        transport = _Transport(clock)
        with (
            _quick(),
            transport.patch_async(),
            enclosing_deadline_at(clock() - 1.0),
            pytest.raises(TimeoutError),
        ):
            await _invoke(_gemini(service_tier=None), "Portfolio Manager")
        assert transport.calls == []

    @pytest.mark.asyncio
    async def test_run_deadline_below_a_useful_attempt_is_refused_by_admission(
        self, clock
    ):
        transport = _Transport(clock)
        with (
            _quick(),
            transport.patch_async(),
            patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock),
            enclosing_deadline_at(clock() + MARGIN + MIN_USEFUL - 1.0),
            pytest.raises(BudgetExhausted),
        ):
            await _invoke(_gemini(service_tier=None), "Portfolio Manager")
        assert transport.calls == []

    @pytest.mark.asyncio
    async def test_node_deadline_inside_the_run_deadline_takes_the_earlier(self, clock):
        transport = _Transport(clock)
        with (
            _quick(),
            transport.patch_async(),
            enclosing_deadline_at(clock() + 100.0),
            enclosing_deadline(240.0),
        ):
            await _invoke(_gemini(service_tier=None), "Portfolio Manager")
        assert transport.calls[0]["timeout"] == pytest.approx(100.0 - MARGIN)

    @pytest.mark.asyncio
    async def test_run_deadline_truncates_the_retry_ladder(self, clock):
        # A 503 at t=156 of a 160 s run allowance: a retry needs a useful
        # attempt plus margin (7 s) that no longer fits, so the loop stops
        # rather than sending a request the watchdog would cut off.
        runnable = AsyncMock()

        async def slow_failure(*_a, **_kw):
            clock.advance(156.0)
            raise _server_error(503)

        runnable.ainvoke = AsyncMock(side_effect=slow_failure)
        with (
            _quick(),
            patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock) as sleep,
            enclosing_deadline_at(clock() + 160.0),
            pytest.raises(genai_errors.ServerError),
        ):
            await _invoke(runnable, "Portfolio Manager")
        assert runnable.ainvoke.await_count == 1
        sleep.assert_not_awaited()


# ---------------------------------------------------------------- semantics


class TestAttemptSemantics:
    def test_llm_attempts_env_precedence_over_the_deprecated_spelling(
        self, monkeypatch
    ):
        from src.config import Settings

        # The single-spelling values are covered by test_settings_hygiene.py.
        monkeypatch.setenv("API_RETRY_ATTEMPTS", "3")
        monkeypatch.setenv("LLM_ATTEMPTS", "4")
        settings = Settings(_env_file=None)
        assert settings.llm_attempts == 4
        assert settings.api_retry_attempts == 4  # alias property


class TestSdkRequestLayer:
    """The per-request options reach the vendor SDK's request layer."""

    @pytest.mark.asyncio
    async def test_google_request_carries_server_deadline_and_one_attempt(self, clock):
        from google.genai import _api_client

        captured: dict = {}

        async def fake_async_request(
            self, http_request, http_options=None, stream=False
        ):
            captured["header"] = http_request.headers.get("X-Server-Timeout")
            captured["timeout"] = http_request.timeout
            options = http_options
            if isinstance(options, dict):
                options = _api_client.HttpOptions(**options)
            captured["attempts"] = options.retry_options.attempts
            body = {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "ok"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ]
            }
            return _api_client.HttpResponse({}, [json.dumps(body)])

        llm = _gemini(service_tier=None, timeout=300)
        with _quick():
            policy = _policy()
            with (
                patch.object(
                    _api_client.BaseApiClient, "_async_request", fake_async_request
                ),
                invocation_scope(policy, clock() + 40.0),
            ):
                await llm._agenerate([HumanMessage(content="hi")])
        assert captured["attempts"] == 1
        assert captured["timeout"] == pytest.approx(38.0)
        assert captured["header"] == "38"
