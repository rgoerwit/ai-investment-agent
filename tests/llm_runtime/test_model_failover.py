"""Same-vendor model failover on the transient retry.

Incident (2026-10-01): the Portfolio Manager on gemini-3.8-flash hit
504 DEADLINE_EXCEEDED on 5 of 40 quick-mode calls, each after Google had already
been degraded to the standard tier, so the only fallback (a tier change on the
same model) had nothing left to give. gemini-3.7-flash had 474 successes and no
failures on the same seat in Aug-Sep. These tests pin the three layers:
the binding plan decides *what*, the retry loop decides *when*, and the tiered
transports decide *how*.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from google.genai import errors as genai_errors
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import src.llms as llms_mod
from src.agents import invoke_with_rate_limit_handling
from src.agents.circuit_breaker import reset_circuit_breaker_for_tests
from src.agents.network_breaker import reset_network_breaker_singleton
from src.config import Settings
from src.llm_runtime.bindings import BindingConfigurationError, resolve_binding_plan
from src.llm_runtime.construction import build_model_for_seat
from src.llm_runtime.failover import (
    ModelFailoverMixin,
    attach_failover,
    failover_attempt,
    failover_target,
)
from src.llm_runtime.seats import SeatId
from src.token_tracker import get_tracker

PRIMARY = "gemini-3.8-flash"
FAILOVER = "gemini-3.7-flash"


@pytest.fixture(autouse=True)
def _reset_breakers():
    reset_circuit_breaker_for_tests()
    reset_network_breaker_singleton()
    yield
    reset_circuit_breaker_for_tests()
    reset_network_breaker_singleton()


def _settings(**overrides) -> Settings:
    values = {
        "llm_base_provider": "google",
        "llm_review_provider": "openai",
        "llm_regional_provider": "deepseek",
        "llm_writer_provider": "anthropic",
        "llm_operational_provider": "google",
        "llm_judge_provider": "google",
        "google_api_key": "google-key",
        "openai_api_key": "openai-key",
        "claude_api_key": "anthropic-key",
        "deepseek_api_key": "deepseek-key",
        "google_llm_fast_model": "gemini-3.1-flash-lite",
        "google_llm_reasoning_model": PRIMARY,
        "google_llm_critical_model": "gemini-3.1-pro-preview",
        "openai_llm_reasoning_model": "gpt-6.1-sol",
        "anthropic_llm_prose_model": "claude-sonnet-5-5",
        "deepseek_llm_reasoning_model": "deepseek-flash",
        "llm_seat_quick_model_overrides": {"portfolio_manager": PRIMARY},
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _server_504() -> Exception:
    return genai_errors.ServerError(
        504,
        {
            "error": {
                "code": 504,
                "message": "Deadline expired before operation could complete.",
                "status": "DEADLINE_EXCEEDED",
            }
        },
    )


def _client_error(code: int, status: str) -> Exception:
    return genai_errors.ClientError(
        code, {"error": {"code": code, "message": status, "status": status}}
    )


def _result(model: str) -> ChatResult:
    message = AIMessage(content=f"answer from {model}")
    message.response_metadata = {"model_name": model, "finish_reason": "STOP"}
    return ChatResult(generations=[ChatGeneration(message=message)])


def _gemini(**overrides) -> llms_mod._TieredChatGoogleGenerativeAI:
    kwargs = {
        "model": PRIMARY,
        "api_key": "test-key",
        "service_tier": "flex",
        "failover_model": FAILOVER,
    }
    kwargs.update(overrides)
    return llms_mod._TieredChatGoogleGenerativeAI(**kwargs)


class _Transport:
    """Stands in for the Google SDK call; records who was asked, at which tier."""

    def __init__(self, failures: dict[str, list[Exception]]):
        self.failures = {k: list(v) for k, v in failures.items()}
        self.calls: list[tuple[str, str | None]] = []

    def patch(self):
        transport = self

        async def fake_agenerate(llm, messages, *args, **kwargs):
            tier = kwargs.get("service_tier") or llm.service_tier
            transport.calls.append((llm.model, tier))
            queue = transport.failures.get(llm.model) or []
            if queue:
                raise queue.pop(0)
            return _result(llm.model)

        return patch.object(ChatGoogleGenerativeAI, "_agenerate", fake_agenerate)


# --------------------------------------------------------------------------- what


class TestBindingPlan:
    def test_every_binding_on_the_source_model_inherits_the_failover(self):
        plan = resolve_binding_plan(_settings(llm_model_failovers={PRIMARY: FAILOVER}))
        all_bindings = [*plan.bindings.values(), *plan.quick_bindings.values()]
        on_primary = [b for b in all_bindings if b.model == PRIMARY]
        assert len(on_primary) > 5  # debate, risk, PM (quick), recovery...
        assert {b.failover_model for b in on_primary} == {FAILOVER}
        assert all(b.failover_model is None for b in all_bindings if b.model != PRIMARY)
        assert (
            plan.for_seat(SeatId.PORTFOLIO_MANAGER, quick_mode=True).failover_model
            == FAILOVER
        )

    def test_telemetry_records_the_failover(self):
        settings = _settings(llm_model_failovers={PRIMARY: FAILOVER})
        seats = resolve_binding_plan(settings).telemetry(settings)["seats"]
        assert seats["portfolio_manager"]["quick_failover_model"] == FAILOVER
        assert seats["market_analyst"]["failover_model"] is None

    def test_no_map_means_no_failover(self):
        plan = resolve_binding_plan(_settings())
        assert all(b.failover_model is None for b in plan.bindings.values())

    @pytest.mark.parametrize(
        ("failovers", "message"),
        [
            ({PRIMARY: PRIMARY}, "cannot fail over to itself"),
            ({PRIMARY: FAILOVER, FAILOVER: "gemini-3.6-flash"}, "one hop"),
            ({PRIMARY: "mystery-model-1"}, "no reviewed capability profile"),
            ({PRIMARY: "gpt-6.1-sol"}, "within one vendor"),
            ({"deepseek-flash": "deepseek-v4-pro"}, "does not support model failover"),
            (
                {"claude-sonnet-5-5": "claude-opus-5-5"},
                "does not support model failover",
            ),
        ],
    )
    def test_unservable_failovers_fail_closed_at_startup(self, failovers, message):
        with pytest.raises(BindingConfigurationError) as raised:
            resolve_binding_plan(_settings(llm_model_failovers=failovers))
        assert any(message in error for error in raised.value.errors), (
            raised.value.errors
        )

    def test_legacy_schema_rejects_a_failover_map(self):
        settings = Settings(
            _env_file=None,
            google_api_key="google-key",
            llm_model_failovers={PRIMARY: FAILOVER},
        )
        with pytest.raises(BindingConfigurationError, match="provider-scoped"):
            resolve_binding_plan(settings)

    def test_unused_source_model_warns_but_resolves(self):
        with patch("src.llm_runtime.bindings.logger") as logger:
            resolve_binding_plan(
                _settings(llm_model_failovers={"gemini-3.6-flash": FAILOVER})
            )
        logger.warning.assert_called_once_with(
            "llm_model_failover_unused", source_model="gemini-3.6-flash"
        )


class TestFailoverRequestShape:
    """The failover copy inherits the source's request settings, so the target
    must accept exactly what this seat sends, or the failover fails when needed."""

    def test_reasoning_value_the_target_lacks_fails_closed(self):
        settings = _settings(
            llm_model_failovers={"gpt-6.1-sol": "gpt-5.4"},
            llm_seat_reasoning_overrides={"consultant": "max"},
        )
        with pytest.raises(BindingConfigurationError) as exc:
            resolve_binding_plan(settings)
        assert any(
            "consultant" in e and "'gpt-5.4'" in e and "reasoning 'max'" in e
            for e in exc.value.errors
        )

    def test_reasoning_value_the_target_supports_resolves(self):
        settings = _settings(
            llm_model_failovers={"gpt-6.1-sol": "gpt-5.4"},
            llm_seat_reasoning_overrides={"consultant": "xhigh"},
        )
        plan = resolve_binding_plan(settings)
        assert plan.for_seat(SeatId.CONSULTANT, quick_mode=False).failover_model == (
            "gpt-5.4"
        )

    def test_default_reasoning_is_checked_without_an_override(self):
        """No override: the value the seat would pick from the source ladder."""
        resolve_binding_plan(_settings(llm_model_failovers={"gpt-6.1-sol": "gpt-5.4"}))
        resolve_binding_plan(_settings(llm_model_failovers={PRIMARY: FAILOVER}))

    def test_differing_temperature_policy_fails_closed(self):
        settings = _settings(llm_model_failovers={"gpt-6.1-sol": "gpt-4o"})
        with pytest.raises(BindingConfigurationError) as exc:
            resolve_binding_plan(settings)
        assert any("temperature_policy" in e for e in exc.value.errors)


class TestConstruction:
    def test_seat_model_carries_its_failover(self):
        settings = _settings(llm_model_failovers={PRIMARY: FAILOVER})
        model = build_model_for_seat(
            SeatId.PORTFOLIO_MANAGER,
            settings=settings,
            plan=resolve_binding_plan(settings),
            quick_mode=True,
        )
        assert isinstance(model, ModelFailoverMixin)
        assert model.failover_model == FAILOVER

    def test_standard_tier_openai_seat_uses_the_failover_capable_transport(self):
        settings = _settings(
            openai_service_tier="auto",
            llm_model_failovers={"gpt-6.1-sol": "gpt-5.4"},
        )
        model = build_model_for_seat(
            SeatId.CONSULTANT, settings=settings, plan=resolve_binding_plan(settings)
        )
        assert isinstance(model, llms_mod._get_flex_fallback_chat_openai_cls())
        assert model.failover_model == "gpt-5.4"
        assert model.service_tier != "flex"  # failover capability, not flex

    def test_openai_seat_without_failover_stays_plain(self):
        settings = _settings(openai_service_tier="auto")
        model = build_model_for_seat(
            SeatId.CONSULTANT, settings=settings, plan=resolve_binding_plan(settings)
        )
        assert not isinstance(model, ModelFailoverMixin)

    def test_attach_refuses_a_transport_that_cannot_serve_it(self):
        with pytest.raises(TypeError, match="cannot serve a model failover"):
            attach_failover(object(), FAILOVER)
        sentinel = object()
        assert attach_failover(sentinel, None) is sentinel


# ---------------------------------------------------------------------------- how


class TestTransport:
    @pytest.mark.asyncio
    async def test_failover_attempt_serves_from_the_failover_model_at_standard(self):
        transport = _Transport({})
        with transport.patch(), failover_attempt():
            result = await _gemini()._agenerate([HumanMessage(content="hi")])
        assert transport.calls == [(FAILOVER, "standard")]
        assert result.generations[0].message.response_metadata["model_name"] == FAILOVER

    @pytest.mark.asyncio
    async def test_ordinary_attempt_uses_the_primary(self):
        transport = _Transport({})
        with transport.patch():
            await _gemini(service_tier=None)._agenerate([HumanMessage(content="hi")])
        assert transport.calls == [(PRIMARY, None)]

    @pytest.mark.asyncio
    async def test_failover_attempt_without_a_target_uses_the_primary(self):
        transport = _Transport({})
        with transport.patch(), failover_attempt():
            await _gemini(failover_model=None, service_tier=None)._agenerate(
                [HumanMessage(content="hi")]
            )
        assert transport.calls == [(PRIMARY, None)]

    @pytest.mark.asyncio
    async def test_failover_does_not_recurse_or_mutate_the_primary(self):
        llm = _gemini()
        transport = _Transport({FAILOVER: [_server_504()]})
        with (
            transport.patch(),
            failover_attempt(),
            pytest.raises(genai_errors.ServerError),
        ):
            await llm._agenerate([HumanMessage(content="hi")])
        assert transport.calls == [(FAILOVER, "standard")]
        assert (llm.model, llm.service_tier, llm.failover_model) == (
            PRIMARY,
            "flex",
            FAILOVER,
        )

    def test_sync_path_fails_over_too(self):
        captured = []

        def fake_generate(llm, messages, *args, **kwargs):
            captured.append((llm.model, llm.service_tier))
            return _result(llm.model)

        with (
            patch.object(ChatGoogleGenerativeAI, "_generate", fake_generate),
            failover_attempt(),
        ):
            _gemini()._generate([HumanMessage(content="hi")])
        assert captured == [(FAILOVER, "standard")]

    @pytest.mark.asyncio
    async def test_openai_transport_fails_over_at_auto(self):
        from langchain_openai import ChatOpenAI

        cls = llms_mod._get_flex_fallback_chat_openai_cls()
        llm = cls(
            model="gpt-6.1-sol",
            api_key="k",
            service_tier="flex",
            failover_model="gpt-5.4",
        )
        captured = []

        async def fake_agenerate(model, messages, *args, **kwargs):
            captured.append((model.model_name, model.service_tier))
            return _result(model.model_name)

        with patch.object(ChatOpenAI, "_agenerate", fake_agenerate), failover_attempt():
            await llm._agenerate([HumanMessage(content="hi")])
        assert captured == [("gpt-5.4", "auto")]

    def test_target_is_found_through_prompt_and_tool_bindings(self):
        llm = _gemini()
        prompt = ChatPromptTemplate.from_messages([("human", "{q}")])
        assert failover_target(llm) == FAILOVER
        assert failover_target(prompt | llm) == FAILOVER
        assert failover_target(prompt | llm.bind(stop=["x"])) == FAILOVER
        assert failover_target(prompt | _gemini(failover_model=None)) is None
        assert failover_target(AsyncMock()) is None


# --------------------------------------------------------------------------- when


async def _invoke(llm, context: str, **kwargs):
    with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
        try:
            result = await invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content="decide")],
                context=context,
                provider="google",
                model_name=llm.model,
                **kwargs,
            )
        finally:
            attempts = [
                (a.model_name, a.status, a.failure_kind)
                for a in get_tracker().call_attempts
                if a.agent_name == context
            ]
    return result, attempts, sleep


class TestRetryLoop:
    @pytest.mark.asyncio
    async def test_flex_504_recovers_on_the_primary_before_any_failover(self):
        """The tier rung comes first: a different tier of the same model has not
        been tried yet, so a flex 504 must not spend the failover."""
        transport = _Transport({PRIMARY: [_server_504()]})
        with transport.patch():
            result, attempts, _ = await _invoke(_gemini(), "PM flex 504")
        assert transport.calls == [(PRIMARY, "flex"), (PRIMARY, "standard")]
        assert result.response_metadata["model_name"] == PRIMARY
        assert attempts == [(PRIMARY, "success", None)]

    @pytest.mark.asyncio
    async def test_504_spends_the_retry_on_the_failover_model(self):
        """The 2026-10-01 incident: 3.8 504s at flex and again at standard."""
        transport = _Transport({PRIMARY: [_server_504(), _server_504()]})
        with transport.patch():
            result, attempts, sleep = await _invoke(_gemini(), "PM failover 504")
        assert transport.calls == [
            (PRIMARY, "flex"),
            (PRIMARY, "standard"),
            (FAILOVER, "standard"),
        ]
        assert result.response_metadata["model_name"] == FAILOVER
        assert attempts == [
            (PRIMARY, "failure", "server_error"),
            (FAILOVER, "success", None),
        ]
        sleep.assert_not_awaited()  # a different model is not waiting out this one

    @pytest.mark.asyncio
    async def test_hard_timeout_fails_over_like_a_504(self):
        transport = _Transport({PRIMARY: [TimeoutError("hard timeout of 60.0s")] * 2})
        with transport.patch():
            _, attempts, _ = await _invoke(_gemini(), "PM failover timeout")
        assert [model for model, _, _ in attempts] == [PRIMARY, FAILOVER]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error", "calls"),
        [
            (_client_error(429, "RESOURCE_EXHAUSTED"), [PRIMARY, PRIMARY]),
            (_client_error(400, "INVALID_ARGUMENT"), [PRIMARY]),
        ],
    )
    async def test_non_health_failures_never_fail_over(self, error, calls):
        transport = _Transport({PRIMARY: [error]})
        with transport.patch():
            try:
                await _invoke(_gemini(), f"PM no failover {error}")
            except genai_errors.ClientError:
                pass
        assert [model for model, _ in transport.calls] == calls

    @pytest.mark.asyncio
    async def test_both_models_failing_stops_after_the_failover(self):
        transport = _Transport(
            {PRIMARY: [_server_504(), _server_504()], FAILOVER: [_server_504()]}
        )
        with transport.patch(), pytest.raises(genai_errors.ServerError):
            await _invoke(_gemini(), "PM failover exhausted")
        assert transport.calls == [
            (PRIMARY, "flex"),
            (PRIMARY, "standard"),
            (FAILOVER, "standard"),
        ]

    @pytest.mark.asyncio
    async def test_a_failing_failover_is_never_failed_over_again(self):
        """Once on the failover model, later retries stay there: the ladder never
        bounces back to the model that just failed, nor hops a second time."""
        transport = _Transport(
            {PRIMARY: [_server_504()] * 9, FAILOVER: [_server_504()]}
        )
        with transport.patch():
            try:
                await _invoke(
                    _gemini(), "PM failover retried", max_transient_attempts=3
                )
            except genai_errors.ServerError:
                pass
        after_failover = transport.calls[
            transport.calls.index((FAILOVER, "standard")) :
        ]
        assert {model for model, _ in after_failover} == {FAILOVER}

    @pytest.mark.asyncio
    async def test_without_a_failover_the_retry_stays_on_the_primary(self):
        transport = _Transport({PRIMARY: [_server_504(), _server_504()]})
        with transport.patch():
            _, _, sleep = await _invoke(_gemini(failover_model=None), "PM plain retry")
        assert {model for model, _ in transport.calls} == {PRIMARY}
        sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_quick_flex_queue_timeout_fails_over_instead_of_giving_up(self):
        """A quick flex queue timeout is never re-queued at flex, but a standard-tier
        failover attempt is not a re-queue."""
        # Both tiers time out, as the outer hard timeout would leave it.
        transport = _Transport({PRIMARY: [TimeoutError("flex queue")] * 2})
        with (
            transport.patch(),
            patch("src.agents.runtime.provider_flex_active", return_value=True),
            patch(
                "src.agents.runtime.get_runtime_config",
                return_value=type(
                    "RC",
                    (),
                    {
                        "quiet_mode": True,
                        "quick_mode_active": True,
                        "llm_call_hard_timeout_seconds": 60.0,
                    },
                )(),
            ),
            patch(
                "src.agents.runtime.quick_mode_hard_timeout_seconds", return_value=60.0
            ),
        ):
            _, attempts, _ = await _invoke(_gemini(), "PM quick flex timeout")
        assert [model for model, _, _ in attempts] == [PRIMARY, FAILOVER]
