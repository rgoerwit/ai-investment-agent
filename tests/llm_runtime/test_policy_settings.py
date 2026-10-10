"""The last three LLM policy values are configured or derived, never constants.

Before October 2026 (round 2, Step 7) three values were module constants in
``src/llm_runtime/timeouts.py`` that no override could reach:

- ``DIRECT_CALLER_TRANSPORT_ATTEMPTS = 3`` -- every OpenAI-family or Anthropic
  client whose callers invoke it directly made three SDK attempts, one more than
  the shipped ``LLM_ATTEMPTS=2`` that Google direct callers already got.
- ``OPENAI_REQUEST_CAP_SECONDS = 120`` -- now the ``OPENAI_REQUEST_CAP_SECONDS``
  setting, read at the call.
- ``APAC_CLIENT_TIMEOUT_SECONDS = 240`` -- disagreed with the 300 s compatible
  ceiling the same APAC request got inside an invocation scope. Both are now the
  APAC node total (``APAC_NODE_TOTAL_TIMEOUT_SECONDS``), its enclosing deadline.

Kept values and the intended change are tested separately. Attempt counts are
measured below the SDK: each client's HTTP transport is replaced with an
``httpx2.MockTransport`` that answers 500 and counts what was sent, so the
vendor SDK's own retry loop runs for real. No network; dummy keys.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import anthropic
import httpx2
import openai
import pytest
from langchain_core.messages import HumanMessage

import src.config as config_module
import src.llms as llms_mod
from src.config import Settings
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.adapters import transports
from src.llm_runtime.adapters.anthropic import AnthropicAdapter
from src.llm_runtime.adapters.base import SeatModelRequest
from src.llm_runtime.bindings import resolve_binding_plan
from src.llm_runtime.construction import build_model_for_seat
from src.llm_runtime.seats import SEATS, SeatId
from src.llm_runtime.timeouts import (
    TimeoutClass,
    apac_client_timeout_seconds,
    construction_transport_attempts,
    direct_caller_transport_attempts,
    invocation_scope,
    openai_request_cap_seconds,
    policy_for_seat,
)
from src.runtime_config import RuntimeConfig, use_runtime_config

_MESSAGES = [HumanMessage(content="hi")]


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "llm_base_provider": "google",
        "llm_review_provider": "openai",
        "llm_regional_provider": "deepseek",
        "llm_writer_provider": "anthropic",
        "llm_operational_provider": "openai",
        "llm_judge_provider": "google",
        "google_api_key": "g",
        "openai_api_key": "o",
        "claude_api_key": "a",
        "deepseek_api_key": "d",
        "anthropic_llm_prose_model": "claude-opus-4-8",
        "openai_service_tier": "auto",
        "google_service_tier": "standard",
        "llm_apac_mode": "required",
        "llm_require_review_independence": False,
        "llm_review_independence_waiver_reason": "policy settings tests",
        "llm_require_regional_independence": False,
        "llm_regional_independence_waiver_reason": "policy settings tests",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def pin(monkeypatch):
    """Make ``settings`` the live configuration the dispatch-time policy reads."""

    def apply(settings: Settings) -> Settings:
        for target in {
            id(t): t for t in (config_module.config, llms_mod.config)
        }.values():
            for name in (
                "llm_attempts",
                "openai_request_cap_seconds",
                "apac_node_total_timeout_seconds",
                "openai_compatible_client_timeout_seconds",
                "llm_call_hard_timeout_seconds",
                "llm_min_useful_attempt_seconds",
                "llm_dispatch_margin_seconds",
                "openai_service_tier",
            ):
                monkeypatch.setattr(
                    target, name, getattr(settings, name), raising=False
                )
        return settings

    return apply


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(llm_timeouts, "now", fake)
    return fake


class Wire:
    """An HTTP transport below the vendor SDK: counts sends, answers 500."""

    def __init__(self) -> None:
        self.sends: list[dict[str, Any]] = []

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.sends.append(
            {"path": request.url.path, "timeout": request.extensions.get("timeout")}
        )
        return httpx2.Response(500, json={"error": {"message": "boom"}})

    def attach(self, llm: Any) -> Any:
        """Swap every SDK client's HTTP transport and its retry sleep."""
        llm.rate_limiter = None
        clients = []
        for attr in ("root_async_client", "root_client", "_async_client", "_client"):
            client = getattr(llm, attr, None)
            if client is not None and hasattr(client, "_sleep_for_retry"):
                clients.append(client)
        assert clients, "no SDK client found to wire"
        for client in clients:
            async_ = "Async" in type(client).__name__
            transport = httpx2.MockTransport(self._handle)
            client._client = (
                httpx2.AsyncClient(transport=transport)
                if async_
                else httpx2.Client(transport=transport)
            )
            if async_:

                async def no_sleep(*_a: Any, **_k: Any) -> None:
                    return None

                client._sleep_for_retry = no_sleep
            else:
                client._sleep_for_retry = lambda *_a, **_k: None
        return llm

    def read_timeout(self, index: int = 0) -> float:
        return float(self.sends[index]["timeout"]["read"])


async def _send(llm: Any) -> None:
    """One async generation; the 500 surfaces once the SDK's attempts are spent."""
    with pytest.raises((openai.InternalServerError, anthropic.InternalServerError)):
        await llm.ainvoke(_MESSAGES)


def _build(seat_id: SeatId, settings: Settings) -> Any:
    with use_runtime_config(RuntimeConfig.from_config(settings)):
        return build_model_for_seat(
            seat_id, settings=settings, plan=resolve_binding_plan(settings)
        )


def _direct_anthropic(settings: Settings) -> Any:
    """A direct-caller seat on the Anthropic adapter (no seat ships one today)."""
    plan = resolve_binding_plan(settings)
    seat = replace(SEATS[SeatId.ARTICLE_WRITER], timeout_class=TimeoutClass.OPERATIONAL)
    request = SeatModelRequest(
        binding=plan.for_seat(SeatId.ARTICLE_WRITER, quick_mode=False),
        seat=seat,
        quick_mode=False,
        settings=settings,
    )
    with use_runtime_config(RuntimeConfig.from_config(settings)):
        return AnthropicAdapter().build(request)


# ------------------------------------------------------------- kept values


class TestKeptValues:
    def test_shipped_defaults_keep_120_and_240(self):
        fields = Settings.model_fields
        assert fields["openai_request_cap_seconds"].default == 120.0
        assert fields["apac_node_total_timeout_seconds"].default == 240.0
        settings = _settings()
        assert openai_request_cap_seconds(settings) == 120.0
        assert apac_client_timeout_seconds(settings) == 240.0

    def test_default_openai_client_and_in_scope_ceiling_stay_120(self, pin):
        settings = pin(_settings())
        llm = _build(SeatId.MACRO_CONTEXT, settings)
        assert llm.request_timeout == pytest.approx(120.0)
        policy = policy_for_seat(SEATS[SeatId.CONSULTANT], provider="openai")
        assert policy.request_cap == pytest.approx(120.0)

    def test_default_apac_client_and_in_scope_ceiling_stay_240(self, pin):
        settings = pin(_settings())
        llm = _build(SeatId.APAC, settings)
        assert llm.request_timeout == pytest.approx(240.0)
        policy = policy_for_seat(SEATS[SeatId.APAC], provider="deepseek")
        # Was the 300 s compatible ceiling: the mismatch is gone.
        assert policy.request_cap == pytest.approx(240.0)

    @pytest.mark.asyncio
    async def test_health_check_explicit_single_attempt_still_wins(self, pin):
        settings = pin(_settings(llm_attempts=5))
        wire = Wire()
        llm = wire.attach(_build(SeatId.HEALTH_CHECK, settings))
        await _send(llm)
        assert len(wire.sends) == 1
        assert llm.max_retries == 0

    def test_runtime_owned_seats_keep_one_attempt(self):
        settings = _settings(llm_attempts=5)
        with use_runtime_config(RuntimeConfig.from_config(settings)):
            for vendor in ("openai", "anthropic", "openai_compatible"):
                assert (
                    construction_transport_attempts(
                        SEATS[SeatId.CONSULTANT], vendor=vendor, cfg=settings
                    )
                    == 1
                )


# ----------------------------------------------------- intended change: attempts


class TestDirectCallerAttemptsFollowLlmAttempts:
    @pytest.mark.parametrize(("configured", "expected"), [(2, 2), (4, 4), (0, 1)])
    def test_the_derivation(self, configured, expected):
        settings = _settings(llm_attempts=configured)
        with use_runtime_config(RuntimeConfig.from_config(settings)):
            assert direct_caller_transport_attempts(settings) == expected
            for vendor in ("openai", "anthropic", "openai_compatible", "google"):
                assert (
                    construction_transport_attempts(None, vendor=vendor, cfg=settings)
                    == expected
                )

    @pytest.mark.asyncio
    async def test_canonical_openai_direct_caller_sends_two_requests(self, pin):
        settings = pin(_settings(llm_attempts=2))
        wire = Wire()
        llm = wire.attach(_build(SeatId.MACRO_CONTEXT, settings))
        await _send(llm)
        # Two total attempts at LLM_ATTEMPTS=2 (was three).
        assert len(wire.sends) == 2
        assert llm.max_retries == 1

    @pytest.mark.asyncio
    async def test_canonical_anthropic_direct_caller_sends_two_requests(self, pin):
        settings = pin(_settings(llm_attempts=2))
        wire = Wire()
        llm = wire.attach(_direct_anthropic(settings))
        await _send(llm)
        assert len(wire.sends) == 2
        assert llm.max_retries == 1

    @pytest.mark.asyncio
    async def test_legacy_openai_direct_caller_sends_two_requests(self, pin):
        settings = pin(_settings(llm_attempts=2))
        with use_runtime_config(RuntimeConfig.from_config(settings)):
            llm = llms_mod._build_openai_chat(
                "gpt-5.4",
                api_key="o",
                callbacks=None,
                max_completion_tokens=1024,
                service_tier_label="openai_sdk_timeout:test",
                unthrottled_kind="test",
                effort_preference=("medium",),
                settings=settings,
            )
        wire = Wire()
        wire.attach(llm)
        await _send(llm)
        assert len(wire.sends) == 2
        assert llm.max_retries == 1

    @pytest.mark.asyncio
    async def test_raising_llm_attempts_raises_direct_attempts(self, pin):
        settings = pin(_settings(llm_attempts=4))
        wire = Wire()
        llm = wire.attach(_build(SeatId.MACRO_CONTEXT, settings))
        await _send(llm)
        assert len(wire.sends) == 4


# ------------------------------------------------------------------ overrides


class TestOpenAIRequestCapOverride:
    def test_non_positive_is_rejected(self):
        from pydantic import ValidationError

        for bad in (0, -1):
            with pytest.raises(ValidationError):
                Settings(_env_file=None, openai_request_cap_seconds=bad)

    def test_read_at_the_call_not_at_import(self, pin):
        pin(_settings(openai_request_cap_seconds=45.0))
        assert openai_request_cap_seconds() == 45.0
        pin(_settings(openai_request_cap_seconds=75.0))
        assert openai_request_cap_seconds() == 75.0

    @pytest.mark.asyncio
    async def test_override_reaches_responses_request_options(self, pin, clock):
        settings = pin(_settings(openai_request_cap_seconds=45.0, llm_attempts=1))
        wire = Wire()
        llm = wire.attach(_build(SeatId.CONSULTANT, settings))
        assert llm.request_timeout == pytest.approx(45.0)
        policy = policy_for_seat(SEATS[SeatId.CONSULTANT], provider="openai")
        with invocation_scope(policy, clock() + 500.0):
            await _send(llm)
        assert wire.sends[0]["path"].endswith("/responses")
        assert wire.read_timeout() == pytest.approx(45.0)

    @pytest.mark.asyncio
    async def test_override_reaches_chat_completions_request_options(self, pin, clock):
        pin(_settings(openai_request_cap_seconds=45.0))
        wire = Wire()
        llm = wire.attach(
            transports.new_openai_chat(
                model="gpt-5.4",
                api_key="k",
                max_retries=0,
                use_responses_api=False,
                dispatch_provider="openai",
                budget_timeout_class=TimeoutClass.CROSS_CHECK.value,
            )
        )
        policy = policy_for_seat(SEATS[SeatId.CONSULTANT], provider="openai")
        with invocation_scope(policy, clock() + 500.0):
            await _send(llm)
        assert wire.sends[0]["path"].endswith("/chat/completions")
        assert wire.read_timeout() == pytest.approx(45.0)

    def test_override_reaches_the_legacy_factories(self, pin, monkeypatch):
        settings = pin(_settings(openai_request_cap_seconds=45.0))
        monkeypatch.setattr(llms_mod.config, "enable_consultant", True)
        monkeypatch.setattr(
            type(llms_mod.config), "get_openai_api_key", lambda self: "o"
        )
        monkeypatch.setattr(
            type(llms_mod.config), "get_openai_api_base", lambda self: None
        )
        consultant = llms_mod.create_consultant_llm(model="gpt-5.4")
        auditor = llms_mod.create_auditor_llm(settings=settings)
        assert consultant.request_timeout == pytest.approx(45.0)
        assert auditor is not None
        assert auditor.request_timeout == pytest.approx(45.0)
        # An explicit caller timeout still wins.
        explicit = llms_mod.create_consultant_llm(model="gpt-5.4", timeout=30.0)
        assert explicit.request_timeout == pytest.approx(30.0)


class TestApacNodeTotalMovesBothLimits:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("total", [100.0, 400.0])
    async def test_canonical_client_and_in_scope_ceiling_follow(
        self, pin, clock, total
    ):
        settings = pin(_settings(apac_node_total_timeout_seconds=total))
        wire = Wire()
        llm = wire.attach(_build(SeatId.APAC, settings))
        assert llm.request_timeout == pytest.approx(total)
        policy = policy_for_seat(SEATS[SeatId.APAC], provider="deepseek")
        assert policy.request_cap == pytest.approx(total)
        with invocation_scope(policy, clock() + 1_000.0):
            await _send(llm)
        assert wire.read_timeout() == pytest.approx(total)

    @pytest.mark.parametrize("total", [100.0, 400.0])
    def test_legacy_factory_follows(self, pin, total):
        settings = pin(
            Settings(
                _env_file=None,
                google_api_key="g",
                enable_apac_specialist=True,
                apac_specialist_api_key="d",
                apac_node_total_timeout_seconds=total,
            )
        )
        llm = llms_mod.create_apac_specialist_llm(settings=settings)
        assert llm is not None
        assert llm.request_timeout == pytest.approx(total)
        policy = policy_for_seat(SEATS[SeatId.APAC], provider="openai_compatible")
        assert policy.request_cap == pytest.approx(total)

    def test_an_explicit_seat_client_timeout_still_wins(self, pin):
        from src.llm_runtime.seats import SeatExecutionPolicy

        settings = pin(_settings())
        with use_runtime_config(RuntimeConfig.from_config(settings)):
            llm = build_model_for_seat(
                SeatId.APAC,
                settings=settings,
                plan=resolve_binding_plan(settings),
                execution_policy=SeatExecutionPolicy(client_timeout_seconds=55.0),
            )
        assert llm.request_timeout == pytest.approx(55.0)
