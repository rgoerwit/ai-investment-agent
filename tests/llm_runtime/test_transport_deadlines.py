"""Every OpenAI-family and Anthropic client is sized at dispatch, to its own vendor.

Before October 2026 (round 2), only the flex-tier OpenAI client and Gemini went
through the per-dispatch check: the canonical OpenAI adapter without a failover,
every OpenAI-compatible client, the Consultant's alternate transport, the legacy
review-plane factories, the APAC factory and both Anthropic writers were plain
SDK classes that sent their construction-time timeout inside an application
invocation. The one deadline-aware OpenAI class also hardcoded
``provider="openai"``, so a compatible vendor was sized against the 120 s OpenAI
request cap instead of its 300 s ceiling, and its failover copy sent
``service_tier="auto"`` to a vendor that sells no tiers.

These tests assert what each vendor library's *request layer* receives, through
real construction: ``openai``'s and ``anthropic``'s ``request`` and
google-genai's ``_async_request`` are patched, nothing else. A fake clock; no
network.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import anthropic
import openai
import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

import src.config as config_module
import src.llms as llms_mod
from src.config import Settings
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.adapters import transports
from src.llm_runtime.bindings import resolve_binding_plan
from src.llm_runtime.construction import build_model_for_seat
from src.llm_runtime.dispatch import StreamingNotQualifiedError
from src.llm_runtime.factory import SeatModelFactory
from src.llm_runtime.failover import failover_attempt
from src.llm_runtime.provider_policy import PROVIDER_GROUP_ALLOWLIST
from src.llm_runtime.rate_limits import reset_fallback_limiters_for_tests
from src.llm_runtime.seats import SEATS, SeatId
from src.llm_runtime.timeouts import (
    BudgetExhausted,
    TimeoutClass,
    invocation_scope,
    policy_for_seat,
    resolve_budget_policy,
)
from src.runtime_config import RuntimeConfig, use_runtime_config
from src.service_tiers import (
    _reset_flex_capability_cache_for_tests,
    _reset_flex_health_for_tests,
    note_flex_fallback,
)

MARGIN = 2.0
MIN_USEFUL = 5.0
COMPATIBLE_CAP = 300.0
OPENAI_CAP = llm_timeouts.OPENAI_REQUEST_CAP_SECONDS


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(llm_timeouts, "now", fake)
    return fake


@pytest.fixture(autouse=True)
def budget_config(monkeypatch):
    """Pin every input the policy reads, on every live config binding."""
    values = {
        "quick_llm_call_hard_timeout_seconds": 60.0,
        "apex_quick_llm_call_hard_timeout_seconds": 180.0,
        "cross_check_quick_llm_call_hard_timeout_seconds": 180.0,
        "llm_call_hard_timeout_seconds": 600.0,
        "llm_min_useful_attempt_seconds": MIN_USEFUL,
        "llm_dispatch_margin_seconds": MARGIN,
        "flex_quick_attempt_fraction": 0.4,
        "api_timeout": 300,
        "quick_llm_api_timeout_seconds": 120,
        "openai_compatible_client_timeout_seconds": int(COMPATIBLE_CAP),
        "flex_llm_timeout_seconds": 900,
        "flex_degrade_threshold": 2,
        "llm_attempts": 2,
    }
    targets = {id(config_module.config): config_module.config}
    targets[id(llms_mod.config)] = llms_mod.config
    for target in targets.values():
        for key, value in values.items():
            monkeypatch.setattr(target, key, value, raising=False)
    _reset_flex_capability_cache_for_tests()
    _reset_flex_health_for_tests()
    reset_fallback_limiters_for_tests()
    yield
    _reset_flex_capability_cache_for_tests()
    _reset_flex_health_for_tests()
    reset_fallback_limiters_for_tests()


@contextmanager
def _mode(quick: bool):
    runtime = RuntimeConfig.from_config(config_module.config).with_overrides(
        quick_mode_active=quick
    )
    with use_runtime_config(runtime):
        yield


class Captured(RuntimeError):
    """Raised by a fake request layer once it has recorded the options."""


class _CapacityError(RuntimeError):
    status_code = 503


class RequestLayer:
    """Patches the vendor SDKs' request layer and records every request."""

    def __init__(self, script: list[BaseException | None] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.script = list(script or [])

    def _next(self) -> None:
        error = self.script.pop(0) if self.script else Captured("captured")
        if error is not None:
            raise error

    def _record(self, vendor: str, client: Any, options: Any) -> None:
        body = options.json_data if isinstance(options.json_data, dict) else {}
        self.calls.append(
            {
                "vendor": vendor,
                "timeout": options.timeout,
                "client_max_retries": client.max_retries,
                "client_timeout": client.timeout,
                "service_tier": body.get("service_tier", "<absent>"),
                "stream": body.get("stream"),
                "model": body.get("model"),
                "base_url": str(client.base_url),
                "url": options.url,
            }
        )

    @contextmanager
    def patched(self):
        layer = self

        async def openai_async(client, cast_to, options, *args, **kwargs):
            layer._record("openai", client, options)
            await asyncio.sleep(0)
            layer._next()

        def openai_sync(client, cast_to, options, *args, **kwargs):
            layer._record("openai", client, options)
            layer._next()

        async def anthropic_async(client, cast_to, options, *args, **kwargs):
            layer._record("anthropic", client, options)
            await asyncio.sleep(0)
            layer._next()

        def anthropic_sync(client, cast_to, options, *args, **kwargs):
            layer._record("anthropic", client, options)
            layer._next()

        async def google_async(client, http_request, http_options=None, stream=False):
            from google.genai import _api_client

            layer.calls.append(
                {
                    "vendor": "google",
                    "timeout": http_request.timeout,
                    "header": http_request.headers.get("X-Server-Timeout"),
                }
            )
            body = {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "ok"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ]
            }
            return _api_client.HttpResponse({}, [json.dumps(body)])

        from google.genai import _api_client

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(openai.AsyncOpenAI, "request", openai_async)
            )
            stack.enter_context(patch.object(openai.OpenAI, "request", openai_sync))
            stack.enter_context(
                patch.object(anthropic.AsyncAnthropic, "request", anthropic_async)
            )
            stack.enter_context(
                patch.object(anthropic.Anthropic, "request", anthropic_sync)
            )
            stack.enter_context(
                patch.object(_api_client.BaseApiClient, "_async_request", google_async)
            )
            yield self


_MESSAGES = [HumanMessage(content="hi")]


async def _send(llm) -> None:
    """One async generation; a fake request layer ends it with ``Captured``."""
    try:
        await llm._agenerate(_MESSAGES)
    except Captured:
        pass


def _send_sync(llm) -> None:
    try:
        llm._generate(_MESSAGES)
    except Captured:
        pass


def _settings(**overrides) -> Settings:
    values: dict[str, Any] = {
        "llm_base_provider": "google",
        "llm_review_provider": "openai",
        "llm_regional_provider": "deepseek",
        "llm_writer_provider": "anthropic",
        "llm_operational_provider": "google",
        "llm_judge_provider": "google",
        "google_api_key": "g",
        "openai_api_key": "o",
        "claude_api_key": "a",
        "deepseek_api_key": "d",
        "zai_api_key": "z",
        "moonshot_api_key": "m",
        "xai_api_key": "x",
        "openai_service_tier": "auto",
        "google_service_tier": "standard",
        "llm_consultant_mode": "required",
        "llm_auditor_mode": "required",
        "llm_editor_mode": "required",
        "llm_apac_mode": "required",
        # The per-seat matrix binds every group to every allowed vendor, so a
        # base and a review plane may share one; independence is not under test.
        "llm_require_review_independence": False,
        "llm_review_independence_waiver_reason": "transport deadline tests",
        "llm_require_regional_independence": False,
        "llm_regional_independence_waiver_reason": "transport deadline tests",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _build(seat_id: SeatId, settings: Settings, *, quick: bool = False):
    return build_model_for_seat(
        seat_id,
        settings=settings,
        plan=resolve_binding_plan(settings),
        quick_mode=quick,
    )


def _scope_for(seat_id: SeatId, provider: str, seconds: float, clock: FakeClock):
    """The scope the runtime loop opens for this seat on this vendor."""
    return invocation_scope(
        policy_for_seat(SEATS[seat_id], provider=provider), clock() + seconds
    )


# ------------------------------------------------------------- construction


class TestExplicitProvider:
    def test_openai_transport_without_a_provider_is_refused(self):
        with pytest.raises(ValidationError, match="dispatch_provider"):
            transports.new_openai_chat(model="gpt-5.4", api_key="k")

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_provider_is_refused(self, value):
        with pytest.raises(ValidationError, match="dispatch_provider is required"):
            transports.new_openai_chat(
                model="gpt-5.4", api_key="k", dispatch_provider=value
            )

    def test_anthropic_transport_without_a_provider_is_refused(self):
        with pytest.raises(ValidationError, match="dispatch_provider"):
            transports.new_anthropic_chat(
                model="claude-opus-4-8", anthropic_api_key="k"
            )

    def test_a_compatible_vendor_cannot_be_built_with_a_service_tier(self):
        with pytest.raises(ValidationError, match="must not receive one"):
            transports.new_openai_chat(
                model="kimi-k3",
                api_key="k",
                base_url="https://api.moonshot.ai/v1",
                dispatch_provider="moonshot",
                service_tier="auto",
            )

    def test_the_plan_dispatch_refuses_a_missing_provider(self):
        from src.llm_runtime.dispatch import _plan_dispatch

        model = SimpleNamespace(_effective_tier=lambda kwargs: None)
        with pytest.raises(ValueError, match="receiving provider"):
            _plan_dispatch(
                model,
                {},
                provider="",
                standard_tier=None,
                model_name="m",
                phase="initial",
                per_request_attempts=False,
            )

    def test_a_patched_vendor_class_is_refused_not_cached(self, monkeypatch):
        import langchain_openai.chat_models.base as base_module

        real = base_module.ChatOpenAI
        monkeypatch.setattr(transports, "_openai_cls", None)
        monkeypatch.setattr(base_module, "ChatOpenAI", MagicMock())
        with pytest.raises(TypeError, match="not a chat model class"):
            transports.deadline_aware_openai_cls()
        assert transports._openai_cls is None
        monkeypatch.setattr(base_module, "ChatOpenAI", real)
        cls = transports.deadline_aware_openai_cls()
        assert issubclass(cls, real)

    def test_patching_the_package_attribute_does_not_reach_the_base(self, monkeypatch):
        import langchain_openai

        monkeypatch.setattr(transports, "_openai_cls", None)
        monkeypatch.setattr(langchain_openai, "ChatOpenAI", MagicMock())
        cls = transports.deadline_aware_openai_cls()
        from langchain_openai.chat_models.base import ChatOpenAI

        assert issubclass(cls, ChatOpenAI)


# -------------------------------------------------- request-layer sizing


class TestCanonicalOpenAI:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("quick", [False, True])
    async def test_responses_request_is_sized_from_the_deadline(self, clock, quick):
        settings = _settings()
        with _mode(quick):
            llm = _build(SeatId.CONSULTANT, settings, quick=quick)
            assert isinstance(llm, transports.deadline_aware_openai_cls())
            assert llm.dispatch_provider == "openai"
            assert llm.use_responses_api is True
            with (
                RequestLayer().patched() as layer,
                _scope_for(SeatId.CONSULTANT, "openai", 33.0, clock),
            ):
                await _send(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(31.0)
        assert call["url"] == "/responses"
        # The Consultant's retries belong to the runtime loop: one request.
        assert call["client_max_retries"] == 0
        assert call["service_tier"] == "<absent>"

    def test_sync_twin_is_sized_the_same_way(self, clock):
        llm = _build(SeatId.CONSULTANT, _settings())
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.CONSULTANT, "openai", 33.0, clock),
        ):
            _send_sync(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(31.0)

    @pytest.mark.asyncio
    async def test_chat_completions_request_is_sized_from_the_deadline(self, clock):
        llm = transports.new_openai_chat(
            model="gpt-5.4",
            api_key="k",
            timeout=OPENAI_CAP,
            max_retries=0,
            use_responses_api=False,
            dispatch_provider="openai",
        )
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.CONSULTANT, "openai", 33.0, clock),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["url"] == "/chat/completions"
        assert call["timeout"] == pytest.approx(31.0)

    @pytest.mark.asyncio
    async def test_below_the_minimum_useful_attempt_nothing_is_sent(self, clock):
        llm = _build(SeatId.CONSULTANT, _settings())
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.CONSULTANT, "openai", MARGIN + 3.0, clock) as scope,
            pytest.raises(BudgetExhausted),
        ):
            await llm._agenerate(_MESSAGES)
        assert layer.calls == []
        assert scope.dispatches[0]["outcome"] == "budget_exhausted"

    @pytest.mark.asyncio
    async def test_direct_caller_keeps_the_constructed_ceiling(self):
        llm = _build(SeatId.CONSULTANT, _settings())
        with RequestLayer().patched() as layer:
            await _send(llm)
        [call] = layer.calls
        # No invocation scope (Step 2 changes this): no per-request option is
        # set, so the client's constructed request cap applies.
        assert not isinstance(call["timeout"], int | float)
        assert call["client_timeout"] == pytest.approx(OPENAI_CAP)

    @pytest.mark.asyncio
    async def test_an_explicit_seat_client_timeout_still_wins(self, clock):
        # The health probe pins a 10 s client timeout in its execution policy.
        settings = _settings(llm_operational_provider="openai")
        llm = _build(SeatId.HEALTH_CHECK, settings)
        assert isinstance(llm, transports.deadline_aware_openai_cls())
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.HEALTH_CHECK, "openai", 33.0, clock),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(10.0)
        assert call["client_max_retries"] == 0


class TestCompatibleVendors:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("quick", [False, True])
    async def test_canonical_compatible_vendor_gets_its_own_ceiling(self, clock, quick):
        settings = _settings(llm_review_provider="moonshot")
        with _mode(quick):
            llm = _build(SeatId.CONSULTANT, settings, quick=quick)
            assert isinstance(llm, transports.deadline_aware_openai_cls())
            assert llm.dispatch_provider == "moonshot"
            with (
                RequestLayer().patched() as layer,
                _scope_for(SeatId.CONSULTANT, "moonshot", 200.0, clock),
            ):
                await _send(llm)
        [call] = layer.calls
        # 198 s, not the 120 s OpenAI request cap.
        assert call["timeout"] == pytest.approx(198.0)
        assert call["url"] == "/chat/completions"
        assert call["service_tier"] == "<absent>"
        assert "moonshot" in call["base_url"]

    @pytest.mark.asyncio
    async def test_canonical_apac_is_sized_at_the_compatible_ceiling(self, clock):
        settings = _settings()
        llm = _build(SeatId.APAC, settings)
        assert llm.dispatch_provider == "deepseek"
        assert llm.budget_timeout_class == TimeoutClass.REGIONAL.value
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.APAC, "deepseek", 200.0, clock),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(198.0)

    @pytest.mark.asyncio
    async def test_legacy_compatible_openai_base_gets_the_compatible_ceiling(
        self, clock, monkeypatch
    ):
        monkeypatch.setattr(
            type(llms_mod.config),
            "get_openai_api_base",
            lambda self: "https://api.moonshot.ai/v1",
        )
        monkeypatch.setattr(llms_mod.config, "enable_consultant", True)
        monkeypatch.setattr(
            type(llms_mod.config), "get_openai_api_key", lambda self: "o"
        )
        monkeypatch.setattr(llms_mod.config, "openai_service_tier", "flex")
        llm = llms_mod.create_consultant_llm(model="kimi-k3")
        assert isinstance(llm, transports.deadline_aware_openai_cls())
        assert llm.dispatch_provider == transports.OPENAI_COMPATIBLE_PROVIDER
        # A compatible host never takes the OpenAI flex tier.
        assert llm.service_tier is None
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.CONSULTANT, "moonshot", 200.0, clock),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(198.0)
        assert call["service_tier"] == "<absent>"

    @pytest.mark.asyncio
    async def test_legacy_apac_factory_is_sized_at_the_compatible_ceiling(
        self, clock, monkeypatch
    ):
        monkeypatch.setattr(llms_mod.config, "enable_apac_specialist", True)
        monkeypatch.setattr(
            type(llms_mod.config), "get_apac_specialist_api_key", lambda self: "d"
        )
        monkeypatch.setattr(
            llms_mod.config, "apac_specialist_base_url", "https://api.deepseek.com"
        )
        llm = llms_mod.create_apac_specialist_llm()
        assert isinstance(llm, transports.deadline_aware_openai_cls())
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.APAC, "deepseek", 200.0, clock),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(198.0)
        assert call["service_tier"] == "<absent>"

    @pytest.mark.asyncio
    async def test_a_compatible_failover_copy_sends_no_service_tier(self, clock):
        llm = transports.new_openai_chat(
            model="kimi-k3",
            api_key="k",
            base_url="https://api.moonshot.ai/v1",
            dispatch_provider="moonshot",
            failover_model="kimi-k2.5",
        )
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.CONSULTANT, "moonshot", 200.0, clock),
            failover_attempt(),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["model"] == "kimi-k2.5"
        assert call["service_tier"] == "<absent>"
        assert call["timeout"] == pytest.approx(198.0)

    @pytest.mark.parametrize("provider", ["openai", "moonshot"])
    def test_flex_health_is_keyed_to_the_dispatch_provider(self, monkeypatch, provider):
        asked: list[str] = []
        monkeypatch.setattr(
            transports, "flex_degraded", lambda key: asked.append(key) or False
        )
        llm = transports.new_openai_chat(
            model="gpt-5.4", api_key="k", dispatch_provider=provider
        )
        llm._flex_ineligible()
        assert asked == [provider]
        assert llm._failover_tier() == ("auto" if provider == "openai" else None)


class TestAlternateTransport:
    """The Consultant's alternate OpenAI transport (``build_transport_fallback``)."""

    @pytest.fixture(autouse=True)
    def _openai_key(self, monkeypatch):
        monkeypatch.setattr(
            type(config_module.config), "get_openai_api_key", lambda self: "o"
        )

    @pytest.mark.asyncio
    async def test_default_endpoint_is_sized_at_the_openai_cap(self, clock):
        llm = SeatModelFactory().build_openai_transport_fallback("gpt-5.4")
        assert isinstance(llm, transports.deadline_aware_openai_cls())
        assert llm.dispatch_provider == "openai"
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.CONSULTANT, "openai", 200.0, clock),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(OPENAI_CAP)
        assert call["url"] == "/responses"

    @pytest.mark.asyncio
    async def test_custom_base_is_a_compatible_endpoint(self, clock, monkeypatch):
        monkeypatch.setattr(
            type(config_module.config),
            "get_openai_api_base",
            lambda self: "https://api.moonshot.ai/v1",
        )
        llm = SeatModelFactory().build_openai_transport_fallback("kimi-k3")
        assert llm.dispatch_provider == transports.OPENAI_COMPATIBLE_PROVIDER
        # Built with the compatible ceiling, not the 120 s OpenAI cap.
        assert llm.request_timeout == pytest.approx(COMPATIBLE_CAP)
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.CONSULTANT, "moonshot", 200.0, clock),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(198.0)
        assert call["url"] == "/chat/completions"
        assert call["service_tier"] == "<absent>"


class TestNativeTiers:
    def _flex_consultant(self):
        settings = _settings(openai_service_tier="flex")
        llm = _build(SeatId.CONSULTANT, settings)
        assert llm.service_tier == "flex"
        return llm

    @pytest.mark.asyncio
    @pytest.mark.parametrize("quick", [False, True])
    async def test_flex_then_standard_reissue_each_sized(self, clock, quick):
        with _mode(quick):
            llm = self._flex_consultant()
            with (
                RequestLayer(
                    script=[_CapacityError("503 flex capacity")]
                ).patched() as layer,
                _scope_for(SeatId.CONSULTANT, "openai", 33.0, clock),
            ):
                await _send(llm)
        first, second = layer.calls
        assert first["service_tier"] == "flex"
        assert second["service_tier"] == "auto"
        assert first["timeout"] == pytest.approx(31.0)
        assert second["timeout"] == pytest.approx(31.0)

    @pytest.mark.asyncio
    async def test_degraded_flex_sends_standard_sized_from_remaining(self, clock):
        llm = self._flex_consultant()
        for _ in range(2):
            note_flex_fallback("openai", reason="latency", model="gpt-5.4")
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.CONSULTANT, "openai", 33.0, clock),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["service_tier"] == "auto"
        assert call["timeout"] == pytest.approx(31.0)

    @pytest.mark.asyncio
    async def test_failover_copy_is_sized_and_sent_at_auto(self, clock):
        settings = _settings(
            openai_llm_reasoning_model="gpt-6.1-sol",
            llm_model_failovers={"gpt-6.1-sol": "gpt-5.4"},
        )
        llm = _build(SeatId.CONSULTANT, settings)
        assert llm.failover_model == "gpt-5.4"
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.CONSULTANT, "openai", 33.0, clock) as scope,
            failover_attempt(),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["model"] == "gpt-5.4"
        assert call["service_tier"] == "auto"
        assert call["timeout"] == pytest.approx(31.0)
        assert scope.dispatches[0]["phase"] == "failover"

    @pytest.mark.asyncio
    async def test_concurrent_reuse_keeps_each_call_on_its_own_deadline(self, clock):
        llm = _build(SeatId.CONSULTANT, _settings())
        layer = RequestLayer()

        async def call(seconds: float) -> None:
            with _scope_for(SeatId.CONSULTANT, "openai", seconds, clock):
                await _send(llm)

        with layer.patched():
            await asyncio.gather(call(20.0), call(40.0), call(60.0))
        assert sorted(c["timeout"] for c in layer.calls) == pytest.approx(
            [18.0, 38.0, 58.0]
        )


class TestAnthropic:
    def _canonical(self):
        settings = _settings(anthropic_llm_prose_model="claude-opus-4-8")
        llm = _build(SeatId.ARTICLE_WRITER, settings)
        assert isinstance(llm, transports.deadline_aware_anthropic_cls())
        assert llm.dispatch_provider == "anthropic"
        assert llm.budget_timeout_class == TimeoutClass.EDITORIAL.value
        return llm

    def _legacy(self):
        llm = llms_mod.create_writer_llm(
            model="claude-opus-4-8", api_key_override="a", allow_fallback=False
        )
        assert isinstance(llm, transports.deadline_aware_anthropic_cls())
        assert llm.dispatch_provider == "anthropic"
        assert llm.budget_timeout_class == TimeoutClass.EDITORIAL.value
        return llm

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["canonical", "legacy"])
    async def test_async_request_is_sized_from_the_deadline(self, clock, path):
        llm = self._canonical() if path == "canonical" else self._legacy()
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.ARTICLE_WRITER, "anthropic", 33.0, clock),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["vendor"] == "anthropic"
        assert call["timeout"] == pytest.approx(31.0)
        # Retries stay fixed on the client (the writer is still a direct
        # caller): the seat policy on the canonical path, the legacy
        # constructor's own default on the legacy one (Step 2 aligns them).
        assert call["client_max_retries"] == (2 if path == "canonical" else 3)

    @pytest.mark.parametrize("path", ["canonical", "legacy"])
    def test_sync_request_is_sized_from_the_deadline(self, clock, path):
        llm = self._canonical() if path == "canonical" else self._legacy()
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.ARTICLE_WRITER, "anthropic", 200.0, clock),
        ):
            _send_sync(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(198.0)

    @pytest.mark.asyncio
    async def test_below_the_minimum_useful_attempt_nothing_is_sent(self, clock):
        llm = self._canonical()
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.ARTICLE_WRITER, "anthropic", MARGIN + 3.0, clock),
            pytest.raises(BudgetExhausted),
        ):
            await llm._agenerate(_MESSAGES)
        assert layer.calls == []

    def test_reasoning_controls_survive(self):
        llm = self._legacy()
        assert llm.thinking == {"type": "adaptive"}
        assert llm.effort == "high"
        assert llm.max_tokens == 16384


class TestStreamingIsRefused:
    def _models(self):
        return [
            transports.new_openai_chat(
                model="gpt-5.4", api_key="k", dispatch_provider="openai"
            ),
            transports.new_anthropic_chat(
                model="claude-opus-4-8",
                anthropic_api_key="k",
                dispatch_provider="anthropic",
            ),
            llms_mod._TieredChatGoogleGenerativeAI(
                model="gemini-3.8-flash", api_key="k"
            ),
        ]

    def test_a_streaming_client_cannot_be_built(self):
        # ChatOpenAI puts ``stream: true`` in its default request params, so a
        # streaming=True client would open a stream even through _generate.
        with pytest.raises(ValidationError, match="streaming=True is not qualified"):
            transports.new_openai_chat(
                model="gpt-5.4", api_key="k", dispatch_provider="openai", streaming=True
            )
        with pytest.raises(ValidationError, match="streaming=True is not qualified"):
            transports.new_anthropic_chat(
                model="claude-opus-4-8",
                anthropic_api_key="k",
                dispatch_provider="anthropic",
                streaming=True,
            )

    def test_private_stream_raises(self):
        for llm in self._models():
            with pytest.raises(StreamingNotQualifiedError):
                llm._stream(_MESSAGES)

    @pytest.mark.asyncio
    async def test_private_astream_raises(self):
        for llm in self._models():
            with pytest.raises(StreamingNotQualifiedError):
                async for _chunk in llm._astream(_MESSAGES):
                    pass

    @pytest.mark.asyncio
    @pytest.mark.parametrize("vendor", ["openai", "anthropic"])
    async def test_public_stream_paths_still_pass_the_dispatch_check(
        self, clock, vendor
    ):
        # .stream()/.astream() would route to _stream/_astream on a plain
        # client; here each is a sized, non-streaming call. An explicit
        # stream kwarg is refused before any request is sent.
        if vendor == "openai":
            llm = transports.new_openai_chat(
                model="gpt-5.4", api_key="k", dispatch_provider="openai"
            )
        else:
            llm = transports.new_anthropic_chat(
                model="claude-opus-4-8",
                anthropic_api_key="k",
                dispatch_provider="anthropic",
            )
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.CONSULTANT, vendor, 33.0, clock),
        ):
            with pytest.raises(Captured):
                await llm.ainvoke(_MESSAGES)
            with pytest.raises(StreamingNotQualifiedError):
                await llm.ainvoke(_MESSAGES, stream=True)
            with pytest.raises(Captured):
                async for _chunk in llm.astream(_MESSAGES):
                    pass
            with pytest.raises(Captured):
                for _chunk in llm.stream(_MESSAGES):
                    pass
        assert len(layer.calls) == 3
        for call in layer.calls:
            assert call["timeout"] == pytest.approx(31.0)
            assert not call["stream"]


# ------------------------------------------------------------ seat matrix
#
# Read from the real registries: every seat x every vendor the application
# allows for that seat's binding group. A new seat or a newly allowed vendor
# joins the matrix without editing this file.


def _matrix_cases() -> list[Any]:
    cases = []
    for provider, groups in sorted(PROVIDER_GROUP_ALLOWLIST.items()):
        for seat_id, spec in SEATS.items():
            if spec.binding_group not in groups:
                continue
            for quick in (False, True):
                if quick and spec.disabled_in_quick_mode:
                    continue
                cases.append(
                    pytest.param(
                        provider,
                        seat_id,
                        quick,
                        id=f"{provider}-{seat_id.value}-{'quick' if quick else 'full'}",
                    )
                )
    return cases


def _expected_class(provider: str) -> type:
    if provider == "google":
        return llms_mod._TieredChatGoogleGenerativeAI
    if provider == "anthropic":
        return transports.deadline_aware_anthropic_cls()
    return transports.deadline_aware_openai_cls()


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "seat_id", "quick"), _matrix_cases())
async def test_every_seat_on_every_allowed_vendor_is_sized_at_dispatch(
    clock, provider: str, seat_id: SeatId, quick: bool
):
    from src.llm_runtime.provider_policy import _GROUP_SELECTOR_FIELDS

    spec = SEATS[seat_id]
    settings = _settings(**{_GROUP_SELECTOR_FIELDS[spec.binding_group]: provider})
    with _mode(quick):
        plan = resolve_binding_plan(settings)
        assert plan.for_seat(seat_id, quick_mode=quick).provider == provider
        llm = build_model_for_seat(
            seat_id, settings=settings, plan=plan, quick_mode=quick
        )
        # Every optional seat is "required" in _settings, so none may be skipped.
        assert llm is not None
        assert isinstance(llm, _expected_class(provider))
        if provider != "google":
            assert llm.dispatch_provider == provider
            assert llm.budget_timeout_class == spec.timeout_class.value
        policy = policy_for_seat(spec, provider=provider)
        expected = min(policy.request_cap, 200.0 - MARGIN)
        with (
            RequestLayer().patched() as layer,
            invocation_scope(policy, clock() + 200.0),
        ):
            await _send(llm)
    [call] = layer.calls
    assert call["vendor"] == (
        "openai" if provider not in ("google", "anthropic") else provider
    )
    assert call["timeout"] == pytest.approx(expected)
    if provider not in ("openai", "google", "anthropic"):
        # A compatible vendor never receives a service tier and is never
        # sized against the OpenAI request cap.
        assert call["service_tier"] == "<absent>"
        assert expected > OPENAI_CAP or policy.request_cap_override is not None
