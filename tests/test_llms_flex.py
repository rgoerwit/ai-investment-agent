"""Tests for flex service-tier wiring in src/llms.py.

Covers: Gemini tier resolution and request injection, capacity fallback,
dynamic capability downgrade (no hardcoded model allowlists), the pinned
standard tier for latency-sensitive callers, and OpenAI factory wiring.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI

import src.llms as llms_mod
from src.config import config
from src.service_tiers import (
    _reset_flex_capability_cache_for_tests,
    _reset_floor_log_cache_for_tests,
    is_flex_unsupported,
)


@pytest.fixture(autouse=True)
def _reset_tier_state():
    _reset_floor_log_cache_for_tests()
    _reset_flex_capability_cache_for_tests()
    yield
    _reset_floor_log_cache_for_tests()
    _reset_flex_capability_cache_for_tests()


def _set_cfg(monkeypatch, **attrs):
    """Patch config attributes on every live binding.

    Other suites reload ``src.config``/``src.llms``, which can leave
    ``llms_mod.config`` pointing at a different Settings object than
    ``src.config.config``. Patch both so these tests are order-independent.
    """
    import src.config as config_module

    targets = {
        id(config_module.config): config_module.config,
        id(llms_mod.config): llms_mod.config,
        id(config): config,
    }
    for target in targets.values():
        for key, value in attrs.items():
            monkeypatch.setattr(target, key, value, raising=False)


def _tiered_llm(**overrides):
    kwargs = {
        "model": "gemini-3.5-flash",
        "api_key": "test-key",
        "service_tier": "flex",
        "flex_fallback_to_standard": True,
    }
    kwargs.update(overrides)
    return llms_mod._TieredChatGoogleGenerativeAI(**kwargs)


def _chat_result(text: str = "ok") -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class TestGeminiTierResolution:
    def test_standard_config_means_no_tier(self, monkeypatch):
        _set_cfg(monkeypatch, gemini_service_tier="standard")
        assert llms_mod._resolve_gemini_service_tier("gemini-3.5-flash", None) is None

    def test_flex_config_applies_to_any_gemini_model(self, monkeypatch):
        _set_cfg(monkeypatch, gemini_service_tier="flex")
        assert llms_mod._resolve_gemini_service_tier("gemini-3.5-flash", None) == "flex"
        # No hardcoded allowlist: unknown/new gemini models attempt flex too
        assert (
            llms_mod._resolve_gemini_service_tier("gemini-9-future-model", None)
            == "flex"
        )

    def test_explicit_standard_pin_wins_over_flex_config(self, monkeypatch):
        _set_cfg(monkeypatch, gemini_service_tier="flex")
        assert (
            llms_mod._resolve_gemini_service_tier("gemini-3.5-flash", "standard")
            is None
        )

    def test_learned_unsupported_model_skips_flex(self, monkeypatch):
        _set_cfg(monkeypatch, gemini_service_tier="flex")
        from src.service_tiers import mark_flex_unsupported

        mark_flex_unsupported("gemini-3.5-flash")
        assert llms_mod._resolve_gemini_service_tier("gemini-3.5-flash", None) is None


class TestGeminiRequestInjection:
    def test_prepare_request_injects_flex_tier(self):
        llm = _tiered_llm()
        request = llm._prepare_request([HumanMessage(content="hi")])
        assert request["config"].service_tier == "flex"

    def test_prepare_request_no_tier_when_unset(self):
        llm = _tiered_llm(service_tier=None)
        request = llm._prepare_request([HumanMessage(content="hi")])
        assert request["config"].service_tier is None

    def test_prepare_request_skips_learned_unsupported_model(self):
        from src.service_tiers import mark_flex_unsupported

        llm = _tiered_llm()
        mark_flex_unsupported("gemini-3.5-flash")
        request = llm._prepare_request([HumanMessage(content="hi")])
        assert request["config"].service_tier is None


class TestGeminiFlexFallback:
    @pytest.mark.asyncio
    async def test_capacity_error_falls_back_to_standard(self):
        llm = _tiered_llm()
        calls: list[dict] = []

        async def fake_agenerate(self, *args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise ValueError("503 UNAVAILABLE: no flex capacity")
            return _chat_result()

        with patch.object(ChatGoogleGenerativeAI, "_agenerate", fake_agenerate):
            result = await llm._agenerate([HumanMessage(content="hi")])

        assert len(calls) == 2
        assert calls[1]["service_tier"] == "standard"
        # Fallback call is priced at standard rates by the tracker
        assert result.llm_output["service_tier"] == "standard"
        # Capacity errors are NOT cached — next call still tries flex
        assert not is_flex_unsupported("gemini-3.5-flash")

    @pytest.mark.asyncio
    async def test_capability_error_downgrades_and_caches(self):
        llm = _tiered_llm(flex_fallback_to_standard=False)
        calls: list[dict] = []

        async def fake_agenerate(self, *args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise ValueError("400 INVALID_ARGUMENT: service_tier not supported")
            return _chat_result()

        with patch.object(ChatGoogleGenerativeAI, "_agenerate", fake_agenerate):
            result = await llm._agenerate([HumanMessage(content="hi")])

        # Capability downgrade happens even with capacity fallback disabled
        assert len(calls) == 2
        assert calls[1]["service_tier"] == "standard"
        assert result.llm_output["service_tier"] == "standard"
        assert is_flex_unsupported("gemini-3.5-flash")
        # Subsequent requests skip flex entirely
        request = llm._prepare_request([HumanMessage(content="hi")])
        assert request["config"].service_tier is None

    @pytest.mark.asyncio
    async def test_capacity_error_propagates_when_fallback_disabled(self):
        llm = _tiered_llm(flex_fallback_to_standard=False)

        async def fake_agenerate(self, *args, **kwargs):
            raise ValueError("429 RESOURCE_EXHAUSTED: flex queue full")

        with (
            patch.object(ChatGoogleGenerativeAI, "_agenerate", fake_agenerate),
            pytest.raises(ValueError, match="429"),
        ):
            await llm._agenerate([HumanMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_non_flex_instance_never_falls_back(self):
        llm = _tiered_llm(service_tier=None)

        async def fake_agenerate(self, *args, **kwargs):
            raise ValueError("503 UNAVAILABLE")

        with (
            patch.object(ChatGoogleGenerativeAI, "_agenerate", fake_agenerate),
            pytest.raises(ValueError, match="503"),
        ):
            await llm._agenerate([HumanMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_success_stamps_flex_tier_for_cost_tracking(self):
        llm = _tiered_llm()

        async def fake_agenerate(self, *args, **kwargs):
            return _chat_result()

        with patch.object(ChatGoogleGenerativeAI, "_agenerate", fake_agenerate):
            result = await llm._agenerate([HumanMessage(content="hi")])

        assert result.llm_output["service_tier"] == "flex"
        message = result.generations[0].message
        assert message.response_metadata["service_tier"] == "flex"


class TestCreateGeminiModelWiring:
    def test_flex_config_builds_tiered_instance_with_floored_timeout(self, monkeypatch):
        _set_cfg(monkeypatch, gemini_service_tier="flex")
        _set_cfg(monkeypatch, flex_llm_timeout_seconds=900)
        llm = llms_mod.create_gemini_model("gemini-3.5-flash", 0.3, 60, 2)
        assert isinstance(llm, llms_mod._TieredChatGoogleGenerativeAI)
        assert llm.service_tier == "flex"
        # SDK client timeout floored so queued flex calls aren't killed
        assert llm.timeout >= 900

    def test_standard_config_leaves_timeout_and_tier_alone(self, monkeypatch):
        _set_cfg(monkeypatch, gemini_service_tier="standard")
        llm = llms_mod.create_gemini_model("gemini-3.5-flash", 0.3, 60, 2)
        assert llm.service_tier is None
        assert llm.timeout == 60

    def test_standard_pin_under_flex_config(self, monkeypatch):
        # The LLM-judge inspector path: must not queue on flex
        _set_cfg(monkeypatch, gemini_service_tier="flex")
        llm = llms_mod.create_gemini_model(
            "gemini-3.5-flash", 0.0, 60, 2, service_tier="standard"
        )
        assert llm.service_tier is None
        assert llm.timeout == 60


class TestOpenAIFlexWiring:
    def test_apply_openai_service_tier_flex(self, monkeypatch):
        _set_cfg(monkeypatch, openai_service_tier="flex")
        _set_cfg(monkeypatch, flex_fallback_to_standard=True)
        _set_cfg(monkeypatch, flex_llm_timeout_seconds=900)
        kwargs = {"model": "gpt-5.4", "timeout": 120}
        llms_mod._apply_openai_service_tier(kwargs, label="test")
        assert kwargs["service_tier"] == "flex"
        assert kwargs["timeout"] >= 900
        assert kwargs["flex_fallback_to_standard"] is True

    def test_apply_openai_service_tier_noop_when_auto(self, monkeypatch):
        _set_cfg(monkeypatch, openai_service_tier="auto")
        kwargs = {"model": "gpt-5.4", "timeout": 120}
        llms_mod._apply_openai_service_tier(kwargs, label="test")
        assert "service_tier" not in kwargs
        assert kwargs["timeout"] == 120

    def test_apply_openai_service_tier_skips_learned_unsupported(self, monkeypatch):
        from src.service_tiers import mark_flex_unsupported

        _set_cfg(monkeypatch, openai_service_tier="flex")
        mark_flex_unsupported("gpt-5.4")
        kwargs = {"model": "gpt-5.4", "timeout": 120}
        llms_mod._apply_openai_service_tier(kwargs, label="test")
        assert "service_tier" not in kwargs

    def test_construct_chat_openai_uses_subclass_only_for_flex(self):
        flex_cls = llms_mod._get_flex_fallback_chat_openai_cls()
        llm = llms_mod._construct_chat_openai(
            {"model": "gpt-5.4", "api_key": "k", "service_tier": "flex"}
        )
        assert isinstance(llm, flex_cls)
        plain = llms_mod._construct_chat_openai({"model": "gpt-5.4", "api_key": "k"})
        assert not isinstance(plain, flex_cls)

    def test_consultant_llm_gets_flex_tier(self, monkeypatch):
        _set_cfg(monkeypatch, openai_service_tier="flex")
        _set_cfg(monkeypatch, enable_consultant=True)
        monkeypatch.setattr(type(config), "get_openai_api_key", lambda self: "test-key")
        llm = llms_mod.create_consultant_llm(model="gpt-5.4")
        assert llm.service_tier == "flex"
        flex_cls = llms_mod._get_flex_fallback_chat_openai_cls()
        assert isinstance(llm, flex_cls)

    @pytest.mark.asyncio
    async def test_openai_capacity_fallback_retries_on_auto(self):
        from langchain_openai import ChatOpenAI

        flex_cls = llms_mod._get_flex_fallback_chat_openai_cls()
        llm = flex_cls(
            model="gpt-5.4",
            api_key="test-key",
            service_tier="flex",
            flex_fallback_to_standard=True,
        )
        calls: list[dict] = []

        async def fake_agenerate(self, *args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise ValueError("429 resource_unavailable: flex capacity")
            return _chat_result()

        with patch.object(ChatOpenAI, "_agenerate", fake_agenerate):
            await llm._agenerate([HumanMessage(content="hi")])

        assert len(calls) == 2
        assert calls[1]["service_tier"] == "auto"

    @pytest.mark.asyncio
    async def test_openai_capability_error_caches_downgrade(self):
        from langchain_openai import ChatOpenAI

        flex_cls = llms_mod._get_flex_fallback_chat_openai_cls()
        llm = flex_cls(
            model="gpt-4o-mini-retired",
            api_key="test-key",
            service_tier="flex",
        )
        calls: list[dict] = []

        async def fake_agenerate(self, *args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise ValueError("Error code: 400 - service_tier flex is not supported")
            return _chat_result()

        with patch.object(ChatOpenAI, "_agenerate", fake_agenerate):
            await llm._agenerate([HumanMessage(content="hi")])

        assert calls[1]["service_tier"] == "auto"
        assert is_flex_unsupported("gpt-4o-mini-retired")
        # Later calls on the same instance never send flex again
        calls.clear()

        async def fake_agenerate_strict(self, *args, **kwargs):
            calls.append(kwargs)
            assert kwargs.get("service_tier") == "auto"
            return _chat_result()

        with patch.object(ChatOpenAI, "_agenerate", fake_agenerate_strict):
            await llm._agenerate([HumanMessage(content="hi")])
        assert len(calls) == 1


class TestLLMJudgePinnedStandard:
    def test_judge_inspector_pins_standard_tier(self, monkeypatch):
        _set_cfg(monkeypatch, gemini_service_tier="flex")
        captured: dict = {}

        def fake_create(**kw):
            captured.update(kw)
            return MagicMock()

        from src.tooling.llm_judge_inspector import LLMJudgeInspector

        inspector = LLMJudgeInspector.__new__(LLMJudgeInspector)
        inspector._llm = None
        inspector._model_name = "gemini-3.1-flash-lite"
        inspector._merge_judge_callbacks = lambda: []
        with patch("src.llms.create_quick_thinking_llm", side_effect=fake_create):
            inspector._get_llm()
        assert captured.get("service_tier") == "standard"


class TestFlexLatencyTimeoutDetection:
    """`_is_flex_latency_timeout` classifies a queued flex call's SDK client
    timeout so the transport can fall back to standard (Fix A3 step 2)."""

    def test_builtin_timeout_error(self):
        assert llms_mod._is_flex_latency_timeout(TimeoutError("timed out"))

    def test_asyncio_timeout_error(self):
        import asyncio

        assert llms_mod._is_flex_latency_timeout(asyncio.TimeoutError())

    def test_class_name_match(self):
        class APITimeoutError(Exception):
            pass

        assert llms_mod._is_flex_latency_timeout(APITimeoutError("boom"))

    def test_deadline_message_match(self):
        assert llms_mod._is_flex_latency_timeout(
            ValueError("504 Deadline Exceeded while awaiting flex worker")
        )

    def test_real_google_504_deadline_exceeded_body(self):
        # Regression: the verbatim Google 504 body uses status
        # 'DEADLINE_EXCEEDED' (underscore) + message 'Deadline expired ...' —
        # neither matched the original ("deadline exceeded"/"deadlineexceeded")
        # markers, so 504s silently bypassed the standard-tier fallback and
        # sank every gate-critical quick-mode ticker (2026-07-06).
        msg = (
            "504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': "
            "'Deadline expired before operation could complete.', 'status': "
            "'DEADLINE_EXCEEDED'}}"
        )

        class ServerError(Exception):
            pass

        assert llms_mod._is_flex_latency_timeout(ServerError(msg))

    def test_unrelated_error_is_not_timeout(self):
        assert not llms_mod._is_flex_latency_timeout(ValueError("bad request"))


class TestGeminiFlexLatencyFallback:
    """Row 5/6/7-8 of the flex-fallback x timeout matrix, Gemini transport."""

    @pytest.mark.asyncio
    async def test_latency_timeout_falls_back_to_standard(self):
        llm = _tiered_llm()
        calls: list[dict] = []

        async def fake_agenerate(self, *args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise TimeoutError("flex attempt exceeded client timeout")
            return _chat_result()

        with patch.object(ChatGoogleGenerativeAI, "_agenerate", fake_agenerate):
            result = await llm._agenerate([HumanMessage(content="hi")])

        assert len(calls) == 2
        assert calls[1]["service_tier"] == "standard"
        assert result.llm_output["service_tier"] == "standard"
        # Latency is transient — must NOT be cached as capability-unsupported.
        assert not is_flex_unsupported("gemini-3.5-flash")

    @pytest.mark.asyncio
    async def test_latency_timeout_propagates_when_fallback_disabled(self):
        llm = _tiered_llm(flex_fallback_to_standard=False)

        async def fake_agenerate(self, *args, **kwargs):
            raise TimeoutError("flex queued too long")

        with (
            patch.object(ChatGoogleGenerativeAI, "_agenerate", fake_agenerate),
            pytest.raises(TimeoutError),
        ):
            await llm._agenerate([HumanMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_standard_reissue_does_not_loop_on_timeout(self):
        """No-loop guard: the standard re-issue's tier is not flex, so a second
        timeout propagates instead of recursing into another fallback."""
        llm = _tiered_llm()
        calls: list[dict] = []

        async def fake_agenerate(self, *args, **kwargs):
            calls.append(kwargs)
            raise TimeoutError("both attempts time out")

        with (
            patch.object(ChatGoogleGenerativeAI, "_agenerate", fake_agenerate),
            pytest.raises(TimeoutError),
        ):
            await llm._agenerate([HumanMessage(content="hi")])

        # Exactly two SDK calls: the flex attempt + one standard re-issue.
        assert len(calls) == 2
        assert calls[1]["service_tier"] == "standard"

    def test_flex_retry_tier_none_for_standard_attempt(self):
        llm = _tiered_llm()
        # A call already at standard tier must never trigger fallback.
        assert (
            llm._flex_retry_tier(TimeoutError("x"), {"service_tier": "standard"})
            is None
        )


class TestOpenAIFlexLatencyFallback:
    @pytest.mark.asyncio
    async def test_latency_timeout_falls_back_to_auto(self):
        from langchain_openai import ChatOpenAI

        flex_cls = llms_mod._get_flex_fallback_chat_openai_cls()
        llm = flex_cls(
            model="gpt-5.4",
            api_key="test-key",
            service_tier="flex",
            flex_fallback_to_standard=True,
        )
        calls: list[dict] = []

        async def fake_agenerate(self, *args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise TimeoutError("flex attempt exceeded client timeout")
            return _chat_result()

        with patch.object(ChatOpenAI, "_agenerate", fake_agenerate):
            await llm._agenerate([HumanMessage(content="hi")])

        assert len(calls) == 2
        assert calls[1]["service_tier"] == "auto"
        assert not is_flex_unsupported("gpt-5.4")

    @pytest.mark.asyncio
    async def test_latency_timeout_propagates_when_fallback_disabled(self):
        from langchain_openai import ChatOpenAI

        flex_cls = llms_mod._get_flex_fallback_chat_openai_cls()
        llm = flex_cls(
            model="gpt-5.4",
            api_key="test-key",
            service_tier="flex",
            flex_fallback_to_standard=False,
        )

        async def fake_agenerate(self, *args, **kwargs):
            raise TimeoutError("flex queued too long")

        with (
            patch.object(ChatOpenAI, "_agenerate", fake_agenerate),
            pytest.raises(TimeoutError),
        ):
            await llm._agenerate([HumanMessage(content="hi")])


class TestFlexHealthGatesTheTransports:
    """A degraded provider must stop *requesting* flex, on both transports."""

    @staticmethod
    def _degrade(provider: str) -> None:
        """Degrade a provider on the *real* monotonic clock.

        The transports call ``flex_degraded()`` with no ``now``, so synthetic
        timestamps would place the cool-off in the distant past and read as
        already expired. Unit tests of the state machine itself inject both
        sides of the clock (see tests/test_service_tiers.py).
        """
        import time

        from src.config import Settings
        from src.service_tiers import note_flex_fallback

        cfg = Settings(_env_file=None, flex_degrade_threshold=2)
        base = time.monotonic()
        note_flex_fallback(provider, reason="latency", now=base, cfg=cfg)
        note_flex_fallback(provider, reason="latency", now=base + 0.1, cfg=cfg)

    def test_gemini_stops_injecting_service_tier_once_degraded(self):
        model = llms_mod._TieredChatGoogleGenerativeAI(
            model="gemini-3.6-flash", google_api_key="k", service_tier="flex"
        )
        assert model._effective_tier({}) == "flex"

        self._degrade("google")

        assert model._effective_tier({}) is None
        assert model._flex_ineligible() is True

    def test_openai_falls_back_to_auto_once_degraded(self):
        cls = llms_mod._get_flex_fallback_chat_openai_cls()
        model = cls(model="gpt-5.4", api_key="k", service_tier="flex")
        assert model._effective_tier({}) == "flex"

        self._degrade("openai")

        assert model._effective_tier({}) == "auto"
        assert model._payload_kwargs({}) == {"service_tier": "auto"}

    def test_degrading_one_provider_leaves_the_other_alone(self):
        """The review plane must not be downgraded by base-plane congestion."""
        gemini = llms_mod._TieredChatGoogleGenerativeAI(
            model="gemini-3.6-flash", google_api_key="k", service_tier="flex"
        )
        openai_cls = llms_mod._get_flex_fallback_chat_openai_cls()
        openai = openai_cls(model="gpt-5.4", api_key="k", service_tier="flex")

        self._degrade("google")

        assert gemini._effective_tier({}) is None
        assert openai._effective_tier({}) == "flex"

    def test_capability_downgrade_still_wins_and_stays_permanent(self):
        """Health is a cool-off; capability is forever. They must not collide."""
        from src.service_tiers import mark_flex_unsupported

        model = llms_mod._TieredChatGoogleGenerativeAI(
            model="gemini-3.6-flash", google_api_key="k", service_tier="flex"
        )
        mark_flex_unsupported("gemini-3.6-flash")

        assert model._flex_ineligible() is True
        # No cool-off can restore a model the vendor rejects.
        assert model._effective_tier({}) is None

    def test_latency_fallback_records_the_failure(self):
        from src.service_tiers import flex_degradation_snapshot

        model = llms_mod._TieredChatGoogleGenerativeAI(
            model="gemini-3.6-flash", google_api_key="k", service_tier="flex"
        )
        exc = TimeoutError("Deadline expired before operation could complete")
        assert llms_mod._is_flex_latency_timeout(exc) is True

        for _ in range(2):
            model._flex_retry_tier(exc, {})

        assert flex_degradation_snapshot()["google"]["degraded"] is True

    def test_eligibility_is_independent_of_the_fallback_setting(self):
        """`flex_fallback_to_standard=false` is about recovery, not about asking.

        A provider already known to be degraded must stop requesting flex even
        when per-call fallback is disabled -- otherwise the operator who turned
        fallback off pays the full queue wait on every call, forever.
        """
        model = llms_mod._TieredChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key="k",
            service_tier="flex",
            flex_fallback_to_standard=False,
        )
        self._degrade("google")

        assert model._effective_tier({}) is None


def test_per_call_tier_resolvers_consult_both_caches():
    """Every per-call flex gate must check capability AND health.

    A hand-maintained list of call sites structurally cannot catch a seventh one
    added later, so this scans the source. The two *construction-time* sites are
    excluded deliberately: `_resolve_gemini_service_tier` and
    `_apply_openai_service_tier` run once per seat at graph-build time and only
    choose which transport class to build. They cannot observe a degradation that
    happens mid-run, and gating them too would create a second source of truth
    for the same question -- `_effective_tier` is consulted per call and is what
    actually decides.
    """
    import ast
    from pathlib import Path

    _CONSTRUCTION_TIME_SITES = {
        "_resolve_gemini_service_tier",
        "_apply_openai_service_tier",
    }

    source = Path("src/llms.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    def _calls(node: ast.AST) -> set[str]:
        return {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        called = _calls(node)
        if "is_flex_unsupported" not in called:
            continue
        if node.name in _CONSTRUCTION_TIME_SITES:
            continue
        if "flex_degraded" not in called:
            offenders.append(node.name)

    assert not offenders, (
        f"per-call flex gates consult is_flex_unsupported but not flex_degraded: "
        f"{offenders}. Either add the health check, or -- if this is a "
        f"construction-time site -- add it to _CONSTRUCTION_TIME_SITES with the "
        f"reason."
    )
