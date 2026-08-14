import sys
from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.llm_runtime.adapters.base import SeatModelRequest
from src.llm_runtime.bindings import resolve_binding_plan
from src.llm_runtime.factory import SeatModelFactory
from src.llm_runtime.seats import SEATS, SeatId
from src.runtime_diagnostics import get_runtime_provider
from src.service_tiers import provider_flex_active


def _plan():
    return resolve_binding_plan(
        Settings(
            _env_file=None,
            llm_base_provider="google",
            llm_review_provider="openai",
            google_api_key="g",
            openai_api_key="o",
            claude_api_key="a",
            deepseek_api_key="d",
        )
    )


def test_factory_selects_native_adapter_by_resolved_identity() -> None:
    plan = _plan()
    factory = SeatModelFactory()
    google = SeatModelRequest(plan.bindings[SeatId.MARKET], SEATS[SeatId.MARKET], False)
    openai = SeatModelRequest(
        plan.bindings[SeatId.CONSULTANT], SEATS[SeatId.CONSULTANT], False
    )
    assert factory.adapter_for(google).kind == "google_native"
    assert factory.adapter_for(openai).kind == "openai_native"


def test_factory_builds_fresh_instance_per_call(monkeypatch) -> None:
    plan = _plan()
    request = SeatModelRequest(
        plan.bindings[SeatId.MARKET], SEATS[SeatId.MARKET], False
    )
    factory = SeatModelFactory()
    adapter = factory.adapter_for(request)
    first, second = MagicMock(), MagicMock()
    monkeypatch.setattr(adapter, "build", MagicMock(side_effect=[first, second]))
    assert factory.build(request) is first
    assert factory.build(request) is second
    assert first is not second


def test_compat_adapter_rejects_unqualified_tool_seat() -> None:
    plan = _plan()
    binding = plan.bindings[SeatId.APAC]
    request = SeatModelRequest(binding, SEATS[SeatId.MARKET], False)
    with pytest.raises(ValueError, match="restricted"):
        SeatModelFactory().build(request)


def test_moonshot_review_adapter_honors_reasoning_override(monkeypatch) -> None:
    import langchain_openai

    settings = Settings(
        _env_file=None,
        llm_base_provider="google",
        llm_review_provider="moonshot",
        google_api_key="g",
        moonshot_api_key="m",
        claude_api_key="a",
        deepseek_api_key="d",
        openai_service_tier="flex",
        llm_seat_reasoning_overrides={"consultant": "low"},
    )
    plan = resolve_binding_plan(settings)
    captured = {}
    model = MagicMock()

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", fake_chat_openai)
    request = SeatModelRequest(
        plan.bindings[SeatId.CONSULTANT],
        SEATS[SeatId.CONSULTANT],
        False,
        output_tokens=4096,
        reasoning_value="low",
        settings=settings,
    )
    assert SeatModelFactory().build(request) is model
    assert captured["reasoning_effort"] == "low"
    assert captured["base_url"] == "https://api.moonshot.ai/v1"
    assert captured["max_completion_tokens"] == 4096 + 2048
    assert captured["max_retries"] == 0
    assert captured["timeout"] == settings.api_timeout
    assert "service_tier" not in captured
    assert "flex_fallback_to_standard" not in captured
    assert get_runtime_provider(model) == "moonshot"
    assert provider_flex_active(get_runtime_provider(model), settings) is False


@pytest.mark.parametrize(
    ("seat_id", "expected_temperature", "expected_timeout", "expected_retries"),
    [
        (SeatId.CONTENT_INSPECTOR, 0.0, None, None),
        (SeatId.MACRO_CONTEXT, 0.1, None, None),
        (SeatId.HEALTH_CHECK, 0.0, 10.0, 1),
    ],
)
def test_google_adapter_honors_specialized_seat_execution_policy(
    monkeypatch,
    seat_id,
    expected_temperature,
    expected_timeout,
    expected_retries,
) -> None:
    from src import llms

    plan = _plan()
    captured = {}
    model = MagicMock()

    def fake_quick(**kwargs):
        captured.update(kwargs)
        return model

    def fake_gemini(_model, temperature, timeout, max_retries, **_kwargs):
        captured.update(
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        return model

    monkeypatch.setattr(llms, "create_quick_thinking_llm", fake_quick)
    monkeypatch.setattr(llms, "create_gemini_model", fake_gemini)
    request = SeatModelRequest(
        plan.bindings[seat_id],
        SEATS[seat_id],
        False,
        settings=Settings(
            _env_file=None,
            llm_base_provider="google",
            llm_review_provider="openai",
            google_api_key="g",
            openai_api_key="o",
        ),
    )

    assert SeatModelFactory().build(request) is model
    assert captured["temperature"] == expected_temperature
    assert captured["timeout"] == expected_timeout
    assert captured["max_retries"] == expected_retries
    if SEATS[seat_id].execution_policy.standard_tier_only:
        assert request.service_tier is None


def test_openai_adapter_omits_temperature_but_keeps_health_bounds(monkeypatch) -> None:
    import langchain_openai

    settings = Settings(
        _env_file=None,
        llm_base_provider="google",
        llm_review_provider="openai",
        llm_operational_provider="openai",
        llm_judge_provider="google",
        google_api_key="g",
        openai_api_key="o",
    )
    plan = resolve_binding_plan(settings)
    captured = {}
    model = MagicMock()

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", fake_chat_openai)
    request = SeatModelRequest(
        plan.bindings[SeatId.HEALTH_CHECK],
        SEATS[SeatId.HEALTH_CHECK],
        False,
        settings=settings,
    )

    assert SeatModelFactory().build(request) is model
    assert captured["timeout"] == 10.0
    assert captured["max_retries"] == 1
    assert "temperature" not in captured


def test_standard_openai_adapter_builds_normalized_budget_contract(monkeypatch) -> None:
    import langchain_openai

    settings = Settings(
        _env_file=None,
        llm_base_provider="openai",
        llm_review_provider="google",
        llm_operational_provider="openai",
        llm_judge_provider="google",
        google_api_key="g",
        openai_api_key="o",
        claude_api_key="a",
        deepseek_api_key="d",
    )
    plan = resolve_binding_plan(settings)
    captured = {}
    model = MagicMock()

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", fake_chat_openai)
    request = SeatModelRequest(
        plan.bindings[SeatId.MARKET],
        SEATS[SeatId.MARKET],
        False,
        output_tokens=2048,
        reasoning_value="medium",
        settings=settings,
    )
    assert SeatModelFactory().build(request) is model
    assert captured["reasoning_effort"] == "medium"
    assert captured["max_completion_tokens"] == 4096
    assert model._configured_max_completion_tokens == 2048
    assert model._configured_reasoning_reserve_tokens == 2048


def test_openai_adapter_honors_reviewed_official_base_url(monkeypatch) -> None:
    import langchain_openai

    settings = Settings(
        _env_file=None,
        llm_base_provider="openai",
        llm_review_provider="google",
        google_api_key="g",
        openai_api_key="o",
        claude_api_key="a",
        openai_api_base="https://api.openai.com/v1",
    )
    plan = resolve_binding_plan(settings)
    captured = {}
    model = MagicMock()

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", fake_chat_openai)
    request = SeatModelRequest(
        plan.bindings[SeatId.MARKET],
        SEATS[SeatId.MARKET],
        False,
        settings=settings,
    )

    assert SeatModelFactory().build(request) is model
    assert captured["base_url"] == "https://api.openai.com/v1"
    assert captured["use_responses_api"] is True
    assert captured["output_version"] == "responses/v1"


def test_standard_openai_adapter_does_not_import_google_facade(monkeypatch) -> None:
    import langchain_openai

    settings = Settings(
        _env_file=None,
        llm_base_provider="openai",
        llm_review_provider="google",
        llm_operational_provider="openai",
        llm_judge_provider="google",
        google_api_key="g",
        openai_api_key="o",
        claude_api_key="a",
        deepseek_api_key="d",
    )
    plan = resolve_binding_plan(settings)
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: MagicMock())
    monkeypatch.delitem(sys.modules, "src.llms", raising=False)
    monkeypatch.delitem(sys.modules, "langchain_google_genai", raising=False)
    request = SeatModelRequest(
        plan.bindings[SeatId.MARKET],
        SEATS[SeatId.MARKET],
        False,
        output_tokens=2048,
        reasoning_value="medium",
        settings=settings,
    )
    SeatModelFactory().build(request)
    assert "src.llms" not in sys.modules
    assert "langchain_google_genai" not in sys.modules


def test_anthropic_adapter_preserves_adaptive_writer_construction(monkeypatch) -> None:
    import langchain_anthropic

    settings = Settings(
        _env_file=None,
        llm_base_provider="google",
        llm_review_provider="openai",
        google_api_key="g",
        openai_api_key="o",
        claude_api_key="a",
        deepseek_api_key="d",
        anthropic_llm_prose_model="claude-opus-4-8",
        llm_seat_reasoning_overrides={"article_writer": "high"},
    )
    plan = resolve_binding_plan(settings)
    captured = {}
    model = MagicMock()

    def fake_chat_anthropic(**kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", fake_chat_anthropic)
    request = SeatModelRequest(
        plan.bindings[SeatId.ARTICLE_WRITER],
        SEATS[SeatId.ARTICLE_WRITER],
        False,
        settings=settings,
    )
    assert SeatModelFactory().build(request) is model
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["effort"] == "high"
    assert "temperature" not in captured


def test_apac_compatible_adapter_honors_reasoning_override(monkeypatch) -> None:
    import langchain_openai

    settings = Settings(
        _env_file=None,
        llm_base_provider="google",
        llm_review_provider="openai",
        llm_regional_provider="zai",
        google_api_key="g",
        openai_api_key="o",
        claude_api_key="a",
        zai_api_key="z",
        llm_apac_mode="required",
        llm_seat_reasoning_overrides={"apac_regional_specialist": "low"},
    )
    plan = resolve_binding_plan(settings)
    captured = {}
    model = MagicMock()

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", fake_chat_openai)
    binding = plan.bindings[SeatId.APAC]
    request = SeatModelRequest(
        binding,
        SEATS[SeatId.APAC],
        False,
        output_tokens=4096,
        reasoning_value="low",
        settings=settings,
    )
    assert SeatModelFactory().build(request) is model
    assert captured["reasoning_effort"] == "low"
    assert captured["extra_body"] == {"thinking": {"type": "enabled"}}
