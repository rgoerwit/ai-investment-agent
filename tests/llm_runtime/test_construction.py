from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.llm_runtime.bindings import resolve_binding_plan
from src.llm_runtime.construction import (
    build_model_for_seat,
    build_required_model_for_seat,
    writer_seat_fallback_chain,
)
from src.llm_runtime.seats import SeatId
from src.token_tracker import TokenTracker, TokenTrackingCallback


def _settings(**overrides):
    values = {
        "llm_base_provider": "openai",
        "llm_review_provider": "google",
        "llm_regional_provider": "zai",
        "llm_writer_provider": "anthropic",
        "llm_operational_provider": "openai",
        "llm_judge_provider": "google",
        "google_api_key": "g",
        "openai_api_key": "o",
        "claude_api_key": "a",
        "zai_api_key": "z",
        "llm_consultant_mode": "required",
        "llm_auditor_mode": "required",
        "llm_editor_mode": "required",
        "llm_apac_mode": "required",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class RecordingFactory:
    def __init__(self) -> None:
        self.requests = []

    def build(self, request):
        self.requests.append(request)
        return MagicMock(name=request.seat.seat_id.value)


def test_operational_and_judge_seats_use_their_own_groups() -> None:
    settings = _settings()
    plan = resolve_binding_plan(settings)
    factory = RecordingFactory()

    build_required_model_for_seat(
        SeatId.MACRO_CONTEXT, settings=settings, plan=plan, factory=factory
    )
    build_required_model_for_seat(
        SeatId.CONTENT_INSPECTOR,
        settings=settings,
        plan=plan,
        factory=factory,
    )
    build_required_model_for_seat(
        SeatId.SEMANTIC_JUDGE, settings=settings, plan=plan, factory=factory
    )

    assert [request.binding.provider for request in factory.requests] == [
        "openai",
        "openai",
        "google",
    ]
    assert factory.requests[1].service_tier == "standard"


def test_writer_fallback_tiers_are_lazy_and_provider_neutral() -> None:
    settings = _settings()
    plan = resolve_binding_plan(settings)
    factory = RecordingFactory()
    tiers = writer_seat_fallback_chain(settings=settings, plan=plan, factory=factory)

    assert factory.requests == []
    assert [tier.label for tier in tiers] == ["review_group", "base_group"]
    tiers[0].build()
    assert factory.requests[0].seat.seat_id is SeatId.ARTICLE_WRITER_REVIEW_FALLBACK
    assert factory.requests[0].binding.provider == "google"


def test_writer_fallback_chain_survives_unavailable_primary_provider() -> None:
    settings = _settings(claude_api_key="")
    plan = resolve_binding_plan(settings)
    factory = RecordingFactory()
    tiers = writer_seat_fallback_chain(settings=settings, plan=plan, factory=factory)
    assert plan.statuses[SeatId.ARTICLE_WRITER].enabled is False
    assert [tier.label for tier in tiers] == ["review_group", "base_group"]
    assert factory.requests == []


def test_disabled_new_schema_seat_returns_none_without_calling_factory() -> None:
    settings = _settings(llm_consultant_mode="off")
    plan = resolve_binding_plan(settings)
    factory = RecordingFactory()
    assert (
        build_model_for_seat(
            SeatId.CONSULTANT, settings=settings, plan=plan, factory=factory
        )
        is None
    )
    assert factory.requests == []


def test_quick_mode_apac_returns_none_without_calling_factory() -> None:
    settings = _settings()
    plan = resolve_binding_plan(settings)
    factory = RecordingFactory()

    assert (
        build_model_for_seat(
            SeatId.APAC,
            settings=settings,
            plan=plan,
            factory=factory,
            quick_mode=True,
        )
        is None
    )
    assert factory.requests == []


def test_semantic_judge_rejects_an_unpinned_cli_override() -> None:
    settings = _settings()
    plan = resolve_binding_plan(settings)
    with pytest.raises(ValueError, match="pin it in environment settings"):
        build_required_model_for_seat(
            SeatId.SEMANTIC_JUDGE,
            settings=settings,
            plan=plan,
            factory=RecordingFactory(),
            model_override="gpt-5.4",
        )


def test_zai_regional_binding_uses_zai_identity_and_endpoint() -> None:
    plan = resolve_binding_plan(_settings())
    binding = plan.bindings[SeatId.APAC]
    assert binding.provider == "zai"
    assert binding.identity.vendor_id == "zai"
    assert binding.endpoint_host == "api.z.ai"


def test_construction_binds_effective_identity_to_tracking_callback() -> None:
    settings = _settings()
    plan = resolve_binding_plan(settings)
    callback = TokenTrackingCallback("Macro", TokenTracker())
    build_required_model_for_seat(
        SeatId.MACRO_CONTEXT,
        settings=settings,
        plan=plan,
        factory=RecordingFactory(),
        callbacks=[callback],
    )
    assert callback.seat_id == SeatId.MACRO_CONTEXT.value
    assert callback.binding_group == "operational"
    assert callback.vendor_id == "openai"
    assert callback.adapter_kind == "openai_native"


def test_factory_model_identity_distinguishes_provider_from_adapter() -> None:
    from src.runtime_diagnostics import get_runtime_provider

    settings = _settings(
        llm_base_provider="google",
        llm_review_provider="moonshot",
        moonshot_api_key="m",
    )
    plan = resolve_binding_plan(settings)
    model = build_required_model_for_seat(
        SeatId.CONSULTANT,
        settings=settings,
        plan=plan,
    )

    assert model._llm_adapter_kind == "openai_compatible"
    assert model._llm_vendor_id == "moonshot"
    assert get_runtime_provider(model) == "moonshot"


def test_reasoning_and_service_tier_overrides_reach_adapter_request() -> None:
    settings = _settings(
        openai_service_tier="flex",
        llm_seat_reasoning_overrides={"macro_context_analyst": "high"},
    )
    plan = resolve_binding_plan(settings)
    factory = RecordingFactory()
    build_required_model_for_seat(
        SeatId.MACRO_CONTEXT, settings=settings, plan=plan, factory=factory
    )
    request = factory.requests[0]
    assert request.reasoning_value == "high"
    assert request.service_tier == "flex"


def test_portfolio_macro_classifier_preserves_deep_normal_and_fast_quick_modes() -> (
    None
):
    settings = _settings(llm_operational_provider="openai")
    plan = resolve_binding_plan(settings)
    factory = RecordingFactory()

    build_required_model_for_seat(
        SeatId.PORTFOLIO_MACRO_CLASSIFIER,
        settings=settings,
        plan=plan,
        factory=factory,
    )
    build_required_model_for_seat(
        SeatId.PORTFOLIO_MACRO_CLASSIFIER,
        settings=settings,
        plan=plan,
        factory=factory,
        quick_mode=True,
    )

    normal, quick = factory.requests
    assert normal.binding.intent.value == "reasoning"
    assert normal.binding.model == settings.openai_llm_reasoning_model
    assert quick.binding.intent.value == "classifier"
    assert quick.binding.model == settings.openai_llm_fast_model


def test_anthropic_writer_uses_editorial_high_effort_not_fast_low_default() -> None:
    settings = _settings()
    plan = resolve_binding_plan(settings)
    factory = RecordingFactory()
    build_required_model_for_seat(
        SeatId.ARTICLE_WRITER, settings=settings, plan=plan, factory=factory
    )
    assert factory.requests[0].reasoning_value == "high"


@pytest.mark.parametrize(
    ("seat_id", "expected_temperature"),
    [
        (SeatId.CONTENT_INSPECTOR, 0.0),
        (SeatId.MACRO_CONTEXT, 0.1),
    ],
)
def test_legacy_auxiliary_seats_keep_sampling_policy(
    monkeypatch, seat_id, expected_temperature
) -> None:
    from src import llms

    captured = {}
    model = MagicMock()

    def fake_quick(**kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(llms, "create_quick_thinking_llm", fake_quick)
    settings = Settings(_env_file=None)

    assert build_required_model_for_seat(seat_id, settings=settings) is model
    assert captured["temperature"] == expected_temperature


def test_legacy_construction_honors_scoped_settings(monkeypatch) -> None:
    from src import llms

    captured = {}
    model = MagicMock()

    def fake_quick(**kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(llms, "create_quick_thinking_llm", fake_quick)
    settings = Settings(
        _env_file=None,
        quick_think_llm="gemini-3.6-flash",
        google_api_key="scoped-google-key",
    )

    assert build_required_model_for_seat(SeatId.MARKET, settings=settings) is model
    assert captured["model"] == "gemini-3.6-flash"
    assert captured["api_key"] == "scoped-google-key"
    assert captured["settings"] is settings
