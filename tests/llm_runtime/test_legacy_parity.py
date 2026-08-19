"""Curated parity between current-equivalent legacy and seat construction."""

import pytest

from src.config import Settings
from src.llm_runtime.adapters.base import SeatModelRequest
from src.llm_runtime.bindings import resolve_binding_plan
from src.llm_runtime.construction import (
    build_required_model_for_seat,
    reasoning_value_for_seat,
)
from src.llm_runtime.contracts import capture_construction_contract
from src.llm_runtime.factory import SeatModelFactory
from src.llm_runtime.seats import SEATS, BindingGroup, SeatId


class _FakeOpenAITransport:
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)


class _FakeGoogleTransport(_FakeOpenAITransport):
    pass


def _contract(llm, seat_id: SeatId):
    return capture_construction_contract(
        llm,
        seat_id=seat_id.value,
        callback_agent=SEATS[seat_id].callback_name,
    )


def _install_fake_google_transport(monkeypatch) -> None:
    from src import llms

    monkeypatch.setattr(llms, "_TieredChatGoogleGenerativeAI", _FakeGoogleTransport)


def _legacy_google_settings() -> Settings:
    return Settings(
        _env_file=None,
        google_api_key="g",
        openai_api_key="o",
        claude_api_key="a",
        deepseek_api_key="d",
        quick_think_llm="gemini-3-flash-preview",
        deep_think_llm="gemini-3.1-pro-preview",
        apex_model="gemini-3.1-pro-preview",
        apex_quick_model="gemini-3.1-pro-preview",
        enable_consultant=True,
    )


def _new_google_settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_base_provider="google",
        llm_review_provider="openai",
        llm_regional_provider="deepseek",
        llm_writer_provider="anthropic",
        llm_operational_provider="google",
        llm_judge_provider="google",
        google_api_key="g",
        openai_api_key="o",
        claude_api_key="a",
        deepseek_api_key="d",
        google_llm_fast_model="gemini-3-flash-preview",
        google_llm_reasoning_model="gemini-3.1-pro-preview",
        google_llm_critical_model="gemini-3.1-pro-preview",
        llm_consultant_mode="required",
        llm_auditor_mode="required",
        llm_editor_mode="required",
    )


_GOOGLE_PARITY_CASES = tuple(
    (seat_id, quick_mode)
    for seat_id, spec in SEATS.items()
    if spec.binding_group
    in {BindingGroup.BASE, BindingGroup.OPERATIONAL, BindingGroup.JUDGE}
    for quick_mode in (False, True)
    if not (quick_mode and spec.disabled_in_quick_mode)
)


@pytest.mark.parametrize(
    ("seat_id", "quick_mode"),
    _GOOGLE_PARITY_CASES,
    ids=lambda value: value.value if isinstance(value, SeatId) else str(value),
)
def test_google_legacy_and_provider_scoped_contracts_match(
    monkeypatch, seat_id: SeatId, quick_mode: bool
) -> None:
    _install_fake_google_transport(monkeypatch)
    legacy_settings = _legacy_google_settings()
    new_settings = _new_google_settings()

    legacy = build_required_model_for_seat(
        seat_id,
        settings=legacy_settings,
        quick_mode=quick_mode,
        output_tokens=4096,
    )
    current = build_required_model_for_seat(
        seat_id,
        settings=new_settings,
        quick_mode=quick_mode,
        output_tokens=4096,
    )

    assert _contract(current, seat_id) == _contract(legacy, seat_id)


def _legacy_openai_settings() -> Settings:
    return Settings(
        _env_file=None,
        google_api_key="g",
        openai_api_key="o",
        claude_api_key="a",
        deepseek_api_key="d",
        enable_consultant=True,
        consultant_model="gpt-5.4",
        consultant_quick_model="gpt-5.4-mini",
        auditor_model="gpt-5.4",
        auditor_quick_model="gpt-5.4-mini",
        auditor_escalation_model="gpt-5.4",
        editor_model="gpt-5.4",
        openai_service_tier="auto",
    )


def _new_openai_settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_base_provider="google",
        llm_review_provider="openai",
        llm_regional_provider="deepseek",
        llm_writer_provider="anthropic",
        llm_operational_provider="google",
        llm_judge_provider="google",
        google_api_key="g",
        openai_api_key="o",
        claude_api_key="a",
        deepseek_api_key="d",
        openai_llm_fast_model="gpt-5.4-mini",
        openai_llm_reasoning_model="gpt-5.4",
        openai_llm_critical_model="gpt-5.4",
        openai_llm_escalation_model="gpt-5.4",
        llm_consultant_mode="required",
        llm_auditor_mode="required",
        llm_editor_mode="required",
        openai_service_tier="auto",
    )


_OPENAI_REVIEW_PARITY_SEATS = (
    SeatId.CONSULTANT,
    SeatId.AUDITOR,
    SeatId.EDITOR,
    SeatId.ARTICLE_WRITER_REVIEW_FALLBACK,
)

_REVIEW_OUTPUT_TOKENS = {
    SeatId.EDITOR: 8192,
    SeatId.ARTICLE_WRITER_REVIEW_FALLBACK: 16_384,
}


@pytest.mark.parametrize(
    "seat_id", _OPENAI_REVIEW_PARITY_SEATS, ids=lambda seat: seat.value
)
@pytest.mark.parametrize("quick_mode", (False, True), ids=("normal", "quick"))
def test_openai_review_legacy_and_provider_scoped_contracts_match(
    monkeypatch, seat_id: SeatId, quick_mode: bool
) -> None:
    import langchain_openai

    from src import llms

    monkeypatch.setattr(
        llms,
        "_construct_chat_openai",
        lambda kwargs, **_ignored: _FakeOpenAITransport(**kwargs),
    )
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _FakeOpenAITransport)
    output_tokens = _REVIEW_OUTPUT_TOKENS.get(seat_id, 4096)
    legacy = build_required_model_for_seat(
        seat_id,
        settings=_legacy_openai_settings(),
        quick_mode=quick_mode,
        output_tokens=output_tokens,
    )
    current = build_required_model_for_seat(
        seat_id,
        settings=_new_openai_settings(),
        quick_mode=quick_mode,
        output_tokens=output_tokens,
    )

    assert _contract(current, seat_id) == _contract(legacy, seat_id)


def test_provider_scoped_auditor_escalation_is_deliberately_stronger_than_legacy(
    monkeypatch,
) -> None:
    import langchain_openai

    from src import llms

    monkeypatch.setattr(
        llms,
        "_construct_chat_openai",
        lambda kwargs, **_ignored: _FakeOpenAITransport(**kwargs),
    )
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _FakeOpenAITransport)
    legacy = build_required_model_for_seat(
        SeatId.AUDITOR_ESCALATION,
        settings=_legacy_openai_settings(),
        output_tokens=4096,
    )
    current = build_required_model_for_seat(
        SeatId.AUDITOR_ESCALATION,
        settings=_new_openai_settings(),
        output_tokens=4096,
    )

    legacy_contract = _contract(legacy, SeatId.AUDITOR_ESCALATION)
    current_contract = _contract(current, SeatId.AUDITOR_ESCALATION)
    assert legacy_contract.reasoning_intent == "medium"
    assert current_contract.reasoning_intent == "xhigh"
    assert current_contract.configured_reasoning_reserve_tokens > (
        legacy_contract.configured_reasoning_reserve_tokens
    )


def test_consultant_legacy_and_new_construction_contracts_match(
    monkeypatch,
) -> None:
    import langchain_openai

    from src import llms

    monkeypatch.setattr(
        llms, "_construct_chat_openai", lambda kwargs: _FakeOpenAITransport(**kwargs)
    )
    monkeypatch.setattr(llms, "_get_openai_rate_limiter", lambda: None)
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _FakeOpenAITransport)
    monkeypatch.setattr(llms.config, "enable_consultant", True)
    monkeypatch.setattr(llms.config, "openai_service_tier", "auto")
    monkeypatch.setattr(llms.config, "consultant_model", "gpt-5.4")
    monkeypatch.setattr(llms.config, "llm_default_reasoning_reserve_tokens", 2048)
    monkeypatch.setattr(llms.config, "llm_deep_reasoning_reserve_tokens", 8192)
    monkeypatch.setattr(type(llms.config), "get_openai_api_key", lambda self: "o")

    legacy = llms.create_consultant_llm(
        model="gpt-5.4", callbacks=[], max_completion_tokens=4096
    )
    settings = Settings(
        _env_file=None,
        llm_base_provider="google",
        llm_review_provider="openai",
        google_api_key="g",
        openai_api_key="o",
        claude_api_key="a",
        deepseek_api_key="d",
        llm_consultant_mode="required",
    )
    plan = resolve_binding_plan(settings)
    binding = plan.bindings[SeatId.CONSULTANT]
    reasoning = reasoning_value_for_seat(binding.profile, binding.intent, adjust=False)
    current = SeatModelFactory().build(
        SeatModelRequest(
            binding,
            SEATS[SeatId.CONSULTANT],
            False,
            output_tokens=4096,
            reasoning_value=reasoning,
            settings=settings,
        )
    )
    assert current is not None

    legacy_contract = capture_construction_contract(
        legacy,
        seat_id=SeatId.CONSULTANT.value,
        callback_agent="Consultant",
        reasoning_intent=reasoning,
    )
    current_contract = capture_construction_contract(
        current,
        seat_id=SeatId.CONSULTANT.value,
        callback_agent="Consultant",
        reasoning_intent=reasoning,
    )
    assert current_contract == legacy_contract
