import pytest

from src.config import Settings
from src.llm_runtime.bindings import (
    BindingConfigurationError,
    SeatStatus,
    resolve_binding_plan,
)
from src.llm_runtime.seats import SeatId


@pytest.mark.parametrize(
    "mode,ready,enabled,reason",
    [
        ("required", True, True, None),
        ("auto", True, True, None),
        ("auto", False, False, "missing key"),
        ("off", True, False, "configured off"),
    ],
)
def test_seat_mode_and_provider_readiness_are_separate(
    mode, ready, enabled, reason
) -> None:
    status = SeatStatus.resolve(mode, provider_ready=ready, reason="missing key")
    assert status.enabled is enabled
    assert status.reason == reason
    assert status.provider_ready is ready


def test_optional_modes_are_independent() -> None:
    plan = resolve_binding_plan(
        Settings(
            _env_file=None,
            llm_base_provider="google",
            llm_review_provider="openai",
            google_api_key="g",
            openai_api_key="o",
            claude_api_key="a",
            deepseek_api_key="d",
            llm_consultant_mode="off",
            llm_auditor_mode="required",
            llm_editor_mode="off",
            llm_apac_mode="required",
        )
    )
    assert plan.statuses[SeatId.CONSULTANT].enabled is False
    assert plan.statuses[SeatId.AUDITOR].enabled is True
    assert plan.statuses[SeatId.EDITOR].enabled is False
    assert plan.statuses[SeatId.APAC].enabled is True


def test_required_unavailable_fails_while_auto_records_reason() -> None:
    common = {
        "_env_file": None,
        "llm_base_provider": "google",
        "llm_review_provider": "openai",
        "google_api_key": "g",
        "claude_api_key": "a",
        "deepseek_api_key": "d",
        "llm_auditor_mode": "off",
        "llm_editor_mode": "off",
    }
    with pytest.raises(
        BindingConfigurationError, match="consultant: missing credential"
    ):
        resolve_binding_plan(
            Settings(**common, llm_consultant_mode="required", openai_api_key="")
        )
    plan = resolve_binding_plan(
        Settings(**common, llm_consultant_mode="auto", openai_api_key="")
    )
    status = plan.statuses[SeatId.CONSULTANT]
    assert status.enabled is False
    assert status.reason == "missing credential"
    assert status.mode == "auto"


def test_apac_availability_is_explicitly_mode_aware() -> None:
    settings = Settings(
        _env_file=None,
        llm_base_provider="google",
        llm_review_provider="openai",
        google_api_key="g",
        openai_api_key="o",
        claude_api_key="a",
        deepseek_api_key="d",
        llm_consultant_mode="off",
        llm_auditor_mode="off",
        llm_editor_mode="off",
        llm_apac_mode="required",
    )
    plan = resolve_binding_plan(settings)

    assert plan.status_for(SeatId.APAC).enabled is True
    quick_status = plan.status_for(SeatId.APAC, quick_mode=True)
    assert quick_status.enabled is False
    assert quick_status.provider_ready is True
    assert quick_status.reason == "disabled in quick mode"
    assert SeatId.APAC not in {
        binding.seat_id for binding in plan.reachable_bindings(quick_mode=True)
    }

    telemetry = plan.telemetry(settings)["seats"][SeatId.APAC.value]
    assert telemetry["enabled"] is True
    assert telemetry["quick_enabled"] is False
    assert telemetry["quick_unavailable_reason"] == "disabled in quick mode"
