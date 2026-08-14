from src.config import Settings
from src.llm_runtime.bindings import resolve_binding_plan
from src.llm_runtime.qualification import (
    offline_contract_findings,
    validate_live_contract_result,
)
from src.llm_runtime.seats import SeatId


def _reversible_plan():
    return resolve_binding_plan(
        Settings(
            _env_file=None,
            llm_base_provider="openai",
            llm_review_provider="google",
            llm_regional_provider="zai",
            llm_writer_provider="anthropic",
            llm_operational_provider="openai",
            llm_judge_provider="google",
            google_api_key="g",
            openai_api_key="o",
            claude_api_key="a",
            zai_api_key="z",
            llm_consultant_mode="required",
            llm_auditor_mode="required",
            llm_editor_mode="required",
            llm_apac_mode="required",
        )
    )


def test_reversed_google_openai_plan_is_offline_contract_capable() -> None:
    assert offline_contract_findings(_reversible_plan()) == ()


def test_live_contract_requires_editor_tools_and_structure() -> None:
    failures = validate_live_contract_result(
        seat_id=SeatId.EDITOR,
        tool_calls_valid=False,
        structured_output_valid=False,
        artifact_valid=True,
        usage_recorded=True,
    )
    assert failures == ("tool_calling", "structured_output")


def test_live_contract_requires_artifact_and_usage_for_text_seat() -> None:
    failures = validate_live_contract_result(
        seat_id=SeatId.APAC,
        tool_calls_valid=False,
        structured_output_valid=False,
        artifact_valid=False,
        usage_recorded=False,
    )
    assert failures == ("artifact_contract", "usage_telemetry")
