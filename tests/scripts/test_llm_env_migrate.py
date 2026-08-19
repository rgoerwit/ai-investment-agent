import json
import sys

import pytest

from scripts.llm_env_migrate import (
    Choices,
    MigrationChoiceRequired,
    _validate_candidate,
    infer_legacy_provider,
    main,
    migrate_llm_values,
    parse_env,
    render_migration,
)


def _legacy(**overrides):
    values = {
        "QUICK_MODEL": "gemini-3-flash-preview",
        "DEEP_MODEL": "gemini-3.1-pro-preview",
        "GOOGLE_API_KEY": "google-secret",
        "OPENAI_API_KEY": "openai-secret",
        "CLAUDE_KEY": "anthropic-secret",
        "CONSULTANT_MODEL": "gpt-5.4",
        "AUDITOR_MODEL": "gpt-5.4",
        "EDITOR_MODEL": "gpt-5.4",
        "ENABLE_CONSULTANT": "true",
        "ENABLE_APAC_SPECIALIST": "true",
        "APAC_SPECIALIST_MODEL": "glm-5.2",
        "APAC_SPECIALIST_BASE_URL": "https://api.z.ai/api/paas/v4/",
        "APAC_SPECIALIST_API_KEY": "zai-secret",
    }
    values.update(overrides)
    return values


def test_zai_apac_endpoint_maps_to_zai_not_deepseek() -> None:
    migrated = migrate_llm_values(_legacy())
    assert migrated["LLM_REGIONAL_PROVIDER"] == "zai"
    assert migrated["ZAI_LLM_REASONING_MODEL"] == "glm-5.2"
    assert "DEEPSEEK_LLM_REASONING_MODEL" not in migrated


def test_empty_apex_preserves_distinct_senior_and_pm_models() -> None:
    migrated = migrate_llm_values(_legacy(APEX_MODEL=""))
    overrides = json.loads(migrated["LLM_SEAT_MODEL_OVERRIDES"])
    assert overrides["fundamentals_analyst"] == "gemini-3-flash-preview"
    assert overrides["portfolio_manager"] == "gemini-3.1-pro-preview"


def test_populated_apex_applies_to_both_and_quick_override() -> None:
    migrated = migrate_llm_values(
        _legacy(
            APEX_MODEL="gemini-3.1-pro-preview",
            APEX_QUICK_MODEL="gemini-3-flash-preview",
        )
    )
    normal = json.loads(migrated["LLM_SEAT_MODEL_OVERRIDES"])
    quick = json.loads(migrated["LLM_SEAT_QUICK_MODEL_OVERRIDES"])
    assert normal["fundamentals_analyst"] == normal["portfolio_manager"]
    assert quick["fundamentals_analyst"] == quick["portfolio_manager"]


def test_populated_apex_preserves_thinking_level_for_both_modes() -> None:
    migrated = migrate_llm_values(
        _legacy(
            APEX_MODEL="gemini-3.1-pro-preview",
            APEX_QUICK_MODEL="gemini-3-flash-preview",
            APEX_THINKING_LEVEL="medium",
        )
    )
    normal = json.loads(migrated["LLM_SEAT_REASONING_OVERRIDES"])
    quick = json.loads(migrated["LLM_SEAT_QUICK_REASONING_OVERRIDES"])
    assert normal == {
        "fundamentals_analyst": "medium",
        "portfolio_manager": "medium",
    }
    assert quick == normal


def test_unknown_host_requires_operator_choice() -> None:
    with pytest.raises(MigrationChoiceRequired, match="explicit provider"):
        infer_legacy_provider(
            "https://llm.example.test/v1", default="openai", explicit=None
        )


def test_reviewed_moonshot_kimi_review_plane_gets_provider_scoped_settings() -> None:
    migrated = migrate_llm_values(
        _legacy(
            OPENAI_API_BASE="https://api.moonshot.ai/v1",
            CONSULTANT_MODEL="kimi-k3",
            CONSULTANT_QUICK_MODEL="kimi-k3",
            AUDITOR_MODEL="kimi-k3",
            AUDITOR_QUICK_MODEL="kimi-k3",
            AUDITOR_ESCALATION_MODEL="kimi-k3",
            EDITOR_MODEL="kimi-k3",
        )
    )
    assert migrated["LLM_REVIEW_PROVIDER"] == "moonshot"
    assert migrated["MOONSHOT_API_BASE"] == "https://api.moonshot.ai/v1"
    assert migrated["MOONSHOT_API_KEY"] == "openai-secret"
    assert "OPENAI_API_KEY" not in migrated
    _validate_candidate({"GOOGLE_API_KEY": "google-secret", **migrated})


def test_cli_reports_migration_choice_without_traceback(
    tmp_path, monkeypatch, capsys
) -> None:
    source = tmp_path / ".env"
    source.write_text("OPENAI_API_BASE=https://unknown.example/v1\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["llm_env_migrate.py", "--check", str(source)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "migration error: unknown endpoint host" in stderr
    assert "Traceback" not in stderr


def test_render_preserves_unknown_keys_and_comments_legacy_bindings() -> None:
    source = "# comment\nCUSTOM_FLAG=yes\nLLM_PROVIDER=openai\nQUICK_MODEL=old\n"
    rendered = render_migration(source, {"LLM_BASE_PROVIDER": "google"})
    assert "CUSTOM_FLAG=yes" in rendered
    assert "# migrated legacy key: QUICK_MODEL=old" in rendered
    assert "# retired metadata-only key: LLM_PROVIDER=openai" in rendered
    assert "LLM_BASE_PROVIDER=google" in rendered


def test_parse_env_handles_quotes_and_inline_comments() -> None:
    assert parse_env('A="x y"\nB=value # note\n') == {"A": "x y", "B": "value"}


def test_generated_json_overrides_validate_as_env_values() -> None:
    source = _legacy(APEX_MODEL="gemini-3.1-pro-preview")
    migrated = migrate_llm_values(source)
    candidate = {"GOOGLE_API_KEY": source["GOOGLE_API_KEY"]}
    candidate.update(migrated)
    _validate_candidate(candidate)


def test_legacy_credentials_and_google_policies_get_new_names() -> None:
    migrated = migrate_llm_values(
        _legacy(GEMINI_RPM_LIMIT="77", GEMINI_SERVICE_TIER="flex")
    )
    assert migrated["ANTHROPIC_API_KEY"] == "anthropic-secret"
    assert migrated["GOOGLE_RPM_LIMIT"] == "77"
    assert migrated["GOOGLE_SERVICE_TIER"] == "flex"
    assert "CLAUDE_KEY" not in migrated
    assert "GEMINI_RPM_LIMIT" not in migrated
    assert "GEMINI_SERVICE_TIER" not in migrated
