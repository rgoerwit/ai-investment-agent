from types import MappingProxyType

import pytest

from src.config import Settings
from src.llm_runtime.bindings import BindingConfigurationError, resolve_binding_plan
from src.llm_runtime.seats import SeatId


def _new_settings(**overrides):
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
        "llm_consultant_mode": "required",
        "llm_auditor_mode": "required",
        "llm_editor_mode": "required",
        "llm_apac_mode": "required",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_new_plan_resolves_group_provider_models() -> None:
    plan = resolve_binding_plan(_new_settings())
    assert plan.schema == "new"
    assert plan.bindings[SeatId.MARKET].model == "gemini-3-flash-preview"
    assert plan.bindings[SeatId.BULL].provider == "google"
    assert plan.bindings[SeatId.CONSULTANT].model == "gpt-5.4"
    assert plan.bindings[SeatId.APAC].provider == "deepseek"
    assert plan.for_seat(SeatId.BULL, quick_mode=True).model == "gemini-3-flash-preview"
    assert isinstance(plan.bindings, MappingProxyType)


def test_legacy_defaults_translate_without_changing_provider_plane() -> None:
    plan = resolve_binding_plan(Settings(_env_file=None, google_api_key="google-key"))
    assert plan.schema == "legacy"
    assert plan.bindings[SeatId.MARKET].provider == "google"
    assert plan.bindings[SeatId.CONSULTANT].provider == "openai"
    assert plan.bindings[SeatId.ARTICLE_WRITER].provider == "anthropic"
    assert plan.bindings[SeatId.SENIOR_FUNDAMENTALS].model == "gemini-3-flash-preview"
    assert plan.bindings[SeatId.PORTFOLIO_MANAGER].model == "gemini-3.1-pro-preview"


def test_legacy_compatible_endpoint_models_are_not_requalified() -> None:
    """The bridge must preserve deployments accepted by the old constructors."""

    plan = resolve_binding_plan(
        Settings(
            _env_file=None,
            google_api_key="google-key",
            consultant_model="kimi-k3",
            auditor_model="kimi-k3",
            apac_specialist_model="glm-5.2",
            apac_specialist_base_url="https://api.z.ai/api/paas/v4/",
        )
    )
    assert plan.schema == "legacy"
    assert plan.bindings[SeatId.CONSULTANT].model == "kimi-k3"
    assert plan.bindings[SeatId.APAC].identity.vendor_id == "zai"


def test_reversing_only_group_providers_selects_scoped_models() -> None:
    plan = resolve_binding_plan(
        _new_settings(
            llm_base_provider="openai",
            llm_review_provider="google",
            llm_operational_provider="openai",
            llm_judge_provider="openai",
        )
    )
    assert plan.bindings[SeatId.MARKET].model == "gpt-5.4-mini"
    assert plan.bindings[SeatId.BULL].model == "gpt-5.4"
    assert plan.bindings[SeatId.CONSULTANT].model == "gemini-3.1-pro-preview"
    assert (
        plan.for_seat(SeatId.CONSULTANT, quick_mode=True).model
        == "gemini-3-flash-preview"
    )


def test_mixed_legacy_and_new_schema_fails_with_names() -> None:
    settings = _new_settings(quick_think_llm="gemini-3.1-flash-lite")
    with pytest.raises(
        BindingConfigurationError, match="llm_base_provider.*quick_think_llm"
    ):
        resolve_binding_plan(settings)


def test_documented_legacy_defaults_in_dotenv_do_not_create_false_mixed_schema(
    tmp_path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "LLM_PROVIDER=google",
                "QUICK_MODEL=gemini-3-flash-preview",
                "DEEP_MODEL=gemini-3.1-pro-preview",
                "ENABLE_CONSULTANT=true",
                "ENABLE_APAC_SPECIALIST=false",
                "LLM_BASE_PROVIDER=google",
                "LLM_REVIEW_PROVIDER=openai",
                "GOOGLE_API_KEY=g",
                "OPENAI_API_KEY=o",
            )
        ),
        encoding="utf-8",
    )
    assert resolve_binding_plan(Settings(_env_file=env_file)).schema == "new"


def test_real_legacy_override_plus_new_selector_is_actionable(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "QUICK_MODEL=custom-legacy-floor",
                "ENABLE_CONSULTANT=false",
                "LLM_BASE_PROVIDER=google",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(BindingConfigurationError) as exc_info:
        resolve_binding_plan(Settings(_env_file=env_file))
    message = str(exc_info.value)
    assert "llm_base_provider" in message
    assert "quick_think_llm" in message
    assert "enable_consultant" not in message


def test_collapsed_review_fails_before_construction() -> None:
    with pytest.raises(BindingConfigurationError, match="review binding must differ"):
        resolve_binding_plan(_new_settings(llm_review_provider="google"))


def test_review_independence_still_applies_when_consultant_is_off() -> None:
    with pytest.raises(BindingConfigurationError, match="review binding must differ"):
        resolve_binding_plan(
            _new_settings(
                llm_review_provider="google",
                llm_consultant_mode="off",
                llm_auditor_mode="required",
            )
        )


def test_waiver_is_single_boolean_plus_required_reason() -> None:
    with pytest.raises(BindingConfigurationError, match="waiver reason is required"):
        resolve_binding_plan(
            _new_settings(
                llm_review_provider="google",
                llm_require_review_independence=False,
            )
        )
    plan = resolve_binding_plan(
        _new_settings(
            llm_review_provider="google",
            llm_require_review_independence=False,
            llm_review_independence_waiver_reason="controlled same-vendor test",
        )
    )
    assert plan.bindings[SeatId.CONSULTANT].provider == "google"


def test_required_missing_credential_is_aggregated() -> None:
    with pytest.raises(BindingConfigurationError) as exc_info:
        resolve_binding_plan(_new_settings(openai_api_key="", deepseek_api_key=""))
    message = str(exc_info.value)
    assert "consultant: missing credential" in message
    assert "apac_regional_specialist: missing credential" in message


def test_inactive_historical_waiver_reason_does_not_disable_enforcement() -> None:
    settings = _new_settings(
        llm_review_independence_waiver_reason="prior controlled same-vendor test"
    )
    plan = resolve_binding_plan(settings)
    review = plan.independence_telemetry(settings)["review"]
    assert review["required"] is True
    assert review["waiver_reason"] is None


def test_transport_capability_does_not_qualify_anthropic_for_base() -> None:
    with pytest.raises(BindingConfigurationError) as exc_info:
        resolve_binding_plan(_new_settings(llm_base_provider="anthropic"))
    message = str(exc_info.value)
    assert "not application-qualified" in message
    assert "binding group 'base'" in message
    assert "has no configured 'fast' model" not in message


def test_telemetry_contains_sanitized_identity_and_no_secrets() -> None:
    settings = _new_settings(
        deepseek_api_base="https://user:secret@api.deepseek.com/v1?token=hidden"
    )
    payload = resolve_binding_plan(settings).telemetry(settings)
    apac = payload["seats"][SeatId.APAC.value]
    assert apac["endpoint_host"] == "api.deepseek.com"
    assert apac["binding_group"] == "regional_review"
    assert payload["independence"]["review"]["satisfied"] is True
    rendered = repr(payload)
    assert "secret" not in rendered
    assert "hidden" not in rendered


def test_env_example_is_a_valid_unmixed_new_schema() -> None:
    plan = resolve_binding_plan(Settings(_env_file=".env.example"))
    assert plan.schema == "new"
    assert plan.bindings[SeatId.BULL].provider == "google"
    assert plan.bindings[SeatId.BEAR].provider == "google"
    assert plan.bindings[SeatId.CONSULTANT].provider == "openai"


def test_custom_openai_base_cannot_masquerade_as_openai_vendor() -> None:
    with pytest.raises(BindingConfigurationError, match="endpoint host.*moonshot"):
        resolve_binding_plan(
            _new_settings(openai_api_base="https://api.moonshot.cn/v1")
        )


def test_malformed_endpoint_is_aggregated_with_seat_and_setting() -> None:
    with pytest.raises(BindingConfigurationError) as exc_info:
        resolve_binding_plan(_new_settings(deepseek_api_base="api.deepseek.com/v1"))
    message = str(exc_info.value)
    assert "apac_regional_specialist: DEEPSEEK_API_BASE" in message
    assert "absolute HTTP(S) URL" in message


def test_unknown_model_error_is_deduplicated_by_model() -> None:
    with pytest.raises(BindingConfigurationError) as exc_info:
        resolve_binding_plan(
            _new_settings(google_llm_fast_model="gemini-4-flash-preview")
        )
    message = str(exc_info.value)
    assert message.count("has no reviewed capability profile") == 1
    assert "market_analyst" in message


def test_moonshot_kimi_review_plane_is_first_class_and_endpoint_scoped() -> None:
    plan = resolve_binding_plan(
        _new_settings(
            llm_review_provider="moonshot",
            moonshot_api_key="moonshot-key",
            moonshot_api_base="https://api.moonshot.ai/v1",
        )
    )
    editor = plan.bindings[SeatId.EDITOR]
    assert editor.provider == "moonshot"
    assert editor.identity.vendor_id == "moonshot"
    assert editor.endpoint_host == "api.moonshot.ai"
    assert editor.model == "kimi-k3"


def test_missing_writer_credential_keeps_lazy_fallbacks_available() -> None:
    plan = resolve_binding_plan(_new_settings(claude_api_key=""))
    assert plan.statuses[SeatId.ARTICLE_WRITER].enabled is False
    assert plan.statuses[SeatId.ARTICLE_WRITER_REVIEW_FALLBACK].enabled is True
    assert plan.statuses[SeatId.ARTICLE_WRITER_BASE_FALLBACK].enabled is True


def test_reasoning_overrides_are_validated_and_resolved_per_mode() -> None:
    plan = resolve_binding_plan(
        _new_settings(
            llm_seat_reasoning_overrides={"portfolio_manager": "medium"},
            llm_seat_quick_reasoning_overrides={"portfolio_manager": "low"},
        )
    )
    assert plan.bindings[SeatId.PORTFOLIO_MANAGER].reasoning_value_override == "medium"
    assert (
        plan.quick_bindings[SeatId.PORTFOLIO_MANAGER].reasoning_value_override == "low"
    )

    with pytest.raises(BindingConfigurationError, match="supported values"):
        resolve_binding_plan(
            _new_settings(llm_seat_reasoning_overrides={"portfolio_manager": "extreme"})
        )


def test_value_trap_adjustment_cannot_silently_degrade() -> None:
    with pytest.raises(BindingConfigurationError, match="one-step reasoning"):
        resolve_binding_plan(
            _new_settings(llm_seat_model_overrides={"value_trap_detector": "gpt-4o"})
        )


class TestBaseGroupCliModelOverrides:
    """`--quick-model`/`--deep-model` under provider-scoped bindings.

    They map to the base group's `fast` and `reasoning` intents respectively --
    and to nothing else. `--deep-model` deliberately does NOT reach `critical`:
    under the legacy schema APEX_MODEL already superseded DEEP_MODEL for the two
    gate-critical seats, so extending it there would grant the flag authority it
    never had over the seats with the densest incident history in the repo.
    """

    @staticmethod
    def _settings(**over):
        return Settings(
            _env_file=None,
            google_api_key="g",
            openai_api_key="o",
            finnhub_api_key="f",
            tavily_api_key="t",
            llm_base_provider="google",
            llm_review_provider="openai",
            **over,
        )

    @staticmethod
    def _bind(**over):
        from types import SimpleNamespace

        from src.runtime_config import bind_runtime_config, build_runtime_config

        args = {"quick": False, "quick_model": None, "deep_model": None}
        args.update(over)
        settings = TestBaseGroupCliModelOverrides._settings()
        bind_runtime_config(build_runtime_config(SimpleNamespace(**args), settings))
        return settings

    def test_quick_model_moves_fast_seats_only(self, restore_runtime_config):
        settings = self._bind(quick_model="gemini-3.6-flash")
        plan = resolve_binding_plan(settings)

        assert plan.bindings[SeatId.MARKET].model == "gemini-3.6-flash"
        assert plan.bindings[SeatId.BULL].model != "gemini-3.6-flash"
        assert plan.bindings[SeatId.PORTFOLIO_MANAGER].model != "gemini-3.6-flash"

    def test_deep_model_moves_reasoning_but_never_critical(
        self, restore_runtime_config
    ):
        settings = self._bind(deep_model="gemini-3-pro-preview")
        plan = resolve_binding_plan(settings)

        assert plan.bindings[SeatId.BULL].model == "gemini-3-pro-preview"
        assert plan.bindings[SeatId.RESEARCH_MANAGER].model == "gemini-3-pro-preview"
        # The APEX seats are `critical`; they keep their configured binding.
        assert plan.bindings[SeatId.PORTFOLIO_MANAGER].model == (
            settings.google_llm_critical_model
        )
        assert plan.bindings[SeatId.SENIOR_FUNDAMENTALS].model == (
            settings.google_llm_critical_model
        )

    def test_flags_never_reach_another_binding_group(self, restore_runtime_config):
        settings = self._bind(
            quick_model="gemini-3.6-flash", deep_model="gemini-3-pro-preview"
        )
        plan = resolve_binding_plan(settings)

        for seat in (SeatId.CONSULTANT, SeatId.AUDITOR, SeatId.EDITOR):
            assert plan.bindings[seat].provider == "openai"
            assert plan.bindings[seat].model.startswith("gpt-")

    def test_cross_vendor_flag_fails_loudly_naming_both_vendors(
        self, restore_runtime_config
    ):
        settings = self._bind(deep_model="gpt-5.4")

        with pytest.raises(BindingConfigurationError) as exc_info:
            resolve_binding_plan(settings)

        joined = "\n".join(exc_info.value.errors)
        assert "belongs to 'openai', not 'google'" in joined

    def test_absent_flags_leave_resolution_untouched(self, restore_runtime_config):
        settings = self._bind()
        plan = resolve_binding_plan(settings)

        assert plan.bindings[SeatId.MARKET].model == settings.google_llm_fast_model
        assert plan.bindings[SeatId.BULL].model == settings.google_llm_reasoning_model
