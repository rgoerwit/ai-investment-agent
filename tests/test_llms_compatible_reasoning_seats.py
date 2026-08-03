"""OpenAI-plane seats must bound reasoning on OpenAI-*compatible* endpoints.

The failure this pins down (1088.HK, 2026-08-02): a reasoning model reached
through ``OPENAI_API_BASE`` got no ``reasoning_effort`` and therefore no
completion-cap reserve, so its hidden reasoning consumed the entire 4096-token
consultant budget and the persisted "review" was 46 characters long. Measured
against the live endpoint the same day: unset effort spent 2553 of 4096 tokens
reasoning, ``high`` spent 1434, ``low`` spent 26.

Nothing here is vendor-specific by construction — the seats resolve effort from
the model-family table, so a model the operator swaps in (GPT, Kimi, or a family
added later) is handled by the same code path.
"""

from unittest.mock import MagicMock, patch

import pytest

import src.llms as llms_mod

MOONSHOT_BASE = "https://api.moonshot.ai/v1"
DEFAULT_RESERVE = 2048
DEEP_RESERVE = 8192


@pytest.fixture(autouse=True)
def _reset_warn_state():
    llms_mod._reset_openai_rate_limiter_for_tests()
    yield
    llms_mod._reset_openai_rate_limiter_for_tests()


def _mock_config(*, base_url: str | None = MOONSHOT_BASE, **attrs):
    """A bare-mock config with only the fields these factories read."""
    cfg = MagicMock()
    cfg.enable_consultant = True
    cfg.get_openai_api_key.return_value = "test-key"
    cfg.get_openai_api_base.return_value = base_url
    for key, value in attrs.items():
        setattr(cfg, key, value)
    return cfg


def _build(factory, cfg, **kwargs):
    """Run a seat factory against a fake ChatOpenAI and return (llm, kwargs)."""
    fake_cls = MagicMock()
    fake_cls.return_value = MagicMock()
    with patch("langchain_openai.ChatOpenAI", fake_cls), patch("src.llms.config", cfg):
        llm = factory(**kwargs)
    assert fake_cls.call_args is not None, "factory did not construct a client"
    return llm, fake_cls.call_args.kwargs


class TestConsultantSeat:
    def test_full_mode_bounds_reasoning_and_reserves_headroom(self):
        cfg = _mock_config(consultant_model="kimi-k3")

        llm, kwargs = _build(
            llms_mod.create_consultant_llm, cfg, max_completion_tokens=4096
        )

        assert kwargs["reasoning_effort"] == "high"
        # The visible-output intent is untouched; the reserve is added on top,
        # so reasoning can no longer eat into the 4096 the seat asked for.
        assert llm._configured_max_completion_tokens == 4096
        assert kwargs["max_completion_tokens"] == 4096 + DEEP_RESERVE
        assert llm._configured_reasoning_reserve_tokens == DEEP_RESERVE

    def test_quick_mode_uses_the_cheap_effort_and_default_reserve(self):
        cfg = _mock_config(consultant_model="kimi-k3", consultant_quick_model="kimi-k3")

        llm, kwargs = _build(
            llms_mod.create_consultant_llm,
            cfg,
            quick_mode=True,
            max_completion_tokens=4096,
        )

        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["max_completion_tokens"] == 4096 + DEFAULT_RESERVE
        assert llm._configured_max_completion_tokens == 4096

    def test_gpt5_seat_is_unchanged_by_the_generalization(self):
        cfg = _mock_config(base_url=None, consultant_model="gpt-5.4")

        llm, kwargs = _build(
            llms_mod.create_consultant_llm, cfg, max_completion_tokens=4096
        )

        assert kwargs["reasoning_effort"] == "medium"
        assert kwargs["max_completion_tokens"] == 4096 + DEFAULT_RESERVE
        assert llm._configured_reasoning_reserve_tokens == DEFAULT_RESERVE

    def test_non_reasoning_model_gets_no_parameter_and_no_reserve(self):
        cfg = _mock_config(base_url=None, consultant_model="gpt-4o")

        llm, kwargs = _build(
            llms_mod.create_consultant_llm, cfg, max_completion_tokens=4096
        )

        assert "reasoning_effort" not in kwargs
        assert kwargs["max_completion_tokens"] == 4096
        assert llm._configured_reasoning_reserve_tokens == 0


class TestAuditorSeat:
    def test_reasoning_is_bounded_and_reserved(self):
        cfg = _mock_config(auditor_model="kimi-k3", consultant_model="kimi-k3")

        llm, kwargs = _build(
            llms_mod.create_auditor_llm, cfg, max_completion_tokens=8192
        )

        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["max_completion_tokens"] == 8192 + DEEP_RESERVE
        assert llm._configured_max_completion_tokens == 8192

    def test_escalation_model_resolves_on_its_own_name(self):
        """Escalation may run a different family than the primary auditor."""
        cfg = _mock_config(auditor_model="gpt-4o", consultant_model="gpt-4o")

        _llm, kwargs = _build(
            llms_mod.create_auditor_llm,
            cfg,
            max_completion_tokens=8192,
            model_name_override="kimi-k3",
        )

        assert kwargs["model"] == "kimi-k3"
        assert kwargs["reasoning_effort"] == "high"

    def test_default_budget_matches_the_centralized_share(self):
        """The old 6144/16384 factory defaults were dead and divergent."""
        cfg = _mock_config(auditor_model="gpt-4o", consultant_model="gpt-4o")
        expected = llms_mod._centralized_output_budget("Global Forensic Auditor")

        for quick_mode in (True, False):
            llm, _kwargs = _build(
                llms_mod.create_auditor_llm, cfg, quick_mode=quick_mode
            )
            assert llm._configured_max_completion_tokens == expected


class TestEditorSeat:
    def test_budget_comes_from_the_central_table(self):
        cfg = _mock_config(base_url=None, editor_model="gpt-5.4")

        llm, kwargs = _build(llms_mod.create_editor_llm, cfg)

        # Preserves the historical 8192 at the default 32768 base.
        assert llm._configured_max_completion_tokens == 8192
        assert kwargs["reasoning_effort"] == "medium"

    def test_budget_scales_with_the_configured_base(self):
        cfg = _mock_config(
            base_url=None, editor_model="gpt-5.4", llm_base_output_tokens=65536
        )

        llm, _kwargs = _build(llms_mod.create_editor_llm, cfg)

        assert llm._configured_max_completion_tokens == 16384

    def test_compatible_reasoning_model_is_bounded(self):
        cfg = _mock_config(editor_model="kimi-k3")

        llm, kwargs = _build(llms_mod.create_editor_llm, cfg)

        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["max_completion_tokens"] == 8192 + DEEP_RESERVE
        assert llm._configured_max_completion_tokens == 8192


class TestWriterFallbackSeat:
    def test_prose_tier_prefers_output_budget_over_reasoning(self):
        cfg = _mock_config(editor_model="kimi-k3")

        llm, kwargs = _build(llms_mod.create_writer_openai_fallback_llm, cfg)

        assert kwargs["reasoning_effort"] == "low"
        assert llm._configured_max_completion_tokens == 16384
        assert kwargs["max_completion_tokens"] == 16384 + DEFAULT_RESERVE


class TestApacSeat:
    def test_registered_family_resolves_its_deepest_documented_effort(self):
        cfg = _mock_config(
            enable_apac_specialist=True,
            apac_specialist_model="kimi-k3",
            apac_specialist_base_url=MOONSHOT_BASE,
        )
        cfg.get_apac_specialist_api_key.return_value = "test-key"

        _llm, kwargs = _build(
            llms_mod.create_apac_specialist_llm, cfg, max_completion_tokens=8192
        )

        assert kwargs["reasoning_effort"] == "max"
        assert kwargs["max_completion_tokens"] == 8192 + DEEP_RESERVE

    def test_unregistered_family_keeps_the_legacy_literal(self):
        """z.ai/DeepSeek behaviour must stay byte-identical."""
        cfg = _mock_config(
            enable_apac_specialist=True,
            apac_specialist_model="glm-5.2",
            apac_specialist_base_url="https://api.z.ai/api/paas/v4",
        )
        cfg.get_apac_specialist_api_key.return_value = "test-key"

        _llm, kwargs = _build(
            llms_mod.create_apac_specialist_llm, cfg, max_completion_tokens=8192
        )

        assert kwargs["reasoning_effort"] == "max"
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


class TestUnknownFamilyIsSurfaced:
    def test_unregistered_model_on_a_custom_base_warns_once(self):
        cfg = _mock_config(consultant_model="brand-new-reasoner-v1")

        with patch.object(llms_mod.logger, "warning") as warn:
            _build(llms_mod.create_consultant_llm, cfg, max_completion_tokens=4096)
            _build(llms_mod.create_consultant_llm, cfg, max_completion_tokens=4096)

        events = [call.args[0] for call in warn.call_args_list]
        assert events.count("openai_reasoning_capability_unknown") == 1
        logged = next(
            call
            for call in warn.call_args_list
            if call.args[0] == "openai_reasoning_capability_unknown"
        )
        assert logged.kwargs["model"] == "brand-new-reasoner-v1"
        # Log-safe host only — never the full base URL.
        assert logged.kwargs["endpoint_host"] == "api.moonshot.ai"

    def test_stock_openai_models_do_not_warn(self):
        cfg = _mock_config(base_url=None, consultant_model="gpt-4o")

        with patch.object(llms_mod.logger, "warning") as warn:
            _build(llms_mod.create_consultant_llm, cfg, max_completion_tokens=4096)

        events = [call.args[0] for call in warn.call_args_list]
        assert "openai_reasoning_capability_unknown" not in events
