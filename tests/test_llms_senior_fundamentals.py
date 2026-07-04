"""Tests for the dedicated Senior Fundamentals LLM factory in src/llms.py.

Covers: legacy fallback to the quick-thinking path when the knob is unset,
dedicated deep-tier construction when set (model, thinking level, temperature,
reserve class), thinking-level stripping for non-thinking models, and the
graph-assembly wiring in src/graph/components.py.
"""

from unittest.mock import patch

import pytest

import src.llms as llms_mod
from src.config import Settings, config


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


class TestSeniorFundamentalsFactory:
    def test_unset_knob_delegates_to_quick_path(self, monkeypatch):
        _set_cfg(monkeypatch, senior_fundamentals_model=None)
        sentinel = object()
        with patch.object(
            llms_mod, "create_quick_thinking_llm", return_value=sentinel
        ) as quick:
            result = llms_mod.create_senior_fundamentals_llm(
                callbacks=[], max_output_tokens=1234
            )
        assert result is sentinel
        quick.assert_called_once_with(callbacks=[], max_output_tokens=1234)

    def test_empty_string_knob_treated_as_unset(self, monkeypatch):
        _set_cfg(monkeypatch, senior_fundamentals_model="")
        sentinel = object()
        with patch.object(
            llms_mod, "create_quick_thinking_llm", return_value=sentinel
        ) as quick:
            result = llms_mod.create_senior_fundamentals_llm()
        assert result is sentinel
        quick.assert_called_once()

    def test_dedicated_model_uses_deep_tier_settings(self, monkeypatch):
        _set_cfg(
            monkeypatch,
            senior_fundamentals_model="gemini-3.5-flash",
            senior_fundamentals_thinking_level="high",
        )
        sentinel = object()
        with patch.object(
            llms_mod, "create_gemini_model", return_value=sentinel
        ) as gemini:
            result = llms_mod.create_senior_fundamentals_llm(
                callbacks=[], max_output_tokens=10923
            )
        assert result is sentinel
        gemini.assert_called_once()
        args, kwargs = gemini.call_args
        assert args[0] == "gemini-3.5-flash"
        assert kwargs["thinking_level"] == "high"
        assert kwargs["reserve_class"] == "deep"
        assert kwargs["max_output_tokens"] == 10923
        # Deterministic scoring wants the deep-tier temperature, not quick's 0.3
        assert kwargs["temperature"] == 0.1

    def test_configured_thinking_level_is_passed_through(self, monkeypatch):
        _set_cfg(
            monkeypatch,
            senior_fundamentals_model="gemini-3.5-flash",
            senior_fundamentals_thinking_level="medium",
        )
        with patch.object(llms_mod, "create_gemini_model") as gemini:
            llms_mod.create_senior_fundamentals_llm()
        assert gemini.call_args.kwargs["thinking_level"] == "medium"

    def test_non_thinking_model_strips_level_and_warns(self, monkeypatch):
        _set_cfg(
            monkeypatch,
            senior_fundamentals_model="gemini-2.0-flash",
            senior_fundamentals_thinking_level="high",
        )
        with patch.object(llms_mod, "create_gemini_model") as gemini:
            with patch.object(llms_mod, "logger") as logger:
                llms_mod.create_senior_fundamentals_llm()
        assert gemini.call_args.kwargs["thinking_level"] is None
        logger.warning.assert_any_call(
            "senior_fund_model_no_thinking_support", model="gemini-2.0-flash"
        )


class TestSettingsFields:
    def test_env_override_is_honored(self, monkeypatch):
        monkeypatch.setenv("SENIOR_FUNDAMENTALS_MODEL", "gemini-3.5-flash")
        monkeypatch.setenv("SENIOR_FUNDAMENTALS_THINKING_LEVEL", "medium")
        settings = Settings()
        assert settings.senior_fundamentals_model == "gemini-3.5-flash"
        assert settings.senior_fundamentals_thinking_level == "medium"

    def test_defaults_preserve_legacy_behavior(self, monkeypatch):
        monkeypatch.setenv("SENIOR_FUNDAMENTALS_MODEL", "")
        monkeypatch.delenv("SENIOR_FUNDAMENTALS_THINKING_LEVEL", raising=False)
        settings = Settings()
        assert not settings.senior_fundamentals_model
        assert settings.senior_fundamentals_thinking_level == "high"

    def test_invalid_thinking_level_rejected(self, monkeypatch):
        monkeypatch.setenv("SENIOR_FUNDAMENTALS_THINKING_LEVEL", "maximal")
        with pytest.raises(ValueError):
            Settings()


class TestGraphWiring:
    def test_senior_uses_dedicated_factory_junior_stays_quick(self, monkeypatch):
        import src.graph.components as components_mod

        _set_cfg(
            monkeypatch,
            senior_fundamentals_model="gemini-3.5-flash",
            senior_fundamentals_thinking_level="high",
        )
        calls: dict[str, object] = {}

        def fake_senior(**kwargs):
            calls["senior"] = kwargs
            return object()

        def fake_quick(**kwargs):
            calls.setdefault("quick", []).append(kwargs)  # type: ignore[union-attr]
            return object()

        with (
            patch.object(
                llms_mod, "create_senior_fundamentals_llm", side_effect=fake_senior
            ),
            patch.object(llms_mod, "create_quick_thinking_llm", side_effect=fake_quick),
        ):
            # The components-module shims lazily import from src.llms, so
            # patching llms_mod is sufficient; call the shims directly.
            components_mod.create_senior_fundamentals_llm(
                callbacks=[], max_output_tokens=10923
            )
            components_mod.create_quick_thinking_llm(
                callbacks=[], max_output_tokens=10923
            )

        assert "senior" in calls
        assert "quick" in calls

    def test_build_uses_senior_factory_for_fundamentals_analyst(self):
        """The assembly source must route Senior through the dedicated factory."""
        import inspect

        import src.graph.components as components_mod

        source = inspect.getsource(components_mod)
        assert "senior_fund_llm = create_senior_fundamentals_llm(" in source
