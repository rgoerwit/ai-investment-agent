"""Tests for the APEX-tier LLM factory in src/llms.py.

The APEX tier is one pin (APEX_MODEL / APEX_QUICK_MODEL / APEX_THINKING_LEVEL)
covering the two gate-critical seats: Senior Fundamentals (DATA_BLOCK rubric
arithmetic feeding the hard <50% gates) and the Portfolio Manager (gate checks,
override logic, PM_BLOCK contract).

Behavior matrix under test:
- APEX_MODEL unset → legacy per-seat tiers (senior: quick; PM: deep in full
  mode, quick in --quick).
- APEX_MODEL set, full mode → both seats get the apex model with deep-tier
  settings (temp 0.1, APEX_THINKING_LEVEL, reserve deep).
- APEX_MODEL set, --quick → APEX_QUICK_MODEL when provided, else the plain
  quick floor (accepted degradation; quick mode stays cheap).
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


class TestApexFactoryUnset:
    """APEX_MODEL unset → legacy per-seat behavior, byte-identical pipeline."""

    def test_senior_full_mode_delegates_to_quick_path(self, monkeypatch):
        _set_cfg(monkeypatch, apex_model="")
        sentinel = object()
        with patch.object(
            llms_mod, "create_quick_thinking_llm", return_value=sentinel
        ) as quick:
            result = llms_mod.create_apex_llm(
                "senior_fundamentals",
                quick_mode=False,
                callbacks=[],
                max_output_tokens=1234,
            )
        assert result is sentinel
        quick.assert_called_once_with(callbacks=[], max_output_tokens=1234)

    def test_pm_full_mode_delegates_to_deep_path(self, monkeypatch):
        _set_cfg(monkeypatch, apex_model="")
        sentinel = object()
        with patch.object(
            llms_mod, "create_deep_thinking_llm", return_value=sentinel
        ) as deep:
            result = llms_mod.create_apex_llm(
                "portfolio_manager",
                quick_mode=False,
                callbacks=[],
                max_output_tokens=4321,
            )
        assert result is sentinel
        deep.assert_called_once_with(callbacks=[], max_output_tokens=4321)

    def test_pm_quick_mode_delegates_to_quick_path(self, monkeypatch):
        _set_cfg(monkeypatch, apex_model="")
        sentinel = object()
        with patch.object(
            llms_mod, "create_quick_thinking_llm", return_value=sentinel
        ) as quick:
            result = llms_mod.create_apex_llm("portfolio_manager", quick_mode=True)
        assert result is sentinel
        quick.assert_called_once()

    def test_senior_quick_mode_delegates_to_quick_path(self, monkeypatch):
        _set_cfg(monkeypatch, apex_model="")
        with patch.object(llms_mod, "create_quick_thinking_llm") as quick:
            llms_mod.create_apex_llm("senior_fundamentals", quick_mode=True)
        quick.assert_called_once()


class TestApexFactorySet:
    """APEX_MODEL set → dedicated construction per the behavior matrix."""

    @pytest.mark.parametrize("seat", ["senior_fundamentals", "portfolio_manager"])
    def test_full_mode_uses_apex_model_with_deep_tier_settings(self, monkeypatch, seat):
        _set_cfg(
            monkeypatch,
            apex_model="gemini-3.1-pro-preview",
            apex_thinking_level="high",
        )
        sentinel = object()
        with patch.object(
            llms_mod, "create_gemini_model", return_value=sentinel
        ) as gemini:
            result = llms_mod.create_apex_llm(
                seat, quick_mode=False, callbacks=[], max_output_tokens=10923
            )
        assert result is sentinel
        args, kwargs = gemini.call_args
        assert args[0] == "gemini-3.1-pro-preview"
        assert kwargs["thinking_level"] == "high"
        assert kwargs["reserve_class"] == "deep"
        assert kwargs["max_output_tokens"] == 10923
        # Deterministic scoring wants the deep-tier temperature, not quick's 0.3
        assert kwargs["temperature"] == 0.1

    @pytest.mark.parametrize("seat", ["senior_fundamentals", "portfolio_manager"])
    def test_quick_mode_without_quick_knob_falls_to_quick_floor(
        self, monkeypatch, seat
    ):
        _set_cfg(
            monkeypatch,
            apex_model="gemini-3.1-pro-preview",
            apex_quick_model="",
        )
        sentinel = object()
        with (
            patch.object(
                llms_mod, "create_quick_thinking_llm", return_value=sentinel
            ) as quick,
            patch.object(llms_mod, "create_gemini_model") as gemini,
        ):
            result = llms_mod.create_apex_llm(seat, quick_mode=True)
        assert result is sentinel
        quick.assert_called_once()
        gemini.assert_not_called()

    def test_quick_mode_with_quick_knob_uses_it_with_apex_thinking(self, monkeypatch):
        _set_cfg(
            monkeypatch,
            apex_model="gemini-3.1-pro-preview",
            apex_quick_model="gemini-3.5-flash",
            apex_thinking_level="high",
        )
        with patch.object(llms_mod, "create_gemini_model") as gemini:
            llms_mod.create_apex_llm("portfolio_manager", quick_mode=True)
        args, kwargs = gemini.call_args
        assert args[0] == "gemini-3.5-flash"
        assert kwargs["thinking_level"] == "high"
        assert kwargs["temperature"] == 0.1

    def test_quick_mode_pins_standard_tier(self, monkeypatch):
        # Gate-critical seat must not queue on best-effort flex under the tight
        # --quick budget: create_gemini_model gets service_tier="standard".
        _set_cfg(
            monkeypatch,
            apex_model="gemini-3.1-pro-preview",
            apex_quick_model="gemini-3.5-flash",
            apex_thinking_level="high",
        )
        with patch.object(llms_mod, "create_gemini_model") as gemini:
            llms_mod.create_apex_llm("portfolio_manager", quick_mode=True)
        assert gemini.call_args.kwargs["service_tier"] == "standard"

    def test_full_mode_leaves_tier_to_config(self, monkeypatch):
        # Full mode keeps config-driven flex (service_tier=None follows config).
        _set_cfg(
            monkeypatch,
            apex_model="gemini-3.1-pro-preview",
            apex_thinking_level="high",
        )
        with patch.object(llms_mod, "create_gemini_model") as gemini:
            llms_mod.create_apex_llm("senior_fundamentals", quick_mode=False)
        assert gemini.call_args.kwargs["service_tier"] is None

    def test_configured_thinking_level_is_passed_through(self, monkeypatch):
        _set_cfg(
            monkeypatch,
            apex_model="gemini-3.5-flash",
            apex_thinking_level="medium",
        )
        with patch.object(llms_mod, "create_gemini_model") as gemini:
            llms_mod.create_apex_llm("senior_fundamentals", quick_mode=False)
        assert gemini.call_args.kwargs["thinking_level"] == "medium"

    def test_non_thinking_model_strips_level_and_warns(self, monkeypatch):
        _set_cfg(
            monkeypatch,
            apex_model="gemini-2.0-flash",
            apex_thinking_level="high",
        )
        with patch.object(llms_mod, "create_gemini_model") as gemini:
            with patch.object(llms_mod, "logger") as logger:
                llms_mod.create_apex_llm("portfolio_manager", quick_mode=False)
        assert gemini.call_args.kwargs["thinking_level"] is None
        logger.warning.assert_any_call(
            "apex_model_no_thinking_support",
            seat="portfolio_manager",
            model="gemini-2.0-flash",
        )

    def test_unknown_seat_raises(self):
        with pytest.raises(ValueError, match="unknown apex seat"):
            llms_mod.create_apex_llm("trader", quick_mode=False)


class TestSettingsFields:
    def test_env_overrides_are_honored(self, monkeypatch):
        monkeypatch.setenv("APEX_MODEL", "gemini-3.1-pro-preview")
        monkeypatch.setenv("APEX_QUICK_MODEL", "gemini-3.5-flash")
        monkeypatch.setenv("APEX_THINKING_LEVEL", "medium")
        settings = Settings()
        assert settings.apex_model == "gemini-3.1-pro-preview"
        assert settings.apex_quick_model == "gemini-3.5-flash"
        assert settings.apex_thinking_level == "medium"

    def test_defaults_preserve_legacy_behavior(self, monkeypatch):
        monkeypatch.setenv("APEX_MODEL", "")
        monkeypatch.setenv("APEX_QUICK_MODEL", "")
        monkeypatch.delenv("APEX_THINKING_LEVEL", raising=False)
        # _env_file=None: an operator .env pinning the thinking level must not
        # flip the default assertion (shell delenv can't mask the .env file).
        settings = Settings(_env_file=None)
        assert not settings.apex_model
        assert not settings.apex_quick_model
        assert settings.apex_thinking_level == "high"

    def test_invalid_thinking_level_rejected(self, monkeypatch):
        monkeypatch.setenv("APEX_THINKING_LEVEL", "xhigh")
        with pytest.raises(ValueError):
            Settings()

    def test_retired_env_vars_warn(self, monkeypatch):
        import src.config as config_module

        monkeypatch.setenv("SENIOR_FUNDAMENTALS_MODEL", "gemini-3.5-flash")
        with patch.object(config_module, "logger") as logger:
            config_module._warn_retired_env_vars()
        warned = " ".join(str(c.args[0]) for c in logger.warning.call_args_list)
        assert "SENIOR_FUNDAMENTALS_MODEL" in warned
        assert "APEX_MODEL" in warned

    def test_no_retired_warning_when_unset(self, monkeypatch):
        import src.config as config_module

        monkeypatch.delenv("SENIOR_FUNDAMENTALS_MODEL", raising=False)
        monkeypatch.delenv("SENIOR_FUNDAMENTALS_THINKING_LEVEL", raising=False)
        monkeypatch.setattr(config_module, "_cached_env_file_values", dict)
        with patch.object(config_module, "logger") as logger:
            config_module._warn_retired_env_vars()
        logger.warning.assert_not_called()


class TestGraphWiring:
    def test_components_shim_routes_to_llms_factory(self, monkeypatch):
        import src.graph.components as components_mod

        _set_cfg(monkeypatch, apex_model="")
        calls: list[tuple] = []

        def fake_apex(seat, **kwargs):
            calls.append((seat, kwargs))
            return object()

        with patch.object(llms_mod, "create_apex_llm", side_effect=fake_apex):
            components_mod.create_apex_llm(
                "senior_fundamentals", quick_mode=False, callbacks=[]
            )
        assert calls and calls[0][0] == "senior_fundamentals"

    def test_build_routes_both_apex_seats_through_factory(self):
        """Assembly routes both gate-critical seats through the seat factory."""
        import inspect

        import src.graph.components as components_mod

        source = inspect.getsource(components_mod)
        assert "senior_fund_llm = seat_model(SeatId.SENIOR_FUNDAMENTALS)" in source
        assert "pm_llm = seat_model(SeatId.PORTFOLIO_MANAGER)" in source
        assert "legacy_builder=_build_legacy_seat_model" in source
        assert "LegacyGraphFactories(" in source
        assert "pm_llm = create_deep_thinking_llm(" not in source
        assert "pm_llm = create_quick_thinking_llm(" not in source
