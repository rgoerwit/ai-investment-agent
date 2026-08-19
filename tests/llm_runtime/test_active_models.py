"""What ran, versus what the legacy config fields claim ran.

The multi-provider migration left ``QUICK_MODEL``/``DEEP_MODEL`` unset for every
new-schema operator, so they sit at code defaults. Four consumers reported them
as fact: 25 of 25 post-migration artifacts recorded a ``quick_model`` the run
never invoked. ``active_models`` is the single source those consumers share.
"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.llm_runtime.bindings import (
    ActiveModels,
    BindingConfigurationError,
    active_models,
)


def _new_schema(**overrides) -> Settings:
    base = {
        "LLM_BASE_PROVIDER": "google",
        "GOOGLE_LLM_FAST_MODEL": "gemini-3.1-flash-lite",
        "GOOGLE_LLM_REASONING_MODEL": "gemini-3.7-flash",
        "GOOGLE_LLM_CRITICAL_MODEL": "gemini-3.1-pro-preview",
        "QUICK_MODEL": "gemini-3-flash-preview",  # the misleading legacy default
        "DEEP_MODEL": "gemini-3.1-pro-preview",
    }
    # _env_file=None: Settings() otherwise reads the operator's .env, whose
    # LLM_*_PROVIDER keys would collide with the legacy fields under test.
    return Settings(_env_file=None, **{**base, **overrides})


class TestNewSchema:
    def test_reports_bound_models_not_legacy_defaults(self):
        active = active_models(_new_schema(), quick_mode=False)

        assert active.fast == "gemini-3.1-flash-lite"
        assert active.reasoning == "gemini-3.7-flash"
        assert active.decision == "gemini-3.1-pro-preview"
        # The regression this exists to prevent.
        assert active.fast != "gemini-3-flash-preview"

    def test_quick_mode_reads_quick_bindings(self):
        """A quick seat pin must be visible, not the full-mode model.

        This is the operator's real configuration shape: the two gate-critical
        seats drop to a cheaper flash model under --quick.
        """
        settings = _new_schema(
            LLM_SEAT_QUICK_MODEL_OVERRIDES={"portfolio_manager": "gemini-3.6-flash"}
        )

        assert active_models(settings, quick_mode=True).decision == "gemini-3.6-flash"
        assert (
            active_models(settings, quick_mode=False).decision
            == "gemini-3.1-pro-preview"
        )

    def test_decision_intent_is_the_binding_layer_enum(self):
        active = active_models(_new_schema(), quick_mode=False)
        assert active.decision_intent == "critical"

    def test_intent_is_stable_under_a_seat_model_pin(self):
        """Pinning a seat's model changes the model, never its role."""
        settings = _new_schema(
            LLM_SEAT_QUICK_MODEL_OVERRIDES={"portfolio_manager": "gemini-3.6-flash"}
        )
        assert active_models(settings, quick_mode=True).decision_intent == "critical"


class TestLegacySchema:
    def test_legacy_returns_the_configured_fields_unchanged(self):
        """No LLM_*_PROVIDER set: the legacy fields ARE authoritative."""
        settings = Settings(
            _env_file=None,
            QUICK_MODEL="gemini-2.5-flash",
            DEEP_MODEL="gemini-2.5-pro",
        )

        active = active_models(settings, quick_mode=False)

        assert active == ActiveModels(
            fast="gemini-2.5-flash",
            reasoning="gemini-2.5-pro",
            decision="gemini-2.5-pro",
            decision_intent="reasoning",
        )


class TestFailureModes:
    def test_an_invalid_binding_surfaces_rather_than_reporting_a_blank(self):
        """A broken plan must raise here so the caller can fall back explicitly.

        Returning "" would put an empty model name into the artifact, which is
        the same class of lie as the legacy default.
        """
        with pytest.raises(BindingConfigurationError):
            active_models(
                _new_schema(LLM_BASE_PROVIDER="not-a-real-provider"), quick_mode=False
            )
