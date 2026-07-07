"""Unit tests for src/service_tiers.py — flex-tier gating, timeout floors,
and the dynamic (error-driven) flex-capability cache."""

from types import SimpleNamespace

import pytest

from src.config import config as global_config
from src.runtime_config import RuntimeConfig, use_runtime_config
from src.service_tiers import (
    FLEX_QUICK_ATTEMPT_FRACTION,
    _reset_flex_capability_cache_for_tests,
    _reset_floor_log_cache_for_tests,
    flex_attempt_client_timeout,
    floor_llm_hard_timeout,
    floor_llm_timeout,
    floor_llm_total_timeout,
    gemini_flex_active,
    is_flex_unsupported,
    is_flex_unsupported_error,
    mark_flex_unsupported,
    normalize_model_name,
    openai_flex_active,
    provider_flex_active,
)


def _cfg(gemini="standard", openai="auto", floor=900, quick_hard_cap=60):
    return SimpleNamespace(
        gemini_service_tier=gemini,
        openai_service_tier=openai,
        flex_llm_timeout_seconds=floor,
        quick_llm_call_hard_timeout_seconds=quick_hard_cap,
    )


def _quick_runtime():
    """A bound RuntimeConfig with quick mode active (source of truth for the
    quick-mode floor bypass; quick_mode_active is run-scoped, not on cfg)."""
    return RuntimeConfig.from_config(global_config).with_overrides(
        quick_mode_active=True
    )


@pytest.fixture(autouse=True)
def _reset_tier_state():
    _reset_floor_log_cache_for_tests()
    _reset_flex_capability_cache_for_tests()
    yield
    _reset_floor_log_cache_for_tests()
    _reset_flex_capability_cache_for_tests()


class TestFlexActivation:
    def test_defaults_are_off(self):
        cfg = _cfg()
        assert not gemini_flex_active(cfg)
        assert not openai_flex_active(cfg)

    def test_gemini_flex_on(self):
        cfg = _cfg(gemini="flex")
        assert gemini_flex_active(cfg)
        assert provider_flex_active("google", cfg)
        assert not provider_flex_active("openai", cfg)

    def test_openai_flex_on(self):
        cfg = _cfg(openai="flex")
        assert openai_flex_active(cfg)
        assert provider_flex_active("openai", cfg)
        assert not provider_flex_active("google", cfg)

    def test_unknown_provider_never_flex(self):
        cfg = _cfg(gemini="flex", openai="flex")
        assert not provider_flex_active("anthropic", cfg)
        assert not provider_flex_active(None, cfg)


class TestTimeoutFloors:
    def test_no_floor_when_standard(self):
        cfg = _cfg()
        assert floor_llm_timeout(120.0, provider="google", cfg=cfg) == 120.0
        assert floor_llm_hard_timeout(60.0, provider="google", cfg=cfg) == 60.0
        assert floor_llm_total_timeout(35.0, provider="openai", cfg=cfg) == 35.0

    def test_per_call_floor(self):
        cfg = _cfg(gemini="flex")
        assert floor_llm_timeout(120.0, provider="google", cfg=cfg) == 900.0

    def test_hard_cap_floor_is_1_5x(self):
        cfg = _cfg(gemini="flex")
        assert floor_llm_hard_timeout(120.0, provider="google", cfg=cfg) == 1350.0

    def test_total_budget_floor_is_2x(self):
        cfg = _cfg(openai="flex")
        assert floor_llm_total_timeout(240.0, provider="openai", cfg=cfg) == 1800.0

    def test_floor_is_provider_scoped(self):
        # Gemini flex must not stretch OpenAI ceilings and vice versa
        cfg = _cfg(gemini="flex")
        assert floor_llm_timeout(120.0, provider="openai", cfg=cfg) == 120.0
        cfg = _cfg(openai="flex")
        assert floor_llm_hard_timeout(60.0, provider="google", cfg=cfg) == 60.0

    def test_larger_configured_value_wins(self):
        cfg = _cfg(gemini="flex", floor=900)
        assert floor_llm_timeout(1800.0, provider="google", cfg=cfg) == 1800.0
        assert floor_llm_hard_timeout(2000.0, provider="google", cfg=cfg) == 2000.0

    def test_custom_floor_setting(self):
        cfg = _cfg(gemini="flex", floor=600)
        assert floor_llm_timeout(120.0, provider="google", cfg=cfg) == 600.0
        assert floor_llm_hard_timeout(120.0, provider="google", cfg=cfg) == 900.0


class TestFlexCapabilityCache:
    def test_normalize_strips_path_prefix(self):
        assert normalize_model_name("models/gemini-3.5-flash") == "gemini-3.5-flash"
        assert normalize_model_name("gpt-5.4") == "gpt-5.4"

    def test_mark_and_lookup(self):
        assert not is_flex_unsupported("gemini-3.5-flash")
        mark_flex_unsupported("models/gemini-3.5-flash")
        assert is_flex_unsupported("gemini-3.5-flash")
        assert is_flex_unsupported("models/gemini-3.5-flash")
        assert not is_flex_unsupported("gemini-3.1-pro-preview")

    def test_mark_is_idempotent(self):
        mark_flex_unsupported("gpt-5.4")
        mark_flex_unsupported("gpt-5.4")
        assert is_flex_unsupported("gpt-5.4")


class TestFlexUnsupportedErrorDetection:
    def test_400_naming_service_tier(self):
        exc = ValueError("400 INVALID_ARGUMENT: service_tier is not supported")
        assert is_flex_unsupported_error(exc)

    def test_unsupported_prose_variant(self):
        exc = ValueError("The requested service tier is not supported for this model.")
        assert is_flex_unsupported_error(exc)

    def test_status_code_attribute(self):
        exc = ValueError("bad service_tier value")
        exc.status_code = 400  # type: ignore[attr-defined]
        assert is_flex_unsupported_error(exc)

    def test_capacity_429_is_not_capability(self):
        exc = ValueError("429 RESOURCE_EXHAUSTED: flex capacity unavailable")
        assert not is_flex_unsupported_error(exc)

    def test_unrelated_400_is_not_capability(self):
        exc = ValueError("400 INVALID_ARGUMENT: contents must not be empty")
        assert not is_flex_unsupported_error(exc)


class TestQuickModeFloorBypass:
    """Fix A1: in quick mode the flex timeout floors must NOT stretch the
    deadline — a floor longer than the pipeline watchdog converts a graceful
    in-process timeout into a hard SIGTERM that discards the ticker."""

    def test_full_mode_floors_apply(self):
        cfg = _cfg(gemini="flex", openai="flex")
        # No quick RuntimeConfig bound => full mode => floors apply.
        assert floor_llm_timeout(120.0, provider="google", cfg=cfg) == 900.0
        assert floor_llm_hard_timeout(120.0, provider="google", cfg=cfg) == 1350.0
        assert floor_llm_total_timeout(240.0, provider="openai", cfg=cfg) == 1800.0

    def test_quick_mode_bypasses_all_three_floors(self):
        cfg = _cfg(gemini="flex", openai="flex")
        with use_runtime_config(_quick_runtime()):
            assert floor_llm_timeout(120.0, provider="google", cfg=cfg) == 120.0
            assert floor_llm_hard_timeout(120.0, provider="google", cfg=cfg) == 120.0
            assert floor_llm_total_timeout(240.0, provider="openai", cfg=cfg) == 240.0

    def test_quick_mode_uses_contextvar_not_partial_cfg(self):
        # The tier cfg has no quick_mode_active field; the bypass must key off
        # the bound RuntimeConfig, and must not crash on a partial cfg.
        cfg = _cfg(gemini="flex")
        with use_runtime_config(_quick_runtime()):
            assert floor_llm_timeout(120.0, provider="google", cfg=cfg) == 120.0
        # Unbound again => full mode => floor returns.
        assert floor_llm_timeout(120.0, provider="google", cfg=cfg) == 900.0


class TestFlexAttemptClientTimeout:
    """Fix A3 step 1: the flex-attempt SDK client timeout. Full mode floors up
    (queued flex calls survive); quick mode caps below the outer hard cap so a
    queued call raises in time for a standard re-issue within the same wrapper."""

    def test_noop_when_flex_off(self):
        cfg = _cfg()  # standard/auto
        assert flex_attempt_client_timeout(120.0, provider="google", cfg=cfg) == 120.0

    def test_full_mode_floors_up(self):
        cfg = _cfg(gemini="flex", floor=900)
        assert flex_attempt_client_timeout(120.0, provider="google", cfg=cfg) == 900.0

    def test_quick_mode_caps_below_outer_hard_cap(self):
        cfg = _cfg(gemini="flex", quick_hard_cap=60)
        with use_runtime_config(_quick_runtime()):
            got = flex_attempt_client_timeout(120.0, provider="google", cfg=cfg)
        assert got == pytest.approx(60 * FLEX_QUICK_ATTEMPT_FRACTION)
        assert got < 60  # strictly below the quick outer hard cap

    def test_quick_mode_never_raises_base(self):
        # A base already below the cap must not be inflated.
        cfg = _cfg(openai="flex", quick_hard_cap=60)
        with use_runtime_config(_quick_runtime()):
            got = flex_attempt_client_timeout(10.0, provider="openai", cfg=cfg)
        assert got == 10.0

    def test_fraction_leaves_headroom_for_standard_reissue(self):
        # The flex attempt AND its standard re-issue share one outer window, so
        # 2 x T_internal must fit under the outer cap with margin.
        assert 0.0 < FLEX_QUICK_ATTEMPT_FRACTION <= 0.45, (
            "FLEX_QUICK_ATTEMPT_FRACTION must leave >=10% margin after two "
            "attempts (flex + standard re-issue) under the outer hard cap."
        )
