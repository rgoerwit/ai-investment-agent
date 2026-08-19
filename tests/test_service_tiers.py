"""Unit tests for src/service_tiers.py — flex-tier gating, timeout floors,
and the dynamic (error-driven) flex-capability cache."""

import pathlib
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
    resolve_google_service_tier,
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


class TestFlexHealthDowngrade:
    """Provider-scoped flex-health cache — the memory the capability cache lacks.

    Measured on 8002.T (2026-08-14): a 134-minute run, ~121 of them waiting on
    four queued flex calls that each burned the 900 s flex floor before falling
    back. Nothing remembered the first timeout, so the run re-learned the same
    outage four times, and the fallbacks billed at the standard rate -- $0.90
    against a $0.48-0.66 norm.

    All timing is injected via ``now`` so nothing sleeps.
    """

    T0 = 1000.0

    @staticmethod
    def _cfg(**over):
        from src.config import Settings

        base = {
            "flex_degrade_enabled": True,
            "flex_degrade_threshold": 2,
            "flex_degrade_window_seconds": 900.0,
            "flex_degrade_cool_off_seconds": 1800.0,
        }
        base.update(over)
        return Settings(_env_file=None, **base)

    def _fail(self, provider, *, at, cfg, reason="latency"):
        from src.service_tiers import note_flex_fallback

        note_flex_fallback(provider, reason=reason, now=at, cfg=cfg)

    def test_threshold_failures_degrade_the_provider(self):
        from src.service_tiers import flex_degraded

        cfg = self._cfg()
        self._fail("google", at=self.T0, cfg=cfg)
        assert flex_degraded("google", now=self.T0 + 1, cfg=cfg) is False
        self._fail("google", at=self.T0 + 60, cfg=cfg)
        assert flex_degraded("google", now=self.T0 + 61, cfg=cfg) is True

    def test_capacity_and_latency_both_count(self):
        """Both mean 'the flex pool did not serve this call'."""
        from src.service_tiers import flex_degraded

        cfg = self._cfg()
        self._fail("google", at=self.T0, cfg=cfg, reason="capacity")
        self._fail("google", at=self.T0 + 5, cfg=cfg, reason="latency")
        assert flex_degraded("google", now=self.T0 + 6, cfg=cfg) is True

    def test_failures_older_than_the_window_age_out(self):
        from src.service_tiers import flex_degraded

        cfg = self._cfg()
        self._fail("google", at=self.T0, cfg=cfg)
        # Second failure lands after the window has slid past the first.
        self._fail("google", at=self.T0 + 1000, cfg=cfg)
        assert flex_degraded("google", now=self.T0 + 1001, cfg=cfg) is False

    def test_providers_are_independent(self):
        """Google congestion must not downgrade the OpenAI review plane."""
        from src.service_tiers import flex_degraded

        cfg = self._cfg()
        self._fail("google", at=self.T0, cfg=cfg)
        self._fail("google", at=self.T0 + 1, cfg=cfg)
        assert flex_degraded("google", now=self.T0 + 2, cfg=cfg) is True
        assert flex_degraded("openai", now=self.T0 + 2, cfg=cfg) is False

    def test_cool_off_expiry_restores_eligibility(self):
        from src.service_tiers import flex_degraded

        cfg = self._cfg()
        self._fail("google", at=self.T0, cfg=cfg)
        self._fail("google", at=self.T0 + 1, cfg=cfg)
        assert flex_degraded("google", now=self.T0 + 1799, cfg=cfg) is True
        assert flex_degraded("google", now=self.T0 + 1802, cfg=cfg) is False

    def test_single_failure_on_probation_re_degrades(self):
        """Without this, a sustained outage re-pays threshold x 900 s per cycle."""
        from src.service_tiers import flex_degraded

        cfg = self._cfg()
        self._fail("google", at=self.T0, cfg=cfg)
        self._fail("google", at=self.T0 + 1, cfg=cfg)
        assert (
            flex_degraded("google", now=self.T0 + 2000, cfg=cfg) is False
        )  # probation
        self._fail("google", at=self.T0 + 2001, cfg=cfg)
        assert flex_degraded("google", now=self.T0 + 2002, cfg=cfg) is True

    def test_failures_during_degradation_do_not_extend_it(self):
        """An in-flight call that already requested flex must not re-arm the clock."""
        from src.service_tiers import flex_degraded

        cfg = self._cfg()
        self._fail("google", at=self.T0, cfg=cfg)
        self._fail("google", at=self.T0 + 1, cfg=cfg)
        self._fail("google", at=self.T0 + 900, cfg=cfg)
        assert flex_degraded("google", now=self.T0 + 1802, cfg=cfg) is False

    @pytest.mark.parametrize("provider", [None, "", "   "])
    def test_unknown_provider_is_inert(self, provider):
        from src.service_tiers import flex_degraded

        cfg = self._cfg()
        self._fail(provider, at=self.T0, cfg=cfg)
        self._fail(provider, at=self.T0 + 1, cfg=cfg)
        assert flex_degraded(provider, now=self.T0 + 2, cfg=cfg) is False

    def test_disabled_switch_makes_it_a_no_op(self):
        from src.service_tiers import flex_degraded

        cfg = self._cfg(flex_degrade_enabled=False)
        self._fail("google", at=self.T0, cfg=cfg)
        self._fail("google", at=self.T0 + 1, cfg=cfg)
        assert flex_degraded("google", now=self.T0 + 2, cfg=cfg) is False

    def test_snapshot_is_secret_free_and_serializable(self):
        """It rides into a persisted artifact, so it must survive json.dumps."""
        import json

        from src.service_tiers import flex_degradation_snapshot

        cfg = self._cfg()
        assert flex_degradation_snapshot(now=self.T0) == {}
        self._fail("google", at=self.T0, cfg=cfg)
        self._fail("google", at=self.T0 + 1, cfg=cfg)
        snap = flex_degradation_snapshot(now=self.T0 + 2)
        assert snap["google"]["episodes"] == 1
        assert snap["google"]["degraded"] is True
        json.dumps(snap)

    def test_concurrent_failures_stay_consistent(self):
        import threading

        from src.service_tiers import flex_degraded

        cfg = self._cfg(flex_degrade_threshold=50)
        barrier = threading.Barrier(8)

        def worker(index: int) -> None:
            barrier.wait()
            for step in range(10):
                self._fail("google", at=self.T0 + index * 10 + step, cfg=cfg)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # 80 failures against a threshold of 50, all inside the window.
        assert flex_degraded("google", now=self.T0 + 200, cfg=cfg) is True


class TestGoogleServiceTierResolution:
    """One resolver, so what we request and what we allow for cannot diverge.

    ``GOOGLE_SERVICE_TIER`` (multi-provider) and ``GEMINI_SERVICE_TIER``
    (legacy) name the same concept. Seat construction read the first while this
    module read the second, so an operator who set only ``GOOGLE_SERVICE_TIER``
    sent flex requests that the runtime did not treat as flex: the timeout
    floors never applied and the degradation cache never engaged. Same class of
    split that ``RuntimeConfig.from_config`` already resolves for RPM.
    """

    @staticmethod
    def _schema_cfg(*, google="standard", gemini="standard", base_provider=None):
        return SimpleNamespace(
            google_service_tier=google,
            gemini_service_tier=gemini,
            llm_base_provider=base_provider,
            openai_service_tier="auto",
            flex_llm_timeout_seconds=900,
            quick_llm_call_hard_timeout_seconds=60,
        )

    def test_new_schema_reads_the_provider_scoped_key(self):
        cfg = self._schema_cfg(google="flex", gemini="standard", base_provider="google")
        assert resolve_google_service_tier(cfg) == "flex"
        assert gemini_flex_active(cfg)
        assert provider_flex_active("google", cfg)

    def test_legacy_schema_reads_the_legacy_key(self):
        cfg = self._schema_cfg(google="standard", gemini="flex", base_provider=None)
        assert resolve_google_service_tier(cfg) == "flex"
        assert gemini_flex_active(cfg)

    def test_new_schema_ignores_a_stale_legacy_key(self):
        """A leftover GEMINI_SERVICE_TIER must not re-enable flex."""
        cfg = self._schema_cfg(google="standard", gemini="flex", base_provider="google")
        assert resolve_google_service_tier(cfg) == "standard"
        assert not gemini_flex_active(cfg)

    def test_legacy_schema_ignores_the_provider_scoped_key(self):
        cfg = self._schema_cfg(google="flex", gemini="standard", base_provider=None)
        assert resolve_google_service_tier(cfg) == "standard"
        assert not gemini_flex_active(cfg)

    def test_missing_fields_default_to_standard(self):
        assert resolve_google_service_tier(SimpleNamespace()) == "standard"

    def test_seat_construction_does_not_read_the_raw_field(self):
        """Construction must go through the resolver, not the Settings field.

        A direct ``settings.google_service_tier`` read there is exactly how the
        two sites drifted apart; a source scan is the only thing that catches
        it reappearing, since both spellings behave identically in the common
        case where the keys agree.
        """
        source = (
            pathlib.Path(__file__).resolve().parents[1]
            / "src"
            / "llm_runtime"
            / "construction.py"
        ).read_text()
        assert "resolve_google_service_tier(" in source
        assert "google_service_tier" not in source.replace(
            "resolve_google_service_tier", ""
        ), (
            "construction.py reads the raw google_service_tier field; route it "
            "through resolve_google_service_tier so the runtime flex gate and "
            "the outgoing request agree."
        )
