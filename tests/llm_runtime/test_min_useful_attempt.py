"""One minimum-useful-attempt rule and one deadline margin, both derived.

A minimum useful attempt is ``MIN_USEFUL_ATTEMPT_FRACTION`` of its plane's
*configured* allowance (``min_useful_seconds``); the margin is
``deadline_margin(cap)``. The LLM, embedding and MCP planes share both. The
memory and MCP planes' boundary tests live beside their planes
(``tests/memory/test_embedding_attempt_deadline.py``,
``tests/mcp/test_mcp_time_left.py``); this file owns the setting's bounds, the
helper, and the LLM policy.

Fractions below are dyadic (0.0625) wherever a test compares at the exact
boundary, so the float arithmetic is exact. No clock but the fake one is read.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import (
    BudgetExhausted,
    TimeoutClass,
    deadline_margin,
    min_useful_seconds,
    request_timeout,
    resolve_budget_policy,
    retry_wait_allowance,
)
from src.runtime_config import RuntimeConfig

FRACTION = 0.0625


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _policy(settings: Settings, *, quick: bool, provider: str | None = None, **kw):
    runtime = RuntimeConfig.from_config(settings).with_overrides(
        quick_mode_active=quick
    )
    return resolve_budget_policy(
        kw.pop("timeout_class", TimeoutClass.STANDARD),
        provider=provider,
        cfg=settings,
        runtime_config=runtime,
        **kw,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "MIN_USEFUL_ATTEMPT_FRACTION",
        "DEADLINE_MARGIN_FRACTION",
        "LLM_DISPATCH_MARGIN_SECONDS",
        "LLM_CALL_HARD_TIMEOUT_SECONDS",
        "QUICK_LLM_CALL_HARD_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def clock(monkeypatch):
    now = [1_000.0]
    monkeypatch.setattr(llm_timeouts, "now", lambda: now[0])
    return lambda: now[0]


# ---------------------------------------------------------------- field bounds


class TestFractionBounds:
    def test_shipped_default(self):
        assert Settings.model_fields["min_useful_attempt_fraction"].default == 0.04

    @pytest.mark.parametrize("value", [1e-9, 0.04, 0.25, 0.5])
    def test_tiny_positive_and_exactly_half_are_accepted(self, value):
        assert (
            _settings(min_useful_attempt_fraction=value).min_useful_attempt_fraction
            == value
        )

    @pytest.mark.parametrize("value", [0.0, -0.04, -1.0, 0.51, 1.0, 5.0, 1e9])
    def test_zero_negative_above_half_and_huge_are_rejected(self, value):
        with pytest.raises(ValidationError):
            _settings(min_useful_attempt_fraction=value)

    @pytest.mark.parametrize("raw", ["0", "-1", "0.51", "abc", "1e9"])
    def test_a_bad_environment_value_is_rejected(self, monkeypatch, raw):
        monkeypatch.setenv("MIN_USEFUL_ATTEMPT_FRACTION", raw)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_the_environment_value_is_read(self, monkeypatch):
        monkeypatch.setenv("MIN_USEFUL_ATTEMPT_FRACTION", "0.1")
        assert Settings(_env_file=None).min_useful_attempt_fraction == 0.1

    @pytest.mark.parametrize(
        "name",
        [
            "llm_min_useful_attempt_seconds",
            "memory_embedding_min_useful_seconds",
            "mcp_retry_min_useful_seconds",
        ],
    )
    def test_the_absolute_minimums_are_gone(self, name):
        assert name not in Settings.model_fields


# ---------------------------------------------------------------------- helper


class TestHelper:
    @pytest.mark.parametrize(
        ("allowance", "expected"),
        [(120.0, 7.5), (30.0, 1.875), (45.0, 2.8125), (0.0, 0.0), (1e6, 62_500.0)],
    )
    def test_is_the_fraction_of_the_allowance(self, allowance, expected):
        cfg = _settings(min_useful_attempt_fraction=FRACTION)
        assert min_useful_seconds(allowance, cfg) == expected

    def test_below_a_positive_allowance_at_the_largest_fraction(self):
        cfg = _settings(min_useful_attempt_fraction=0.5)
        assert min_useful_seconds(0.001, cfg) < 0.001

    def test_without_cfg_it_reads_the_live_settings(self, monkeypatch):
        import src.config as config_module

        monkeypatch.setattr(config_module.config, "min_useful_attempt_fraction", 0.1)
        assert min_useful_seconds(10.0) == pytest.approx(1.0)


# ------------------------------------------------------------------ LLM policy


class TestLlmPolicy:
    def test_quick_and_full_caps_give_their_own_minimums(self):
        settings = _settings(
            min_useful_attempt_fraction=FRACTION,
            quick_llm_call_hard_timeout_seconds=60.0,
            llm_call_hard_timeout_seconds=120.0,
        )
        quick = _policy(settings, quick=True)
        full = _policy(settings, quick=False)
        assert (quick.invocation_cap, quick.min_useful_attempt) == (60.0, 3.75)
        assert (full.invocation_cap, full.min_useful_attempt) == (120.0, 7.5)

    def test_shipped_defaults(self):
        policy = _policy(_settings(), quick=False)
        assert policy.invocation_cap == 120.0
        assert policy.min_useful_attempt == pytest.approx(4.8)
        assert policy.margin == deadline_margin(120.0, _settings())

    def test_the_minimum_is_a_share_of_one_attempt_not_of_the_invocation(self):
        """A 600 s invocation cap must not demand 24 s of a 120 s request."""
        settings = _settings(llm_call_hard_timeout_seconds=600.0)
        policy = _policy(settings, quick=False, provider="openai")
        assert (policy.invocation_cap, policy.request_cap) == (600.0, 120.0)
        assert policy.min_useful_attempt == pytest.approx(4.8)

    def test_the_flex_floor_does_not_inflate_the_minimum(self):
        settings = _settings(
            min_useful_attempt_fraction=FRACTION,
            llm_call_hard_timeout_seconds=120.0,
            gemini_service_tier="flex",
            flex_llm_timeout_seconds=900.0,
        )
        policy = _policy(settings, quick=False, provider="google")
        assert policy.invocation_cap == 1350.0  # floored: 1.5 x 900
        assert policy.min_useful_attempt == 120.0 * FRACTION

    def test_an_override_of_the_fraction_moves_the_minimum(self):
        base = _settings(llm_call_hard_timeout_seconds=100.0)
        doubled = _settings(
            llm_call_hard_timeout_seconds=100.0, min_useful_attempt_fraction=0.08
        )
        assert _policy(base, quick=False).min_useful_attempt == pytest.approx(4.0)
        assert _policy(doubled, quick=False).min_useful_attempt == pytest.approx(8.0)

    def test_a_tiny_cap_with_a_small_fraction_still_admits(self, clock):
        settings = _settings(llm_call_hard_timeout_seconds=0.1)
        policy = _policy(settings, quick=False)
        assert policy.min_useful_attempt == pytest.approx(0.004)
        deadline = clock() + policy.margin + policy.min_useful_attempt
        assert request_timeout(policy, "standard", deadline=deadline) == pytest.approx(
            policy.min_useful_attempt
        )

    def test_remaining_exactly_the_minimum_is_admitted(self, clock):
        policy = _policy(_settings(min_useful_attempt_fraction=FRACTION), quick=False)
        assert (policy.margin, policy.min_useful_attempt) == (2.0, 7.5)
        deadline = clock() + 2.0 + 7.5
        assert request_timeout(policy, "standard", deadline=deadline) == 7.5

    def test_just_below_the_minimum_is_refused(self, clock):
        policy = _policy(_settings(min_useful_attempt_fraction=FRACTION), quick=False)
        deadline = clock() + 2.0 + 7.5 - 0.001
        with pytest.raises(BudgetExhausted) as refused:
            request_timeout(policy, "standard", deadline=deadline)
        assert refused.value.minimum == 7.5

    def test_the_minimum_is_not_shrunk_by_a_short_deadline(self, clock):
        """5 s left of a 120 s cap: a minimum from the remainder (0.31 s) would admit."""
        policy = _policy(_settings(min_useful_attempt_fraction=FRACTION), quick=False)
        with pytest.raises(BudgetExhausted):
            request_timeout(policy, "standard", deadline=clock() + 5.0 + 2.0)

    def test_the_retry_wait_keeps_the_derived_minimum_free(self, clock):
        policy = _policy(_settings(min_useful_attempt_fraction=FRACTION), quick=False)
        # 20 s left, margin 2, minimum 7.5: a 30 s transient backoff is cut to 10.5 s.
        allowance = retry_wait_allowance(
            policy, clock() + 20.0, 30.0, rate_limited=False
        )
        assert allowance == pytest.approx(10.5)
        assert (
            retry_wait_allowance(policy, clock() + 9.4, 1.0, rate_limited=False) is None
        )


# ------------------------------------------------- shipped defaults, pinned

# (class, provider) -> (quick minimum, full minimum) at shipped Settings(): 0.04 x
# min(invocation cap, per-request ceiling). Pinned so a cap/allowance mix-up
# (the minimum taken from the invocation cap) cannot recur silently.
_SHIPPED_MINIMUMS = {
    ("STANDARD", "openai"): (2.4, 4.8),
    ("STANDARD", "google"): (2.4, 4.8),
    ("STANDARD", "anthropic"): (2.4, 4.8),
    ("APEX", "openai"): (4.8, 4.8),
    ("APEX", "google"): (7.2, 4.8),
    ("APEX", "anthropic"): (7.2, 4.8),
    ("CROSS_CHECK", "openai"): (4.8, 4.8),
    ("CROSS_CHECK", "google"): (4.8, 4.8),
    ("CROSS_CHECK", "anthropic"): (7.2, 4.8),
    ("REGIONAL", "openai"): (2.4, 4.8),
    ("REGIONAL", "google"): (2.4, 4.8),
    ("REGIONAL", "anthropic"): (2.4, 4.8),
    ("EDITORIAL", "openai"): (4.8, 4.8),
    ("EDITORIAL", "google"): (12.0, 12.0),
    ("EDITORIAL", "anthropic"): (12.0, 12.0),
    ("OPERATIONAL", "openai"): (2.4, 4.8),
    ("OPERATIONAL", "google"): (2.4, 4.8),
    ("OPERATIONAL", "anthropic"): (2.4, 4.8),
    ("AUXILIARY", "openai"): (2.4, 4.8),
    ("AUXILIARY", "google"): (2.4, 4.8),
    ("AUXILIARY", "anthropic"): (2.4, 4.8),
}


class TestShippedDefaults:
    @pytest.mark.parametrize(("key", "expected"), sorted(_SHIPPED_MINIMUMS.items()))
    def test_every_class_provider_and_mode(self, key, expected):
        name, provider = key
        settings = _settings()
        for quick, minimum in zip((True, False), expected, strict=True):
            policy = _policy(
                settings,
                quick=quick,
                provider=provider,
                timeout_class=TimeoutClass[name],
            )
            assert policy.min_useful_attempt == pytest.approx(minimum)

    def test_the_table_covers_every_class(self):
        assert {name for name, _ in _SHIPPED_MINIMUMS} == {c.name for c in TimeoutClass}

    def test_no_default_minimum_exceeds_the_fraction_of_the_request_ceiling(self):
        """Never above 0.04 x the request ceiling (12 s at the 300 s ceiling)."""
        settings = _settings()
        for name, provider in _SHIPPED_MINIMUMS:
            for quick in (True, False):
                policy = _policy(
                    settings,
                    quick=quick,
                    provider=provider,
                    timeout_class=TimeoutClass[name],
                )
                assert policy.min_useful_attempt <= 0.04 * policy.request_cap + 1e-9


# ---------------------------------------------------------------------- margin


class TestOneMargin:
    @pytest.mark.parametrize("cap", [0.01, 0.5, 1.0, 10.0, 40.0, 60.0, 120.0, 10_000.0])
    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"deadline_margin_fraction": 0.1},
            {"llm_dispatch_margin_seconds": 0.5},
            {"llm_dispatch_margin_seconds": 0.0},
        ],
    )
    def test_the_policy_margin_is_deadline_margin_of_the_cap(self, cap, overrides):
        settings = _settings(llm_call_hard_timeout_seconds=cap, **overrides)
        policy = _policy(settings, quick=False)
        assert policy.invocation_cap == cap
        assert policy.margin == deadline_margin(cap, settings)
        assert policy.margin < cap

    def test_the_margin_follows_the_shared_fraction_setting(self):
        settings = _settings(
            llm_call_hard_timeout_seconds=10.0,
            deadline_margin_fraction=0.1,
            llm_dispatch_margin_seconds=5.0,
        )
        assert _policy(settings, quick=False).margin == pytest.approx(1.0)
