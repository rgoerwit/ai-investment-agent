"""Tests for the P2-7 LLM circuit breaker."""

from __future__ import annotations

import pytest

from src.agents.circuit_breaker import (
    CircuitOpenError,
    LLMCircuitBreaker,
    get_circuit_breaker,
    reset_circuit_breaker_for_tests,
)


def _make() -> LLMCircuitBreaker:
    return LLMCircuitBreaker(
        threshold=3,
        window_seconds=300.0,
        cool_off_seconds=60.0,
    )


_KEY = {
    "agent_name": "Junior Fundamentals Analyst",
    "provider": "google",
    "model_name": "gemini-3.1-flash-lite",
}


# ---- happy path -----------------------------------------------------------


def test_closed_circuit_does_not_block():
    cb = _make()
    cb.before_call(now=100.0, **_KEY)  # no error
    cb.record_outcome(now=101.0, ok=True, **_KEY)
    assert cb.snapshot(**_KEY)["state"] == "closed"


def test_threshold_opens_after_three_timeouts_in_window():
    cb = _make()
    for ts in (10.0, 20.0, 30.0):
        cb.before_call(now=ts, **_KEY)
        cb.record_outcome(now=ts, ok=False, failure_kind="timeout", **_KEY)
    assert cb.snapshot(**_KEY)["state"] == "open"
    with pytest.raises(CircuitOpenError) as info:
        cb.before_call(now=31.0, **_KEY)
    assert info.value.agent_name == _KEY["agent_name"]
    assert info.value.opens_remaining_seconds > 0


# ---- only timeouts count --------------------------------------------------


def test_non_timeout_failures_do_not_open_circuit():
    cb = _make()
    for ts in (10.0, 20.0, 30.0, 40.0):
        cb.before_call(now=ts, **_KEY)
        cb.record_outcome(now=ts, ok=False, failure_kind="application_error", **_KEY)
    assert cb.snapshot(**_KEY)["state"] == "closed"
    # Still allows calls.
    cb.before_call(now=50.0, **_KEY)


def test_mixed_failures_only_timeouts_count():
    cb = _make()
    cb.before_call(now=10.0, **_KEY)
    cb.record_outcome(now=10.0, ok=False, failure_kind="timeout", **_KEY)
    cb.before_call(now=20.0, **_KEY)
    cb.record_outcome(now=20.0, ok=False, failure_kind="rate_limit", **_KEY)
    cb.before_call(now=30.0, **_KEY)
    cb.record_outcome(now=30.0, ok=False, failure_kind="timeout", **_KEY)
    # Two timeouts; below threshold.
    assert cb.snapshot(**_KEY)["state"] == "closed"


# ---- sliding window ------------------------------------------------------


def test_failures_outside_window_do_not_count():
    cb = LLMCircuitBreaker(threshold=3, window_seconds=10.0, cool_off_seconds=60.0)
    cb.before_call(now=0.0, **_KEY)
    cb.record_outcome(now=0.0, ok=False, failure_kind="timeout", **_KEY)
    cb.before_call(now=2.0, **_KEY)
    cb.record_outcome(now=2.0, ok=False, failure_kind="timeout", **_KEY)
    # Old failures (>10s ago) pruned at the third failure.
    cb.before_call(now=15.0, **_KEY)
    cb.record_outcome(now=15.0, ok=False, failure_kind="timeout", **_KEY)
    assert cb.snapshot(**_KEY)["state"] == "closed"
    assert cb.snapshot(**_KEY)["failures_in_window"] == 1


# ---- cool-off + half-open ------------------------------------------------


def test_cool_off_transitions_to_half_open_and_allows_one_probe():
    cb = _make()
    # Open the circuit at t=30.
    for ts in (10.0, 20.0, 30.0):
        cb.before_call(now=ts, **_KEY)
        cb.record_outcome(now=ts, ok=False, failure_kind="timeout", **_KEY)
    # During cool-off, blocked.
    with pytest.raises(CircuitOpenError):
        cb.before_call(now=60.0, **_KEY)
    # After 60s cool-off, half-open: one probe allowed.
    cb.before_call(now=95.0, **_KEY)
    snap = cb.snapshot(**_KEY)
    assert snap["state"] == "half_open"
    assert snap["probe_in_flight"] is True
    # A second concurrent caller during the probe is denied.
    with pytest.raises(CircuitOpenError):
        cb.before_call(now=95.5, **_KEY)


def test_half_open_success_closes_circuit():
    cb = _make()
    for ts in (10.0, 20.0, 30.0):
        cb.before_call(now=ts, **_KEY)
        cb.record_outcome(now=ts, ok=False, failure_kind="timeout", **_KEY)
    cb.before_call(now=95.0, **_KEY)
    cb.record_outcome(now=96.0, ok=True, **_KEY)
    snap = cb.snapshot(**_KEY)
    assert snap["state"] == "closed"
    assert snap["failures_in_window"] == 0


def test_half_open_failure_reopens_circuit():
    cb = _make()
    for ts in (10.0, 20.0, 30.0):
        cb.before_call(now=ts, **_KEY)
        cb.record_outcome(now=ts, ok=False, failure_kind="timeout", **_KEY)
    cb.before_call(now=95.0, **_KEY)
    cb.record_outcome(now=96.0, ok=False, failure_kind="timeout", **_KEY)
    assert cb.snapshot(**_KEY)["state"] == "open"
    # Re-open extends cool-off by another full window.
    with pytest.raises(CircuitOpenError):
        cb.before_call(now=120.0, **_KEY)


# ---- key isolation -------------------------------------------------------


def test_key_isolation_across_agents():
    cb = _make()
    for ts in (10.0, 20.0, 30.0):
        cb.before_call(
            now=ts,
            agent_name="A",
            provider="google",
            model_name="m",
        )
        cb.record_outcome(
            now=ts,
            ok=False,
            failure_kind="timeout",
            agent_name="A",
            provider="google",
            model_name="m",
        )
    assert (
        cb.snapshot(agent_name="A", provider="google", model_name="m")["state"]
        == "open"
    )
    # A different agent on same provider/model is untouched.
    cb.before_call(now=31.0, agent_name="B", provider="google", model_name="m")


def test_key_normalization_case_insensitive():
    cb = _make()
    for ts in (10.0, 20.0, 30.0):
        cb.before_call(
            now=ts,
            agent_name="Junior FUNDAMENTALS Analyst",
            provider="GOOGLE",
            model_name="Gemini-3.1-Flash-Lite",
        )
        cb.record_outcome(
            now=ts,
            ok=False,
            failure_kind="timeout",
            agent_name="Junior FUNDAMENTALS Analyst",
            provider="GOOGLE",
            model_name="Gemini-3.1-Flash-Lite",
        )
    # Different casing should resolve to the same key.
    with pytest.raises(CircuitOpenError):
        cb.before_call(
            now=31.0,
            agent_name="junior fundamentals analyst",
            provider="google",
            model_name="gemini-3.1-flash-lite",
        )


# ---- success closes regardless of prior state ----------------------------


def test_success_resets_failure_window():
    cb = _make()
    cb.before_call(now=10.0, **_KEY)
    cb.record_outcome(now=10.0, ok=False, failure_kind="timeout", **_KEY)
    cb.before_call(now=20.0, **_KEY)
    cb.record_outcome(now=20.0, ok=True, **_KEY)
    assert cb.snapshot(**_KEY)["failures_in_window"] == 0


# ---- construction guards -------------------------------------------------


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        LLMCircuitBreaker(threshold=0)
    with pytest.raises(ValueError):
        LLMCircuitBreaker(window_seconds=0.0)
    with pytest.raises(ValueError):
        LLMCircuitBreaker(cool_off_seconds=0.0)


# ---- singleton + config integration --------------------------------------


def test_get_circuit_breaker_reads_config(monkeypatch):
    from src.config import config

    monkeypatch.setattr(config, "llm_circuit_breaker_threshold", 5)
    monkeypatch.setattr(config, "llm_circuit_breaker_window_seconds", 120.0)
    monkeypatch.setattr(config, "llm_circuit_breaker_cool_off_seconds", 45.0)
    reset_circuit_breaker_for_tests()
    cb = get_circuit_breaker()
    assert cb.threshold == 5
    assert cb.window_seconds == 120.0
    assert cb.cool_off_seconds == 45.0
    reset_circuit_breaker_for_tests()
