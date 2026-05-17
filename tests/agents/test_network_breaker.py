"""Unit tests for the process-global NetworkBreaker.

Three classes:
- happy / threshold behavior (closed → open → half-open → closed cycle)
- edge cases (window pruning, ineligible failure kinds, concurrent half-open)
- error path (concurrent threads, reset)

Note: tests pass explicit `now=` arguments rather than relying on real
`time.monotonic()` so failures don't drift across the window boundary.
"""

from __future__ import annotations

import threading

import pytest

from src.agents.network_breaker import (
    NetworkBreaker,
    NetworkBreakerOpenError,
    get_network_breaker,
    reset_network_breaker_singleton,
)


@pytest.fixture(autouse=True)
def _reset_breaker_singleton():
    reset_network_breaker_singleton()
    yield
    reset_network_breaker_singleton()


class TestHappyPath:
    def test_below_threshold_stays_closed(self):
        b = NetworkBreaker(threshold=4, window_seconds=30.0, cool_off_seconds=45.0)
        for i in range(3):
            b.record_outcome(ok=False, failure_kind="dns_resolution", now=i * 1.0)
        snap = b.snapshot()
        assert snap["state"] == "closed"
        assert snap["failures_in_window"] == 3
        # before_call must not raise
        b.before_call(now=4.0)

    def test_threshold_opens_breaker(self):
        b = NetworkBreaker(threshold=4, window_seconds=30.0, cool_off_seconds=45.0)
        for i in range(4):
            b.record_outcome(ok=False, failure_kind="dns_resolution", now=i * 1.0)
        snap = b.snapshot()
        assert snap["state"] == "open"
        with pytest.raises(NetworkBreakerOpenError):
            b.before_call(now=5.0)

    def test_cool_off_then_half_open_probe_success_closes(self):
        b = NetworkBreaker(threshold=4, window_seconds=30.0, cool_off_seconds=45.0)
        for i in range(4):
            b.record_outcome(ok=False, failure_kind="dns_resolution", now=i * 1.0)
        # Past the cool-off window — should be allowed through as half-open.
        b.before_call(now=100.0)
        assert b.snapshot()["state"] == "half_open"
        b.record_outcome(ok=True, now=101.0)
        assert b.snapshot()["state"] == "closed"
        assert b.snapshot()["failures_in_window"] == 0

    def test_cool_off_then_half_open_probe_failure_reopens(self):
        b = NetworkBreaker(threshold=4, window_seconds=30.0, cool_off_seconds=45.0)
        for i in range(4):
            b.record_outcome(ok=False, failure_kind="connect_error", now=i * 1.0)
        b.before_call(now=100.0)  # half-open probe admitted
        b.record_outcome(ok=False, failure_kind="dns_resolution", now=101.0)
        snap = b.snapshot()
        assert snap["state"] == "open"
        # Re-open extends cool-off by 45s from the probe failure timestamp.
        assert snap["open_until_monotonic"] == pytest.approx(101.0 + 45.0)


class TestEdgeCases:
    def test_failures_outside_window_dont_count(self):
        b = NetworkBreaker(threshold=4, window_seconds=30.0, cool_off_seconds=45.0)
        # Spread 4 failures across > 30s → pruning drops the oldest each time.
        for i in range(4):
            b.record_outcome(ok=False, failure_kind="dns_resolution", now=i * 12.0)
        snap = b.snapshot()
        # By the 4th failure at t=36, t=0 is outside the 30s window and pruned.
        assert snap["state"] == "closed"
        assert snap["failures_in_window"] <= 3

    def test_ineligible_failure_kinds_dont_count(self):
        b = NetworkBreaker(threshold=2, window_seconds=30.0, cool_off_seconds=45.0)
        # `timeout`, `server_error`, `application_error` are handled
        # elsewhere — they must not trip this breaker.
        for kind in ("timeout", "server_error", "application_error", "rate_limit"):
            b.record_outcome(ok=False, failure_kind=kind, now=0.0)
        assert b.snapshot()["state"] == "closed"
        assert b.snapshot()["failures_in_window"] == 0

    def test_success_clears_failure_window(self):
        b = NetworkBreaker(threshold=4, window_seconds=30.0, cool_off_seconds=45.0)
        for i in range(3):
            b.record_outcome(ok=False, failure_kind="dns_resolution", now=i * 1.0)
        b.record_outcome(ok=True, now=4.0)
        assert b.snapshot()["failures_in_window"] == 0

    def test_concurrent_half_open_admits_only_one_probe(self):
        b = NetworkBreaker(threshold=4, window_seconds=30.0, cool_off_seconds=45.0)
        for i in range(4):
            b.record_outcome(ok=False, failure_kind="dns_resolution", now=i * 1.0)
        # First caller past cool-off becomes the probe.
        b.before_call(now=100.0)
        # Second concurrent caller is denied (probe_in_flight=True).
        with pytest.raises(NetworkBreakerOpenError):
            b.before_call(now=100.5)


class TestConstructorValidation:
    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            NetworkBreaker(threshold=0)

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            NetworkBreaker(window_seconds=0)

    def test_invalid_cool_off(self):
        with pytest.raises(ValueError):
            NetworkBreaker(cool_off_seconds=-1)


class TestThreadSafety:
    def test_concurrent_record_outcome_does_not_double_open(self):
        """Hammer record_outcome from many threads; verify the open count
        is consistent (no missing-or-doubled failure tallies)."""
        b = NetworkBreaker(threshold=50, window_seconds=300.0, cool_off_seconds=45.0)
        n_threads = 8
        per_thread = 10

        def worker():
            for _ in range(per_thread):
                b.record_outcome(ok=False, failure_kind="dns_resolution")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = b.snapshot()
        # 80 failures, threshold 50 → should be open. Window long enough
        # that all 80 still in the deque.
        assert snap["state"] == "open"


class TestSingleton:
    def test_get_returns_same_instance(self):
        a = get_network_breaker()
        b = get_network_breaker()
        assert a is b

    def test_reset_clears_singleton(self):
        a = get_network_breaker()
        reset_network_breaker_singleton()
        b = get_network_breaker()
        assert a is not b
