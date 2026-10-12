"""``stalled_worker`` must release and join on every exit path."""

from __future__ import annotations

import threading

import pytest

from src.blocking_io import BlockingCallPolicy, InFlightLimit, run_blocking_call_sync
from tests.helpers.stalled_workers import join_workers, stalled_worker

LABEL = "stalled-worker-helper-test"


def _park(release: threading.Event) -> None:
    policy = BlockingCallPolicy(LABEL, 0.05, in_flight=InFlightLimit(1))
    with pytest.raises(TimeoutError):
        run_blocking_call_sync(policy, release.wait)


def _alive() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == f"blocking:{LABEL}"]


def test_worker_is_released_and_joined_on_normal_exit():
    with stalled_worker(LABEL) as release:
        _park(release)
        assert len(_alive()) == 1

    assert _alive() == []


def test_worker_is_released_and_joined_when_the_body_fails():
    with pytest.raises(AssertionError, match="body failed"):
        with stalled_worker(LABEL) as release:
            _park(release)
            raise AssertionError("body failed")

    assert _alive() == []


def test_a_worker_that_ignores_the_release_is_reported(monkeypatch):
    stuck = threading.Event()
    monkeypatch.setattr("tests.helpers.stalled_workers.JOIN_TIMEOUT_SECONDS", 0.05)
    try:
        with pytest.raises(AssertionError, match="outlived their release"):
            with stalled_worker(LABEL):
                _park(stuck)
    finally:
        stuck.set()
    assert join_workers(LABEL) == []
