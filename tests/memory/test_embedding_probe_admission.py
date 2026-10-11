"""The embeddings startup probe: admitted whole or deferred, and run off the loop.

The probe's result is cached for the process, so it is never clamped to a
caller's deadline (a truncated probe would disable embeddings for every later
caller). It is *admitted* instead: it starts only when its full bound fits
before the operational deadline, and a deferral caches nothing. The async entry
points run it through ``prepare_shared_embeddings`` on a daemon thread, so the
synchronous constructors that follow find it cached and never block the loop.

One caller owns each binding's probe, from before the client is built until
the worker publishes. Every other caller, sync or async, on any thread or loop,
waits on the same completion under its own deadline, and cannot cancel it. A
probe from an older binding (or from before a test reset) cannot publish.

Every test counts probe requests and client constructions. Sizing uses a frozen
deadline clock; elapsed time is measured only against a stall the test releases.
Each stalled worker is released and joined before the test's patches unwind.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

import src.memory as memory_module
from src.config import config
from src.embeddings import resolve_embedding_binding
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline
from src.memory import FinancialSituationMemory

_NOW = 1_000.0
_PROBE_BOUND = 15.0


class Probe:
    """The embeddings client the memory builds: counts builds and probe sends."""

    def __init__(self, behaviour: Callable[[], Any] | None = None) -> None:
        self.builds = 0
        self.sends = 0
        self.behaviour = behaviour
        self.result: list[float] = [0.1] * 8

    def build(self, *_args: Any, **_kwargs: Any) -> Any:
        self.builds += 1
        client = MagicMock()
        client.embed_query.side_effect = self._embed
        return client

    def _embed(self, _text: str) -> list[float]:
        self.sends += 1
        if self.behaviour is not None:
            result = self.behaviour()
            if isinstance(result, BaseException):
                raise result
        return self.result


@pytest.fixture
def probe(monkeypatch) -> Probe:
    """A credentialed memory with Chroma off and a counting embeddings client."""
    fake_config = MagicMock()
    fake_config.get_google_api_key.return_value = "test-key"
    monkeypatch.setattr(memory_module, "config", fake_config)
    monkeypatch.setattr(
        memory_module, "embedding_credential", lambda *_a, **_k: "test-key"
    )
    monkeypatch.setattr(
        memory_module,
        "resolve_embedding_binding",
        lambda *_a, **_k: resolve_embedding_binding(config),
    )
    monkeypatch.setattr(
        FinancialSituationMemory,
        "_get_shared_chroma_client",
        classmethod(lambda cls, _name: None),
    )
    monkeypatch.setattr(
        config, "memory_embedding_healthcheck_timeout_seconds", _PROBE_BOUND
    )
    counter = Probe()
    monkeypatch.setattr(memory_module, "build_embeddings", counter.build)
    # The root _reset_shared_memory_state fixture clears the class state.
    return counter


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    clock = [_NOW]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


# --- Constructor admission ------------------------------------------------------


@pytest.mark.parametrize(
    ("allowance", "admitted"),
    [
        (None, True),
        (_PROBE_BOUND + 60.0, True),
        (_PROBE_BOUND, True),  # exactly the bound: admitted
        (_PROBE_BOUND - 0.001, False),
        (0.0, False),
        (-5.0, False),
    ],
    ids=["no_deadline", "ample", "exactly_the_bound", "just_below", "zero", "spent"],
)
def test_the_probe_starts_only_when_its_full_bound_fits(
    probe, fake_clock, allowance, admitted
):
    with enclosing_deadline(allowance):
        memory = FinancialSituationMemory("TEST_probe_admission")

    assert probe.sends == (1 if admitted else 0)
    assert probe.builds == (1 if admitted else 0)
    assert memory.embeddings_available is admitted


def test_five_memories_under_a_spent_deadline_build_and_send_nothing(probe, fake_clock):
    with enclosing_deadline(-1.0):
        memories = [FinancialSituationMemory(f"TEST_spent_{i}") for i in range(5)]

    assert probe.builds == probe.sends == 0
    assert all(m.embeddings is None and not m.available for m in memories)


def test_a_deferral_caches_nothing_and_the_next_caller_probes_once(probe, fake_clock):
    with enclosing_deadline(1.0):
        FinancialSituationMemory("TEST_deferred")
    assert probe.sends == 0

    first = FinancialSituationMemory("TEST_after_deferral_1")
    second = FinancialSituationMemory("TEST_after_deferral_2")
    assert probe.sends == 1
    assert first.embeddings_available and second.embeddings_available


def test_a_probe_that_fits_runs_at_its_full_bound_unclamped(
    probe, fake_clock, monkeypatch
):
    seen: list[float] = []
    real = memory_module.start_blocking_call

    def spy(policy, fn):
        seen.append(policy.hard_timeout_seconds)
        return real(policy, fn)

    monkeypatch.setattr(memory_module, "start_blocking_call", spy)
    with enclosing_deadline(_PROBE_BOUND + 30.0):
        FinancialSituationMemory("TEST_unclamped")
    assert seen == [_PROBE_BOUND]


def test_a_failed_probe_is_cached_and_never_resent(probe):
    probe.behaviour = lambda: ConnectionError("unreachable")

    first = FinancialSituationMemory("TEST_failed_1")
    second = FinancialSituationMemory("TEST_failed_2")

    assert probe.sends == 1
    assert first.embeddings_available is False
    assert second.embeddings_available is False


def test_an_empty_probe_vector_is_a_failure(probe):
    probe.result = []

    memory = FinancialSituationMemory("TEST_empty_vector")

    assert probe.sends == 1
    assert memory.embeddings_available is False


def test_a_stalled_probe_returns_at_its_bound(probe, monkeypatch):
    release = threading.Event()
    probe.behaviour = lambda: release.wait(5)
    monkeypatch.setattr(config, "memory_embedding_healthcheck_timeout_seconds", 0.05)

    start = time.monotonic()
    try:
        memory = FinancialSituationMemory("TEST_stalled")
    finally:
        release.set()
    assert time.monotonic() - start < 1.0
    assert memory.embeddings_available is False


def test_no_credential_builds_and_sends_nothing(probe, monkeypatch):
    monkeypatch.setattr(memory_module, "embedding_credential", lambda *_a, **_k: "")

    memory = FinancialSituationMemory("TEST_no_key")

    assert probe.builds == probe.sends == 0
    assert memory.available is False


def test_a_client_that_cannot_be_built_is_not_cached(probe, monkeypatch):
    def broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("bad client")

    monkeypatch.setattr(memory_module, "build_embeddings", broken)
    assert FinancialSituationMemory("TEST_unbuildable").embeddings is None

    monkeypatch.setattr(memory_module, "build_embeddings", probe.build)
    assert FinancialSituationMemory("TEST_buildable").embeddings_available is True
    assert probe.sends == 1


# --- Off the event loop ---------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_runs_the_probe_off_the_loop(probe):
    release = threading.Event()
    probe.behaviour = lambda: release.wait(5) and None
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while not release.is_set():
            ticks += 1
            await asyncio.sleep(0.005)

    ticking = asyncio.create_task(ticker())
    asyncio.get_running_loop().call_later(0.1, release.set)
    assert await FinancialSituationMemory.prepare_shared_embeddings() is True
    await ticking

    # The loop kept running while the probe waited: a blocking join would
    # have left the ticker at zero.
    assert ticks >= 5
    assert probe.sends == 1


@pytest.mark.asyncio
async def test_constructors_after_prepare_reuse_its_result(probe):
    await FinancialSituationMemory.prepare_shared_embeddings()
    memories = [FinancialSituationMemory(f"TEST_after_prepare_{i}") for i in range(5)]

    assert probe.builds == probe.sends == 1
    assert all(m.embeddings_available for m in memories)


@pytest.mark.asyncio
async def test_prepare_under_a_spent_deadline_defers(probe, fake_clock):
    with enclosing_deadline(-1.0):
        assert await FinancialSituationMemory.prepare_shared_embeddings() is False
    assert probe.builds == probe.sends == 0
    # Nothing cached: a later prepare with time probes.
    assert await FinancialSituationMemory.prepare_shared_embeddings() is True
    assert probe.sends == 1


@pytest.mark.asyncio
async def test_prepare_caches_a_failure(probe):
    probe.behaviour = lambda: ConnectionError("unreachable")
    assert await FinancialSituationMemory.prepare_shared_embeddings() is False
    memory = FinancialSituationMemory("TEST_after_failed_prepare")
    assert memory.embeddings_available is False
    assert probe.sends == 1


@pytest.mark.asyncio
async def test_a_stalled_prepare_returns_at_its_bound(probe, monkeypatch):
    release = threading.Event()
    probe.behaviour = lambda: release.wait(5) and None
    monkeypatch.setattr(config, "memory_embedding_healthcheck_timeout_seconds", 0.05)

    start = time.monotonic()
    try:
        assert await FinancialSituationMemory.prepare_shared_embeddings() is False
    finally:
        release.set()
    assert time.monotonic() - start < 1.0


# --- One owner per binding, any thread or loop ------------------------------------


class Stall:
    """The first probe send waits for the test's release; later sends do not."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.outcome: BaseException | None = None

    def __call__(self) -> BaseException | None:
        if not self.started.is_set():
            self.started.set()
            self.release.wait(5)
        return self.outcome


def _join_probe_workers() -> None:
    for worker in threading.enumerate():
        if worker.name == "blocking:memory_embedding_healthcheck":
            worker.join(5)


@pytest.fixture
def stall(probe):
    stalled = Stall()
    probe.behaviour = stalled
    try:
        yield stalled
    finally:
        stalled.release.set()
        _join_probe_workers()


async def _until(event: threading.Event) -> None:
    for _ in range(5000):
        if event.is_set():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("the probe never started")


def _prepare() -> Any:
    return FinancialSituationMemory.prepare_shared_embeddings()


def test_every_probe_takes_the_same_slot():
    assert memory_module._probe_policy().in_flight is not None
    assert memory_module._probe_policy().in_flight is (
        memory_module._probe_policy().in_flight
    )


@pytest.mark.asyncio
async def test_concurrent_prepares_build_and_probe_once(probe, stall):
    first = asyncio.ensure_future(_prepare())
    await _until(stall.started)
    second = asyncio.ensure_future(_prepare())
    await asyncio.sleep(0)  # the second caller arrives while the probe runs
    stall.release.set()

    assert await asyncio.gather(first, second) == [True, True]
    assert probe.builds == probe.sends == 1


@pytest.mark.asyncio
async def test_a_sync_constructor_joins_an_async_owner(probe, stall):
    owner = asyncio.ensure_future(_prepare())
    await _until(stall.started)
    joined: list[FinancialSituationMemory] = []
    joiner = threading.Thread(
        target=lambda: joined.append(FinancialSituationMemory("TEST_sync_joiner"))
    )
    joiner.start()
    await asyncio.sleep(0.05)  # ordering only: the joiner arrives mid-probe
    stall.release.set()

    assert await owner is True
    await asyncio.to_thread(joiner.join, 5)
    assert joined[0].embeddings_available is True
    assert probe.builds == probe.sends == 1


def test_two_event_loops_share_one_probe(probe, stall):
    results: list[bool] = []

    def run_loop() -> None:
        results.append(asyncio.run(_prepare()))

    first = threading.Thread(target=run_loop)
    first.start()
    assert stall.started.wait(5)
    second = threading.Thread(target=run_loop)
    second.start()
    time.sleep(0.05)  # ordering only: the second loop arrives mid-probe
    stall.release.set()
    first.join(5)
    second.join(5)

    assert results == [True, True]
    assert probe.builds == probe.sends == 1


def test_a_loop_shut_down_mid_probe_still_publishes_for_the_next_run(probe, stall):
    async def start_and_leave() -> None:
        asyncio.ensure_future(_prepare())
        await _until(stall.started)

    # asyncio.run cancels the owner's pending task as its loop shuts down.
    asyncio.run(start_and_leave())
    stall.release.set()
    _join_probe_workers()

    assert asyncio.run(_prepare()) is True
    assert probe.builds == probe.sends == 1


@pytest.mark.asyncio
async def test_a_cancelled_owner_cannot_cancel_the_probe(probe, stall):
    owner = asyncio.ensure_future(_prepare())
    await _until(stall.started)
    waiter = asyncio.ensure_future(_prepare())
    await asyncio.sleep(0)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    stall.release.set()

    # The other waiter still gets the probe's result, and the worker published
    # although its owner was gone: nothing is resent.
    assert await waiter is True
    await asyncio.to_thread(_join_probe_workers)
    assert await _prepare() is True
    assert probe.builds == probe.sends == 1


def test_a_success_after_the_bound_is_discarded(probe, stall, monkeypatch):
    monkeypatch.setattr(config, "memory_embedding_healthcheck_timeout_seconds", 0.05)

    first = FinancialSituationMemory("TEST_late_1")
    assert first.embeddings_available is False
    stall.release.set()
    _join_probe_workers()

    # The timeout failure stays cached; the late vector changed nothing.
    second = FinancialSituationMemory("TEST_late_2")
    assert second.embeddings_available is False
    assert probe.builds == probe.sends == 1


def test_a_probe_still_stalled_settles_as_failed_at_its_bound(
    probe, stall, monkeypatch
):
    monkeypatch.setattr(config, "memory_embedding_healthcheck_timeout_seconds", 0.05)

    assert FinancialSituationMemory("TEST_stalled_owner").available is False
    for _ in range(1000):  # the expiry fires at the bound, worker still stalled
        if FinancialSituationMemory._probe_run is None:
            break
        time.sleep(0.001)

    assert not stall.release.is_set()
    assert FinancialSituationMemory._probe_run is None
    # The timeout failure is cached with its client, as a failed probe is.
    assert FinancialSituationMemory._shared_embeddings is not None
    assert FinancialSituationMemory._shared_embeddings_available is False


@pytest.mark.asyncio
async def test_an_old_bindings_late_probe_cannot_publish_into_the_new_one(
    probe, stall, monkeypatch
):
    owner = asyncio.ensure_future(_prepare())
    await _until(stall.started)
    monkeypatch.setattr(
        memory_module, "embedding_credential", lambda *_a, **_k: "rotated-key"
    )

    # The old probe still holds the one probe slot: the new binding's probe is
    # not sent, and nothing is cached for it.
    blocked = FinancialSituationMemory("TEST_new_binding_blocked")
    assert blocked.embeddings is None
    assert probe.sends == 1

    stall.release.set()
    assert await owner is False
    await asyncio.to_thread(_join_probe_workers)
    assert FinancialSituationMemory._shared_embeddings is None

    fresh = FinancialSituationMemory("TEST_new_binding")
    assert fresh.embeddings_available is True
    assert FinancialSituationMemory._shared_embeddings is fresh.embeddings
    assert (probe.builds, probe.sends) == (3, 2)


@pytest.mark.asyncio
async def test_a_reset_during_a_probe_discards_its_late_result(probe, stall):
    owner = asyncio.ensure_future(_prepare())
    await _until(stall.started)
    FinancialSituationMemory._reset_shared_state_for_tests()
    stall.release.set()

    assert await owner is False
    await asyncio.to_thread(_join_probe_workers)
    assert FinancialSituationMemory._shared_embeddings is None
    assert FinancialSituationMemory._shared_embeddings_available is False


@pytest.mark.asyncio
async def test_waiters_on_a_failed_probe_share_its_one_failure(probe, stall):
    stall.outcome = ConnectionError("unreachable")
    first = asyncio.ensure_future(_prepare())
    await _until(stall.started)
    second = asyncio.ensure_future(_prepare())
    await asyncio.sleep(0)
    stall.release.set()

    assert await asyncio.gather(first, second) == [False, False]
    assert FinancialSituationMemory("TEST_after_failure").embeddings_available is False
    assert probe.builds == probe.sends == 1


def test_a_failed_build_releases_ownership_to_the_next_caller(probe, monkeypatch):
    building = threading.Event()
    fail = threading.Event()
    attempts: list[int] = []

    def slow_broken(*_a: Any, **_k: Any) -> Any:
        attempts.append(1)
        building.set()
        fail.wait(5)
        raise RuntimeError("bad client")

    monkeypatch.setattr(memory_module, "build_embeddings", slow_broken)
    built: list[FinancialSituationMemory] = []

    def construct(name: str) -> threading.Thread:
        thread = threading.Thread(
            target=lambda: built.append(FinancialSituationMemory(name))
        )
        thread.start()
        return thread

    owner = construct("TEST_build_owner")
    assert building.wait(5)
    joiner = construct("TEST_build_joiner")
    time.sleep(0.05)  # ordering only: the joiner arrives mid-build
    fail.set()
    owner.join(5)
    joiner.join(5)

    assert [m.embeddings for m in built] == [None, None]
    assert attempts == [1]
    # Nothing was cached: the next caller owns a fresh attempt.
    monkeypatch.setattr(memory_module, "build_embeddings", probe.build)
    assert FinancialSituationMemory("TEST_build_retry").embeddings_available is True
    assert probe.builds == probe.sends == 1


@pytest.mark.asyncio
async def test_a_waiter_keeps_its_own_deadline(probe, stall):
    owner = asyncio.ensure_future(_prepare())
    await _until(stall.started)

    with enclosing_deadline(-1.0):
        spent = FinancialSituationMemory("TEST_spent_waiter")
    start = time.monotonic()
    with enclosing_deadline(0.05):
        short = await _prepare()
    elapsed = time.monotonic() - start

    assert spent.embeddings is None
    assert short is False
    assert elapsed < 1.0
    # Neither waiter cached anything or touched the owner's probe.
    stall.release.set()
    assert await owner is True
    assert probe.builds == probe.sends == 1


# --- The CLI entry points prepare before building any memory --------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_memory", [True, False], ids=["memory", "no_memory"])
async def test_the_cli_prepares_embeddings_only_with_memory_on(probe, enable_memory):
    from src.main import _prepare_memory_embeddings
    from src.runtime_config import get_runtime_config, use_runtime_config

    runtime = get_runtime_config(config).with_overrides(enable_memory=enable_memory)
    with use_runtime_config(runtime):
        await _prepare_memory_embeddings()

    assert probe.sends == (1 if enable_memory else 0)
    assert probe.builds == (1 if enable_memory else 0)
