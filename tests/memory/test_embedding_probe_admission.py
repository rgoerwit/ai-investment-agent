"""The embeddings startup probe: admitted whole or deferred, and run off the loop.

The probe's result is cached for the process, so it is never clamped to a
caller's deadline (a truncated probe would disable embeddings for every later
caller). It is *admitted* instead: it starts only when its full bound fits
before the operational deadline, and a deferral caches nothing. The async entry
points run it through ``prepare_shared_embeddings`` on a daemon thread, so the
synchronous constructors that follow find it cached and never block the loop.

Every test counts probe requests and client constructions. Sizing uses a frozen
deadline clock; elapsed time is measured only against a stall the test releases.
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
    FinancialSituationMemory._reset_shared_state_for_tests()
    yield counter
    FinancialSituationMemory._reset_shared_state_for_tests()


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
    real = memory_module.run_blocking_call_sync

    def spy(policy, fn):
        seen.append(policy.hard_timeout_seconds)
        return real(policy, fn)

    monkeypatch.setattr(memory_module, "run_blocking_call_sync", spy)
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
async def test_prepare_twice_probes_once(probe):
    assert await FinancialSituationMemory.prepare_shared_embeddings() is True
    assert await FinancialSituationMemory.prepare_shared_embeddings() is True
    assert probe.sends == 1


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


@pytest.mark.asyncio
async def test_a_cancelled_prepare_caches_nothing(probe):
    release = threading.Event()
    started = threading.Event()

    def stall() -> None:
        started.set()
        release.wait(5)

    probe.behaviour = stall
    task = asyncio.ensure_future(FinancialSituationMemory.prepare_shared_embeddings())
    while not started.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()

    # The cancelled probe recorded nothing, so the next caller probes again.
    probe.behaviour = None
    assert await FinancialSituationMemory.prepare_shared_embeddings() is True
    assert probe.sends == 2


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
