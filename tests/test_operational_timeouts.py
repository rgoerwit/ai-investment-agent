"""Operational deadlines and the last market-data sites: one setting per scope.

Each scope is checked for its shipped default (read from the field definition,
never by instantiating ``Settings``), an override reaching the real wait,
request or flush call, and its existing fallback on timeout. Where a scope
clamps to a caller's enclosing deadline, a nested deadline wins and a spent one
starts no work. Where it deliberately does not clamp (the health check, the
embeddings startup probe and the two shutdown paths), a spent caller deadline
leaves the bound unchanged.

Fake clocks, fakes and tiny timeouts only; no network, and the memory tests
never construct a Chroma client.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.blocking_io as blocking_io
import src.cleanup as cleanup
import src.health_check as health_check
import src.main as main_module
import src.memory as memory_module
import src.observability as observability
import src.output as output
import src.retrospective as retrospective
import src.stocktwits_api as stocktwits_api
import src.tools.ownership as ownership
from src.blocking_io import (
    OUTPUT_COMPANY_NAME_INFO_POLICY,
    OUTPUT_COMPANY_NAME_POLICY,
    BlockingCallSaturatedError,
)
from src.config import Settings, config
from src.embeddings import resolve_embedding_binding
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline
from src.memory import EmbeddingAllowanceSpent, FinancialSituationMemory

_NOW = 1_000.0

SHIPPED_DEFAULTS = {
    # Market data (leftovers)
    "ownership_section_timeout_seconds": 8.0,
    "stocktwits_http_timeout_seconds": 10.0,
    "retrospective_price_fetch_timeout_seconds": 15.0,
    "market_context_fetch_timeout_seconds": 10.0,
    "price_snapshot_fetch_timeout_seconds": 10.0,
    "output_company_name_timeout_seconds": 6.0,
    # Operations
    "health_check_connectivity_timeout_seconds": 15.0,
    "memory_embedding_healthcheck_timeout_seconds": 15.0,
    "memory_embedding_call_timeout_seconds": 30.0,
    "memory_embedding_min_useful_seconds": 1.0,
    "genai_client_close_timeout_seconds": 2.0,
    "langfuse_flush_timeout_seconds": 2.0,
    # LLM request ceilings (round 2, Step 7)
    "openai_request_cap_seconds": 120.0,
    "apac_node_total_timeout_seconds": 240.0,
    "llm_attempts": 2,
}


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    """A frozen monotonic clock for the deadline helpers; advance via clock[0]."""
    clock = [_NOW]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


def _spy_hard_timeout(monkeypatch, module, result: Any = None) -> list[float]:
    """Replace ``module.run_with_hard_timeout``; record each timeout, never run."""
    seen: list[float] = []

    async def spy(coro, *, timeout, label):
        del label
        seen.append(timeout)
        coro.close()
        return result

    monkeypatch.setattr(module, "run_with_hard_timeout", spy)
    return seen


# --- Defaults --------------------------------------------------------------


@pytest.mark.parametrize("field, expected", sorted(SHIPPED_DEFAULTS.items()))
def test_shipped_default_is_unchanged(field, expected):
    assert Settings.model_fields[field].default == expected


@pytest.mark.parametrize(
    "policy, field",
    [
        (OUTPUT_COMPANY_NAME_POLICY, "output_company_name_timeout_seconds"),
        (OUTPUT_COMPANY_NAME_INFO_POLICY, "company_name_lookup_timeout_seconds"),
    ],
    ids=lambda value: getattr(value, "label", value),
)
def test_policy_constant_matches_its_setting_default(policy, field):
    # The constant's number is only a default; the call site applies the field.
    assert policy.hard_timeout_seconds == Settings.model_fields[field].default


def test_sync_name_lookup_policy_carries_a_slot_cap():
    # run_blocking_call_sync refuses a policy without one.
    assert OUTPUT_COMPANY_NAME_INFO_POLICY.in_flight is not None


# --- Health check ----------------------------------------------------------


def _bind_probe(monkeypatch, llm: Any) -> None:
    import src.llm_runtime.construction as construction

    monkeypatch.setattr(
        construction, "build_required_model_for_seat", lambda *_a, **_k: llm
    )
    monkeypatch.setattr(health_check, "get_runtime_provider", lambda _llm: "fake")
    monkeypatch.setattr(health_check, "get_model_name", lambda _llm: "fake-model")


def _ok_llm() -> Any:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="OK"))
    return llm


@pytest.mark.asyncio
async def test_health_check_override_reaches_the_probe_wait(monkeypatch):
    _bind_probe(monkeypatch, _ok_llm())
    monkeypatch.setattr(config, "health_check_connectivity_timeout_seconds", 4.5)
    seen = _spy_hard_timeout(
        monkeypatch, health_check, result=SimpleNamespace(content="OK")
    )

    assert await health_check.check_llm_connectivity() is True
    assert seen == [4.5]


@pytest.mark.asyncio
async def test_health_check_is_not_clamped_to_a_caller_deadline(
    monkeypatch, fake_clock
):
    # A top-level entrypoint: even a spent caller deadline leaves its bound.
    _bind_probe(monkeypatch, _ok_llm())
    seen = _spy_hard_timeout(
        monkeypatch, health_check, result=SimpleNamespace(content="OK")
    )
    monkeypatch.setattr(config, "health_check_connectivity_timeout_seconds", 15.0)
    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        assert await health_check.check_llm_connectivity() is True
    assert seen == [15.0]


@pytest.mark.asyncio
async def test_health_check_stalled_probe_fails_at_its_bound(monkeypatch):
    llm = MagicMock()

    async def _stall(*_a, **_k):
        await asyncio.sleep(5)

    llm.ainvoke = _stall
    _bind_probe(monkeypatch, llm)
    monkeypatch.setattr(config, "health_check_connectivity_timeout_seconds", 0.05)

    start = time.monotonic()
    assert await health_check.check_llm_connectivity() is False
    assert time.monotonic() - start < 2.0


# --- Embeddings ------------------------------------------------------------


@pytest.fixture
def isolated_memory_state():
    FinancialSituationMemory._reset_shared_state_for_tests()
    yield
    FinancialSituationMemory._reset_shared_state_for_tests()


class _InstantLimiter:
    async def aacquire(self, *, blocking: bool = True) -> bool:
        return True


def _bare_memory(monkeypatch, embeddings: Any) -> FinancialSituationMemory:
    """A memory object built without its constructor: no Chroma client at all."""
    import src.llms as llms

    monkeypatch.setattr(llms, "GLOBAL_RATE_LIMITER", _InstantLimiter())
    memory = FinancialSituationMemory.__new__(FinancialSituationMemory)
    memory.name = "TEST_operational_memory"
    memory.embedding_binding = resolve_embedding_binding(config)
    memory.available = True
    memory.embeddings = embeddings
    memory.situation_collection = MagicMock()
    return memory


def _fast_embeddings() -> Any:
    embeddings = MagicMock()
    embeddings.aembed_query = AsyncMock(return_value=[0.1] * 8)
    return embeddings


@pytest.mark.asyncio
async def test_embedding_call_override_reaches_the_request_wait(
    monkeypatch, fake_clock
):
    # Frozen clock: the request gets what is left of the attempt's deadline.
    memory = _bare_memory(monkeypatch, _fast_embeddings())
    monkeypatch.setattr(config, "memory_embedding_call_timeout_seconds", 12.5)
    seen = _spy_hard_timeout(monkeypatch, memory_module, result=[0.1] * 8)

    assert await memory._get_embedding("text") == [0.1] * 8
    assert seen == [12.5]


@pytest.mark.asyncio
async def test_embedding_call_nested_caller_deadline_wins(monkeypatch, fake_clock):
    memory = _bare_memory(monkeypatch, _fast_embeddings())
    seen = _spy_hard_timeout(monkeypatch, memory_module, result=[0.1] * 8)

    with enclosing_deadline(3.0):
        await memory._get_embedding("text")
    assert seen == [pytest.approx(3.0)]


@pytest.mark.asyncio
async def test_embedding_call_spent_deadline_sends_nothing_and_is_not_retried(
    monkeypatch, fake_clock
):
    embeddings = _fast_embeddings()
    memory = _bare_memory(monkeypatch, embeddings)

    start = time.monotonic()
    with enclosing_deadline(1.0):
        fake_clock[0] += 2.0
        with pytest.raises(EmbeddingAllowanceSpent):
            await memory._get_embedding("text")
    # Not retried: tenacity's 2 s + 4 s backoff would show here.
    assert time.monotonic() - start < 1.0
    embeddings.aembed_query.assert_not_called()


@pytest.mark.asyncio
async def test_spent_deadline_query_takes_the_empty_result_fallback(
    monkeypatch, fake_clock
):
    embeddings = _fast_embeddings()
    memory = _bare_memory(monkeypatch, embeddings)

    with enclosing_deadline(1.0):
        fake_clock[0] += 2.0
        assert await memory.query_similar_situations("text") == []
    embeddings.aembed_query.assert_not_called()
    memory.situation_collection.query.assert_not_called()


@pytest.mark.asyncio
async def test_stalled_embedding_call_raises_timeout_at_its_bound(monkeypatch):
    async def _stall(*_a, **_k):
        await asyncio.sleep(5)

    embeddings = MagicMock()
    embeddings.aembed_query = _stall
    memory = _bare_memory(monkeypatch, embeddings)
    monkeypatch.setattr(config, "memory_embedding_call_timeout_seconds", 0.05)
    monkeypatch.setattr(memory_module, "logger", MagicMock())

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        # Undecorated: the bound itself, without tenacity's backoff.
        await memory._get_embedding.__wrapped__(memory, "text")
    assert time.monotonic() - start < 2.0


def _probe_memory(monkeypatch, embed_query) -> FinancialSituationMemory:
    """Run the constructor's embeddings probe with Chroma switched off."""
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
    embeddings = MagicMock()
    embeddings.embed_query.side_effect = embed_query
    monkeypatch.setattr(
        memory_module, "GoogleGenerativeAIEmbeddings", lambda **_k: embeddings
    )
    monkeypatch.setattr(memory_module, "build_embeddings", lambda *_a, **_k: embeddings)
    monkeypatch.setattr(
        FinancialSituationMemory,
        "_get_shared_chroma_client",
        classmethod(lambda cls, _name: None),
    )
    return FinancialSituationMemory("TEST_operational_probe")


def test_embedding_probe_override_bounds_the_constructor(
    monkeypatch, isolated_memory_state
):
    release = threading.Event()
    monkeypatch.setattr(config, "memory_embedding_healthcheck_timeout_seconds", 0.05)

    start = time.monotonic()
    try:
        memory = _probe_memory(monkeypatch, lambda *_a: release.wait(5) or [0.1])
    finally:
        release.set()
    assert time.monotonic() - start < 2.0
    assert memory.embeddings_available is False


def test_embedding_probe_is_not_clamped_to_a_caller_deadline(
    monkeypatch, isolated_memory_state, fake_clock
):
    # Its failure is cached for the process, so a spent caller deadline must
    # not turn embeddings off for everyone.
    with enclosing_deadline(1.0):
        fake_clock[0] += 2.0
        memory = _probe_memory(monkeypatch, lambda *_a: [0.1] * 8)
    assert memory.embeddings_available is True


# --- Shutdown: GenAI client close ------------------------------------------


def _bind_genai_client(monkeypatch, aclose) -> SimpleNamespace:
    client = SimpleNamespace(aclose=aclose)
    llm = SimpleNamespace(async_client=client)
    monkeypatch.setitem(
        sys.modules,
        "src.llms",
        SimpleNamespace(get_all_llm_instances=lambda: {"quick": llm}),
    )
    return llm


@pytest.mark.asyncio
async def test_client_close_override_reaches_the_close_wait(monkeypatch):
    # A close slower than a tiny bound but inside the override completes.
    closed: list[bool] = []

    async def _close():
        await asyncio.sleep(0.1)
        closed.append(True)

    llm = _bind_genai_client(monkeypatch, _close)
    monkeypatch.setattr(config, "genai_client_close_timeout_seconds", 1.5)
    mock_logger = MagicMock()
    monkeypatch.setattr(cleanup, "logger", mock_logger)

    await cleanup._cleanup_genai_clients()
    assert closed == [True]
    mock_logger.warning.assert_not_called()
    assert llm.async_client is None


@pytest.mark.asyncio
async def test_client_close_is_not_clamped_to_a_caller_deadline(
    monkeypatch, fake_clock
):
    # Shutdown runs after any caller deadline; a spent one must not skip it.
    aclose = AsyncMock()
    llm = _bind_genai_client(monkeypatch, aclose)
    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        await cleanup._cleanup_genai_clients()
    aclose.assert_awaited_once()
    assert llm.async_client is None


@pytest.mark.asyncio
async def test_stalled_client_close_is_dropped_at_its_bound(monkeypatch):
    async def _slow_close():
        await asyncio.sleep(5)

    llm = _bind_genai_client(monkeypatch, _slow_close)
    monkeypatch.setattr(config, "genai_client_close_timeout_seconds", 0.02)
    mock_logger = MagicMock()
    monkeypatch.setattr(cleanup, "logger", mock_logger)

    start = time.monotonic()
    await cleanup._cleanup_genai_clients()
    assert time.monotonic() - start < 2.0
    assert llm.async_client is None
    mock_logger.warning.assert_called_once_with(
        "cleanup_timeout", resource="genai_async_client_quick", timeout_seconds=0.02
    )


# --- Shutdown: Langfuse flush ----------------------------------------------


def _bind_langfuse(monkeypatch, client: Any) -> None:
    monkeypatch.setattr(observability, "_runtime_langfuse_enabled", lambda _s: True)
    monkeypatch.setattr(observability, "_langfuse_keys_present", lambda _s: True)
    monkeypatch.setitem(
        sys.modules, "langfuse", SimpleNamespace(get_client=lambda: client)
    )


def _spy_flush(monkeypatch) -> list[float]:
    seen: list[float] = []

    def spy(action, *, timeout_seconds, operation_name):
        del action, operation_name
        seen.append(timeout_seconds)
        return True

    monkeypatch.setattr(observability, "_run_with_timeout", spy)
    return seen


def test_flush_override_reaches_the_flush_wait(monkeypatch):
    _bind_langfuse(monkeypatch, MagicMock())
    seen = _spy_flush(monkeypatch)
    monkeypatch.setattr(config, "langfuse_flush_timeout_seconds", 0.4)

    observability.flush_traces()
    assert seen == [0.4]


def test_flush_explicit_timeout_still_wins(monkeypatch):
    _bind_langfuse(monkeypatch, MagicMock())
    seen = _spy_flush(monkeypatch)

    observability.flush_traces(timeout_seconds=0.3)
    assert seen == [0.3]


def test_flush_is_not_clamped_to_a_caller_deadline(monkeypatch, fake_clock):
    _bind_langfuse(monkeypatch, MagicMock())
    seen = _spy_flush(monkeypatch)
    monkeypatch.setattr(config, "langfuse_flush_timeout_seconds", 2.0)
    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        observability.flush_traces()
    assert seen == [2.0]


def test_stalled_flush_is_abandoned_at_its_bound(monkeypatch):
    release = threading.Event()
    client = MagicMock()
    client.shutdown.side_effect = lambda: release.wait(5)
    _bind_langfuse(monkeypatch, client)
    monkeypatch.setattr(config, "langfuse_flush_timeout_seconds", 0.05)
    mock_logger = MagicMock()
    monkeypatch.setattr(observability, "logger", mock_logger)

    start = time.monotonic()
    try:
        observability.flush_traces()
    finally:
        release.set()
    assert time.monotonic() - start < 2.0
    mock_logger.warning.assert_any_call(
        "langfuse_flush_timeout", operation="shutdown", timeout_seconds=0.05
    )


# --- Market data: ownership sections ---------------------------------------


class _OwnershipTicker:
    """Records each ownership property read; optionally stalls on it."""

    def __init__(self, stall: threading.Event | None = None):
        self.reads: list[str] = []
        self._stall = stall

    def _read(self, name: str) -> Any:
        self.reads.append(name)
        if self._stall is not None:
            self._stall.wait(5)
        return None

    @property
    def institutional_holders(self):
        return self._read("institutional_holders")

    @property
    def insider_transactions(self):
        return self._read("insider_transactions")

    @property
    def major_holders(self):
        return self._read("major_holders")


@pytest.mark.asyncio
async def test_ownership_override_reaches_each_section_wait(monkeypatch):
    monkeypatch.setattr(config, "ownership_section_timeout_seconds", 6.5)
    seen = _spy_hard_timeout(monkeypatch, ownership)

    await ownership._load_ownership_property(
        _OwnershipTicker(), "major_holders", "7203.T"
    )
    assert seen == [6.5]


@pytest.mark.asyncio
async def test_ownership_nested_caller_deadline_wins(monkeypatch, fake_clock):
    seen = _spy_hard_timeout(monkeypatch, ownership)
    with enclosing_deadline(2.0):
        await ownership._load_ownership_property(
            _OwnershipTicker(), "major_holders", "7203.T"
        )
    assert seen == [pytest.approx(2.0)]


@pytest.mark.asyncio
async def test_ownership_spent_deadline_reads_nothing(monkeypatch, fake_clock):
    fake = _OwnershipTicker()
    monkeypatch.setattr(ownership.yf, "Ticker", lambda _symbol: fake)

    with enclosing_deadline(1.0):
        fake_clock[0] += 2.0
        payload = json.loads(
            await ownership.get_ownership_structure.ainvoke({"ticker": "7203.T"})
        )
    assert fake.reads == []
    assert payload["institutional_holders_status"] == "DATA_UNAVAILABLE"
    assert payload["insider_transactions_status"] == "DATA_UNAVAILABLE"
    assert payload["major_holders_status"] == "DATA_UNAVAILABLE"


@pytest.mark.asyncio
async def test_stalled_ownership_section_is_unavailable_at_its_bound(monkeypatch):
    release = threading.Event()
    fake = _OwnershipTicker(stall=release)
    monkeypatch.setattr(ownership.yf, "Ticker", lambda _symbol: fake)
    monkeypatch.setattr(config, "ownership_section_timeout_seconds", 0.05)

    start = time.monotonic()
    try:
        payload = json.loads(
            await ownership.get_ownership_structure.ainvoke({"ticker": "7203.T"})
        )
    finally:
        release.set()
    assert time.monotonic() - start < 2.0
    assert payload["institutional_holders_status"] == "DATA_UNAVAILABLE"
    assert payload["data_quality"] == "PARTIAL"


# --- Market data: StockTwits -----------------------------------------------


class _TimingOutSession:
    """aiohttp.ClientSession stand-in that records its timeout and times out."""

    def __init__(self, record: list[Any], *_a, timeout=None, **_k):
        record.append(timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, *_a, timeout=None, **_k):
        del timeout
        raise asyncio.TimeoutError()


def _bind_stocktwits_session(monkeypatch) -> list[Any]:
    record: list[Any] = []
    monkeypatch.setattr(
        stocktwits_api.aiohttp,
        "ClientSession",
        lambda *a, **k: _TimingOutSession(record, *a, **k),
    )
    return record


@pytest.mark.asyncio
async def test_stocktwits_override_reaches_the_session(monkeypatch):
    record = _bind_stocktwits_session(monkeypatch)
    monkeypatch.setattr(config, "stocktwits_http_timeout_seconds", 7.5)

    result = await stocktwits_api.StockTwitsAPI().get_sentiment("AAPL")
    assert [t.total for t in record] == [7.5]
    # Error handling: a request timeout is today's error payload.
    assert result == {"error": "TimeoutError (details in operator logs)"}


@pytest.mark.asyncio
async def test_stocktwits_nested_caller_deadline_wins(monkeypatch, fake_clock):
    record = _bind_stocktwits_session(monkeypatch)
    with enclosing_deadline(2.0):
        await stocktwits_api.StockTwitsAPI().get_sentiment("AAPL")
    assert [t.total for t in record] == [pytest.approx(2.0)]


@pytest.mark.asyncio
async def test_stocktwits_spent_deadline_opens_no_session(monkeypatch, fake_clock):
    record = _bind_stocktwits_session(monkeypatch)
    with enclosing_deadline(1.0):
        fake_clock[0] += 2.0
        result = await stocktwits_api.StockTwitsAPI().get_sentiment("AAPL")
    assert record == []
    assert result == {"error": "TimeoutError (details in operator logs)"}


# --- Market data: retrospective price fetch --------------------------------

_OLD_SNAPSHOT = {
    "ticker": "7203.T",
    "analysis_date": "2025-01-06",
    "verdict": "BUY",
    "current_price": 100.0,
}


@pytest.mark.asyncio
async def test_retrospective_override_reaches_the_fetch_wait(monkeypatch):
    monkeypatch.setattr(config, "retrospective_price_fetch_timeout_seconds", 9.5)
    seen = _spy_hard_timeout(monkeypatch, retrospective, result={})

    assert await retrospective.compare_to_reality(dict(_OLD_SNAPSHOT)) is None
    assert seen == [9.5]


@pytest.mark.asyncio
async def test_retrospective_nested_caller_deadline_wins(monkeypatch, fake_clock):
    seen = _spy_hard_timeout(monkeypatch, retrospective, result={})
    with enclosing_deadline(4.0):
        await retrospective.compare_to_reality(dict(_OLD_SNAPSHOT))
    assert seen == [pytest.approx(4.0)]


@pytest.mark.asyncio
async def test_retrospective_spent_deadline_skips_the_snapshot(monkeypatch, fake_clock):
    seen = _spy_hard_timeout(monkeypatch, retrospective, result={})
    with enclosing_deadline(1.0):
        fake_clock[0] += 2.0
        assert await retrospective.compare_to_reality(dict(_OLD_SNAPSHOT)) is None
    assert seen == []


@pytest.mark.asyncio
async def test_retrospective_timeout_skips_the_snapshot(monkeypatch):
    async def _times_out(coro, *, timeout, label):
        del timeout, label
        coro.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr(retrospective, "run_with_hard_timeout", _times_out)
    assert await retrospective.compare_to_reality(dict(_OLD_SNAPSHOT)) is None


# --- Market data: advisory market note and price snapshot ------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field, call",
    [
        (
            "market_context_fetch_timeout_seconds",
            lambda: main_module._fetch_market_context("7203.T", "2026-03-05"),
        ),
        (
            "price_snapshot_fetch_timeout_seconds",
            lambda: main_module._fetch_price_snapshot("7203.T"),
        ),
    ],
    ids=["market_context", "price_snapshot"],
)
class TestAdvisoryHistoryFetches:
    async def test_override_reaches_the_fetch_wait(self, monkeypatch, field, call):
        monkeypatch.setattr(config, field, 3.5)
        seen = _spy_hard_timeout(monkeypatch, main_module)
        await call()
        assert seen == [3.5]

    async def test_nested_caller_deadline_wins(
        self, monkeypatch, fake_clock, field, call
    ):
        seen = _spy_hard_timeout(monkeypatch, main_module)
        with enclosing_deadline(1.5):
            await call()
        assert seen == [pytest.approx(1.5)]

    async def test_spent_deadline_starts_no_fetch(
        self, monkeypatch, fake_clock, field, call
    ):
        seen = _spy_hard_timeout(monkeypatch, main_module)
        with enclosing_deadline(1.0):
            fake_clock[0] += 2.0
            result = await call()
        assert seen == []
        assert not result  # "" for the note, None for the snapshot

    async def test_timeout_takes_the_advisory_fallback(self, monkeypatch, field, call):
        async def _times_out(coro, *, timeout, label):
            del timeout, label
            coro.close()
            raise asyncio.TimeoutError()

        monkeypatch.setattr(main_module, "run_with_hard_timeout", _times_out)
        assert not await call()


# --- Market data: report-header company name -------------------------------


def _spy_blocking_call(monkeypatch, result: Any = None) -> list[Any]:
    seen: list[Any] = []

    async def spy(policy, fn):
        del fn
        seen.append(policy)
        return result

    monkeypatch.setattr(blocking_io, "run_blocking_call", spy)
    return seen


def _spy_blocking_call_sync(monkeypatch, result: Any = None) -> list[Any]:
    seen: list[Any] = []

    def spy(policy, fn):
        del fn
        seen.append(policy)
        return result

    monkeypatch.setattr(blocking_io, "run_blocking_call_sync", spy)
    return seen


@pytest.mark.asyncio
async def test_output_name_override_reaches_the_whole_lookup(monkeypatch):
    monkeypatch.setattr(config, "output_company_name_timeout_seconds", 4.5)
    seen = _spy_blocking_call(monkeypatch, result="Toyota Motor Corporation")

    name = await output._load_company_name_for_output_async("7203.T")
    assert name == "Toyota Motor Corporation"
    assert [p.hard_timeout_seconds for p in seen] == [4.5]


@pytest.mark.asyncio
async def test_output_name_nested_caller_deadline_wins(monkeypatch, fake_clock):
    seen = _spy_blocking_call(monkeypatch)
    with enclosing_deadline(2.5):
        await output._load_company_name_for_output_async("7203.T")
    assert [p.hard_timeout_seconds for p in seen] == [pytest.approx(2.5)]


@pytest.mark.asyncio
async def test_output_name_spent_deadline_starts_no_lookup(monkeypatch, fake_clock):
    seen = _spy_blocking_call(monkeypatch)
    with enclosing_deadline(1.0):
        fake_clock[0] += 2.0
        assert await output._load_company_name_for_output_async("7203.T") is None
    assert seen == []


def test_output_candidate_override_reaches_each_worker(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", MagicMock())
    monkeypatch.setattr(config, "company_name_lookup_timeout_seconds", 3.25)
    seen = _spy_blocking_call_sync(
        monkeypatch, result={"longName": "Toyota Motor Corporation"}
    )

    assert output._load_company_name_for_output("7203.T") == (
        "Toyota Motor Corporation"
    )
    assert [p.hard_timeout_seconds for p in seen] == [3.25]
    # The slot cap rides along on the per-call copy.
    assert seen[0].in_flight is OUTPUT_COMPANY_NAME_INFO_POLICY.in_flight


def test_output_candidate_nested_caller_deadline_wins(monkeypatch, fake_clock):
    monkeypatch.setitem(sys.modules, "yfinance", MagicMock())
    seen = _spy_blocking_call_sync(monkeypatch, result={})
    with enclosing_deadline(1.25):
        output._load_company_name_for_output("7203.T")
    assert seen
    assert all(p.hard_timeout_seconds == pytest.approx(1.25) for p in seen), [
        p.hard_timeout_seconds for p in seen
    ]


def test_output_candidate_spent_deadline_starts_no_worker(monkeypatch, fake_clock):
    monkeypatch.setitem(sys.modules, "yfinance", MagicMock())
    seen = _spy_blocking_call_sync(monkeypatch)
    with enclosing_deadline(1.0):
        fake_clock[0] += 2.0
        assert output._load_company_name_for_output("7203.T") is None
    assert seen == []


def test_output_candidate_saturation_falls_back_to_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", MagicMock())

    def saturated(policy, fn):
        del fn
        raise BlockingCallSaturatedError(policy.label)

    monkeypatch.setattr(blocking_io, "run_blocking_call_sync", saturated)
    assert output._load_company_name_for_output("7203.T") is None


@pytest.mark.asyncio
async def test_stalled_name_lookup_returns_by_the_deadline_via_async_path(
    monkeypatch,
):
    """End to end: the async caller returns at its bound; the worker is a daemon."""
    release = threading.Event()
    workers: list[threading.Thread] = []

    class _StalledTicker:
        def __init__(self, _symbol):
            pass

        @property
        def info(self):
            workers.append(threading.current_thread())
            release.wait(5)
            return {}

    fake_yf = MagicMock()
    fake_yf.Ticker.side_effect = _StalledTicker
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    monkeypatch.setattr(config, "output_company_name_timeout_seconds", 1.0)
    monkeypatch.setattr(config, "company_name_lookup_timeout_seconds", 0.05)

    start = time.monotonic()
    try:
        assert await output._load_company_name_for_output_async("7203.T") is None
    finally:
        release.set()
    assert time.monotonic() - start < 1.0
    assert workers and all(w.daemon for w in workers)
