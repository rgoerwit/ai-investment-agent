"""One deadline per embedding attempt; physical requests counted per provider.

Each attempt of ``FinancialSituationMemory._get_embedding`` has one absolute
deadline covering the rate-limiter wait and the request. The unthrottled
fallback is decided only at setup (no limiter can be had), never after an error
from the wait or the request. Tenacity's backoff is clamped to the caller's
enclosing deadline and stops once less than the embedding minimum remains,
re-raising the attempt's own error.

Every test calls the real decorated method. Sizing uses a frozen deadline
clock; elapsed-time checks run against a stall the test releases itself. The
provider tests patch the HTTP layer each SDK really uses (aiohttp for the Google
GenAI async client, ``httpx2`` for the OpenAI SDK), with dummy keys and sockets
disabled, so the SDK's own retry machinery runs and no request can leave.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlparse

import aiohttp
import pytest
from google.genai import errors as genai_errors
from langchain_openai import OpenAIEmbeddings
from openai import InternalServerError

import src.memory as memory_module
from src.config import config
from src.embeddings import (
    EmbeddingBinding,
    GoogleGenerativeAIEmbeddings,
    build_embeddings,
    resolve_embedding_binding,
)
from src.llm_runtime import rate_limits
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline
from src.memory import EmbeddingAllowanceSpent, FinancialSituationMemory
from src.runtime_services import (
    ProviderRuntime,
    ProviderRuntimeKey,
    RuntimeServices,
    use_runtime_services,
)
from src.tooling.inspection_service import InspectionService
from src.tooling.runtime import ToolExecutionService

_NOW = 1_000.0
_VECTOR = [0.1] * 8


# --- Fixtures and fakes ------------------------------------------------------


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    """A frozen deadline clock; advance via clock[0]."""
    clock = [_NOW]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


@pytest.fixture
def retry_sleeps(monkeypatch) -> list[float]:
    """Record tenacity's backoff sleeps instead of sleeping."""
    sleeps: list[float] = []

    async def _record(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(FinancialSituationMemory._get_embedding.retry, "sleep", _record)
    return sleeps


class CountingLimiter:
    """An instant limiter that counts acquisitions."""

    def __init__(self) -> None:
        self.acquired = 0

    async def aacquire(self, *, blocking: bool = True) -> bool:
        self.acquired += 1
        return True


class GatedLimiter:
    """A limiter whose wait stalls until the test releases it."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.entered = 0
        self.acquired = 0

    async def aacquire(self, *, blocking: bool = True) -> bool:
        self.entered += 1
        await self.gate.wait()
        self.acquired += 1
        return True


class ClockLimiter:
    """An instant limiter whose wait spends fake-clock time."""

    def __init__(self, clock: list[float], seconds: float) -> None:
        self.clock = clock
        self.seconds = seconds
        self.acquired = 0

    async def aacquire(self, *, blocking: bool = True) -> bool:
        self.clock[0] += self.seconds
        self.acquired += 1
        return True


class Requests:
    """A fake ``aembed_query`` that counts sends and runs a scripted behaviour."""

    def __init__(self, behaviour: Callable[[], Any] | None = None) -> None:
        self.count = 0
        self.behaviour = behaviour

    async def __call__(self, _text: str) -> list[float]:
        self.count += 1
        if self.behaviour is not None:
            result = self.behaviour()
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, BaseException):
                raise result
        return _VECTOR


def _memory(monkeypatch, embeddings: Any, limiter: Any) -> FinancialSituationMemory:
    """A memory object built without its constructor: no Chroma client at all.

    ``limiter`` stands in for whatever the binding's provider resolves to.
    """
    monkeypatch.setattr(memory_module, "_embedding_rate_limiter", lambda _b: limiter)
    memory = FinancialSituationMemory.__new__(FinancialSituationMemory)
    memory.name = "TEST_embedding_deadline"
    memory.embedding_binding = resolve_embedding_binding(config)
    memory.available = True
    memory.embeddings = embeddings
    memory.situation_collection = MagicMock()
    return memory


def _embeddings(requests: Requests) -> Any:
    embeddings = MagicMock()
    embeddings.aembed_query = requests
    return embeddings


def _small_scopes(monkeypatch, *, call: float, minimum: float) -> None:
    """Deliberately small per-attempt and minimum overrides."""
    monkeypatch.setattr(config, "memory_embedding_call_timeout_seconds", call)
    monkeypatch.setattr(config, "min_useful_attempt_fraction", minimum / call)


# --- One deadline covers the limiter wait and the request -------------------


@pytest.mark.asyncio
async def test_slow_limiter_returns_at_a_20ms_deadline_and_sends_nothing(
    monkeypatch, retry_sleeps
):
    requests = Requests()
    limiter = GatedLimiter()
    memory = _memory(monkeypatch, _embeddings(requests), limiter)
    _small_scopes(monkeypatch, call=0.02, minimum=0.005)
    # Released by the test after 1 s: without the bound the call would return
    # then, with a request sent, and fail both assertions below.
    asyncio.get_running_loop().call_later(1.0, limiter.gate.set)

    start = time.monotonic()
    with enclosing_deadline(0.02), pytest.raises(TimeoutError, match="rate-limiter"):
        await memory._get_embedding("text")
    elapsed = time.monotonic() - start

    assert 0.015 <= elapsed < 0.5
    assert requests.count == 0
    assert limiter.acquired == 0
    # The single attempt used the whole deadline: no retry was admitted.
    assert limiter.entered == 1
    assert retry_sleeps == []
    limiter.gate.set()


@pytest.mark.asyncio
async def test_limiter_wait_and_request_share_one_deadline(monkeypatch, fake_clock):
    seen: list[float] = []

    async def spy(coro, *, timeout, label):
        del label
        coro.close()
        seen.append(timeout)
        return _VECTOR

    limiter = ClockLimiter(fake_clock, seconds=4.0)
    memory = _memory(monkeypatch, _embeddings(Requests()), limiter)
    monkeypatch.setattr(config, "memory_embedding_call_timeout_seconds", 10.0)
    monkeypatch.setattr(memory_module, "run_with_hard_timeout", spy)

    assert await memory._get_embedding("text") == _VECTOR
    assert seen == [pytest.approx(6.0)]


@pytest.mark.asyncio
async def test_limiter_wait_that_spends_the_attempt_sends_nothing(
    monkeypatch, fake_clock, retry_sleeps
):
    requests = Requests()
    limiter = ClockLimiter(fake_clock, seconds=10.0)
    memory = _memory(monkeypatch, _embeddings(requests), limiter)
    monkeypatch.setattr(config, "memory_embedding_call_timeout_seconds", 10.0)

    with enclosing_deadline(10.0), pytest.raises(TimeoutError, match="before the"):
        await memory._get_embedding("text")
    assert requests.count == 0
    assert limiter.acquired == 1
    assert retry_sleeps == []


# --- Provider errors never re-send outside the limiter ----------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [AttributeError("attr"), TypeError("type"), RuntimeError("provider")]
)
async def test_provider_error_is_not_re_sent_directly(
    monkeypatch, fake_clock, retry_sleeps, error
):
    requests = Requests(lambda: error)
    limiter = CountingLimiter()
    memory = _memory(monkeypatch, _embeddings(requests), limiter)
    monkeypatch.setattr(memory_module, "logger", MagicMock())

    with pytest.raises(type(error)):
        await memory._get_embedding("text")
    # One request per attempt, each behind a limiter token: no fallback send.
    assert requests.count == limiter.acquired == 3
    memory_module.logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_provider_timeout_sends_one_request_and_surfaces_the_timeout(
    monkeypatch, retry_sleeps
):
    release = asyncio.Event()

    async def stall() -> None:
        await release.wait()

    requests = Requests(stall)
    limiter = CountingLimiter()
    memory = _memory(monkeypatch, _embeddings(requests), limiter)
    _small_scopes(monkeypatch, call=0.05, minimum=0.03)
    asyncio.get_running_loop().call_later(1.0, release.set)
    monkeypatch.setattr(memory_module, "logger", MagicMock())

    start = time.monotonic()
    try:
        with (
            enclosing_deadline(0.06),
            pytest.raises(TimeoutError, match="hard timeout"),
        ):
            await memory._get_embedding("text")
        assert time.monotonic() - start < 0.5
        assert requests.count == 1
        assert limiter.acquired == 1
    finally:
        # The stalled request is an orphaned loop task, not a thread worker:
        # release it, then let the loop finish it before the patches are undone.
        release.set()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_limiter_wait_error_is_the_attempts_error_not_a_fallback(
    monkeypatch, retry_sleeps
):
    class FailingLimiter:
        async def aacquire(self, *, blocking: bool = True) -> bool:
            raise RuntimeError("limiter wait failed")

    requests = Requests()
    memory = _memory(monkeypatch, _embeddings(requests), FailingLimiter())

    with pytest.raises(RuntimeError, match="limiter wait failed"):
        await memory._get_embedding("text")
    assert requests.count == 0


# --- The limiter comes from the embedding binding's provider -----------------


def _runtime_with(limiters: dict[str, Any]) -> RuntimeServices:
    """Runtime services whose provider runtime holds one limiter per vendor."""
    providers = ProviderRuntime(
        fetcher=SimpleNamespace(),
        rate_limiter=limiters.get("google", CountingLimiter()),
        rate_limiters={ProviderRuntimeKey(v): lim for v, lim in limiters.items()},
    )
    return RuntimeServices(
        tool_service=ToolExecutionService(),
        inspection_service=InspectionService(),
        providers=providers,
    )


def _bound_memory(monkeypatch, provider: str, requests: Requests) -> Any:
    """A constructor-free memory whose limiter is resolved for real."""
    memory = FinancialSituationMemory.__new__(FinancialSituationMemory)
    memory.name = "TEST_embedding_limiter"
    memory.embedding_binding = EmbeddingBinding(
        provider=provider, model="m", dimension=8
    )
    memory.available = True
    memory.embeddings = _embeddings(requests)
    memory.situation_collection = MagicMock()
    return memory


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["google", "openai"])
async def test_the_binding_provider_selects_the_runtime_limiter(monkeypatch, provider):
    limiters = {"google": CountingLimiter(), "openai": CountingLimiter()}
    requests = Requests()
    memory = _bound_memory(monkeypatch, provider, requests)

    with use_runtime_services(_runtime_with(limiters)):
        assert await memory._get_embedding("text") == _VECTOR

    other = "openai" if provider == "google" else "google"
    assert limiters[provider].acquired == 1
    assert limiters[other].acquired == 0
    assert requests.count == 1


@pytest.mark.real_rate_limiter
def test_an_unscoped_caller_gets_the_providers_process_fallback(monkeypatch):
    rate_limits.reset_fallback_limiters_for_tests()
    binding = EmbeddingBinding(provider="openai", model="m", dimension=8)

    limiter = memory_module._embedding_rate_limiter(binding)

    assert limiter is rate_limits.limiter_for_binding(config, "openai", None)
    assert limiter.requests_per_second == pytest.approx(
        config.openai_rpm_limit / 60.0 * 0.8
    )
    rate_limits.reset_fallback_limiters_for_tests()


def test_an_explicit_openai_base_keys_the_limiter_by_its_host(monkeypatch):
    seen: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        memory_module,
        "limiter_for_binding",
        lambda _s, vendor, host: seen.append((vendor, host)),
    )
    monkeypatch.setattr(config, "openai_api_base", "https://api.openai.com/v1")
    memory_module._embedding_rate_limiter(
        EmbeddingBinding(provider="openai", model="m", dimension=8)
    )
    # A compatible chat-plane base is not where embeddings go: no host.
    monkeypatch.setattr(config, "openai_api_base", "https://api.moonshot.ai/v1")
    memory_module._embedding_rate_limiter(
        EmbeddingBinding(provider="openai", model="m", dimension=8)
    )
    assert seen == [("openai", "api.openai.com"), ("openai", None)]


@pytest.mark.asyncio
async def test_no_configured_limiter_sends_without_a_wait_or_a_warning(
    monkeypatch, retry_sleeps
):
    requests = Requests()
    memory = _bound_memory(monkeypatch, "unrated-vendor", requests)
    monkeypatch.setattr(memory_module, "logger", MagicMock())

    with use_runtime_services(_runtime_with({})):
        assert await memory._get_embedding("text") == _VECTOR
    assert requests.count == 1
    memory_module.logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_a_limiter_that_cannot_be_resolved_fails_the_attempt_unsent(
    monkeypatch, retry_sleeps
):
    def broken(_binding: Any) -> Any:
        raise RuntimeError("limiter construction failed")

    requests = Requests()
    memory = _memory(monkeypatch, _embeddings(requests), None)
    monkeypatch.setattr(memory_module, "_embedding_rate_limiter", broken)

    with pytest.raises(RuntimeError, match="limiter construction failed"):
        await memory._get_embedding("text")
    # Never sent outside the limiter, on any of the three attempts.
    assert requests.count == 0
    assert len(retry_sleeps) == 2


@pytest.mark.asyncio
async def test_no_limiter_request_still_has_the_attempt_deadline(
    monkeypatch, fake_clock
):
    seen: list[float] = []

    async def spy(coro, *, timeout, label):
        del label
        coro.close()
        seen.append(timeout)
        return _VECTOR

    memory = _memory(monkeypatch, _embeddings(Requests()), None)
    monkeypatch.setattr(memory_module, "run_with_hard_timeout", spy)

    with enclosing_deadline(4.0):
        await memory._get_embedding("text")
    assert seen == [pytest.approx(4.0)]


# --- Tenacity: clamped backoff, deadline stop, original error ---------------


@pytest.mark.asyncio
async def test_backoff_is_clamped_to_leave_a_useful_attempt(monkeypatch, fake_clock):
    sleeps: list[float] = []

    async def advance(seconds: float) -> None:
        sleeps.append(seconds)
        fake_clock[0] += seconds

    monkeypatch.setattr(FinancialSituationMemory._get_embedding.retry, "sleep", advance)
    requests = Requests(lambda: RuntimeError("provider"))
    memory = _memory(monkeypatch, _embeddings(requests), CountingLimiter())
    _small_scopes(monkeypatch, call=25.0, minimum=1.0)

    with enclosing_deadline(3.5), pytest.raises(RuntimeError, match="provider"):
        await memory._get_embedding("text")
    # Unclamped backoff is 2 s then 2 s; the second is cut to leave the 1 s minimum (0.04 x 25 s).
    assert sleeps == [pytest.approx(2.0), pytest.approx(0.5)]
    assert requests.count == 3


@pytest.mark.asyncio
async def test_backoff_that_cannot_fit_stops_and_reraises_the_original_error(
    monkeypatch, fake_clock
):
    sleeps: list[float] = []

    async def advance(seconds: float) -> None:
        sleeps.append(seconds)
        fake_clock[0] += seconds

    def slow_failure() -> BaseException:
        fake_clock[0] += 1.0
        return RuntimeError("provider")

    monkeypatch.setattr(FinancialSituationMemory._get_embedding.retry, "sleep", advance)
    requests = Requests(slow_failure)
    memory = _memory(monkeypatch, _embeddings(requests), CountingLimiter())
    _small_scopes(monkeypatch, call=25.0, minimum=1.0)

    # t=0 attempt fails at t=1; sleep 1 s; t=2 attempt fails at t=3: nothing fits.
    with enclosing_deadline(3.0), pytest.raises(RuntimeError, match="provider"):
        await memory._get_embedding("text")
    assert sleeps == [pytest.approx(1.0)]
    assert requests.count == 2


@pytest.mark.asyncio
async def test_less_than_the_minimum_left_refuses_before_any_token(
    monkeypatch, fake_clock, retry_sleeps
):
    requests = Requests()
    limiter = CountingLimiter()
    memory = _memory(monkeypatch, _embeddings(requests), limiter)

    with enclosing_deadline(0.5), pytest.raises(EmbeddingAllowanceSpent):
        await memory._get_embedding("text")
    assert requests.count == limiter.acquired == 0
    assert retry_sleeps == []


@pytest.mark.asyncio
async def test_small_minimum_override_admits_a_short_remaining_allowance(
    monkeypatch, fake_clock
):
    seen: list[float] = []

    async def spy(coro, *, timeout, label):
        del label
        coro.close()
        seen.append(timeout)
        return _VECTOR

    memory = _memory(monkeypatch, _embeddings(Requests()), CountingLimiter())
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.1 / 30.0)
    monkeypatch.setattr(memory_module, "run_with_hard_timeout", spy)

    with enclosing_deadline(0.5):
        await memory._get_embedding("text")
    assert seen == [pytest.approx(0.5)]


def _spy_timeouts(monkeypatch) -> list[float]:
    seen: list[float] = []

    async def spy(coro, *, timeout, label):
        del label
        coro.close()
        seen.append(timeout)
        return _VECTOR

    monkeypatch.setattr(memory_module, "run_with_hard_timeout", spy)
    return seen


@pytest.mark.asyncio
async def test_remaining_exactly_the_derived_minimum_is_admitted(
    monkeypatch, fake_clock
):
    """Minimum = fraction x the configured allowance: 0.0625 x 32 s = 2 s."""
    monkeypatch.setattr(config, "memory_embedding_call_timeout_seconds", 32.0)
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.0625)
    seen = _spy_timeouts(monkeypatch)
    requests = Requests()
    memory = _memory(monkeypatch, _embeddings(requests), CountingLimiter())

    with enclosing_deadline(2.0):
        await memory._get_embedding("text")
    assert seen == [2.0]


@pytest.mark.asyncio
async def test_remaining_just_below_the_derived_minimum_sends_nothing(
    monkeypatch, fake_clock, retry_sleeps
):
    monkeypatch.setattr(config, "memory_embedding_call_timeout_seconds", 32.0)
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.0625)
    requests = Requests()
    limiter = CountingLimiter()
    memory = _memory(monkeypatch, _embeddings(requests), limiter)

    with enclosing_deadline(2.0 - 0.001), pytest.raises(EmbeddingAllowanceSpent):
        await memory._get_embedding("text")
    assert requests.count == limiter.acquired == 0
    assert retry_sleeps == []


@pytest.mark.asyncio
async def test_minimum_follows_the_configured_allowance_not_the_clamped_one(
    monkeypatch, fake_clock
):
    """A 2.5 s remainder clamps the attempt to 2.5 s; the minimum stays 0.0625 x 32.

    Derived from the clamped remainder (2.5 s) the minimum would be 0.16 s and a
    0.5 s remainder would pass; it must be refused at the configured 2 s.
    """
    monkeypatch.setattr(config, "memory_embedding_call_timeout_seconds", 32.0)
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.0625)
    seen = _spy_timeouts(monkeypatch)
    memory = _memory(monkeypatch, _embeddings(Requests()), CountingLimiter())

    with enclosing_deadline(2.5):
        await memory._get_embedding("text")
    assert seen == [2.5]
    with enclosing_deadline(0.5), pytest.raises(EmbeddingAllowanceSpent):
        await memory._get_embedding("text")


@pytest.mark.asyncio
async def test_tiny_allowance_with_a_small_fraction_still_admits(
    monkeypatch, fake_clock
):
    monkeypatch.setattr(config, "memory_embedding_call_timeout_seconds", 0.01)
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.04)
    seen = _spy_timeouts(monkeypatch)
    memory = _memory(monkeypatch, _embeddings(Requests()), CountingLimiter())

    with enclosing_deadline(0.01):
        await memory._get_embedding("text")
    assert seen == [pytest.approx(0.01)]


@pytest.mark.asyncio
async def test_a_fraction_override_moves_the_embedding_minimum(monkeypatch, fake_clock):
    """At the default 0.04 x 30 s = 1.2 s a 1 s remainder is refused; at 0.025 it is not."""
    seen = _spy_timeouts(monkeypatch)
    memory = _memory(monkeypatch, _embeddings(Requests()), CountingLimiter())
    with enclosing_deadline(1.0), pytest.raises(EmbeddingAllowanceSpent):
        await memory._get_embedding("text")
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.025)
    with enclosing_deadline(1.0):
        await memory._get_embedding("text")
    assert seen == [1.0]


# --- Cancellation and concurrent callers ------------------------------------


@pytest.mark.asyncio
async def test_cancel_during_limiter_wait_sends_nothing_and_is_not_retried(
    monkeypatch, retry_sleeps
):
    requests = Requests()
    limiter = GatedLimiter()
    memory = _memory(monkeypatch, _embeddings(requests), limiter)

    task = asyncio.ensure_future(memory._get_embedding("text"))
    while limiter.entered == 0:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    limiter.gate.set()
    await asyncio.sleep(0)
    assert requests.count == 0
    assert limiter.entered == 1
    assert retry_sleeps == []


@pytest.mark.asyncio
async def test_cancel_during_request_cancels_it_and_is_not_retried(
    monkeypatch, retry_sleeps
):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def stall() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    requests = Requests(stall)
    limiter = CountingLimiter()
    memory = _memory(monkeypatch, _embeddings(requests), limiter)

    task = asyncio.ensure_future(memory._get_embedding("text"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    assert requests.count == limiter.acquired == 1
    assert retry_sleeps == []


@pytest.mark.asyncio
async def test_concurrent_callers_keep_their_own_deadlines(monkeypatch, fake_clock):
    seen: dict[str, float] = {}

    async def spy(coro, *, timeout, label):
        del label
        coro.close()
        await asyncio.sleep(0)
        seen[asyncio.current_task().get_name()] = timeout
        return _VECTOR

    memory = _memory(monkeypatch, _embeddings(Requests()), CountingLimiter())
    monkeypatch.setattr(memory_module, "run_with_hard_timeout", spy)

    async def caller(seconds: float) -> None:
        with enclosing_deadline(seconds):
            await memory._get_embedding("text")

    await asyncio.gather(
        asyncio.create_task(caller(3.0), name="short"),
        asyncio.create_task(caller(8.0), name="long"),
    )
    assert seen == {"short": pytest.approx(3.0), "long": pytest.approx(8.0)}


@pytest.mark.asyncio
async def test_a_timed_out_waiter_does_not_disturb_a_concurrent_one(
    monkeypatch, retry_sleeps
):
    requests = Requests()
    limiter = GatedLimiter()
    memory = _memory(monkeypatch, _embeddings(requests), limiter)
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.005 / 30.0)

    async def short() -> None:
        with enclosing_deadline(0.02):
            await memory._get_embedding("text")

    patient = asyncio.ensure_future(memory._get_embedding("text"))
    with pytest.raises(TimeoutError):
        await short()
    assert requests.count == 0
    limiter.gate.set()
    assert await asyncio.wait_for(patient, timeout=1.0) == _VECTOR
    assert requests.count == 1


# --- Physical requests per provider, at the HTTP layer -----------------------


@pytest.fixture
def no_httpx_escape(monkeypatch) -> None:
    """Fail loudly if any request reaches the plain httpx transport."""
    import httpx

    async def refuse(self, request):
        raise AssertionError(f"unexpected httpx request to {request.url}")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)


@pytest.fixture
def openai_wire(monkeypatch, no_httpx_escape) -> list[str]:
    """The OpenAI SDK's transport (httpx2): answer 503, ask for a 1 ms retry."""
    import httpx2

    sent: list[str] = []

    async def respond(self, request):
        sent.append(str(request.url))
        return httpx2.Response(
            503,
            json={"error": {"message": "unavailable"}},
            headers={"retry-after-ms": "1"},
            request=request,
        )

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", respond)
    return sent


def _openai_embeddings(*, max_retries: int | None = None) -> Any:
    settings = SimpleNamespace(
        openai_api_key="sk-test-dummy-not-a-key",
        get_openai_api_base=lambda: None,
    )
    embeddings = build_embeddings(
        EmbeddingBinding(
            provider="openai", model="text-embedding-3-small", dimension=8
        ),
        settings,  # type: ignore[arg-type]
    )
    # Raw text, not tiktoken tokens: tokenizing could fetch an encoding file.
    embeddings.check_embedding_ctx_length = False
    if max_retries is not None:
        # The control: the same class at the SDK's retry setting.
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key="sk-test-dummy-not-a-key",
            dimensions=8,
            max_retries=max_retries,
            check_embedding_ctx_length=False,
        )
    return embeddings


@pytest.mark.security
@pytest.mark.asyncio
async def test_openai_sends_one_physical_request_per_attempt(
    monkeypatch, retry_sleeps, openai_wire
):
    memory = _memory(monkeypatch, _openai_embeddings(), CountingLimiter())

    with pytest.raises(InternalServerError, match="unavailable"):
        await memory._get_embedding("text")
    assert len(openai_wire) == 3  # three application attempts, one request each
    assert all(url.endswith("/embeddings") for url in openai_wire)


@pytest.mark.security
@pytest.mark.asyncio
async def test_openai_transport_counter_sees_sdk_retries(
    monkeypatch, retry_sleeps, openai_wire
):
    # Control: the SDK default (two retries) would triple every attempt.
    memory = _memory(monkeypatch, _openai_embeddings(max_retries=2), CountingLimiter())

    with pytest.raises(InternalServerError, match="unavailable"):
        await memory._get_embedding("text")
    assert len(openai_wire) == 9


class _InstantSleepAsyncio:
    """The asyncio module as google-genai sees it, with an instant sleep."""

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    @staticmethod
    async def sleep(_seconds: float) -> None:
        return None


@pytest.fixture
def google_wire(monkeypatch, no_httpx_escape) -> SimpleNamespace:
    """The Google GenAI async transport (aiohttp): count and script each send."""
    wire = SimpleNamespace(sent=[], disconnect=False)

    async def respond(self, method, url, **kwargs):
        del self, method, kwargs
        wire.sent.append(str(url))
        if wire.disconnect:
            raise aiohttp.ServerDisconnectedError()
        # A replay-shaped error response, which raise_for_async_response reads.
        return SimpleNamespace(
            status_code=503,
            body_segments=[
                {"error": {"code": 503, "message": "x", "status": "UNAVAILABLE"}}
            ],
            headers={},
        )

    monkeypatch.setattr(aiohttp.ClientSession, "_request", respond)
    return wire


def _google_embeddings() -> Any:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key="test-dummy-not-a-key",
        task_type="retrieval_document",
        output_dimensionality=8,
    )


@pytest.mark.security
@pytest.mark.asyncio
async def test_google_sends_one_physical_request_per_attempt_on_http_error(
    monkeypatch, retry_sleeps, google_wire
):
    memory = _memory(monkeypatch, _google_embeddings(), CountingLimiter())

    with pytest.raises(Exception, match="503") as raised:
        await memory._get_embedding("text")
    # The SDK's own error for the HTTP response, not a wrapper's.
    assert isinstance(raised.value.__cause__, genai_errors.ServerError)
    assert len(google_wire.sent) == 3  # retry_options=None: one SDK attempt
    assert all(
        urlparse(url).hostname == "generativelanguage.googleapis.com"
        for url in google_wire.sent
    )


@pytest.mark.security
@pytest.mark.asyncio
async def test_google_aiohttp_disconnect_resends_once_within_the_attempt(
    monkeypatch, retry_sleeps, google_wire
):
    """Documents what remains: google-genai's aiohttp path re-sends once after a
    connection-level error (1-10 s sleep), outside ``retry_options``."""
    import google.genai._api_client as api_client

    monkeypatch.setattr(api_client, "asyncio", _InstantSleepAsyncio())
    google_wire.disconnect = True
    memory = _memory(monkeypatch, _google_embeddings(), CountingLimiter())

    with pytest.raises(Exception, match="Server disconnected") as raised:
        await memory._get_embedding("text")
    assert isinstance(raised.value.__cause__, aiohttp.ServerDisconnectedError)
    assert len(google_wire.sent) == 6  # three attempts, two sends each


@pytest.mark.security
@pytest.mark.asyncio
async def test_google_resend_is_cut_off_by_the_attempt_deadline(
    monkeypatch, retry_sleeps, google_wire
):
    # The SDK's real 1-10 s pre-resend sleep outlasts a 50 ms attempt deadline.
    google_wire.disconnect = True
    memory = _memory(monkeypatch, _google_embeddings(), CountingLimiter())
    _small_scopes(monkeypatch, call=0.05, minimum=0.03)

    start = time.monotonic()
    with enclosing_deadline(0.06), pytest.raises(TimeoutError):
        await memory._get_embedding("text")
    assert time.monotonic() - start < 0.5
    assert len(google_wire.sent) == 1
