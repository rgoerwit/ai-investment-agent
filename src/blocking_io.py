"""Hard wall-clock bounds for blocking calls, from async and sync callers.

Two dispatch modes share one policy type:

* An unlimited policy (``in_flight=None``) runs the call through
  ``asyncio.to_thread`` under ``run_with_hard_timeout``. The caller is bounded;
  a stalled thread is orphaned in the loop's default executor.
* A policy with an ``InFlightLimit`` runs each call on a dedicated **daemon**
  thread and holds a slot until that thread actually finishes. Repeated stalls
  therefore cannot grow the orphan population past the limit, a saturated
  policy fails fast instead of queueing, and a stuck worker does not hold the
  interpreter open at exit (``ThreadPoolExecutor`` workers — including the
  ``to_thread`` default executor — are joined at exit; daemon threads are not).

Neither mode stops the stalled work itself: a thread parked in a C-level
socket read cannot be interrupted from Python. What is bounded is how long the
*caller* waits and, for limited policies, how many such threads can pile up.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import TypeVar

import structlog

from src.async_utils import run_with_hard_timeout

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class BlockingCallSaturatedError(TimeoutError):
    """Every in-flight slot of a policy is held by a call that has not finished.

    Subclasses ``TimeoutError`` so callers that already treat a timeout as
    "no data" handle saturation the same way without new except clauses.
    """


class BlockingCallTimeoutError(TimeoutError):
    """``run_blocking_call_sync``'s own deadline passed; the worker is orphaned.

    Distinct from a ``TimeoutError`` raised *by* the wrapped call, which is
    re-raised unchanged.
    """


class InFlightLimit:
    """Thread-safe count of worker threads that have started and not finished."""

    def __init__(self, max_in_flight: int) -> None:
        if max_in_flight <= 0:
            raise ValueError(f"max_in_flight must be positive, got {max_in_flight!r}")
        self._max = max_in_flight
        self._in_flight = 0
        self._lock = threading.Lock()

    @property
    def max_in_flight(self) -> int:
        return self._max

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def try_acquire(self) -> bool:
        with self._lock:
            if self._in_flight >= self._max:
                return False
            self._in_flight += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._in_flight <= 0:
                raise RuntimeError("InFlightLimit released more times than acquired")
            self._in_flight -= 1


@dataclass(frozen=True)
class BlockingCallPolicy:
    label: str
    hard_timeout_seconds: float
    # Shared by every copy made with with_label()/with_timeout(), so the bound
    # applies to the policy as a whole rather than per label.
    in_flight: InFlightLimit | None = field(default=None, compare=False, repr=False)

    def with_label(self, label: str) -> BlockingCallPolicy:
        return replace(self, label=label)

    def with_timeout(self, hard_timeout_seconds: float) -> BlockingCallPolicy:
        return replace(self, hard_timeout_seconds=hard_timeout_seconds)


def _validate(policy: BlockingCallPolicy) -> None:
    if policy.hard_timeout_seconds <= 0:
        raise ValueError(
            "hard_timeout_seconds must be positive, "
            f"got {policy.hard_timeout_seconds!r}"
        )


def _start_daemon_call(
    policy: BlockingCallPolicy, limit: InFlightLimit, fn: Callable[[], T]
) -> Future[T]:
    """Start *fn* on a daemon thread holding one slot of *limit* until it ends."""
    if not limit.try_acquire():
        logger.warning(
            "blocking_call_saturated",
            label=policy.label,
            max_in_flight=limit.max_in_flight,
        )
        raise BlockingCallSaturatedError(
            f"{policy.label!r} has {limit.max_in_flight} unfinished call(s) in flight"
        )

    future: Future[T] = Future()
    # Same context propagation as asyncio.to_thread: scoped runtime services
    # bound in a ContextVar stay visible inside the worker.
    context = contextvars.copy_context()

    def _worker() -> None:
        try:
            if not future.set_running_or_notify_cancel():
                return
            try:
                result = context.run(fn)
            except BaseException as exc:  # noqa: BLE001 - delivered to the caller
                future.set_exception(exc)
            else:
                future.set_result(result)
        finally:
            limit.release()

    thread = threading.Thread(
        target=_worker, name=f"blocking:{policy.label}"[:120], daemon=True
    )
    try:
        thread.start()
    except BaseException:
        limit.release()
        raise
    return future


async def run_blocking_call(policy: BlockingCallPolicy, fn: Callable[[], T]) -> T:
    _validate(policy)
    if policy.in_flight is None:
        return await run_with_hard_timeout(
            asyncio.to_thread(fn),
            timeout=policy.hard_timeout_seconds,
            label=policy.label,
        )
    future = _start_daemon_call(policy, policy.in_flight, fn)
    return await run_with_hard_timeout(
        asyncio.wrap_future(future),
        timeout=policy.hard_timeout_seconds,
        label=policy.label,
    )


def run_blocking_call_sync(policy: BlockingCallPolicy, fn: Callable[[], T]) -> T:
    """Synchronous twin of ``run_blocking_call`` for non-async call sites.

    Requires an ``InFlightLimit``: a sync caller has no event-loop executor to
    orphan into, and an unlimited per-call thread is exactly the unbounded
    accumulation this module exists to prevent. Raises ``TimeoutError`` when
    the deadline passes (``BlockingCallTimeoutError``; the worker is left
    running, holding its slot) and
    ``BlockingCallSaturatedError`` when no slot is free.
    """
    _validate(policy)
    if policy.in_flight is None:
        raise ValueError(
            f"run_blocking_call_sync requires an InFlightLimit on {policy.label!r}"
        )
    future = _start_daemon_call(policy, policy.in_flight, fn)
    try:
        return future.result(timeout=policy.hard_timeout_seconds)
    except TimeoutError:
        if future.done():
            # Either fn raised its own TimeoutError, or it finished at the
            # deadline; result() re-raises the former and returns the latter.
            return future.result()
        logger.warning(
            "hard_timeout_exceeded",
            label=policy.label,
            timeout_seconds=policy.hard_timeout_seconds,
            in_flight=policy.in_flight.in_flight,
        )
        raise BlockingCallTimeoutError(
            f"{policy.label!r} exceeded hard timeout of "
            f"{policy.hard_timeout_seconds:.1f}s"
        ) from None


# The yfinance-info, yahooquery, FX and gap-fill policies carry a label and (for
# gap fill) a shared in-flight cap; their numbers are only the shipped defaults.
# Each call site applies its Settings field with with_timeout(), clamped to a
# caller's enclosing deadline: COMPANY_NAME_LOOKUP_TIMEOUT_SECONDS and
# FX_RATE_FETCH_TIMEOUT_SECONDS. tests/data/test_market_data_timeouts.py pins
# these numbers to the Settings defaults.
YFINANCE_INFO_POLICY = BlockingCallPolicy("yfinance.info", 5.0)
YAHOOQUERY_QUOTE_TYPE_POLICY = BlockingCallPolicy("yahooquery.quote_type", 5.0)
OUTPUT_COMPANY_NAME_POLICY = BlockingCallPolicy("output_company_name", 6.0)
FX_RATE_POLICY = BlockingCallPolicy("fx_rate", 8.0)
# Gap-fill company-name lookup: same 5 s budget as YFINANCE_INFO_POLICY, but
# bounded so a stalled Yahoo endpoint cannot accumulate one orphan per ticker
# across a batch. On saturation the caller falls back to the bare symbol.
GAP_FILL_YFINANCE_INFO_POLICY = BlockingCallPolicy(
    "gap_fill.yfinance.info", 5.0, in_flight=InFlightLimit(4)
)
