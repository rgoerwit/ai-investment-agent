"""Process-global circuit breaker for host-level network failures.

Distinct from `LLMCircuitBreaker` (per-(agent, provider, model), counts
only `timeout`). This breaker watches `dns_resolution` and `connect_error`
across **all** contexts. Rationale: those failure kinds reflect a problem
with the host's resolver / network, not with any specific provider or
model — so during a real outage every parallel analyst would otherwise
burn its full retry budget independently against the same dead resolver.

The May 2026 overnight macOS DNS outage produced exactly this cascade:
~7 parallel analysts × 2 retries × ~30s of socket-level wait, blowing past
the 360s pipeline ticker watchdog while each call kept trying. A shared
breaker collapses that into a single short-circuit window of
`cool_off_seconds`, after which one probe re-establishes service.

Thresholds (defaults):
- 4 network failures within 30s opens the breaker.
- Open state lasts 45s.
- Half-open: the next call is the probe; success closes, failure re-opens.

Invariants:
- Only `failure_kind in {"dns_resolution", "connect_error"}` counts.
- A successful outcome (`record_outcome(ok=True)`) closes the breaker
  immediately and clears the failure window.
- `threading.RLock`-guarded to handle concurrent graph branches without
  needing asyncio integration.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)


NetworkBreakerState = Literal["closed", "open", "half_open"]


# Failure kinds eligible to trip this breaker. Kept narrow on purpose:
# `timeout` and `server_error` are routed to other recovery paths.
_ELIGIBLE_KINDS: frozenset[str] = frozenset({"dns_resolution", "connect_error"})


class NetworkBreakerOpenError(RuntimeError):
    """Raised when the network breaker is open. Carries the remaining
    cool-off so the caller can attribute the fast-failure correctly."""

    def __init__(self, *, opens_remaining_seconds: float) -> None:
        self.opens_remaining_seconds = opens_remaining_seconds
        super().__init__(
            f"Network breaker open; reopens in {opens_remaining_seconds:.1f}s"
        )


class NetworkBreaker:
    """Process-global, thread-safe network failure breaker.

    Unlike `LLMCircuitBreaker`, state is **global** (not keyed) because the
    underlying signal — host DNS / TCP — is shared across all callers.
    """

    def __init__(
        self,
        *,
        threshold: int = 4,
        window_seconds: float = 30.0,
        cool_off_seconds: float = 45.0,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if cool_off_seconds <= 0:
            raise ValueError("cool_off_seconds must be > 0")

        self.threshold = threshold
        self.window_seconds = window_seconds
        self.cool_off_seconds = cool_off_seconds

        self._failures: deque[float] = deque()
        self._state: NetworkBreakerState = "closed"
        self._open_until: float = 0.0
        self._probe_in_flight: bool = False
        self._lock = threading.RLock()

    # ---- helpers ------------------------------------------------------

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

    # ---- public API ---------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        """Debug helper; not synchronized with concurrent mutators by
        anything stronger than the lock that produced the dict."""
        with self._lock:
            return {
                "state": self._state,
                "failures_in_window": len(self._failures),
                "open_until_monotonic": self._open_until,
                "probe_in_flight": self._probe_in_flight,
            }

    def before_call(self, *, now: float | None = None) -> None:
        """Check the breaker before dispatching a call. Raises
        `NetworkBreakerOpenError` when the breaker is open. Half-open
        admits exactly one probe at a time; concurrent half-open callers
        are denied until the probe records an outcome."""
        ts = time.monotonic() if now is None else now
        with self._lock:
            if self._state == "open":
                if ts >= self._open_until:
                    self._state = "half_open"
                    self._probe_in_flight = False
                    logger.info("network_breaker_half_open")
                else:
                    raise NetworkBreakerOpenError(
                        opens_remaining_seconds=max(0.0, self._open_until - ts)
                    )
            if self._state == "half_open":
                if self._probe_in_flight:
                    raise NetworkBreakerOpenError(opens_remaining_seconds=0.0)
                self._probe_in_flight = True

    def record_outcome(
        self,
        *,
        ok: bool,
        failure_kind: str | None = None,
        now: float | None = None,
    ) -> None:
        """Record the outcome of a network-touching call.

        `ok=True` always closes the breaker and clears the failure window.
        On `ok=False`, only `failure_kind` in `_ELIGIBLE_KINDS` is counted
        toward the threshold."""
        ts = time.monotonic() if now is None else now
        with self._lock:
            previous_state = self._state
            self._probe_in_flight = False

            if ok:
                self._failures.clear()
                if previous_state != "closed":
                    self._state = "closed"
                    self._open_until = 0.0
                    logger.info("network_breaker_closed")
                return

            if failure_kind not in _ELIGIBLE_KINDS:
                return

            self._failures.append(ts)
            self._prune(ts)

            if previous_state == "half_open":
                self._state = "open"
                self._open_until = ts + self.cool_off_seconds
                logger.warning(
                    "network_breaker_reopened",
                    failure_kind=failure_kind,
                    cool_off_seconds=self.cool_off_seconds,
                )
                return

            if len(self._failures) >= self.threshold:
                self._state = "open"
                self._open_until = ts + self.cool_off_seconds
                logger.warning(
                    "network_breaker_opened",
                    failure_kind=failure_kind,
                    failures_in_window=len(self._failures),
                    window_seconds=self.window_seconds,
                    cool_off_seconds=self.cool_off_seconds,
                )

    def reset(self) -> None:
        """Forget state. Used by tests + admin entrypoints."""
        with self._lock:
            self._failures.clear()
            self._state = "closed"
            self._open_until = 0.0
            self._probe_in_flight = False


# ---- module-level singleton ----------------------------------------------

_global_breaker: NetworkBreaker | None = None
_global_breaker_lock = threading.Lock()


def get_network_breaker() -> NetworkBreaker:
    """Lazily construct the process-global breaker using current config."""
    global _global_breaker
    with _global_breaker_lock:
        if _global_breaker is None:
            try:
                from src.config import config as settings_config

                _global_breaker = NetworkBreaker(
                    threshold=int(
                        getattr(settings_config, "network_breaker_threshold", 4)
                    ),
                    window_seconds=float(
                        getattr(settings_config, "network_breaker_window_seconds", 30.0)
                    ),
                    cool_off_seconds=float(
                        getattr(
                            settings_config, "network_breaker_cool_off_seconds", 45.0
                        )
                    ),
                )
            except Exception:
                _global_breaker = NetworkBreaker()
        return _global_breaker


def reset_network_breaker_singleton() -> None:
    """Drop the cached singleton (for tests that change config)."""
    global _global_breaker
    with _global_breaker_lock:
        _global_breaker = None
