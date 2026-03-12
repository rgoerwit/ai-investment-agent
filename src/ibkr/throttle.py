"""
Centralized request pacing for the IBKR API.

IBKRThrottle owns:
- Inter-request rate limiting (monotonic clock, no backward jumps)
- Named warm-up pauses for endpoints with engine-init requirements
- Automatic 429 back-off and retry
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class IBKRThrottle:
    """
    Rate limiter and retry wrapper for ibind API calls.

    Usage::

        throttle = IBKRThrottle(rate_per_sec=10.0)

        # Standard call
        result = throttle.call(lambda: ibkr.portfolio_accounts())

        # Endpoint with engine warm-up (e.g. orders engine)
        orders = throttle.call_with_warmup(
            preflight=lambda: ibkr.live_orders(force=True),
            request=lambda: ibkr.live_orders(),
            warm_up_secs=1.0,
            label="live_orders",
        )
    """

    def __init__(self, rate_per_sec: float = 10.0, max_retries: int = 3) -> None:
        self._min_interval = 1.0 / max(rate_per_sec, 1e-9)
        self._max_retries = max_retries
        self._last_call_time: float = 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    def call(self, fn: Callable[[], T]) -> T:
        """Rate-limited call with automatic 429 backoff retry."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            self._pace()
            try:
                return fn()
            except Exception as exc:
                if self._is_rate_limited(exc):
                    backoff = 2**attempt
                    logger.warning(
                        "ibkr_429_backoff",
                        attempt=attempt + 1,
                        max_retries=self._max_retries,
                        backoff_secs=backoff,
                    )
                    time.sleep(backoff)
                    last_exc = exc
                else:
                    raise
        # Exhausted retries — re-raise the last 429 exception
        raise last_exc  # type: ignore[misc]

    def call_with_warmup(
        self,
        preflight: Callable[[], Any],
        request: Callable[[], T],
        warm_up_secs: float = 1.0,
        *,
        label: str = "endpoint",
    ) -> T:
        """Pre-flight call → named warm-up pause → real call.

        Encodes IBKR's pattern for endpoints whose backend engine must
        initialise before it returns data (e.g. the orders engine).
        The preflight result is discarded; only the request result is returned.
        """
        self.call(preflight)
        logger.debug("ibkr_warmup", label=label, duration_secs=warm_up_secs)
        time.sleep(warm_up_secs)
        return self.call(request)

    # ── Internals ───────────────────────────────────────────────────────────

    def _pace(self) -> None:
        """Block until the minimum inter-request interval has elapsed."""
        now = time.monotonic()
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.monotonic()

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        """Return True if the ibind exception signals HTTP 429."""
        text = str(exc).lower()
        return "429" in text or "too many requests" in text
