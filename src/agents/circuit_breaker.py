"""P2-7: Per-process circuit breaker for chronically-slow LLM calls.

Motivation: when a provider/model starts serving back-to-back hard-timeouts
(e.g., Gemini Flash regional degradation), every quick run pays
``hard_timeout × api_retry_attempts`` of dead wall-clock per affected agent
until conditions recover. The breaker watches a sliding window of timeouts
per ``(agent_name, provider, model)`` key and opens after ``threshold``
hits within ``window_seconds``. While open, subsequent calls raise
``CircuitOpenError`` immediately and the runtime records a stub failure
attempt instead of waiting another full timeout.

After ``cool_off_seconds`` the breaker enters ``half_open`` and lets one
probe call through. A successful probe closes the circuit; a failed probe
re-opens it for another cool-off window.

This is deliberately *in-process only* — a single pipeline run is the
correct scope, since the real failure mode is "this run is hitting a wall
right now." Persisted state would conflict with the recovery story.

Key invariants:
- Only ``failure_kind == "timeout"`` counts toward the threshold. A normal
  application error (e.g., 400 from a bad prompt) is not breaker-eligible.
- ``record_outcome(ok=True, ...)`` from any state closes the circuit.
- ``before_call`` and ``record_outcome`` use a ``threading.RLock`` so calls
  from concurrent graph branches see consistent state without needing
  ``asyncio`` integration.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)


CircuitState = Literal["closed", "open", "half_open"]


class CircuitOpenError(RuntimeError):
    """Raised by the LLM runtime when the breaker is open for a key.

    Carries the canonical metadata so callers and the saved diagnostics
    can attribute the fast-failure correctly.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        provider: str,
        model_name: str,
        opens_remaining_seconds: float,
    ) -> None:
        self.agent_name = agent_name
        self.provider = provider
        self.model_name = model_name
        self.opens_remaining_seconds = opens_remaining_seconds
        super().__init__(
            f"LLM circuit open for {agent_name!r} "
            f"(provider={provider}, model={model_name}); "
            f"reopens in {opens_remaining_seconds:.1f}s"
        )


@dataclass
class _KeyState:
    """Internal per-key state. Use only with the parent breaker's lock held."""

    failures: deque[float] = field(default_factory=deque)
    state: CircuitState = "closed"
    open_until: float = 0.0
    probe_in_flight: bool = False


class LLMCircuitBreaker:
    """In-memory, thread-safe circuit breaker keyed on
    ``(agent_name, provider, model_name)``.
    """

    def __init__(
        self,
        *,
        threshold: int = 3,
        window_seconds: float = 300.0,
        cool_off_seconds: float = 60.0,
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
        self._states: dict[tuple[str, str, str], _KeyState] = {}
        self._lock = threading.RLock()

    # ---- helpers ------------------------------------------------------

    @staticmethod
    def _normalize_key(
        agent_name: str, provider: str, model_name: str
    ) -> tuple[str, str, str]:
        return (
            (agent_name or "unknown").strip().lower(),
            (provider or "unknown").strip().lower(),
            (model_name or "unknown").strip().lower(),
        )

    def _state_for(self, key: tuple[str, str, str]) -> _KeyState:
        state = self._states.get(key)
        if state is None:
            state = _KeyState()
            self._states[key] = state
        return state

    def _prune_old_failures(self, ks: _KeyState, *, now: float) -> None:
        cutoff = now - self.window_seconds
        while ks.failures and ks.failures[0] < cutoff:
            ks.failures.popleft()

    # ---- public API ---------------------------------------------------

    def snapshot(
        self, agent_name: str, provider: str, model_name: str
    ) -> dict[str, object]:
        """Debug helper — returns the visible state for a key."""
        key = self._normalize_key(agent_name, provider, model_name)
        with self._lock:
            ks = self._states.get(key)
            if ks is None:
                return {"state": "closed", "failures_in_window": 0}
            return {
                "state": ks.state,
                "failures_in_window": len(ks.failures),
                "open_until_monotonic": ks.open_until,
                "probe_in_flight": ks.probe_in_flight,
            }

    def before_call(
        self,
        *,
        agent_name: str,
        provider: str,
        model_name: str,
        now: float | None = None,
    ) -> None:
        """Check the circuit before an LLM call. Raises ``CircuitOpenError``
        if the call must not be dispatched. When the breaker is half-open,
        marks the call as the probe so a concurrent half-open caller will
        still be denied.
        """
        ts = time.monotonic() if now is None else now
        key = self._normalize_key(agent_name, provider, model_name)
        with self._lock:
            ks = self._state_for(key)
            if ks.state == "open":
                if ts >= ks.open_until:
                    ks.state = "half_open"
                    ks.probe_in_flight = False
                    logger.info(
                        "llm_circuit_half_open",
                        agent=agent_name,
                        provider=provider,
                        model=model_name,
                    )
                else:
                    raise CircuitOpenError(
                        agent_name=agent_name,
                        provider=provider,
                        model_name=model_name,
                        opens_remaining_seconds=max(0.0, ks.open_until - ts),
                    )
            if ks.state == "half_open":
                if ks.probe_in_flight:
                    raise CircuitOpenError(
                        agent_name=agent_name,
                        provider=provider,
                        model_name=model_name,
                        opens_remaining_seconds=0.0,
                    )
                ks.probe_in_flight = True

    def record_outcome(
        self,
        *,
        agent_name: str,
        provider: str,
        model_name: str,
        ok: bool,
        failure_kind: str | None = None,
        now: float | None = None,
    ) -> None:
        """Record the outcome of a call.

        Only ``failure_kind == "timeout"`` counts toward opening the circuit.
        Any successful outcome closes the circuit and clears the failure
        history.
        """
        ts = time.monotonic() if now is None else now
        key = self._normalize_key(agent_name, provider, model_name)
        with self._lock:
            ks = self._state_for(key)
            previous_state = ks.state
            ks.probe_in_flight = False

            if ok:
                ks.failures.clear()
                if previous_state != "closed":
                    ks.state = "closed"
                    ks.open_until = 0.0
                    logger.info(
                        "llm_circuit_closed",
                        agent=agent_name,
                        provider=provider,
                        model=model_name,
                    )
                return

            if failure_kind != "timeout":
                # Don't penalize non-timeout failures; they're typically
                # request-shape errors (400s) that retries won't fix and
                # are unrelated to the regional-slow-tail scenario we
                # want to short-circuit.
                return

            ks.failures.append(ts)
            self._prune_old_failures(ks, now=ts)

            if previous_state == "half_open":
                ks.state = "open"
                ks.open_until = ts + self.cool_off_seconds
                logger.warning(
                    "llm_circuit_reopened",
                    agent=agent_name,
                    provider=provider,
                    model=model_name,
                    cool_off_seconds=self.cool_off_seconds,
                )
                return

            if len(ks.failures) >= self.threshold:
                ks.state = "open"
                ks.open_until = ts + self.cool_off_seconds
                logger.warning(
                    "llm_circuit_opened",
                    agent=agent_name,
                    provider=provider,
                    model=model_name,
                    failures_in_window=len(ks.failures),
                    window_seconds=self.window_seconds,
                    cool_off_seconds=self.cool_off_seconds,
                )

    def reset(self) -> None:
        """Forget all per-key state (for tests + admin)."""
        with self._lock:
            self._states.clear()


# ---- module-level singleton ----------------------------------------------

_global_breaker: LLMCircuitBreaker | None = None
_global_breaker_lock = threading.Lock()


def get_circuit_breaker() -> LLMCircuitBreaker:
    """Return the process-wide singleton breaker, lazily configured from
    the live ``Settings`` values.

    Re-reading settings on each construction keeps tests (which monkeypatch
    ``config``) honest, while still amortizing the breaker across calls.
    """
    global _global_breaker
    with _global_breaker_lock:
        if _global_breaker is None:
            from src.config import config

            _global_breaker = LLMCircuitBreaker(
                threshold=int(getattr(config, "llm_circuit_breaker_threshold", 3)),
                window_seconds=float(
                    getattr(config, "llm_circuit_breaker_window_seconds", 300.0)
                ),
                cool_off_seconds=float(
                    getattr(config, "llm_circuit_breaker_cool_off_seconds", 60.0)
                ),
            )
        return _global_breaker


def reset_circuit_breaker_for_tests() -> None:
    """Test helper — clears the global singleton so the next get() rebuilds
    from current config values."""
    global _global_breaker
    with _global_breaker_lock:
        _global_breaker = None
