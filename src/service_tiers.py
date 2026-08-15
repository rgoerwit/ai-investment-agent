"""
Service-tier (flex inference) helpers.

Flex tiers halve per-token cost on Gemini (``GEMINI_SERVICE_TIER=flex``) and
OpenAI (``OPENAI_SERVICE_TIER=flex``) in exchange for variable latency —
flex calls may queue 1-15 minutes and capacity is best-effort. This module
centralizes two concerns so every call site agrees:

1. **Eligibility** — which vendor/model combinations may request flex.
2. **Timeout floors** — any wall-clock ceiling that could kill a
   legitimately-queued flex call (SDK timeouts, ``run_with_hard_timeout``
   caps, consultant loop budgets) must be floored while flex is active,
   otherwise flex degrades into a timeout-retry generator that costs *more*
   than the standard tier.

Floors are provider-aware: enabling Gemini flex must not stretch OpenAI
ceilings and vice versa. Fallback-to-standard behavior lives in the LLM
subclasses in ``src/llms.py``, not here.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import structlog

import src.config as config_module

logger = structlog.get_logger(__name__)

_floor_log_lock = threading.Lock()
_logged_floor_labels: set[str] = set()

# Per-process negative capability cache. Neither Gemini nor OpenAI exposes
# service-tier support via model introspection (models.get / /v1/models say
# nothing about tiers), so eligibility is discovered dynamically: the first
# flex attempt on an unsupported model fails fast with a 400-class error,
# the model is recorded here, and every subsequent call in this process
# skips the flex request entirely. This replaces a hardcoded model-name
# allowlist — no per-model name checks needed, and newly flex-enabled
# models start working without a code change.
_flex_capability_lock = threading.Lock()
_flex_unsupported_models: set[str] = set()

# Per-process flex *health* cache — the sibling of the capability cache above,
# and keyed differently on purpose.
#
# Capability is a property of a MODEL: flex either exists for it or does not, and
# the answer never changes, so that cache is model-keyed and permanent.
# Congestion is a property of a VENDOR'S POOL: models queue together. Measured on
# 8002.T (2026-08-14), `gemini-3.6-flash` timed out three times and
# `gemini-3.1-pro-preview` once in a single run, ~121 minutes of waiting, because
# nothing remembered the first timeout — each flex attempt is bounded by the
# 900 s flex floor, and the run re-learned the same outage four times. Keying
# this by model would merely make it re-learn per model.
#
# The provider is a plain string; nothing here is specific to Google or OpenAI
# (today's only tiered vendors). A compatible vendor that gains a tier plugs in
# unchanged.
_flex_health_lock = threading.Lock()


@dataclass
class _FlexHealth:
    """Per-provider flex-pool health. Use only with ``_flex_health_lock`` held."""

    failures: deque[float] = field(default_factory=deque)
    degraded_until: float = 0.0
    # Set when a cool-off expires: the provider is eligible again, but a single
    # further failure re-degrades it. Without this, a sustained outage re-pays
    # ``threshold x flex floor`` on every cool-off cycle. Mirrors the half-open
    # probe in ``agents/circuit_breaker.LLMCircuitBreaker``.
    probation: bool = False
    episodes: int = 0


_flex_health: dict[str, _FlexHealth] = {}


def _cfg(cfg: Any = None) -> Any:
    return cfg if cfg is not None else config_module.config


def normalize_model_name(model_name: str) -> str:
    """Strip vendor path prefixes (e.g. "models/gemini-...") for cache keys."""
    return model_name.rsplit("/", 1)[-1]


def is_flex_unsupported(model_name: str) -> bool:
    """Whether this process has learned the model rejects the flex tier."""
    with _flex_capability_lock:
        return normalize_model_name(model_name) in _flex_unsupported_models


def mark_flex_unsupported(model_name: str) -> None:
    """Record that the vendor rejected a flex request for this model."""
    normalized = normalize_model_name(model_name)
    with _flex_capability_lock:
        if normalized in _flex_unsupported_models:
            return
        _flex_unsupported_models.add(normalized)
    logger.warning(
        "flex_unsupported_model_downgrade",
        model=normalized,
        note=(
            "vendor rejected service_tier=flex for this model; "
            "using the standard tier for the rest of this process"
        ),
    )


def is_flex_unsupported_error(exc: BaseException) -> bool:
    """Detect a vendor rejection of the flex tier itself (not capacity).

    Capability rejections are 400/INVALID_ARGUMENT-class errors whose message
    names the service tier; capacity exhaustion is 429/503 and is handled
    separately (retry/fall back per call, no capability caching).
    """
    combined = str(exc).lower()
    if "service_tier" not in combined and "service tier" not in combined:
        return False
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 400:
        return True
    return any(
        marker in combined
        for marker in (
            "invalid_argument",
            "invalid argument",
            "not supported",
            "unsupported",
            "invalid_request",
        )
    )


def _normalize_provider(provider: str | None) -> str:
    return (provider or "").strip().lower()


def flex_degraded(
    provider: str | None, *, now: float | None = None, cfg: Any = None
) -> bool:
    """Whether this process has learned the provider's flex pool is not serving.

    True only while a cool-off is active. Expiry is evaluated here rather than by
    a timer: the check happens on every call anyway, so the transition is free and
    there is no background state to leak.
    """
    key = _normalize_provider(provider)
    if not key or not bool(getattr(_cfg(cfg), "flex_degrade_enabled", True)):
        return False
    ts = time.monotonic() if now is None else now
    with _flex_health_lock:
        health = _flex_health.get(key)
        if health is None or not health.degraded_until:
            return False
        if ts < health.degraded_until:
            return True
        # Cool-off expired: eligible again, but on a hair trigger.
        health.degraded_until = 0.0
        health.probation = True
        health.failures.clear()
        return False


def note_flex_fallback(
    provider: str | None,
    *,
    reason: str,
    model: str = "",
    now: float | None = None,
    cfg: Any = None,
) -> None:
    """Record that a flex attempt fell back, and degrade the provider past threshold.

    ``reason`` is the flex-attributable failure class — ``"latency"`` (a queued
    call that never returned) or ``"capacity"`` (429/503). Both mean "the flex
    pool did not serve this call", which is the signal. Capability rejections are
    deliberately excluded: they are already cached permanently by
    ``mark_flex_unsupported`` and are not congestion.
    """
    key = _normalize_provider(provider)
    settings = _cfg(cfg)
    if not key or not bool(getattr(settings, "flex_degrade_enabled", True)):
        return
    threshold = int(getattr(settings, "flex_degrade_threshold", 2))
    window = float(getattr(settings, "flex_degrade_window_seconds", 900.0))
    cool_off = float(getattr(settings, "flex_degrade_cool_off_seconds", 1800.0))
    ts = time.monotonic() if now is None else now

    with _flex_health_lock:
        health = _flex_health.setdefault(key, _FlexHealth())
        if health.degraded_until and ts < health.degraded_until:
            return  # already degraded; nothing to learn
        cutoff = ts - window
        while health.failures and health.failures[0] < cutoff:
            health.failures.popleft()
        health.failures.append(ts)
        # One failure is enough while on probation — the cool-off just expired and
        # the pool immediately failed again.
        effective_threshold = 1 if health.probation else threshold
        if len(health.failures) < effective_threshold:
            return
        health.degraded_until = ts + cool_off
        health.probation = False
        health.episodes += 1
        failures = len(health.failures)

    logger.warning(
        "flex_provider_degraded",
        provider=key,
        reason=reason,
        model=normalize_model_name(model) if model else None,
        failures_in_window=failures,
        cool_off_seconds=round(cool_off, 1),
        note=(
            "flex fallbacks reached the threshold; requesting the standard tier "
            "for this provider until the cool-off expires"
        ),
    )


def flex_degradation_snapshot(*, now: float | None = None) -> dict[str, dict[str, Any]]:
    """Secret-free, JSON-serializable view of flex health, for run artifacts.

    Reports every provider that degraded at least once this process, so a slow or
    expensive run explains itself without anyone reading the logs.
    """
    ts = time.monotonic() if now is None else now
    snapshot: dict[str, dict[str, Any]] = {}
    with _flex_health_lock:
        for key, health in _flex_health.items():
            if not health.episodes:
                continue
            snapshot[key] = {
                "episodes": health.episodes,
                "degraded": bool(health.degraded_until and ts < health.degraded_until),
                "seconds_remaining": (
                    round(max(0.0, health.degraded_until - ts), 1)
                    if health.degraded_until
                    else 0.0
                ),
            }
    return snapshot


def _reset_flex_capability_cache_for_tests() -> None:
    with _flex_capability_lock:
        _flex_unsupported_models.clear()


def _reset_flex_health_for_tests() -> None:
    with _flex_health_lock:
        _flex_health.clear()


def resolve_google_service_tier(cfg: Any = None) -> str:
    """The effective Google tier, from whichever key the active schema uses.

    ``GOOGLE_SERVICE_TIER`` is the multi-provider key; ``GEMINI_SERVICE_TIER``
    is its legacy predecessor. They must never be read at different call sites:
    seat construction read the first while this module read the second, so an
    operator who set only ``GOOGLE_SERVICE_TIER=flex`` sent flex requests while
    the runtime believed flex was inactive — the timeout floors and the
    degradation cache silently never engaged. That is the same divergence
    ``RuntimeConfig.from_config`` already resolves for RPM, and it is resolved
    the same way here: one function, consulted by every reader.
    """
    settings = _cfg(cfg)
    if getattr(settings, "llm_base_provider", None) is not None:
        return getattr(settings, "google_service_tier", "standard")
    return getattr(settings, "gemini_service_tier", "standard")


def gemini_flex_active(cfg: Any = None) -> bool:
    return resolve_google_service_tier(cfg) == "flex"


def openai_flex_active(cfg: Any = None) -> bool:
    return getattr(_cfg(cfg), "openai_service_tier", "auto") == "flex"


def provider_flex_active(provider: str | None, cfg: Any = None) -> bool:
    """Whether flex is configured for the given provider ("google"/"openai")."""
    if provider == "google":
        return gemini_flex_active(cfg)
    if provider == "openai":
        return openai_flex_active(cfg)
    return False


def flex_floor_seconds(cfg: Any = None) -> float:
    return float(getattr(_cfg(cfg), "flex_llm_timeout_seconds", 900))


def _quick_mode_active(cfg: Any = None) -> bool:
    """Whether the current run is a ``--quick`` (latency-bounded) screening run.

    Quick mode imposes a tight per-ticker wall-clock budget (the pipeline
    watchdog) and a 120s in-process hard cap. Stretching a timeout *floor*
    past that budget converts a graceful in-process timeout into a hard
    external SIGTERM that discards the whole ticker, so the flex floors below
    must not apply in quick mode. Flex *eligibility* is unaffected — a fast
    flex call still gets the discount; a queued one is bounded by the quick
    deadline and falls back to standard (see ``src/llms.py``).

    ``quick_mode_active`` is run-scoped (the ``RuntimeConfig`` ContextVar set by
    ``--quick``), not a per-tier ``cfg`` field, so the ContextVar is the source
    of truth and the no-binding fallback snapshots the full global config —
    never the (possibly partial) tier ``cfg`` passed to the floor helpers.
    """
    from src.runtime_config import get_runtime_config

    return bool(get_runtime_config(config_module.config).quick_mode_active)


def _log_floor_once(label: str, base: float, floored: float, kind: str) -> None:
    with _floor_log_lock:
        if label in _logged_floor_labels:
            return
        _logged_floor_labels.add(label)
    logger.info(
        "flex_timeout_floor_applied",
        label=label,
        kind=kind,
        configured_seconds=round(base, 1),
        effective_seconds=round(floored, 1),
    )


def floor_llm_timeout(
    base_seconds: float,
    *,
    provider: str | None,
    cfg: Any = None,
    label: str = "llm_call",
) -> float:
    """Floor a per-call wall-clock ceiling while flex is active for provider.

    Returns ``base_seconds`` unchanged when flex is off for the provider,
    when quick mode is active (the floor must not outlast the quick deadline),
    or when the configured ceiling already exceeds the floor. Logs once per
    label so operators can see which ceilings were stretched.
    """
    if not provider_flex_active(provider, cfg) or _quick_mode_active(cfg):
        return base_seconds
    floor = flex_floor_seconds(cfg)
    if base_seconds >= floor:
        return base_seconds
    _log_floor_once(label, base_seconds, floor, kind="per_call")
    return floor


def floor_llm_hard_timeout(
    base_seconds: float,
    *,
    provider: str | None,
    cfg: Any = None,
    label: str = "llm_hard_cap",
) -> float:
    """Floor a hard wall-clock cap that wraps a whole ``ainvoke`` call.

    Uses 1.5x the per-call floor: a flex attempt that queues up to the SDK
    timeout may then fall back to a standard-tier attempt inside the same
    wrapper window, so the cap must outlast attempt + fallback. Skipped in
    quick mode, where the tight watchdog forbids stretching the deadline.
    """
    if not provider_flex_active(provider, cfg) or _quick_mode_active(cfg):
        return base_seconds
    floor = flex_floor_seconds(cfg) * 1.5
    if base_seconds >= floor:
        return base_seconds
    _log_floor_once(label, base_seconds, floor, kind="hard_cap")
    return floor


def floor_llm_total_timeout(
    base_seconds: float,
    *,
    provider: str | None,
    cfg: Any = None,
    label: str = "llm_total",
) -> float:
    """Floor a multi-call wall-clock budget (e.g. consultant tool loop).

    Uses 2x the per-call floor so a budget spanning several flex calls is not
    consumed entirely by the first queued call. Skipped in quick mode, where
    the tight watchdog forbids stretching the deadline.
    """
    if not provider_flex_active(provider, cfg) or _quick_mode_active(cfg):
        return base_seconds
    floor = flex_floor_seconds(cfg) * 2.0
    if base_seconds >= floor:
        return base_seconds
    _log_floor_once(label, base_seconds, floor, kind="total_budget")
    return floor


# Fraction of the quick-mode outer hard cap allotted to a single flex attempt's
# SDK client timeout, so a flex attempt PLUS its standard-tier re-issue both fit
# inside one run_with_hard_timeout window (2 x fraction < 1, with margin).
FLEX_QUICK_ATTEMPT_FRACTION = 0.4


def flex_attempt_client_timeout(
    base_seconds: float,
    *,
    provider: str | None,
    cfg: Any = None,
    label: str = "llm_call",
) -> float:
    """SDK client timeout for a flex-tier model instance.

    - **Full mode**: floor up to ``FLEX_LLM_TIMEOUT_SECONDS`` so a legitimately
      queued flex call (1-15 min) is not killed by a short socket timeout.
    - **Quick mode**: cap *below* the quick outer hard cap
      (``quick_llm_call_hard_timeout_seconds``) so a queued flex call raises a
      timeout in time for a standard-tier re-issue within the same
      ``run_with_hard_timeout`` window, instead of being guillotined by the
      outer cap (which would bypass the transport fallback). See the
      flex-fallback x timeout matrix in the mitigation plan.

    No-op (returns ``base_seconds``) when flex is off for the provider.
    """
    if not provider_flex_active(provider, cfg):
        return base_seconds
    if _quick_mode_active(cfg):
        outer = float(getattr(_cfg(cfg), "quick_llm_call_hard_timeout_seconds", 60.0))
        return min(base_seconds, max(5.0, outer * FLEX_QUICK_ATTEMPT_FRACTION))
    return floor_llm_timeout(base_seconds, provider=provider, cfg=cfg, label=label)


def _reset_floor_log_cache_for_tests() -> None:
    with _floor_log_lock:
        _logged_floor_labels.clear()
