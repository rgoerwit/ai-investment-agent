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


def _reset_flex_capability_cache_for_tests() -> None:
    with _flex_capability_lock:
        _flex_unsupported_models.clear()


def gemini_flex_active(cfg: Any = None) -> bool:
    return getattr(_cfg(cfg), "gemini_service_tier", "standard") == "flex"


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

    Returns ``base_seconds`` unchanged when flex is off for the provider or
    the configured ceiling already exceeds the floor. Logs once per label so
    operators can see which ceilings were stretched.
    """
    if not provider_flex_active(provider, cfg):
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
    wrapper window, so the cap must outlast attempt + fallback.
    """
    if not provider_flex_active(provider, cfg):
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
    consumed entirely by the first queued call.
    """
    if not provider_flex_active(provider, cfg):
        return base_seconds
    floor = flex_floor_seconds(cfg) * 2.0
    if base_seconds >= floor:
        return base_seconds
    _log_floor_once(label, base_seconds, floor, kind="total_budget")
    return floor


def _reset_floor_log_cache_for_tests() -> None:
    with _floor_log_lock:
        _logged_floor_labels.clear()
