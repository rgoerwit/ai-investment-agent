"""
LLM configuration and initialization module.
Updated for Google Gemini 3 with Safety Settings and Rate Limiting.
Includes token tracking for cost monitoring.
UPDATED: Configurable rate limits via GEMINI_RPM_LIMIT environment variable.
UPDATED: Added OpenAI consultant LLM for cross-validation (Dec 2025).
"""

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec
from numbers import Real
from typing import Any, Literal
from urllib.parse import urlsplit

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

import src.config as config_module
from src.config import config
from src.error_safety import summarize_exception
from src.llm_budgets import (
    GenerationBudget,
    get_agent_output_budget,
    get_generation_budget,
)
from src.runtime_config import get_runtime_config
from src.runtime_services import get_current_provider_runtime
from src.service_tiers import (
    flex_attempt_client_timeout,
    flex_degraded,
    gemini_flex_active,
    is_flex_unsupported,
    is_flex_unsupported_error,
    mark_flex_unsupported,
    normalize_model_name,
    note_flex_fallback,
    openai_flex_active,
)

logger = structlog.get_logger(__name__)


def _settings_or_default(settings: Any | None) -> Any:
    return config if settings is None else settings


_logged_model_init_configs: set[tuple[str, str, int, int, str | None]] = set()


def _langchain_openai_available() -> bool:
    """Availability guard tolerant of test-injected module mocks.

    ``find_spec`` raises ValueError for modules present in ``sys.modules``
    with ``__spec__ = None`` (how tests stub ``langchain_openai``), so check
    ``sys.modules`` first.
    """
    if "langchain_openai" in sys.modules:
        return True
    try:
        return find_spec("langchain_openai") is not None
    except ValueError:
        return True


_THINKING_BUDGETS = {"low": 512, "medium": 4096, "high": 16384}

# Ordered low → high so a "bump" is just the next element, clamped at the
# ceiling. Keeps the one-notch-up mechanism generic (not tied to any specific
# agent or baseline level): if a quick-tier agent's baseline ever changes, the
# bump recomputes automatically.
_THINKING_LEVEL_ORDER: tuple[Literal["low", "medium", "high"], ...] = (
    "low",
    "medium",
    "high",
)


def bump_thinking_level(
    level: Literal["low", "medium", "high"] | None,
) -> Literal["low", "medium", "high"] | None:
    """Return the next thinking level above ``level``, or ``level`` at the ceiling.

    ``None`` (model does not support ``thinking_level``) passes through
    unchanged, so a bump requested on a non-thinking model is a safe no-op.
    """
    if level is None:
        return None
    try:
        idx = _THINKING_LEVEL_ORDER.index(level)
    except ValueError:
        return level
    if idx + 1 < len(_THINKING_LEVEL_ORDER):
        return _THINKING_LEVEL_ORDER[idx + 1]
    return level


# Reasoning-effort capabilities per OpenAI-compatible model family.
#
# Every seat on the OpenAI plane (consultant, auditor, editor, writer
# fallback, APAC specialist) may point at an OpenAI-*compatible* endpoint via
# OPENAI_API_BASE / APAC_SPECIALIST_BASE_URL, so this table is keyed by model
# family rather than by vendor.  Registering a family here is what bounds its
# hidden reasoning: on these models the reasoning tokens are drawn from the
# same completion-token pool as the visible answer, so an *unset* effort lets
# the model spend the whole budget thinking.  Measured against the live
# Moonshot endpoint on 2026-08-02 with a consultant-scale prompt under the
# production 4096-token cap: no effort -> 2553 reasoning tokens, high -> 1434,
# low -> 26.  The unbounded case is what produced the 1088.HK consultant
# "review" of 46 characters.
#
# Longest matching prefix wins, so entries are order-independent (same rule as
# ``token_tracker._lookup_model_pricing``).  Values are the vendor-documented
# settings only — an undocumented value that today happens to be accepted is
# not a contract.
_OPENAI_REASONING_EFFORTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("gpt-5.6", frozenset({"none", "low", "medium", "high", "xhigh", "max"})),
    ("gpt-5.5", frozenset({"none", "low", "medium", "high", "xhigh"})),
    ("gpt-5.4", frozenset({"none", "low", "medium", "high", "xhigh"})),
    ("gpt-5.2", frozenset({"none", "low", "medium", "high", "xhigh"})),
    ("gpt-5.1", frozenset({"none", "low", "medium", "high"})),
    ("gpt-5", frozenset({"minimal", "low", "medium", "high"})),
    # Moonshot Kimi K3 — documented low|high|max, defaulting to ``max``.
    ("kimi-k3", frozenset({"low", "high", "max"})),
)

# Substring markers for variants that reject the reasoning parameter outright
# (OpenAI's ``-pro`` tier).  Checked against the whole normalized model id.
_OPENAI_NO_REASONING_MARKERS: tuple[str, ...] = ("pro",)

# Efforts deep enough that the small "default" reserve cannot cover the hidden
# reasoning; these earn the "deep" reserve instead.
_DEEP_REASONING_EFFORTS = frozenset({"high", "xhigh", "max"})

# Ordered preferences, resolved against a family's supported set.  ``low`` is
# deliberately preferred over ``minimal`` for quick mode even where a legacy
# model accepts both: current GPT-5.6 models do not document ``minimal``.
_EFFORT_PREFERENCE_QUICK: tuple[str, ...] = ("low", "minimal")
_EFFORT_PREFERENCE_FULL: tuple[str, ...] = ("medium", "high")
# Long-form prose wants output budget, not reasoning depth.
_EFFORT_PREFERENCE_PROSE: tuple[str, ...] = ("low", "minimal")
# The APAC regional specialist is a deliberately deep single-shot seat.
_EFFORT_PREFERENCE_DEEPEST: tuple[str, ...] = ("max", "xhigh", "high")

_warned_unknown_openai_reasoning: set[str] = set()


def _normalized_openai_model_id(model_name: str) -> str:
    """Lowercase a model id and drop a single leading ``vendor/`` segment."""
    normalized = model_name.strip().lower()
    if normalized.count("/") == 1:
        normalized = normalized.split("/", 1)[1]
    return normalized


def _openai_supported_reasoning_efforts(model_name: str) -> frozenset[str] | None:
    """Return a model family's documented efforts, or None when unregistered."""
    normalized_name = _normalized_openai_model_id(model_name)
    if any(marker in normalized_name for marker in _OPENAI_NO_REASONING_MARKERS):
        return None

    best_prefix_length = -1
    supported_efforts: frozenset[str] | None = None
    for prefix, efforts in _OPENAI_REASONING_EFFORTS:
        if normalized_name.startswith(prefix) and len(prefix) > best_prefix_length:
            best_prefix_length = len(prefix)
            supported_efforts = efforts
    return supported_efforts


def _openai_reasoning_effort(
    model_name: str, *, preference: tuple[str, ...]
) -> str | None:
    """Resolve the first preferred effort the model family documents.

    ``None`` means "no reasoning parameter for this model" — either the family
    is unregistered or it rejects the parameter.  Callers must then leave the
    parameter off entirely rather than guessing a value.
    """
    supported_efforts = _openai_supported_reasoning_efforts(model_name)
    if supported_efforts is None:
        return None
    for effort in preference:
        if effort in supported_efforts:
            return effort
    return None


def _effort_preference_for_mode(quick_mode: bool) -> tuple[str, ...]:
    return _EFFORT_PREFERENCE_QUICK if quick_mode else _EFFORT_PREFERENCE_FULL


def _reserve_class_for_effort(effort: str | None) -> Literal["default", "deep"]:
    """Size the completion-cap reserve to the reasoning depth requested."""
    return "deep" if effort in _DEEP_REASONING_EFFORTS else "default"


def _centralized_output_budget(agent_name: str) -> int:
    """Resolve an agent's share of ``LLM_BASE_OUTPUT_TOKENS``.

    The graph resolves these budgets itself; this is for the seats built
    outside the graph (editor) and for direct factory calls, so no caller has
    to re-literalize a token count that ``AGENT_OUTPUT_BUDGET_FRACTIONS`` owns.
    """
    return get_agent_output_budget(
        agent_name,
        _coerce_int_setting(getattr(config, "llm_base_output_tokens", None), 32768),
    )


# Relax safety settings slightly for financial/market analysis context
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

_GEMINI_GENERATION_FIELD_KEYS = {
    "candidate_count": frozenset({"candidate_count", "candidateCount"}),
    "temperature": frozenset({"temperature"}),
    "top_p": frozenset({"top_p", "topP"}),
    "top_k": frozenset({"top_k", "topK"}),
}


def _normalized_gemini_model_id(model_name: str) -> str:
    """Return a comparable Gemini model ID without path or numeric-version suffix."""
    normalized = normalize_model_name(model_name).lower()
    return re.sub(r"-\d{3}$", "", normalized)


def _gemini_version(model_name: str) -> tuple[int, int] | None:
    """Parse the major/minor Gemini version without inferring model capabilities."""
    match = re.match(
        r"gemini-(\d+)(?:\.(\d+))?",
        _normalized_gemini_model_id(model_name),
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _gemini_generation_fields_to_omit(model_name: str) -> frozenset[str]:
    """Return generation fields Google says to omit for a Gemini model.

    Gemini 3.x does not support ``candidate_count``. Google also recommends
    removing ``temperature``, ``top_p``, and ``top_k`` from every Gemini 3.x
    request because those models are optimized for their sampling defaults.
    The sampling fields are deprecated and ignored beginning with Gemini 3.6
    Flash and Gemini 3.5 Flash-Lite, and future model generations may reject
    them.
    """
    version = _gemini_version(model_name)
    if version is None or version[0] < 3:
        return frozenset()
    return frozenset(_GEMINI_GENERATION_FIELD_KEYS)


def _without_gemini_generation_fields(
    values: dict[str, Any],
    omitted_fields: frozenset[str],
) -> dict[str, Any]:
    """Copy a request mapping without snake_case or API-alias forms of fields."""
    omitted_keys = {
        key for field in omitted_fields for key in _GEMINI_GENERATION_FIELD_KEYS[field]
    }
    return {key: value for key, value in values.items() if key not in omitted_keys}


def _is_gemini_v3_or_greater(model_name: str) -> bool:
    """
    Checks if a Gemini model supports 'thinking_level' parameter.

    Includes:
    - Gemini 3.0+ models (e.g., gemini-3-pro-preview)
    - Any model with 'thinking' in the name (e.g., gemini-2.0-flash-thinking-exp)
    """
    if not model_name.startswith("gemini-"):
        return False

    # Explicit support for "thinking" models regardless of version number
    if "thinking" in model_name.lower():
        return True

    match = re.search(r"gemini-([0-9.]+)", model_name)
    if not match:
        return False

    version_str = match.group(1)
    try:
        major_version = int(version_str.split(".")[0])
        return major_version >= 3
    except (ValueError, IndexError):
        return False


def _is_gemini_v2_5(model_name: str) -> bool:
    """Checks if a Gemini model is in the 2.5 family."""
    if not model_name.startswith("gemini-"):
        return False

    match = re.search(r"gemini-([0-9]+)\.([0-9]+)", model_name)
    if not match:
        return False

    try:
        major_version = int(match.group(1))
        minor_version = int(match.group(2))
        return major_version == 2 and minor_version >= 5
    except ValueError:
        return False


def is_gemini_v3_or_greater(model_name: str) -> bool:
    """
    Public wrapper to check if a Gemini model is version 3.0 or greater.

    Used by agents to determine if retry with high thinking_level is beneficial.
    Only Gemini 3+ models support the thinking_level parameter.

    Args:
        model_name: The model name string (e.g., "gemini-3-pro-preview")

    Returns:
        True if model is Gemini 3.0 or greater, False otherwise
    """
    return _is_gemini_v3_or_greater(model_name)


def _create_rate_limiter_from_rpm(rpm: int) -> InMemoryRateLimiter:
    """Compatibility wrapper around the provider-neutral limiter factory."""

    from src.llm_runtime.rate_limits import create_process_rate_limiter

    return create_process_rate_limiter(rpm)


def create_process_rate_limiter(rpm: int | None = None) -> InMemoryRateLimiter:
    """Create an owned rate limiter for a long-lived process/runtime.

    Bare calls intentionally use base config. CLI-scoped runs pass an explicit
    rpm from ``RuntimeConfig`` through ``ProviderRuntime``.
    """
    effective_rpm = rpm if rpm is not None else config_module.config.gemini_rpm_limit
    return _create_rate_limiter_from_rpm(effective_rpm)


def _reset_init_log_cache_for_tests() -> None:
    """Reset one-time init logging state for tests."""
    _logged_model_init_configs.clear()


def _log_model_init_once(
    kind: str,
    model_name: str,
    timeout: int,
    retries: int,
    thinking_level: str | None,
) -> None:
    key = (kind, model_name, timeout, retries, thinking_level)
    if key in _logged_model_init_configs:
        return
    _logged_model_init_configs.add(key)
    logger.debug(
        "llm_initialized",
        kind=kind,
        model=model_name,
        timeout=timeout,
        retries=retries,
        thinking_level=thinking_level,
    )


class _LazyRateLimiterProxy(BaseRateLimiter):
    """Lazily construct the shared Gemini rate limiter on first use."""

    def __init__(self, factory: Callable[[], BaseRateLimiter]):
        self._factory = factory
        self._instance: BaseRateLimiter | None = None

    def _get_instance(self) -> BaseRateLimiter:
        provider_runtime = get_current_provider_runtime()
        if (
            provider_runtime is not None
            and provider_runtime.rate_limiter is not None
            and provider_runtime.rate_limiter is not self
        ):
            runtime_limiter = provider_runtime.limiter_for("google")
            if runtime_limiter is not None:
                return runtime_limiter
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def acquire(self, *, blocking: bool = True) -> bool:
        return self._get_instance().acquire(blocking=blocking)

    async def aacquire(self, *, blocking: bool = True) -> bool:
        return await self._get_instance().aacquire(blocking=blocking)

    async def __aenter__(self):
        await self.aacquire(blocking=True)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)

    def __repr__(self) -> str:
        status = "initialized" if self._instance is not None else "lazy"
        return f"<_LazyRateLimiterProxy {status}>"


GLOBAL_RATE_LIMITER = _LazyRateLimiterProxy(
    # Legacy fallback for unscoped callers; CLI-scoped runs bind a
    # ProviderRuntime with an explicit rate limiter built from RuntimeConfig.
    lambda: _create_rate_limiter_from_rpm(config_module.config.gemini_rpm_limit)
)

# Lazily-constructed OpenAI rate limiter.  None when OPENAI_RPM_LIMIT is unset
# (the default) so existing deployments are not throttled without opt-in.
_openai_rate_limiter: InMemoryRateLimiter | None = None
_openai_rate_limiter_initialized: bool = False
_warned_openai_unthrottled: set[str] = set()


def _get_openai_rate_limiter(*, settings: Any | None = None) -> BaseRateLimiter | None:
    """Return the shared OpenAI rate limiter, initializing it on first call."""
    settings = _settings_or_default(settings)
    if settings is not config:
        from src.llm_runtime.identities import sanitize_endpoint_host
        from src.llm_runtime.rate_limits import limiter_for_binding

        return limiter_for_binding(
            settings,
            "openai",
            sanitize_endpoint_host(settings.get_openai_api_base()),
        )
    global _openai_rate_limiter, _openai_rate_limiter_initialized
    if not _openai_rate_limiter_initialized:
        _openai_rate_limiter_initialized = True
        rpm = config_module.config.openai_rpm_limit
        if rpm is not None:
            _openai_rate_limiter = _create_rate_limiter_from_rpm(rpm)
    return _openai_rate_limiter


def _get_openai_rate_limiter_for_settings(
    settings: Any,
) -> BaseRateLimiter | None:
    if settings is config:
        return _get_openai_rate_limiter()
    return _get_openai_rate_limiter(settings=settings)


def _warn_openai_unthrottled_once(llm_kind: str) -> None:
    """Log once per LLM kind when no OpenAI rate limiter is configured."""
    if llm_kind not in _warned_openai_unthrottled:
        _warned_openai_unthrottled.add(llm_kind)
        logger.debug("openai_llm_unthrottled", llm_kind=llm_kind)


def _reset_openai_rate_limiter_for_tests() -> None:
    """Reset OpenAI rate-limiter state between tests."""
    global _openai_rate_limiter, _openai_rate_limiter_initialized
    _openai_rate_limiter = None
    _openai_rate_limiter_initialized = False
    _warned_openai_unthrottled.clear()
    _warned_unknown_openai_reasoning.clear()


# Track LLM instances for cleanup
_llm_instances: dict = {}
_llm_instance_counter: int = 0


def _coerce_int_setting(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, Real):
        return int(float(value))
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _reasoning_counts_against_completion_cap(
    *,
    provider: str,
    model_name: str,
    thinking_level: str | None = None,
    reasoning_effort: str | None = None,
    thinking: Any = None,
    thinking_budget: int | None = None,
) -> bool:
    if provider == "google":
        return bool(thinking_level and _is_gemini_v3_or_greater(model_name))
    if provider == "openai":
        return reasoning_effort is not None
    if provider == "anthropic":
        return thinking is not None and thinking_budget is None
    return False


def _resolve_generation_budget(
    *,
    intent_tokens: int,
    reserve_class: Literal["default", "deep"],
    reserve_enabled: bool,
    settings: Any | None = None,
) -> GenerationBudget:
    settings = _settings_or_default(settings)
    return get_generation_budget(
        intent_tokens=intent_tokens,
        reserve_class=reserve_class,
        reserve_enabled=reserve_enabled,
        default_reserve_tokens=_coerce_int_setting(
            getattr(settings, "llm_default_reasoning_reserve_tokens", None),
            2048,
        ),
        deep_reserve_tokens=_coerce_int_setting(
            getattr(settings, "llm_deep_reasoning_reserve_tokens", None),
            8192,
        ),
    )


def _stamp_budget_metadata(
    llm: BaseChatModel,
    *,
    callbacks: list[BaseCallbackHandler] | None,
    budget: GenerationBudget,
    intent_attr: str,
    api_attr: str,
) -> None:
    setattr(llm, intent_attr, budget.intent_tokens)
    setattr(llm, api_attr, budget.api_cap_tokens)
    llm._configured_reasoning_reserve_tokens = budget.reserve_tokens  # type: ignore[attr-defined]

    for callback in callbacks or []:
        if hasattr(callback, "output_token_cap"):
            callback.output_token_cap = budget.intent_tokens  # type: ignore[attr-defined]
            callback.api_output_token_cap = budget.api_cap_tokens  # type: ignore[attr-defined]
            callback.reasoning_reserve_tokens = budget.reserve_tokens  # type: ignore[attr-defined]


def _is_flex_capacity_error(exc: BaseException) -> bool:
    """Detect flex-tier capacity exhaustion (429/503-class errors).

    Marker style mirrors ``runtime_diagnostics.classify_failure``. Genuine
    rate-limit 429s also match; falling back to the standard tier is a valid
    (if full-price) recovery for those too, and the process rate limiter
    keeps them rare.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, 503):
        return True
    combined = str(exc).lower()
    return any(
        marker in combined
        for marker in (
            "429",
            "503",
            "rate limit",
            "too many requests",
            "resource_exhausted",
            "resource exhausted",
            "resource_unavailable",
            "unavailable",
        )
    )


def _is_flex_latency_timeout(exc: BaseException) -> bool:
    """Detect a flex-attempt SDK client timeout (a queued call that never returned).

    A flex request may queue silently for minutes; the model instance's SDK
    client timeout (bounded below the outer hard cap in quick mode — see
    ``flex_attempt_client_timeout``) surfaces that queue as a raised
    timeout/deadline error, which we treat like a capacity signal and fall back
    to the standard tier. Only errors raised *inside* the SDK call reach here;
    the outer ``run_with_hard_timeout`` fires in ``runtime.py``, never in the
    transport, so this cannot swallow the outer wall-clock cap.
    """
    if isinstance(exc, TimeoutError):  # asyncio.TimeoutError aliases this on 3.11+
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name or "deadline" in name:
        return True
    combined = str(exc).lower()
    return any(
        marker in combined
        for marker in (
            "timed out",
            "timeout",
            "deadline exceeded",
            "deadlineexceeded",
            # Google's actual 504 body: status 'DEADLINE_EXCEEDED' (underscore)
            # and message 'Deadline expired ...' — neither matched the two
            # forms above, so 504s silently bypassed the standard-tier fallback.
            "deadline_exceeded",
            "deadline expired",
        )
    )


def _stamp_service_tier(result: Any, tier: str) -> None:
    """Record the effective service tier on a ChatResult for cost tracking.

    Stamped on both ``llm_output`` and each message's ``response_metadata``
    so the token tracker can price flex calls at flex rates regardless of
    how the callback receives the result.
    """
    try:
        if getattr(result, "llm_output", None) is None:
            result.llm_output = {}
        result.llm_output["service_tier"] = tier
        for generation in getattr(result, "generations", []) or []:
            message = getattr(generation, "message", None)
            if message is not None:
                message.response_metadata["service_tier"] = tier
    except Exception:  # noqa: BLE001 - accounting must never break the call
        logger.debug("service_tier_stamp_failed", tier=tier)


class _TieredChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with Gemini request compatibility and flex support.

    langchain-google-genai 4.2.6 does not expose ``service_tier`` (see
    langchain-ai/langchain-google#1682), but its ``_prepare_request`` forwards
    unconsumed invoke kwargs into ``GenerateContentConfig``, which the
    installed google-genai SDK (>=1.75) accepts. We inject the tier there —
    ``_generate``/``_agenerate``/``_stream``/``_astream`` all route through
    ``_prepare_request``, so one override covers every path.

    Two error paths on a flex call, handled in ``_generate``/``_agenerate``:

    - **Capability rejection** (400-class naming the service tier): the model
      does not offer flex. The model is recorded in the process-wide negative
      cache (``mark_flex_unsupported``) so no further flex attempts are made,
      and the call is re-issued at the standard tier. This replaces any
      hardcoded model allowlist — eligibility is discovered, not declared.
    - **Capacity exhaustion** (429/503 after SDK-level retries): re-issue the
      call once at the standard tier (unless ``flex_fallback_to_standard`` is
      off). Not cached — the next call tries flex again.

    Streaming paths get tier injection but no fallback (this repo constructs
    non-streaming Gemini models; outer retry machinery covers streams).
    """

    service_tier: str | None = None
    flex_fallback_to_standard: bool = True

    def _prepare_params(
        self,
        stop: list[str] | None,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply model-specific field omissions after generation config is merged."""
        omitted_fields = _gemini_generation_fields_to_omit(self.model)
        if omitted_fields:
            kwargs = _without_gemini_generation_fields(kwargs, omitted_fields)
            if generation_config is not None:
                generation_config = _without_gemini_generation_fields(
                    generation_config,
                    omitted_fields,
                )

        params = super()._prepare_params(
            stop,
            generation_config=generation_config,
            **kwargs,
        )
        if not omitted_fields:
            return params

        payload = params.model_dump(exclude_unset=True)
        for field in omitted_fields:
            payload.pop(field, None)
        return type(params).model_validate(payload)

    def _prepare_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        omitted_fields = _gemini_generation_fields_to_omit(self.model)
        if omitted_fields:
            kwargs = _without_gemini_generation_fields(kwargs, omitted_fields)
        if (
            self.service_tier is not None
            and "service_tier" not in kwargs
            and not self._flex_ineligible()
        ):
            kwargs["service_tier"] = self.service_tier
        return super()._prepare_request(*args, **kwargs)

    def _flex_ineligible(self) -> bool:
        """Whether this call must not request flex — capability or health.

        Capability is permanent and model-scoped; health is a cool-off and
        provider-scoped (models share a vendor's queue). One predicate so the
        request builder and the tier resolver cannot disagree.
        """
        return is_flex_unsupported(self.model) or flex_degraded("google")

    def _effective_tier(self, kwargs: dict[str, Any]) -> str | None:
        requested_tier = kwargs.get("service_tier")
        if isinstance(requested_tier, str):
            return requested_tier
        if self.service_tier == "flex" and self._flex_ineligible():
            return None
        return self.service_tier

    def _flex_retry_tier(
        self, exc: BaseException, kwargs: dict[str, Any]
    ) -> str | None:
        """Tier to retry a failed flex call at, or None to re-raise."""
        if self._effective_tier(kwargs) != "flex":
            return None
        if is_flex_unsupported_error(exc):
            # Capability, not capacity: cache the downgrade regardless of
            # the capacity-fallback setting — flex can never work here.
            mark_flex_unsupported(self.model)
            return "standard"
        if self.flex_fallback_to_standard and _is_flex_capacity_error(exc):
            logger.warning(
                "flex_fallback_to_standard",
                model=self.model,
                **summarize_exception(exc, operation="gemini_flex_capacity"),
            )
            note_flex_fallback("google", reason="capacity", model=self.model)
            return "standard"
        if self.flex_fallback_to_standard and _is_flex_latency_timeout(exc):
            # Queued-too-long: the flex attempt exceeded its SDK client timeout.
            # Re-issue at standard rather than re-queue at flex, and record it —
            # past the threshold the provider stops being asked for the cool-off,
            # so one run cannot pay this wait once per call.
            logger.warning(
                "flex_fallback_to_standard",
                model=self.model,
                **summarize_exception(exc, operation="gemini_flex_latency"),
            )
            note_flex_fallback("google", reason="latency", model=self.model)
            return "standard"
        return None

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        try:
            result = super()._generate(*args, **kwargs)
        except Exception as exc:
            retry_tier = self._flex_retry_tier(exc, kwargs)
            if retry_tier is None:
                raise
            result = super()._generate(*args, **{**kwargs, "service_tier": retry_tier})
            _stamp_service_tier(result, retry_tier)
            return result
        tier = self._effective_tier(kwargs)
        if tier is not None:
            _stamp_service_tier(result, tier)
        return result

    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        try:
            result = await super()._agenerate(*args, **kwargs)
        except Exception as exc:
            retry_tier = self._flex_retry_tier(exc, kwargs)
            if retry_tier is None:
                raise
            result = await super()._agenerate(
                *args, **{**kwargs, "service_tier": retry_tier}
            )
            _stamp_service_tier(result, retry_tier)
            return result
        tier = self._effective_tier(kwargs)
        if tier is not None:
            _stamp_service_tier(result, tier)
        return result


_flex_fallback_chat_openai_cls: type[BaseChatModel] | None = None


def _get_flex_fallback_chat_openai_cls() -> type[BaseChatModel]:
    """Lazily define the flex-capable ChatOpenAI subclass.

    Defined inside a factory because ``langchain_openai`` is an optional
    dependency imported function-locally throughout this module.

    ``service_tier`` is a first-class ChatOpenAI field; the subclass only
    adds capacity fallback: langchain-openai merges invoke kwargs over
    ``_default_params`` (kwargs win), so re-calling with
    ``service_tier="auto"`` overrides the constructor's ``"flex"`` for the
    fallback attempt. Per OpenAI docs, flex-capacity 429s are not billed.
    """
    global _flex_fallback_chat_openai_cls
    if _flex_fallback_chat_openai_cls is not None:
        return _flex_fallback_chat_openai_cls

    from langchain_openai import ChatOpenAI

    class _FlexFallbackChatOpenAI(ChatOpenAI):
        flex_fallback_to_standard: bool = True

        def _flex_ineligible(self) -> bool:
            """Whether this call must not request flex — capability or health.

            Mirror of the Gemini transport's predicate: capability is permanent
            and model-scoped, health is a cool-off and provider-scoped.
            """
            return is_flex_unsupported(self.model_name) or flex_degraded("openai")

        def _effective_tier(self, kwargs: dict[str, Any]) -> str | None:
            requested_tier = kwargs.get("service_tier")
            if isinstance(requested_tier, str):
                return requested_tier
            if self.service_tier == "flex" and self._flex_ineligible():
                return "auto"
            return self.service_tier

        def _payload_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
            # A model learned to be flex-incapable — or a provider learned to be
            # congested — must not send "flex" from the constructor field; invoke
            # kwargs override _default_params.
            if (
                self.service_tier == "flex"
                and "service_tier" not in kwargs
                and self._flex_ineligible()
            ):
                return {**kwargs, "service_tier": "auto"}
            return kwargs

        def _flex_retry_tier(
            self, exc: BaseException, kwargs: dict[str, Any]
        ) -> str | None:
            """Tier to retry a failed flex call at, or None to re-raise."""
            if self._effective_tier(kwargs) != "flex":
                return None
            if is_flex_unsupported_error(exc):
                mark_flex_unsupported(self.model_name)
                return "auto"
            if self.flex_fallback_to_standard and _is_flex_capacity_error(exc):
                logger.warning(
                    "flex_fallback_to_standard",
                    model=self.model_name,
                    **summarize_exception(exc, operation="openai_flex_capacity"),
                )
                note_flex_fallback("openai", reason="capacity", model=self.model_name)
                return "auto"
            if self.flex_fallback_to_standard and _is_flex_latency_timeout(exc):
                # Queued-too-long: re-issue at standard (auto) rather than
                # re-queue at flex, and record it — past the threshold the
                # provider stops being asked for the cool-off.
                logger.warning(
                    "flex_fallback_to_standard",
                    model=self.model_name,
                    **summarize_exception(exc, operation="openai_flex_latency"),
                )
                note_flex_fallback("openai", reason="latency", model=self.model_name)
                return "auto"
            return None

        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            kwargs = self._payload_kwargs(kwargs)
            try:
                return super()._generate(*args, **kwargs)
            except Exception as exc:
                retry_tier = self._flex_retry_tier(exc, kwargs)
                if retry_tier is None:
                    raise
                return super()._generate(
                    *args, **{**kwargs, "service_tier": retry_tier}
                )

        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            kwargs = self._payload_kwargs(kwargs)
            try:
                return await super()._agenerate(*args, **kwargs)
            except Exception as exc:
                retry_tier = self._flex_retry_tier(exc, kwargs)
                if retry_tier is None:
                    raise
                return await super()._agenerate(
                    *args, **{**kwargs, "service_tier": retry_tier}
                )

    _flex_fallback_chat_openai_cls = _FlexFallbackChatOpenAI
    return _FlexFallbackChatOpenAI


def _apply_openai_service_tier(
    kwargs: dict[str, Any],
    *,
    label: str,
    service_tier: str | None = None,
    settings: Any | None = None,
) -> None:
    """Mutate ChatOpenAI constructor kwargs for OPENAI_SERVICE_TIER=flex.

    Sets the tier, enables fallback per config, and floors the client
    timeout (OpenAI recommends ~15 min for flex requests). No-op when the
    standard/auto tier is configured, or when this process has already
    learned the model rejects flex. Not applied to the APAC specialist,
    whose OpenAI-compatible backend is a different vendor.

    **Also a no-op for any other OpenAI-*compatible* base** (``OPENAI_API_BASE``
    pointing somewhere other than ``api.openai.com``). A service tier is an
    OpenAI pricing/queueing product; a compatible vendor does not sell one, and
    sending it either 400s or is silently swallowed. Before Aug 2026 this
    function also owned the client-timeout floor, which made
    ``OPENAI_SERVICE_TIER=flex`` the *only* way to give a compatible vendor more
    than the OpenAI-shaped default — an operator had to set a pricing flag to
    buy a timeout. That timeout is now
    ``OPENAI_COMPATIBLE_CLIENT_TIMEOUT_SECONDS``, applied by
    ``_apply_openai_api_base``, and the two concerns are independent.
    """
    settings = _settings_or_default(settings)
    if _openai_compatible_host(settings=settings) is not None:
        return
    if service_tier is None:
        flex_active = openai_flex_active(settings)
    else:
        flex_active = service_tier == "flex"
    if not flex_active:
        return
    if is_flex_unsupported(str(kwargs.get("model", ""))):
        return
    kwargs["service_tier"] = "flex"
    kwargs["flex_fallback_to_standard"] = settings.flex_fallback_to_standard
    kwargs["timeout"] = int(
        flex_attempt_client_timeout(
            float(kwargs.get("timeout", 120)),
            provider="openai",
            cfg=settings,
            label=label,
        )
    )


def _openai_compatible_host(*, settings: Any | None = None) -> str | None:
    """Return the host when ``OPENAI_API_BASE`` names a non-OpenAI vendor.

    ``None`` means "the default OpenAI endpoint", so every OpenAI-only behavior
    (service tiers, the Responses API) stays byte-identical.
    """
    settings = _settings_or_default(settings)
    base_url = settings.get_openai_api_base()
    if not isinstance(base_url, str) or not base_url:
        return None
    host = urlsplit(base_url).hostname
    if not host or host == "api.openai.com":
        return None
    return host


def _apply_openai_api_base(
    kwargs: dict[str, Any], *, settings: Any | None = None
) -> None:
    """Route OpenAI-plane calls to a custom base URL when configured.

    Single chokepoint for the consultant, auditor, editor, and writer-fallback
    seats (all of which construct via ``_construct_chat_openai``). A custom base
    is treated as an OpenAI-*compatible* — not OpenAI — endpoint: it speaks the
    Chat Completions API, so the OpenAI-only Responses API fields are dropped
    (mirrors the APAC/DeepSeek path), and it takes its own client timeout from
    ``OPENAI_COMPATIBLE_CLIENT_TIMEOUT_SECONDS`` rather than the OpenAI-shaped
    per-seat default. No-op when unset, so the default OpenAI path is
    byte-identical.
    """
    settings = _settings_or_default(settings)
    base_url = settings.get_openai_api_base()
    # Act only on a real, non-empty URL string. The accessor returns
    # ``str | None`` in production, so this guard is a no-op there — but it also
    # keeps the default OpenAI path byte-identical when ``config`` is a bare
    # test mock, whose attribute access yields a truthy ``MagicMock`` (not a
    # string) and would otherwise be injected as ``base_url``.
    if not isinstance(base_url, str) or not base_url:
        return
    kwargs["base_url"] = base_url
    if urlsplit(base_url).hostname != "api.openai.com":
        kwargs.pop("use_responses_api", None)
        kwargs.pop("output_version", None)
        kwargs["timeout"] = int(settings.openai_compatible_client_timeout_seconds)


def _openai_base_url_override(*, settings: Any | None = None) -> str | None:
    """Return the configured OpenAI-compatible base URL, or None when unset."""
    settings = _settings_or_default(settings)
    base_url = settings.get_openai_api_base()
    if not isinstance(base_url, str) or not base_url:
        return None
    return base_url


def _openai_endpoint_host(*, settings: Any | None = None) -> str | None:
    """Log-safe host of the configured base URL — never the full URL.

    Mirrors ``runtime_diagnostics.get_endpoint_host``: a base URL may carry a
    path, a query string, or embedded credentials, none of which may be logged.
    """
    base_url = _openai_base_url_override(settings=settings)
    if base_url is None:
        return None
    try:
        return urlsplit(base_url).hostname
    except ValueError:
        return None


def _warn_unknown_openai_reasoning_capability(model_name: str) -> None:
    """Warn once when a compatible endpoint serves an unregistered family.

    Silence here is how the kimi-k3 starvation went unnoticed: an unregistered
    reasoning model gets no effort bound *and* no completion-cap reserve, so
    its hidden reasoning quietly eats the visible-output budget.  Stock OpenAI
    models are not warned about — the table covers every reasoning family
    OpenAI serves, so a miss there is a plain non-reasoning model.
    """
    if _openai_base_url_override() is None:
        return
    if model_name in _warned_unknown_openai_reasoning:
        return
    _warned_unknown_openai_reasoning.add(model_name)
    logger.warning(
        "openai_reasoning_capability_unknown",
        model=model_name,
        endpoint_host=_openai_endpoint_host(),
        reason=(
            "no reasoning-effort profile registered for this model family; "
            "if it is a reasoning model its hidden reasoning shares the "
            "completion-token cap and can starve the visible output. Add the "
            "family to _OPENAI_REASONING_EFFORTS in src/llms.py."
        ),
    )


def _apply_openai_reasoning_effort(
    kwargs: dict[str, Any], *, model_name: str, preference: tuple[str, ...]
) -> str | None:
    """Set ``reasoning_effort`` when the model family documents one."""
    effort = _openai_reasoning_effort(model_name, preference=preference)
    if effort is None:
        _warn_unknown_openai_reasoning_capability(model_name)
        return None
    kwargs["reasoning_effort"] = effort
    return effort


def _apply_openai_generation_budget(
    kwargs: dict[str, Any],
    *,
    model_name: str,
    effort: str | None,
    settings: Any | None = None,
) -> GenerationBudget:
    """Convert an intent budget into an API cap with a reasoning reserve."""
    settings = _settings_or_default(settings)
    budget = _resolve_generation_budget(
        intent_tokens=kwargs["max_completion_tokens"],
        reserve_class=_reserve_class_for_effort(effort),
        reserve_enabled=_reasoning_counts_against_completion_cap(
            provider="openai",
            model_name=model_name,
            reasoning_effort=effort,
        ),
        settings=settings,
    )
    kwargs["max_completion_tokens"] = budget.api_cap_tokens
    return budget


def _construct_chat_openai(
    kwargs: dict[str, Any], *, settings: Any | None = None
) -> BaseChatModel:
    """Build ChatOpenAI, using the flex-fallback subclass when tiered."""
    settings = _settings_or_default(settings)
    _apply_openai_api_base(kwargs, settings=settings)
    if kwargs.get("service_tier") == "flex":
        return _get_flex_fallback_chat_openai_cls()(**kwargs)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**kwargs)


def _construct_chat_openai_for_settings(
    kwargs: dict[str, Any], settings: Any
) -> BaseChatModel:
    if settings is config:
        return _construct_chat_openai(kwargs)
    return _construct_chat_openai(kwargs, settings=settings)


class _LazyLLMProxy:
    """Lazily construct a default LLM on first use."""

    def __init__(self, factory):
        self._factory = factory
        self._instance = None

    def _get_instance(self):
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get_instance(), name)

    def __repr__(self) -> str:
        status = "initialized" if self._instance is not None else "lazy"
        return f"<_LazyLLMProxy {status}>"


def is_openai_consultant_available() -> bool:
    """Return whether OpenAI-backed consultant/auditor nodes can be enabled."""
    if not config.enable_consultant:
        return False
    if not config.get_openai_api_key():
        return False
    return find_spec("langchain_openai") is not None


def get_all_llm_instances() -> dict:
    """
    Get all tracked LLM instances for cleanup.

    Returns:
        Dict mapping instance names to LLM objects
    """
    return _llm_instances.copy()


def _resolve_gemini_service_tier(
    model_name: str, service_tier: str | None, *, settings: Any | None = None
) -> str | None:
    """Resolve the effective Gemini tier for a new model instance.

    ``None`` means "follow config": flex when ``GEMINI_SERVICE_TIER=flex``,
    unless this process has already learned the model rejects flex (no
    hardcoded model allowlist — eligibility is discovered via the vendor's
    own capability error and cached; see ``src/service_tiers.py``). An
    explicit ``"standard"`` pins the instance to the standard tier (e.g. the
    LLM-judge content inspector, an inline security path that must not queue).
    """
    settings = _settings_or_default(settings)
    if service_tier is not None:
        return service_tier if service_tier == "flex" else None
    if (
        gemini_flex_active(settings)
        and normalize_model_name(model_name).startswith("gemini-")
        and not is_flex_unsupported(model_name)
    ):
        return "flex"
    return None


def create_gemini_model(
    model_name: str,
    temperature: float,
    timeout: int,
    max_retries: int,
    streaming: bool = False,
    callbacks: list[BaseCallbackHandler] | None = None,
    thinking_level: str | None = None,
    max_output_tokens: int | None = None,
    reserve_class: Literal["default", "deep"] = "default",
    service_tier: str | None = None,
    api_key: str | None = None,
    settings: Any | None = None,
) -> BaseChatModel:
    """
    Generic factory for Gemini models.
    All created instances are tracked for proper cleanup at shutdown.

    ``service_tier=None`` follows ``GEMINI_SERVICE_TIER`` (flex only for
    documented flex-eligible models); pass ``"standard"`` to pin an instance
    to the standard tier regardless of config. Flex instances get their SDK
    timeout floored to ``FLEX_LLM_TIMEOUT_SECONDS`` — flex calls may queue
    1-15 minutes and a short socket timeout would turn that into retry churn.

    Note: API key is explicitly passed from config to avoid dependency on
    os.environ being populated by load_dotenv() (Pydantic Settings handles
    .env loading for our config, but third-party libs like LangChain still
    expect explicit api_key or os.environ values).
    """
    settings = _settings_or_default(settings)
    global _llm_instance_counter

    resolved_tier = _resolve_gemini_service_tier(
        model_name, service_tier, settings=settings
    )
    if resolved_tier == "flex":
        timeout = int(
            flex_attempt_client_timeout(
                float(timeout),
                provider="google",
                cfg=settings,
                label=f"gemini_sdk_timeout:{model_name}",
            )
        )

    intent_tokens = max_output_tokens or settings.llm_base_output_tokens
    thinking_budget = None
    if thinking_level and _is_gemini_v2_5(model_name):
        thinking_budget = _THINKING_BUDGETS.get(thinking_level, 4096)

    budget = _resolve_generation_budget(
        intent_tokens=intent_tokens,
        reserve_class=reserve_class,
        reserve_enabled=_reasoning_counts_against_completion_cap(
            provider="google",
            model_name=model_name,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
        ),
        settings=settings,
    )

    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
        "safety_settings": SAFETY_SETTINGS,
        "streaming": streaming,
        "rate_limiter": GLOBAL_RATE_LIMITER,
        "convert_system_message_to_human": False,
        "max_output_tokens": budget.api_cap_tokens,
        "callbacks": callbacks or [],
        "api_key": api_key if api_key is not None else settings.get_google_api_key(),
    }

    if thinking_level and _is_gemini_v3_or_greater(model_name):
        kwargs["thinking_level"] = thinking_level
        logger.debug(
            "thinking_level_applied", thinking_level=thinking_level, model=model_name
        )
    elif thinking_level and _is_gemini_v2_5(model_name):
        kwargs["thinking_budget"] = thinking_budget
        logger.debug(
            "thinking_budget_applied",
            thinking_level=thinking_level,
            thinking_budget=kwargs["thinking_budget"],
            model=model_name,
        )

    if resolved_tier is not None:
        kwargs["service_tier"] = resolved_tier
        kwargs["flex_fallback_to_standard"] = settings.flex_fallback_to_standard

    llm = _TieredChatGoogleGenerativeAI(**kwargs)
    _stamp_budget_metadata(
        llm,
        callbacks=kwargs["callbacks"],
        budget=budget,
        intent_attr="_configured_max_output_tokens",
        api_attr="_configured_api_output_tokens",
    )

    # Track instance for cleanup
    _llm_instance_counter += 1
    instance_name = f"gemini_{model_name}_{_llm_instance_counter}"
    _llm_instances[instance_name] = llm

    return llm


def create_quick_thinking_llm(
    temperature: float = 0.3,
    model: str | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
    max_output_tokens: int | None = None,
    service_tier: str | None = None,
    thinking_level_bump: bool = False,
    api_key: str | None = None,
    settings: Any | None = None,
) -> BaseChatModel:
    """
    Create a quick thinking LLM.
    If the QUICK_MODEL is Gemini 3+ or Gemini 2.5, this will set low reasoning.

    ``service_tier`` defaults to config (``GEMINI_SERVICE_TIER``); pass
    ``"standard"`` for latency-sensitive callers that must not queue on the
    flex tier (e.g. the LLM-judge content inspector).

    ``thinking_level_bump`` raises the thinking level one notch above the
    baseline (low → medium), clamped at the ceiling, for a quick-tier agent
    whose task is genuine synthesis rather than extraction (e.g. the Value
    Trap Detector distinguishing "announced" from "executed" corporate
    actions). No-op on models that do not support ``thinking_level``.
    """
    settings = _settings_or_default(settings)
    runtime_config = get_runtime_config(settings)
    model_name = model or runtime_config.quick_think_llm
    final_timeout = (
        timeout
        if timeout is not None
        else min(settings.api_timeout, settings.quick_llm_api_timeout_seconds)
    )
    final_retries = (
        max_retries if max_retries is not None else runtime_config.api_retry_attempts
    )

    thinking_level: Literal["low", "medium", "high"] | None = None
    if _is_gemini_v3_or_greater(model_name) or _is_gemini_v2_5(model_name):
        thinking_level = "low"
        if thinking_level_bump:
            thinking_level = bump_thinking_level(thinking_level)
    elif model_name.startswith("gemini-"):
        # Gemini model but NOT 3+ (likely 2.x)
        logger.warning("quick_model_gemini_2x_warning", model=model_name)

    _log_model_init_once(
        "quick", model_name, final_timeout, final_retries, thinking_level
    )
    return create_gemini_model(
        model_name,
        temperature,
        final_timeout,
        final_retries,
        callbacks=callbacks,
        thinking_level=thinking_level,
        max_output_tokens=max_output_tokens,
        reserve_class="default",
        service_tier=service_tier,
        api_key=api_key,
        settings=settings,
    )


def create_deep_thinking_llm(
    temperature: float = 0.1,
    model: str | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
    max_output_tokens: int | None = None,
    api_key: str | None = None,
    settings: Any | None = None,
) -> BaseChatModel:
    """
    Create a deep thinking LLM.
    If the DEEP_MODEL is Gemini 3+ or Gemini 2.5, this will set high reasoning.
    """
    settings = _settings_or_default(settings)
    runtime_config = get_runtime_config(settings)
    model_name = model or runtime_config.deep_think_llm
    final_timeout = timeout if timeout is not None else settings.api_timeout
    final_retries = (
        max_retries if max_retries is not None else runtime_config.api_retry_attempts
    )

    thinking_level: Literal["low", "medium", "high"] | None = None
    if _is_gemini_v3_or_greater(model_name) or _is_gemini_v2_5(model_name):
        thinking_level = "high"

    _log_model_init_once(
        "deep", model_name, final_timeout, final_retries, thinking_level
    )
    return create_gemini_model(
        model_name,
        temperature,
        final_timeout,
        final_retries,
        callbacks=callbacks,
        thinking_level=thinking_level,
        max_output_tokens=max_output_tokens,
        reserve_class="deep",
        api_key=api_key,
        settings=settings,
    )


APEX_SEATS = ("senior_fundamentals", "portfolio_manager")


def create_apex_llm(
    seat: str,
    *,
    quick_mode: bool,
    callbacks: list[BaseCallbackHandler] | None = None,
    max_output_tokens: int | None = None,
    settings: Any | None = None,
) -> BaseChatModel:
    """
    Create the LLM for a gate-critical (APEX) seat.

    The two seats — Senior Fundamentals (DATA_BLOCK rubric arithmetic feeding
    the hard <50% health/growth gates) and Portfolio Manager (gate checks,
    override logic, PM_BLOCK contract) — share the largest, most rule-dense
    prompts in the pipeline and fail flash-tier models at a steady rate.

    APEX_MODEL unset → legacy per-seat behavior (senior: quick tier; PM: deep
    tier in full mode, quick tier in --quick). Set → APEX_MODEL with deep-tier
    settings in full mode; in --quick, APEX_QUICK_MODEL when provided, else
    the plain quick floor — quick mode stays cheap and the degradation is an
    accepted trade-off.
    """
    explicit_settings = settings is not None
    settings = _settings_or_default(settings)
    if seat not in APEX_SEATS:
        raise ValueError(f"unknown apex seat: {seat!r} (expected one of {APEX_SEATS})")

    if not settings.apex_model:
        delegated_kwargs: dict[str, Any] = {
            "callbacks": callbacks,
            "max_output_tokens": max_output_tokens,
        }
        if explicit_settings:
            delegated_kwargs["settings"] = settings
        if seat == "portfolio_manager" and not quick_mode:
            return create_deep_thinking_llm(**delegated_kwargs)
        return create_quick_thinking_llm(**delegated_kwargs)

    model_name = settings.apex_model
    if quick_mode:
        if not settings.apex_quick_model:
            delegated_kwargs = {
                "callbacks": callbacks,
                "max_output_tokens": max_output_tokens,
            }
            if explicit_settings:
                delegated_kwargs["settings"] = settings
            return create_quick_thinking_llm(**delegated_kwargs)
        model_name = settings.apex_quick_model

    thinking_level: str | None = settings.apex_thinking_level
    if not (_is_gemini_v3_or_greater(model_name) or _is_gemini_v2_5(model_name)):
        logger.warning("apex_model_no_thinking_support", seat=seat, model=model_name)
        thinking_level = None

    runtime_config = get_runtime_config(settings)
    final_timeout = settings.api_timeout
    final_retries = runtime_config.api_retry_attempts
    _log_model_init_once(
        f"apex:{seat}", model_name, final_timeout, final_retries, thinking_level
    )
    return create_gemini_model(
        model_name,
        temperature=0.1,
        timeout=final_timeout,
        max_retries=final_retries,
        callbacks=callbacks,
        thinking_level=thinking_level,
        max_output_tokens=max_output_tokens,
        reserve_class="deep",
        # Gate-critical seat on the critical path under the tight --quick
        # budget: pin to standard so best-effort flex 503/504 queueing can't
        # burn the per-call budget (and lose the DATA_BLOCK / PM_BLOCK) before
        # any fallback. Full mode keeps config-driven flex (the flex floors +
        # the _is_flex_latency_timeout fallback give it room to recover).
        service_tier="standard" if quick_mode else None,
        api_key=settings.get_google_api_key(),
        settings=settings,
    )


def create_writer_fallback_llm(
    temperature: float = 0.1,
    callbacks: list[BaseCallbackHandler] | None = None,
    model: str | None = None,
    service_tier: str | None = None,
    api_key: str | None = None,
    settings: Any | None = None,
) -> BaseChatModel:
    """
    Create the Gemini fallback used when the Claude article writer is
    unavailable (e.g. missing key, billing failure).

    Unlike ``create_deep_thinking_llm`` (the analyst/PM deep-reasoning
    profile), this mirrors the *writer's* intent: long-form output with
    minimal reasoning. Article generation needs output budget, not a large
    hidden-reasoning budget — and for Gemini 3+ the hidden reasoning shares
    the same completion-token pool as the visible text, so a high
    ``thinking_level`` here starves the article and truncates it mid-sentence
    (the June 2026 1928.T failure). We therefore cap thinking at ``"low"`` and
    pin an explicit 16384-token visible budget (matching the Claude writer's
    ``max_tokens``); the budget machinery adds only the small "default" reserve
    on top, so the 16384 visible tokens can never be cannibalized by reasoning.
    """
    settings = _settings_or_default(settings)
    runtime_config = get_runtime_config(settings)
    model_name = model or runtime_config.deep_think_llm
    thinking_level: Literal["low"] | None = None
    if _is_gemini_v3_or_greater(model_name) or _is_gemini_v2_5(model_name):
        thinking_level = "low"
    return create_gemini_model(
        model_name,
        temperature,
        settings.api_timeout,
        runtime_config.api_retry_attempts,
        callbacks=callbacks,
        thinking_level=thinking_level,
        max_output_tokens=16384,
        reserve_class="default",
        service_tier=service_tier,
        api_key=api_key,
        settings=settings,
    )


@dataclass(frozen=True)
class WriterTier:
    """One tier of the article-writer fallback chain.

    ``label`` is a family-neutral tier name (never a model name — the honest-
    messaging invariant); ``build`` is lazy: the tier's LLM is constructed only
    when the tier is actually attempted, so an unused tier never creates a
    client or pollutes the instance registry.
    """

    label: str
    build: Callable[[], BaseChatModel]


def _writer_openai_tier_available(*, settings: Any | None = None) -> bool:
    """OpenAI usable for the writer fallback tier: lib + key + ENABLE_CONSULTANT.

    ``enable_consultant`` is the de-facto OpenAI master switch (consultant,
    auditor, and editor all gate on it) — the writer tier honors it too, so a
    deployment that disabled every other OpenAI agent never gets a GPT writer.
    """
    settings = _settings_or_default(settings)
    return (
        _langchain_openai_available()
        and settings.enable_consultant
        and bool(settings.get_openai_api_key())
    )


def create_writer_openai_fallback_llm(
    callbacks: list[BaseCallbackHandler] | None = None,
    *,
    model: str | None = None,
    settings: Any | None = None,
) -> BaseChatModel | None:
    """EDITOR_MODEL (OpenAI) as the preferred article-writer fallback.

    GPT-class prose with the writer's long-form output budget (16384, matching
    the Claude writer's ``max_tokens`` — the editor's 8192 cap would truncate
    articles) and low reasoning effort (prose needs output budget, not
    reasoning depth). Returns None when the tier is unavailable (missing
    lib/key or ``ENABLE_CONSULTANT=false``).
    """
    settings = _settings_or_default(settings)
    if not _writer_openai_tier_available(settings=settings):
        return None

    model_name = model or settings.editor_model or settings.consultant_model or "gpt-4o"
    logger.info("writer_fallback_tier_init", tier="editor_model", model=model_name)
    return _build_openai_chat(
        model_name,
        api_key=settings.get_openai_api_key(),
        callbacks=callbacks,
        max_completion_tokens=16384,
        service_tier_label="openai_sdk_timeout:writer_fallback",
        unthrottled_kind="writer_fallback",
        effort_preference=_EFFORT_PREFERENCE_PROSE,
        settings=settings,
    )


def writer_fallback_chain(
    callbacks: list[BaseCallbackHandler] | None = None,
    temperature: float = 0.7,
    *,
    settings: Any | None = None,
) -> list[WriterTier]:
    """Ordered, key/switch-aware fallback tiers for the article writer.

    EDITOR_MODEL (OpenAI) first when usable, then the Gemini floor — always
    present, so the chain is never empty. Tiers are lazy factories: nothing is
    constructed until a tier is actually attempted. ``temperature`` reaches
    only the Gemini tier (OpenAI reasoning models ignore sampling temperature);
    it defaults to the writer's 0.7 prose setting.
    """

    settings = _settings_or_default(settings)

    def _build_openai_tier() -> BaseChatModel:
        llm = create_writer_openai_fallback_llm(
            callbacks=callbacks,
            settings=settings,
        )
        if llm is None:  # availability changed between chain build and attempt
            raise RuntimeError("OpenAI writer fallback tier unavailable")
        return llm

    tiers: list[WriterTier] = []
    if _writer_openai_tier_available(settings=settings):
        tiers.append(WriterTier("editor_model", _build_openai_tier))
    tiers.append(
        WriterTier(
            "gemini_last_resort",
            lambda: create_writer_fallback_llm(
                temperature=temperature,
                callbacks=callbacks,
                settings=settings,
            ),
        )
    )
    return tiers


# Lazily initialize default instances so importing src.llms does not construct
# network-capable clients during test collection or light-weight CLI paths.
quick_thinking_llm = _LazyLLMProxy(create_quick_thinking_llm)
deep_thinking_llm = _LazyLLMProxy(create_deep_thinking_llm)


# ... (rest of the file is the same)
def create_consultant_llm(
    temperature: float = 0.3,
    model: str | None = None,
    timeout: int = 120,
    max_retries: int = 0,
    quick_mode: bool = False,
    callbacks: list[BaseCallbackHandler] | None = None,
    max_completion_tokens: int | None = None,
    settings: Any | None = None,
) -> BaseChatModel:
    """
    Create an OpenAI consultant LLM for cross-validation.

    Uses OpenAI (ChatGPT) instead of Gemini to provide independent perspective
    on Gemini's analysis outputs. This helps detect biases and groupthink.

    Args:
        temperature: Deprecated, ignored. Kept for API compatibility.
        model: Model name (overrides env vars if provided)
        timeout: Request timeout in seconds
        max_retries: Max retry attempts for failed requests
        quick_mode: If True, use CONSULTANT_QUICK_MODEL env var (default False)
        callbacks: Optional callback handlers for token tracking

    Returns:
        Configured ChatOpenAI instance

    Raises:
        ValueError: If OPENAI_API_KEY not found in environment
        ImportError: If langchain-openai package not installed

    Notes:
        - Requires OPENAI_API_KEY environment variable
        - Normal mode: Uses CONSULTANT_MODEL env var
        - Quick mode: Uses CONSULTANT_QUICK_MODEL env var (defaults to gpt-4o-mini)
        - Optional ENABLE_CONSULTANT env var (defaults to true)

    Example:
        >>> consultant_llm = create_consultant_llm()
        >>> result = consultant_llm.invoke("Review this analysis...")
        >>> quick_llm = create_consultant_llm(quick_mode=True)
    """
    settings = _settings_or_default(settings)
    if not _langchain_openai_available():
        raise ImportError(
            "langchain-openai package not found. Install with: "
            "pip install langchain-openai>=0.3.0"
        )

    # Check if consultant is enabled (via config, not os.environ)
    if not settings.enable_consultant:
        raise ValueError(
            "Consultant LLM is disabled. Set ENABLE_CONSULTANT=true to enable."
        )

    # Get OpenAI API key via config (SecretStr protected)
    api_key = settings.get_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "The consultant node requires an OpenAI API key for cross-validation. "
            "Add OPENAI_API_KEY to your .env file or set ENABLE_CONSULTANT=false."
        )

    # Get model name from config (not os.environ)
    if model:
        model_name = model
    elif quick_mode:
        model_name = settings.consultant_quick_model or settings.consultant_model
    else:
        model_name = settings.consultant_model

    logger.info(
        "consultant_llm_init", model=model_name, timeout=timeout, retries=max_retries
    )

    # Do NOT set temperature — multiple OpenAI model families (o-series,
    # gpt-5.x, and potentially future models) reject temperature != 1.0.
    # The consultant's precision comes from its structured prompt and
    # spot-check tool methodology, not from temperature settings.

    kwargs: dict[str, Any] = {
        "model": model_name,
        "timeout": timeout,
        "max_retries": max_retries,
        "api_key": api_key,
        "callbacks": callbacks or [],
        # Keep the consultant concise enough for bounded latency on multi-turn
        # tool use while leaving ample room for structured critique.
        "max_completion_tokens": max_completion_tokens or 8192,
        "streaming": False,
        "use_responses_api": True,
        "output_version": "responses/v1",
    }
    _apply_openai_service_tier(
        kwargs,
        label="openai_sdk_timeout:consultant",
        settings=settings,
    )
    _rl = _get_openai_rate_limiter_for_settings(settings)
    if _rl is not None:
        kwargs["rate_limiter"] = _rl
    else:
        _warn_openai_unthrottled_once("consultant")

    reasoning_effort = _apply_openai_reasoning_effort(
        kwargs,
        model_name=model_name,
        preference=_effort_preference_for_mode(quick_mode),
    )
    budget = _apply_openai_generation_budget(
        kwargs,
        model_name=model_name,
        effort=reasoning_effort,
        settings=settings,
    )

    llm = _construct_chat_openai_for_settings(kwargs, settings)
    _stamp_budget_metadata(
        llm,
        callbacks=kwargs["callbacks"],
        budget=budget,
        intent_attr="_configured_max_completion_tokens",
        api_attr="_configured_api_completion_tokens",
    )

    return llm


def create_auditor_llm(
    callbacks: list[BaseCallbackHandler] | None = None,
    max_completion_tokens: int | None = None,
    quick_mode: bool = False,
    model_name_override: str | None = None,
    settings: Any | None = None,
) -> BaseChatModel | None:
    """
    Create Auditor LLM with fallback logic.
    Returns None if ENABLE_CONSULTANT is false.

    Logic:
    1. If ENABLE_CONSULTANT is False -> None
    2. quick_mode=True and AUDITOR_QUICK_MODEL is set -> Use it
    3. If AUDITOR_MODEL is set -> Use it
    4. If CONSULTANT_MODEL is set -> Use it (Fallback)
    5. Default -> gpt-4o

    In quick mode, current GPT-5.x models use the documented ``low`` effort;
    normal mode uses ``medium``.  This includes GPT-5.6 Sol/Terra/Luna, which
    do not support the legacy ``minimal`` setting.
    """
    settings = _settings_or_default(settings)
    if not _langchain_openai_available():
        logger.warning("langchain_openai_missing")
        return None

    if not settings.enable_consultant:
        return None

    # Get OpenAI API key via config
    api_key = settings.get_openai_api_key()
    if not api_key:
        logger.warning("auditor_no_api_key")
        return None

    # Determine model: quick override -> specific -> consultant -> default
    if model_name_override:
        model_name = model_name_override
    elif quick_mode and settings.auditor_quick_model:
        model_name = settings.auditor_quick_model
    else:
        model_name = settings.auditor_model or settings.consultant_model or "gpt-4o"

    logger.info("auditor_llm_init", model=model_name, quick_mode=quick_mode)

    # Do NOT set temperature — multiple OpenAI model families (o-series reasoning
    # models, gpt-5.x) reject temperature != 1.0.  Forensic precision comes from
    # the structured prompt and deterministic tool calls, not from temperature=0.
    # Omitting temperature lets the SDK use each model's default safely.

    kwargs: dict[str, Any] = {
        "model": model_name,
        "timeout": 120,
        "max_retries": 3,
        "api_key": api_key,
        "callbacks": callbacks or [],
        # Fallback only: the graph always passes the auditor's centralized
        # budget explicitly. Deriving the same value here keeps a direct call
        # (tests, scripts) from silently running on a different cap.
        "max_completion_tokens": max_completion_tokens
        or _centralized_output_budget("Global Forensic Auditor"),
        "streaming": False,
        "use_responses_api": True,
        "output_version": "responses/v1",
    }
    _apply_openai_service_tier(
        kwargs,
        label="openai_sdk_timeout:auditor",
        settings=settings,
    )
    _rl = _get_openai_rate_limiter_for_settings(settings)
    if _rl is not None:
        kwargs["rate_limiter"] = _rl
    else:
        _warn_openai_unthrottled_once("auditor")

    reasoning_effort = _apply_openai_reasoning_effort(
        kwargs,
        model_name=model_name,
        preference=_effort_preference_for_mode(quick_mode),
    )
    budget = _apply_openai_generation_budget(
        kwargs,
        model_name=model_name,
        effort=reasoning_effort,
        settings=settings,
    )

    llm = _construct_chat_openai_for_settings(kwargs, settings)
    _stamp_budget_metadata(
        llm,
        callbacks=kwargs["callbacks"],
        budget=budget,
        intent_attr="_configured_max_completion_tokens",
        api_attr="_configured_api_completion_tokens",
    )
    return llm


def create_apac_specialist_llm(
    *,
    callbacks: list[BaseCallbackHandler] | None = None,
    max_completion_tokens: int | None = None,
    quick_mode: bool = False,
    thinking_enabled: bool = True,
    settings: Any | None = None,
) -> BaseChatModel | None:
    """Create the optional APAC Regional Specialist LLM."""
    settings = _settings_or_default(settings)
    if quick_mode:
        return None
    if not settings.enable_apac_specialist:
        return None

    api_key = settings.get_apac_specialist_api_key()
    if not api_key:
        logger.warning("apac_specialist_no_api_key")
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        logger.warning(
            "langchain_openai_missing_for_apac_specialist",
            **summarize_exception(
                exc, operation="langchain_openai_missing_for_apac_specialist"
            ),
        )
        return None

    model_name = settings.apac_specialist_model
    logger.info(
        "apac_specialist_llm_init",
        model=model_name,
        base_url=settings.apac_specialist_base_url,
    )

    kwargs: dict[str, Any] = {
        "model": model_name,
        "base_url": settings.apac_specialist_base_url,
        "api_key": api_key,
        "timeout": 240,
        "max_retries": 1 if thinking_enabled else 0,
        "callbacks": callbacks or [],
        "max_completion_tokens": max_completion_tokens or 8192,
        "streaming": False,
    }
    if thinking_enabled:
        # Deliberately the deepest setting the family documents — this seat is
        # a single-shot regional audit with a deep reserve behind it. An
        # unregistered family keeps the literal "max" (byte-identical to the
        # long-standing z.ai/DeepSeek behaviour); a registered one that does
        # not document "max" degrades to its deepest documented setting rather
        # than sending a value the vendor would reject.
        kwargs["reasoning_effort"] = (
            _openai_reasoning_effort(model_name, preference=_EFFORT_PREFERENCE_DEEPEST)
            or "max"
        )
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    budget = _resolve_generation_budget(
        intent_tokens=kwargs["max_completion_tokens"],
        reserve_class="deep",
        reserve_enabled=True,
        settings=settings,
    )
    kwargs["max_completion_tokens"] = budget.api_cap_tokens

    llm = ChatOpenAI(**kwargs)
    _stamp_budget_metadata(
        llm,
        callbacks=kwargs["callbacks"],
        budget=budget,
        intent_attr="_configured_max_completion_tokens",
        api_attr="_configured_api_completion_tokens",
    )
    return llm


def _is_claude_opus_adaptive_thinking_model(model_name: str) -> bool:
    match = re.search(r"opus-4-(\d+)", model_name)
    return bool(match and int(match.group(1)) >= 6)


def create_writer_llm(
    temperature: float = 0.7,
    timeout: int | None = None,
    max_retries: int = 3,
    callbacks: list[BaseCallbackHandler] | None = None,
    model: str | None = None,
    allow_fallback: bool = True,
    api_key_override: str | None = None,
    settings: Any | None = None,
) -> BaseChatModel:
    """
    Create the LLM for article writing.

    Prefers Claude (Anthropic) when CLAUDE_KEY is configured. Otherwise
    resolves the first available tier of writer_fallback_chain():
    EDITOR_MODEL (OpenAI) when usable, else the Gemini/DEEP_MODEL floor.

    Args:
        temperature: Sampling temperature. NOTE: Overridden to 1.0
                     when Claude adaptive thinking is active (API constraint).
        timeout: Request timeout in seconds (default from config)
        max_retries: Max retry attempts
        callbacks: Optional callback handlers

    Returns:
        ChatAnthropic or ChatGoogleGenerativeAI instance
    """
    settings = _settings_or_default(settings)
    api_key = (
        api_key_override
        if api_key_override is not None
        else settings.get_claude_api_key()
    )

    if not api_key and allow_fallback:
        logger.warning("writer_no_claude_key")
        return writer_fallback_chain(
            callbacks=callbacks,
            temperature=temperature,
            settings=settings,
        )[0].build()

    if not api_key:
        raise ValueError("Anthropic writer binding requires CLAUDE_KEY")

    # --- Claude path ---
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        if not allow_fallback:
            raise
        logger.warning("langchain_anthropic_missing")
        return writer_fallback_chain(
            callbacks=callbacks,
            temperature=temperature,
            settings=settings,
        )[0].build()

    model_name = model or settings.writer_model
    final_timeout = float(timeout if timeout is not None else settings.api_timeout)

    # Build kwargs — base configuration
    kwargs: dict = {
        "model": model_name,
        "max_tokens": 16384,
        "max_retries": max_retries,
        "timeout": final_timeout,
        "callbacks": callbacks or [],
        "anthropic_api_key": api_key,
    }

    # Thinking configuration — model-dependent
    if _is_claude_opus_adaptive_thinking_model(model_name):
        # Opus 4.6+: adaptive thinking (Claude decides when/how much to think)
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["effort"] = "high"
        # CRITICAL: Anthropic returns 400 if temperature != 1.0 with thinking.
        # Omit temperature entirely — SDK defaults to 1.0.
        logger.info(
            "writer_llm_adaptive_thinking", model=model_name, requested_temp=temperature
        )
    elif "sonnet" in model_name or "opus" in model_name:
        # Other Claude 4.x models: manual extended thinking
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8192}
        # Same temperature constraint applies
        logger.info("writer_llm_extended_thinking", model=model_name)
    else:
        # Haiku or unknown models: no thinking, use requested temperature
        kwargs["temperature"] = temperature
        logger.info("writer_llm_no_thinking", model=model_name, temperature=temperature)

    llm = ChatAnthropic(**kwargs)
    llm._configured_reasoning_reserve_tokens = (  # type: ignore[attr-defined]
        kwargs.get("thinking", {}).get("budget_tokens") or 0
        if isinstance(kwargs.get("thinking"), dict)
        else 0
    )

    # Track instance for cleanup (consistent with Gemini tracking)
    global _llm_instance_counter
    _llm_instance_counter += 1
    instance_name = f"claude_{model_name}_{_llm_instance_counter}"
    _llm_instances[instance_name] = llm

    logger.info("writer_llm_init", model=model_name, timeout=final_timeout)

    return llm


def _build_openai_chat(
    model_name: str,
    *,
    api_key: str,
    callbacks: list[BaseCallbackHandler] | None,
    max_completion_tokens: int,
    service_tier_label: str,
    service_tier: str | None = None,
    unthrottled_kind: str,
    effort_preference: tuple[str, ...],
    settings: Any | None = None,
) -> BaseChatModel:
    """Shared ChatOpenAI construction for the editor and writer-fallback tiers.

    Service-tier floor, process rate limiter, reasoning-effort resolution, and
    generation-budget handling live here once so the two callers cannot drift.
    ``effort_preference`` is an ordered wish list, not a literal: it is resolved
    against whatever the configured model family actually documents, so these
    seats behave correctly on an OpenAI-compatible endpoint too.
    """
    settings = _settings_or_default(settings)
    kwargs: dict[str, Any] = {
        "model": model_name,
        "timeout": 120,
        "max_retries": 3,
        "api_key": api_key,
        "callbacks": callbacks or [],
        "max_completion_tokens": max_completion_tokens,
        "streaming": False,
        "use_responses_api": True,
        "output_version": "responses/v1",
    }
    _apply_openai_service_tier(
        kwargs,
        label=service_tier_label,
        service_tier=service_tier,
        settings=settings,
    )
    _rl = _get_openai_rate_limiter_for_settings(settings)
    if _rl is not None:
        kwargs["rate_limiter"] = _rl
    else:
        _warn_openai_unthrottled_once(unthrottled_kind)
    reasoning_effort = _apply_openai_reasoning_effort(
        kwargs, model_name=model_name, preference=effort_preference
    )
    budget = _apply_openai_generation_budget(
        kwargs,
        model_name=model_name,
        effort=reasoning_effort,
        settings=settings,
    )

    llm = _construct_chat_openai_for_settings(kwargs, settings)
    _stamp_budget_metadata(
        llm,
        callbacks=kwargs["callbacks"],
        budget=budget,
        intent_attr="_configured_max_completion_tokens",
        api_attr="_configured_api_completion_tokens",
    )
    return llm


def create_editor_llm(
    callbacks: list[BaseCallbackHandler] | None = None,
    *,
    model: str | None = None,
    settings: Any | None = None,
) -> BaseChatModel | None:
    """
    Create Editor-in-Chief LLM for article revision and fact-checking.

    Returns None if ENABLE_CONSULTANT is false or OPENAI_API_KEY missing.

    Fallback chain: EDITOR_MODEL -> CONSULTANT_MODEL -> "gpt-4o"

    Args:
        callbacks: Optional callback handlers for token tracking

    Returns:
        ChatOpenAI instance or None if editor unavailable
    """
    settings = _settings_or_default(settings)
    if not _langchain_openai_available():
        logger.warning("langchain_openai_missing")
        return None

    if not settings.enable_consultant:
        logger.info("editor_disabled")
        return None

    api_key = settings.get_openai_api_key()
    if not api_key:
        logger.warning("editor_no_api_key")
        return None

    # Fallback chain: EDITOR_MODEL -> CONSULTANT_MODEL -> gpt-4o
    model_name = model or settings.editor_model or settings.consultant_model or "gpt-4o"

    logger.info("editor_llm_init", model=model_name)

    return _build_openai_chat(
        model_name,
        api_key=api_key,
        callbacks=callbacks,
        max_completion_tokens=_centralized_output_budget("Article Editor"),
        service_tier_label="openai_sdk_timeout:editor",
        unthrottled_kind="editor",
        effort_preference=_EFFORT_PREFERENCE_FULL,
        settings=settings,
    )


# Legacy symbol kept for compatibility with older tests/importers. Consultant
# instances are now created per call so quick/full mode configuration and
# per-run callbacks cannot bleed into one another.
_consultant_llm_instance = None


def get_consultant_llm(
    callbacks: list[BaseCallbackHandler] | None = None,
    quick_mode: bool = False,
    max_completion_tokens: int | None = None,
    model: str | None = None,
    settings: Any | None = None,
) -> BaseChatModel | None:
    """
    Get a consultant LLM instance for the current run.

    Uses lazy dependency checks to gracefully handle missing OPENAI_API_KEY.
    If consultant is disabled or API key is missing, returns None.

    Args:
        callbacks: Optional callback handlers for token tracking
        quick_mode: If True, use CONSULTANT_QUICK_MODEL (gpt-4o-mini by default)

    Returns:
        ChatOpenAI instance or None if consultant disabled/unavailable

    """
    settings = _settings_or_default(settings)
    # Check if consultant is enabled (via config, not os.environ)
    if not settings.enable_consultant:
        logger.info("consultant_disabled")
        return None

    # Check if API key exists (via config with SecretStr protection)
    if not settings.get_openai_api_key():
        logger.warning("consultant_no_api_key")
        return None

    try:
        return create_consultant_llm(
            callbacks=callbacks,
            quick_mode=quick_mode,
            max_completion_tokens=max_completion_tokens,
            model=model,
            settings=settings,
        )
    except Exception as e:
        logger.error(
            "consultant_llm_init_failed",
            model=(
                settings.consultant_quick_model or settings.consultant_model
                if quick_mode
                else settings.consultant_model
            ),
            quick_mode=quick_mode,
            exc_info=True,
            **summarize_exception(e, operation="consultant_llm_init"),
        )
        return None
