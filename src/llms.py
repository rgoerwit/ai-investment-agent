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
from src.llm_budgets import GenerationBudget, get_generation_budget
from src.runtime_config import get_runtime_config
from src.runtime_services import get_current_provider_runtime
from src.service_tiers import (
    flex_attempt_client_timeout,
    gemini_flex_active,
    is_flex_unsupported,
    is_flex_unsupported_error,
    mark_flex_unsupported,
    normalize_model_name,
    openai_flex_active,
)

logger = structlog.get_logger(__name__)
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

# Relax safety settings slightly for financial/market analysis context
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


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
    """
    Create a rate limiter from RPM (requests per minute) setting.
    """
    safety_factor = 0.8
    rps = (rpm / 60.0) * safety_factor
    max_bucket = max(5, int(rpm * 0.1))
    logger.info(
        "rate_limiter_configured", rpm=rpm, rps=round(rps, 2), max_bucket=max_bucket
    )
    return InMemoryRateLimiter(
        requests_per_second=rps, check_every_n_seconds=0.1, max_bucket_size=max_bucket
    )


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
            return provider_runtime.rate_limiter
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


def _get_openai_rate_limiter() -> InMemoryRateLimiter | None:
    """Return the shared OpenAI rate limiter, initializing it on first call."""
    global _openai_rate_limiter, _openai_rate_limiter_initialized
    if not _openai_rate_limiter_initialized:
        _openai_rate_limiter_initialized = True
        rpm = config_module.config.openai_rpm_limit
        if rpm is not None:
            _openai_rate_limiter = _create_rate_limiter_from_rpm(rpm)
    return _openai_rate_limiter


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
) -> GenerationBudget:
    return get_generation_budget(
        intent_tokens=intent_tokens,
        reserve_class=reserve_class,
        reserve_enabled=reserve_enabled,
        default_reserve_tokens=_coerce_int_setting(
            getattr(config, "llm_default_reasoning_reserve_tokens", None),
            2048,
        ),
        deep_reserve_tokens=_coerce_int_setting(
            getattr(config, "llm_deep_reasoning_reserve_tokens", None),
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
    """ChatGoogleGenerativeAI with Gemini flex-tier support.

    langchain-google-genai 4.2.5 does not expose ``service_tier`` (see
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

    def _prepare_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if (
            self.service_tier is not None
            and "service_tier" not in kwargs
            and not is_flex_unsupported(self.model)
        ):
            kwargs["service_tier"] = self.service_tier
        return super()._prepare_request(*args, **kwargs)

    def _effective_tier(self, kwargs: dict[str, Any]) -> str | None:
        if kwargs.get("service_tier") is not None:
            return kwargs["service_tier"]
        if self.service_tier == "flex" and is_flex_unsupported(self.model):
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
            return "standard"
        if self.flex_fallback_to_standard and _is_flex_latency_timeout(exc):
            # Queued-too-long: the flex attempt exceeded its (short, quick-mode)
            # SDK client timeout. Re-issue at standard rather than re-queue at
            # flex. Not cached — the queue is transient.
            logger.warning(
                "flex_fallback_to_standard",
                model=self.model,
                **summarize_exception(exc, operation="gemini_flex_latency"),
            )
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


_flex_fallback_chat_openai_cls: type | None = None


def _get_flex_fallback_chat_openai_cls() -> type:
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

        def _effective_tier(self, kwargs: dict[str, Any]) -> str | None:
            if kwargs.get("service_tier") is not None:
                return kwargs["service_tier"]
            if self.service_tier == "flex" and is_flex_unsupported(self.model_name):
                return "auto"
            return self.service_tier

        def _payload_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
            # A model learned to be flex-incapable must not send "flex" from
            # the constructor field; invoke kwargs override _default_params.
            if (
                self.service_tier == "flex"
                and "service_tier" not in kwargs
                and is_flex_unsupported(self.model_name)
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
                return "auto"
            if self.flex_fallback_to_standard and _is_flex_latency_timeout(exc):
                # Queued-too-long: re-issue at standard (auto) rather than
                # re-queue at flex. Not cached — the queue is transient.
                logger.warning(
                    "flex_fallback_to_standard",
                    model=self.model_name,
                    **summarize_exception(exc, operation="openai_flex_latency"),
                )
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


def _apply_openai_service_tier(kwargs: dict[str, Any], *, label: str) -> None:
    """Mutate ChatOpenAI constructor kwargs for OPENAI_SERVICE_TIER=flex.

    Sets the tier, enables fallback per config, and floors the client
    timeout (OpenAI recommends ~15 min for flex requests). No-op when the
    standard/auto tier is configured, or when this process has already
    learned the model rejects flex. Not applied to the APAC specialist,
    whose OpenAI-compatible backend is a different vendor.
    """
    if not openai_flex_active():
        return
    if is_flex_unsupported(str(kwargs.get("model", ""))):
        return
    kwargs["service_tier"] = "flex"
    kwargs["flex_fallback_to_standard"] = config.flex_fallback_to_standard
    kwargs["timeout"] = int(
        flex_attempt_client_timeout(
            float(kwargs.get("timeout", 120)),
            provider="openai",
            label=label,
        )
    )


def _construct_chat_openai(kwargs: dict[str, Any]) -> BaseChatModel:
    """Build ChatOpenAI, using the flex-fallback subclass when tiered."""
    if kwargs.get("service_tier") == "flex":
        return _get_flex_fallback_chat_openai_cls()(**kwargs)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**kwargs)


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
    model_name: str, service_tier: str | None
) -> str | None:
    """Resolve the effective Gemini tier for a new model instance.

    ``None`` means "follow config": flex when ``GEMINI_SERVICE_TIER=flex``,
    unless this process has already learned the model rejects flex (no
    hardcoded model allowlist — eligibility is discovered via the vendor's
    own capability error and cached; see ``src/service_tiers.py``). An
    explicit ``"standard"`` pins the instance to the standard tier (e.g. the
    LLM-judge content inspector, an inline security path that must not queue).
    """
    if service_tier is not None:
        return service_tier if service_tier == "flex" else None
    if (
        gemini_flex_active()
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
    global _llm_instance_counter

    resolved_tier = _resolve_gemini_service_tier(model_name, service_tier)
    if resolved_tier == "flex":
        timeout = int(
            flex_attempt_client_timeout(
                float(timeout),
                provider="google",
                label=f"gemini_sdk_timeout:{model_name}",
            )
        )

    intent_tokens = max_output_tokens or config.llm_base_output_tokens
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
        "api_key": config.get_google_api_key(),  # Explicit API key from config
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
        kwargs["flex_fallback_to_standard"] = config.flex_fallback_to_standard

    llm = _TieredChatGoogleGenerativeAI(**kwargs)
    _stamp_budget_metadata(
        llm,
        callbacks=kwargs["callbacks"],
        budget=budget,
        intent_attr="_configured_max_output_tokens",
        api_attr="_configured_api_output_tokens",
    )
    if thinking_level and _is_gemini_v3_or_greater(model_name):
        llm.thinking_level = thinking_level  # type: ignore[attr-defined,assignment]

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
) -> BaseChatModel:
    """
    Create a quick thinking LLM.
    If the QUICK_MODEL is Gemini 3+ or Gemini 2.5, this will set low reasoning.

    ``service_tier`` defaults to config (``GEMINI_SERVICE_TIER``); pass
    ``"standard"`` for latency-sensitive callers that must not queue on the
    flex tier (e.g. the LLM-judge content inspector).
    """
    runtime_config = get_runtime_config(config)
    model_name = model or runtime_config.quick_think_llm
    final_timeout = (
        timeout
        if timeout is not None
        else min(config.api_timeout, config.quick_llm_api_timeout_seconds)
    )
    final_retries = (
        max_retries if max_retries is not None else runtime_config.api_retry_attempts
    )

    thinking_level: Literal["low", "medium", "high"] | None = None
    if _is_gemini_v3_or_greater(model_name) or _is_gemini_v2_5(model_name):
        thinking_level = "low"
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
    )


def create_deep_thinking_llm(
    temperature: float = 0.1,
    model: str | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
    max_output_tokens: int | None = None,
) -> BaseChatModel:
    """
    Create a deep thinking LLM.
    If the DEEP_MODEL is Gemini 3+ or Gemini 2.5, this will set high reasoning.
    """
    runtime_config = get_runtime_config(config)
    model_name = model or runtime_config.deep_think_llm
    final_timeout = timeout if timeout is not None else config.api_timeout
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
    )


APEX_SEATS = ("senior_fundamentals", "portfolio_manager")


def create_apex_llm(
    seat: str,
    *,
    quick_mode: bool,
    callbacks: list[BaseCallbackHandler] | None = None,
    max_output_tokens: int | None = None,
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
    if seat not in APEX_SEATS:
        raise ValueError(f"unknown apex seat: {seat!r} (expected one of {APEX_SEATS})")

    if not config.apex_model:
        if seat == "portfolio_manager" and not quick_mode:
            return create_deep_thinking_llm(
                callbacks=callbacks, max_output_tokens=max_output_tokens
            )
        return create_quick_thinking_llm(
            callbacks=callbacks, max_output_tokens=max_output_tokens
        )

    model_name = config.apex_model
    if quick_mode:
        if not config.apex_quick_model:
            return create_quick_thinking_llm(
                callbacks=callbacks, max_output_tokens=max_output_tokens
            )
        model_name = config.apex_quick_model

    thinking_level: str | None = config.apex_thinking_level
    if not (_is_gemini_v3_or_greater(model_name) or _is_gemini_v2_5(model_name)):
        logger.warning("apex_model_no_thinking_support", seat=seat, model=model_name)
        thinking_level = None

    runtime_config = get_runtime_config(config)
    final_timeout = config.api_timeout
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
    )


def create_writer_fallback_llm(
    temperature: float = 0.1,
    callbacks: list[BaseCallbackHandler] | None = None,
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
    runtime_config = get_runtime_config(config)
    model_name = runtime_config.deep_think_llm
    thinking_level: Literal["low"] | None = None
    if _is_gemini_v3_or_greater(model_name) or _is_gemini_v2_5(model_name):
        thinking_level = "low"
    return create_gemini_model(
        model_name,
        temperature,
        config.api_timeout,
        runtime_config.api_retry_attempts,
        callbacks=callbacks,
        thinking_level=thinking_level,
        max_output_tokens=16384,
        reserve_class="default",
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


def _writer_openai_tier_available() -> bool:
    """OpenAI usable for the writer fallback tier: lib + key + ENABLE_CONSULTANT.

    ``enable_consultant`` is the de-facto OpenAI master switch (consultant,
    auditor, and editor all gate on it) — the writer tier honors it too, so a
    deployment that disabled every other OpenAI agent never gets a GPT writer.
    """
    return (
        _langchain_openai_available()
        and config.enable_consultant
        and bool(config.get_openai_api_key())
    )


def create_writer_openai_fallback_llm(
    callbacks: list[BaseCallbackHandler] | None = None,
) -> BaseChatModel | None:
    """EDITOR_MODEL (OpenAI) as the preferred article-writer fallback.

    GPT-class prose with the writer's long-form output budget (16384, matching
    the Claude writer's ``max_tokens`` — the editor's 8192 cap would truncate
    articles) and low reasoning effort (prose needs output budget, not
    reasoning depth). Returns None when the tier is unavailable (missing
    lib/key or ``ENABLE_CONSULTANT=false``).
    """
    if not _writer_openai_tier_available():
        return None

    model_name = config.editor_model or config.consultant_model or "gpt-4o"
    logger.info("writer_fallback_tier_init", tier="editor_model", model=model_name)
    return _build_openai_chat(
        model_name,
        api_key=config.get_openai_api_key(),
        callbacks=callbacks,
        max_completion_tokens=16384,
        service_tier_label="openai_sdk_timeout:writer_fallback",
        unthrottled_kind="writer_fallback",
        gpt5_reasoning_effort="low",
    )


def writer_fallback_chain(
    callbacks: list[BaseCallbackHandler] | None = None,
    temperature: float = 0.7,
) -> list[WriterTier]:
    """Ordered, key/switch-aware fallback tiers for the article writer.

    EDITOR_MODEL (OpenAI) first when usable, then the Gemini floor — always
    present, so the chain is never empty. Tiers are lazy factories: nothing is
    constructed until a tier is actually attempted. ``temperature`` reaches
    only the Gemini tier (OpenAI reasoning models ignore sampling temperature);
    it defaults to the writer's 0.7 prose setting.
    """

    def _build_openai_tier() -> BaseChatModel:
        llm = create_writer_openai_fallback_llm(callbacks=callbacks)
        if llm is None:  # availability changed between chain build and attempt
            raise RuntimeError("OpenAI writer fallback tier unavailable")
        return llm

    tiers: list[WriterTier] = []
    if _writer_openai_tier_available():
        tiers.append(WriterTier("editor_model", _build_openai_tier))
    tiers.append(
        WriterTier(
            "gemini_last_resort",
            lambda: create_writer_fallback_llm(
                temperature=temperature, callbacks=callbacks
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
    if not _langchain_openai_available():
        raise ImportError(
            "langchain-openai package not found. Install with: "
            "pip install langchain-openai>=0.3.0"
        )

    # Check if consultant is enabled (via config, not os.environ)
    if not config.enable_consultant:
        raise ValueError(
            "Consultant LLM is disabled. Set ENABLE_CONSULTANT=true to enable."
        )

    # Get OpenAI API key via config (SecretStr protected)
    api_key = config.get_openai_api_key()
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
        model_name = config.consultant_quick_model or config.consultant_model
    else:
        model_name = config.consultant_model

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
    _apply_openai_service_tier(kwargs, label="openai_sdk_timeout:consultant")
    _rl = _get_openai_rate_limiter()
    if _rl is not None:
        kwargs["rate_limiter"] = _rl
    else:
        _warn_openai_unthrottled_once("consultant")

    # GPT-5 non-pro models support configurable reasoning effort. Quick mode
    # uses a lower setting to keep the consultant active without paying full
    # synthesis cost; normal mode preserves the current medium effort.
    # Note: gpt-5.x-mini variants reject "minimal" — only full gpt-5.x accepts it.
    if model_name.startswith("gpt-5") and "pro" not in model_name:
        if quick_mode:
            kwargs["reasoning_effort"] = "low" if "mini" in model_name else "minimal"
        else:
            kwargs["reasoning_effort"] = "medium"

    budget = _resolve_generation_budget(
        intent_tokens=kwargs["max_completion_tokens"],
        reserve_class="default",
        reserve_enabled=_reasoning_counts_against_completion_cap(
            provider="openai",
            model_name=model_name,
            reasoning_effort=kwargs.get("reasoning_effort"),
        ),
    )
    kwargs["max_completion_tokens"] = budget.api_cap_tokens

    llm = _construct_chat_openai(kwargs)
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

    In quick mode, gpt-5 reasoning effort is dropped to "minimal" to keep the
    auditor cheap on screening passes; normal mode preserves "medium".
    """
    if not _langchain_openai_available():
        logger.warning("langchain_openai_missing")
        return None

    if not config.enable_consultant:
        return None

    # Get OpenAI API key via config
    api_key = config.get_openai_api_key()
    if not api_key:
        logger.warning("auditor_no_api_key")
        return None

    # Determine model: quick override -> specific -> consultant -> default
    if quick_mode and config.auditor_quick_model:
        model_name = config.auditor_quick_model
    else:
        model_name = config.auditor_model or config.consultant_model or "gpt-4o"

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
        "max_completion_tokens": max_completion_tokens
        or (6144 if quick_mode else 16384),
        "streaming": False,
        "use_responses_api": True,
        "output_version": "responses/v1",
    }
    _apply_openai_service_tier(kwargs, label="openai_sdk_timeout:auditor")
    _rl = _get_openai_rate_limiter()
    if _rl is not None:
        kwargs["rate_limiter"] = _rl
    else:
        _warn_openai_unthrottled_once("auditor")

    # gpt-5.x-mini variants reject "minimal" — only full gpt-5.x accepts it.
    if model_name.startswith("gpt-5") and "pro" not in model_name:
        if quick_mode:
            kwargs["reasoning_effort"] = "low" if "mini" in model_name else "minimal"
        else:
            kwargs["reasoning_effort"] = "medium"

    budget = _resolve_generation_budget(
        intent_tokens=kwargs["max_completion_tokens"],
        reserve_class="default",
        reserve_enabled=_reasoning_counts_against_completion_cap(
            provider="openai",
            model_name=model_name,
            reasoning_effort=kwargs.get("reasoning_effort"),
        ),
    )
    kwargs["max_completion_tokens"] = budget.api_cap_tokens

    llm = _construct_chat_openai(kwargs)
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
) -> BaseChatModel | None:
    """Create the optional APAC Regional Specialist LLM."""
    if quick_mode:
        return None
    if not config.enable_apac_specialist:
        return None

    api_key = config.get_apac_specialist_api_key()
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

    model_name = config.apac_specialist_model
    logger.info(
        "apac_specialist_llm_init",
        model=model_name,
        base_url=config.apac_specialist_base_url,
    )

    kwargs: dict[str, Any] = {
        "model": model_name,
        "base_url": config.apac_specialist_base_url,
        "api_key": api_key,
        "timeout": 240,
        "max_retries": 1,
        "callbacks": callbacks or [],
        "max_completion_tokens": max_completion_tokens or 8192,
        "streaming": False,
        "reasoning_effort": "max",
        "extra_body": {"thinking": {"type": "enabled"}},
    }

    budget = _resolve_generation_budget(
        intent_tokens=kwargs["max_completion_tokens"],
        reserve_class="deep",
        reserve_enabled=True,
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
    api_key = config.get_claude_api_key()

    if not api_key:
        logger.warning("writer_no_claude_key")
        return writer_fallback_chain(callbacks=callbacks, temperature=temperature)[
            0
        ].build()

    # --- Claude path ---
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        logger.warning("langchain_anthropic_missing")
        return writer_fallback_chain(callbacks=callbacks, temperature=temperature)[
            0
        ].build()

    model_name = config.writer_model
    final_timeout = float(timeout if timeout is not None else config.api_timeout)

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
    unthrottled_kind: str,
    gpt5_reasoning_effort: str,
) -> BaseChatModel:
    """Shared ChatOpenAI construction for the editor and writer-fallback tiers.

    Service-tier floor, process rate limiter, gpt-5 reasoning effort, and
    generation-budget handling live here once so the two callers cannot drift.
    """
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
    _apply_openai_service_tier(kwargs, label=service_tier_label)
    _rl = _get_openai_rate_limiter()
    if _rl is not None:
        kwargs["rate_limiter"] = _rl
    else:
        _warn_openai_unthrottled_once(unthrottled_kind)
    if model_name.startswith("gpt-5") and "pro" not in model_name:
        kwargs["reasoning_effort"] = gpt5_reasoning_effort
    budget = _resolve_generation_budget(
        intent_tokens=kwargs["max_completion_tokens"],
        reserve_class="default",
        reserve_enabled=_reasoning_counts_against_completion_cap(
            provider="openai",
            model_name=model_name,
            reasoning_effort=kwargs.get("reasoning_effort"),
        ),
    )
    kwargs["max_completion_tokens"] = budget.api_cap_tokens

    llm = _construct_chat_openai(kwargs)
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
    if not _langchain_openai_available():
        logger.warning("langchain_openai_missing")
        return None

    if not config.enable_consultant:
        logger.info("editor_disabled")
        return None

    api_key = config.get_openai_api_key()
    if not api_key:
        logger.warning("editor_no_api_key")
        return None

    # Fallback chain: EDITOR_MODEL -> CONSULTANT_MODEL -> gpt-4o
    model_name = config.editor_model or config.consultant_model or "gpt-4o"

    logger.info("editor_llm_init", model=model_name)

    return _build_openai_chat(
        model_name,
        api_key=api_key,
        callbacks=callbacks,
        max_completion_tokens=8192,
        service_tier_label="openai_sdk_timeout:editor",
        unthrottled_kind="editor",
        gpt5_reasoning_effort="medium",
    )


# Legacy symbol kept for compatibility with older tests/importers. Consultant
# instances are now created per call so quick/full mode configuration and
# per-run callbacks cannot bleed into one another.
_consultant_llm_instance = None


def get_consultant_llm(
    callbacks: list[BaseCallbackHandler] | None = None,
    quick_mode: bool = False,
    max_completion_tokens: int | None = None,
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
    # Check if consultant is enabled (via config, not os.environ)
    if not config.enable_consultant:
        logger.info("consultant_disabled")
        return None

    # Check if API key exists (via config with SecretStr protection)
    if not config.get_openai_api_key():
        logger.warning("consultant_no_api_key")
        return None

    try:
        return create_consultant_llm(
            callbacks=callbacks,
            quick_mode=quick_mode,
            max_completion_tokens=max_completion_tokens,
        )
    except Exception as e:
        logger.error(
            "consultant_llm_init_failed",
            model=(
                config.consultant_quick_model or config.consultant_model
                if quick_mode
                else config.consultant_model
            ),
            quick_mode=quick_mode,
            exc_info=True,
            **summarize_exception(e, operation="consultant_llm_init"),
        )
        return None
