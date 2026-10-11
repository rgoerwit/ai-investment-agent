"""Deadline-aware OpenAI-family and Anthropic chat transports.

Every OpenAI, OpenAI-compatible and Anthropic client in this repository is built
here, by ``new_openai_chat`` and ``new_anthropic_chat``. There is no plain-client
branch: a plain ``ChatOpenAI``/``ChatAnthropic`` never passes the per-dispatch
check (``src/llm_runtime/dispatch.py``), so inside an application invocation it
would send its construction-time timeout instead of one sized from the
invocation's absolute deadline.

**The receiving provider is a required field, set by every constructor.**
``dispatch_provider`` selects the request ceiling (``policy_for_provider``) and
keys flex health. It is never inferred and has no default: an OpenAI-compatible
vendor that silently defaulted to ``"openai"`` would be sized against the
OpenAI request cap and keyed into OpenAI's flex health (the defect this module
closes). The factory's ``_llm_runtime_provider`` stamp arrives only after
construction, and legacy constructors never receive it, so it cannot serve.

A compatible vendor (any ``dispatch_provider`` other than ``"openai"``) has no
service tiers: it may not be built with one, its failover copy carries none,
a dispatch never adds one, and one passed at invocation or through bound kwargs
is refused before anything is sent.

**One SDK attempt per request is a property of the class.** ``max_retries``
defaults to zero retries and any other value is refused at construction
(``require_single_sdk_attempt``): the runtime loop owns every retry, so no
constructor passes an attempt count.

The classes are defined lazily (``langchain_openai``/``langchain_anthropic``
are optional) from the vendor package's defining module, and a base that is
not a real chat model is refused rather than cached, so a test that patches
the package attribute cannot poison the class for the rest of the process.
Tests replace ``new_openai_chat``/``new_anthropic_chat`` -- the one patch seam
for constructing these transports.
"""

import importlib
import threading
from typing import Any, ClassVar, Literal

import structlog
from langchain_core.language_models import BaseChatModel
from pydantic import Field, model_validator

from src.error_safety import summarize_exception
from src.llm_runtime.dispatch import (
    _Dispatch,
    _is_flex_capacity_error,
    _is_flex_latency_timeout,
    _plan_dispatch,
    _run_dispatch_async,
    _run_dispatch_sync,
    refuse_streaming,
    require_single_sdk_attempt,
)
from src.llm_runtime.failover import ModelFailoverMixin
from src.llm_runtime.timeouts import BudgetExhausted
from src.service_tiers import (
    flex_degraded,
    is_flex_unsupported,
    is_flex_unsupported_error,
    mark_flex_latency_fallback,
    mark_flex_unsupported,
    mark_tier_attempt,
    note_flex_fallback,
)

logger = structlog.get_logger(__name__)

# The ``dispatch_provider`` of the native OpenAI endpoint, the only one that
# sells service tiers. Every other value is an OpenAI-compatible vendor.
OPENAI_NATIVE_PROVIDER = "openai"
# The ``dispatch_provider`` of a legacy client pointed at a non-OpenAI host
# (``OPENAI_API_BASE`` or the APAC specialist base URL), where no binding names
# the vendor. Any label other than ``"openai"`` takes the compatible ceiling.
OPENAI_COMPATIBLE_PROVIDER = "openai_compatible"
ANTHROPIC_PROVIDER = "anthropic"

_openai_cls: type[BaseChatModel] | None = None
_anthropic_cls: type[BaseChatModel] | None = None
_CLASS_LOCK = threading.Lock()


def _normalized_provider(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "dispatch_provider is required: every constructor names the vendor "
            "that receives the request"
        )
    return value.strip().lower()


def _refuse_streaming_field(model: Any) -> None:
    # ``streaming=True`` puts ``stream: true`` into ChatOpenAI's default request
    # params, so even the non-streaming ``_generate`` would open a stream.
    if getattr(model, "streaming", False):
        raise ValueError(
            "streaming=True is not qualified: streaming bypasses the "
            "per-dispatch request deadline"
        )


def _real_base(module_name: str, class_name: str) -> type[BaseChatModel]:
    base = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(base, type) and issubclass(base, BaseChatModel)):
        raise TypeError(
            f"{module_name}.{class_name} is not a chat model class; patch "
            "src.llm_runtime.adapters.transports.new_openai_chat or "
            "new_anthropic_chat instead of the vendor class"
        )
    return base


def deadline_aware_openai_cls() -> type[BaseChatModel]:
    """The OpenAI and OpenAI-compatible transport class, defined once on first use.

    Under a lock: two first callers racing would otherwise define two classes,
    and an instance of the loser would fail every ``isinstance`` check.
    """
    global _openai_cls
    with _CLASS_LOCK:
        if _openai_cls is None:
            _openai_cls = _define_openai_cls()
        return _openai_cls


def deadline_aware_anthropic_cls() -> type[BaseChatModel]:
    """The Anthropic transport class, defined once on first use."""
    global _anthropic_cls
    with _CLASS_LOCK:
        if _anthropic_cls is None:
            _anthropic_cls = _define_anthropic_cls()
        return _anthropic_cls


def _define_openai_cls() -> type[BaseChatModel]:
    chat_openai = _real_base("langchain_openai.chat_models.base", "ChatOpenAI")

    class _FlexFallbackChatOpenAI(ModelFailoverMixin, chat_openai):  # type: ignore[valid-type,misc]
        """ChatOpenAI sized at dispatch, with flex fallback on the native endpoint.

        ``service_tier`` is a first-class ChatOpenAI field; langchain-openai
        merges invoke kwargs over ``_default_params`` (kwargs win), so
        re-calling with ``service_tier="auto"`` overrides the constructor's
        ``"flex"`` for the fallback attempt. Per OpenAI docs, flex-capacity
        429s are not billed. The per-request ``timeout`` reaches both the
        Responses and Chat Completions ``create()`` calls; the client is built
        for one attempt (the SDK has no per-request retry count).
        """

        _failover_model_field: ClassVar[str] = "model_name"
        _failover_standard_tier: ClassVar[str] = "auto"

        # Required, no default: see the module docstring.
        dispatch_provider: str
        # Zero SDK retries; the parent's None would mean the SDK default of 2.
        max_retries: int | None = 0
        flex_fallback_to_standard: bool = True
        failover_model: str | None = None
        # The seat's TimeoutClass value, used to size a direct caller's flex request.
        budget_timeout_class: str | None = None
        # The seat policy's explicit client timeout, stamped at construction
        # (``attach_request_cap_override``) and resolved at every dispatch.
        request_cap_override: float | None = Field(default=None, gt=0)
        # Streaming would bypass the dispatch check; route every public
        # stream/invoke path through ``_generate``/``_agenerate`` instead.
        disable_streaming: bool | Literal["tool_calling"] = True

        @model_validator(mode="after")
        def _validate_dispatch_provider(self) -> "_FlexFallbackChatOpenAI":
            self.dispatch_provider = _normalized_provider(self.dispatch_provider)
            _refuse_streaming_field(self)
            require_single_sdk_attempt(self, 0)
            if self.service_tier is not None:
                self._refuse_tier_off_native()
            return self

        def _refuse_tier_off_native(self) -> None:
            if not self._native():
                raise ValueError(
                    "service_tier is an OpenAI product; the OpenAI-compatible "
                    f"vendor {self.dispatch_provider!r} must not receive one"
                )

        def _native(self) -> bool:
            return self.dispatch_provider == OPENAI_NATIVE_PROVIDER

        def _standard_tier(self) -> str | None:
            return "auto" if self._native() else None

        def _failover_tier(self) -> str | None:
            return self._standard_tier()

        def _flex_ineligible(self) -> bool:
            """Whether this call must not request flex — capability or health.

            Mirror of the Gemini transport's predicate: capability is permanent
            and model-scoped, health is a cool-off and provider-scoped.
            """
            return is_flex_unsupported(self.model_name) or flex_degraded(
                self.dispatch_provider
            )

        def _effective_tier(self, kwargs: dict[str, Any]) -> str | None:
            requested_tier = kwargs.get("service_tier")
            if isinstance(requested_tier, str):
                return requested_tier
            if self.service_tier == "flex" and self._flex_ineligible():
                return self._standard_tier()
            tier = self.service_tier
            return tier if isinstance(tier, str) else None

        def _payload_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
            if kwargs.get("service_tier") is not None:
                # From invocation or bound kwargs: refused before any send, as
                # at construction.
                self._refuse_tier_off_native()
            elif "service_tier" in kwargs and not self._native():
                # An explicit None is no tier: a compatible vendor gets no field.
                kwargs = {k: v for k, v in kwargs.items() if k != "service_tier"}
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
            if (
                isinstance(exc, BudgetExhausted)
                or self._effective_tier(kwargs) != "flex"
            ):
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
                note_flex_fallback(
                    self.dispatch_provider, reason="capacity", model=self.model_name
                )
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
                note_flex_fallback(
                    self.dispatch_provider, reason="latency", model=self.model_name
                )
                mark_flex_latency_fallback()
                return "auto"
            return None

        def _plan(self, kwargs: dict[str, Any], phase: str) -> _Dispatch:
            return _plan_dispatch(
                self,
                kwargs,
                provider=self.dispatch_provider,
                standard_tier=self._standard_tier(),
                model_name=self.model_name,
                phase=phase,
            )

        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            if (delegate := self._failover_delegate()) is not None:
                return delegate._generate(*args, **kwargs)
            kwargs = self._payload_kwargs(kwargs)
            dispatch = self._plan(kwargs, "initial")
            mark_tier_attempt(dispatch.tier)
            try:
                return _run_dispatch_sync(super()._generate, args, dispatch)
            except Exception as exc:
                retry_tier = self._flex_retry_tier(exc, dispatch.kwargs)
                if retry_tier is None:
                    raise
                retry = self._plan({**kwargs, "service_tier": retry_tier}, "reissue")
                mark_tier_attempt(retry.tier)
                return _run_dispatch_sync(super()._generate, args, retry)

        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            if (delegate := self._failover_delegate()) is not None:
                return await delegate._agenerate(*args, **kwargs)
            kwargs = self._payload_kwargs(kwargs)
            dispatch = self._plan(kwargs, "initial")
            mark_tier_attempt(dispatch.tier)
            try:
                return await _run_dispatch_async(super()._agenerate, args, dispatch)
            except Exception as exc:
                retry_tier = self._flex_retry_tier(exc, dispatch.kwargs)
                if retry_tier is None:
                    raise
                retry = self._plan({**kwargs, "service_tier": retry_tier}, "reissue")
                mark_tier_attempt(retry.tier)
                return await _run_dispatch_async(super()._agenerate, args, retry)

        def _stream(self, *args: Any, **kwargs: Any) -> Any:
            refuse_streaming(self)

        async def _astream(self, *args: Any, **kwargs: Any) -> Any:
            refuse_streaming(self)
            yield  # pragma: no cover - makes this an async generator

    return _FlexFallbackChatOpenAI


def _define_anthropic_cls() -> type[BaseChatModel]:
    chat_anthropic = _real_base("langchain_anthropic.chat_models", "ChatAnthropic")

    class _BudgetedChatAnthropic(chat_anthropic):  # type: ignore[valid-type,misc]
        """ChatAnthropic sized at dispatch.

        langchain-anthropic forwards a per-call ``timeout`` through
        ``_get_request_payload`` into ``messages.create(timeout=...)`` on both
        the sync and async paths. ``create()`` takes no retry count, so the
        client is built for one attempt. No service tiers and no model
        failover: Anthropic serves only the writer group.
        """

        dispatch_provider: str
        # Zero SDK retries; the parent's default is 2.
        max_retries: int = 0
        budget_timeout_class: str | None = None
        request_cap_override: float | None = Field(default=None, gt=0)
        disable_streaming: bool | Literal["tool_calling"] = True

        @model_validator(mode="after")
        def _validate_dispatch_provider(self) -> "_BudgetedChatAnthropic":
            self.dispatch_provider = _normalized_provider(self.dispatch_provider)
            _refuse_streaming_field(self)
            require_single_sdk_attempt(self, 0)
            return self

        def _effective_tier(self, kwargs: dict[str, Any]) -> str | None:
            return None

        def _plan(self, kwargs: dict[str, Any]) -> _Dispatch:
            return _plan_dispatch(
                self,
                kwargs,
                provider=self.dispatch_provider,
                standard_tier=None,
                model_name=self.model,
                phase="initial",
            )

        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            dispatch = self._plan(kwargs)
            mark_tier_attempt(dispatch.tier)
            return _run_dispatch_sync(super()._generate, args, dispatch)

        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            dispatch = self._plan(kwargs)
            mark_tier_attempt(dispatch.tier)
            return await _run_dispatch_async(super()._agenerate, args, dispatch)

        def _stream(self, *args: Any, **kwargs: Any) -> Any:
            refuse_streaming(self)

        async def _astream(self, *args: Any, **kwargs: Any) -> Any:
            refuse_streaming(self)
            yield  # pragma: no cover - makes this an async generator

    return _BudgetedChatAnthropic


def new_openai_chat(**kwargs: Any) -> BaseChatModel:
    """Construct an OpenAI or OpenAI-compatible client. ``dispatch_provider`` is required."""
    return deadline_aware_openai_cls()(**kwargs)


def new_anthropic_chat(**kwargs: Any) -> BaseChatModel:
    """Construct an Anthropic client. ``dispatch_provider`` is required."""
    return deadline_aware_anthropic_cls()(**kwargs)
