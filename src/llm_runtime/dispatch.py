"""Per-dispatch request sizing shared by every deadline-aware chat transport.

Every request -- flex, the standard re-issue, a call sent at standard because
flex is degraded or skipped, and the failover copy -- is sized here from the
invocation's absolute deadline (a ContextVar set by the runtime loop), never
from a timeout fixed on the instance at construction. Outside an application
invocation (a direct caller), a request is admitted and sized against the
enclosing deadline when one is bound (``enclosing_deadline``), under the
model's class policy; with none bound, only a flex request is sized (to the
flex ceiling) and a standard request keeps the client's constructed timeout.
Every request is one SDK attempt (``require_single_sdk_attempt``): the runtime
loop is the only retry owner.

Provider-neutral and SDK-free on purpose: the Google transport in
``src/llms.py`` and the OpenAI and Anthropic transports in
``src/llm_runtime/adapters/transports.py`` share it, and importing it must not
pull in any vendor SDK.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

import structlog

from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.failover import in_failover_attempt
from src.llm_runtime.timeouts import (
    BudgetExhausted,
    BudgetPolicy,
    DispatchScope,
    TimeoutClass,
    current_dispatch_scope,
    current_enclosing_deadline,
    policy_for_provider,
    request_timeout,
    resolve_budget_policy,
)
from src.service_tiers import is_flex_unsupported_error

logger = structlog.get_logger(__name__)


class StreamingNotQualifiedError(NotImplementedError):
    """A deadline-aware transport was asked to stream.

    Streaming would bypass the per-dispatch sizing in ``_generate``/``_agenerate``.
    No application code streams; this stays an error until a streaming path is
    sized at dispatch and qualified.
    """


def refuse_streaming(model: Any) -> NoReturn:
    raise StreamingNotQualifiedError(
        f"{type(model).__name__} does not stream: streaming bypasses the "
        "per-dispatch request deadline and is not qualified in this repository"
    )


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

    A flex request may queue silently for minutes; its per-dispatch request
    timeout (a share of the invocation allowance in quick mode — see
    ``request_timeout`` in ``src/llm_runtime/timeouts.py``) surfaces that queue
    as a raised timeout/deadline error, which we treat like a capacity signal
    and fall back to the standard tier. Only errors raised *inside* the SDK call
    reach here; the outer ``run_with_hard_timeout`` fires in ``runtime.py``,
    never in the transport, so this cannot swallow the outer wall-clock cap.
    A ``BudgetExhausted`` refusal is a TimeoutError but never a queue signal;
    the transports reject it before asking this predicate.
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


@dataclass
class _Dispatch:
    """One planned transport request and its telemetry record."""

    tier: str | None
    kwargs: dict[str, Any]
    record: dict[str, Any]
    scope: DispatchScope | None


def _dispatch_outcome(exc: BaseException | None) -> str:
    """Secret-free outcome label for one dispatch record."""
    if exc is None:
        return "ok"
    if isinstance(exc, BudgetExhausted):
        return "budget_exhausted"
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    if is_flex_unsupported_error(exc):
        return "tier_unsupported"
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return "rate_limited"
    if _is_flex_latency_timeout(exc):
        return "timeout"
    if (isinstance(code, int) and code >= 500) or _is_flex_capacity_error(exc):
        return "server_error"
    return "error"


def require_single_sdk_attempt(model: Any, single_attempt: int) -> None:
    """Refuse a transport built to retry inside the SDK.

    Retries have one owner, the runtime loop (``invoke_with_rate_limit_handling``),
    which re-admits each one against the enclosing deadline. An SDK retry would
    re-send a request with the timeout sized for the first one, running past
    the deadline it was sized from, and multiply the loop's attempts.
    ``single_attempt`` is the vendor's spelling of one request: google-genai
    counts total attempts (where ``0`` means its own default), OpenAI and
    Anthropic count retries after the first request.
    """
    configured = getattr(model, "max_retries", None)
    if configured != single_attempt:
        raise ValueError(
            f"{type(model).__name__} sends one SDK attempt per request "
            f"(max_retries={single_attempt}), not max_retries={configured!r}: "
            "retries belong to the runtime loop"
        )


def _construction_timeout(model: Any) -> float | None:
    # ChatOpenAI and Gemini name it ``request_timeout``/``timeout``;
    # ChatAnthropic stores the ``timeout`` alias as ``default_request_timeout``.
    for attr in ("timeout", "request_timeout", "default_request_timeout"):
        raw = getattr(model, attr, None)
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            return float(raw)
    return None


REQUEST_CAP_OVERRIDE_FIELD = "request_cap_override"


def attach_request_cap_override(model: Any, seconds: float | None) -> Any:
    """Stamp a seat policy's explicit client timeout on a constructed transport.

    The two construction choke points (``SeatModelFactory.build`` and
    ``build_legacy_model``) call this with ``client_timeout_seconds``, so every
    dispatch of the model -- scoped or direct, a ``model_copy`` and the
    failover copy included -- resolves the override through
    ``policy_for_provider``. ``None`` stamps nothing. A model that cannot
    carry an override fails loudly rather than silently sending another
    ceiling.
    """
    if seconds is None:
        return model
    if not hasattr(model, REQUEST_CAP_OVERRIDE_FIELD):
        raise TypeError(
            f"{type(model).__name__} cannot carry a {REQUEST_CAP_OVERRIDE_FIELD}; "
            "only the deadline-aware transports size requests at dispatch"
        )
    setattr(model, REQUEST_CAP_OVERRIDE_FIELD, float(seconds))
    return model


def _dispatch_budget(
    model: Any, provider: str
) -> tuple[BudgetPolicy, float | None, DispatchScope | None]:
    # The request ceiling's precedence (policy_for_provider): this model's
    # explicit override, the canonical seat policy, the receiving vendor's.
    override = getattr(model, REQUEST_CAP_OVERRIDE_FIELD, None)
    scope = current_dispatch_scope()
    if scope is not None:
        # The deadline is the invocation's; the request ceiling is resolved for
        # the vendor that actually receives this request (a delegate could be
        # another one).
        policy = policy_for_provider(
            scope.policy, provider, request_cap_override=override
        )
        return policy, scope.deadline, scope
    # A direct caller (no invocation scope). Its requests are still admitted
    # and sized against an enclosing deadline when one is bound, under the
    # model's own class policy; with none bound, nothing changes.
    raw_class = getattr(model, "budget_timeout_class", None)
    timeout_class = (
        TimeoutClass(raw_class)
        if isinstance(raw_class, str)
        else TimeoutClass.AUXILIARY
    )
    return (
        resolve_budget_policy(
            timeout_class, provider=provider, request_cap_override=override
        ),
        current_enclosing_deadline(),
        None,
    )


def _plan_dispatch(
    model: Any,
    kwargs: dict[str, Any],
    *,
    provider: str,
    standard_tier: str | None,
    model_name: str,
    phase: str,
) -> _Dispatch:
    """Size one request from the deadline, or refuse it with ``BudgetExhausted``.

    ``provider`` is the vendor that receives the request, passed explicitly by
    the transport that owns it (never inferred): it selects the request ceiling.
    ``standard_tier`` is the vendor's name for "not flex", or ``None`` for a
    vendor with no service tiers. A per-call ``max_retries`` is refused: the
    transport sends one SDK attempt and the runtime loop owns retries.
    """
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("a dispatch needs the receiving provider, explicitly")
    if kwargs.get("stream"):
        # ``disable_streaming`` routes a stream request here instead of to
        # ``_stream``; sending it on would open a streaming request that the
        # non-streaming result parser cannot read.
        refuse_streaming(model)
    if "max_retries" in kwargs:
        raise ValueError(
            "a dispatch sends one SDK attempt; retries belong to the runtime loop"
        )
    policy, deadline, scope = _dispatch_budget(model, provider)
    tier = model._effective_tier(kwargs)
    record: dict[str, Any] = {
        "phase": "failover" if in_failover_attempt() else phase,
        "tier": tier or standard_tier,
        "model": model_name,
        "timeout_s": None,
        "remaining_s": (
            round(deadline - llm_timeouts.now(), 3) if deadline is not None else None
        ),
        "outcome": "in_flight",
        "elapsed_s": None,
    }
    if scope is not None:
        scope.record(record)
    call_kwargs = dict(kwargs)
    try:
        if tier == "flex":
            timeout = request_timeout(policy, "flex", deadline=deadline)
            if timeout < policy.min_useful_attempt:
                # A flex request this short cannot usefully queue; spend the
                # allowance at standard instead (not a flex-health event).
                tier = standard_tier
                if standard_tier is not None:
                    call_kwargs["service_tier"] = standard_tier
                else:
                    call_kwargs.pop("service_tier", None)
                record["tier"] = standard_tier
                record["flex_skipped"] = True
        if tier != "flex":
            timeout = request_timeout(policy, tier, deadline=deadline)
    except BudgetExhausted:
        record["outcome"] = "budget_exhausted"
        record["elapsed_s"] = 0.0
        logger.warning(
            "llm_dispatch_refused",
            model=model_name,
            tier=record["tier"],
            phase=record["phase"],
            remaining_seconds=record["remaining_s"],
            min_useful_attempt_seconds=round(policy.min_useful_attempt, 3),
        )
        raise
    if scope is not None or tier == "flex" or deadline is not None:
        # Outside an invocation, a request under an enclosing deadline is sized
        # from it, as a flex request is from its ceiling.
        call_kwargs["timeout"] = timeout
        record["timeout_s"] = round(timeout, 3)
    else:
        constructed = _construction_timeout(model)
        record["timeout_s"] = round(constructed, 3) if constructed else None
    return _Dispatch(tier=tier, kwargs=call_kwargs, record=record, scope=scope)


def _finish_dispatch(
    dispatch: _Dispatch, started: float, exc: BaseException | None
) -> None:
    dispatch.record["outcome"] = _dispatch_outcome(exc)
    dispatch.record["elapsed_s"] = round(llm_timeouts.now() - started, 3)


async def _run_dispatch_async(
    call: Callable[..., Any], args: tuple[Any, ...], dispatch: _Dispatch
) -> Any:
    started = llm_timeouts.now()
    try:
        result = await call(*args, **dispatch.kwargs)
    except BaseException as exc:
        _finish_dispatch(dispatch, started, exc)
        raise
    _finish_dispatch(dispatch, started, None)
    return result


def _run_dispatch_sync(
    call: Callable[..., Any], args: tuple[Any, ...], dispatch: _Dispatch
) -> Any:
    started = llm_timeouts.now()
    try:
        result = call(*args, **dispatch.kwargs)
    except BaseException as exc:
        _finish_dispatch(dispatch, started, exc)
        raise
    _finish_dispatch(dispatch, started, None)
    return result
