"""One budget policy per LLM dispatch, and the absolute deadline it is sized from.

Every request deadline sent to a provider is derived at *dispatch* time from an
absolute monotonic deadline owned by the invocation boundary
(``invoke_with_rate_limit_handling`` in ``src/agents/runtime.py``). Nothing here
is fixed on a model instance at construction, because a construction-time
timeout survives every later tier change: Google enforces the client timeout
server-side (``X-Server-Timeout``), so a 24 s flex-attempt timeout baked into a
flex-built client was also sent on the standard re-issue, on calls sent at
standard after flex degradation, and on the model failover copy (October 2026
quick Portfolio Manager 504s). See ``docs/LLM_PROVIDERS.md``.

Three pieces:

- ``TimeoutClass`` -- carried on every registered ``SeatSpec``; replaces the old
  display-name string matching that chose a seat's per-call cap.
- ``BudgetPolicy`` -- immutable, resolved centrally from (class, run mode,
  provider). All numbers come from ``Settings``/``RuntimeConfig``.
- A ContextVar ``DispatchScope`` -- the invocation's absolute deadline plus the
  per-dispatch telemetry list. It is never stored on the model object, so
  concurrent calls on one model stay isolated, and the failover copy reads the
  same deadline rather than a fresh clock.

Retries have one owner: the runtime loop, whose budget is ``llm_attempts``
(total attempts). Every client sends exactly one SDK attempt per request -- a
property of the transport classes, never a construction argument -- so an SDK
retry can neither multiply the loop's attempts nor re-send a request past the
deadline it was sized from.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from src.config import resolve_settings

if TYPE_CHECKING:
    from src.llm_runtime.seats import SeatSpec
    from src.runtime_config import RuntimeConfig


class TimeoutClass(StrEnum):
    """Which per-call budget family a seat belongs to."""

    # Ordinary graph seats: the flat per-call cap.
    STANDARD = "standard"
    # Gate-critical seats (Senior Fundamentals, Portfolio Manager): the largest,
    # most rule-dense prompts, given a larger --quick cap.
    APEX = "apex"
    # The review plane (Consultant, Auditor, Article Editor): a vendor-latency
    # allowance, not a criticality one.
    CROSS_CHECK = "cross_check"
    # The APAC regional specialist, whose node total is its enclosing deadline.
    REGIONAL = "regional"
    # The writer and its fallbacks. Invoked through the runtime loop (which owns
    # their retries) inside the article workflow's own deadline; quick-invariant,
    # like the writer's model selection, with ``api_timeout`` per attempt.
    EDITORIAL = "editorial"
    # Operational helpers (macro context, injection judge, retrospective, health
    # check, signal processor, semantic judge). All but the health check run
    # under the runtime loop.
    OPERATIONAL = "operational"
    # Reserved for callers that are not registered seats; never valid on a SeatSpec.
    AUXILIARY = "auxiliary"


class BudgetExhausted(TimeoutError):
    """Too little of the invocation allowance remains for a useful request.

    Raised *before* any transport call. A ``TimeoutError`` so every existing
    timeout handler treats it as one, but never a flex-latency signal and never
    evidence of provider health.
    """

    def __init__(self, remaining: float, minimum: float, *, tier: str | None = None):
        self.remaining = remaining
        self.minimum = minimum
        self.tier = tier
        super().__init__(
            f"llm budget exhausted: {remaining:.1f}s left < "
            f"{minimum:.1f}s minimum useful attempt"
        )


@dataclass(frozen=True)
class BudgetPolicy:
    """Per-(class, mode, provider) call budget. Seconds throughout."""

    timeout_class: TimeoutClass
    quick_mode: bool
    provider: str | None
    # Wall-clock allowance for one application attempt (run_with_hard_timeout).
    invocation_cap: float
    # Share of the invocation cap a flex attempt may queue for (quick mode).
    flex_fraction: float
    # Ceiling for one flex request: quick = cap x fraction; full = the flex floor.
    flex_request_cap: float
    # Ceiling for one standard request.
    request_cap: float
    # Below this remaining allowance a request is refused, not sent.
    min_useful_attempt: float
    # Kept back so the request's own deadline fires before the hard cap.
    margin: float
    # Total application attempts for transient failures (``llm_attempts``).
    attempts: int
    # A seat's explicit client timeout (``client_timeout_seconds``), kept so a
    # re-resolution for another provider still honours it.
    request_cap_override: float | None = None


@dataclass
class DispatchScope:
    """The active invocation's absolute deadline and its dispatch records."""

    policy: BudgetPolicy
    deadline: float
    dispatches: list[dict[str, Any]] = field(default_factory=list)

    def record(self, entry: dict[str, Any]) -> None:
        self.dispatches.append(entry)


_DISPATCH_SCOPE: ContextVar[DispatchScope | None] = ContextVar(
    "llm_dispatch_scope", default=None
)
_ENCLOSING_DEADLINE: ContextVar[float | None] = ContextVar(
    "llm_enclosing_deadline", default=None
)
# The deadline operational work inside the tool call in progress must meet, so
# a call made from inside a tool -- a web search, a market-data fetch, an MCP
# spot check routed back through the tool service -- sizes against what the
# runner has left instead of a fresh allowance. Deliberately separate from the
# LLM enclosing deadline: binding that around a tool call would also cap the
# injection judge in the after() hooks, and the judge fails *open* when
# admission refuses it, where the tool scope's own timeout fails closed. Only
# ``ToolExecutionService`` binds it (``tool_call_deadline``).
_TOOL_CALL_DEADLINE: ContextVar[float | None] = ContextVar(
    "tool_call_deadline", default=None
)


def now() -> float:
    """The monotonic clock every deadline here is measured on (patchable)."""
    return time.monotonic()


def current_dispatch_scope() -> DispatchScope | None:
    return _DISPATCH_SCOPE.get()


def current_enclosing_deadline() -> float | None:
    """The LLM enclosing deadline alone: for LLM admission, never operational work.

    Operational call sites read ``operational_deadline`` (through
    ``require_allowance`` or ``clamp_to_operational_deadline``), which also
    sees the tool runner they may run under. An architecture test
    (``tests/architecture/test_deadline_readers.py``) keeps this reader to the
    LLM dispatch modules.
    """
    return _ENCLOSING_DEADLINE.get()


def deadline_margin(cap: float, cfg: Any = None) -> float:
    """Seconds an inner deadline is kept back from a hard cap of ``cap`` seconds.

    ``DEADLINE_MARGIN_FRACTION`` of the cap, at most
    ``LLM_DISPATCH_MARGIN_SECONDS``. Work sized inside a hard cap gets its own
    deadline this much earlier, so its own timeout -- and the typed result it
    produces -- fires before the cap's cancellation. Equal deadlines race:
    asyncio orders timers due at the same instant arbitrarily. Always below a
    positive cap.
    """
    settings = resolve_settings(cfg)
    return min(
        float(settings.llm_dispatch_margin_seconds),
        float(cap) * float(settings.deadline_margin_fraction),
    )


def min_useful_seconds(allowance: float, cfg: Any = None) -> float:
    """The least time worth starting one attempt for, from its configured allowance.

    ``MIN_USEFUL_ATTEMPT_FRACTION`` of ``allowance``: the one rule for the LLM,
    embedding and MCP planes. Pass the plane's *configured* per-attempt
    allowance for one attempt (for the LLM, the per-request ceiling or the
    invocation cap if shorter -- never the cap of a whole multi-attempt
    invocation), never a remainder already clamped to a deadline -- the
    minimum must not shrink with the time left. Always below a positive allowance.
    """
    return float(allowance) * float(resolve_settings(cfg).min_useful_attempt_fraction)


def operational_deadline() -> float | None:
    """The absolute deadline operational work must meet, or None.

    The earlier of the caller's enclosing deadline and the tool call this work
    runs under. Read-only: for a retry loop that needs the instant itself, to
    gate a backoff with ``wait_allowance``.
    """
    bound = [
        deadline
        for deadline in (_ENCLOSING_DEADLINE.get(), _TOOL_CALL_DEADLINE.get())
        if deadline is not None
    ]
    return min(bound) if bound else None


def operational_time_left() -> float:
    """Seconds left before ``operational_deadline``; ``inf`` when none is bound."""
    deadline = operational_deadline()
    return math.inf if deadline is None else deadline - now()


def clamp_to_operational_deadline(default_seconds: float) -> float:
    """``min(default_seconds, operational_time_left())``.

    With nothing bound the default stands. A result ``<= 0`` means the
    allowance is already spent, and the caller must fail as a timeout without
    starting work. Operational scopes read these deadlines but never bind the
    LLM one (see ``_TOOL_CALL_DEADLINE``).
    """
    return min(float(default_seconds), operational_time_left())


def require_allowance(default_seconds: float, *, label: str) -> float:
    """``clamp_to_operational_deadline``, raising ``TimeoutError`` once it is spent.

    For a call site whose existing timeout handling is the fallback: a spent
    allowance takes the same path as the call's own timeout, without starting
    the call (``run_with_hard_timeout`` rejects a non-positive bound).
    """
    seconds = clamp_to_operational_deadline(default_seconds)
    if seconds <= 0:
        raise TimeoutError(f"{label!r}: deadline already passed")
    return seconds


@contextmanager
def tool_call_deadline(seconds: float) -> Iterator[float]:
    """Bind the operational deadline for a tool scope hard-capped at ``seconds``.

    The bound instant is ``now + seconds - deadline_margin(seconds)``: work
    inside sizes against it, so its own timeout fires before the scope's hard
    cap does. Nested bindings can only shorten it. A task created inside the
    block (the hard-timeout wrapper's inner task) copies the binding, so every
    tool call it makes sees the outer deadline. Only ``ToolExecutionService``
    binds it.
    """
    deadline = now() + float(seconds) - deadline_margin(seconds)
    outer = _TOOL_CALL_DEADLINE.get()
    bound = deadline if outer is None else min(deadline, outer)
    token = _TOOL_CALL_DEADLINE.set(bound)
    try:
        yield bound
    finally:
        _TOOL_CALL_DEADLINE.reset(token)


@contextmanager
def invocation_scope(policy: BudgetPolicy, deadline: float) -> Iterator[DispatchScope]:
    """Bind one application attempt's deadline for the dispatches made inside it.

    Reset in ``finally`` so a cancelled or failed dispatch cannot leak its
    deadline into the next call. A task created inside the block (the hard-cap
    wrapper's inner task) holds its own context copy, so an orphaned request
    keeps the deadline it was admitted under and cannot disturb a later one.
    """
    scope = DispatchScope(policy=policy, deadline=deadline)
    token = _DISPATCH_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _DISPATCH_SCOPE.reset(token)


@contextmanager
def enclosing_deadline(seconds: float | None) -> Iterator[float | None]:
    """Cap every LLM invocation inside the block by an absolute node deadline.

    Nested scopes can only shorten the deadline: a correction or recovery call
    gets its own invocation allowance but can never renew the node's.
    """
    if seconds is None:
        with enclosing_deadline_at(None) as outer:
            yield outer
        return
    with enclosing_deadline_at(now() + float(seconds)) as deadline:
        yield deadline


@contextmanager
def enclosing_deadline_at(deadline: float | None) -> Iterator[float | None]:
    """``enclosing_deadline`` for an absolute monotonic deadline already computed.

    The ticker run deadline (``src/llm_runtime/run_deadline.py``) arrives as an
    absolute instant, not a duration; binding it here makes every node scope
    inside it ``min(node, run)``. ``None`` binds nothing.
    """
    outer = _ENCLOSING_DEADLINE.get()
    if deadline is None:
        yield outer
        return
    bound = float(deadline) if outer is None else min(float(deadline), outer)
    token = _ENCLOSING_DEADLINE.set(bound)
    try:
        yield bound
    finally:
        _ENCLOSING_DEADLINE.reset(token)


def _runtime(settings: Any, runtime_config: RuntimeConfig | None) -> Any:
    if runtime_config is not None:
        return runtime_config
    from src.runtime_config import get_runtime_config

    return get_runtime_config(settings)


def openai_request_cap_seconds(cfg: Any = None) -> float:
    """The standard per-request ceiling on the OpenAI plane, read at the call.

    ``OPENAI_REQUEST_CAP_SECONDS`` (a ``Settings`` field). Was the module
    constant of the same name, which no override could reach.
    """
    return float(resolve_settings(cfg).openai_request_cap_seconds)


def apac_client_timeout_seconds(cfg: Any = None) -> float:
    """The APAC specialist's request ceiling: its node total.

    ``apac_node_total_timeout_seconds`` is the enclosing deadline every APAC
    request is sized from, so no single request can usefully run longer. The
    client is built with it and the in-scope ceiling (``REGIONAL`` class on a
    compatible vendor) is the same value, on the legacy and canonical paths.
    Before October 2026 the client used a separate 240 s constant while the
    in-scope ceiling was the generic compatible-vendor 300 s.
    """
    return float(resolve_settings(cfg).apac_node_total_timeout_seconds)


def _editorial_invocation_cap(settings: Any) -> float:
    """The writer's per-attempt cap, in both run modes.

    The writer is quick-invariant (its model ignores ``--quick``) and writes a
    long article in one request, so neither the flat quick cap nor the graph's
    per-call cap fits it; ``api_timeout`` is the per-request ceiling its clients
    have always been built with. The article workflow's own deadline
    (``article_workflow_timeout_seconds``) bounds the whole chain.
    """
    return float(settings.api_timeout)


def quick_invocation_cap(timeout_class: TimeoutClass, cfg: Any = None) -> float:
    """The ``--quick`` per-attempt cap for a class, before any flex floor."""
    settings = resolve_settings(cfg)
    if timeout_class is TimeoutClass.EDITORIAL:
        return _editorial_invocation_cap(settings)
    if timeout_class is TimeoutClass.APEX:
        return float(settings.apex_quick_llm_call_hard_timeout_seconds)
    if timeout_class is TimeoutClass.CROSS_CHECK:
        return float(settings.cross_check_quick_llm_call_hard_timeout_seconds)
    return float(settings.quick_llm_call_hard_timeout_seconds)


def _standard_request_cap(
    timeout_class: TimeoutClass, provider: str | None, settings: Any, quick: bool
) -> float:
    """The provider's standard per-request ceiling: the value clients were built with."""
    family = (provider or "").strip().lower()
    api_timeout = float(settings.api_timeout)
    if family == "openai":
        return openai_request_cap_seconds(settings)
    if family == "anthropic":
        return api_timeout
    if family in ("", "google", "gemini"):
        if quick and timeout_class not in (TimeoutClass.APEX, TimeoutClass.EDITORIAL):
            return min(
                api_timeout,
                float(settings.quick_llm_api_timeout_seconds),
            )
        return api_timeout
    if timeout_class is TimeoutClass.REGIONAL:
        return apac_client_timeout_seconds(settings)
    return float(settings.openai_compatible_client_timeout_seconds)


def resolve_budget_policy(
    timeout_class: TimeoutClass,
    *,
    provider: str | None = None,
    cfg: Any = None,
    runtime_config: RuntimeConfig | None = None,
    request_cap_override: float | None = None,
) -> BudgetPolicy:
    """Resolve the budget for one (class, run mode, provider) from configuration.

    Quick mode has one source: the run-scoped ``RuntimeConfig``.
    """
    from src.service_tiers import flex_floor_seconds, floor_llm_hard_timeout

    settings = resolve_settings(cfg)
    runtime = _runtime(settings, runtime_config)
    quick = bool(runtime.quick_mode_active)
    if timeout_class is TimeoutClass.EDITORIAL:
        cap = _editorial_invocation_cap(settings)
    elif quick:
        cap = quick_invocation_cap(timeout_class, settings)
    else:
        cap = float(runtime.llm_call_hard_timeout_seconds)
    request_cap = (
        float(request_cap_override)
        if request_cap_override is not None
        else _standard_request_cap(timeout_class, provider, settings, quick)
    )
    # The minimum is a share of one attempt's allowance: the per-request
    # ceiling, or the invocation cap when that is shorter. The flex floor below
    # lengthens the invocation, not the least attempt worth sending.
    min_useful = min_useful_seconds(min(cap, request_cap), settings)
    # Full-mode flex: the cap must outlast a queued flex request plus its
    # standard re-issue. A no-op in quick mode and when flex is off.
    cap = floor_llm_hard_timeout(
        cap,
        provider=provider,
        cfg=settings,
        label=f"invoke_hard_timeout:{provider}",
    )
    fraction = float(settings.flex_quick_attempt_fraction)
    flex_request_cap = cap * fraction if quick else float(flex_floor_seconds(settings))
    margin = deadline_margin(cap, settings)
    attempts = max(1, int(runtime.llm_attempts))
    return BudgetPolicy(
        timeout_class=timeout_class,
        quick_mode=quick,
        provider=provider,
        invocation_cap=cap,
        flex_fraction=fraction,
        flex_request_cap=flex_request_cap,
        request_cap=request_cap,
        min_useful_attempt=min_useful,
        margin=margin,
        attempts=attempts,
        request_cap_override=(
            float(request_cap_override) if request_cap_override is not None else None
        ),
    )


def _provider_family(provider: str | None) -> str:
    """The request-ceiling family a provider label belongs to."""
    family = (provider or "").strip().lower()
    return "google" if family in ("", "google", "gemini") else family


def policy_for_provider(
    policy: BudgetPolicy,
    provider: str | None,
    *,
    request_cap_override: float | None = None,
    cfg: Any = None,
) -> BudgetPolicy:
    """``policy`` with its request ceiling resolved for the request being sent.

    The one precedence chain for a request's ceiling: the sending model's
    explicit ``request_cap_override``, then the policy's own (the canonical
    seat's client timeout), then the ceiling of ``provider``, the vendor that
    actually receives the request -- a delegate on another vendor must not
    send the original vendor's ceiling. An override replaces the ceiling, so
    it may exceed the vendor's; the deadline bounds every request anyway
    (``request_timeout``).

    The invocation's allowance (cap, margin, attempts) belongs to the
    invocation and is kept, as is the deadline it was opened with. The
    minimum useful attempt can only fall, to the share of a shorter ceiling
    (``min_useful_seconds``): a request ceiling shorter than the minimum
    would refuse every request.
    """
    override = (
        float(request_cap_override)
        if request_cap_override is not None
        else policy.request_cap_override
    )
    same_vendor = _provider_family(policy.provider) == _provider_family(provider)
    if same_vendor and override == policy.request_cap_override:
        return policy
    if override is not None:
        request_cap = override
    else:
        request_cap = _standard_request_cap(
            policy.timeout_class, provider, resolve_settings(cfg), policy.quick_mode
        )
    return replace(
        policy,
        provider=provider,
        request_cap=request_cap,
        request_cap_override=override,
        min_useful_attempt=min(
            policy.min_useful_attempt, min_useful_seconds(request_cap, cfg)
        ),
    )


def request_timeout(
    policy: BudgetPolicy,
    tier: str | None,
    *,
    deadline: float | None,
    clock: Any = None,
) -> float:
    """Size one request from the absolute deadline, or refuse it.

    ``deadline=None`` (a direct caller, outside any application invocation)
    returns the tier's ceiling. Remaining time exactly equal to the minimum
    useful attempt is admitted.
    """
    cap = policy.flex_request_cap if tier == "flex" else policy.request_cap
    if deadline is None:
        return cap
    remaining = deadline - (clock or now)() - policy.margin
    if remaining < policy.min_useful_attempt:
        raise BudgetExhausted(remaining, policy.min_useful_attempt, tier=tier)
    return min(cap, remaining)


def retry_wait_allowance(
    policy: BudgetPolicy,
    enclosing: float | None,
    wait: float,
    *,
    rate_limited: bool,
    clock: Any = None,
) -> float | None:
    """How long the retry loop may sleep before another attempt, or None to stop.

    Admission is re-checked against the *enclosing* deadline: an application
    retry gets a fresh invocation allowance, but never past the node or ticker
    deadline. A transient backoff is clamped so a useful attempt still fits; a
    rate-limit backoff is not clamped -- sleeping less than the provider asked
    is a wasted request -- so it is refused instead.
    """
    return wait_allowance(
        enclosing,
        wait,
        minimum=policy.min_useful_attempt,
        margin=policy.margin,
        rate_limited=rate_limited,
        clock=clock,
    )


def wait_allowance(
    enclosing: float | None,
    wait: float,
    *,
    minimum: float,
    margin: float = 0.0,
    rate_limited: bool = False,
    clock: Any = None,
) -> float | None:
    """``retry_wait_allowance`` for a retry loop with no ``BudgetPolicy``.

    The same rule, with the minimum useful attempt and margin passed directly:
    sleep at most until ``minimum + margin`` is left before ``enclosing``, or
    return None when even that does not fit. The memory embedding retry and the
    MCP tool-call retry (against the tool-plane deadline) use it.
    """
    if enclosing is None:
        return wait
    available = enclosing - (clock or now)() - margin - minimum
    if available < 0:
        return None
    if rate_limited and available < wait:
        return None
    return max(0.0, min(wait, available))


def _seat_index() -> dict[str, SeatSpec]:
    """Display name -> seat, for the facts an invocation's budget depends on.

    Several seats may share a display name (the two writer fallbacks); they
    must then agree on everything the budget reads, or the name is ambiguous.
    """
    from src.llm_runtime.seats import SEATS

    index: dict[str, SeatSpec] = {}
    for spec in SEATS.values():
        existing = index.get(spec.callback_name)
        if existing is not None:
            if existing.timeout_class is not spec.timeout_class:
                raise ValueError(
                    f"display name {spec.callback_name!r} maps to two timeout classes"
                )
            if (
                existing.execution_policy.client_timeout_seconds
                != spec.execution_policy.client_timeout_seconds
            ):
                raise ValueError(
                    f"display name {spec.callback_name!r} maps to two client timeouts"
                )
            continue
        index[spec.callback_name] = spec
    return index


def seat_for_agent(name: str | None) -> SeatSpec | None:
    """The registered seat for a (possibly decorated) agent name, if any."""
    if not name:
        return None
    from src.token_tracker import canonical_display_name

    return _seat_index().get(canonical_display_name(name))


def timeout_class_for_agent(name: str | None) -> TimeoutClass | None:
    """The registered seat's class for a (possibly decorated) agent name, if any."""
    seat = seat_for_agent(name)
    return seat.timeout_class if seat is not None else None


def policy_for_seat(
    seat: SeatSpec,
    *,
    provider: str | None = None,
    cfg: Any = None,
    runtime_config: RuntimeConfig | None = None,
    timeout_class: TimeoutClass | None = None,
) -> BudgetPolicy:
    """The budget for a registered seat, honouring its explicit client timeout.

    ``timeout_class`` lets a caller choose a different budget family for one
    invocation; the seat's client timeout is a property of its client and
    applies either way.
    """
    return resolve_budget_policy(
        timeout_class or seat.timeout_class,
        provider=provider,
        cfg=cfg,
        runtime_config=runtime_config,
        request_cap_override=seat.execution_policy.client_timeout_seconds,
    )
