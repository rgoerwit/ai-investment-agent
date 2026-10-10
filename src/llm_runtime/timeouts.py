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

Attempt counts have one meaning: **total** attempts. ``llm_attempts`` is the
application retry budget; a dispatch made under an application-owned
invocation sends exactly one SDK attempt.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

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
    # Operational helpers (macro context, inspectors, retrospective, health check,
    # semantic judge). Mixed: some run under the runtime loop, some directly, so
    # a seat that the loop owns says so with ``transport_attempts=1``.
    OPERATIONAL = "operational"
    # Reserved for callers that are not registered seats; never valid on a SeatSpec.
    AUXILIARY = "auxiliary"

    @property
    def retries_owned_by_runtime(self) -> bool:
        """Whether this class's calls go through the application retry loop.

        Used only where the SDK fixes its retry count at client construction
        (OpenAI, Anthropic, OpenAI-compatible): those clients are built with one
        attempt when the runtime loop owns retries, and keep a small SDK budget
        for classes whose callers still invoke the model directly.
        """
        return self in _RUNTIME_OWNED_CLASSES


_RUNTIME_OWNED_CLASSES = frozenset(
    {
        TimeoutClass.STANDARD,
        TimeoutClass.APEX,
        TimeoutClass.CROSS_CHECK,
        TimeoutClass.REGIONAL,
        TimeoutClass.EDITORIAL,
    }
)


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
    # SDK attempts per dispatch under an application-owned invocation.
    transport_attempts: int = 1
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


def now() -> float:
    """The monotonic clock every deadline here is measured on (patchable)."""
    return time.monotonic()


def current_dispatch_scope() -> DispatchScope | None:
    return _DISPATCH_SCOPE.get()


def current_enclosing_deadline() -> float | None:
    return _ENCLOSING_DEADLINE.get()


def clamp_to_enclosing_deadline(default_seconds: float) -> float:
    """``min(default_seconds, remaining)`` under the caller's enclosing deadline.

    The enclosing deadline is the absolute node deadline bound by
    ``enclosing_deadline``; with none bound the default stands. A result
    ``<= 0`` means the caller's allowance is already spent, and the caller must
    fail as a timeout without starting work.

    Operational scopes (tool calls, market-data fetches) read this deadline but
    never bind it: binding it around such a call would also cap any LLM
    inspection judge inside it, and the judge fails *open* when refused, where
    the scope's own timeout fails closed.
    """
    deadline = current_enclosing_deadline()
    if deadline is None:
        return float(default_seconds)
    return min(float(default_seconds), deadline - now())


def require_allowance(default_seconds: float, *, label: str) -> float:
    """``clamp_to_enclosing_deadline``, raising ``TimeoutError`` once it is spent.

    For a call site whose existing timeout handling is the fallback: a spent
    caller allowance takes the same path as the call's own timeout, without
    starting the call (``run_with_hard_timeout`` rejects a non-positive bound).
    """
    seconds = clamp_to_enclosing_deadline(default_seconds)
    if seconds <= 0:
        raise TimeoutError(f"{label!r}: enclosing deadline already passed")
    return seconds


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


def _settings(cfg: Any) -> Any:
    if cfg is not None:
        return cfg
    import src.config as config_module

    return config_module.config


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
    return float(_settings(cfg).openai_request_cap_seconds)


def apac_client_timeout_seconds(cfg: Any = None) -> float:
    """The APAC specialist's request ceiling: its node total.

    ``apac_node_total_timeout_seconds`` is the enclosing deadline every APAC
    request is sized from, so no single request can usefully run longer. The
    client is built with it and the in-scope ceiling (``REGIONAL`` class on a
    compatible vendor) is the same value, on the legacy and canonical paths.
    Before October 2026 the client used a separate 240 s constant while the
    in-scope ceiling was the generic compatible-vendor 300 s.
    """
    return float(_settings(cfg).apac_node_total_timeout_seconds)


def direct_caller_transport_attempts(cfg: Any = None) -> int:
    """Total SDK attempts for a client whose callers still invoke it directly.

    The run-scoped ``llm_attempts`` (``LLM_ATTEMPTS``; 0 behaves as 1), on every
    vendor. Was a fixed 3 on the OpenAI-family and Anthropic paths, so those
    clients made one more attempt than the configured default of 2.
    """
    settings = _settings(cfg)
    return max(1, int(_runtime(settings, None).llm_attempts))


def _editorial_invocation_cap(settings: Any) -> float:
    """The writer's per-attempt cap, in both run modes.

    The writer is quick-invariant (its model ignores ``--quick``) and writes a
    long article in one request, so neither the flat quick cap nor the graph's
    per-call cap fits it; ``api_timeout`` is the per-request ceiling its clients
    have always been built with. The article workflow's own deadline
    (``article_workflow_timeout_seconds``) bounds the whole chain.
    """
    return float(getattr(settings, "api_timeout", 300))


def quick_invocation_cap(timeout_class: TimeoutClass, cfg: Any = None) -> float:
    """The ``--quick`` per-attempt cap for a class, before any flex floor."""
    settings = _settings(cfg)
    if timeout_class is TimeoutClass.EDITORIAL:
        return _editorial_invocation_cap(settings)
    if timeout_class is TimeoutClass.APEX:
        return float(
            getattr(settings, "apex_quick_llm_call_hard_timeout_seconds", 180.0)
        )
    if timeout_class is TimeoutClass.CROSS_CHECK:
        return float(
            getattr(settings, "cross_check_quick_llm_call_hard_timeout_seconds", 180.0)
        )
    return float(getattr(settings, "quick_llm_call_hard_timeout_seconds", 60.0))


def _standard_request_cap(
    timeout_class: TimeoutClass, provider: str | None, settings: Any, quick: bool
) -> float:
    """The provider's standard per-request ceiling: the value clients were built with."""
    family = (provider or "").strip().lower()
    api_timeout = float(getattr(settings, "api_timeout", 300))
    if family == "openai":
        return openai_request_cap_seconds(settings)
    if family == "anthropic":
        return api_timeout
    if family in ("", "google", "gemini"):
        if quick and timeout_class not in (TimeoutClass.APEX, TimeoutClass.EDITORIAL):
            return min(
                api_timeout,
                float(getattr(settings, "quick_llm_api_timeout_seconds", 120)),
            )
        return api_timeout
    if timeout_class is TimeoutClass.REGIONAL:
        return apac_client_timeout_seconds(settings)
    return float(getattr(settings, "openai_compatible_client_timeout_seconds", 300))


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

    settings = _settings(cfg)
    runtime = _runtime(settings, runtime_config)
    quick = bool(getattr(runtime, "quick_mode_active", False))
    if timeout_class is TimeoutClass.EDITORIAL:
        cap = _editorial_invocation_cap(settings)
    elif quick:
        cap = quick_invocation_cap(timeout_class, settings)
    else:
        cap = float(runtime.llm_call_hard_timeout_seconds)
    # Full-mode flex: the cap must outlast a queued flex request plus its
    # standard re-issue. A no-op in quick mode and when flex is off.
    cap = floor_llm_hard_timeout(
        cap,
        provider=provider,
        cfg=settings,
        label=f"invoke_hard_timeout:{provider}",
    )
    fraction = float(getattr(settings, "flex_quick_attempt_fraction", 0.4))
    flex_request_cap = cap * fraction if quick else float(flex_floor_seconds(settings))
    request_cap = (
        float(request_cap_override)
        if request_cap_override is not None
        else _standard_request_cap(timeout_class, provider, settings, quick)
    )
    # Scale the floors for a deliberately tiny cap so it stays admissible.
    min_useful = min(
        float(getattr(settings, "llm_min_useful_attempt_seconds", 5.0)), cap / 2.0
    )
    margin = min(
        float(getattr(settings, "llm_dispatch_margin_seconds", 2.0)), cap * 0.05
    )
    attempts = max(1, int(getattr(runtime, "llm_attempts", 2)))
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
    policy: BudgetPolicy, provider: str | None, *, cfg: Any = None
) -> BudgetPolicy:
    """The scope's policy with its request ceiling re-derived for ``provider``.

    The invocation's allowance (cap, margin, minimum useful attempt, attempts)
    belongs to the invocation and is kept, as is the deadline it was opened
    with. Only the per-request ceiling is a property of the vendor that
    actually receives the request: a delegate on another vendor must not send
    the original vendor's ceiling. A seat's explicit client timeout still wins.
    """
    if _provider_family(policy.provider) == _provider_family(provider):
        return policy
    if policy.request_cap_override is not None:
        request_cap = policy.request_cap_override
    else:
        request_cap = _standard_request_cap(
            policy.timeout_class, provider, _settings(cfg), policy.quick_mode
        )
    return replace(policy, provider=provider, request_cap=request_cap)


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


def construction_transport_attempts(
    seat: SeatSpec | None, *, vendor: str, cfg: Any = None
) -> int:
    """Total SDK attempts a client is *built* with (direct callers rely on it).

    A seat's explicit ``transport_attempts`` (or its deprecated alias,
    translated) always wins. Google sends per-request attempts under an
    application invocation, so its built value only serves direct callers.
    OpenAI-family and Anthropic SDKs fix retries on the client, so a seat whose
    retries the runtime owns is built with one attempt. Every other client gets
    the run-scoped ``llm_attempts`` (``direct_caller_transport_attempts``).
    """
    if seat is not None:
        explicit = seat.execution_policy.resolved_transport_attempts(vendor)
        if explicit is not None:
            return explicit
    if (
        vendor != "google"
        and seat is not None
        and seat.timeout_class.retries_owned_by_runtime
    ):
        return 1
    return direct_caller_transport_attempts(cfg)


def sdk_max_retries(vendor: str, attempts: int) -> int:
    """Translate total attempts into the vendor SDK's ``max_retries`` argument.

    langchain-google-genai maps ``max_retries`` to ``HttpRetryOptions(attempts=N)``
    (total attempts); OpenAI and Anthropic count retries after the first request.
    """
    total = max(1, int(attempts))
    return total if vendor == "google" else total - 1
