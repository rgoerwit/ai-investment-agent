from __future__ import annotations

import asyncio
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import structlog

from src.agents.circuit_breaker import CircuitOpenError, get_circuit_breaker
from src.agents.network_breaker import (
    NetworkBreakerOpenError,
    get_network_breaker,
)
from src.async_utils import run_with_hard_timeout
from src.config import config as settings_config
from src.error_safety import summarize_exception
from src.llm_usage import extract_token_usage_breakdown
from src.runtime_config import get_runtime_config
from src.runtime_diagnostics import (
    classify_failure,
    get_base_url,
    get_class_name,
    get_model_name,
    infer_provider,
)
from src.service_tiers import floor_llm_hard_timeout, provider_flex_active
from src.token_tracker import canonical_display_name

logger = structlog.get_logger(__name__)


# Graph node labels (agent_name / context) of the two gate-critical APEX seats.
# These run the biggest, most rule-dense prompts on the APEX model and need a
# larger --quick per-call budget than the flat quick cap sized for cheap flash
# data-gathering agents. Kept in lockstep with APEX_SEATS in src/llms.py via
# tests/agents/test_runtime_hard_timeout.py (the seats' prompt agent_name).
APEX_SEAT_CONTEXTS = frozenset({"Fundamentals Analyst", "Portfolio Manager"})
# Canonical display names — see canonical_display_name() below on why these are
# not the raw context strings.
CROSS_CHECK_SEAT_CONTEXTS = frozenset({"Consultant", "Global Forensic Auditor"})


def quick_mode_hard_timeout_seconds(
    context: str,
    cfg: Any = settings_config,
    canonical_agent: str | None = None,
) -> float:
    """Per-call hard wall-clock cap for a single ``--quick`` LLM ainvoke.

    Three budgets, by why the seat needs one:

    * Gate-critical APEX seats (Senior Fundamentals, Portfolio Manager) get
      ``apex_quick_llm_call_hard_timeout_seconds`` so the tight quick cap can't
      guillotine the DATA_BLOCK / PM_BLOCK the hard gates depend on.
    * The OpenAI cross-check plane (Consultant, Forensic Auditor) gets
      ``cross_check_quick_llm_call_hard_timeout_seconds`` — a *vendor-latency*
      allowance, not a criticality one, because these seats can be pointed at an
      OpenAI-compatible endpoint whose reasoning model exceeds the flat cap.
    * Everything else (cheap, fast flash agents) keeps
      ``quick_llm_call_hard_timeout_seconds`` so quick mode still surfaces a hung
      provider quickly.

    Seat identity comes from the explicit ``canonical_agent`` when the call site
    supplies one, and is otherwise *derived* from the decorated context. Deriving
    alone is not sufficient and was a live defect twice over: a raw membership
    test missed ``"Fundamentals Analyst (RETRY-HIGH)"`` (a gate-critical seat's
    deep-model retry running on a *smaller* per-call budget than its first
    attempt), and even after canonicalization the free-form correction contexts
    — ``"Portfolio Manager structure correction"``, ``"pm_verdict_recovery"``,
    ``"global_forensic_auditor_repair"`` — carry no recognizable suffix and fell
    back to the 60s flat cap. Those corrections enforce the final
    recommendation's evidence contract, so they are at least as gate-critical as
    the first call. Call sites that decorate their context must pass
    ``canonical_agent``; ``tests/agents/test_runtime_hard_timeout.py::
    TestSeatBudgetCoversEveryCallSite`` scans the source and fails if one does not.
    """
    seat = canonical_display_name(canonical_agent or context)
    if seat in APEX_SEAT_CONTEXTS:
        return float(getattr(cfg, "apex_quick_llm_call_hard_timeout_seconds", 180.0))
    if seat in CROSS_CHECK_SEAT_CONTEXTS:
        return float(
            getattr(cfg, "cross_check_quick_llm_call_hard_timeout_seconds", 180.0)
        )
    return float(getattr(cfg, "quick_llm_call_hard_timeout_seconds", 60.0))


@contextmanager
def _accounting_hook(label: str):
    """Wrap an accounting/observability side-effect so its failures never bubble.

    Accounting paths (capture manager, token tracker) can touch provider error
    strings that include URLs or partial payloads — even at DEBUG level we
    route through summarize_exception() instead of raw str(exc) so that
    redaction and structured fields apply uniformly.

    Uses ``except Exception`` (not ``BaseException``) so KeyboardInterrupt
    and SystemExit propagate normally.
    """
    try:
        yield
    except Exception as exc:
        logger.debug(
            "accounting_hook_failed",
            hook=label,
            **summarize_exception(exc, operation=f"accounting:{label}"),
        )


_get_capture_manager: Any
_normalize_reasoning_level: Any
_extract_vendor_reasoning_config: Any
_extract_token_usage: Any
_normalize_for_json: Any

try:
    from src.eval import get_active_capture_manager
    from src.eval.llm_capture_meta import (
        extract_token_usage,
        extract_vendor_reasoning_config,
        normalize_reasoning_level,
    )
    from src.eval.serialization import normalize_for_json

    _get_capture_manager = get_active_capture_manager
    _normalize_reasoning_level = normalize_reasoning_level
    _extract_vendor_reasoning_config = extract_vendor_reasoning_config
    _extract_token_usage = extract_token_usage
    _normalize_for_json = normalize_for_json
except ImportError:

    def _fallback_capture_manager() -> None:
        return None

    def _fallback_normalize_reasoning_level(runnable: Any, model_name: Any) -> None:
        return None

    def _fallback_extract_vendor_reasoning_config(runnable: Any, provider: Any) -> None:
        return None

    def _fallback_extract_token_usage(result: Any) -> dict[str, int | None]:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "thinking_tokens": None,
            "total_tokens": None,
        }

    def _fallback_normalize_for_json(value: Any) -> Any:
        return value

    _get_capture_manager = _fallback_capture_manager
    _normalize_reasoning_level = _fallback_normalize_reasoning_level
    _extract_vendor_reasoning_config = _fallback_extract_vendor_reasoning_config
    _extract_token_usage = _fallback_extract_token_usage
    _normalize_for_json = _fallback_normalize_for_json


def _detect_provider_partial_response(result: Any) -> str | None:
    """Return a non-None reason string iff *result* is a suspected partial
    response from the provider — otherwise None.

    The canonical signal is `response.response_metadata.finish_reason`.
    Each provider returns a value here on a successful call:

    - ``"stop"``: model emitted a stop sequence or naturally finished. ✓
    - ``"tool_calls"`` / ``"tool_use"``: paused for tool execution. ✓
      (the agent loop continues with tool results.)
    - ``"length"``: hit ``max_completion_tokens`` cap before finishing.
      Treat as partial — re-running rarely helps but at least surfaces
      the issue and lets `agent_output_truncated` correctly classify it.
    - ``"content_filter"``: provider safety filter; not a transient
      glitch — propagate as-is, do NOT retry.
    - missing / ``None``: response was not finalized cleanly; commonly a
      provider-side stream interruption (the May 2026 2382.HK auditor
      truncation). Worth a retry.

    Some response shapes (Gemini, certain LangChain wrappers) place the
    finish reason in different keys; we accept any of them. Returning
    None means "looks like a clean finish, don't retry."
    """
    # An active tool-loop step is not a partial response: the model paused
    # to call a tool, content can legitimately be empty. The agent loop
    # will continue with tool results. Skip the entire partial check here
    # — finish_reason might also be missing on Mock-style test responses
    # that legitimately use tool_calls.
    tool_calls = getattr(result, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        return None

    response_metadata = getattr(result, "response_metadata", None)
    if not isinstance(response_metadata, dict):
        response_metadata = {}

    # OpenAI Responses API path: when create_consultant_llm /
    # create_auditor_llm / create_editor_llm pass `use_responses_api=True`
    # + `output_version="responses/v1"`, langchain_openai populates
    # response_metadata with {created_at, id, incomplete_details, metadata,
    # object, status, user, model, service_tier} — but NOT finish_reason.
    # Status values: "completed" (clean), "incomplete" (length/refusal —
    # see incomplete_details.reason), "failed" (error). Recognize this
    # shape FIRST so we don't false-positive on every successful auditor /
    # consultant call.
    status = response_metadata.get("status")
    if status == "completed":
        return None
    if status == "incomplete":
        details = response_metadata.get("incomplete_details") or {}
        reason = (
            details.get("reason") if isinstance(details, dict) else None
        ) or "incomplete"
        return f"responses_api_incomplete:{reason}"
    if status == "failed":
        return "responses_api_failed"

    finish_reason = (
        response_metadata.get("finish_reason")
        or response_metadata.get("stop_reason")
        or response_metadata.get("done_reason")
    )

    if finish_reason in ("stop", "end_turn", "STOP", "tool_calls", "tool_use"):
        return None
    if finish_reason in ("length", "max_tokens", "MAX_TOKENS"):
        return f"finish_reason_{finish_reason}"
    if finish_reason in ("content_filter", "SAFETY"):
        # Intentional stop, not a partial. Caller decides what to do.
        return None

    # finish_reason missing. This is the suspect case BUT we want to avoid
    # flagging non-provider AIMessages (test mocks, agents constructing
    # synthetic messages) as partial. Two corroborating signals tell us
    # we're looking at an actual provider response that came back wrong:
    #
    # 1. ``response_metadata`` is non-empty (provider sent SOMETHING but
    #    omitted finish_reason).
    # 2. ``content`` is empty or whitespace-only.
    #
    # Either alone is enough; both together is the canonical 2382.HK
    # auditor truncation pattern. Pure ``AIMessage(content="...")`` with
    # no metadata is treated as a clean response.
    has_provider_metadata = bool(response_metadata)
    content = getattr(result, "content", None)
    has_empty_content = isinstance(content, str) and content.strip() == ""

    if has_provider_metadata and not has_empty_content:
        # Provider returned other metadata but no finish_reason; suspect.
        return "finish_reason_missing"
    if has_empty_content:
        # Content vanished — definitely not a clean finish.
        return "empty_content_no_finish_reason"

    # No corroborating signal — let it through. Common for AIMessage
    # constructed in tests or for non-streaming providers that don't
    # populate response_metadata.
    return None


def response_partial_reason(result: Any) -> str | None:
    """The provider's reason for cutting a response short, if it did.

    Broader than :func:`response_hit_output_cap` on purpose. A caller deciding
    whether it is holding a *finished answer* cares about every partial the
    classifier recognizes — ``finish_reason_missing``, ``responses_api_failed``,
    ``responses_api_incomplete:*`` — not only the token-cap subset. Keying
    recovery on the cap alone let a nonempty fragment with no finish reason be
    published as a complete Consultant review.

    Refusals are deliberately excluded upstream (``content_filter``/``SAFETY``
    return ``None`` from the classifier): those are intentional stops owned by
    the refusal path, and re-asking an identical prompt would not clear them.
    """
    return _detect_provider_partial_response(result)


def response_hit_output_cap(result: Any) -> bool:
    """True when the provider says the response stopped at the token cap.

    Derived from the one partial-response classifier above so there is a single
    definition of "the provider cut this off". The retry loop already raises on
    a partial, but the *final* attempt returns the truncated message as-is —
    callers that publish an artifact need to know they are holding a fragment
    rather than a finished answer. Fragments are truthy, so an emptiness check
    alone cannot see this (the 1088.HK consultant "review" of 46 characters).
    """
    reason = _detect_provider_partial_response(result)
    if reason is None:
        return False
    lowered = reason.lower()
    return any(
        marker in lowered for marker in ("length", "max_tokens", "max_output_tokens")
    )


class ProviderPartialResponseError(RuntimeError):
    """Raised when an LLM call returned successfully but the response
    looks like a provider-side partial (no finish_reason, length cap,
    etc). The marker `provider_partial_response` in the message routes
    the failure to the transient-retry branch in
    `invoke_with_rate_limit_handling`."""


@dataclass(frozen=True)
class RefusalSignal:
    """A recognized provider safety/content-policy block on a returned
    response. ``reason_code`` is a stable lowercase token
    (``content_filter``/``safety``/``recitation``/``prohibited_content``/
    ``refusal``/``prompt_blocked:<x>``); ``stage`` is ``"response"`` (the
    model produced output that was blocked/refused) or ``"request"`` (the
    prompt itself was blocked before generation)."""

    reason_code: str
    stage: str


class ProviderRefusalError(RuntimeError):
    """Raised when an LLM call returned a provider safety block / content
    refusal instead of a usable response. The message carries the marker
    `provider_safety_block` so `classify_failure` maps it to the
    ``provider_safety_block`` kind — which is in neither the rate-limit nor
    the transient-retry set, so it is non-retryable by construction and
    degrades through the node-level `except -> failure_artifact` wrapper."""

    def __init__(self, signal: RefusalSignal):
        self.reason_code = signal.reason_code
        self.stage = signal.stage
        super().__init__(
            f"provider_safety_block: {signal.reason_code} (stage={signal.stage})"
        )


def _detect_provider_refusal(result: Any) -> RefusalSignal | None:
    """Return a `RefusalSignal` iff *result* is a provider safety block /
    content-policy refusal — otherwise None.

    Metadata-only by design: this inspects `finish_reason`/`stop_reason`,
    the OpenAI Responses-API `incomplete_details.reason`, the OpenAI Chat
    structured `.refusal` field, and Gemini `prompt_feedback.block_reason`.
    It never matches on the model's prose — a keyword scan of output text
    would false-positive on legitimate financial content (e.g. a report that
    discusses "safety") and kill a good run. Callers must short-circuit
    active tool-call turns before calling this (an empty content on a
    tool_calls turn is normal, not a refusal)."""
    response_metadata = getattr(result, "response_metadata", None)
    if not isinstance(response_metadata, dict):
        response_metadata = {}

    # OpenAI Responses API: an incomplete status names the reason.
    if response_metadata.get("status") == "incomplete":
        details = response_metadata.get("incomplete_details") or {}
        reason = details.get("reason") if isinstance(details, dict) else None
        if reason in ("content_filter", "refusal"):
            return RefusalSignal(reason, "response")

    finish_reason = response_metadata.get("finish_reason") or response_metadata.get(
        "stop_reason"
    )
    if finish_reason == "content_filter":
        return RefusalSignal("content_filter", "response")
    if finish_reason == "SAFETY":
        return RefusalSignal("safety", "response")
    if finish_reason == "RECITATION":
        return RefusalSignal("recitation", "response")
    if finish_reason in ("PROHIBITED_CONTENT", "SPII", "BLOCKLIST"):
        return RefusalSignal(str(finish_reason).lower(), "response")
    if finish_reason == "refusal":  # Anthropic stop_reason
        return RefusalSignal("refusal", "response")

    # OpenAI Chat Completions structured refusal (langchain surfaces it in
    # additional_kwargs; the response text is empty in this case).
    additional_kwargs = getattr(result, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict) and additional_kwargs.get("refusal"):
        return RefusalSignal("refusal", "response")

    # Gemini prompt-level block, when the wrapper surfaces prompt_feedback
    # (best-effort — not currently exposed by every langchain-google version).
    prompt_feedback = response_metadata.get("prompt_feedback")
    if isinstance(prompt_feedback, dict) and prompt_feedback.get("block_reason"):
        return RefusalSignal(
            f"prompt_blocked:{prompt_feedback['block_reason']}", "request"
        )

    return None


def _timeout_failure_origin(exc: BaseException) -> str | None:
    """Classify timeout source without changing the stable failure_kind."""
    text = f"{type(exc).__name__}: {exc}".lower()
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    cause_type = type(cause).__name__ if cause is not None else ""

    if "exceeded hard timeout" in text:
        return "hard_timeout"
    if type(exc).__name__ == "TimeoutError" and cause_type == "CancelledError":
        return "provider_sdk_timeout"
    if "node timeout" in text or "nodetimeout" in text:
        return "graph_node_timeout"
    if "watchdog" in text or "pipeline" in text:
        return "pipeline_watchdog"
    return None


async def invoke_with_rate_limit_handling(
    runnable,
    input_data: dict[str, Any] | list[Any],
    max_attempts: int = 3,
    max_transient_attempts: int = 2,
    context: str = "LLM",
    provider: str | None = None,
    model_name: str | None = None,
    canonical_agent: str | None = None,
    overall_timeout_seconds: float | None = None,
) -> Any:
    """
    Invoke an LLM with explicit 429 and transient error handling.

    ``canonical_agent`` is the display-namespace identity for cost/diagnostic
    joins; when omitted the token tracker derives it from ``context`` (which may
    carry a round/retry suffix).
    """
    runtime_config = get_runtime_config(settings_config)
    quiet_mode = runtime_config.quiet_mode
    resolved_model = model_name or get_model_name(runnable)
    class_name = get_class_name(runnable)
    resolved_provider = provider or infer_provider(
        model_name=resolved_model,
        class_name=class_name,
    )

    if not quiet_mode:
        logger.debug(
            "llm_call_start",
            context=context,
            provider=resolved_provider,
            model=resolved_model,
            runnable_class=class_name,
            max_attempts=max_attempts,
            max_transient_attempts=max_transient_attempts,
            overall_timeout_seconds=overall_timeout_seconds,
        )

    if runtime_config.quick_mode_active:
        hard_timeout = quick_mode_hard_timeout_seconds(
            context, settings_config, canonical_agent=canonical_agent
        )
    else:
        hard_timeout = float(runtime_config.llm_call_hard_timeout_seconds)
    # Flex tier: queued calls may take minutes; floor the hard cap so a
    # legitimately-slow flex call (plus its potential standard-tier fallback
    # attempt) isn't killed and fed to the circuit breaker as a timeout.
    hard_timeout = floor_llm_hard_timeout(
        hard_timeout,
        provider=resolved_provider,
        cfg=settings_config,
        label=f"invoke_hard_timeout:{resolved_provider}",
    )
    deadline = (
        time.monotonic() + float(overall_timeout_seconds)
        if overall_timeout_seconds is not None
        else None
    )

    breaker_enabled = bool(
        getattr(settings_config, "llm_circuit_breaker_enabled", True)
    )
    breaker = get_circuit_breaker() if breaker_enabled else None
    network_breaker_enabled = bool(
        getattr(settings_config, "network_breaker_enabled", True)
    )
    network_breaker = get_network_breaker() if network_breaker_enabled else None

    for attempt in range(max_attempts):
        attempt_started = time.monotonic()
        try:
            effective_timeout = hard_timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"{context} exceeded {overall_timeout_seconds:.1f}s "
                        "overall LLM timeout"
                    )
                effective_timeout = min(hard_timeout, max(0.1, remaining))

            if breaker is not None:
                breaker.before_call(
                    agent_name=context,
                    provider=resolved_provider,
                    model_name=resolved_model or "",
                )
            if network_breaker is not None:
                network_breaker.before_call()

            result = await run_with_hard_timeout(
                runnable.ainvoke(input_data),
                timeout=effective_timeout,
                label=f"llm:{context}:{resolved_provider}:{resolved_model}",
            )
            # Refusal / provider safety block: a returned response that the
            # provider blocked or refused (finish_reason SAFETY/content_filter,
            # Anthropic stop_reason=refusal, OpenAI structured .refusal, Gemini
            # prompt block). An active tool-call turn is never a refusal, so
            # skip the check there. On a hit, emit the dedicated event (safe
            # metadata only — no prompt contents) and raise; the generic
            # `except` below records the single ledger failure via
            # classify_failure -> provider_safety_block (non-retryable).
            _tool_calls = getattr(result, "tool_calls", None)
            if not (isinstance(_tool_calls, list) and _tool_calls):
                refusal = _detect_provider_refusal(result)
                if refusal is not None:
                    _content = getattr(result, "content", "") or ""
                    logger.warning(
                        "llm_refusal_detected",
                        context=context,
                        provider=resolved_provider,
                        model=resolved_model,
                        stage=refusal.stage,
                        reason_code=refusal.reason_code,
                        output_len=len(_content) if isinstance(_content, str) else None,
                        input_messages=(
                            len(input_data["messages"])
                            if isinstance(input_data, dict)
                            and isinstance(input_data.get("messages"), list)
                            else None
                        ),
                    )
                    raise ProviderRefusalError(refusal)
            # Inspect finish_reason: providers occasionally return a "200 OK"
            # with truncated content and no finish_reason under load. Raise
            # the marker exception so the existing transient-retry branch
            # handles it like any other recoverable failure.
            partial_reason = _detect_provider_partial_response(result)
            transient_max_attempts = max(1, min(max_attempts, max_transient_attempts))
            if partial_reason is None and breaker is not None:
                breaker.record_outcome(
                    agent_name=context,
                    provider=resolved_provider,
                    model_name=resolved_model or "",
                    ok=True,
                )
            if partial_reason is None and network_breaker is not None:
                # Any successful call demonstrates the host network is fine.
                network_breaker.record_outcome(ok=True)
            if partial_reason is not None and attempt < transient_max_attempts - 1:
                raise ProviderPartialResponseError(
                    f"provider_partial_response: {partial_reason} "
                    f"(context={context}, provider={resolved_provider}, "
                    f"model={resolved_model})"
                )
            with _accounting_hook("capture_manager_success"):
                capture_manager = _get_capture_manager()
                if capture_manager is not None:
                    token_usage = _extract_token_usage(result)
                    response_metadata = getattr(result, "response_metadata", None)
                    response_model = None
                    if isinstance(response_metadata, dict):
                        response_model = response_metadata.get(
                            "model_name"
                        ) or response_metadata.get("model")
                    capture_manager.record_llm_call(
                        {
                            "status": "success",
                            "context": context,
                            "provider": resolved_provider,
                            "model": resolved_model,
                            "response_model": response_model,
                            "runnable_class": class_name,
                            "reasoning_level": _normalize_reasoning_level(
                                runnable, resolved_model
                            ),
                            "thinking_config_raw": _extract_vendor_reasoning_config(
                                runnable, resolved_provider
                            ),
                            "attempt": attempt + 1,
                            **token_usage,
                            "input": _normalize_for_json(input_data),
                            "response": _normalize_for_json(result),
                        }
                    )
            with _accounting_hook("token_tracker_success"):
                from src.token_tracker import get_tracker

                usage = extract_token_usage_breakdown(result)
                get_tracker().record_call_attempt(
                    agent_name=context,
                    canonical_agent=canonical_agent,
                    provider=resolved_provider,
                    model_name=resolved_model or "",
                    status="success",
                    attempt=attempt + 1,
                    elapsed_seconds=time.monotonic() - attempt_started,
                    prompt_tokens=usage.input_tokens,
                    completion_tokens=usage.total_output_tokens,
                    total_tokens=usage.total_tokens,
                )
            if not quiet_mode:
                logger.info(
                    "llm_call_success",
                    context=context,
                    provider=resolved_provider,
                    model=resolved_model,
                    runnable_class=class_name,
                    attempt=attempt + 1,
                )
            return result
        except CircuitOpenError as exc:
            # Fast-fail: the breaker already classified this provider/model
            # as bad. Record a stub failure attempt so call_diagnostics
            # surface the short-circuit, then propagate without sleeping
            # or retrying — the whole point is to skip the wait.
            elapsed_seconds = time.monotonic() - attempt_started
            with _accounting_hook("token_tracker_circuit_open"):
                from src.token_tracker import get_tracker

                get_tracker().record_call_attempt(
                    agent_name=context,
                    canonical_agent=canonical_agent,
                    provider=resolved_provider,
                    model_name=resolved_model or "",
                    status="failure",
                    attempt=attempt + 1,
                    elapsed_seconds=elapsed_seconds,
                    failure_kind="circuit_open",
                    failure_origin="circuit_breaker",
                    retryable=False,
                )
            logger.warning(
                "llm_call_circuit_open",
                context=context,
                provider=resolved_provider,
                model=resolved_model,
                attempt=attempt + 1,
                reopens_in_seconds=round(exc.opens_remaining_seconds, 2),
            )
            raise
        except NetworkBreakerOpenError as exc:
            # Process-global network breaker is open — short-circuit
            # without retry. The host's DNS/TCP is unhealthy; sleeping
            # 30s then retrying achieves nothing. The graceful-degradation
            # paths in analyst/researcher nodes already handle this
            # exception the same way they handle a final llm_call_failed.
            elapsed_seconds = time.monotonic() - attempt_started
            with _accounting_hook("token_tracker_network_breaker_open"):
                from src.token_tracker import get_tracker

                get_tracker().record_call_attempt(
                    agent_name=context,
                    canonical_agent=canonical_agent,
                    provider=resolved_provider,
                    model_name=resolved_model or "",
                    status="failure",
                    attempt=attempt + 1,
                    elapsed_seconds=elapsed_seconds,
                    failure_kind="network_breaker_open",
                    failure_origin="network_breaker",
                    retryable=False,
                )
            logger.warning(
                "llm_call_network_breaker_open",
                context=context,
                provider=resolved_provider,
                model=resolved_model,
                attempt=attempt + 1,
                reopens_in_seconds=round(exc.opens_remaining_seconds, 2),
            )
            raise
        except Exception as exc:
            details = classify_failure(
                exc,
                provider=resolved_provider,
                model_name=resolved_model,
                class_name=class_name,
                # `provider` names the transport family, which is "openai" for
                # every OpenAI-compatible endpoint. The endpoint host is what
                # identifies the vendor that actually served (or refused) the
                # call, and it needs no vendor lookup table to do so.
                base_url=get_base_url(runnable),
            )
            elapsed_seconds = time.monotonic() - attempt_started
            # Quick-mode flex latency timeout: a queued flex call the transport
            # could not fall back to standard (fallback disabled, or the flex
            # client timeout raced the outer hard cap). It is a queue event, not
            # a provider fault — so it must NOT re-queue at flex (non-retryable)
            # and must be excluded from the circuit breakers, which would
            # otherwise trip and fast-fail sibling agents on transient queueing.
            # See the mitigation plan's flex-fallback x timeout matrix (rows 6-8).
            flex_latency_timeout_quick = (
                details.kind == "timeout"
                and runtime_config.quick_mode_active
                and provider_flex_active(resolved_provider, settings_config)
            )
            # A provider safety block / content refusal is content-specific, not
            # evidence the provider or model is unhealthy — exclude it from the
            # circuit breakers so a run's repeated refusals cannot trip them and
            # fast-fail unrelated sibling agents. (Same rationale as the flex
            # queue-event exclusion above.)
            exclude_from_breakers = (
                flex_latency_timeout_quick or details.kind == "provider_safety_block"
            )
            if breaker is not None and not exclude_from_breakers:
                breaker.record_outcome(
                    agent_name=context,
                    provider=resolved_provider,
                    model_name=resolved_model or "",
                    ok=False,
                    failure_kind=details.kind,
                )
            if network_breaker is not None and not exclude_from_breakers:
                network_breaker.record_outcome(ok=False, failure_kind=details.kind)
            with _accounting_hook("capture_manager_failure"):
                capture_manager = _get_capture_manager()
                if capture_manager is not None:
                    capture_manager.record_llm_call(
                        {
                            "status": "failure",
                            "context": context,
                            "provider": resolved_provider,
                            "model": resolved_model,
                            "runnable_class": class_name,
                            "reasoning_level": _normalize_reasoning_level(
                                runnable, resolved_model
                            ),
                            "thinking_config_raw": _extract_vendor_reasoning_config(
                                runnable, resolved_provider
                            ),
                            "attempt": attempt + 1,
                            "input_tokens": None,
                            "output_tokens": None,
                            "thinking_tokens": None,
                            "total_tokens": None,
                            "input": _normalize_for_json(input_data),
                            "failure_kind": details.kind,
                            "retryable": details.retryable,
                            "error_type": details.error_type,
                            "root_cause_type": details.root_cause_type,
                            "host": details.host,
                            "endpoint_host": details.endpoint_host,
                            "error_message": details.message,
                        }
                    )
            with _accounting_hook("token_tracker_failure"):
                from src.token_tracker import get_tracker

                failure_origin = (
                    _timeout_failure_origin(exc) if details.kind == "timeout" else None
                )
                get_tracker().record_call_attempt(
                    agent_name=context,
                    canonical_agent=canonical_agent,
                    provider=details.provider,
                    model_name=resolved_model or "",
                    status="failure",
                    attempt=attempt + 1,
                    elapsed_seconds=elapsed_seconds,
                    failure_kind=details.kind,
                    failure_origin=failure_origin,
                    retryable=details.retryable,
                )

            is_rate_limit = details.kind in {"rate_limit", "quota_error"}
            # ``classify_failure`` owns retryability. Keep only the distinct
            # rate-limit branch here because it has a different backoff policy.
            is_transient = details.retryable and not is_rate_limit

            if is_rate_limit and attempt < max_attempts - 1:
                jitter = random.uniform(1, 10)
                wait_time = (60 * (attempt + 1)) + jitter
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.error(
                            "llm_call_failed",
                            context=context,
                            provider=resolved_provider,
                            model=resolved_model,
                            runnable_class=class_name,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            failure_kind=details.kind,
                            host=details.host,
                            retryable=details.retryable,
                            error_type=details.error_type,
                            root_cause_type=details.root_cause_type,
                            error_message=details.message,
                        )
                        raise
                    wait_time = min(wait_time, max(0.0, remaining))
                logger.warning(
                    "llm_call_retry",
                    context=context,
                    provider=resolved_provider,
                    model=resolved_model,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    failure_kind=details.kind,
                    host=details.host,
                    retryable=details.retryable,
                    wait_seconds=f"{wait_time:.1f}",
                    error_type=details.error_type,
                    root_cause_type=details.root_cause_type,
                    error_message=details.message,
                )
                await asyncio.sleep(wait_time)
                continue

            transient_max_attempts = max(1, min(max_attempts, max_transient_attempts))
            if (
                is_transient
                and not flex_latency_timeout_quick
                and attempt < transient_max_attempts - 1
            ):
                wait_time = 5 * (attempt + 1) + random.uniform(1, 3)
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.error(
                            "llm_call_failed",
                            context=context,
                            provider=resolved_provider,
                            model=resolved_model,
                            runnable_class=class_name,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            failure_kind=details.kind,
                            host=details.host,
                            retryable=details.retryable,
                            error_type=details.error_type,
                            root_cause_type=details.root_cause_type,
                            error_message=details.message,
                        )
                        raise
                    wait_time = min(wait_time, max(0.0, remaining))
                logger.warning(
                    "llm_call_retry",
                    context=context,
                    provider=resolved_provider,
                    model=resolved_model,
                    attempt=attempt + 1,
                    max_attempts=transient_max_attempts,
                    failure_kind=details.kind,
                    host=details.host,
                    retryable=details.retryable,
                    wait_seconds=f"{wait_time:.1f}",
                    error_type=details.error_type,
                    root_cause_type=details.root_cause_type,
                    error_message=details.message,
                )
                await asyncio.sleep(wait_time)
                continue

            logger.error(
                "llm_call_failed",
                context=context,
                provider=resolved_provider,
                model=resolved_model,
                runnable_class=class_name,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                failure_kind=details.kind,
                host=details.host,
                retryable=details.retryable,
                error_type=details.error_type,
                root_cause_type=details.root_cause_type,
                error_message=details.message,
            )
            raise
