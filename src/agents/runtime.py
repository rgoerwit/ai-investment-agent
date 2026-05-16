from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import structlog

from src.agents.circuit_breaker import CircuitOpenError, get_circuit_breaker
from src.async_utils import run_with_hard_timeout
from src.config import config as settings_config
from src.llm_usage import extract_token_usage_breakdown
from src.runtime_diagnostics import (
    classify_failure,
    get_class_name,
    get_model_name,
    infer_provider,
)

logger = structlog.get_logger(__name__)

try:
    from src.eval import get_active_capture_manager as _get_capture_manager
    from src.eval.llm_capture_meta import (
        extract_token_usage as _extract_token_usage,
    )
    from src.eval.llm_capture_meta import (
        extract_vendor_reasoning_config as _extract_vendor_reasoning_config,
    )
    from src.eval.llm_capture_meta import (
        normalize_reasoning_level as _normalize_reasoning_level,
    )
    from src.eval.serialization import normalize_for_json as _normalize_for_json
except ImportError:

    def _get_capture_manager():
        return None

    def _normalize_reasoning_level(runnable, model_name):
        return None

    def _extract_vendor_reasoning_config(runnable, provider):
        return None

    def _extract_token_usage(result):
        return {
            "input_tokens": None,
            "output_tokens": None,
            "thinking_tokens": None,
            "total_tokens": None,
        }

    def _normalize_for_json(value):
        return value


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


class ProviderPartialResponseError(RuntimeError):
    """Raised when an LLM call returned successfully but the response
    looks like a provider-side partial (no finish_reason, length cap,
    etc). The marker `provider_partial_response` in the message routes
    the failure to the transient-retry branch in
    `invoke_with_rate_limit_handling`."""


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
    overall_timeout_seconds: float | None = None,
) -> Any:
    """
    Invoke an LLM with explicit 429 and transient error handling.
    """
    quiet_mode = settings_config.quiet_mode
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

    if getattr(settings_config, "quick_mode_active", False):
        hard_timeout = float(
            getattr(settings_config, "quick_llm_call_hard_timeout_seconds", 60.0)
        )
    else:
        hard_timeout = float(
            getattr(settings_config, "llm_call_hard_timeout_seconds", 120.0)
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

            result = await run_with_hard_timeout(
                runnable.ainvoke(input_data),
                timeout=effective_timeout,
                label=f"llm:{context}:{resolved_provider}:{resolved_model}",
            )
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
            if partial_reason is not None and attempt < transient_max_attempts - 1:
                raise ProviderPartialResponseError(
                    f"provider_partial_response: {partial_reason} "
                    f"(context={context}, provider={resolved_provider}, "
                    f"model={resolved_model})"
                )
            try:
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
            except Exception as _acct_exc:
                logger.debug("accounting_hook_failed", error=str(_acct_exc))
            try:
                from src.token_tracker import get_tracker

                usage = extract_token_usage_breakdown(result)
                get_tracker().record_call_attempt(
                    agent_name=context,
                    provider=resolved_provider,
                    model_name=resolved_model or "",
                    status="success",
                    attempt=attempt + 1,
                    elapsed_seconds=time.monotonic() - attempt_started,
                    prompt_tokens=usage.input_tokens,
                    completion_tokens=usage.total_output_tokens,
                    total_tokens=usage.total_tokens,
                )
            except Exception as _acct_exc:
                logger.debug("accounting_hook_failed", error=str(_acct_exc))
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
            try:
                from src.token_tracker import get_tracker

                get_tracker().record_call_attempt(
                    agent_name=context,
                    provider=resolved_provider,
                    model_name=resolved_model or "",
                    status="failure",
                    attempt=attempt + 1,
                    elapsed_seconds=elapsed_seconds,
                    failure_kind="circuit_open",
                    failure_origin="circuit_breaker",
                    retryable=False,
                )
            except Exception as _acct_exc:
                logger.debug("accounting_hook_failed", error=str(_acct_exc))
            logger.warning(
                "llm_call_circuit_open",
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
            )
            elapsed_seconds = time.monotonic() - attempt_started
            if breaker is not None:
                breaker.record_outcome(
                    agent_name=context,
                    provider=resolved_provider,
                    model_name=resolved_model or "",
                    ok=False,
                    failure_kind=details.kind,
                )
            try:
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
                            "error_message": details.message,
                        }
                    )
            except Exception as _acct_exc:
                logger.debug("accounting_hook_failed", error=str(_acct_exc))
            try:
                from src.token_tracker import get_tracker

                failure_origin = (
                    _timeout_failure_origin(exc) if details.kind == "timeout" else None
                )
                get_tracker().record_call_attempt(
                    agent_name=context,
                    provider=details.provider,
                    model_name=resolved_model or "",
                    status="failure",
                    attempt=attempt + 1,
                    elapsed_seconds=elapsed_seconds,
                    failure_kind=details.kind,
                    failure_origin=failure_origin,
                    retryable=details.retryable,
                )
            except Exception as _acct_exc:
                logger.debug("accounting_hook_failed", error=str(_acct_exc))

            is_rate_limit = details.kind in {"rate_limit", "quota_error"}
            is_transient = details.kind in {
                "dns_resolution",
                "connect_error",
                "timeout",
                "server_error",
                "provider_partial_response",
            }

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
            if is_transient and attempt < transient_max_attempts - 1:
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
