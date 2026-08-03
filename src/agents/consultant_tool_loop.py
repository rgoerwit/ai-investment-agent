from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, ToolMessage

from src.error_safety import summarize_exception
from src.runtime_services import get_current_tool_service
from src.tooling.runtime import ToolInvocation

from . import message_utils
from . import runtime as agent_runtime

logger = structlog.get_logger(__name__)


@dataclass
class ConsultantLoopResult:
    content: str
    response: object | None
    had_tool_errors: bool
    tool_failure_count: int
    tool_call_count: int = 0
    failed_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConsultantToolLoopPolicy:
    max_tool_iterations: int
    max_tool_calls_per_turn: int
    deadline: float
    total_timeout: float


InvokeWithDeadline = Callable[
    [object, list],
    Awaitable[object],
]


def remaining_consultant_budget(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _build_consultant_fmp_skip_payload(
    *, ticker: str, metric: str, failure_kind: str
) -> str:
    reason = (
        "current FMP plan does not cover this request"
        if failure_kind == "auth_error"
        else "FMP cooldown is active after a quota or rate-limit response"
    )
    return json.dumps(
        {
            "error": f"SKIPPED: spot_check_metric_alt disabled after prior FMP {failure_kind}",
            "suggestion": "Use official filings or primary-source evidence for cross-checks in this run",
            "ticker": ticker,
            "metric": metric,
            "provider": "fmp",
            "failure_kind": failure_kind,
            "retryable": failure_kind == "rate_limit",
            "skipped": True,
            "reason": reason,
        }
    )


async def run_bounded_consultant_loop(
    *,
    active_llm,
    fallback_llm,
    messages: list,
    tools_by_name: dict[str, Any],
    policy: ConsultantToolLoopPolicy,
    invoke_with_deadline: InvokeWithDeadline,
    tool_service_getter: Callable = get_current_tool_service,
    agent_name: str,
    agent_key: str,
    ticker: str,
) -> ConsultantLoopResult:
    """Run the Consultant's bounded tool loop under a shared deadline."""
    content_str = ""
    had_tool_errors = False
    tool_failure_count = 0
    tool_call_count = 0
    failed_tools: list[str] = []
    fmp_alt_disabled_kind: str | None = None
    response: object | None = None

    for iteration in range(policy.max_tool_iterations + 1):
        try:
            response = await invoke_with_deadline(active_llm, messages)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(
                "consultant_deadline_mid_loop",
                ticker=ticker,
                iteration=iteration,
                tool_failures_so_far=tool_failure_count,
            )
            break
        tool_calls = getattr(response, "tool_calls", None)
        if (
            not isinstance(tool_calls, list)
            or not tool_calls
            or iteration == policy.max_tool_iterations
        ):
            content_str = message_utils.extract_string_content(
                getattr(response, "content", "")
            )
            break

        messages.append(response)
        capped = tool_calls[: policy.max_tool_calls_per_turn]
        all_suppressed_this_iter = True
        if len(tool_calls) > policy.max_tool_calls_per_turn:
            logger.warning(
                "consultant_tool_calls_capped",
                ticker=ticker,
                requested=len(tool_calls),
                cap=policy.max_tool_calls_per_turn,
            )

        for tool_call in capped:
            tool_fn = tools_by_name.get(tool_call["name"])
            tool_call_id = tool_call.get("id", tool_call["name"])
            tool_call_count += 1
            result_failed = False
            count_failure = True
            if tool_fn:
                if (
                    tool_call["name"] == "spot_check_metric_alt"
                    and fmp_alt_disabled_kind is not None
                ):
                    result = _build_consultant_fmp_skip_payload(
                        ticker=tool_call["args"].get("ticker", ticker),
                        metric=tool_call["args"].get("metric", "unknown"),
                        failure_kind=fmp_alt_disabled_kind,
                    )
                    count_failure = False
                else:
                    if remaining_consultant_budget(policy.deadline) <= 0:
                        raise TimeoutError(
                            "Consultant node exceeded total wall-clock timeout "
                            f"of {policy.total_timeout:.0f}s for {ticker}"
                        )
                    try:

                        async def _run_tool(args: dict[str, Any], tool=tool_fn) -> Any:
                            return await tool.ainvoke(args)

                        tool_result = await tool_service_getter().execute(
                            ToolInvocation(
                                name=tool_call["name"],
                                args=tool_call["args"],
                                source="consultant",
                                agent_key=agent_key,
                            ),
                            runner=_run_tool,
                        )
                        result = tool_result.value
                    except Exception as tool_err:
                        result_failed = True
                        logger.warning(
                            "consultant_tool_failed",
                            ticker=ticker,
                            tool=tool_call["name"],
                            **summarize_exception(
                                tool_err, operation="consultant_tool_failed"
                            ),
                        )
                        result = f"TOOL_ERROR: {type(tool_err).__name__}"
            else:
                result_failed = True
                result = f"Unknown tool: {tool_call['name']}"

            if isinstance(result, str):
                stripped = result.strip()
                if stripped.startswith(("TOOL_ERROR:", "TOOL_BLOCKED:")):
                    result_failed = True
                else:
                    try:
                        payload = json.loads(stripped)
                    except (TypeError, ValueError):
                        payload = None
                    if isinstance(payload, dict) and payload.get("error"):
                        is_managed_unavailability = bool(payload.get("skipped"))
                        if (
                            tool_call["name"] == "spot_check_metric_alt"
                            and payload.get("provider") == "fmp"
                            and payload.get("failure_kind")
                            in {"auth_error", "rate_limit"}
                        ):
                            is_managed_unavailability = True
                            if fmp_alt_disabled_kind != payload.get("failure_kind"):
                                fmp_alt_disabled_kind = payload["failure_kind"]
                                logger.debug(
                                    "consultant_fmp_disabled",
                                    ticker=ticker,
                                    tool=tool_call["name"],
                                    failure_kind=fmp_alt_disabled_kind,
                                )
                        result_failed = not is_managed_unavailability
                        count_failure = not is_managed_unavailability
                        if is_managed_unavailability:
                            logger.debug(
                                "consultant_tool_suppressed",
                                ticker=ticker,
                                tool=tool_call["name"],
                                failure_kind=payload.get("failure_kind"),
                            )
            if result_failed:
                had_tool_errors = True
                if count_failure:
                    tool_failure_count += 1
                    failed_tools.append(tool_call["name"])
                all_suppressed_this_iter = False
            elif count_failure:
                all_suppressed_this_iter = False
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))

        for tool_call in tool_calls[policy.max_tool_calls_per_turn :]:
            skip_id = tool_call.get("id", f"skip_{tool_call['name']}")
            messages.append(
                ToolMessage(
                    content="SKIPPED: Too many tool calls in one turn.",
                    tool_call_id=skip_id,
                )
            )

        logger.debug(
            "consultant_tool_iteration",
            ticker=ticker,
            iteration=iteration + 1,
            tools_called=[tool_call["name"] for tool_call in capped],
        )

        if capped and all_suppressed_this_iter:
            messages.append(
                HumanMessage(
                    content=(
                        "All external verification tools requested in the last "
                        "turn were unavailable or suppressed. Provide your "
                        "final consultant review now using the evidence already "
                        "available and note any verification limits."
                    )
                )
            )
            response = await invoke_with_deadline(fallback_llm, messages)
            content_str = message_utils.extract_string_content(
                getattr(response, "content", "")
            )
            break

    # A response the provider cut off at the token cap is a fragment, not a
    # review — and a fragment is truthy, so the emptiness check above cannot
    # see it. Re-ask once (tool-free) so the model spends its budget on the
    # answer instead of on reasoning it already did.
    #
    # Cost is bounded to exactly one extra call: this runs once, after the
    # loop, and the consultant invokes with ``max_transient_attempts=1``, so
    # the runtime does not separately retry a partial before we get here. The
    # re-ask is strictly additive — any failure (deadline exhausted, provider
    # error) leaves whatever content we already had, so widening the trigger
    # from "empty" to "empty or truncated" cannot lose a usable fragment.
    truncated = agent_runtime.response_hit_output_cap(response)
    if (not content_str or truncated) and (
        tools_by_name or policy.max_tool_iterations > 0
    ):
        if truncated:
            logger.warning(
                "consultant_response_truncated_at_cap",
                ticker=ticker,
                agent=agent_name,
                partial_content_chars=len(content_str),
            )
        retry_content = ""
        retry_response: object | None = None
        try:
            retry_response = await invoke_with_deadline(fallback_llm, messages)
            retry_content = message_utils.extract_string_content(
                getattr(retry_response, "content", "")
            )
        except Exception as synthesis_exc:
            if not content_str:
                raise
            logger.warning(
                "consultant_truncation_resynthesis_failed",
                ticker=ticker,
                agent=agent_name,
                partial_content_chars=len(content_str),
                **summarize_exception(
                    synthesis_exc, operation="consultant_truncation_resynthesis"
                ),
            )
        # Never trade usable text for nothing: keep the fragment when the
        # re-ask comes back empty.
        if retry_content or not content_str:
            response = retry_response
            content_str = retry_content

    return ConsultantLoopResult(
        content=content_str,
        response=response,
        had_tool_errors=had_tool_errors,
        tool_failure_count=tool_failure_count,
        tool_call_count=tool_call_count,
        failed_tools=tuple(failed_tools),
    )
