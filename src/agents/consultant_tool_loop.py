from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, ToolMessage

from src.config import Settings, config
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
    # Provider-partial reason of the response actually being returned, after any
    # re-ask. None means the caller is holding a finished answer. The re-ask can
    # itself come back partial, error, or return nothing — in each case the
    # original fragment is retained, and without this the node saw only the
    # required headers and marked a truncated cross-check complete.
    partial_reason: str | None = None


@dataclass(frozen=True)
class ConsultantToolLoopPolicy:
    max_tool_iterations: int
    max_tool_calls_per_turn: int
    deadline: float
    total_timeout: float

    @classmethod
    def from_settings(
        cls,
        *,
        quick_mode: bool,
        deadline: float,
        total_timeout: float,
        settings: Settings = config,
    ) -> ConsultantToolLoopPolicy:
        """Build the loop budget from config, mirroring ``AuditorBudgetPolicy``.

        Quick mode pins a single tool round: that is structural, not a tuning
        choice — one round is all that fits inside
        ``consultant_quick_total_timeout_seconds``. The per-turn fan-out is the
        same in both modes because the calls run concurrently, so a wider turn
        costs the wall clock of its slowest call rather than their sum.
        """
        return cls(
            max_tool_iterations=(
                1 if quick_mode else settings.consultant_max_tool_iterations
            ),
            max_tool_calls_per_turn=settings.consultant_max_tool_calls_per_turn,
            deadline=deadline,
            total_timeout=total_timeout,
        )


@dataclass(frozen=True)
class _ToolOutcome:
    """Accounting for one executed tool call, folded back in request order."""

    tool_name: str
    result_failed: bool
    count_failure: bool
    fmp_disabled_kind: str | None = None


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


async def _execute_tool_call(
    tool_call: dict[str, Any],
    *,
    tools_by_name: dict[str, Any],
    fmp_alt_disabled_kind: str | None,
    tool_service_getter: Callable,
    agent_key: str,
    ticker: str,
) -> tuple[ToolMessage, _ToolOutcome]:
    """Run one tool call; failures are contained per task.

    Returns the ToolMessage to append plus the accounting the caller folds in
    request order, so concurrency cannot make the ledger order-dependent.
    """
    tool_fn = tools_by_name.get(tool_call["name"])
    tool_call_id = tool_call.get("id", tool_call["name"])
    result_failed = False
    count_failure = True
    disabled_kind: str | None = None

    if tool_fn:
        if tool_call["name"] == "spot_check_metric_alt" and fmp_alt_disabled_kind:
            result = _build_consultant_fmp_skip_payload(
                ticker=tool_call["args"].get("ticker", ticker),
                metric=tool_call["args"].get("metric", "unknown"),
                failure_kind=fmp_alt_disabled_kind,
            )
            count_failure = False
        else:
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
                    **summarize_exception(tool_err, operation="consultant_tool_failed"),
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
                    and payload.get("failure_kind") in {"auth_error", "rate_limit"}
                ):
                    is_managed_unavailability = True
                    if fmp_alt_disabled_kind != payload.get("failure_kind"):
                        disabled_kind = payload["failure_kind"]
                        logger.debug(
                            "consultant_fmp_disabled",
                            ticker=ticker,
                            tool=tool_call["name"],
                            failure_kind=disabled_kind,
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

    return (
        ToolMessage(content=str(result), tool_call_id=tool_call_id),
        _ToolOutcome(
            tool_name=tool_call["name"],
            result_failed=result_failed,
            count_failure=count_failure,
            fmp_disabled_kind=disabled_kind,
        ),
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

        # The deadline bounds the turn, not each call: check once, then fan out.
        # The prompt mandates plan-then-batch, so a turn is one batch and running
        # it concurrently costs the wall clock of its slowest call rather than
        # their sum (same pattern as the Auditor loop and the graph tool node).
        if remaining_consultant_budget(policy.deadline) <= 0:
            raise TimeoutError(
                "Consultant node exceeded total wall-clock timeout "
                f"of {policy.total_timeout:.0f}s for {ticker}"
            )
        # gather preserves argument order, so folding the outcomes below keeps
        # the ledger identical regardless of which task finishes first.
        executed = await asyncio.gather(
            *[
                _execute_tool_call(
                    tool_call,
                    tools_by_name=tools_by_name,
                    # Every call in a turn reads the pre-turn cooldown state.
                    # The cross-turn skip — the load-bearing half, since an FMP
                    # auth failure is sticky — is preserved by the fold below;
                    # serializing to also catch it within a turn is the defect
                    # this concurrency exists to fix.
                    fmp_alt_disabled_kind=fmp_alt_disabled_kind,
                    tool_service_getter=tool_service_getter,
                    agent_key=agent_key,
                    ticker=ticker,
                )
                for tool_call in capped
            ]
        )

        for tool_message, outcome in executed:
            tool_call_count += 1
            if outcome.fmp_disabled_kind:
                fmp_alt_disabled_kind = outcome.fmp_disabled_kind
            if outcome.result_failed:
                had_tool_errors = True
                if outcome.count_failure:
                    tool_failure_count += 1
                    failed_tools.append(outcome.tool_name)
            if outcome.result_failed or outcome.count_failure:
                all_suppressed_this_iter = False
            messages.append(tool_message)

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
    # Every recognized partial, not just the token-cap subset: a nonempty
    # fragment whose response carried provider metadata but no finish_reason is
    # equally not a finished review, and keying on the cap alone published it.
    partial_reason = agent_runtime.response_partial_reason(response)
    if (not content_str or partial_reason) and (
        tools_by_name or policy.max_tool_iterations > 0
    ):
        if partial_reason:
            logger.warning(
                "consultant_response_partial",
                ticker=ticker,
                agent=agent_name,
                partial_reason=partial_reason,
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

    # Re-evaluate against whatever is actually being returned: the re-ask may
    # have been partial too, or failed and left the original fragment in place.
    final_partial_reason = agent_runtime.response_partial_reason(response)
    if final_partial_reason:
        logger.warning(
            "consultant_response_partial_unrecovered",
            ticker=ticker,
            agent=agent_name,
            partial_reason=final_partial_reason,
            content_chars=len(content_str),
        )

    return ConsultantLoopResult(
        content=content_str,
        response=response,
        partial_reason=final_partial_reason,
        had_tool_errors=had_tool_errors,
        tool_failure_count=tool_failure_count,
        tool_call_count=tool_call_count,
        failed_tools=tuple(failed_tools),
    )
