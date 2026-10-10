"""Shared tool execution service for all tool-calling surfaces.

All agentic tool loops must go through ``TOOL_SERVICE`` so pre-call argument
policy and post-call untrusted-content inspection run consistently.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

import structlog

from src.async_utils import run_with_hard_timeout
from src.config import config as settings_config
from src.error_safety import redact_sensitive_text
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import clamp_to_enclosing_deadline
from src.observability import start_tool_observation
from src.runtime_diagnostics import classify_failure

logger = structlog.get_logger(__name__)

# The absolute deadline (on ``llm_timeouts.now``) of the tool call in progress,
# so a tool call made from inside another one -- an MCP spot check routed back
# through the service by a Consultant tool -- clamps to what the outer call has
# left instead of opening a fresh allowance. Deliberately not the LLM
# enclosing deadline: binding that would also cap the injection judge in the
# after() hooks, and the judge fails *open* when admission refuses it, where
# this scope's own timeout fails closed. Only the tool plane reads this one.
_TOOL_CALL_DEADLINE: ContextVar[float | None] = ContextVar(
    "tool_call_deadline", default=None
)


def _clamp_to_tool_call(default_seconds: float) -> float:
    """``default_seconds`` shrunk to the caller's deadline and any outer tool call's."""
    seconds = clamp_to_enclosing_deadline(default_seconds)
    outer = _TOOL_CALL_DEADLINE.get()
    if outer is None:
        return seconds
    return min(seconds, outer - llm_timeouts.now())


def tool_execution_timeout_seconds() -> float:
    """The runner's allowance for this call: the setting, clamped to the caller."""
    return _clamp_to_tool_call(settings_config.tool_execution_timeout_seconds)


def tool_execution_with_inspection_timeout_seconds() -> float:
    """This call's allowance for the hooks *and* the runner, clamped to the caller.

    The one policy for the combined scope: ``ToolExecutionService.execute``
    bounds every call with it, and the graph tool node reads the same function
    for its own wrapper so the two cannot drift. A result ``<= 0`` means the
    allowance is already spent and the call must fail as a timeout unstarted.
    """
    return _clamp_to_tool_call(
        settings_config.tool_execution_with_inspection_timeout_seconds
    )


@contextmanager
def tool_call_deadline(seconds: float) -> Iterator[float]:
    """Bind ``now + seconds`` as the tool-call deadline for the block.

    Nested bindings can only shorten it. A task created inside the block (the
    hard-timeout wrapper's inner task) copies the binding, so every tool call
    it makes sees the outer deadline.
    """
    deadline = llm_timeouts.now() + float(seconds)
    outer = _TOOL_CALL_DEADLINE.get()
    bound = deadline if outer is None else min(deadline, outer)
    token = _TOOL_CALL_DEADLINE.set(bound)
    try:
        yield bound
    finally:
        _TOOL_CALL_DEADLINE.reset(token)


class _RunnerTimeout(TimeoutError):
    """The runner's own bound fired (already logged); not the combined scope."""


ToolSource: TypeAlias = Literal[
    "toolnode", "preflight", "consultant", "editor", "legal_counsel", "auditor"
]


class ToolCallBlocked(RuntimeError):
    """Raised by a hook to block tool execution before the runner is invoked."""


@dataclass
class ToolInvocation:
    """Metadata describing a requested tool call."""

    name: str
    args: dict[str, Any]
    source: ToolSource
    agent_key: str | None = None


@dataclass
class ToolResult:
    """Execution result plus optional audit findings."""

    value: Any
    blocked: bool = False
    findings: list[str] | None = None


class ToolHook(Protocol):
    """Hook contract for observing or changing tool execution."""

    async def before(self, call: ToolInvocation) -> ToolInvocation: ...

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult: ...


class ToolExecutionService:
    """Execute tools through an ordered hook chain.

    This is the mandatory execution plane for prompt-bound tool outputs.
    """

    def __init__(self, hooks: list[ToolHook] | None = None) -> None:
        self._hooks: list[ToolHook] = list(hooks or [])

    def with_hooks(self, hooks: list[ToolHook]) -> ToolExecutionService:
        """Return a new service with the provided hook chain."""
        return ToolExecutionService(hooks)

    def with_extra_hooks(self, hooks: list[ToolHook]) -> ToolExecutionService:
        """Return a new service with additional hooks appended."""
        return ToolExecutionService([*self._hooks, *hooks])

    def set_hooks(self, hooks: list[ToolHook]) -> None:
        self._hooks = list(hooks)

    def add_hook(self, hook: ToolHook) -> None:
        self._hooks.append(hook)

    def clear_hooks(self) -> None:
        self._hooks.clear()

    @property
    def hooks(self) -> list[ToolHook]:
        return list(self._hooks)

    async def execute(
        self,
        call: ToolInvocation,
        runner: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> ToolResult:
        """Run ``before()`` hooks, the runner and ``after()`` hooks under one bound.

        The combined bound (``tool_execution_with_inspection_timeout_seconds``)
        is deadline-only: when it fires the caller gets ``TimeoutError`` at
        once and the late result, inspected or not, is discarded with the
        orphaned task. The runner keeps its own narrower bound inside it.
        """
        allowance = tool_execution_with_inspection_timeout_seconds()
        if allowance <= 0:
            # The caller's allowance is already spent: run no hook, no tool.
            logger.warning(
                "tool_call_timeout",
                tool=call.name,
                source=call.source,
                agent_key=call.agent_key,
                timeout_seconds=allowance,
                scope="with_inspection",
            )
            raise TimeoutError(
                f"Tool call '{call.name}' not started: allowance already spent"
            )
        # Bound before the inner task exists, so it (and any tool call made
        # from inside it) inherits this call's deadline.
        with tool_call_deadline(allowance):
            try:
                return await run_with_hard_timeout(
                    self._execute_hooked(call, runner),
                    timeout=allowance,
                    label=f"tool_with_hooks:{call.name}",
                )
            except _RunnerTimeout:
                raise
            except asyncio.TimeoutError as exc:
                logger.warning(
                    "tool_call_timeout",
                    tool=call.name,
                    source=call.source,
                    agent_key=call.agent_key,
                    timeout_seconds=allowance,
                    scope="with_inspection",
                )
                raise TimeoutError(
                    f"Tool call '{call.name}' with its hooks exceeded {allowance:.1f}s"
                ) from exc

    async def _execute_hooked(
        self,
        call: ToolInvocation,
        runner: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> ToolResult:
        try:
            for hook in self._hooks:
                call = await hook.before(call)
        except ToolCallBlocked as exc:
            logger.warning(
                "tool_call_blocked",
                tool=call.name,
                source=call.source,
                agent_key=call.agent_key,
                reason=redact_sensitive_text(str(exc)),
            )
            return ToolResult(
                value=f"TOOL_BLOCKED: {exc}",
                blocked=True,
                findings=[redact_sensitive_text(str(exc))],
            )

        # Read after the before() hooks, so time they spent counts against a
        # caller deadline rather than extending it.
        timeout_seconds = tool_execution_timeout_seconds()
        try:
            if timeout_seconds <= 0:
                # The caller's allowance is already spent: never start the tool.
                raise asyncio.TimeoutError("enclosing deadline already passed")
            with (
                start_tool_observation(
                    tool_name=call.name,
                    input_payload={
                        "arg_keys": sorted(call.args.keys()),
                        "ticker": call.args.get("ticker"),
                    },
                    metadata={
                        "tool_name": call.name,
                        "tool_source": call.source,
                        "agent_key": call.agent_key,
                    },
                ),
                tool_call_deadline(timeout_seconds),
            ):
                # The runner's deadline is bound too: a tool call the runner
                # makes (an MCP spot check routed back through this service)
                # gets what this runner has left, never a fresh allowance.
                result = ToolResult(
                    value=await run_with_hard_timeout(
                        runner(call.args),
                        timeout=timeout_seconds,
                        label=f"tool:{call.name}",
                    )
                )
        except asyncio.TimeoutError as exc:
            logger.warning(
                "tool_call_timeout",
                tool=call.name,
                source=call.source,
                agent_key=call.agent_key,
                timeout_seconds=timeout_seconds,
            )
            raise _RunnerTimeout(
                f"Tool call '{call.name}' exceeded {max(timeout_seconds, 0.0):.1f}s"
            ) from exc
        except Exception as exc:
            # after() hooks only run for successfully produced tool outputs.
            # Failed executions propagate immediately so callers keep the original
            # stack and error semantics.
            details = classify_failure(exc, provider="unknown")
            logger.error(
                "tool_call_runner_failed",
                tool=call.name,
                source=call.source,
                agent_key=call.agent_key,
                failure_kind=details.kind,
                retryable=details.retryable,
                host=details.host,
                error_type=details.error_type,
                root_cause_type=details.root_cause_type,
                error_message=details.message,
            )
            raise

        deadline = _TOOL_CALL_DEADLINE.get()
        for hook in reversed(self._hooks):
            if deadline is not None and deadline - llm_timeouts.now() <= 0:
                # Spent before this hook could start: discard the result rather
                # than return it uninspected, start a judge nobody will wait
                # for, or -- in an orphan whose hook outlived cancellation --
                # let a later hook record a result the caller never received.
                raise TimeoutError(
                    f"Tool call '{call.name}': no allowance left for its hooks"
                )
            result = await hook.after(call, result)
        return result


TOOL_SERVICE = ToolExecutionService()
