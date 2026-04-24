"""Shared tool execution service for all tool-calling surfaces.

All agentic tool loops must go through ``TOOL_SERVICE`` so pre-call argument
policy and post-call untrusted-content inspection run consistently.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

import structlog

from src.error_safety import redact_sensitive_text
from src.observability import start_tool_observation
from src.runtime_diagnostics import classify_failure

logger = structlog.get_logger(__name__)

TOOL_CALL_TIMEOUT_SECONDS = 45.0

ToolSource: TypeAlias = Literal[
    "toolnode", "consultant", "editor", "legal_counsel", "auditor"
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

        try:
            with start_tool_observation(
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
            ):
                result = ToolResult(
                    value=await asyncio.wait_for(
                        runner(call.args), timeout=TOOL_CALL_TIMEOUT_SECONDS
                    )
                )
        except asyncio.TimeoutError as exc:
            logger.warning(
                "tool_call_timeout",
                tool=call.name,
                source=call.source,
                agent_key=call.agent_key,
                timeout_seconds=TOOL_CALL_TIMEOUT_SECONDS,
            )
            raise TimeoutError(
                f"Tool call '{call.name}' exceeded {TOOL_CALL_TIMEOUT_SECONDS:.1f}s"
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
                exc_info=True,
            )
            raise

        for hook in reversed(self._hooks):
            result = await hook.after(call, result)
        return result


TOOL_SERVICE = ToolExecutionService()
