"""Shared tool execution service for all tool-calling surfaces."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)


class ToolCallBlocked(RuntimeError):
    """Raised by a hook to block tool execution before the runner is invoked."""


@dataclass
class ToolInvocation:
    """Metadata describing a requested tool call."""

    name: str
    args: dict[str, Any]
    source: str
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
    """Execute tools through an ordered hook chain."""

    def __init__(self, hooks: list[ToolHook] | None = None) -> None:
        self._hooks: list[ToolHook] = list(hooks or [])

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
                reason=str(exc),
            )
            return ToolResult(
                value=f"TOOL_BLOCKED: {exc}",
                blocked=True,
                findings=[str(exc)],
            )

        try:
            result = ToolResult(value=await runner(call.args))
        except Exception:
            logger.error(
                "tool_call_runner_failed",
                tool=call.name,
                source=call.source,
                agent_key=call.agent_key,
                exc_info=True,
            )
            raise

        for hook in reversed(self._hooks):
            result = await hook.after(call, result)
        return result


from src.tooling.audit import LoggingToolAuditHook

TOOL_SERVICE = ToolExecutionService(hooks=[LoggingToolAuditHook()])
