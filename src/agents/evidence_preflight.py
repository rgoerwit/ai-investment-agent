"""Shared execution helper for code-owned disclosure preflights."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from src.error_safety import summarize_exception
from src.tooling.evidence_recorder import (
    EvidenceStatus,
    ExecutionStatus,
    classify_evidence_value,
)
from src.tooling.runtime import ToolInvocation

PreflightCall = tuple[str, Any, dict[str, Any]]


@dataclass(frozen=True)
class PreflightOutcome:
    """Structured execution and evidence semantics for one preflight call."""

    label: str
    execution_status: ExecutionStatus
    evidence_status: EvidenceStatus
    content: str = ""
    reason: str | None = None

    @property
    def legacy_status(self) -> str:
        if self.execution_status != "SUCCEEDED":
            return self.execution_status
        if self.evidence_status in {
            "NO_RESULTS",
            "UNAVAILABLE",
            "AUTH_ERROR",
            "INSUFFICIENT",
        }:
            return "INSUFFICIENT_DATA"
        return "COMPLETED"

    def render(self) -> str:
        lines = [
            f"STATUS: {self.legacy_status}",
            f"EXECUTION_STATUS: {self.execution_status}",
            f"EVIDENCE_STATUS: {self.evidence_status}",
        ]
        if self.reason:
            lines.append(f"REASON: {self.reason}")
        if self.content:
            lines.append(self.content)
        return "\n".join(lines)


def skipped_preflight_outcome(label: str, reason: str) -> PreflightOutcome:
    return PreflightOutcome(
        label=label,
        execution_status="SKIPPED",
        evidence_status="UNAVAILABLE",
        reason=reason,
    )


async def run_preflight_calls(
    calls: list[PreflightCall],
    *,
    agent_key: str,
    source: str,
    ticker: str,
    failure_event: str,
    logger: Any,
) -> tuple[list[PreflightOutcome], dict[str, int]]:
    """Execute independent tool calls concurrently through the runtime hook chain."""
    from src.runtime_services import get_current_tool_service

    durations_ms: dict[str, int] = {}

    async def _execute(call: PreflightCall) -> PreflightOutcome:
        label, tool, args = call
        started_at = time.monotonic()

        async def _runner(call_args: dict[str, Any]) -> Any:
            return await tool.ainvoke(call_args)

        try:
            result = await get_current_tool_service().execute(
                ToolInvocation(
                    name=tool.name,
                    args=args,
                    source=source,
                    agent_key=agent_key,
                ),
                runner=_runner,
            )
            value = str(result.value)
            execution_status, evidence_status, reason, content = (
                classify_evidence_value(
                    tool.name,
                    value,
                    blocked=bool(getattr(result, "blocked", False)),
                )
            )
            return PreflightOutcome(
                label=label,
                execution_status=execution_status,
                evidence_status=evidence_status,
                content=content,
                reason=reason,
            )
        except Exception as exc:
            logger.warning(
                failure_event,
                ticker=ticker,
                query_label=label,
                **summarize_exception(exc, operation=failure_event),
            )
            return PreflightOutcome(
                label=label,
                execution_status="FAILED",
                evidence_status="UNAVAILABLE",
                reason=type(exc).__name__.upper(),
            )
        finally:
            durations_ms[label] = round((time.monotonic() - started_at) * 1000)

    return list(await asyncio.gather(*(_execute(call) for call in calls))), durations_ms
