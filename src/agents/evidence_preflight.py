"""Shared execution helper for code-owned disclosure preflights."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.error_safety import summarize_exception
from src.tooling.runtime import ToolInvocation

PreflightCall = tuple[str, Any, dict[str, Any]]
PreflightOutcome = tuple[str, str]


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
            if getattr(result, "blocked", False):
                status = "BLOCKED"
            elif value.strip().upper().startswith("STATUS: INSUFFICIENT_DATA"):
                status = "INSUFFICIENT_DATA"
            else:
                status = "COMPLETED"
            return label, f"STATUS: {status}\n{value}"
        except Exception as exc:
            logger.warning(
                failure_event,
                ticker=ticker,
                query_label=label,
                **summarize_exception(exc, operation=failure_event),
            )
            return label, f"STATUS: FAILED ({type(exc).__name__})"
        finally:
            durations_ms[label] = round((time.monotonic() - started_at) * 1000)

    return list(await asyncio.gather(*(_execute(call) for call in calls))), durations_ms
