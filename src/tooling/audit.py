"""Default logging-oriented tool audit hook."""

from __future__ import annotations

import json
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        return len(str(value))


def _serialize_preview(value: Any, limit: int = 200) -> str:
    try:
        rendered = json.dumps(value, default=str)
    except Exception:
        rendered = str(value)
    return rendered[:limit]


class LoggingToolAuditHook:
    """Log tool start/end events without mutating calls or results."""

    async def before(self, call):
        logger.info(
            "tool_call_start",
            tool=call.name,
            source=call.source,
            agent_key=call.agent_key,
            args_preview=_serialize_preview(call.args),
            args_len=_safe_len(call.args),
        )
        return call

    async def after(self, call, result):
        logger.debug(
            "tool_call_end",
            tool=call.name,
            source=call.source,
            agent_key=call.agent_key,
            blocked=result.blocked,
            findings_count=len(result.findings or []),
            output_len=_safe_len(result.value),
        )
        return result
