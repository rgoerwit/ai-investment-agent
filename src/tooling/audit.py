"""Default logging-oriented tool audit hook."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from src.error_safety import redact_sensitive_text

if TYPE_CHECKING:
    from src.tooling.runtime import ToolInvocation, ToolResult

logger = structlog.get_logger(__name__)
_ERROR_PREFIXES = (
    "TOOL_ERROR:",
    "TOOL_BLOCKED:",
    "FETCH_FAILED:",
    "SEARCH_FAILED:",
)


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return len(str(value))


def _extract_error_like_details(value: Any) -> dict[str, Any] | None:
    """Extract failure metadata from legacy string prefixes or JSON error payloads.

    Supported string prefixes are:
    - ``TOOL_ERROR:``
    - ``TOOL_BLOCKED:``
    - ``FETCH_FAILED:``
    - ``SEARCH_FAILED:``

    These prefixes are the compatibility contract for older manual-tool paths that
    still return string sentinel values instead of structured error payloads.
    """
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    if text.startswith(_ERROR_PREFIXES):
        prefix, _, detail = text.partition(":")
        return {
            "failure_kind": prefix.lower(),
            "message": redact_sensitive_text(detail.strip(), max_chars=64),
        }

    try:
        payload = json.loads(text)
    except Exception:
        return None

    if isinstance(payload, dict) and payload.get("error"):
        return {
            "failure_kind": payload.get("failure_kind", "tool_error"),
            "retryable": payload.get("retryable"),
            "provider": payload.get("provider"),
            "endpoint": payload.get("fmp_endpoint") or payload.get("endpoint"),
            "message": redact_sensitive_text(str(payload.get("error")), max_chars=64),
        }
    return None


class LoggingToolAuditHook:
    """Log tool start/end events without mutating calls or results."""

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        logger.info(
            "tool_call_start",
            tool=call.name,
            source=call.source,
            agent_key=call.agent_key,
            arg_keys=sorted(call.args.keys()),
            args_len=_safe_len(call.args),
        )
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        error_details = _extract_error_like_details(result.value)
        logger.debug(
            "tool_call_end",
            tool=call.name,
            source=call.source,
            agent_key=call.agent_key,
            blocked=result.blocked,
            findings_count=len(result.findings or []),
            output_len=_safe_len(result.value),
        )
        if error_details:
            logger.warning(
                "tool_call_result_error",
                tool=call.name,
                source=call.source,
                agent_key=call.agent_key,
                failure_kind=error_details.get("failure_kind"),
                retryable=error_details.get("retryable"),
                provider=error_details.get("provider"),
                endpoint=error_details.get("endpoint"),
                message=error_details.get("message"),
            )
        return result
