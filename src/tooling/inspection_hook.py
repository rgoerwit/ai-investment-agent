"""TOOL_SERVICE hook adapter for content inspection.

ContentInspectionHook plugs into the ToolExecutionService hook chain and
inspects every tool output via an InspectionService before the result reaches
an LLM context.

The hook operates exclusively in after() — it never raises ToolCallBlocked
because after() is called outside the before()-try/catch block in
ToolExecutionService.execute(). Instead, blocked content is returned as
a modified ToolResult with the legacy ``TOOL_BLOCKED:`` sentinel and
blocked=True.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from src.mcp.errors import parse_mcp_tool_name
from src.runtime_services import (
    get_current_inspection_service,
    get_current_runtime_services,
)
from src.tooling.inspection_service import InspectionService
from src.tooling.inspector import InspectionEnvelope, SourceKind
from src.tooling.runtime import ToolInvocation, ToolResult

logger = structlog.get_logger(__name__)

_MAX_INSPECTION_INPUT_CHARS = 50_000


def _json_fallback(value: Any) -> str:
    return f"<{type(value).__name__}>"


def _serialize_for_inspection(
    value: Any,
    *,
    max_chars: int = _MAX_INSPECTION_INPUT_CHARS,
) -> tuple[str, dict[str, Any]]:
    """Serialize tool output defensively before inspection.

    Preserve the original raw value for shape retention, but keep inspection
    input bounded and avoid arbitrary ``__str__`` execution for complex objects.
    """
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict | list | tuple):
        text = json.dumps(
            value,
            ensure_ascii=False,
            default=_json_fallback,
            sort_keys=True,
        )
    elif value is None:
        text = "null"
    elif isinstance(value, int | float | bool):
        text = str(value)
    else:
        text = f"<{type(value).__name__}>"

    original_length = len(text)
    truncated = original_length > max_chars
    if truncated:
        text = text[:max_chars] + "\n...[truncated for inspection]"

    return text, {
        "original_length": original_length,
        "truncated_for_inspection": truncated,
    }


def _build_mcp_metadata(
    *,
    server_id: str,
    tool_name: str,
    result_value: Any,
    serialization_meta: dict[str, Any],
    call: ToolInvocation,
) -> dict[str, Any]:
    metadata = {
        **serialization_meta,
        "source": call.source,
        "args_keys": list(call.args.keys()) if call.args else [],
        "mcp_server_id": server_id,
        "mcp_tool_name": tool_name,
        "payload_profile": "free_text",
        "trust_tier": "unknown",
    }

    if isinstance(result_value, dict):
        payload_profile = result_value.get("payload_profile")
        if isinstance(payload_profile, str) and payload_profile:
            metadata["payload_profile"] = payload_profile

    services = get_current_runtime_services()
    runtime = services.mcp_runtime if services is not None else None
    spec = runtime.specs.get(server_id) if runtime is not None else None
    if spec is not None:
        metadata["trust_tier"] = spec.trust_tier
        metadata["transport"] = spec.transport

    return metadata


class ContentInspectionHook:
    """Inspect tool outputs through an InspectionService.

    Install via::

        TOOL_SERVICE.add_hook(ContentInspectionHook())
    """

    def __init__(self, inspection_service: InspectionService | None = None) -> None:
        self._inspection_service = inspection_service

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        return call  # pass-through

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        if result.blocked:
            # Already blocked by a prior hook — nothing to inspect.
            return result

        try:
            content_text, serialization_meta = _serialize_for_inspection(result.value)
        except Exception as exc:
            logger.warning(
                "inspection_input_serialization_failed",
                tool=call.name,
                source=call.source,
                agent_key=call.agent_key,
                error=str(exc),
            )
            return result

        mcp_parts = parse_mcp_tool_name(call.name)
        if mcp_parts is not None:
            mcp_server_id, mcp_inner_tool = mcp_parts
            source_kind = SourceKind.mcp_tool_output
            source_name = mcp_server_id
            tool_name_for_envelope = mcp_inner_tool
            metadata = _build_mcp_metadata(
                server_id=mcp_server_id,
                tool_name=mcp_inner_tool,
                result_value=result.value,
                serialization_meta=serialization_meta,
                call=call,
            )
        else:
            source_kind = SourceKind.tool_output
            source_name = call.name
            tool_name_for_envelope = call.name
            metadata = {
                "source": call.source,
                "args_keys": list(call.args.keys()) if call.args else [],
                **serialization_meta,
            }

        envelope = InspectionEnvelope(
            content_text=content_text,
            raw_content=result.value,
            source_kind=source_kind,
            source_name=source_name,
            tool_name=tool_name_for_envelope,
            agent_key=call.agent_key,
            metadata=metadata,
        )

        service = self._inspection_service or get_current_inspection_service()
        approved = await service.check(envelope)

        if approved is result.value:
            return result

        # Content was sanitized or blocked.
        blocked = isinstance(approved, str) and approved.startswith("TOOL_BLOCKED:")
        findings = result.findings or []
        if blocked:
            logger.warning(
                "tool_call_blocked",
                tool=call.name,
                source=call.source,
                agent_key=call.agent_key,
                reason=approved,
            )
            findings = findings + [approved]

        return ToolResult(
            value=approved,
            blocked=blocked,
            findings=findings if findings else None,
        )
