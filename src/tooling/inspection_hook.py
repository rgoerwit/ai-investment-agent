"""TOOL_SERVICE hook adapter for content inspection.

ContentInspectionHook plugs into the ToolExecutionService hook chain and
inspects every tool output via the global INSPECTION_SERVICE singleton
before the result reaches an LLM context.

The hook operates exclusively in after() — it never raises ToolCallBlocked
because after() is called outside the before()-try/catch block in
ToolExecutionService.execute(). Instead, blocked content is returned as
a modified ToolResult with the legacy ``TOOL_BLOCKED:`` sentinel and
blocked=True.
"""

from __future__ import annotations

import structlog

from src.tooling.inspection_service import INSPECTION_SERVICE
from src.tooling.inspector import InspectionEnvelope, SourceKind
from src.tooling.runtime import ToolInvocation, ToolResult

logger = structlog.get_logger(__name__)


class ContentInspectionHook:
    """Inspect tool outputs through the global INSPECTION_SERVICE.

    Install via::

        TOOL_SERVICE.add_hook(ContentInspectionHook())
    """

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        return call  # pass-through

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        if result.blocked:
            # Already blocked by a prior hook — nothing to inspect.
            return result

        content_text = (
            result.value if isinstance(result.value, str) else str(result.value)
        )

        envelope = InspectionEnvelope(
            content_text=content_text,
            raw_content=result.value,
            source_kind=SourceKind.tool_output,
            source_name=call.name,
            tool_name=call.name,
            agent_key=call.agent_key,
            metadata={
                "source": call.source,
                "args_keys": list(call.args.keys()) if call.args else [],
            },
        )

        approved = await INSPECTION_SERVICE.check(envelope)

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
