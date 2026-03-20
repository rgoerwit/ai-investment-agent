"""Tests for ContentInspectionHook — the TOOL_SERVICE hook adapter."""

from __future__ import annotations

import pytest

from src.tooling.inspection_hook import ContentInspectionHook
from src.tooling.inspection_service import INSPECTION_SERVICE, InspectionService
from src.tooling.inspector import (
    InspectionDecision,
    InspectionEnvelope,
    NullInspector,
    SourceKind,
)
from src.tooling.runtime import ToolInvocation, ToolResult


def _invocation(name: str = "test_tool", source: str = "toolnode") -> ToolInvocation:
    return ToolInvocation(name=name, args={}, source=source, agent_key="test_agent")


def _result(value: str = "output") -> ToolResult:
    return ToolResult(value=value)


# ---------------------------------------------------------------------------
# before() is always a pass-through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_before_passthrough():
    hook = ContentInspectionHook()
    call = _invocation()
    returned = await hook.before(call)
    assert returned is call


# ---------------------------------------------------------------------------
# after() with NullInspector — no mutation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_after_null_inspector_passthrough():
    """NullInspector → result unchanged."""
    original_inspector = INSPECTION_SERVICE._inspector
    original_mode = INSPECTION_SERVICE._mode
    try:
        INSPECTION_SERVICE.configure(NullInspector(), mode="warn")
        hook = ContentInspectionHook()
        result = await hook.after(_invocation(), _result("original"))
        assert result.value == "original"
        assert not result.blocked
    finally:
        INSPECTION_SERVICE.configure(original_inspector, mode=original_mode)


# ---------------------------------------------------------------------------
# after() in warn mode — content passes through even if flagged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_after_warn_mode_content_passes_through():
    class FlagInspector:
        async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
            return InspectionDecision(
                action="block",
                threat_level="high",
                findings=["injection"],
                reason="test",
            )

    original_inspector = INSPECTION_SERVICE._inspector
    original_mode = INSPECTION_SERVICE._mode
    try:
        INSPECTION_SERVICE.configure(FlagInspector(), mode="warn")
        hook = ContentInspectionHook()
        result = await hook.after(_invocation(), _result("dangerous"))
        # warn mode → original content returned
        assert result.value == "dangerous"
        assert not result.blocked
    finally:
        INSPECTION_SERVICE.configure(original_inspector, mode=original_mode)


# ---------------------------------------------------------------------------
# after() in block mode — TOOL_BLOCKED returned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_after_block_mode_returns_blocked_result():
    class BlockInspector:
        async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
            return InspectionDecision(
                action="block",
                threat_level="critical",
                findings=["injection"],
                reason="blocked for test",
            )

    original_inspector = INSPECTION_SERVICE._inspector
    original_mode = INSPECTION_SERVICE._mode
    try:
        INSPECTION_SERVICE.configure(BlockInspector(), mode="block")
        hook = ContentInspectionHook()
        result = await hook.after(_invocation(), _result("evil content"))
        assert result.blocked is True
        assert isinstance(result.value, str)
        assert "TOOL_BLOCKED" in result.value
    finally:
        INSPECTION_SERVICE.configure(original_inspector, mode=original_mode)


# ---------------------------------------------------------------------------
# after() skips already-blocked results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_after_skips_already_blocked():
    class AlwaysBlockInspector:
        call_count = 0

        async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
            AlwaysBlockInspector.call_count += 1
            return InspectionDecision(action="block", threat_level="high")

    original_inspector = INSPECTION_SERVICE._inspector
    original_mode = INSPECTION_SERVICE._mode
    try:
        inspector = AlwaysBlockInspector()
        INSPECTION_SERVICE.configure(inspector, mode="block")
        hook = ContentInspectionHook()
        # Pre-blocked result
        pre_blocked = ToolResult(value="TOOL_BLOCKED: earlier hook", blocked=True)
        result = await hook.after(_invocation(), pre_blocked)
        # Should return same blocked result without calling inspector
        assert result.blocked is True
        assert AlwaysBlockInspector.call_count == 0
    finally:
        INSPECTION_SERVICE.configure(original_inspector, mode=original_mode)


# ---------------------------------------------------------------------------
# after() in sanitize mode — content replaced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_after_sanitize_mode_replaces_content():
    class SanitizeInspector:
        async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
            return InspectionDecision(
                action="sanitize",
                threat_level="medium",
                sanitized_content="[safe]",
            )

    original_inspector = INSPECTION_SERVICE._inspector
    original_mode = INSPECTION_SERVICE._mode
    try:
        INSPECTION_SERVICE.configure(SanitizeInspector(), mode="sanitize")
        hook = ContentInspectionHook()
        result = await hook.after(_invocation(), _result("dirty content"))
        assert result.value == "[safe]"
        assert not result.blocked
    finally:
        INSPECTION_SERVICE.configure(original_inspector, mode=original_mode)


# ---------------------------------------------------------------------------
# InspectionEnvelope is populated correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_envelope_populated_from_invocation():
    captured: list[InspectionEnvelope] = []

    class CapturingInspector:
        async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
            captured.append(envelope)
            return InspectionDecision(action="allow", threat_level="safe")

    original_inspector = INSPECTION_SERVICE._inspector
    original_mode = INSPECTION_SERVICE._mode
    try:
        INSPECTION_SERVICE.configure(CapturingInspector(), mode="warn")
        hook = ContentInspectionHook()
        call = ToolInvocation(
            name="my_tool",
            args={"q": "search"},
            source="consultant",
            agent_key="my_agent",
        )
        await hook.after(call, _result("content"))
        assert len(captured) == 1
        env = captured[0]
        assert env.source_name == "my_tool"
        assert env.tool_name == "my_tool"
        assert env.agent_key == "my_agent"
        assert env.source_kind == SourceKind.tool_output
        assert env.content_text == "content"
    finally:
        INSPECTION_SERVICE.configure(original_inspector, mode=original_mode)
