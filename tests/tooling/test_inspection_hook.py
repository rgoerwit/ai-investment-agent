"""Tests for ContentInspectionHook — the TOOL_SERVICE hook adapter."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.mcp.client import MCPRuntime
from src.mcp.config import MCPServerSpec
from src.runtime_services import RuntimeServices, use_runtime_services
from src.tooling.evidence_recorder import EvidenceRecorder
from src.tooling.inspection_hook import ContentInspectionHook
from src.tooling.inspection_service import INSPECTION_SERVICE, InspectionService
from src.tooling.inspector import (
    InspectionDecision,
    InspectionEnvelope,
    NullInspector,
    SourceKind,
)
from src.tooling.runtime import (
    ToolCallFailed,
    ToolExecutionService,
    ToolInvocation,
    ToolResult,
    tool_error_text,
)


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


@pytest.mark.asyncio
async def test_mcp_envelope_includes_payload_profile_and_trust_tier(tmp_path):
    captured: list[InspectionEnvelope] = []

    class CapturingInspector:
        async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
            captured.append(envelope)
            return InspectionDecision(action="allow", threat_level="safe")

    runtime = MCPRuntime(
        [
            MCPServerSpec(
                id="fmp_remote",
                description="FMP",
                transport="streamable_http",
                base_url="https://example.test/mcp",
                scopes=["consultant"],
                tool_allowlist=["ratios"],
                trust_tier="official_vendor",
            )
        ],
        budget_db_path=str(tmp_path / "mcp_usage.db"),
    )
    inspection_service = InspectionService(CapturingInspector(), mode="warn")
    services = RuntimeServices(
        tool_service=ToolExecutionService([]),
        inspection_service=inspection_service,
        mcp_runtime=runtime,
    )

    with use_runtime_services(services):
        hook = ContentInspectionHook(inspection_service)
        result = await hook.after(
            _invocation(name="mcp__fmp_remote__ratios", source="consultant"),
            ToolResult(
                value={
                    "payload_profile": "structured_financial",
                    "structured_content": {"data": [{"priceEarningsRatio": 14.2}]},
                }
            ),
        )

    assert result.blocked is False
    assert len(captured) == 1
    metadata = captured[0].metadata or {}
    assert metadata["payload_profile"] == "structured_financial"
    assert metadata["trust_tier"] == "official_vendor"
    assert metadata["transport"] == "streamable_http"


@pytest.mark.asyncio
async def test_after_serializes_dict_without_calling_str():
    captured: list[InspectionEnvelope] = []

    class Explosive:
        def __str__(self) -> str:
            raise AssertionError("unsafe __str__ should not be called")

    class CapturingInspector:
        async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
            captured.append(envelope)
            return InspectionDecision(action="allow", threat_level="safe")

    original_inspector = INSPECTION_SERVICE._inspector
    original_mode = INSPECTION_SERVICE._mode
    try:
        INSPECTION_SERVICE.configure(CapturingInspector(), mode="warn")
        hook = ContentInspectionHook()
        result = await hook.after(
            _invocation(),
            ToolResult(value={"payload": ["ok", {"nested": Explosive()}]}),
        )
        assert result.blocked is False
        assert captured
        assert "<Explosive>" in captured[0].content_text
    finally:
        INSPECTION_SERVICE.configure(original_inspector, mode=original_mode)


@pytest.mark.asyncio
async def test_after_truncates_large_content_before_inspection():
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
        huge = "x" * 60_000
        result = await hook.after(_invocation(), _result(huge))
        assert result.value == huge
        assert captured
        env = captured[0]
        assert env.metadata["original_length"] == 60_000
        assert env.metadata["truncated_for_inspection"] is True
        assert env.content_text.endswith("...[truncated for inspection]")
        assert len(env.content_text) < 60_000
    finally:
        INSPECTION_SERVICE.configure(original_inspector, mode=original_mode)


# ---------------------------------------------------------------------------
# after() branches on the enforced disposition, never on the approved text
# ---------------------------------------------------------------------------


class _Decides:
    def __init__(self, decision: InspectionDecision) -> None:
        self.decision = decision

    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
        del envelope
        return self.decision


class _Raises:
    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
        raise RuntimeError("judge down sk-secret-token-in-exception")


@pytest.mark.asyncio
async def test_a_tool_output_that_merely_looks_blocked_is_not_blocked():
    """Allowed content whose own text starts with the placeholder stays data."""
    service = InspectionService(
        _Decides(InspectionDecision(action="block", threat_level="high")), mode="warn"
    )
    value = "TOOL_BLOCKED: quoted from a filing"

    result = await ContentInspectionHook(service).after(_invocation(), _result(value))

    assert result.blocked is False
    assert result.value == value


@pytest.mark.asyncio
async def test_a_blocked_disposition_is_logged_by_code_without_reason_prose():
    service = InspectionService(
        _Decides(
            InspectionDecision(
                action="block", threat_level="high", reason="judge prose: evil text"
            )
        ),
        mode="block",
    )

    with patch("src.tooling.inspection_hook.logger") as log:
        result = await ContentInspectionHook(service).after(
            _invocation(), _result("evil")
        )

    assert result.blocked is True
    (call,) = log.warning.call_args_list
    assert call.args == ("tool_call_blocked",)
    assert call.kwargs["disposition"] == "blocked"
    assert "judge prose" not in repr(call.kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["warn", "sanitize", "block"])
async def test_a_failed_inspection_fails_the_call_and_the_ledger_records_failed(mode):
    service = InspectionService(_Raises(), mode=mode, fail_policy="fail_closed")
    recorder = EvidenceRecorder()
    tools = ToolExecutionService([recorder, ContentInspectionHook(service)])

    async def runner(args):
        del args
        return "tool output"

    with pytest.raises(ToolCallFailed) as raised:
        await tools.execute(_invocation(), runner=runner)

    assert tool_error_text(raised.value) == "TOOL_ERROR: INSPECTION_FAILED"
    (record,) = recorder.snapshot()
    assert (record.execution_status, record.reason) == ("FAILED", "INSPECTION_FAILED")
    assert "sk-secret" not in record.content


@pytest.mark.asyncio
async def test_a_fail_open_backend_error_returns_the_tool_output():
    service = InspectionService(_Raises(), mode="block", fail_policy="fail_open")

    result = await ContentInspectionHook(service).after(
        _invocation(), _result("tool output")
    )

    assert (result.value, result.blocked) == ("tool output", False)
