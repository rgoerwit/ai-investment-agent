"""End-to-end test that the consultant MCP wrappers route through
``ToolExecutionService`` so audit, argument-policy and content-inspection
hooks all observe the call under its canonical ``mcp__<server>__<tool>`` name,
and that the MCP runtime counts and gates its budget beneath them.

This is the load-bearing assertion for R-2: the MCP path must not bypass the
shared hook chain.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from src.consultant_tools import spot_check_metric_mcp_fmp
from src.mcp.client import MCPRuntime
from src.mcp.config import MCPServerSpec
from src.runtime_services import RuntimeServices, use_runtime_services
from src.tooling.inspection_hook import ContentInspectionHook
from src.tooling.inspection_service import InspectionService
from src.tooling.inspector import InspectionDecision, InspectionEnvelope, SourceKind
from src.tooling.runtime import (
    ToolExecutionService,
    ToolHook,
    ToolInvocation,
    ToolResult,
)


class _RecordingHook:
    """A pass-through hook that records the canonical tool names it sees."""

    def __init__(self) -> None:
        self.before_calls: list[str] = []
        self.after_calls: list[tuple[str, bool]] = []

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        self.before_calls.append(call.name)
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        self.after_calls.append((call.name, result.blocked))
        return result


class _RecordingInspector:
    def __init__(self) -> None:
        self.envelopes: list[InspectionEnvelope] = []

    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
        self.envelopes.append(envelope)
        return InspectionDecision(action="allow", threat_level="safe")


def _runtime_with_streaming(tmp_path: Path) -> MCPRuntime:
    spec = MCPServerSpec(
        id="fmp_remote",
        description="FMP",
        transport="streamable_http",
        base_url="https://example.test/mcp",
        scopes=["consultant"],
        tool_allowlist=["statements"],
        daily_call_limit=10,
        per_run_limit=10,
        trust_tier="official_vendor",
    )
    return MCPRuntime([spec], budget_db_path=str(tmp_path / "mcp_usage.db"))


@pytest.mark.asyncio
async def test_consultant_mcp_call_flows_through_tool_service(
    tmp_path: Path, monkeypatch
):
    runtime = _runtime_with_streaming(tmp_path)

    @asynccontextmanager
    async def fake_open_session(_spec):
        class _Session:
            async def list_tools(self, *, params=None):
                # Listed before every call, so the output schema is cached
                # within the time left (src/mcp/client.py::_call_in_time).
                return ListToolsResult(
                    tools=[Tool(name="statements", inputSchema={"type": "object"})]
                )

            async def call_tool(self, name, arguments, read_timeout_seconds=None):
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text='{"data": [{"priceEarningsRatio": 14.2}]}',
                        )
                    ],
                    structuredContent=None,
                    isError=False,
                )

        yield _Session()

    monkeypatch.setattr(runtime, "_open_session", fake_open_session)

    # A real hook chain: an inspector hook plus a recording probe so we can
    # assert on the canonical tool name.
    inspector = _RecordingInspector()
    inspection_service = InspectionService(inspector=inspector)
    recording_hook: ToolHook = _RecordingHook()
    hooks: list[ToolHook] = [
        recording_hook,
        ContentInspectionHook(inspection_service),
    ]
    tool_service = ToolExecutionService(hooks)

    services = RuntimeServices(
        tool_service=tool_service,
        inspection_service=inspection_service,
        mcp_runtime=runtime,
    )

    with use_runtime_services(services):
        # Make get_consultant_tools-style sanity gate happy and exercise the wrapper.
        with patch(
            "src.consultant_tools.get_current_runtime_services", return_value=services
        ):
            import json

            result = json.loads(
                await spot_check_metric_mcp_fmp.ainvoke(
                    {"ticker": "ABC", "metric": "trailingPE"}
                )
            )

    assert result["value"] == 14.2

    # The hook chain must have observed the canonical mcp__server__tool name.
    assert recording_hook.before_calls == ["mcp__fmp_remote__statements"]
    assert recording_hook.after_calls == [("mcp__fmp_remote__statements", False)]

    # ContentInspectionHook must have built an envelope with the MCP source kind
    # and the inner tool name (stripped of the mcp__server__ prefix).
    assert len(inspector.envelopes) == 1
    envelope = inspector.envelopes[0]
    assert envelope.source_kind is SourceKind.mcp_tool_output
    assert envelope.source_name == "fmp_remote"
    assert envelope.tool_name == "statements"
    assert envelope.metadata["payload_profile"] == "structured_financial"
    assert envelope.metadata["trust_tier"] == "official_vendor"
    assert envelope.metadata["transport"] == "streamable_http"

    # The runtime counted the one tools/call it sent.
    assert runtime.budget._run_calls["fmp_remote"] == 1


@pytest.mark.asyncio
async def test_consultant_mcp_call_blocked_by_budget_returns_failure(
    tmp_path: Path, monkeypatch
):
    runtime = _runtime_with_streaming(tmp_path)
    # Pre-saturate the per-run limit so the runtime refuses before any session.
    spec = runtime.specs["fmp_remote"]
    runtime.budget._run_calls["fmp_remote"] = spec.per_run_limit

    @asynccontextmanager
    async def fake_open_session(_spec):  # pragma: no cover - never invoked when blocked
        raise AssertionError("no session may open once the budget is spent")
        yield None

    monkeypatch.setattr(runtime, "_open_session", fake_open_session)

    inspection_service = InspectionService()
    tool_service = ToolExecutionService([])
    services = RuntimeServices(
        tool_service=tool_service,
        inspection_service=inspection_service,
        mcp_runtime=runtime,
    )

    with use_runtime_services(services):
        with patch(
            "src.consultant_tools.get_current_runtime_services", return_value=services
        ):
            import json

            result = json.loads(
                await spot_check_metric_mcp_fmp.ainvoke(
                    {"ticker": "ABC", "metric": "trailingPE"}
                )
            )

    assert result["provider"] == "fmp"
    assert result["source"] == "fmp_mcp"
    assert result["failure_kind"] == "budget"
    assert "budget exhausted" in result["error"].lower()
