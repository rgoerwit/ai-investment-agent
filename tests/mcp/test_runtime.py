from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from src.mcp.catalog import ToolDescriptor
from src.mcp.client import MCPRuntime
from src.mcp.config import MCPServerSpec
from src.mcp.errors import MCPCallError, MCPErrorCategory
from src.tooling.inspector import InspectionDecision


class _FakeSession:
    def __init__(self, result: CallToolResult | None = None):
        self._result = result

    async def call_tool(self, name, arguments):
        assert name
        assert isinstance(arguments, dict)
        return self._result

    async def list_tools(self):
        return ListToolsResult(
            tools=[
                Tool(
                    name="quote",
                    description="Quote tool",
                    inputSchema={"type": "object"},
                    outputSchema={"type": "object"},
                )
            ]
        )


class _AllowInspector:
    async def evaluate(self, envelope):
        return InspectionDecision(
            action="allow", threat_level="safe"
        ), envelope.raw_content


class _SanitizeInspector:
    async def evaluate(self, envelope):
        return (
            InspectionDecision(
                action="sanitize",
                threat_level="medium",
                sanitized_content="sanitized text",
            ),
            "sanitized text",
        )


class _BlockInspector:
    async def evaluate(self, envelope):
        return (
            InspectionDecision(action="block", threat_level="high", reason="blocked"),
            "TOOL_BLOCKED: blocked",
        )


def _runtime(tmp_path: Path) -> MCPRuntime:
    spec = MCPServerSpec(
        id="fmp_remote",
        description="FMP",
        transport="streamable_http",
        base_url="https://example.test/mcp",
        scopes=["consultant"],
        tool_allowlist=["quote"],
        daily_call_limit=10,
        per_run_limit=10,
        trust_tier="official_vendor",
    )
    return MCPRuntime([spec], budget_db_path=str(tmp_path / "mcp_usage.db"))


@pytest.mark.asyncio
async def test_call_tool_success_returns_normalized_payload(
    tmp_path: Path, monkeypatch
):
    runtime = _runtime(tmp_path)
    runtime._inspection_service = _AllowInspector()

    @asynccontextmanager
    async def fake_open_session(_spec):
        yield _FakeSession(
            CallToolResult(
                content=[TextContent(type="text", text='{"data": [{"price": 12.3}]}')],
                structuredContent=None,
                isError=False,
            )
        )

    monkeypatch.setattr(runtime, "_open_session", fake_open_session)

    result = await runtime.call_tool(
        "fmp_remote",
        "quote",
        {"symbol": "ABC"},
        scope="consultant",
    )

    assert result["parsed_text_json"] == {"data": [{"price": 12.3}]}


@pytest.mark.asyncio
async def test_call_tool_sanitize_returns_text_payload(tmp_path: Path, monkeypatch):
    runtime = _runtime(tmp_path)
    runtime._inspection_service = _SanitizeInspector()

    @asynccontextmanager
    async def fake_open_session(_spec):
        yield _FakeSession(
            CallToolResult(
                content=[TextContent(type="text", text='{"data": [{"price": 12.3}]}')],
                structuredContent=None,
                isError=False,
            )
        )

    monkeypatch.setattr(runtime, "_open_session", fake_open_session)
    result = await runtime.call_tool(
        "fmp_remote",
        "quote",
        {"symbol": "ABC"},
        scope="consultant",
    )
    assert result["text_content"] == "sanitized text"


@pytest.mark.asyncio
async def test_call_tool_blocks_inspected_payload_and_records_budget(
    tmp_path: Path, monkeypatch
):
    runtime = _runtime(tmp_path)
    runtime._inspection_service = _BlockInspector()

    @asynccontextmanager
    async def fake_open_session(_spec):
        yield _FakeSession(
            CallToolResult(
                content=[TextContent(type="text", text='{"data": [{"price": 12.3}]}')],
                structuredContent=None,
                isError=False,
            )
        )

    monkeypatch.setattr(runtime, "_open_session", fake_open_session)

    with pytest.raises(MCPCallError) as exc_info:
        await runtime.call_tool(
            "fmp_remote",
            "quote",
            {"symbol": "ABC"},
            scope="consultant",
        )

    assert exc_info.value.category is MCPErrorCategory.INSPECTION
    assert runtime._budget._run_calls["fmp_remote"] == 1


@pytest.mark.asyncio
async def test_call_tool_rejects_non_allowlisted_tool(tmp_path: Path):
    runtime = _runtime(tmp_path)
    with pytest.raises(MCPCallError) as exc_info:
        await runtime.call_tool(
            "fmp_remote",
            "secret_tool",
            {"symbol": "ABC"},
            scope="consultant",
        )
    assert exc_info.value.category is MCPErrorCategory.CONFIG


@pytest.mark.asyncio
async def test_call_tool_rejects_wrong_scope(tmp_path: Path):
    runtime = _runtime(tmp_path)
    with pytest.raises(MCPCallError) as exc_info:
        await runtime.call_tool(
            "fmp_remote",
            "quote",
            {"symbol": "ABC"},
            scope="auditor",
        )
    assert exc_info.value.category is MCPErrorCategory.CONFIG


@pytest.mark.asyncio
async def test_call_tool_raises_for_upstream_tool_error(tmp_path: Path, monkeypatch):
    runtime = _runtime(tmp_path)
    runtime._inspection_service = _AllowInspector()

    @asynccontextmanager
    async def fake_open_session(_spec):
        yield _FakeSession(
            CallToolResult(
                content=[
                    TextContent(type="text", text='{"error": "provider said no"}')
                ],
                structuredContent=None,
                isError=True,
            )
        )

    monkeypatch.setattr(runtime, "_open_session", fake_open_session)

    with pytest.raises(MCPCallError) as exc_info:
        await runtime.call_tool(
            "fmp_remote",
            "quote",
            {"symbol": "ABC"},
            scope="consultant",
        )
    assert exc_info.value.category is MCPErrorCategory.TOOL_ERROR


@pytest.mark.asyncio
async def test_list_tools_for_scope_filters_allowlist(tmp_path: Path, monkeypatch):
    runtime = _runtime(tmp_path)

    @asynccontextmanager
    async def fake_open_session(_spec):
        yield _FakeSession()

    monkeypatch.setattr(runtime, "_open_session", fake_open_session)
    tools = await runtime.list_tools_for_scope("consultant")

    assert tools == [
        ToolDescriptor(
            server_id="fmp_remote",
            name="quote",
            description="Quote tool",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
    ]


@pytest.mark.asyncio
async def test_open_session_streamable_http_uses_http_client_for_headers(
    tmp_path: Path, monkeypatch
):
    runtime = _runtime(tmp_path)
    runtime._resolved["fmp_remote"].headers["Authorization"] = "Bearer token"
    spec = runtime.specs["fmp_remote"]
    captured: dict[str, object] = {}

    class _FakeAsyncClient:
        def __init__(self, *, headers=None, follow_redirects=True):
            captured["headers"] = headers
            captured["follow_redirects"] = follow_redirects

        async def __aenter__(self):
            captured["http_client"] = self
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    @asynccontextmanager
    async def fake_streamable_http_client(url, *, http_client):
        captured["url"] = url
        captured["passed_http_client"] = http_client
        yield "read", "write", None

    class _FakeClientSession:
        def __init__(self, read, write):
            captured["read"] = read
            captured["write"] = write

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            captured["initialized"] = True

    monkeypatch.setattr("src.mcp.client.httpx.AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(
        "src.mcp.client.streamable_http_client",
        fake_streamable_http_client,
    )
    monkeypatch.setattr("src.mcp.client.ClientSession", _FakeClientSession)

    async with runtime._open_session(spec) as session:
        assert isinstance(session, _FakeClientSession)

    assert captured["headers"] == {"Authorization": "Bearer token"}
    assert captured["follow_redirects"] is True
    assert captured["passed_http_client"] is captured["http_client"]
    assert captured["url"] == "https://example.test/mcp"
    assert captured["initialized"] is True
