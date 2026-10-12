from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from src.mcp.client import MCPRuntime
from src.mcp.config import MCPAuthSpec, MCPServerSpec
from src.mcp.errors import MCPCallError, MCPErrorCategory


class _FakeSession:
    def __init__(self, result: CallToolResult | None = None):
        self._result = result
        self.calls = 0

    async def call_tool(self, name, arguments, read_timeout_seconds=None):
        assert name
        assert isinstance(arguments, dict)
        assert read_timeout_seconds is not None and read_timeout_seconds > 0
        self.calls += 1
        return self._result

    async def list_tools(self, *, params=None):
        assert params is None
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


def _serve(monkeypatch, runtime: MCPRuntime, result: CallToolResult) -> _FakeSession:
    session = _FakeSession(result)

    @asynccontextmanager
    async def fake_open_session(_spec):
        yield session

    monkeypatch.setattr(runtime, "_open_session", fake_open_session)
    return session


@pytest.mark.asyncio
async def test_execute_raw_returns_the_normalized_payload(tmp_path: Path, monkeypatch):
    runtime = _runtime(tmp_path)
    _serve(
        monkeypatch,
        runtime,
        CallToolResult(
            content=[TextContent(type="text", text='{"data": [{"price": 12.3}]}')],
            structuredContent=None,
            isError=False,
        ),
    )

    result = await runtime.execute_raw(
        "fmp_remote", "quote", {"symbol": "ABC"}, scope="consultant"
    )

    assert result["parsed_text_json"] == {"data": [{"price": 12.3}]}
    assert runtime.budget._run_calls["fmp_remote"] == 1


@pytest.mark.asyncio
async def test_an_upstream_error_payload_is_returned_and_still_counted(
    tmp_path: Path, monkeypatch
):
    runtime = _runtime(tmp_path)
    _serve(
        monkeypatch,
        runtime,
        CallToolResult(
            content=[TextContent(type="text", text='{"error": "provider said no"}')],
            structuredContent=None,
            isError=True,
        ),
    )

    result = await runtime.execute_raw(
        "fmp_remote", "quote", {"symbol": "ABC"}, scope="consultant"
    )

    assert result["is_error"] is True
    assert runtime.budget._run_calls["fmp_remote"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "scope"), [("secret_tool", "consultant"), ("quote", "auditor")]
)
async def test_a_non_allowlisted_tool_or_wrong_scope_is_refused_unsent(
    tmp_path: Path, monkeypatch, tool: str, scope: str
):
    runtime = _runtime(tmp_path)
    session = _serve(monkeypatch, runtime, CallToolResult(content=[], isError=False))

    with pytest.raises(MCPCallError) as exc_info:
        await runtime.execute_raw("fmp_remote", tool, {"symbol": "ABC"}, scope=scope)

    assert exc_info.value.category is MCPErrorCategory.CONFIG
    assert session.calls == 0
    assert runtime.budget._run_calls == {}


@pytest.mark.asyncio
async def test_open_session_streamable_http_uses_http_client_for_headers(
    tmp_path: Path, monkeypatch
):
    runtime = _runtime(tmp_path)
    runtime._resolved["fmp_remote"].headers["Authorization"] = "Bearer token"
    spec = runtime.specs["fmp_remote"]
    captured: dict[str, object] = {}

    class _FakeAsyncClient:
        def __init__(self, *, headers=None, event_hooks=None, timeout=None):
            captured["headers"] = headers
            captured["event_hooks"] = event_hooks
            captured["timeout"] = timeout

        async def __aenter__(self):
            captured["http_client"] = self
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    @asynccontextmanager
    async def fake_streamable_http_client(url, *, http_client):
        captured["url"] = url
        captured["passed_http_client"] = http_client
        yield "read", "write"

    class _FakeClientSession:
        def __init__(self, read, write, read_timeout_seconds=None):
            captured["read"] = read
            captured["write"] = write
            captured["read_timeout_seconds"] = read_timeout_seconds

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            captured["initialized"] = True

    monkeypatch.setattr("src.mcp.client.httpx2.AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(
        "src.mcp.client.streamable_http_client",
        fake_streamable_http_client,
    )
    monkeypatch.setattr("src.mcp.client.ClientSession", _FakeClientSession)

    async with runtime._open_session(spec) as session:
        assert isinstance(session, _FakeClientSession)

    assert captured["headers"] == {"Authorization": "Bearer token"}
    assert len(captured["event_hooks"]["response"]) == 1
    assert captured["passed_http_client"] is captured["http_client"]
    assert captured["url"] == "https://example.test/mcp"
    assert captured["initialized"] is True
    # The client and session bounds are set explicitly (tests/mcp/test_mcp_timeouts.py).
    assert captured["timeout"] is not None
    assert captured["read_timeout_seconds"] is not None


class TestPartialServerAvailability:
    """One vendor's absent credential must not disable the MCP plane.

    Before this, ``resolve_auth`` raised a bare ValueError on a missing key,
    ``MCPRuntime.__init__`` let it escape, and ``build_runtime_services_from_config``
    caught it and set ``mcp_runtime = None`` — so commenting out FMP_API_KEY
    turned off every MCP server, present and future.
    """

    @staticmethod
    def _spec(server_id: str, env_var: str, *, enabled: bool = True) -> MCPServerSpec:
        return MCPServerSpec(
            id=server_id,
            description=server_id,
            transport="streamable_http",
            base_url="https://example.test/mcp",
            auth=MCPAuthSpec(type="query_api_key", param="apikey", env_var=env_var),
            enabled=enabled,
            scopes=["consultant"],
            tool_allowlist=["quote"],
        )

    def test_one_missing_key_leaves_the_other_server_usable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        monkeypatch.setenv("OTHER_API_KEY", "present")
        runtime = MCPRuntime(
            servers=[
                self._spec("fmp_remote", "FMP_API_KEY"),
                self._spec("other_remote", "OTHER_API_KEY"),
            ],
            budget_db_path=str(tmp_path / "usage.db"),
        )

        assert runtime.usable_server_ids == frozenset({"other_remote"})
        assert runtime.unavailable_servers == {"fmp_remote": "FMP_API_KEY"}
        assert runtime.is_tool_available("other_remote", "quote", scope="consultant")

    def test_an_unusable_server_exposes_no_tools(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        """Enabled != usable.

        Exposure must track resolution, otherwise the tool is offered and then
        fails mid-review with a CONFIG error that counts against the
        Consultant's partial-tool-failure ratio.
        """
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        runtime = MCPRuntime(
            servers=[self._spec("fmp_remote", "FMP_API_KEY")],
            budget_db_path=str(tmp_path / "usage.db"),
        )

        assert runtime.specs["fmp_remote"].enabled is True
        assert not runtime.is_tool_available("fmp_remote", "quote", scope="consultant")

    def test_zero_usable_servers_is_a_supported_degenerate_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        """MCP on with nothing wired up must construct cleanly, not raise."""
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        runtime = MCPRuntime(
            servers=[self._spec("fmp_remote", "FMP_API_KEY")],
            budget_db_path=str(tmp_path / "usage.db"),
        )
        assert runtime.usable_server_ids == frozenset()

    def test_a_disabled_server_is_not_reported_as_missing_a_credential(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        """Deliberately off is not the same as broken."""
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        runtime = MCPRuntime(
            servers=[self._spec("fmp_remote", "FMP_API_KEY", enabled=False)],
            budget_db_path=str(tmp_path / "usage.db"),
        )
        assert runtime.unavailable_servers == {}
        assert runtime.usable_server_ids == frozenset()
