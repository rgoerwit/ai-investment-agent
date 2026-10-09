"""MCP deadlines: one ``Settings`` field per scope, applied at the SDK's own knobs.

Three scopes: the httpx2 client's per-phase timeout, the ``ClientSession`` read
timeout that bounds the initialize handshake and a tools/list catalog request,
and the per-request read timeout on one tool call. Each is checked for its
shipped default (read from the field definition), an override reaching the real
SDK object, a caller's enclosing deadline winning over a looser default, and a
spent allowance opening nothing. The stall tests drive the *real* mcp client
over an ``httpx2.MockTransport`` (or fake stdio streams) whose server never
answers, and prove the bound fires, the error is a retryable TRANSPORT error,
and the transport is torn down. No network, tiny timeouts only.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import httpx2
import pytest
from mcp import ClientSession
from mcp.types import REQUEST_TIMEOUT
from pydantic import ValidationError

from src.config import Settings, config
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline
from src.mcp.client import MCPRuntime
from src.mcp.config import MCPServerSpec
from src.mcp.errors import MCPCallError, MCPErrorCategory

_URL = "https://example.test/mcp"
_NOW = 1_000.0
# Generous ceiling for a stall test: the bound under test is a fraction of it.
_STALL_CEILING_SECONDS = 10.0

SHIPPED_DEFAULTS = {
    "mcp_http_timeout_seconds": 5.0,
    "mcp_initialize_timeout_seconds": 30.0,
    "mcp_tool_call_timeout_seconds": 45.0,
}


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    """A frozen monotonic clock for the deadline helpers; advance via clock[0]."""
    clock = [_NOW]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


def _http_runtime(tmp_path: Path) -> MCPRuntime:
    spec = MCPServerSpec(
        id="fmp_remote",
        description="FMP",
        transport="streamable_http",
        base_url=_URL,
        scopes=["consultant"],
        tool_allowlist=["quote"],
    )
    return MCPRuntime([spec], budget_db_path=str(tmp_path / "mcp_usage.db"))


def _stdio_runtime(tmp_path: Path) -> MCPRuntime:
    spec = MCPServerSpec(
        id="local_stdio",
        description="stdio",
        transport="stdio",
        command="fake-mcp-server",
        scopes=["consultant"],
        tool_allowlist=["quote"],
    )
    return MCPRuntime([spec], budget_db_path=str(tmp_path / "mcp_usage.db"))


class _FakeServer:
    """A streamable-HTTP MCP server behind ``httpx2.MockTransport``.

    Answers initialize, tools/list and tools/call, accepts notifications, and
    parks any method named in ``stall`` on an event that never fires, counting
    how many parked requests were cancelled by the client's teardown.
    """

    def __init__(self, *, stall: frozenset[str] = frozenset()) -> None:
        self.stall = stall
        self.methods: list[str] = []
        self.parked = 0
        self.cancelled = 0
        self.clients: list[httpx2.AsyncClient] = []

    async def handler(self, request: httpx2.Request) -> httpx2.Response:
        if request.method != "POST":
            return httpx2.Response(405)
        message = json.loads(request.content)
        method = message.get("method")
        self.methods.append(method)
        if "id" not in message:
            return httpx2.Response(202)
        if method in self.stall:
            self.parked += 1
            try:
                await asyncio.Event().wait()
            except BaseException:
                self.cancelled += 1
                raise
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {"name": "quote", "inputSchema": {"type": "object"}},
                ]
            }
        else:
            result = {"content": [{"type": "text", "text": "{}"}], "isError": False}
        return httpx2.Response(
            200, json={"jsonrpc": "2.0", "id": message["id"], "result": result}
        )

    def install(self, monkeypatch) -> None:
        """Route the runtime's own httpx2 client to this server, keeping kwargs."""
        real_client = httpx2.AsyncClient

        def _client(**kwargs):
            client = real_client(transport=httpx2.MockTransport(self.handler), **kwargs)
            self.clients.append(client)
            return client

        monkeypatch.setattr("src.mcp.client.httpx2.AsyncClient", _client)


class _RecordingSession(ClientSession):
    """The real ClientSession, recording the timeouts it is handed."""

    session_timeouts: list[float | None] = []
    call_timeouts: list[float | None] = []

    def __init__(self, read, write, read_timeout_seconds=None, **kwargs):
        type(self).session_timeouts.append(read_timeout_seconds)
        super().__init__(read, write, read_timeout_seconds, **kwargs)

    async def call_tool(self, name, arguments=None, read_timeout_seconds=None, **kw):
        type(self).call_timeouts.append(read_timeout_seconds)
        return await super().call_tool(name, arguments, read_timeout_seconds, **kw)


@pytest.fixture
def recording_session(monkeypatch) -> type[_RecordingSession]:
    _RecordingSession.session_timeouts = []
    _RecordingSession.call_timeouts = []
    monkeypatch.setattr("src.mcp.client.ClientSession", _RecordingSession)
    return _RecordingSession


def _other_tasks(before: set[asyncio.Task]) -> set[asyncio.Task]:
    return {
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and task not in before
    }


# --- Shipped defaults ------------------------------------------------------


@pytest.mark.parametrize("field, expected", sorted(SHIPPED_DEFAULTS.items()))
def test_shipped_default_is_unchanged(field, expected):
    assert Settings.model_fields[field].default == expected


def test_http_default_is_the_timeout_the_client_had_implicitly():
    # Before it was a setting the client passed no timeout, so it ran on
    # httpx2's own default. The setting must reproduce exactly that.
    implicit = httpx2.AsyncClient().timeout
    shipped = Settings.model_fields["mcp_http_timeout_seconds"].default
    assert implicit == httpx2.Timeout(shipped)


def test_tool_call_default_matches_the_tool_runner_default():
    # Under the runner the call's own bound must never fire first by default.
    assert (
        Settings.model_fields["mcp_tool_call_timeout_seconds"].default
        == Settings.model_fields["tool_execution_timeout_seconds"].default
    )


@pytest.mark.parametrize("field", sorted(SHIPPED_DEFAULTS))
def test_a_non_positive_override_is_rejected(monkeypatch, field):
    monkeypatch.setenv(Settings.model_fields[field].validation_alias, "0")
    with pytest.raises(ValidationError):
        Settings()


# --- Overrides reach the real SDK objects ----------------------------------


@pytest.mark.asyncio
async def test_overrides_reach_the_client_session_and_call(
    tmp_path, monkeypatch, recording_session
):
    server = _FakeServer()
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_http_timeout_seconds", 7.0)
    monkeypatch.setattr(config, "mcp_initialize_timeout_seconds", 3.0)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", 9.0)
    runtime = _http_runtime(tmp_path)

    payload = await asyncio.wait_for(
        runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
        timeout=_STALL_CEILING_SECONDS,
    )

    assert payload["is_error"] is False
    (client,) = server.clients
    assert client.timeout == httpx2.Timeout(7.0)
    assert client.is_closed
    assert recording_session.session_timeouts == [3.0]
    # Explicit, not inherited from the session's handshake bound.
    assert recording_session.call_timeouts == [9.0]


@pytest.mark.asyncio
async def test_catalog_request_carries_the_initialize_bound(
    tmp_path, monkeypatch, recording_session
):
    server = _FakeServer()
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_initialize_timeout_seconds", 4.0)
    runtime = _http_runtime(tmp_path)

    tools = await asyncio.wait_for(
        runtime.list_tools_for_scope("consultant"), timeout=_STALL_CEILING_SECONDS
    )

    assert [tool.name for tool in tools] == ["quote"]
    assert recording_session.session_timeouts == [4.0]
    assert recording_session.call_timeouts == []


# --- The caller's enclosing deadline ---------------------------------------


@pytest.mark.asyncio
async def test_a_nested_caller_deadline_wins_over_every_default(
    tmp_path, monkeypatch, recording_session, fake_clock
):
    server = _FakeServer()
    server.install(monkeypatch)
    runtime = _http_runtime(tmp_path)

    with enclosing_deadline(2.0):
        await asyncio.wait_for(
            runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
            timeout=_STALL_CEILING_SECONDS,
        )

    (client,) = server.clients
    assert client.timeout == httpx2.Timeout(2.0)
    assert recording_session.session_timeouts == [2.0]
    assert recording_session.call_timeouts == [2.0]


@pytest.mark.asyncio
async def test_a_looser_caller_deadline_leaves_the_defaults(
    tmp_path, monkeypatch, recording_session, fake_clock
):
    server = _FakeServer()
    server.install(monkeypatch)
    runtime = _http_runtime(tmp_path)

    with enclosing_deadline(600.0):
        await asyncio.wait_for(
            runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
            timeout=_STALL_CEILING_SECONDS,
        )

    (client,) = server.clients
    assert client.timeout == httpx2.Timeout(
        SHIPPED_DEFAULTS["mcp_http_timeout_seconds"]
    )
    assert recording_session.session_timeouts == [
        SHIPPED_DEFAULTS["mcp_initialize_timeout_seconds"]
    ]
    assert recording_session.call_timeouts == [
        SHIPPED_DEFAULTS["mcp_tool_call_timeout_seconds"]
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_factory", [_http_runtime, _stdio_runtime])
async def test_a_spent_allowance_opens_no_session(
    tmp_path, monkeypatch, fake_clock, runtime_factory
):
    def _forbidden(*args, **kwargs):
        raise AssertionError("no transport may be created once the deadline passed")

    monkeypatch.setattr("src.mcp.client.httpx2.AsyncClient", _forbidden)
    monkeypatch.setattr("src.mcp.client.stdio_client", _forbidden)
    monkeypatch.setattr("src.mcp.client.ClientSession", _forbidden)
    runtime = runtime_factory(tmp_path)
    (server_id,) = runtime.specs

    with enclosing_deadline(5.0):
        fake_clock[0] += 5.0
        with pytest.raises(MCPCallError) as exc_info:
            await runtime.execute_raw(server_id, "quote", {}, scope="consultant")

    err = exc_info.value
    assert err.category is MCPErrorCategory.TRANSPORT
    assert err.retryable is False
    assert err.details == {
        "timeout": "enclosing_deadline_spent",
        "scope": "mcp_initialize",
    }


@pytest.mark.asyncio
async def test_an_allowance_spent_while_opening_sends_no_call(
    tmp_path, monkeypatch, fake_clock
):
    runtime = _http_runtime(tmp_path)
    opened: list[str] = []
    closed: list[str] = []
    sent: list[str] = []

    class _Session:
        async def call_tool(self, name, *args, **kwargs):  # pragma: no cover
            # Recorded, not asserted here: the runtime would reclassify an
            # AssertionError raised inside the session into an MCPCallError.
            sent.append(name)
            raise RuntimeError("the call must not be sent")

    @asynccontextmanager
    async def slow_open(spec):
        opened.append(spec.id)
        fake_clock[0] += 5.0  # the handshake used the caller's whole allowance
        try:
            yield _Session()
        finally:
            closed.append(spec.id)

    monkeypatch.setattr(runtime, "_open_session", slow_open)

    with enclosing_deadline(5.0):
        with pytest.raises(MCPCallError) as exc_info:
            await runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant")

    assert sent == []
    err = exc_info.value
    assert err.category is MCPErrorCategory.TRANSPORT
    assert err.retryable is False
    assert err.tool_name == "quote"
    assert err.details["scope"] == "mcp_tool_call"
    # One session, closed normally, and no retry of a call that was never sent.
    assert opened == closed == ["fmp_remote"]


# --- Stalls are bounded, classified, and torn down -------------------------


@pytest.mark.asyncio
async def test_a_stalled_http_initialize_returns_within_its_bound(
    tmp_path, monkeypatch
):
    server = _FakeServer(stall=frozenset({"initialize"}))
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_initialize_timeout_seconds", 0.2)
    runtime = _http_runtime(tmp_path)
    spec = runtime.specs["fmp_remote"]
    before = set(asyncio.all_tasks())

    started = time.monotonic()
    with pytest.raises(MCPCallError) as exc_info:
        async with runtime._open_session(spec):
            pytest.fail("initialize must not succeed")  # pragma: no cover
    elapsed = time.monotonic() - started

    err = exc_info.value
    assert err.category is MCPErrorCategory.TRANSPORT
    assert err.retryable is True
    assert err.json_rpc_code == REQUEST_TIMEOUT
    assert 0.2 <= elapsed < _STALL_CEILING_SECONDS
    # Torn down: the parked request was cancelled, the client closed, and no
    # transport task outlives the session.
    assert server.parked == server.cancelled == 1
    (client,) = server.clients
    assert client.is_closed
    assert _other_tasks(before) == set()


@pytest.mark.asyncio
async def test_a_stalled_http_tool_call_returns_within_its_bound(tmp_path, monkeypatch):
    server = _FakeServer(stall=frozenset({"tools/call"}))
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", 0.2)
    runtime = _http_runtime(tmp_path)
    before = set(asyncio.all_tasks())

    started = time.monotonic()
    with pytest.raises(MCPCallError) as exc_info:
        await runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant")
    elapsed = time.monotonic() - started

    err = exc_info.value
    assert err.category is MCPErrorCategory.TRANSPORT
    assert err.retryable is True
    assert err.json_rpc_code == REQUEST_TIMEOUT
    # Two attempts of 0.2 s plus the 0.5 s backoff between them.
    assert 0.4 <= elapsed < _STALL_CEILING_SECONDS
    assert server.methods.count("tools/call") == 2
    # Each timed-out call told the server to stop, and was torn down.
    assert server.methods.count("notifications/cancelled") == 2
    assert server.parked == server.cancelled == 2
    assert len(server.clients) == 2
    assert all(client.is_closed for client in server.clients)
    assert _other_tasks(before) == set()


@pytest.mark.asyncio
async def test_a_stalled_catalog_request_returns_within_the_initialize_bound(
    tmp_path, monkeypatch
):
    server = _FakeServer(stall=frozenset({"tools/list"}))
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_initialize_timeout_seconds", 0.2)
    runtime = _http_runtime(tmp_path)

    with pytest.raises(MCPCallError) as exc_info:
        await asyncio.wait_for(
            runtime.list_tools_for_scope("consultant"),
            timeout=_STALL_CEILING_SECONDS,
        )

    assert exc_info.value.category is MCPErrorCategory.TRANSPORT
    assert exc_info.value.json_rpc_code == REQUEST_TIMEOUT
    assert server.parked == server.cancelled == 1
    assert all(client.is_closed for client in server.clients)


@pytest.mark.asyncio
async def test_a_stalled_stdio_initialize_returns_and_closes_the_transport(
    tmp_path, monkeypatch
):
    """A stdio server that never answers: the real session over fake pipes.

    The real ``stdio_client`` owns process shutdown (shielded, bounded); what
    this proves is that our side exits its context, so that shutdown runs.
    """
    transport_closed: list[bool] = []
    written: list[str] = []

    @asynccontextmanager
    async def silent_stdio_client(params):
        assert params.command == "fake-mcp-server"
        to_client, read = anyio.create_memory_object_stream(10)
        write, from_client = anyio.create_memory_object_stream(10)
        try:
            yield read, write
        finally:
            while True:
                try:
                    written.append(from_client.receive_nowait().message.method)
                except (anyio.WouldBlock, anyio.EndOfStream):
                    break
            for stream in (to_client, read, write, from_client):
                await stream.aclose()
            transport_closed.append(True)

    monkeypatch.setattr("src.mcp.client.stdio_client", silent_stdio_client)
    monkeypatch.setattr(config, "mcp_initialize_timeout_seconds", 0.2)
    runtime = _stdio_runtime(tmp_path)
    spec = runtime.specs["local_stdio"]

    started = time.monotonic()
    with pytest.raises(MCPCallError) as exc_info:
        async with runtime._open_session(spec):
            pytest.fail("initialize must not succeed")  # pragma: no cover
    elapsed = time.monotonic() - started

    err = exc_info.value
    assert err.category is MCPErrorCategory.TRANSPORT
    assert err.retryable is True
    assert err.json_rpc_code == REQUEST_TIMEOUT
    assert 0.2 <= elapsed < _STALL_CEILING_SECONDS
    assert written[:1] == ["initialize"]
    assert transport_closed == [True]
