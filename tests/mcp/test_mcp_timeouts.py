"""MCP deadlines: one ``Settings`` field per scope, applied at the SDK's own knobs.

Three scopes: the httpx2 client's per-phase timeout, the ``ClientSession`` read
timeout that bounds the initialize handshake,
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
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import httpx2
import pytest
from mcp.types import REQUEST_TIMEOUT
from pydantic import ValidationError

import src.mcp.client as mcp_client
from src.config import Settings, config
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline
from src.mcp.client import MCPRuntime
from src.mcp.config import MCPServerSpec
from src.mcp.errors import MCPCallError, MCPErrorCategory
from tests.helpers.mcp_server import (
    FakeMCPServer,
    RecordingSession,
    http_runtime,
    other_tasks,
)

_NOW = 1_000.0
# Generous ceiling for a stall test: the bound under test is a fraction of it.
_STALL_CEILING_SECONDS = 10.0

SHIPPED_DEFAULTS = {
    "mcp_http_timeout_seconds": 5.0,
    "mcp_initialize_timeout_seconds": 30.0,
}


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    """A frozen monotonic clock for the deadline helpers; advance via clock[0]."""
    clock = [_NOW]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


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


@pytest.fixture
def recording_session(monkeypatch) -> type[RecordingSession]:
    RecordingSession.reset()
    monkeypatch.setattr("src.mcp.client.ClientSession", RecordingSession)
    return RecordingSession


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


def test_tool_call_timeout_ships_unset_so_it_follows_the_tool_runner():
    # A second literal would drift from the runner's; unset derives from it.
    assert Settings.model_fields["mcp_tool_call_timeout_seconds"].default is None


def test_unset_tool_call_timeout_follows_the_tool_runner(monkeypatch):
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", None)
    monkeypatch.setattr(config, "tool_execution_timeout_seconds", 45.0)
    assert mcp_client._configured_tool_call_timeout() == 45.0
    monkeypatch.setattr(config, "tool_execution_timeout_seconds", 80.0)
    assert mcp_client._configured_tool_call_timeout() == 80.0
    assert mcp_client._tool_call_allowance() == 80.0


def test_an_explicit_tool_call_timeout_wins_over_the_runner(monkeypatch):
    monkeypatch.setattr(config, "tool_execution_timeout_seconds", 80.0)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", 9.0)
    assert mcp_client._configured_tool_call_timeout() == 9.0


@pytest.mark.parametrize("explicit", [None, 600.0])
def test_the_tool_call_allowance_is_clamped_to_the_deadline(
    monkeypatch, fake_clock, explicit
):
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", explicit)
    monkeypatch.setattr(config, "tool_execution_timeout_seconds", 600.0)
    with enclosing_deadline(2.0):
        assert mcp_client._tool_call_allowance() == 2.0


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
    server = FakeMCPServer()
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_http_timeout_seconds", 7.0)
    monkeypatch.setattr(config, "mcp_initialize_timeout_seconds", 3.0)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", 9.0)
    runtime = http_runtime(tmp_path)

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


# --- The caller's enclosing deadline ---------------------------------------


@pytest.mark.asyncio
async def test_a_nested_caller_deadline_wins_over_every_default(
    tmp_path, monkeypatch, recording_session, fake_clock
):
    server = FakeMCPServer()
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

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
    server = FakeMCPServer()
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", None)
    monkeypatch.setattr(config, "tool_execution_timeout_seconds", 45.0)
    runtime = http_runtime(tmp_path)

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
    assert recording_session.call_timeouts == [45.0]


@pytest.mark.asyncio
async def test_an_unset_call_timeout_reaches_the_sdk_as_the_runners(
    tmp_path, monkeypatch, recording_session
):
    server = FakeMCPServer()
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", None)
    monkeypatch.setattr(config, "tool_execution_timeout_seconds", 80.0)
    runtime = http_runtime(tmp_path)

    await asyncio.wait_for(
        runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
        timeout=_STALL_CEILING_SECONDS,
    )

    assert recording_session.call_timeouts == [80.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_factory", [http_runtime, _stdio_runtime])
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
    # The budget check, off the loop before any session, is the first step refused.
    assert err.details == {
        "timeout": "enclosing_deadline_spent",
        "scope": "mcp_budget",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_factory", [http_runtime, _stdio_runtime])
async def test_an_allowance_spent_after_the_budget_check_opens_no_session(
    tmp_path, monkeypatch, fake_clock, runtime_factory
):
    def _forbidden(*args, **kwargs):
        raise AssertionError("no transport may be created once the deadline passed")

    monkeypatch.setattr("src.mcp.client.httpx2.AsyncClient", _forbidden)
    monkeypatch.setattr("src.mcp.client.stdio_client", _forbidden)
    monkeypatch.setattr("src.mcp.client.ClientSession", _forbidden)
    runtime = runtime_factory(tmp_path)
    (server_id,) = runtime.specs

    async def admitted_as_the_deadline_passes(*args) -> bool:
        fake_clock[0] += 5.0
        return True

    monkeypatch.setattr(runtime.budget, "has_room", admitted_as_the_deadline_passes)

    with enclosing_deadline(5.0):
        with pytest.raises(MCPCallError) as exc_info:
            await runtime.execute_raw(server_id, "quote", {}, scope="consultant")

    assert exc_info.value.retryable is False
    assert exc_info.value.details == {
        "timeout": "enclosing_deadline_spent",
        "scope": "mcp_initialize",
    }


@pytest.mark.asyncio
async def test_an_allowance_spent_while_opening_sends_no_call(
    tmp_path, monkeypatch, fake_clock
):
    runtime = http_runtime(tmp_path)
    opened: list[str] = []
    closed: list[str] = []
    sent: list[str] = []

    class _Session:
        async def list_tools(self, *, params=None):  # pragma: no cover
            sent.append("tools/list")
            raise RuntimeError("the catalogue must not be requested")

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
    # The catalogue, listed before the call, is the first step refused.
    assert err.details == {
        "timeout": "enclosing_deadline_spent",
        "scope": "mcp_list_tools",
    }
    # One session, closed normally, and no retry of a call that was never sent.
    assert opened == closed == ["fmp_remote"]


# --- Stalls are bounded, classified, and torn down -------------------------


@pytest.mark.asyncio
async def test_a_stalled_http_initialize_returns_within_its_bound(
    tmp_path, monkeypatch
):
    server = FakeMCPServer(stall=frozenset({"initialize"}))
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_initialize_timeout_seconds", 0.2)
    runtime = http_runtime(tmp_path)
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
    assert other_tasks(before) == set()


@pytest.mark.asyncio
async def test_a_stalled_http_tool_call_returns_within_its_bound(tmp_path, monkeypatch):
    server = FakeMCPServer(stall=frozenset({"tools/call"}))
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", 0.2)
    runtime = http_runtime(tmp_path)
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
    assert other_tasks(before) == set()


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
