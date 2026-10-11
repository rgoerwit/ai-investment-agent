"""MCP steps stay inside the time left: catalogue, call, and the retry backoff.

The session read timeout is fixed when the session opens, so a tools/list sent
after the handshake used to get a fresh full bound; and the retry loop slept its
backoff without asking whether another attempt could still fit. Both now read
the time left -- the caller's enclosing deadline *and* the tool runner's
``tool_call_deadline`` -- through the operational clamp, which MCP reads and
never binds. The listing also decides whether the call is sent at all: a tool
the server does not offer is refused, so the SDK never re-lists after a call.

Everything drives the real mcp ``ClientSession`` over an ``httpx2.MockTransport``
server, or over fake stdio pipes; no request leaves the process. Stalls park on
an event the test can release, deadlines run on a fake monotonic clock, and a
bound is proven by the value handed to ``asyncio.timeout``, never by elapsed
wall time.
"""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import httpx2
import pytest
from mcp import ClientSession
from mcp import types as mcp_types
from mcp.shared.message import SessionMessage
from mcp.types import REQUEST_TIMEOUT

import src.mcp.client as mcp_client
from src.config import config
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline, tool_call_deadline
from src.mcp.client import MCPRuntime
from src.mcp.config import MCPServerSpec
from src.mcp.errors import MCPCallError, MCPErrorCategory
from src.tooling.runtime import ToolExecutionService, ToolInvocation
from tests.helpers.mcp_server import (
    FakeMCPServer,
    RecordingSession,
    http_runtime,
    other_tasks,
)

_NOW = 1_000.0
# Generous ceiling for a stalled test: every bound under test is a fraction of it.
_STALL_CEILING_SECONDS = 10.0

_QUOTE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"price": {"type": "number"}},
    "required": ["price"],
}


@pytest.fixture(autouse=True)
def _no_deadline_margin(monkeypatch) -> None:
    """Bind tool deadlines exactly, so the arithmetic here is MCP's alone.

    ``tool_call_deadline`` keeps its binding back from the scope's hard cap
    by ``deadline_margin``; that margin is tested where it is defined
    (``tests/llm_runtime/test_operational_deadline.py``).
    """
    monkeypatch.setattr(config, "deadline_margin_fraction", 0.0)


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    """A frozen monotonic clock for the deadline helpers; advance via clock[0]."""
    clock = [_NOW]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


class _AsyncioRecorder:
    """Stands in for the MCP client's ``asyncio``: records sleeps and bounds.

    Patched on the client module's own name, so the SDK's and anyio's internal
    sleeps and timeouts are neither recorded nor disturbed. Both still run.
    """

    def __init__(self) -> None:
        self.sleeps: list[float] = []
        self.bounds: list[float] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        await asyncio.sleep(delay)

    def timeout(self, delay: float) -> asyncio.Timeout:
        self.bounds.append(delay)
        return asyncio.timeout(delay)


@pytest.fixture
def recorder(monkeypatch) -> _AsyncioRecorder:
    recorded = _AsyncioRecorder()
    monkeypatch.setattr(mcp_client, "asyncio", recorded)
    return recorded


@pytest.fixture
def sleeps(recorder: _AsyncioRecorder) -> list[float]:
    """Every backoff sleep the MCP client asks for (and still sleeps)."""
    return recorder.sleeps


@pytest.fixture
def recording_session(monkeypatch) -> type[RecordingSession]:
    RecordingSession.reset()
    monkeypatch.setattr("src.mcp.client.ClientSession", RecordingSession)
    return RecordingSession


# --- The retry minimum is a share of the tool call's configured allowance -----

# 0.0625 x 45 s = 2.8125 s: a dyadic minimum, so the boundary compares exactly.
_MINIMUM = 0.0625 * 45.0


async def _retry_after_502(tmp_path, monkeypatch, *, deadline: float):
    server = FakeMCPServer(status={"tools/call": (502, {})})
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)
    with enclosing_deadline(deadline):
        with pytest.raises(MCPCallError) as exc_info:
            await runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant")
    return server, exc_info.value


@pytest.mark.asyncio
async def test_a_retry_with_exactly_the_derived_minimum_left_is_sent(
    tmp_path, monkeypatch, fake_clock, sleeps
):
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.0625)
    server, err = await _retry_after_502(tmp_path, monkeypatch, deadline=_MINIMUM)
    assert server.methods.count("tools/call") == 2
    assert sleeps == [0.0]
    assert "retry" not in err.details


@pytest.mark.asyncio
async def test_a_retry_with_just_under_the_derived_minimum_left_is_never_sent(
    tmp_path, monkeypatch, fake_clock, sleeps
):
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.0625)
    server, err = await _retry_after_502(
        tmp_path, monkeypatch, deadline=_MINIMUM - 0.001
    )
    assert server.methods.count("tools/call") == 1
    assert sleeps == []
    assert err.details["retry"] == "refused_deadline"


@pytest.mark.asyncio
async def test_the_minimum_is_not_derived_from_the_time_left(
    tmp_path, monkeypatch, fake_clock, sleeps
):
    """1 s left: a minimum taken from the clamped remainder would be 0.0625 s."""
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.0625)
    server, err = await _retry_after_502(tmp_path, monkeypatch, deadline=1.0)
    assert server.methods.count("tools/call") == 1
    assert err.details["retry"] == "refused_deadline"


@pytest.mark.asyncio
async def test_a_tiny_allowance_with_a_small_fraction_still_retries(
    tmp_path, monkeypatch, fake_clock, sleeps
):
    """0.04 x 0.2 s = 0.008 s: 0.01 s left fits one retry, after a 0.002 s sleep."""
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", 0.2)
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.04)
    server, _ = await _retry_after_502(tmp_path, monkeypatch, deadline=0.01)
    assert server.methods.count("tools/call") == 2
    assert sleeps == [pytest.approx(0.002)]


@pytest.mark.asyncio
async def test_a_fraction_override_moves_the_mcp_minimum(
    tmp_path, monkeypatch, fake_clock, sleeps
):
    """4 s left retries at the default (1.8 s); at 0.5 x 45 s = 22.5 s it does not."""
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.5)
    server, err = await _retry_after_502(tmp_path, monkeypatch, deadline=4.0)
    assert server.methods.count("tools/call") == 1
    assert err.details["retry"] == "refused_deadline"


# --- Catalogue: the listing gets what the handshake left ---------------------


@pytest.mark.asyncio
async def test_the_listing_gets_only_what_the_handshake_left(
    tmp_path, monkeypatch, fake_clock, recorder
):
    """The handshake uses 0.04 s of a 0.05 s deadline; the listing gets 0.01 s.

    Before, the listing got the session's fixed 0.05 s on top, failed as a
    retryable timeout, and the retry slept 0.5 s. Now its bound is what the
    handshake left, and the retry is refused: nothing sleeps, no call is sent.
    """

    def handshake_uses_most_of_the_deadline() -> None:
        fake_clock[0] += 0.04

    server = FakeMCPServer(
        stall=frozenset({"tools/list"}),
        on_request={"initialize": handshake_uses_most_of_the_deadline},
    )
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)
    before = set(asyncio.all_tasks())

    with enclosing_deadline(0.05):
        with pytest.raises(MCPCallError) as exc_info:
            await asyncio.wait_for(
                runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
                timeout=_STALL_CEILING_SECONDS,
            )

    err = exc_info.value
    assert recorder.bounds == [pytest.approx(0.01)]
    assert recorder.sleeps == []
    # Sent, then timed out; the retry refused, never sent.
    assert err.category is MCPErrorCategory.TRANSPORT
    assert err.json_rpc_code == REQUEST_TIMEOUT
    assert err.details["timeout"] == "time_left_elapsed"
    assert err.details["scope"] == "mcp_list_tools"
    assert err.details["retry"] == "refused_deadline"
    assert err.retryable is False
    assert isinstance(err.__cause__, MCPCallError)
    assert err.__cause__.retryable is True
    assert server.methods.count("initialize") == 1
    assert server.methods.count("tools/list") == 1
    assert server.calls_sent == 0
    assert server.parked == server.cancelled == 1
    assert not server.release.is_set()
    assert server.all_closed()
    assert other_tasks(before) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("handshake_seconds", [2.0, 2.5])
async def test_a_handshake_that_spends_the_deadline_sends_no_catalogue(
    tmp_path, monkeypatch, fake_clock, recorder, handshake_seconds
):
    """Exactly at the deadline, and past it: nothing is left to list with."""

    def handshake_spends_the_deadline() -> None:
        fake_clock[0] += handshake_seconds

    server = FakeMCPServer(on_request={"initialize": handshake_spends_the_deadline})
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    with enclosing_deadline(2.0):
        with pytest.raises(MCPCallError) as exc_info:
            await runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant")

    err = exc_info.value
    assert err.retryable is False
    assert err.details == {
        "timeout": "enclosing_deadline_spent",
        "scope": "mcp_list_tools",
    }
    assert recorder.bounds == []
    assert "tools/list" not in server.methods
    assert server.calls_sent == 0
    assert server.all_closed()


@pytest.mark.asyncio
async def test_a_stdio_catalogue_is_bounded_by_the_time_left(
    tmp_path, monkeypatch, fake_clock, recorder
):
    """The same bound over stdio pipes: the server answers initialize only."""
    received: list[str] = []
    transport_closed: list[bool] = []

    @asynccontextmanager
    async def answering_stdio_client(params):
        to_client, read = anyio.create_memory_object_stream(10)
        write, from_client = anyio.create_memory_object_stream(10)

        async def serve() -> None:
            async for item in from_client:
                message = item.message
                method = getattr(message, "method", None)
                received.append(method)
                if method == "initialize":
                    fake_clock[0] += 1.96
                    response = mcp_types.JSONRPCResponse(
                        jsonrpc="2.0",
                        id=message.id,
                        result={
                            "protocolVersion": "2025-11-25",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "fake", "version": "1"},
                        },
                    )
                    await to_client.send(SessionMessage(response))
                # Anything else is never answered.

        async with anyio.create_task_group() as group:
            group.start_soon(serve)
            try:
                yield read, write
            finally:
                group.cancel_scope.cancel()
                for stream in (to_client, read, write, from_client):
                    await stream.aclose()
                transport_closed.append(True)

    monkeypatch.setattr("src.mcp.client.stdio_client", answering_stdio_client)
    spec = MCPServerSpec(
        id="local_stdio",
        description="stdio",
        transport="stdio",
        command="fake-mcp-server",
        scopes=["consultant"],
        tool_allowlist=["quote"],
    )
    runtime = MCPRuntime([spec], budget_db_path=str(tmp_path / "mcp.db"))

    with enclosing_deadline(2.0):
        with pytest.raises(MCPCallError) as exc_info:
            await asyncio.wait_for(
                runtime.execute_raw("local_stdio", "quote", {}, scope="consultant"),
                timeout=_STALL_CEILING_SECONDS,
            )

    assert recorder.bounds == [pytest.approx(0.04)]
    assert exc_info.value.details["timeout"] == "time_left_elapsed"
    assert received[:3] == ["initialize", "notifications/initialized", "tools/list"]
    assert "tools/call" not in received
    assert transport_closed == [True]


# --- Catalogue: the SDK's own processing survives the bound ------------------


@pytest.mark.asyncio
async def test_a_call_is_validated_against_the_schema_listed_before_it(
    tmp_path, monkeypatch
):
    server = FakeMCPServer(
        tools=[
            {
                "name": "quote",
                "inputSchema": {"type": "object"},
                "outputSchema": _QUOTE_OUTPUT_SCHEMA,
            }
        ],
        call_result={
            "content": [{"type": "text", "text": "{}"}],
            "structuredContent": {"price": "not-a-number"},
            "isError": False,
        },
    )
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    with pytest.raises(MCPCallError) as exc_info:
        await asyncio.wait_for(
            runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
            timeout=_STALL_CEILING_SECONDS,
        )

    assert "Invalid structured content" in exc_info.value.message
    # Listed once, before the call, inside the time left; the SDK validated
    # against that cached schema and sent no listing of its own afterwards.
    assert server.methods == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]


@pytest.mark.asyncio
async def test_a_conforming_result_passes_with_one_listing(tmp_path, monkeypatch):
    server = FakeMCPServer(
        tools=[
            {
                "name": "quote",
                "inputSchema": {"type": "object"},
                "outputSchema": _QUOTE_OUTPUT_SCHEMA,
            }
        ],
        call_result={
            "content": [{"type": "text", "text": '{"price": 1.5}'}],
            "structuredContent": {"price": 1.5},
            "isError": False,
        },
    )
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    payload = await asyncio.wait_for(
        runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
        timeout=_STALL_CEILING_SECONDS,
    )

    assert payload["is_error"] is False
    assert server.methods.count("tools/list") == 1
    assert server.all_closed()


# --- The listing decides whether the call is sent ----------------------------


_OTHER_TOOL = {"name": "profile", "inputSchema": {"type": "object"}}
_QUOTE_TOOL = {"name": "quote", "inputSchema": {"type": "object"}}


async def _refused(runtime: MCPRuntime, tool: str = "quote") -> MCPCallError:
    with pytest.raises(MCPCallError) as exc_info:
        await asyncio.wait_for(
            runtime.execute_raw("fmp_remote", tool, {}, scope="consultant"),
            timeout=_STALL_CEILING_SECONDS,
        )
    return exc_info.value


@pytest.mark.asyncio
@pytest.mark.parametrize("listed", [[_OTHER_TOOL], []], ids=["other-tool", "empty"])
async def test_a_tool_the_server_does_not_list_is_refused_unsent(
    tmp_path, monkeypatch, listed
):
    """Sent anyway, the SDK would re-list after the call under the stale bound."""
    server = FakeMCPServer(pages=[listed])
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    err = await _refused(runtime)

    assert err.category is MCPErrorCategory.CONFIG
    assert err.retryable is False
    assert err.tool_name == "quote"
    assert err.details == {"refused": "tool_not_listed"}
    # One listing, no call, no re-list, no retry, nothing counted, no cooldown.
    assert server.methods == ["initialize", "notifications/initialized", "tools/list"]
    assert len(server.clients) == 1
    assert server.all_closed()
    assert runtime.budget._run_calls == {}
    assert runtime._cooldowns == {}


@pytest.mark.asyncio
async def test_a_tool_on_a_later_page_is_found_and_validated_by_its_schema(
    tmp_path, monkeypatch
):
    quote = {**_QUOTE_TOOL, "outputSchema": _QUOTE_OUTPUT_SCHEMA}
    server = FakeMCPServer(
        pages=[[_OTHER_TOOL], [quote]],
        call_result={
            "content": [{"type": "text", "text": "{}"}],
            "structuredContent": {"price": "not-a-number"},
            "isError": False,
        },
    )
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    err = await _refused(runtime)

    # The page-2 schema was cached and applied; the SDK listed nothing after.
    assert "Invalid structured content" in err.message
    assert server.list_cursors == [None, "1"]
    assert server.methods[2:] == ["tools/list", "tools/list", "tools/call"]


@pytest.mark.asyncio
async def test_listing_stops_at_the_page_that_names_the_tool(tmp_path, monkeypatch):
    server = FakeMCPServer(pages=[[_QUOTE_TOOL], [_OTHER_TOOL], [_OTHER_TOOL]])
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    payload = await asyncio.wait_for(
        runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
        timeout=_STALL_CEILING_SECONDS,
    )

    assert payload["is_error"] is False
    assert server.list_cursors == [None]
    assert server.calls_sent == 1


@pytest.mark.asyncio
async def test_a_repeated_cursor_ends_the_listing_and_refuses_the_call(
    tmp_path, monkeypatch
):
    server = FakeMCPServer(
        pages=[[_OTHER_TOOL], [_OTHER_TOOL]], next_cursors=["1", "1"]
    )
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    err = await _refused(runtime)

    assert err.details == {"refused": "tool_not_listed"}
    assert server.list_cursors == [None, "1"]
    assert server.calls_sent == 0


@pytest.mark.asyncio
async def test_every_page_shares_one_bound(tmp_path, monkeypatch, fake_clock, recorder):
    """A second page that stalls is cut off by the listing's single bound."""
    stalled = asyncio.Event()

    async def stall_the_second_page() -> None:
        if server.methods.count("tools/list") == 2:
            stalled.set()
            await asyncio.Event().wait()

    server = FakeMCPServer(
        pages=[[_OTHER_TOOL], [_QUOTE_TOOL]],
        on_request={"tools/list": stall_the_second_page},
    )
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    with tool_call_deadline(0.05):
        err = await _refused(runtime)

    assert stalled.is_set()
    assert recorder.bounds == [pytest.approx(0.05)]
    assert err.details["timeout"] == "time_left_elapsed"
    assert err.details["retry"] == "refused_deadline"
    assert server.methods.count("tools/list") == 2
    assert server.calls_sent == 0
    assert server.all_closed()


@pytest.mark.asyncio
async def test_a_malformed_listing_is_classified_and_sends_no_call(
    tmp_path, monkeypatch
):
    server = FakeMCPServer(pages=[[{"name": 5}]])
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    err = await _refused(runtime)

    # classify_mcp_error's fallback for a schema error: not retried.
    assert "validation error" in err.message
    assert err.retryable is False
    assert len(server.clients) == 1
    assert server.calls_sent == 0
    assert runtime.budget._run_calls == {}


class _ModernSession(ClientSession):
    """Adopts the 2026-07-28 protocol after the legacy handshake.

    ``initialize()`` only negotiates legacy versions, where the SDK applies no
    ``x-mcp-header`` filter; adopting a modern discover result (a public SDK
    call) is the way to exercise the filter through ``list_tools()``. Production
    never does this: the filter is inert there (docs/MCP.md).
    """

    async def initialize(self):
        result = await super().initialize()
        self.adopt(
            mcp_types.DiscoverResult(
                supported_versions=["2026-07-28"],
                capabilities=mcp_types.ServerCapabilities(
                    tools=mcp_types.ToolsCapability()
                ),
            )
        )
        return result


_BAD_HEADER_TOOL = {
    "name": "bad",
    "inputSchema": {
        "type": "object",
        "properties": {"symbol": {"type": "object", "x-mcp-header": "Symbol"}},
    },
}


@pytest.mark.asyncio
async def test_a_tool_dropped_for_an_invalid_header_annotation_is_refused_unsent(
    tmp_path, monkeypatch
):
    server = FakeMCPServer(
        tools=[_QUOTE_TOOL, _BAD_HEADER_TOOL],
        # A 2026-07-28 result carries its result type.
        call_result={
            "content": [{"type": "text", "text": "{}"}],
            "isError": False,
            "resultType": "complete",
        },
    )
    server.install(monkeypatch)
    monkeypatch.setattr("src.mcp.client.ClientSession", _ModernSession)
    runtime = http_runtime(tmp_path, allowlist=["quote", "bad"])

    err = await _refused(runtime, "bad")
    payload = await runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant")

    assert err.details == {"refused": "tool_not_listed"}
    assert payload["is_error"] is False
    assert server.calls_sent == 1


@pytest.mark.asyncio
async def test_on_a_legacy_session_the_header_filter_is_inert(tmp_path, monkeypatch):
    """What production negotiates: the annotated tool is listed and called."""
    server = FakeMCPServer(tools=[_QUOTE_TOOL, _BAD_HEADER_TOOL])
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path, allowlist=["quote", "bad"])

    payload = await runtime.execute_raw("fmp_remote", "bad", {}, scope="consultant")

    assert payload["is_error"] is False
    assert server.calls_sent == 1


# --- Retry backoff: admitted against the time left ---------------------------


def _stall_advancing(fake_clock: list[float], seconds: float) -> Callable[[], None]:
    def advance() -> None:
        fake_clock[0] += seconds

    return advance


@pytest.mark.asyncio
async def test_a_call_timeout_near_the_runner_deadline_does_not_sleep_the_backoff(
    tmp_path, monkeypatch, fake_clock, sleeps
):
    """The call times out with 0.05 s left of the runner's deadline; the minimum is 0.1 s.

    The 500 ms backoff is refused, never slept, and no second attempt is made.
    """
    server = FakeMCPServer(
        stall=frozenset({"tools/call"}),
        on_request={"tools/call": _stall_advancing(fake_clock, 1.45)},
    )
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", 0.2)
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.5)
    runtime = http_runtime(tmp_path)
    before = set(asyncio.all_tasks())

    with tool_call_deadline(1.5):
        with pytest.raises(MCPCallError) as exc_info:
            await asyncio.wait_for(
                runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
                timeout=_STALL_CEILING_SECONDS,
            )

    err = exc_info.value
    assert sleeps == []
    assert server.methods.count("tools/call") == 1
    assert len(server.clients) == 1
    # Sent, then timed out -- and the retry refused, never sent.
    assert err.category is MCPErrorCategory.TRANSPORT
    assert err.json_rpc_code == REQUEST_TIMEOUT
    assert err.retryable is False
    assert err.details["retry"] == "refused_deadline"
    assert err.details["backoff_seconds"] == 0.5
    assert isinstance(err.__cause__, MCPCallError)
    assert err.__cause__.retryable is True
    assert server.parked == server.cancelled == 1
    assert server.all_closed()
    assert other_tasks(before) == set()


@pytest.mark.asyncio
async def test_a_backoff_is_shortened_to_leave_a_useful_attempt(
    tmp_path, monkeypatch, fake_clock, sleeps
):
    """0.3 s left after the failure, minimum 0.1 s: sleep 0.2 s, then retry."""
    server = FakeMCPServer(
        stall=frozenset({"tools/call"}),
        on_request={"tools/call": _stall_advancing(fake_clock, 1.2)},
    )
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", 0.2)
    monkeypatch.setattr(config, "min_useful_attempt_fraction", 0.5)
    runtime = http_runtime(tmp_path)

    with tool_call_deadline(1.5):
        with pytest.raises(MCPCallError) as exc_info:
            await asyncio.wait_for(
                runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
                timeout=_STALL_CEILING_SECONDS,
            )

    assert sleeps == [pytest.approx(0.2)]
    assert server.methods.count("tools/call") == 2
    # The second attempt was sent and timed out on its own: not a refusal.
    assert exc_info.value.json_rpc_code == REQUEST_TIMEOUT
    assert "retry" not in exc_info.value.details
    assert server.all_closed()


@pytest.mark.asyncio
async def test_with_no_deadline_the_backoff_is_unchanged(tmp_path, monkeypatch, sleeps):
    server = FakeMCPServer(stall=frozenset({"tools/call"}))
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", 0.1)
    runtime = http_runtime(tmp_path)

    with pytest.raises(MCPCallError):
        await asyncio.wait_for(
            runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
            timeout=_STALL_CEILING_SECONDS,
        )

    assert sleeps == [0.5]
    assert server.methods.count("tools/call") == 2


@pytest.mark.asyncio
async def test_a_rate_limit_wait_that_does_not_fit_is_refused_and_still_cools_down(
    tmp_path, monkeypatch, fake_clock, sleeps
):
    """A 429 asking for 2 s with 1.5 s left: not shortened, refused.

    The HTTP classification survives the refusal, and the 429 still registers
    the server's rate-limit cooldown.
    """
    server = FakeMCPServer(status={"tools/call": (429, {"retry-after": "2"})})
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    with tool_call_deadline(1.5):
        with pytest.raises(MCPCallError) as exc_info:
            await runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant")

    err = exc_info.value
    assert sleeps == []
    assert server.methods.count("tools/call") == 1
    assert err.category is MCPErrorCategory.TRANSPORT
    assert err.http_status == 429
    assert err.retry_after_seconds == 2
    assert err.retryable is False
    assert err.details["retry"] == "refused_deadline"
    assert "fmp_remote" in runtime._cooldowns


@pytest.mark.asyncio
async def test_a_backoff_that_overruns_is_rechecked_before_the_retry(
    tmp_path, monkeypatch, fake_clock
):
    """The sleep returns later than asked: the retry is refused, not opened."""
    server = FakeMCPServer(status={"tools/call": (502, {})})
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    class _OverrunningAsyncio:
        def __getattr__(self, name: str) -> Any:
            return getattr(asyncio, name)

        @staticmethod
        async def sleep(delay: float) -> None:
            # 4 s deadline, 1.8 s minimum (0.04 x 45 s): a 0.5 s sleep is
            # admitted, and returning 3 s late leaves 0.5 s, below it.
            fake_clock[0] += delay + 3.0

    monkeypatch.setattr(mcp_client, "asyncio", _OverrunningAsyncio())

    with enclosing_deadline(4.0):
        with pytest.raises(MCPCallError) as exc_info:
            await runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant")

    err = exc_info.value
    assert len(server.clients) == 1
    assert server.methods.count("tools/call") == 1
    assert err.http_status == 502
    assert err.details["retry"] == "refused_deadline"


# --- Cancellation -------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["initialize", "tools/list", "tools/call"])
async def test_external_cancellation_propagates_and_tears_down(
    tmp_path, monkeypatch, method
):
    server = FakeMCPServer(stall=frozenset({method}))
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)
    before = set(asyncio.all_tasks())

    task = asyncio.create_task(
        runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant")
    )
    await asyncio.wait_for(server.parked_event.wait(), timeout=_STALL_CEILING_SECONDS)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=_STALL_CEILING_SECONDS)

    # Not converted into an MCPCallError, not retried, and torn down.
    assert server.methods.count(method) == 1
    assert server.parked == server.cancelled == 1
    assert server.all_closed()
    assert other_tasks(before) == set()


# --- The tool-call deadline: read, never bound --------------------------------


@pytest.mark.asyncio
async def test_mcp_bounds_honour_the_tool_call_deadline(
    tmp_path, monkeypatch, fake_clock, recording_session
):
    server = FakeMCPServer()
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    with tool_call_deadline(2.0) as bound:
        await asyncio.wait_for(
            runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
            timeout=_STALL_CEILING_SECONDS,
        )

    (client,) = server.clients
    assert client.timeout == httpx2.Timeout(2.0)
    assert recording_session.session_timeouts == [2.0]
    assert recording_session.call_timeouts == [2.0]
    # Read, never bound: the call ran under exactly the outer binding, and
    # no LLM enclosing deadline was bound on the way.
    assert recording_session.deadlines_at_call == [(bound, None)]


@pytest.mark.asyncio
async def test_mcp_bounds_honour_the_service_runner_deadline(
    tmp_path, monkeypatch, fake_clock, recording_session
):
    """Through the real service: the runner's allowance reaches the SDK."""
    server = FakeMCPServer()
    server.install(monkeypatch)
    monkeypatch.setattr(config, "tool_execution_timeout_seconds", 3.0)
    runtime = http_runtime(tmp_path)
    service = ToolExecutionService([])

    async def runner(args: dict[str, Any]) -> Any:
        return await runtime.execute_raw(
            "fmp_remote", "quote", args, scope="consultant"
        )

    result = await asyncio.wait_for(
        service.execute(
            ToolInvocation(name="mcp__fmp_remote__quote", args={}, source="consultant"),
            runner=runner,
        ),
        timeout=_STALL_CEILING_SECONDS,
    )

    assert result.value["is_error"] is False
    assert recording_session.session_timeouts == [3.0]
    assert recording_session.call_timeouts == [3.0]
    assert recording_session.deadlines_at_call == [(_NOW + 3.0, None)]


_BINDERS = frozenset(
    {
        "tool_call_deadline",
        "enclosing_deadline",
        "enclosing_deadline_at",
        "invocation_scope",
    }
)


def test_the_mcp_client_never_binds_a_deadline():
    """Static guard: MCP reads the deadlines; only the service binds them."""
    tree = ast.parse(Path(mcp_client.__file__).read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _BINDERS:
            offenders.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in _BINDERS | {
            "_TOOL_CALL_DEADLINE",
            "_ENCLOSING_DEADLINE",
        }:
            offenders.append(node.attr)
        elif isinstance(node, ast.alias) and node.name in _BINDERS:
            offenders.append(node.name)
    assert offenders == []
