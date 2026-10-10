"""MCP steps stay inside the time left: catalogue, call, and the retry backoff.

The session read timeout is fixed when the session opens, so a tools/list sent
after the handshake used to get a fresh full bound; and the retry loop slept its
backoff without asking whether another attempt could still fit. Both now read
the time left -- the caller's enclosing deadline *and* the tool runner's
``tool_call_deadline`` -- through the operational clamp, which MCP reads and
never binds.

Everything drives the real mcp ``ClientSession`` over an ``httpx2.MockTransport``
server, or over fake stdio pipes; no request leaves the process. Stalls park on
an event the test can release, deadlines run on a fake monotonic clock, and the
one real-clock test discriminates a ~0.05 s return from the ~0.6 s the old code
took, not a bound from a ceiling.
"""

from __future__ import annotations

import ast
import asyncio
import json
import time
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
from pydantic import ValidationError

import src.mcp.client as mcp_client
from src.config import Settings, config
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline, tool_call_deadline
from src.mcp.client import MCPRuntime
from src.mcp.config import MCPServerSpec
from src.mcp.errors import MCPCallError, MCPErrorCategory
from src.tooling.runtime import ToolExecutionService, ToolInvocation

_URL = "https://example.test/mcp"
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


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Record every sleep the MCP client asks for, and still sleep it.

    Patched on the client module's own ``asyncio`` name, so the SDK's and
    anyio's internal sleeps are neither recorded nor disturbed.
    """
    recorded: list[float] = []
    real_sleep = asyncio.sleep

    class _AsyncioProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(asyncio, name)

        @staticmethod
        async def sleep(delay: float) -> None:
            recorded.append(delay)
            await real_sleep(delay)

    monkeypatch.setattr(mcp_client, "asyncio", _AsyncioProxy())
    return recorded


def _spec(*, allowlist: list[str] | None = None) -> MCPServerSpec:
    return MCPServerSpec(
        id="fmp_remote",
        description="FMP",
        transport="streamable_http",
        base_url=_URL,
        scopes=["consultant"],
        tool_allowlist=allowlist if allowlist is not None else ["quote"],
    )


def _http_runtime(tmp_path: Path, **kwargs: Any) -> MCPRuntime:
    return MCPRuntime([_spec(**kwargs)], budget_db_path=str(tmp_path / "mcp.db"))


def _other_tasks(before: set[asyncio.Task]) -> set[asyncio.Task]:
    return {
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and task not in before
    }


class _Server:
    """A streamable-HTTP MCP server behind ``httpx2.MockTransport``.

    ``on_request`` runs a hook per method when the request arrives (advance a
    fake clock, sleep for real). Methods in ``stall`` park until ``release``
    is set, counting parked and cancelled requests. ``status`` answers a
    method with a bare HTTP error instead.
    """

    def __init__(
        self,
        *,
        stall: frozenset[str] = frozenset(),
        tools: list[dict[str, Any]] | None = None,
        call_result: dict[str, Any] | None = None,
        status: dict[str, tuple[int, dict[str, str]]] | None = None,
        on_request: dict[str, Callable[[], Any]] | None = None,
    ) -> None:
        self.stall = stall
        self.tools = tools or [{"name": "quote", "inputSchema": {"type": "object"}}]
        self.call_result = call_result or {
            "content": [{"type": "text", "text": "{}"}],
            "isError": False,
        }
        self.status = status or {}
        self.on_request = on_request or {}
        self.methods: list[str] = []
        self.parked = 0
        self.cancelled = 0
        self.parked_event = asyncio.Event()
        self.release = asyncio.Event()
        self.clients: list[httpx2.AsyncClient] = []

    async def handler(self, request: httpx2.Request) -> httpx2.Response:
        if request.method != "POST":
            return httpx2.Response(405)
        message = json.loads(request.content)
        method = message.get("method")
        self.methods.append(method)
        if "id" not in message:
            return httpx2.Response(202)
        hook = self.on_request.get(method)
        if hook is not None:
            outcome = hook()
            if asyncio.iscoroutine(outcome):
                await outcome
        if method in self.status:
            code, headers = self.status[method]
            return httpx2.Response(code, headers=headers, text="denied")
        if method in self.stall:
            self.parked += 1
            self.parked_event.set()
            try:
                await self.release.wait()
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
            # The cache fields are required at 2026-07-28 and dropped as a
            # later revision's fields on a legacy session.
            result = {
                "tools": self.tools,
                "ttlMs": 0,
                "cacheScope": "private",
                "resultType": "complete",
            }
        else:
            result = self.call_result
        return httpx2.Response(
            200, json={"jsonrpc": "2.0", "id": message["id"], "result": result}
        )

    def install(self, monkeypatch) -> None:
        """Route the runtime's own httpx2 client here, keeping its kwargs."""
        real_client = httpx2.AsyncClient

        def _client(**kwargs):
            client = real_client(transport=httpx2.MockTransport(self.handler), **kwargs)
            self.clients.append(client)
            return client

        monkeypatch.setattr("src.mcp.client.httpx2.AsyncClient", _client)

    def all_closed(self) -> bool:
        return bool(self.clients) and all(client.is_closed for client in self.clients)


class _RecordingSession(ClientSession):
    """The real ClientSession, recording its bounds and the deadlines it ran under."""

    session_timeouts: list[float | None] = []
    call_timeouts: list[float | None] = []
    deadlines_at_call: list[tuple[float | None, float | None]] = []

    def __init__(self, read, write, read_timeout_seconds=None, **kwargs):
        type(self).session_timeouts.append(read_timeout_seconds)
        super().__init__(read, write, read_timeout_seconds, **kwargs)

    async def call_tool(self, name, arguments=None, read_timeout_seconds=None, **kw):
        type(self).call_timeouts.append(read_timeout_seconds)
        type(self).deadlines_at_call.append(
            (
                llm_timeouts._TOOL_CALL_DEADLINE.get(),
                llm_timeouts.current_enclosing_deadline(),
            )
        )
        return await super().call_tool(name, arguments, read_timeout_seconds, **kw)


@pytest.fixture
def recording_session(monkeypatch) -> type[_RecordingSession]:
    _RecordingSession.session_timeouts = []
    _RecordingSession.call_timeouts = []
    _RecordingSession.deadlines_at_call = []
    monkeypatch.setattr("src.mcp.client.ClientSession", _RecordingSession)
    return _RecordingSession


# --- The retry minimum is a setting -----------------------------------------


def test_retry_minimum_shipped_default():
    assert Settings.model_fields["mcp_retry_min_useful_seconds"].default == 1.0


@pytest.mark.parametrize("value", ["0", "45", "60"])
def test_retry_minimum_must_be_positive_and_below_the_call_bound(monkeypatch, value):
    monkeypatch.setenv("MCP_RETRY_MIN_USEFUL_SECONDS", value)
    with pytest.raises(ValidationError):
        Settings()


# --- Catalogue: the listing gets what the handshake left ---------------------


@pytest.mark.asyncio
async def test_a_50ms_allowance_with_a_40ms_handshake_returns_within_the_bound(
    tmp_path, monkeypatch, sleeps
):
    """Real clock: the handshake takes 40 ms of 50; the catalogue then stalls.

    Before: the catalogue got the session's fixed 50 ms on top, failed as a
    retryable timeout, and the retry slept 500 ms -- about 0.6 s in all. Now
    it gets the ~10 ms left and the retry is refused, so the call returns at
    about the deadline. 0.25 s sits between the two with room on both sides.
    """

    async def slow_handshake() -> None:
        await asyncio.sleep(0.04)

    server = _Server(
        stall=frozenset({"tools/list"}),
        on_request={"initialize": slow_handshake},
    )
    server.install(monkeypatch)
    runtime = _http_runtime(tmp_path)
    before = set(asyncio.all_tasks())

    started = time.monotonic()
    with enclosing_deadline(0.05):
        with pytest.raises(MCPCallError) as exc_info:
            await asyncio.wait_for(
                runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
                timeout=_STALL_CEILING_SECONDS,
            )
    elapsed = time.monotonic() - started

    err = exc_info.value
    assert elapsed < 0.25
    assert err.category is MCPErrorCategory.TRANSPORT
    assert err.retryable is False
    assert sleeps == []
    assert "tools/call" not in server.methods
    assert server.methods.count("initialize") == 1
    assert not server.release.is_set()
    assert server.all_closed()
    assert _other_tasks(before) == set()


@pytest.mark.asyncio
async def test_the_catalogue_is_bounded_by_the_time_the_handshake_left(
    tmp_path, monkeypatch, fake_clock
):
    """The handshake uses 1.96 s of a 2 s deadline; the listing gets 0.04 s.

    The session's own read timeout is 2 s (fixed at open), so the old branch
    would park the stalled listing for 2 s and fail with no ``details``.
    """

    def handshake_uses_most_of_the_deadline() -> None:
        fake_clock[0] += 1.96

    server = _Server(
        stall=frozenset({"tools/list"}),
        on_request={"initialize": handshake_uses_most_of_the_deadline},
    )
    server.install(monkeypatch)
    runtime = _http_runtime(tmp_path)
    before = set(asyncio.all_tasks())

    started = time.monotonic()
    with enclosing_deadline(2.0):
        with pytest.raises(MCPCallError) as exc_info:
            await asyncio.wait_for(
                runtime.list_tools_for_scope("consultant"),
                timeout=_STALL_CEILING_SECONDS,
            )
    elapsed = time.monotonic() - started

    err = exc_info.value
    assert elapsed < 1.0  # the old bound was 2.0 s of real time
    assert err.category is MCPErrorCategory.TRANSPORT
    assert err.retryable is True
    assert err.json_rpc_code == REQUEST_TIMEOUT
    assert err.details == {"timeout": "time_left_elapsed", "scope": "mcp_list_tools"}
    # Sent, then torn down: the parked request was cancelled and the client
    # closed, with the SDK contexts left normally.
    assert server.methods.count("tools/list") == 1
    assert server.parked == server.cancelled == 1
    assert server.all_closed()
    assert _other_tasks(before) == set()


@pytest.mark.asyncio
async def test_a_handshake_that_spends_the_deadline_sends_no_catalogue(
    tmp_path, monkeypatch, fake_clock
):
    def handshake_spends_the_deadline() -> None:
        fake_clock[0] += 2.0

    server = _Server(on_request={"initialize": handshake_spends_the_deadline})
    server.install(monkeypatch)
    runtime = _http_runtime(tmp_path)

    with enclosing_deadline(2.0):
        with pytest.raises(MCPCallError) as exc_info:
            await runtime.list_tools_for_scope("consultant")

    err = exc_info.value
    assert err.retryable is False
    assert err.details == {
        "timeout": "enclosing_deadline_spent",
        "scope": "mcp_list_tools",
    }
    assert "tools/list" not in server.methods
    assert server.all_closed()


@pytest.mark.asyncio
async def test_a_stdio_catalogue_is_bounded_by_the_time_left(
    tmp_path, monkeypatch, fake_clock
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

    started = time.monotonic()
    with enclosing_deadline(2.0):
        with pytest.raises(MCPCallError) as exc_info:
            await asyncio.wait_for(
                runtime.list_tools_for_scope("consultant"),
                timeout=_STALL_CEILING_SECONDS,
            )
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert exc_info.value.details["timeout"] == "time_left_elapsed"
    assert received[:3] == ["initialize", "notifications/initialized", "tools/list"]
    assert transport_closed == [True]


# --- Catalogue: the SDK's own processing survives the bound ------------------


@pytest.mark.asyncio
async def test_a_call_is_validated_against_the_schema_listed_before_it(
    tmp_path, monkeypatch
):
    server = _Server(
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
    runtime = _http_runtime(tmp_path)

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
    server = _Server(
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
    runtime = _http_runtime(tmp_path)

    payload = await asyncio.wait_for(
        runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
        timeout=_STALL_CEILING_SECONDS,
    )

    assert payload["is_error"] is False
    assert server.methods.count("tools/list") == 1
    assert server.all_closed()


class _ModernSession(ClientSession):
    """Adopts the 2026-07-28 protocol after the legacy handshake.

    ``initialize()`` only negotiates legacy versions, where the SDK applies no
    ``x-mcp-header`` filter; adopting a modern discover result (a public SDK
    call) is the way to exercise the filter through ``list_tools()``.
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


@pytest.mark.asyncio
async def test_an_invalid_header_annotation_is_still_dropped(tmp_path, monkeypatch):
    bad_input_schema = {
        "type": "object",
        "properties": {
            "symbol": {"type": "object", "x-mcp-header": "Symbol"},
        },
    }
    server = _Server(
        tools=[
            {"name": "quote", "inputSchema": {"type": "object"}},
            {"name": "bad", "inputSchema": bad_input_schema},
        ]
    )
    server.install(monkeypatch)
    monkeypatch.setattr("src.mcp.client.ClientSession", _ModernSession)
    runtime = _http_runtime(tmp_path, allowlist=["quote", "bad"])

    tools = await asyncio.wait_for(
        runtime.list_tools_for_scope("consultant"), timeout=_STALL_CEILING_SECONDS
    )

    assert [tool.name for tool in tools] == ["quote"]


# --- Retry backoff: admitted against the time left ---------------------------


def _stall_advancing(fake_clock: list[float], seconds: float) -> Callable[[], None]:
    def advance() -> None:
        fake_clock[0] += seconds

    return advance


@pytest.mark.asyncio
async def test_a_call_timeout_near_the_runner_deadline_does_not_sleep_the_backoff(
    tmp_path, monkeypatch, fake_clock, sleeps
):
    """The call times out with 0.5 s left of the runner's deadline; the minimum is 1 s.

    The 500 ms backoff is refused, never slept, and no second attempt is made.
    """
    server = _Server(
        stall=frozenset({"tools/call"}),
        on_request={"tools/call": _stall_advancing(fake_clock, 1.0)},
    )
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", 0.2)
    runtime = _http_runtime(tmp_path)
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
    assert _other_tasks(before) == set()


@pytest.mark.asyncio
async def test_a_backoff_is_shortened_to_leave_a_useful_attempt(
    tmp_path, monkeypatch, fake_clock, sleeps
):
    """1.3 s left after the failure, minimum 1 s: sleep 0.3 s, then retry."""
    server = _Server(
        stall=frozenset({"tools/call"}),
        on_request={"tools/call": _stall_advancing(fake_clock, 0.2)},
    )
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", 0.2)
    runtime = _http_runtime(tmp_path)

    with tool_call_deadline(1.5):
        with pytest.raises(MCPCallError) as exc_info:
            await asyncio.wait_for(
                runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
                timeout=_STALL_CEILING_SECONDS,
            )

    assert sleeps == [pytest.approx(0.3)]
    assert server.methods.count("tools/call") == 2
    # The second attempt was sent and timed out on its own: not a refusal.
    assert exc_info.value.json_rpc_code == REQUEST_TIMEOUT
    assert "retry" not in exc_info.value.details
    assert server.all_closed()


@pytest.mark.asyncio
async def test_with_no_deadline_the_backoff_is_unchanged(tmp_path, monkeypatch, sleeps):
    server = _Server(stall=frozenset({"tools/call"}))
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", 0.1)
    runtime = _http_runtime(tmp_path)

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
    server = _Server(status={"tools/call": (429, {"retry-after": "2"})})
    server.install(monkeypatch)
    runtime = _http_runtime(tmp_path)

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
    server = _Server(status={"tools/call": (502, {})})
    server.install(monkeypatch)
    runtime = _http_runtime(tmp_path)

    class _OverrunningAsyncio:
        def __getattr__(self, name: str) -> Any:
            return getattr(asyncio, name)

        @staticmethod
        async def sleep(delay: float) -> None:
            # 4 s deadline, 1 s minimum: a 0.5 s sleep is admitted, and
            # returning 3 s late leaves 0.5 s, below the minimum.
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
    server = _Server(stall=frozenset({method}))
    server.install(monkeypatch)
    runtime = _http_runtime(tmp_path)
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
    assert _other_tasks(before) == set()


@pytest.mark.asyncio
async def test_external_cancellation_of_the_catalogue_entrypoint_propagates(
    tmp_path, monkeypatch
):
    server = _Server(stall=frozenset({"tools/list"}))
    server.install(monkeypatch)
    runtime = _http_runtime(tmp_path)
    before = set(asyncio.all_tasks())

    task = asyncio.create_task(runtime.list_tools_for_scope("consultant"))
    await asyncio.wait_for(server.parked_event.wait(), timeout=_STALL_CEILING_SECONDS)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=_STALL_CEILING_SECONDS)

    assert server.parked == server.cancelled == 1
    assert server.all_closed()
    assert _other_tasks(before) == set()


# --- The tool-call deadline: read, never bound --------------------------------


@pytest.mark.asyncio
async def test_mcp_bounds_honour_the_tool_call_deadline(
    tmp_path, monkeypatch, fake_clock, recording_session
):
    server = _Server()
    server.install(monkeypatch)
    runtime = _http_runtime(tmp_path)

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
    server = _Server()
    server.install(monkeypatch)
    monkeypatch.setattr(config, "tool_execution_timeout_seconds", 3.0)
    runtime = _http_runtime(tmp_path)
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
