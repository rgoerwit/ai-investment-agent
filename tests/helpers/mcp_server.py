"""A fake streamable-HTTP MCP server and the real SDK session, recorded.

Shared by the MCP suites that drive the real ``ClientSession`` over an
``httpx2.MockTransport``: no request leaves the process.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
from mcp import ClientSession

import src.mcp.client as mcp_client
from src.llm_runtime import timeouts as llm_timeouts
from src.mcp.client import MCPRuntime
from src.mcp.config import MCPServerSpec

URL = "https://example.test/mcp"


def http_spec(*, allowlist: list[str] | None = None, **fields: Any) -> MCPServerSpec:
    return MCPServerSpec(
        id="fmp_remote",
        description="FMP",
        transport="streamable_http",
        base_url=URL,
        scopes=["consultant"],
        tool_allowlist=allowlist if allowlist is not None else ["quote"],
        **fields,
    )


def http_runtime(tmp_path: Path, **spec_fields: Any) -> MCPRuntime:
    return MCPRuntime(
        [http_spec(**spec_fields)], budget_db_path=str(tmp_path / "mcp.db")
    )


class FrozenWallClock:
    """The UTC clock the cooldowns read, frozen at ``start``; ``advance`` moves it.

    A skipped sleep must advance it, or a retry would meet its own cooldown.
    """

    def __init__(self, monkeypatch, start: dt.datetime | None = None) -> None:
        frozen = start or dt.datetime.now(dt.UTC)
        self.elapsed = 0.0

        clock = self

        class _Datetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return frozen + dt.timedelta(seconds=clock.elapsed)

        monkeypatch.setattr(
            mcp_client,
            "dt",
            types.SimpleNamespace(
                UTC=dt.UTC, timedelta=dt.timedelta, datetime=_Datetime
            ),
        )

    def advance(self, seconds: float) -> None:
        self.elapsed += seconds


def other_tasks(before: set[asyncio.Task]) -> set[asyncio.Task]:
    """Tasks started since ``before`` that are still alive."""
    return {
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and task not in before
    }


class FakeMCPServer:
    """A streamable-HTTP MCP server behind ``httpx2.MockTransport``.

    ``on_request`` runs a hook per method when the request arrives (advance a
    fake clock, sleep for real). Methods in ``stall`` park until ``release``
    is set, counting parked and cancelled requests. ``status`` answers a
    method with a bare HTTP error instead; a list value answers successive
    requests in turn, ``None`` meaning a normal answer. ``pages`` splits the
    tool listing over cursored pages; ``next_cursors`` overrides the cursor
    each page returns (to make one repeat).
    """

    def __init__(
        self,
        *,
        stall: frozenset[str] = frozenset(),
        tools: list[dict[str, Any]] | None = None,
        pages: list[list[dict[str, Any]]] | None = None,
        next_cursors: list[str | None] | None = None,
        call_result: dict[str, Any] | None = None,
        status: dict[str, Any] | None = None,
        on_request: dict[str, Callable[[], Any]] | None = None,
    ) -> None:
        self.stall = stall
        tools = tools or [{"name": "quote", "inputSchema": {"type": "object"}}]
        self.pages = pages if pages is not None else [tools]
        self.next_cursors = next_cursors or [
            str(index + 1) if index + 1 < len(self.pages) else None
            for index in range(len(self.pages))
        ]
        self.call_result = call_result or {
            "content": [{"type": "text", "text": "{}"}],
            "isError": False,
        }
        self.status = status or {}
        self.on_request = on_request or {}
        self.methods: list[str] = []
        self.list_cursors: list[str | None] = []
        self.parked = 0
        self.cancelled = 0
        self.parked_event = asyncio.Event()
        self.release = asyncio.Event()
        self.clients: list[httpx2.AsyncClient] = []

    def _status_for(self, method: str) -> tuple[int, dict[str, str]] | None:
        planned = self.status.get(method)
        if isinstance(planned, list):
            return planned.pop(0) if planned else None
        return planned

    def _listing(self, params: dict[str, Any] | None) -> dict[str, Any]:
        cursor = (params or {}).get("cursor")
        self.list_cursors.append(cursor)
        index = 0 if cursor is None else int(cursor)
        result: dict[str, Any] = {
            "tools": self.pages[index],
            # The cache fields are required at 2026-07-28 and dropped as a
            # later revision's fields on a legacy session.
            "ttlMs": 0,
            "cacheScope": "private",
            "resultType": "complete",
        }
        if self.next_cursors[index] is not None:
            result["nextCursor"] = self.next_cursors[index]
        return result

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
        status = self._status_for(method)
        if status is not None:
            code, headers = status
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
            result = self._listing(message.get("params"))
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

    @property
    def calls_sent(self) -> int:
        return self.methods.count("tools/call")


class RecordingSession(ClientSession):
    """The real ClientSession, recording its bounds and the deadlines it ran under.

    Reset the class lists (``reset``) before use: they are shared.
    """

    session_timeouts: list[float | None] = []
    call_timeouts: list[float | None] = []
    deadlines_at_call: list[tuple[float | None, float | None]] = []

    @classmethod
    def reset(cls) -> None:
        cls.session_timeouts = []
        cls.call_timeouts = []
        cls.deadlines_at_call = []

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
