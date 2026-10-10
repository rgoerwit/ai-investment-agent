"""MCP call budgets: every attempt admitted, every sent tools/call counted.

The budget used to be counted by a tool-plane ``after()`` hook, which runs only
when the runner succeeds and the combined deadline still has room. A call the
vendor received and then timed out, failed, or whose result the tool plane
discarded was never counted, and a retried call counted once for two sends.
The runtime now admits each attempt with ``can_call`` before a session opens
and records each tools/call as it is sent. Each test checks the count against
the number of tools/call requests the fake server actually received.

The real mcp ``ClientSession`` runs over an ``httpx2.MockTransport``; nothing
leaves the process.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any

import pytest

import src.mcp.client as mcp_client
from src.config import config
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline
from src.mcp.client import MCPRuntime
from src.mcp.errors import MCPCallError, MCPErrorCategory
from src.tooling.runtime import ToolExecutionService, ToolInvocation, ToolResult
from tests.helpers.mcp_server import FakeMCPServer, http_runtime

_NOW = 1_000.0
_STALL_CEILING_SECONDS = 10.0
_QUOTE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"price": {"type": "number"}},
    "required": ["price"],
}


@pytest.fixture(autouse=True)
def _no_deadline_margin(monkeypatch) -> None:
    monkeypatch.setattr(config, "deadline_margin_fraction", 0.0)


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    clock = [_NOW]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """The MCP client's backoff sleeps, recorded and skipped."""
    recorded: list[float] = []

    class _NoSleep:
        def __getattr__(self, name: str) -> Any:
            return getattr(asyncio, name)

        @staticmethod
        async def sleep(delay: float) -> None:
            recorded.append(delay)

    monkeypatch.setattr(mcp_client, "asyncio", _NoSleep())
    return recorded


def _run_count(runtime: MCPRuntime) -> int:
    return runtime.budget._run_calls.get("fmp_remote", 0)


def _daily_count(runtime: MCPRuntime) -> int:
    with sqlite3.connect(runtime.budget._db_path) as conn:
        row = conn.execute(
            "SELECT call_count FROM mcp_usage WHERE server_id='fmp_remote'"
        ).fetchone()
    return row[0] if row else 0


async def _execute(runtime: MCPRuntime) -> dict[str, Any]:
    return await asyncio.wait_for(
        runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant"),
        timeout=_STALL_CEILING_SECONDS,
    )


# --- Sent means counted, whatever follows -------------------------------------


@pytest.mark.asyncio
async def test_a_successful_call_counts_once_per_run_and_per_day(tmp_path, monkeypatch):
    server = FakeMCPServer()
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    await _execute(runtime)

    assert server.calls_sent == 1
    assert _run_count(runtime) == _daily_count(runtime) == 1


_FAILED_CALLS = {
    "error_payload": {
        "call_result": {
            "content": [{"type": "text", "text": "provider said no"}],
            "isError": True,
        }
    },
    "invalid_structured_content": {
        "tools": [
            {
                "name": "quote",
                "inputSchema": {"type": "object"},
                "outputSchema": _QUOTE_OUTPUT_SCHEMA,
            }
        ],
        "call_result": {
            "content": [{"type": "text", "text": "{}"}],
            "structuredContent": {"price": "not-a-number"},
            "isError": False,
        },
    },
    "http_429": {"status": {"tools/call": (429, {"retry-after": "1"})}},
    "http_503": {"status": {"tools/call": (503, {})}},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("server_kwargs", _FAILED_CALLS.values(), ids=_FAILED_CALLS)
async def test_a_call_the_vendor_received_counts_even_when_it_fails(
    tmp_path, monkeypatch, sleeps, server_kwargs
):
    server = FakeMCPServer(**server_kwargs)
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    try:
        await _execute(runtime)
    except MCPCallError:
        pass

    # A retryable failure is retried, and the retry counts too.
    assert server.calls_sent >= 1
    assert _run_count(runtime) == _daily_count(runtime) == server.calls_sent


@pytest.mark.asyncio
async def test_a_timed_out_call_counts(tmp_path, monkeypatch, sleeps):
    server = FakeMCPServer(stall=frozenset({"tools/call"}))
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_tool_call_timeout_seconds", 0.05)
    runtime = http_runtime(tmp_path)

    with pytest.raises(MCPCallError) as exc_info:
        await _execute(runtime)

    assert exc_info.value.category is MCPErrorCategory.TRANSPORT
    assert server.calls_sent == 2
    assert server.parked == server.cancelled == 2
    assert _run_count(runtime) == 2


@pytest.mark.asyncio
async def test_a_retry_that_succeeds_counts_both_sends(tmp_path, monkeypatch, sleeps):
    server = FakeMCPServer(status={"tools/call": [(503, {}), None]})
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    payload = await _execute(runtime)

    assert payload["is_error"] is False
    assert sleeps == [0.5]
    assert server.calls_sent == 2
    assert _run_count(runtime) == _daily_count(runtime) == 2


# --- Never sent means never counted -------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_kwargs",
    [
        {"status": {"initialize": (401, {})}},
        {"status": {"tools/list": (500, {})}},
        {"pages": [[]]},
    ],
    ids=["handshake_refused", "listing_failed", "tool_not_listed"],
)
async def test_an_attempt_that_never_sends_the_call_counts_nothing(
    tmp_path, monkeypatch, sleeps, server_kwargs
):
    server = FakeMCPServer(**server_kwargs)
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    with pytest.raises(MCPCallError):
        await _execute(runtime)

    assert server.calls_sent == 0
    assert _run_count(runtime) == _daily_count(runtime) == 0


@pytest.mark.asyncio
async def test_a_spent_allowance_counts_nothing(tmp_path, monkeypatch, fake_clock):
    server = FakeMCPServer()
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    with enclosing_deadline(1.0):
        fake_clock[0] += 1.0
        with pytest.raises(MCPCallError):
            await _execute(runtime)

    assert server.methods == []
    assert _run_count(runtime) == 0


# --- Admission ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("limit_field", ["per_run_limit", "daily_call_limit"])
async def test_a_spent_budget_opens_no_session(tmp_path, monkeypatch, limit_field):
    server = FakeMCPServer()
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path, **{limit_field: 2})

    await _execute(runtime)
    await _execute(runtime)  # exactly at the limit: still admitted
    with pytest.raises(MCPCallError) as exc_info:
        await _execute(runtime)

    err = exc_info.value
    assert err.category is MCPErrorCategory.BUDGET
    assert err.retryable is False
    assert err.message == "MCP budget exhausted for fmp_remote"
    assert server.calls_sent == 2
    assert server.methods.count("initialize") == 2
    assert _run_count(runtime) == 2


@pytest.mark.asyncio
async def test_the_daily_budget_survives_a_new_runtime(tmp_path, monkeypatch):
    server = FakeMCPServer()
    server.install(monkeypatch)
    await _execute(http_runtime(tmp_path, daily_call_limit=1))

    fresh = http_runtime(tmp_path, daily_call_limit=1)
    with pytest.raises(MCPCallError) as exc_info:
        await _execute(fresh)

    assert exc_info.value.category is MCPErrorCategory.BUDGET
    assert server.calls_sent == 1


@pytest.mark.asyncio
async def test_a_zero_limit_means_unlimited(tmp_path, monkeypatch):
    server = FakeMCPServer()
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path, per_run_limit=0, daily_call_limit=0)

    for _ in range(3):
        await _execute(runtime)

    assert server.calls_sent == _run_count(runtime) == 3


@pytest.mark.asyncio
async def test_a_retry_the_budget_cannot_cover_is_refused_unsent(
    tmp_path, monkeypatch, sleeps
):
    server = FakeMCPServer(status={"tools/call": (503, {})})
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path, per_run_limit=1)

    with pytest.raises(MCPCallError) as exc_info:
        await _execute(runtime)

    err = exc_info.value
    # The sent attempt's classification survives; the retry was never sent.
    assert err.http_status == 503
    assert err.retryable is False
    assert err.details["retry"] == "refused_budget"
    assert isinstance(err.__cause__, MCPCallError)
    assert sleeps == []
    assert len(server.clients) == 1
    assert server.calls_sent == _run_count(runtime) == 1


# --- Through the tool plane -----------------------------------------------------


class _PassThroughHook:
    def __init__(self) -> None:
        self.after_calls = 0

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        self.after_calls += 1
        return result


@pytest.mark.asyncio
async def test_a_result_the_tool_plane_discards_is_still_counted(
    tmp_path, monkeypatch, fake_clock
):
    """The original undercount: the call returns after the combined deadline.

    The service discards the result before any after() hook runs -- so the old
    budget hook never counted a call the vendor had answered.
    """

    def answer_after_the_deadline() -> None:
        fake_clock[0] += 1_000.0

    server = FakeMCPServer(on_request={"tools/call": answer_after_the_deadline})
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)
    hook = _PassThroughHook()

    async def runner(args: dict[str, Any]) -> Any:
        return await runtime.execute_raw(
            "fmp_remote", "quote", args, scope="consultant"
        )

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            ToolExecutionService([hook]).execute(
                ToolInvocation(
                    name="mcp__fmp_remote__quote", args={}, source="consultant"
                ),
                runner=runner,
            ),
            timeout=_STALL_CEILING_SECONDS,
        )

    assert hook.after_calls == 0
    assert server.calls_sent == _run_count(runtime) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "counted"), [("initialize", 0), ("tools/list", 0), ("tools/call", 1)]
)
async def test_external_cancellation_counts_only_a_sent_call(
    tmp_path, monkeypatch, method, counted
):
    server = FakeMCPServer(stall=frozenset({method}))
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    task = asyncio.create_task(
        runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant")
    )
    await asyncio.wait_for(server.parked_event.wait(), timeout=_STALL_CEILING_SECONDS)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=_STALL_CEILING_SECONDS)

    assert server.calls_sent == _run_count(runtime) == counted
