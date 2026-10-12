"""MCP call budgets: one atomic reservation per tools/call, made just before it is sent.

The budget used to be counted by a tool-plane ``after()`` hook, which runs only
when the runner succeeds and the combined deadline still has room. A call the
vendor received and then timed out, failed, or whose result the tool plane
discarded was never counted, and a retried call counted once for two sends.
Then the runtime checked (``can_call``) and recorded in two transactions with
the session handshake between them, on the event loop: concurrent callers all
passed the check and each was sent past the limit.

The runtime now makes one atomic reservation (``BudgetTracker.reserve``) off the
event loop, immediately before each tools/call, and counts admitted
reservations: never refunded, so a reservation whose call is then not sent
(cancelled, deadline spent, a worker that committed after its caller gave up)
still counts. An advisory check before the session opens keeps a spent budget
from opening one. Each test checks the counts against the tools/call requests
the fake server actually received.

The real mcp ``ClientSession`` runs over an ``httpx2.MockTransport``; nothing
leaves the process.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from typing import Any

import pytest

import src.mcp.client as mcp_client
from src.blocking_io import InFlightLimit
from src.config import config
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline, tool_call_deadline
from src.mcp import budget as budget_module
from src.mcp.client import MCPRuntime
from src.mcp.errors import MCPCallError, MCPErrorCategory
from src.tooling.runtime import ToolExecutionService, ToolInvocation, ToolResult
from tests.helpers.mcp_server import FakeMCPServer, FrozenWallClock, http_runtime
from tests.helpers.stalled_workers import stalled_worker

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
def wall_clock(monkeypatch) -> FrozenWallClock:
    return FrozenWallClock(monkeypatch)


@pytest.fixture
def sleeps(monkeypatch, wall_clock: FrozenWallClock) -> list[float]:
    """The MCP client's backoff sleeps, recorded and skipped; the wall clock moves."""
    recorded: list[float] = []

    class _NoSleep:
        def __getattr__(self, name: str) -> Any:
            return getattr(asyncio, name)

        @staticmethod
        async def sleep(delay: float) -> None:
            recorded.append(delay)
            wall_clock.advance(delay)

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


# --- Atomic reservation -----------------------------------------------------------


_BUDGET_WORKERS = "mcp_budget."


def _stall_reservations(
    runtime: MCPRuntime, monkeypatch, release: threading.Event, *, on_entry: Any = None
) -> threading.Event:
    """Park each reservation's worker on *release*; then reserve for real.

    Returns the event set when a worker has entered. The caller owns *release*
    through ``stalled_worker``, which joins the workers on every path.
    """
    entered = threading.Event()
    real = runtime.budget.try_reserve

    def stalled(*args: Any, **kwargs: Any) -> bool:
        if on_entry is not None:
            on_entry()
        entered.set()
        release.wait(_STALL_CEILING_SECONDS)
        return real(*args, **kwargs)

    monkeypatch.setattr(runtime.budget, "try_reserve", stalled)
    return entered


def _store_refusal(err: MCPCallError) -> None:
    assert err.category is MCPErrorCategory.BUDGET
    assert err.retryable is False
    assert err.details["refused"] == "budget_store_unavailable"


@pytest.mark.asyncio
async def test_quota_one_with_concurrent_callers_sends_exactly_one_call(
    tmp_path, monkeypatch
):
    server = FakeMCPServer()
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path, daily_call_limit=1)

    results = await asyncio.wait_for(
        asyncio.gather(
            *(
                runtime.execute_raw("fmp_remote", "quote", {}, scope="consultant")
                for _ in range(5)
            ),
            return_exceptions=True,
        ),
        timeout=_STALL_CEILING_SECONDS,
    )

    refused = [r for r in results if isinstance(r, BaseException)]
    assert len(refused) == 4
    assert all(
        isinstance(r, MCPCallError) and r.category is MCPErrorCategory.BUDGET
        for r in refused
    )
    assert server.calls_sent == _run_count(runtime) == _daily_count(runtime) == 1


@pytest.mark.asyncio
async def test_a_retry_that_loses_the_last_slot_to_a_fresh_call_is_refused_unsent(
    tmp_path, monkeypatch
):
    """A 503, and during its backoff a fresh call takes the last of two slots."""
    server = FakeMCPServer(status={"tools/call": [(503, {}), None]})
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path, per_run_limit=2)
    fresh: list[dict[str, Any]] = []

    class _FreshCallDuringBackoff:
        def __getattr__(self, name: str) -> Any:
            return getattr(asyncio, name)

        @staticmethod
        async def sleep(delay: float) -> None:
            fresh.append(await _execute(runtime))

    monkeypatch.setattr(mcp_client, "asyncio", _FreshCallDuringBackoff())

    with pytest.raises(MCPCallError) as exc_info:
        await _execute(runtime)

    err = exc_info.value
    assert fresh[0]["is_error"] is False
    # The sent attempt's classification survives; its retry was never sent.
    assert err.http_status == 503
    assert err.retryable is False
    assert err.details["retry"] == "refused_budget"
    assert server.calls_sent == _run_count(runtime) == _daily_count(runtime) == 2


@pytest.mark.asyncio
async def test_cancellation_during_admission_sends_nothing_and_the_late_commit_counts(
    tmp_path, monkeypatch
):
    server = FakeMCPServer()
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)
    with stalled_worker(_BUDGET_WORKERS) as release:
        entered = _stall_reservations(runtime, monkeypatch, release)
        task = asyncio.create_task(_execute(runtime))
        # A bounded wait in the thread, so the helper thread always ends.
        assert await asyncio.to_thread(entered.wait, _STALL_CEILING_SECONDS)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=_STALL_CEILING_SECONDS)

    assert server.calls_sent == 0
    # Admitted reservations are counted, never refunded.
    assert _run_count(runtime) == _daily_count(runtime) == 1


@pytest.mark.asyncio
async def test_a_reservation_that_outlives_its_bound_is_refused_unsent_and_counts(
    tmp_path, monkeypatch
):
    server = FakeMCPServer()
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_http_timeout_seconds", 0.2)
    runtime = http_runtime(tmp_path)
    with stalled_worker(_BUDGET_WORKERS) as release:
        _stall_reservations(runtime, monkeypatch, release)
        with pytest.raises(MCPCallError) as exc_info:
            await _execute(runtime)

    _store_refusal(exc_info.value)
    assert server.calls_sent == 0
    # The worker committed after its caller had given up: that still counts.
    assert _run_count(runtime) == _daily_count(runtime) == 1


@pytest.mark.asyncio
async def test_a_deadline_spent_during_admission_sends_nothing_and_counts(
    tmp_path, monkeypatch, fake_clock
):
    """The allowance is recomputed after admission, not reused from before it."""
    server = FakeMCPServer()
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    def spend_the_deadline() -> None:
        fake_clock[0] += 5.0

    with stalled_worker(_BUDGET_WORKERS) as release:
        _stall_reservations(runtime, monkeypatch, release, on_entry=spend_the_deadline)
        release.set()
        with enclosing_deadline(5.0):
            with pytest.raises(MCPCallError) as exc_info:
                await _execute(runtime)

    assert exc_info.value.details == {
        "timeout": "enclosing_deadline_spent",
        "scope": "mcp_tool_call",
    }
    assert server.calls_sent == 0
    assert _run_count(runtime) == _daily_count(runtime) == 1


@pytest.mark.asyncio
async def test_admission_waits_on_a_locked_store_off_the_event_loop(
    tmp_path, monkeypatch
):
    """The loop stays free while admission waits: it releases the lock itself."""
    server = FakeMCPServer()
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)
    holder = sqlite3.connect(runtime.budget._db_path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    heartbeats = 0
    try:
        task = asyncio.create_task(_execute(runtime))
        while budget_module._BUDGET_DB_IN_FLIGHT.in_flight == 0:
            heartbeats += 1
            await asyncio.sleep(0.005)
            assert heartbeats < _STALL_CEILING_SECONDS / 0.005
        assert not task.done()
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    payload = await asyncio.wait_for(task, timeout=_STALL_CEILING_SECONDS)

    assert payload["is_error"] is False
    assert server.calls_sent == _run_count(runtime) == _daily_count(runtime) == 1


@pytest.mark.asyncio
async def test_a_store_locked_past_its_busy_timeout_fails_closed_unsent(
    tmp_path, monkeypatch
):
    """SQLite's own timeout, a margin inside the hard bound, fires first."""
    server = FakeMCPServer()
    server.install(monkeypatch)
    monkeypatch.setattr(config, "mcp_http_timeout_seconds", 0.2)
    monkeypatch.setattr(config, "deadline_margin_fraction", 0.5)
    runtime = http_runtime(tmp_path)
    holder = sqlite3.connect(runtime.budget._db_path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    with stalled_worker(_BUDGET_WORKERS):
        try:
            with pytest.raises(MCPCallError) as exc_info:
                await _execute(runtime)
        finally:
            holder.execute("ROLLBACK")
            holder.close()

    _store_refusal(exc_info.value)
    assert exc_info.value.details["reason"] == "OperationalError"
    assert server.methods == []
    assert _run_count(runtime) == 0


@pytest.mark.asyncio
async def test_a_broken_store_fails_closed_without_its_raw_error(tmp_path, monkeypatch):
    server = FakeMCPServer()
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)
    runtime.budget._db_path.write_bytes(b"this is not a sqlite database" * 64)

    with pytest.raises(MCPCallError) as exc_info:
        await _execute(runtime)

    err = exc_info.value
    _store_refusal(err)
    # The SQLite text ("file is not a database") never reaches the message.
    assert err.message == "MCP budget store unavailable for fmp_remote"
    assert server.methods == []


@pytest.mark.asyncio
async def test_a_saturated_store_fails_closed_unsent(tmp_path, monkeypatch):
    server = FakeMCPServer()
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)
    full = InFlightLimit(1)
    assert full.try_acquire()
    monkeypatch.setattr(budget_module, "_BUDGET_DB_IN_FLIGHT", full)

    with pytest.raises(MCPCallError) as exc_info:
        await _execute(runtime)

    _store_refusal(exc_info.value)
    assert exc_info.value.details["reason"] == "BlockingCallSaturatedError"
    assert server.methods == []
    full.release()


# --- A cooldown registered mid-call is honoured immediately before the send ------


class _ListingBarrier:
    """Parks the first tools/list until ``gate`` is set; later listings pass."""

    def __init__(self) -> None:
        self.parked = asyncio.Event()
        self.gate = asyncio.Event()
        self._first = True

    async def __call__(self) -> None:
        if self._first:
            self._first = False
            self.parked.set()
            await self.gate.wait()


async def _execute_in(runtime: MCPRuntime, seconds: float) -> dict[str, Any]:
    with tool_call_deadline(seconds):
        return await _execute(runtime)


_COOLDOWNS = {
    "rate_limited": (429, {"retry-after": "120"}),
    "auth": (401, {}),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", _COOLDOWNS.values(), ids=_COOLDOWNS)
async def test_a_cooldown_registered_while_a_caller_listed_refuses_its_send(
    tmp_path, monkeypatch, fake_clock, wall_clock, failure
):
    barrier = _ListingBarrier()
    server = FakeMCPServer(
        on_request={"tools/list": barrier}, status={"tools/call": failure}
    )
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    # B passes admission (no cooldown yet), then parks inside its listing.
    caller_b = asyncio.create_task(_execute_in(runtime, 5.0))
    await asyncio.wait_for(barrier.parked.wait(), _STALL_CEILING_SECONDS)
    assert "fmp_remote" not in runtime._cooldowns

    # A is sent, fails, and registers the cooldown while B is still listing.
    with pytest.raises(MCPCallError):
        await _execute_in(runtime, 1.5)
    assert "fmp_remote" in runtime._cooldowns
    assert server.calls_sent == 1

    barrier.gate.set()
    with pytest.raises(MCPCallError) as exc_info:
        await caller_b

    err = exc_info.value
    assert "is in cooldown for" in err.message
    assert err.category is MCPErrorCategory.TRANSPORT
    # B was refused before the send; its reservation was taken and stays counted.
    assert server.calls_sent == 1
    assert _run_count(runtime) == _daily_count(runtime) == 2


@pytest.mark.asyncio
async def test_a_cooldown_that_expired_while_a_caller_listed_does_not_refuse_it(
    tmp_path, monkeypatch, fake_clock, wall_clock
):
    barrier = _ListingBarrier()
    server = FakeMCPServer(on_request={"tools/list": barrier})
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)

    caller_b = asyncio.create_task(_execute(runtime))
    await asyncio.wait_for(barrier.parked.wait(), _STALL_CEILING_SECONDS)
    runtime._register_cooldown("fmp_remote", duration_seconds=30)
    wall_clock.advance(31.0)
    barrier.gate.set()

    payload = await caller_b

    assert payload["is_error"] is False
    assert server.calls_sent == 1
    assert "fmp_remote" not in runtime._cooldowns
    assert _run_count(runtime) == 1


@pytest.mark.asyncio
async def test_a_call_is_admitted_once_the_cooldown_has_passed(
    tmp_path, monkeypatch, wall_clock
):
    server = FakeMCPServer()
    server.install(monkeypatch)
    runtime = http_runtime(tmp_path)
    runtime._register_cooldown("fmp_remote", duration_seconds=30)

    with pytest.raises(MCPCallError):
        await _execute(runtime)
    assert server.methods == []

    wall_clock.advance(31.0)
    payload = await _execute(runtime)

    assert payload["is_error"] is False
    assert server.calls_sent == 1
