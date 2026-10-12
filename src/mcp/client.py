from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from typing import Any, Literal, NoReturn

import httpx2
import structlog
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import REQUEST_TIMEOUT, CallToolResult, PaginatedRequestParams

from src.config import config as settings_config
from src.error_safety import summarize_exception
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import (
    clamp_to_operational_deadline,
    operational_deadline,
    operational_time_left,
)
from src.mcp.auth import MCPCredentialMissing, MCPResolvedServer, resolve_auth
from src.mcp.budget import BudgetTracker
from src.mcp.config import MCPServerSpec
from src.mcp.errors import (
    HTTPErrorRecord,
    MCPCallError,
    MCPErrorCategory,
    classify_mcp_error,
)
from src.mcp.normalize import normalize_result

logger = structlog.get_logger(__name__)

_DEFAULT_AUTH_COOLDOWN_SECONDS = 300
_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 30
# Total sends per call: the first one plus a single retry.
_MAX_ATTEMPTS = 2


def _spent_allowance_error(
    label: str, *, server_id: str, tool_name: str | None = None
) -> MCPCallError:
    """The caller's or the tool runner's deadline passed before this step could start.

    A non-retryable TRANSPORT error: the MCP category every other timeout
    takes, and not worth a retry that would be refused the same way. Nothing
    was sent for this step.
    """
    return MCPCallError(
        message=f"MCP {label} for {server_id}: enclosing deadline already passed",
        category=MCPErrorCategory.TRANSPORT,
        server_id=server_id,
        tool_name=tool_name,
        retryable=False,
        details={"timeout": "enclosing_deadline_spent", "scope": label},
    )


def _budget_exhausted_error(server_id: str, tool_name: str) -> MCPCallError:
    """The server's per-run or daily call budget is spent; nothing was sent."""
    return MCPCallError(
        message=f"MCP budget exhausted for {server_id}",
        category=MCPErrorCategory.BUDGET,
        server_id=server_id,
        tool_name=tool_name,
    )


def _budget_store_error(
    server_id: str, tool_name: str, exc: BaseException
) -> MCPCallError:
    """The budget store could not admit the call in time, or failed; nothing was sent."""
    return MCPCallError(
        message=f"MCP budget store unavailable for {server_id}",
        category=MCPErrorCategory.BUDGET,
        server_id=server_id,
        tool_name=tool_name,
        details={"refused": "budget_store_unavailable", "reason": type(exc).__name__},
    )


def _is_rate_limited(err: MCPCallError) -> bool:
    return err.category is MCPErrorCategory.TRANSPORT and err.http_status == 429


def _server_wait_seconds(err: MCPCallError) -> int | None:
    """The wait the server asked for before another call, or None.

    Its ``Retry-After`` when it sent a usable one; for a rate limit without
    one, the default rate-limit cooldown. One value, so the rate-limited
    retry and every fresh call (through the cooldown) wait the same.
    """
    if err.retry_after_seconds is not None:
        return err.retry_after_seconds
    if _is_rate_limited(err):
        return _DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
    return None


def _tool_call_allowance() -> float:
    """The tool call's configured allowance, clamped to the time left now."""
    return clamp_to_operational_deadline(settings_config.mcp_tool_call_timeout_seconds)


def _tool_not_listed_error(server_id: str, tool_name: str) -> MCPCallError:
    """The server does not offer ``tool_name``: a registry error, so never sent.

    Sent anyway, the SDK would re-list after the call to look for its output
    schema, under the session's handshake bound, and then skip validation.
    """
    return MCPCallError(
        message=f"MCP tool {tool_name!r} is not offered by {server_id}",
        category=MCPErrorCategory.CONFIG,
        server_id=server_id,
        tool_name=tool_name,
        details={"refused": "tool_not_listed"},
    )


_RetryRefusal = Literal["deadline", "budget"]
_RETRY_REFUSAL_REASONS: dict[_RetryRefusal, str] = {
    "deadline": "deadline leaves no room",
    "budget": "call budget exhausted",
}


def _retry_refused_error(
    err: MCPCallError, *, backoff_seconds: float, reason: _RetryRefusal
) -> MCPCallError:
    """``err`` (sent, then failed), marked as a retry the deadline or budget refused.

    The attempt's own classification is kept -- category, HTTP status, JSON-RPC
    code -- so a 429 still reads as one. ``details.retry`` records that the
    retry was never sent, and it is no longer retryable: a caller retry would
    be refused the same way.
    """
    return MCPCallError(
        message=f"{err.message} (retry refused: {_RETRY_REFUSAL_REASONS[reason]})",
        category=err.category,
        server_id=err.server_id,
        tool_name=err.tool_name,
        retryable=False,
        http_status=err.http_status,
        json_rpc_code=err.json_rpc_code,
        retry_after_seconds=err.retry_after_seconds,
        details={
            **err.details,
            "retry": f"refused_{reason}",
            "backoff_seconds": backoff_seconds,
        },
    )


def _mcp_allowance(default_seconds: float, *, label: str, server_id: str) -> float:
    """A deadline setting, read now and clamped to the caller's and the runner's.

    Read through the operational clamp, so a call made under the tool runner
    (``execute_raw``) sizes against what the runner has left as well as any
    enclosing deadline. A spent allowance raises before any client, subprocess
    or session exists.
    """
    seconds = clamp_to_operational_deadline(default_seconds)
    if seconds <= 0:
        raise _spent_allowance_error(label, server_id=server_id)
    return seconds


async def _lists_tool(session: ClientSession, tool_name: str) -> bool:
    """Whether the server lists ``tool_name``, following its cursor until it does.

    Stops at the page that names the tool, at the last page, or at a cursor
    the server repeats; the caller's time bound covers the rest.
    """
    seen: set[str] = set()
    cursor: str | None = None
    while True:
        page = await session.list_tools(
            params=None if cursor is None else PaginatedRequestParams(cursor=cursor)
        )
        if any(tool.name == tool_name for tool in page.tools):
            return True
        cursor = page.next_cursor
        if cursor is None or cursor in seen:
            return False
        seen.add(cursor)


async def _list_tools_in_time(
    session: ClientSession, *, server_id: str, tool_name: str
) -> MCPCallError | None:
    """List the server's tools until ``tool_name`` appears, within the time left.

    None when the tool is listed; otherwise the error to raise -- the tool is
    not offered, the allowance is spent, or the bound expired. Runs inside the
    session body. The public ``list_tools()`` is called as is: it caches the
    output schema the SDK validates a result against (listed first, the SDK
    never re-lists after the call) and, on a 2026-07-28 session only, drops
    tools with an invalid ``x-mcp-header`` annotation. ``initialize()``
    negotiates a legacy version, where that filter is inert. It takes no
    per-request timeout, and the session's read timeout was fixed when the
    session opened, before the handshake spent part of the allowance, so
    every page is bounded here by what is left now.

    The error is *returned*: the caller lets the SDK contexts exit normally
    and raises it outside them, where the SDK's task groups cannot wrap it.
    External cancellation propagates.
    """
    remaining = clamp_to_operational_deadline(
        settings_config.mcp_initialize_timeout_seconds
    )
    if remaining <= 0:
        return _spent_allowance_error(
            "mcp_list_tools", server_id=server_id, tool_name=tool_name
        )
    bound = asyncio.timeout(remaining)
    try:
        async with bound:
            listed = await _lists_tool(session, tool_name)
        return None if listed else _tool_not_listed_error(server_id, tool_name)
    except TimeoutError:
        if not bound.expired():
            raise
        # Sent, then timed out: the same retryable transport timeout the SDK's
        # own read timeout raises, marked with the bound that fired.
        return MCPCallError(
            message=f"MCP tools/list for {server_id} timed out after {remaining:.2f}s",
            category=MCPErrorCategory.TRANSPORT,
            server_id=server_id,
            tool_name=tool_name,
            retryable=True,
            json_rpc_code=REQUEST_TIMEOUT,
            details={"timeout": "time_left_elapsed", "scope": "mcp_list_tools"},
        )


class MCPRuntime:
    """Run MCP tool calls against configured servers, within their call budgets.

    Each tools/call takes one atomic budget reservation immediately before it
    is sent, and the reservation counts whatever happens after: a request the
    vendor received costs quota even when it times out, fails, or its result
    is discarded (see ``BudgetTracker``).
    """

    def __init__(self, servers: list[MCPServerSpec], budget_db_path: str) -> None:
        self._specs = {spec.id: spec for spec in servers}
        self._resolved: dict[str, MCPResolvedServer] = {}
        self._unavailable: dict[str, str] = {}
        for spec in servers:
            # A missing credential disables that one server; every other server
            # still comes up. Anything else is a malformed registry and should
            # surface loudly.
            try:
                resolved = resolve_auth(spec)
            except MCPCredentialMissing as exc:
                self._unavailable[spec.id] = exc.env_var
                logger.info(
                    "mcp_server_unavailable",
                    server_id=spec.id,
                    reason="credential_not_set",
                    env_var=exc.env_var,
                )
                continue
            if resolved is not None:
                self._resolved[spec.id] = resolved

        self._budget = BudgetTracker(budget_db_path)
        self._cooldowns: dict[str, dt.datetime] = {}

    @property
    def specs(self) -> dict[str, MCPServerSpec]:
        return dict(self._specs)

    @property
    def usable_server_ids(self) -> frozenset[str]:
        """Enabled servers whose credentials actually resolved."""
        return frozenset(self._resolved)

    @property
    def unavailable_servers(self) -> dict[str, str]:
        """Server id -> the env var whose absence disabled it."""
        return dict(self._unavailable)

    @property
    def budget(self) -> BudgetTracker:
        return self._budget

    def _get_spec(self, server_id: str) -> MCPServerSpec:
        try:
            return self._specs[server_id]
        except KeyError as exc:
            raise MCPCallError(
                message=f"MCP server {server_id} is not configured",
                category=MCPErrorCategory.CONFIG,
                server_id=server_id,
            ) from exc

    def _check_cooldown(self, spec: MCPServerSpec) -> None:
        cooldown = self._cooldowns.get(spec.id)
        if cooldown is None:
            return
        now = dt.datetime.now(dt.UTC)
        if now >= cooldown:
            self._cooldowns.pop(spec.id, None)
            return
        remaining = max(int((cooldown - now).total_seconds()), 1)
        raise MCPCallError(
            message=f"MCP server {spec.id} is in cooldown for {remaining}s",
            category=MCPErrorCategory.TRANSPORT,
            server_id=spec.id,
            retryable=True,
            retry_after_seconds=remaining,
        )

    def _register_cooldown(self, server_id: str, *, duration_seconds: int) -> None:
        if duration_seconds <= 0:
            return
        until = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=duration_seconds)
        existing = self._cooldowns.get(server_id)
        if existing is None or until > existing:
            self._cooldowns[server_id] = until

    def _require_access(self, spec: MCPServerSpec, *, scope: str | None) -> None:
        if not spec.enabled:
            raise MCPCallError(
                message=f"MCP server {spec.id} is disabled",
                category=MCPErrorCategory.CONFIG,
                server_id=spec.id,
            )
        if not spec.supports_scope(scope):
            raise MCPCallError(
                message=f"MCP server {spec.id} is not available for scope {scope!r}",
                category=MCPErrorCategory.CONFIG,
                server_id=spec.id,
            )
        self._check_cooldown(spec)

    def _require_tool_allowed(self, spec: MCPServerSpec, tool_name: str) -> None:
        if spec.tool_allowlist and tool_name not in set(spec.tool_allowlist):
            raise MCPCallError(
                message=f"MCP tool {tool_name!r} is not allowlisted for {spec.id}",
                category=MCPErrorCategory.CONFIG,
                server_id=spec.id,
                tool_name=tool_name,
            )

    def is_tool_available(
        self,
        server_id: str,
        tool_name: str,
        *,
        scope: str | None = None,
    ) -> bool:
        """Return whether a server/tool is configured for a given agent scope.

        This check intentionally ignores transient cooldown state so tool
        exposure does not flap during short-lived vendor throttles.
        """
        spec = self._specs.get(server_id)
        if spec is None or not spec.enabled:
            return False
        # Enabled is not the same as usable. A server whose credential is absent
        # resolves to nothing, and offering its tools anyway would trade a clean
        # "not available" for a mid-review CONFIG failure that counts against
        # the Consultant's partial-tool-failure ratio.
        if server_id not in self._resolved:
            return False
        if not spec.supports_scope(scope):
            return False
        if spec.tool_allowlist and tool_name not in set(spec.tool_allowlist):
            return False
        return True

    @asynccontextmanager
    async def _open_session(self, spec: MCPServerSpec):
        resolved = self._resolved.get(spec.id)
        if resolved is None:
            raise MCPCallError(
                message=f"MCP server {spec.id} is enabled in config but could not be resolved",
                category=MCPErrorCategory.CONFIG,
                server_id=spec.id,
            )

        # Read now and shrunk to the caller's and the tool runner's deadline
        # before the client, subprocess or session exists, so a spent allowance
        # opens nothing. The SDK applies each one itself: the session read
        # timeout bounds the initialize handshake (connecting included) and
        # raises MCPError(REQUEST_TIMEOUT) after tearing down the request; the
        # httpx2 timeout bounds each phase of each HTTP request. A tools/list
        # request is re-bounded by the time left (``_list_tools_in_time``).
        request_timeout = _mcp_allowance(
            settings_config.mcp_initialize_timeout_seconds,
            label="mcp_initialize",
            server_id=spec.id,
        )
        http_error = HTTPErrorRecord()

        async def _record_http_error(response: httpx2.Response) -> None:
            # Only POSTs carry calls; a 405 on the optional GET stream is normal.
            if response.status_code >= 400 and response.request.method == "POST":
                http_error.status = response.status_code
                http_error.retry_after = response.headers.get("retry-after")

        try:
            if spec.transport == "streamable_http":
                if resolved.url is None:
                    raise MCPCallError(
                        message=f"MCP server {spec.id} has no streamable HTTP URL",
                        category=MCPErrorCategory.CONFIG,
                        server_id=spec.id,
                    )
                # Must be an httpx2 client: mcp 2.x degrades silently (server-sent
                # messages stop arriving) when handed a plain httpx one. No
                # follow_redirects: the SDK ignores it and applies its own
                # same-origin redirect rule. The SDK keeps a supplied client's
                # timeout (its 30 s / 300 s read defaults apply only to a client
                # it builds), so this one is set explicitly.
                http_timeout = _mcp_allowance(
                    settings_config.mcp_http_timeout_seconds,
                    label="mcp_http",
                    server_id=spec.id,
                )
                async with httpx2.AsyncClient(
                    headers=resolved.headers or None,
                    event_hooks={"response": [_record_http_error]},
                    timeout=httpx2.Timeout(http_timeout),
                ) as http_client:
                    async with streamable_http_client(
                        resolved.url,
                        http_client=http_client,
                    ) as (read, write):
                        async with ClientSession(
                            read, write, read_timeout_seconds=request_timeout
                        ) as session:
                            await session.initialize()
                            yield session
                return

            params = StdioServerParameters(
                command=spec.command or "",
                args=spec.args,
                env=resolved.stdio_env or None,
                cwd=resolved.cwd,
            )
            # stdio_client's own shutdown (close stdin, wait, kill the process
            # tree) is shielded and bounded by the SDK, so a timed-out handshake
            # cannot leak the server process.
            async with stdio_client(params) as (read, write):
                async with ClientSession(
                    read, write, read_timeout_seconds=request_timeout
                ) as session:
                    await session.initialize()
                    yield session
        except MCPCallError:
            raise
        except Exception as exc:
            raise classify_mcp_error(
                exc, server_id=spec.id, http_error=http_error
            ) from exc

    def _on_call_failure(self, err: MCPCallError) -> None:
        if err.category is MCPErrorCategory.AUTH:
            self._register_cooldown(
                err.server_id,
                duration_seconds=_DEFAULT_AUTH_COOLDOWN_SECONDS,
            )
        elif _is_rate_limited(err):
            wait = _server_wait_seconds(err)
            if wait is not None:
                self._register_cooldown(err.server_id, duration_seconds=wait)

    async def _admission(
        self,
        admitted: Awaitable[bool],
        spec: MCPServerSpec,
        tool_name: str,
    ) -> MCPCallError | None:
        """None when the budget admits the call; otherwise the refusal, never sent.

        Fails closed: a store that is locked past its bound, saturated, or
        broken refuses the call. The refusal is returned, for a caller inside
        the session body to raise once the session has closed.
        """
        try:
            if await admitted:
                return None
        except (TimeoutError, sqlite3.Error) as exc:
            if operational_time_left() <= 0:
                return _spent_allowance_error(
                    "mcp_budget", server_id=spec.id, tool_name=tool_name
                )
            logger.warning(
                "mcp_budget_store_unavailable",
                server_id=spec.id,
                tool=tool_name,
                **summarize_exception(exc, operation="mcp_budget_admission"),
            )
            return _budget_store_error(spec.id, tool_name, exc)
        logger.warning("mcp_budget_exhausted", server_id=spec.id, tool=tool_name)
        return _budget_exhausted_error(spec.id, tool_name)

    async def _call_in_time(
        self,
        session: ClientSession,
        spec: MCPServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult | MCPCallError:
        """List, reserve, then call, inside the session body, each within the time left.

        Listing first is what keeps the catalogue inside the deadline: the SDK
        validates a result against the tool's output schema and, in a fresh
        session, lists the tools *after* the call to fetch it, under the
        session's read timeout fixed at open. Listed first, under the time
        left, the schema is cached and that post-call request is never sent;
        a tool the listing lacks is refused unsent. A refusal or a timed-out
        listing is returned, never raised, for the caller to raise once the
        session has closed normally.
        """
        refusal = await _list_tools_in_time(
            session, server_id=spec.id, tool_name=tool_name
        )
        if refusal is not None:
            return refusal
        if _tool_call_allowance() <= 0:
            return _spent_allowance_error(
                "mcp_tool_call", server_id=spec.id, tool_name=tool_name
            )
        # Reserved immediately before the send. An admitted reservation is
        # counted whatever follows -- the call times out, fails, is discarded
        # by the tool plane, or is never sent -- and is never refunded.
        refusal = await self._admission(
            self._budget.reserve(spec.id, spec), spec, tool_name
        )
        if refusal is not None:
            return refusal
        # Recomputed after admission: time spent waiting on the store comes out
        # of the call's allowance. Explicit, because the call would otherwise
        # inherit the session's handshake bound.
        call_timeout = _tool_call_allowance()
        if call_timeout <= 0:
            return _spent_allowance_error(
                "mcp_tool_call", server_id=spec.id, tool_name=tool_name
            )
        result: CallToolResult = await session.call_tool(
            tool_name, arguments, read_timeout_seconds=call_timeout
        )
        return result

    async def _attempt(
        self, spec: MCPServerSpec, tool_name: str, arguments: dict[str, Any]
    ) -> CallToolResult | MCPCallError:
        """One session: the result, or the classified error. External cancellation propagates."""
        try:
            async with self._open_session(spec) as session:
                return await self._call_in_time(session, spec, tool_name, arguments)
        except MCPCallError as err:
            return err
        except Exception as exc:
            return classify_mcp_error(exc, server_id=spec.id, tool_name=tool_name)

    def _refuse_retry(
        self, err: MCPCallError, *, backoff_seconds: float, reason: _RetryRefusal
    ) -> NoReturn:
        """Raise ``err`` as a refused retry; its cooldown was registered already."""
        logger.info(
            "mcp_retry_refused",
            server_id=err.server_id,
            tool_name=err.tool_name,
            category=err.category.value,
            backoff_seconds=backoff_seconds,
            reason=reason,
        )
        raise _retry_refused_error(
            err, backoff_seconds=backoff_seconds, reason=reason
        ) from err

    async def _execute_with_retry(
        self,
        spec: MCPServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        # Advisory, before any session opens: a spent budget opens nothing.
        # The reservation made just before the send is the authority.
        refusal = await self._admission(
            self._budget.has_room(spec.id, spec), spec, tool_name
        )
        if refusal is not None:
            raise refusal
        # The failed attempt being retried, and the backoff slept before it.
        retrying: tuple[MCPCallError, float] | None = None
        for attempt in range(_MAX_ATTEMPTS):
            outcome = await self._attempt(spec, tool_name, arguments)
            if not isinstance(outcome, MCPCallError):
                return outcome
            if retrying is not None and outcome.category is MCPErrorCategory.BUDGET:
                # The retry lost the budget at its reservation: report the
                # attempt that was sent, marked as a retry never sent.
                sent_err, backoff = retrying
                self._refuse_retry(sent_err, backoff_seconds=backoff, reason="budget")
            # Before the retry decision, so fresh calls see a rate limit at once.
            self._on_call_failure(outcome)
            if not outcome.retryable or attempt + 1 >= _MAX_ATTEMPTS:
                raise outcome
            backoff = await self._retry_backoff(outcome, spec, tool_name, attempt)
            retrying = (outcome, backoff)
        raise RuntimeError("unreachable retry exit")  # pragma: no cover

    async def _retry_backoff(
        self, err: MCPCallError, spec: MCPServerSpec, tool_name: str, attempt: int
    ) -> float:
        """Sleep before a retry of ``err`` and return the backoff, or refuse the retry.

        The retry is admitted like an LLM retry: sleep only while a useful
        attempt still fits before the caller's or the runner's deadline. A
        transient backoff is shortened to fit; a wait the server asked for is
        honoured in full or refused, never shortened. Refused means never sent.
        """
        server_wait = _server_wait_seconds(err)
        # Exponential, with no cap of its own: wait_allowance below already
        # shortens a transient backoff to the time left.
        backoff = float(server_wait) if server_wait is not None else 0.5 * 2**attempt
        refusal = await self._admission(
            self._budget.has_room(spec.id, spec), spec, tool_name
        )
        if refusal is not None:
            reason: _RetryRefusal = (
                "budget" if refusal.category is MCPErrorCategory.BUDGET else "deadline"
            )
            self._refuse_retry(err, backoff_seconds=backoff, reason=reason)
        minimum = llm_timeouts.min_useful_seconds(
            settings_config.mcp_tool_call_timeout_seconds, settings_config
        )
        wait = llm_timeouts.wait_allowance(
            operational_deadline(),
            backoff,
            minimum=minimum,
            rate_limited=server_wait is not None,
        )
        if wait is None:
            self._refuse_retry(err, backoff_seconds=backoff, reason="deadline")
        await asyncio.sleep(wait)
        # Re-checked after the sleep: it may overrun, and the next attempt
        # must not open a session it cannot use.
        if operational_time_left() < minimum:
            self._refuse_retry(err, backoff_seconds=backoff, reason="deadline")
        return backoff

    async def execute_raw(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        scope: str | None = None,
        agent_key: str | None = None,
    ) -> dict[str, Any]:
        """Execute an MCP tool and return the normalized payload.

        Performs config/scope/allowlist/cooldown checks, reserves the call budget
        immediately before each tools/call, and retries a retryable transport
        error. It does **not** inspect content: run it as
        the runner of a ``ToolExecutionService`` call, whose hooks do.

        ``agent_key`` is accepted from callers but not used here; observability
        metadata travels via the surrounding :class:`ToolInvocation` instead.
        """
        del agent_key
        spec = self._get_spec(server_id)
        self._require_access(spec, scope=scope)
        self._require_tool_allowed(spec, tool_name)
        result = await self._execute_with_retry(spec, tool_name, arguments)
        return normalize_result(result, server_id=server_id, tool_name=tool_name)
