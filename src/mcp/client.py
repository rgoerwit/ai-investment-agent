from __future__ import annotations

import asyncio
import datetime as dt
import json
from contextlib import asynccontextmanager
from typing import Any

import httpx

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, ListToolsResult
from src.mcp.auth import MCPResolvedServer, resolve_auth
from src.mcp.budget import BudgetTracker
from src.mcp.catalog import ToolCatalog, ToolDescriptor
from src.mcp.config import MCPServerSpec
from src.mcp.errors import (
    MCPCallError,
    MCPErrorCategory,
    classify_mcp_error,
)
from src.mcp.normalize import normalize_result
from src.tooling.inspection_service import INSPECTION_SERVICE
from src.tooling.inspector import InspectionDecision, InspectionEnvelope, SourceKind

_DEFAULT_AUTH_COOLDOWN_SECONDS = 300
_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 30
_MAX_RETRY_ATTEMPTS = 2
_MAX_BACKOFF_SECONDS = 5.0


class MCPRuntime:
    """Run MCP tool calls against configured servers with inspection and budgeting."""

    def __init__(self, servers: list[MCPServerSpec], budget_db_path: str) -> None:
        self._specs = {spec.id: spec for spec in servers}
        self._resolved: dict[str, MCPResolvedServer] = {}
        for spec in servers:
            resolved = resolve_auth(spec)
            if resolved is not None:
                self._resolved[spec.id] = resolved

        self._catalog = ToolCatalog()
        self._budget = BudgetTracker(budget_db_path)
        self._inspection_service = INSPECTION_SERVICE
        self._cooldowns: dict[str, dt.datetime] = {}

    @property
    def specs(self) -> dict[str, MCPServerSpec]:
        return dict(self._specs)

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

        try:
            if spec.transport == "streamable_http":
                if resolved.url is None:
                    raise MCPCallError(
                        message=f"MCP server {spec.id} has no streamable HTTP URL",
                        category=MCPErrorCategory.CONFIG,
                        server_id=spec.id,
                    )
                async with httpx.AsyncClient(
                    headers=resolved.headers or None,
                    follow_redirects=True,
                ) as http_client:
                    async with streamable_http_client(
                        resolved.url,
                        http_client=http_client,
                    ) as (read, write, _):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            yield session
                return

            params = StdioServerParameters(
                command=spec.command or "",
                args=spec.args,
                env=resolved.stdio_env or None,
                cwd=resolved.cwd,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        except MCPCallError:
            raise
        except Exception as exc:
            raise classify_mcp_error(exc, server_id=spec.id) from exc

    def _on_call_failure(self, err: MCPCallError) -> None:
        if err.category is MCPErrorCategory.AUTH:
            self._register_cooldown(
                err.server_id,
                duration_seconds=_DEFAULT_AUTH_COOLDOWN_SECONDS,
            )
        elif err.category is MCPErrorCategory.TRANSPORT and err.http_status == 429:
            duration = err.retry_after_seconds or _DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
            self._register_cooldown(err.server_id, duration_seconds=duration)

    async def _execute_with_retry(
        self,
        spec: MCPServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        last_err: MCPCallError | None = None
        for attempt in range(_MAX_RETRY_ATTEMPTS):
            try:
                async with self._open_session(spec) as session:
                    result: CallToolResult = await session.call_tool(
                        tool_name, arguments
                    )
                    return result
            except MCPCallError as err:
                last_err = err
            except Exception as exc:
                last_err = classify_mcp_error(
                    exc, server_id=spec.id, tool_name=tool_name
                )

            if last_err is None:  # pragma: no cover - belt-and-braces
                raise RuntimeError("unreachable retry state")

            is_last_attempt = attempt + 1 >= _MAX_RETRY_ATTEMPTS
            if not last_err.retryable or is_last_attempt:
                self._on_call_failure(last_err)
                raise last_err

            backoff = (
                last_err.retry_after_seconds
                if last_err.retry_after_seconds is not None
                else 0.5 * (2**attempt)
            )
            await asyncio.sleep(min(float(backoff), _MAX_BACKOFF_SECONDS))

        if last_err is not None:  # pragma: no cover - safety
            raise last_err
        raise RuntimeError("unreachable retry exit")

    async def _inspect(
        self,
        normalized: dict[str, Any],
        *,
        spec: MCPServerSpec,
        tool_name: str,
        agent_key: str | None,
    ) -> tuple[InspectionDecision, Any]:
        envelope = InspectionEnvelope(
            content_text=json.dumps(normalized, default=str),
            raw_content=normalized,
            source_kind=SourceKind.mcp_tool_output,
            source_name=spec.id,
            tool_name=tool_name,
            agent_key=agent_key,
            metadata={
                "transport": spec.transport,
                "trust_tier": spec.trust_tier,
                "payload_profile": normalized.get("payload_profile", "free_text"),
            },
        )
        return await self._inspection_service.evaluate(envelope)

    async def list_tools_for_scope(self, scope: str) -> list[ToolDescriptor]:
        descriptors: list[ToolDescriptor] = []
        for spec in self._specs.values():
            if not spec.enabled or not spec.supports_scope(scope):
                continue

            async with self._open_session(spec) as session:
                tool_result: ListToolsResult = await session.list_tools()

            server_tools = [
                ToolDescriptor(
                    server_id=spec.id,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                    output_schema=tool.outputSchema,
                )
                for tool in tool_result.tools
            ]
            self._catalog.update(spec.id, server_tools)
            descriptors.extend(
                self._catalog.list_for_server(
                    spec.id,
                    allowlist=spec.tool_allowlist,
                )
            )
        return descriptors

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

        Performs config/scope/allowlist/cooldown checks and an N-attempt retry
        on retryable transport errors. Does **not** record budget consumption
        or run content inspection — those are the responsibility of the
        ``ToolExecutionService`` hook chain when MCP calls are routed through
        it.

        ``agent_key`` is accepted for symmetry with :meth:`call_tool` but is
        not used here; observability metadata travels via the surrounding
        :class:`ToolInvocation` instead.
        """
        del agent_key  # future hooks may consume; current path does not need it
        spec = self._get_spec(server_id)
        self._require_access(spec, scope=scope)
        self._require_tool_allowed(spec, tool_name)
        result = await self._execute_with_retry(spec, tool_name, arguments)
        return normalize_result(result, server_id=server_id, tool_name=tool_name)

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        agent_key: str | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Convenience entry point that bundles budget, retry, and inspection.

        Prefer routing through :class:`ToolExecutionService` with
        :meth:`execute_raw` as the runner so audit/argument-policy/content-
        inspection hooks observe MCP calls under their canonical
        ``mcp__<server>__<tool>`` tool names.
        """
        spec = self._get_spec(server_id)
        self._require_access(spec, scope=scope)
        self._require_tool_allowed(spec, tool_name)

        if not self._budget.can_call(server_id, spec):
            raise MCPCallError(
                message=f"MCP budget exhausted for {server_id}",
                category=MCPErrorCategory.BUDGET,
                server_id=server_id,
                tool_name=tool_name,
            )

        result = await self._execute_with_retry(spec, tool_name, arguments)
        normalized = normalize_result(result, server_id=server_id, tool_name=tool_name)

        decision, approved_content = await self._inspect(
            normalized,
            spec=spec,
            tool_name=tool_name,
            agent_key=agent_key,
        )

        if decision.action in ("block", "degrade"):
            self._budget.record_upstream_consumption(server_id)
            raise MCPCallError(
                message="MCP content blocked by inspection",
                category=MCPErrorCategory.INSPECTION,
                server_id=server_id,
                tool_name=tool_name,
                details={"decision": decision.action},
            )

        if normalized.get("is_error"):
            self._budget.record_upstream_consumption(server_id)
            raise MCPCallError(
                message="MCP tool returned an error payload",
                category=MCPErrorCategory.TOOL_ERROR,
                server_id=server_id,
                tool_name=tool_name,
                details=normalized,
            )

        self._budget.record_upstream_consumption(server_id)
        if isinstance(approved_content, dict):
            return approved_content
        return {
            "server": server_id,
            "tool": tool_name,
            "payload_profile": "free_text",
            "text_content": str(approved_content),
            "is_error": False,
        }
