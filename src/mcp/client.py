from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult

from src.mcp.auth import MCPResolvedServer, resolve_auth
from src.mcp.budget import BudgetTracker
from src.mcp.catalog import ToolCatalog
from src.mcp.config import MCPServerSpec
from src.mcp.errors import MCPCallError
from src.mcp.normalize import normalize_result
from src.tooling.inspection_service import INSPECTION_SERVICE
from src.tooling.inspector import InspectionDecision, InspectionEnvelope, SourceKind

logger = logging.getLogger(__name__)


class MCPRuntime:
    """Runs MCP tool calls against configured servers, with inspection and budgeting."""

    def __init__(self, servers: list[MCPServerSpec], budget_db_path: str) -> None:
        self._specs = {s.id: s for s in servers}
        self._resolved: dict[str, MCPResolvedServer] = {}
        for s in servers:
            resolved = resolve_auth(s)
            if resolved is not None:
                self._resolved[s.id] = resolved

        self._catalog = ToolCatalog()
        self._budget = BudgetTracker(budget_db_path)
        self._inspection_service = INSPECTION_SERVICE

    # ------------------------------------------------------------------
    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        agent_key: str | None = None,
    ) -> str:
        """Execute one MCP tool call and return a compact JSON string."""
        resolved = self._resolved.get(server_id)
        if resolved is None:
            raise MCPCallError(
                f"MCP server {server_id} is not enabled or configured",
                "config",
                server_id,
                tool_name,
                retryable=False,
            )

        spec = self._specs[server_id]
        if not self._budget.can_call(server_id, spec):
            return json.dumps(
                {"error": "budget_exhausted", "server": server_id, "tool": tool_name}
            )

        try:
            async with streamable_http_client(
                resolved.url, headers=resolved.headers
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result: CallToolResult = await session.call_tool(
                        tool_name, arguments
                    )
        except Exception as exc:
            raise MCPCallError(
                str(exc), "transport", server_id, tool_name, retryable=True
            ) from exc

        # Normalize into repo‑owned structure
        normalized = normalize_result(result, server_id=server_id, tool_name=tool_name)

        # Run content inspection *before* allowing content into prompt context
        decision = await self._inspect(
            normalized,
            server_id=server_id,
            tool_name=tool_name,
            agent_key=agent_key,
        )
        if decision.action == "block":
            raise MCPCallError(
                "Content blocked by inspection",
                "inspection",
                server_id,
                tool_name,
                retryable=False,
            )

        # Apply sanitized content if available
        if (
            decision.action == "sanitize"
            and decision.sanitized_content is not None
        ):
            return decision.sanitized_content

        # Record usage after a successful call
        self._budget.record_call(server_id)
        return json.dumps(normalized, default=str)

    # ------------------------------------------------------------------
    async def _inspect(
        self,
        normalized: dict[str, Any],
        *,
        server_id: str,
        tool_name: str,
        agent_key: str | None = None,
    ) -> InspectionDecision:
        text = json.dumps(normalized, default=str)
        envelope = InspectionEnvelope(
            content_text=text,
            raw_content=normalized,
            source_kind=SourceKind.mcp_tool_output,
            source_name=server_id,
            tool_name=tool_name,
            agent_key=agent_key,
            metadata={
                "transport": self._specs.get(server_id, MCPServerSpec(id="", description="", transport="streamable_http")).transport,
                "trust_tier": self._specs.get(server_id, MCPServerSpec(id="", description="", transport="streamable_http")).trust_tier,
                "payload_profile": "structured_financial"
                if isinstance(normalized, dict)
                else "free_text",
            },
        )
        return await self._inspection_service.inspect(envelope)
