from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDescriptor:
    server_id: str
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None


class ToolCatalog:
    """In-memory MCP tool catalog keyed by server id."""

    def __init__(self) -> None:
        self._tools: dict[str, list[ToolDescriptor]] = {}

    def update(self, server_id: str, tools: list[ToolDescriptor]) -> None:
        self._tools[server_id] = tools

    def list_for_server(
        self,
        server_id: str,
        *,
        allowlist: list[str] | None = None,
    ) -> list[ToolDescriptor]:
        tools = self._tools.get(server_id, [])
        if not allowlist:
            return list(tools)
        allowed = set(allowlist)
        return [tool for tool in tools if tool.name in allowed]
