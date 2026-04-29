from __future__ import annotations

from typing import Any


class ToolCatalog:
    """In-memory tool catalog per MCP server, filtered by allowlists."""

    def __init__(self) -> None:
        self._tools: dict[str, list[dict[str, Any]]] = {}

    def update(self, server_id: str, tools: list[dict[str, Any]]) -> None:
        self._tools[server_id] = tools

    def search(self, server_id: str, allowlist: list[str]) -> list[dict[str, Any]]:
        tools = self._tools.get(server_id, [])
        if allowlist:
            names = set(allowlist)
            return [t for t in tools if t.get("name") in names]
        return tools

    def detail(self, server_id: str, tool_name: str) -> dict[str, Any] | None:
        for t in self._tools.get(server_id, []):
            if t.get("name") == tool_name:
                return t
        return None
