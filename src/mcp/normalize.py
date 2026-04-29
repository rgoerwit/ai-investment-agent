from __future__ import annotations

import json
from typing import Any

from mcp.types import CallToolResult, TextContent


def normalize_result(
    result: CallToolResult,
    server_id: str,
    tool_name: str,
) -> dict[str, Any]:
    """Convert a raw MCP CallToolResult into a dict suitable for inspection."""

    text_parts: list[str] = []
    for item in result.content:
        if isinstance(item, TextContent):
            text_parts.append(item.text)

    combined = "\n".join(text_parts)

    try:
        parsed = json.loads(combined)
        return {
            "server": server_id,
            "tool": tool_name,
            "result": parsed,
            "is_error": result.isError,
        }
    except (json.JSONDecodeError, TypeError):
        return {
            "server": server_id,
            "tool": tool_name,
            "text": combined,
            "is_error": result.isError,
        }
