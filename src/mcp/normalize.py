from __future__ import annotations

import json
from typing import Any

from mcp.types import CallToolResult, TextContent


def _looks_structured_financial(value: Any) -> bool:
    if isinstance(value, dict):
        lowered = {str(key).lower() for key in value.keys()}
        financial_markers = {
            "data",
            "price",
            "symbol",
            "marketcap",
            "revenue",
            "netincome",
            "rsi",
            "volume",
        }
        if lowered & financial_markers:
            return True
        return any(_looks_structured_financial(item) for item in value.values())
    if isinstance(value, list):
        return any(_looks_structured_financial(item) for item in value[:5])
    return isinstance(value, int | float)


def _parse_json_text(text: str) -> Any | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def normalize_result(
    result: CallToolResult,
    server_id: str,
    tool_name: str,
) -> dict[str, Any]:
    """Convert a raw MCP CallToolResult into a stable repo-owned shape."""

    text_parts: list[str] = [
        item.text for item in result.content if isinstance(item, TextContent)
    ]
    combined_text = "\n".join(text_parts).strip()
    parsed_text_json = _parse_json_text(combined_text)
    structured_content = result.structuredContent

    if structured_content is not None:
        payload_profile = (
            "structured_financial"
            if _looks_structured_financial(structured_content)
            else "structured_json"
        )
    elif parsed_text_json is not None:
        payload_profile = (
            "structured_financial"
            if _looks_structured_financial(parsed_text_json)
            else "structured_json"
        )
    else:
        payload_profile = "free_text"

    normalized: dict[str, Any] = {
        "server": server_id,
        "tool": tool_name,
        "is_error": result.isError,
        "payload_profile": payload_profile,
    }
    if structured_content is not None:
        normalized["structured_content"] = structured_content
    if parsed_text_json is not None:
        normalized["parsed_text_json"] = parsed_text_json
    if combined_text:
        normalized["text_content"] = combined_text
    return normalized
