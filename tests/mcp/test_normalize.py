from __future__ import annotations

from mcp.types import CallToolResult, TextContent
from src.mcp.normalize import normalize_result


def test_normalize_result_prefers_structured_content():
    result = CallToolResult(
        content=[TextContent(type="text", text='{"ignored": true}')],
        structuredContent={"data": [{"price": 123.4}]},
        isError=False,
    )

    normalized = normalize_result(result, "server", "tool")

    assert normalized["payload_profile"] == "structured_financial"
    assert normalized["structured_content"] == {"data": [{"price": 123.4}]}


def test_normalize_result_parses_json_text():
    result = CallToolResult(
        content=[TextContent(type="text", text='{"data": [{"rsi": 54.2}]}')],
        structuredContent=None,
        isError=False,
    )

    normalized = normalize_result(result, "server", "tool")

    assert normalized["payload_profile"] == "structured_financial"
    assert normalized["parsed_text_json"] == {"data": [{"rsi": 54.2}]}


def test_normalize_result_marks_free_text_fallback():
    result = CallToolResult(
        content=[TextContent(type="text", text="plain text response")],
        structuredContent=None,
        isError=True,
    )

    normalized = normalize_result(result, "server", "tool")

    assert normalized["payload_profile"] == "free_text"
    assert normalized["text_content"] == "plain text response"
    assert normalized["is_error"] is True
