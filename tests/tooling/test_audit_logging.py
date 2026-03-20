from unittest.mock import patch

import pytest

from src.tooling.audit import LoggingToolAuditHook
from src.tooling.runtime import ToolInvocation, ToolResult


@pytest.mark.asyncio
async def test_audit_logs_error_like_json_tool_outputs():
    hook = LoggingToolAuditHook()
    call = ToolInvocation(
        name="spot_check_metric_alt",
        args={"ticker": "1308.HK"},
        source="consultant",
        agent_key="consultant",
    )
    result = ToolResult(
        value='{"error":"invalid key","provider":"fmp","failure_kind":"auth_error","retryable":false,"fmp_endpoint":"ratios"}'
    )

    with patch("src.tooling.audit.logger") as mock_logger:
        returned = await hook.after(call, result)

    assert returned == result
    mock_logger.warning.assert_called_once()
    kwargs = mock_logger.warning.call_args.kwargs
    assert kwargs["failure_kind"] == "auth_error"
    assert kwargs["provider"] == "fmp"
    assert kwargs["endpoint"] == "ratios"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_kind", "expected_message"),
    [
        ("FETCH_FAILED: HTTP 403", "fetch_failed", "HTTP 403"),
        ("SEARCH_FAILED: rate limit", "search_failed", "rate limit"),
        ("TOOL_ERROR: malformed payload", "tool_error", "malformed payload"),
        ("TOOL_BLOCKED: safety policy", "tool_blocked", "safety policy"),
    ],
)
async def test_audit_logs_prefix_style_tool_failures(
    payload, expected_kind, expected_message
):
    hook = LoggingToolAuditHook()
    call = ToolInvocation(
        name="fetch_reference_content",
        args={"url": "https://example.com"},
        source="editor",
        agent_key="editor",
    )
    result = ToolResult(value=payload)

    with patch("src.tooling.audit.logger") as mock_logger:
        returned = await hook.after(call, result)

    assert returned == result
    mock_logger.warning.assert_called_once()
    kwargs = mock_logger.warning.call_args.kwargs
    assert kwargs["failure_kind"] == expected_kind
    assert kwargs["message"] == expected_message
