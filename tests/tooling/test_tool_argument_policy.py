"""Tests for ToolArgumentPolicyHook — outbound argument guard."""

import pytest

from src.tooling.runtime import ToolCallBlocked, ToolInvocation
from src.tooling.tool_argument_policy import ToolArgumentPolicyHook


def _call(
    name: str = "fetch_reference_content",
    args: dict | None = None,
    source: str = "editor",
) -> ToolInvocation:
    return ToolInvocation(
        name=name,
        args=args or {},
        source=source,
        agent_key="test_agent",
    )


# ---------------------------------------------------------------------------
# Non-editor calls pass through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_editor_passes_through():
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(source="toolnode", args={"url": "file:///etc/passwd"})
    result = await hook.before(call)
    assert result is call  # unmodified


# ---------------------------------------------------------------------------
# fetch_reference_content — URL validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_url_passes():
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "https://www.reuters.com/article/toyota-earnings"})
    result = await hook.before(call)
    assert result is call


@pytest.mark.asyncio
async def test_non_http_scheme_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "ftp://internal.server.com/secret"})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_non_http_scheme_warn_mode():
    hook = ToolArgumentPolicyHook(mode="warn")
    call = _call(args={"url": "ftp://internal.server.com/secret"})
    # Warn mode should not raise — just log and pass through.
    result = await hook.before(call)
    assert result is call


@pytest.mark.asyncio
async def test_long_query_string_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    long_query = "a=" + "x" * 600
    call = _call(args={"url": f"https://example.com/page?{long_query}"})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_webhook_site_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "https://webhook.site/abc-123-def"})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_requestbin_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "https://requestbin.com/r/abc123"})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_ngrok_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "https://abc123.ngrok.io/exfiltrate"})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_no_hostname_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "https://"})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_localhost_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "http://localhost:8000/internal"})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_private_ipv4_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    for url in (
        "http://10.0.0.5/secret",
        "http://172.16.9.1/secret",
        "http://192.168.1.20/secret",
        "http://169.254.169.254/latest/meta-data",
        "http://127.0.0.1:8080/debug",
    ):
        with pytest.raises(ToolCallBlocked):
            await hook.before(_call(args={"url": url}))


@pytest.mark.asyncio
async def test_private_ipv6_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    for url in (
        "http://[::1]/admin",
        "http://[fc00::1]/admin",
        "http://[fe80::1]/admin",
    ):
        with pytest.raises(ToolCallBlocked):
            await hook.before(_call(args={"url": url}))


@pytest.mark.asyncio
async def test_cloud_metadata_hostname_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    for url in (
        "http://metadata.google.internal/computeMetadata/v1",
        "http://metadata.google/computeMetadata/v1",
        "http://instance-data/latest/meta-data",
    ):
        with pytest.raises(ToolCallBlocked):
            await hook.before(_call(args={"url": url}))


# ---------------------------------------------------------------------------
# search_claim — query validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_search_query_passes():
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(
        name="search_claim",
        args={"query": "Toyota Ultraman Card Game launch date 2024"},
    )
    result = await hook.before(call)
    assert result is call


@pytest.mark.asyncio
async def test_oversized_query_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(
        name="search_claim",
        args={"query": "x" * 600},
    )
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_pasted_payload_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    pasted = "line1\nline2\nline3\nline4\nline5"  # 4+ newlines
    call = _call(
        name="search_claim",
        args={"query": pasted},
    )
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_special_char_heavy_query_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    # Over 15% special characters in 60-char query.
    query = "a" * 40 + "{}<>[]" * 4
    call = _call(name="search_claim", args={"query": query})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_oversized_query_warn_mode():
    hook = ToolArgumentPolicyHook(mode="warn")
    call = _call(name="search_claim", args={"query": "x" * 600})
    result = await hook.before(call)
    assert result is call  # Warn mode passes through


# ---------------------------------------------------------------------------
# after() is pass-through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_after_passes_through():
    from src.tooling.runtime import ToolResult

    hook = ToolArgumentPolicyHook(mode="block")
    call = _call()
    result = ToolResult(value="test")
    out = await hook.after(call, result)
    assert out is result


# ---------------------------------------------------------------------------
# Edge cases — empty/missing args
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_url_blocked():
    """Empty URL string has no scheme → blocked."""
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": ""})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_missing_url_key_blocked():
    """Missing 'url' key defaults to empty string → blocked."""
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_url_value_none_blocked():
    """URL value set to None (not missing key) should be blocked."""
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": None})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_data_uri_blocked():
    """data: URI scheme should be blocked."""
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "data:text/html,<script>alert(1)</script>"})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_javascript_uri_blocked():
    """javascript: URI scheme should be blocked."""
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "javascript:alert(document.cookie)"})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_file_uri_blocked():
    """file: URI scheme should be blocked."""
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "file:///etc/passwd"})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


# ---------------------------------------------------------------------------
# Non-guarded tools pass through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_other_editor_tool_passes_through():
    """Editor tools other than fetch_reference_content/search_claim pass through."""
    hook = ToolArgumentPolicyHook(mode="block")
    call = ToolInvocation(
        name="get_news",
        args={"query": "x" * 1000},  # Would fail if checked
        source="editor",
        agent_key="test_agent",
    )
    result = await hook.before(call)
    assert result is call


# ---------------------------------------------------------------------------
# search_claim edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_length_query_passes():
    """Query under 500 chars without pasted-payload signals passes."""
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(
        name="search_claim",
        args={"query": "Toyota Q3 earnings 2024 analyst consensus estimate"},
    )
    result = await hook.before(call)
    assert result is call


@pytest.mark.asyncio
async def test_query_exactly_500_passes():
    """Query at exactly 500 chars should pass (boundary)."""
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(
        name="search_claim",
        args={"query": "a" * 500},
    )
    result = await hook.before(call)
    assert result is call


@pytest.mark.asyncio
async def test_query_501_blocked():
    """Query at 501 chars should be blocked (just over limit)."""
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(
        name="search_claim",
        args={"query": "a" * 501},
    )
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_empty_search_query_passes():
    """Empty search query passes (not suspicious, just useless)."""
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(name="search_claim", args={"query": ""})
    result = await hook.before(call)
    assert result is call


@pytest.mark.asyncio
async def test_missing_query_key_passes():
    """Missing 'query' key defaults to '' → passes (empty is not suspicious)."""
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(name="search_claim", args={})
    result = await hook.before(call)
    assert result is call


# ---------------------------------------------------------------------------
# URL edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_url_with_port_passes():
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "https://example.com:8080/page"})
    result = await hook.before(call)
    assert result is call


@pytest.mark.asyncio
async def test_url_with_auth_passes():
    """URL with basic auth in netloc should pass if otherwise valid."""
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "https://user:pass@example.com/page"})
    result = await hook.before(call)
    assert result is call


@pytest.mark.asyncio
async def test_pipedream_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "https://eo1234.pipedream.net/collect"})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)


@pytest.mark.asyncio
async def test_burpcollaborator_blocked():
    hook = ToolArgumentPolicyHook(mode="block")
    call = _call(args={"url": "https://abc.burpcollaborator.net/ping"})
    with pytest.raises(ToolCallBlocked):
        await hook.before(call)
