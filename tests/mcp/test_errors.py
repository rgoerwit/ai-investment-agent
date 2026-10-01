"""Error classification under mcp 2.x (httpx2 transport, MCPError).

The v1→v2 migration had silent failure modes: ``httpx`` isinstance checks that
import fine and never match, and HTTP statuses that the SDK folds into a
stand-in JSON-RPC error. Each would collapse AUTH and 429 into generic errors,
disabling the AUTH cooldown and the rate-limit backoff without failing anything
loudly. The end-to-end tests drive the *real* SDK over a mock transport, so
they also catch the SDK changing how it surfaces HTTP failures.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import httpx2
import pytest
from mcp.shared.exceptions import MCPError

from src.mcp.client import MCPRuntime
from src.mcp.config import MCPServerSpec
from src.mcp.errors import (
    HTTPErrorRecord,
    MCPCallError,
    MCPErrorCategory,
    classify_mcp_error,
)

_URL = "https://example.test/mcp"


def _status_error(status: int, headers: dict[str, str] | None = None):
    request = httpx2.Request("POST", _URL)
    response = httpx2.Response(status, headers=headers, request=request)
    return httpx2.HTTPStatusError("boom", request=request, response=response)


class TestClassifyHttpx2:
    def test_401_is_auth_and_not_retryable(self):
        err = classify_mcp_error(_status_error(401), server_id="s")
        assert err.category is MCPErrorCategory.AUTH
        assert err.http_status == 401
        assert err.retryable is False

    def test_429_is_retryable_transport_with_retry_after(self):
        err = classify_mcp_error(
            _status_error(429, {"retry-after": "7"}), server_id="s"
        )
        assert err.category is MCPErrorCategory.TRANSPORT
        assert err.retryable is True
        assert err.retry_after_seconds == 7

    def test_503_is_retryable_transport(self):
        err = classify_mcp_error(_status_error(503), server_id="s")
        assert err.category is MCPErrorCategory.TRANSPORT
        assert err.retryable is True

    def test_other_4xx_is_protocol(self):
        err = classify_mcp_error(_status_error(422), server_id="s")
        assert err.category is MCPErrorCategory.PROTOCOL
        assert err.retryable is False

    def test_connect_error_is_retryable(self):
        exc = httpx2.ConnectError("down", request=httpx2.Request("POST", _URL))
        err = classify_mcp_error(exc, server_id="s")
        assert err.category is MCPErrorCategory.TRANSPORT
        assert err.retryable is True

    def test_plain_httpx_error_is_not_the_sdk_layer(self):
        """The SDK never raises ``httpx`` types now; one reaching here is foreign.

        Pins that classification is keyed to httpx2: a regression back to
        ``import httpx`` would make this retryable and the httpx2 test above not.
        """
        exc = httpx.ConnectError("down", request=httpx.Request("POST", _URL))
        err = classify_mcp_error(exc, server_id="s")
        assert err.retryable is False


class TestClassifyMcpError:
    def test_json_rpc_code_is_protocol(self):
        err = classify_mcp_error(MCPError(-32602, "bad params"), server_id="s")
        assert err.category is MCPErrorCategory.PROTOCOL
        assert err.json_rpc_code == -32602
        assert err.retryable is False

    def test_retryable_json_rpc_code(self):
        err = classify_mcp_error(MCPError(-32603, "internal"), server_id="s")
        assert err.retryable is True

    def test_recorded_401_overrides_stand_in_code(self):
        record = HTTPErrorRecord(status=401)
        err = classify_mcp_error(
            MCPError(-32603, "Server returned an error response"),
            server_id="s",
            http_error=record,
        )
        assert err.category is MCPErrorCategory.AUTH
        assert err.http_status == 401

    def test_recorded_429_keeps_retry_after(self):
        record = HTTPErrorRecord(status=429, retry_after="12")
        err = classify_mcp_error(
            MCPError(-32603, "Server returned an error response"),
            server_id="s",
            http_error=record,
        )
        assert err.category is MCPErrorCategory.TRANSPORT
        assert err.retry_after_seconds == 12

    def test_recorded_plain_4xx_defers_to_json_rpc_code(self):
        """A 400 carrying a real JSON-RPC error body keeps the server's code."""
        record = HTTPErrorRecord(status=400)
        err = classify_mcp_error(
            MCPError(-32602, "bad params"), server_id="s", http_error=record
        )
        assert err.category is MCPErrorCategory.PROTOCOL
        assert err.json_rpc_code == -32602

    def test_nested_exception_group_is_unwrapped(self):
        """The SDK's task groups wrap the MCPError two levels deep."""
        inner = MCPError(-32602, "bad params")
        group = ExceptionGroup("outer", [ExceptionGroup("inner", [inner])])  # noqa: F821
        err = classify_mcp_error(group, server_id="s")
        assert err.category is MCPErrorCategory.PROTOCOL
        assert err.json_rpc_code == -32602

    def test_group_prefers_known_layer_over_noise(self):
        group = ExceptionGroup(  # noqa: F821
            "g", [RuntimeError("cleanup noise"), _status_error(401)]
        )
        err = classify_mcp_error(group, server_id="s")
        assert err.category is MCPErrorCategory.AUTH

    def test_group_of_unknown_errors_falls_back(self):
        group = ExceptionGroup("g", [RuntimeError("a"), ValueError("b")])  # noqa: F821
        err = classify_mcp_error(group, server_id="s")
        assert err.category is MCPErrorCategory.TRANSPORT
        assert err.retryable is False

    def test_empty_record_is_ignored(self):
        err = classify_mcp_error(
            MCPError(-32601, "nope"), server_id="s", http_error=HTTPErrorRecord()
        )
        assert err.category is MCPErrorCategory.PROTOCOL
        assert err.json_rpc_code == -32601


# ---------------------------------------------------------------------------
# End to end through the real mcp 2.x client over a mock transport
# ---------------------------------------------------------------------------


def _runtime(tmp_path: Path) -> MCPRuntime:
    spec = MCPServerSpec(
        id="fmp_remote",
        description="FMP",
        transport="streamable_http",
        base_url=_URL,
        scopes=["consultant"],
        tool_allowlist=["quote"],
    )
    return MCPRuntime([spec], budget_db_path=str(tmp_path / "mcp_usage.db"))


def _patch_transport(monkeypatch, handler) -> None:
    """Make the runtime's own httpx2 client talk to ``handler``."""
    real_client = httpx2.AsyncClient

    def _client(**kwargs):
        return real_client(transport=httpx2.MockTransport(handler), **kwargs)

    monkeypatch.setattr("src.mcp.client.httpx2.AsyncClient", _client)


async def _open_and_fail(runtime: MCPRuntime) -> MCPCallError:
    spec = runtime.specs["fmp_remote"]
    with pytest.raises(MCPCallError) as exc_info:
        async with runtime._open_session(spec):
            pass  # pragma: no cover - initialize must fail first
    return exc_info.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "headers", "category", "retryable", "retry_after"),
    [
        (401, {}, MCPErrorCategory.AUTH, False, None),
        (403, {}, MCPErrorCategory.AUTH, False, None),
        (429, {"retry-after": "30"}, MCPErrorCategory.TRANSPORT, True, 30),
        (502, {}, MCPErrorCategory.TRANSPORT, True, None),
    ],
)
async def test_http_status_survives_the_real_sdk(
    tmp_path, monkeypatch, status, headers, category, retryable, retry_after
):
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            return httpx2.Response(status, headers=headers, text="denied")
        return httpx2.Response(405)

    _patch_transport(monkeypatch, handler)
    err = await asyncio.wait_for(_open_and_fail(_runtime(tmp_path)), timeout=10)

    assert err.category is category
    assert err.http_status == status
    assert err.retryable is retryable
    assert err.retry_after_seconds == retry_after


@pytest.mark.asyncio
async def test_json_rpc_error_body_keeps_server_code(tmp_path, monkeypatch):
    """A spec-correct 400 with a JSON-RPC error body stays a protocol error."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method != "POST":
            return httpx2.Response(405)
        msg = json.loads(request.content)
        body = {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "error": {"code": -32602, "message": "Invalid params"},
        }
        return httpx2.Response(400, json=body)

    _patch_transport(monkeypatch, handler)
    err = await asyncio.wait_for(_open_and_fail(_runtime(tmp_path)), timeout=10)

    assert err.category is MCPErrorCategory.PROTOCOL
    assert err.json_rpc_code == -32602
    assert err.retryable is False
