from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.error_safety import redact_sensitive_text


class MCPErrorCategory(str, Enum):
    CONFIG = "config"
    AUTH = "auth"
    TRANSPORT = "transport"
    PROTOCOL = "protocol"
    TOOL_ERROR = "tool_error"
    INSPECTION = "inspection"
    BUDGET = "budget"


@dataclass(eq=False)
class MCPCallError(RuntimeError):
    """Structured MCP runtime error with redacted operator-facing details."""

    message: str
    category: MCPErrorCategory
    server_id: str
    tool_name: str | None = None
    retryable: bool = False
    http_status: int | None = None
    json_rpc_code: int | None = None
    retry_after_seconds: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)


_NON_RETRYABLE_JSON_RPC_CODES = frozenset({-32700, -32600, -32601, -32602})


def _sanitize(text: str, *, max_chars: int = 512) -> str:
    return redact_sensitive_text(text, max_chars=max_chars)


def _parse_retry_after(value: str | None) -> int | None:
    if not value:
        return None
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return seconds


def classify_mcp_error(
    exc: BaseException,
    *,
    server_id: str,
    tool_name: str | None = None,
) -> MCPCallError:
    """Translate a transport/protocol exception into a structured MCPCallError.

    Recognized layers (in order):
      * ``mcp.shared.exceptions.McpError`` — protocol-level JSON-RPC error
      * ``httpx.HTTPStatusError`` — HTTP layer (401/403→AUTH, 429/5xx→TRANSPORT, other 4xx→PROTOCOL)
      * ``httpx`` connection/timeout errors — TRANSPORT, retryable
      * everything else — TRANSPORT, non-retryable

    The original exception is preserved by the caller via ``raise X from exc``.
    """

    # Protocol-level: MCP McpError carries a JSON-RPC code.
    mcp_error_type: type[Any] | None
    try:
        from mcp.shared.exceptions import McpError as _McpError

        mcp_error_type = _McpError
    except ImportError:  # pragma: no cover - mcp dep guaranteed at runtime
        mcp_error_type = None

    if mcp_error_type is not None and isinstance(exc, mcp_error_type):
        err_obj = getattr(exc, "error", None)
        json_rpc_code = getattr(err_obj, "code", None) if err_obj is not None else None
        message = (
            getattr(err_obj, "message", None) if err_obj is not None else None
        ) or "MCP protocol error"
        retryable = json_rpc_code not in _NON_RETRYABLE_JSON_RPC_CODES
        return MCPCallError(
            message=_sanitize(str(message)),
            category=MCPErrorCategory.PROTOCOL,
            server_id=server_id,
            tool_name=tool_name,
            json_rpc_code=json_rpc_code,
            retryable=retryable,
        )

    # HTTP layer: differentiate auth/rate-limit/server-error/other.
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a transitive dep of mcp
        httpx = None  # type: ignore[assignment]

    if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        retry_after = _parse_retry_after(exc.response.headers.get("retry-after"))
        if status in (401, 403):
            return MCPCallError(
                message=_sanitize(f"Upstream HTTP {status}"),
                category=MCPErrorCategory.AUTH,
                server_id=server_id,
                tool_name=tool_name,
                http_status=status,
                retryable=False,
            )
        if status == 429:
            return MCPCallError(
                message=_sanitize("Upstream rate-limited"),
                category=MCPErrorCategory.TRANSPORT,
                server_id=server_id,
                tool_name=tool_name,
                http_status=status,
                retry_after_seconds=retry_after,
                retryable=True,
            )
        if 500 <= status < 600:
            return MCPCallError(
                message=_sanitize(f"Upstream HTTP {status}"),
                category=MCPErrorCategory.TRANSPORT,
                server_id=server_id,
                tool_name=tool_name,
                http_status=status,
                retry_after_seconds=retry_after,
                retryable=True,
            )
        return MCPCallError(
            message=_sanitize(f"Upstream HTTP {status}"),
            category=MCPErrorCategory.PROTOCOL,
            server_id=server_id,
            tool_name=tool_name,
            http_status=status,
            retryable=False,
        )

    if httpx is not None and isinstance(
        exc,
        httpx.ConnectError
        | httpx.TimeoutException
        | httpx.ReadError
        | httpx.WriteError
        | httpx.RemoteProtocolError,
    ):
        return MCPCallError(
            message=_sanitize(str(exc)),
            category=MCPErrorCategory.TRANSPORT,
            server_id=server_id,
            tool_name=tool_name,
            retryable=True,
        )

    # Generic fallback: unknown error layer.
    return MCPCallError(
        message=_sanitize(str(exc)),
        category=MCPErrorCategory.TRANSPORT,
        server_id=server_id,
        tool_name=tool_name,
        retryable=False,
    )


_MCP_TOOL_PREFIX = "mcp__"


def parse_mcp_tool_name(name: str) -> tuple[str, str] | None:
    """Parse a hook-friendly tool name like ``mcp__<server>__<tool>``.

    Returns ``(server_id, tool_name)`` or ``None`` if the name is not an MCP-prefixed
    tool. ``tool`` may itself contain ``__`` (we only split on the first separator
    after the prefix).
    """
    if not name.startswith(_MCP_TOOL_PREFIX):
        return None
    rest = name[len(_MCP_TOOL_PREFIX) :]
    server_id, sep, tool_name = rest.partition("__")
    if not sep or not server_id or not tool_name:
        return None
    return server_id, tool_name


def make_mcp_tool_name(server_id: str, tool_name: str) -> str:
    """Build the canonical hook-friendly MCP tool name."""
    return f"{_MCP_TOOL_PREFIX}{server_id}__{tool_name}"
