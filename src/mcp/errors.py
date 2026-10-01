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


@dataclass
class HTTPErrorRecord:
    """The last HTTP error status our MCP client saw on a POST.

    mcp 2.x no longer raises ``HTTPStatusError`` for a non-2xx POST: it turns the
    response into a JSON-RPC error with a stand-in code (``INTERNAL_ERROR``,
    "Server returned an error response") and drops the status. Without this
    record a 401 or 429 would classify as a generic protocol error, silently
    disabling the AUTH cooldown and the rate-limit backoff. ``client.py``
    fills it from an httpx2 response hook on the client it hands the SDK.
    """

    status: int | None = None
    retry_after: str | None = None


# Statuses that carry operational meaning beyond the JSON-RPC code the SDK
# substitutes for them: AUTH cooldown, 429 backoff, 5xx retry.
_STATUS_OVERRIDES_JSON_RPC = frozenset({401, 403, 429})


def _status_overrides_json_rpc(status: int) -> bool:
    return status in _STATUS_OVERRIDES_JSON_RPC or 500 <= status < 600


def _classify_http_status(
    status: int,
    retry_after_header: str | None,
    *,
    server_id: str,
    tool_name: str | None,
) -> MCPCallError:
    """401/403→AUTH, 429/5xx→TRANSPORT (retryable), other 4xx→PROTOCOL."""
    retry_after = _parse_retry_after(retry_after_header)
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


def _unwrap_exception_group(
    exc: BaseException, preferred: type[BaseException] | Any
) -> BaseException:
    """Return the classifiable leaf of an anyio task-group ``ExceptionGroup``.

    The SDK's transport runs in nested task groups, so an ``MCPError`` from
    ``initialize`` / ``call_tool`` arrives two groups deep. Classifying the group
    itself lands in the generic non-retryable fallback, which hides AUTH and
    429. Prefer the first leaf of a known layer; otherwise a lone leaf.
    """
    # noqa F821 below: BaseExceptionGroup is a 3.11+ builtin and the project is
    # 3.12, but [tool.ruff] target-version still says py310.
    if not isinstance(exc, BaseExceptionGroup):  # noqa: F821
        return exc
    leaves: list[BaseException] = []
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop(0)
        if isinstance(current, BaseExceptionGroup):  # noqa: F821
            stack[:0] = list(current.exceptions)
        else:
            leaves.append(current)
    for leaf in leaves:
        if isinstance(leaf, preferred):
            return leaf
    return leaves[0] if len(leaves) == 1 else exc


def classify_mcp_error(
    exc: BaseException,
    *,
    server_id: str,
    tool_name: str | None = None,
    http_error: HTTPErrorRecord | None = None,
) -> MCPCallError:
    """Translate a transport/protocol exception into a structured MCPCallError.

    Recognized layers (in order):
      * ``mcp.shared.exceptions.MCPError`` — protocol-level JSON-RPC error. When
        ``http_error`` recorded a 401/403/429/5xx POST, the HTTP status wins:
        the SDK's JSON-RPC code for it is a stand-in (see ``HTTPErrorRecord``)
      * ``httpx2.HTTPStatusError`` — HTTP layer (401/403→AUTH, 429/5xx→TRANSPORT, other 4xx→PROTOCOL)
      * ``httpx2`` connection/timeout errors — TRANSPORT, retryable
      * everything else — TRANSPORT, non-retryable

    These are ``httpx2`` types, not ``httpx``: mcp 2.x is built on the fork, and
    this repo also depends on ``httpx`` directly, so an ``httpx`` isinstance check
    would import fine and silently never match.

    The original exception is preserved by the caller via ``raise X from exc``.
    """
    import httpx2
    from mcp.shared.exceptions import MCPError

    exc = _unwrap_exception_group(exc, MCPError | httpx2.HTTPError)

    if isinstance(exc, MCPError):
        if (
            http_error is not None
            and http_error.status is not None
            and _status_overrides_json_rpc(http_error.status)
        ):
            return _classify_http_status(
                http_error.status,
                http_error.retry_after,
                server_id=server_id,
                tool_name=tool_name,
            )
        json_rpc_code = exc.error.code
        message = exc.error.message or "MCP protocol error"
        retryable = json_rpc_code not in _NON_RETRYABLE_JSON_RPC_CODES
        return MCPCallError(
            message=_sanitize(str(message)),
            category=MCPErrorCategory.PROTOCOL,
            server_id=server_id,
            tool_name=tool_name,
            json_rpc_code=json_rpc_code,
            retryable=retryable,
        )

    if isinstance(exc, httpx2.HTTPStatusError):
        return _classify_http_status(
            exc.response.status_code,
            exc.response.headers.get("retry-after"),
            server_id=server_id,
            tool_name=tool_name,
        )

    if isinstance(
        exc,
        httpx2.ConnectError
        | httpx2.TimeoutException
        | httpx2.ReadError
        | httpx2.WriteError
        | httpx2.RemoteProtocolError,
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
