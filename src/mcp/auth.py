from __future__ import annotations

from dataclasses import dataclass, field

from src.config import get_env_value
from src.mcp.config import MCPServerSpec


@dataclass(frozen=True)
class MCPResolvedServer:
    server_id: str
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    stdio_env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


def resolve_auth(spec: MCPServerSpec) -> MCPResolvedServer | None:
    if not spec.enabled:
        return None

    url = spec.base_url
    headers: dict[str, str] = {}
    stdio_env: dict[str, str] = {}

    # Use get_env_value() so MCP keys defined only in .env are visible — the
    # repo's pydantic Settings does not reflect arbitrary keys to os.environ.
    for env_name in spec.env_vars:
        value = get_env_value(env_name)
        if value:
            stdio_env[env_name] = value

    if spec.auth is not None and spec.auth.type != "none":
        auth_env_name = spec.auth.env_var
        if not auth_env_name:
            raise ValueError(
                f"Server {spec.id} auth requires an 'env_var' in auth config."
            )
        key = get_env_value(auth_env_name)
        if not key:
            raise ValueError(
                f"Required env var {auth_env_name!r} not set for MCP server {spec.id}"
            )
        if spec.transport == "stdio":
            stdio_env[auth_env_name] = key

        if spec.auth.type == "query_api_key":
            param = spec.auth.param or "apikey"
            separator = "&" if "?" in (url or "") else "?"
            url = f"{url}{separator}{param}={key}"
        elif spec.auth.type == "header_bearer":
            headers["Authorization"] = f"Bearer {key}"
        elif spec.auth.type == "header_static":
            header_name = spec.auth.param or "X-API-Key"
            headers[header_name] = key

    return MCPResolvedServer(
        server_id=spec.id,
        url=url,
        headers=headers,
        stdio_env=stdio_env,
        cwd=spec.cwd,
    )
