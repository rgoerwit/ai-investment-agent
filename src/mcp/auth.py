from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from src.mcp.config import MCPServerSpec


@dataclass(frozen=True)
class MCPResolvedServer:
    server_id: str
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    stdio_env: dict[str, str] | None = None


def resolve_auth(spec: MCPServerSpec) -> MCPResolvedServer | None:
    if not spec.enabled:
        return None

    url = spec.base_url
    headers: dict[str, str] = {}

    if spec.auth and spec.auth.type != "none":
        env_name = spec.auth.env_var
        if not env_name:
            raise ValueError(
                f"Server {spec.id} auth requires an 'env_var' in auth config."
            )
        key = os.getenv(env_name)
        if not key:
            raise ValueError(
                f"Required env var {env_name!r} not set for MCP server {spec.id}"
            )

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
    )
