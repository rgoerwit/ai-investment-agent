from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SUPPORTED_TRANSPORTS = frozenset({"streamable_http", "stdio"})
SUPPORTED_AUTH_TYPES = frozenset(
    {"none", "query_api_key", "header_bearer", "header_static"}
)
SUPPORTED_TRUST_TIERS = frozenset({"official_vendor", "community", "unknown"})


@dataclass(frozen=True)
class MCPAuthSpec:
    type: Literal["none", "query_api_key", "header_bearer", "header_static"] = "none"
    param: str | None = None
    env_var: str | None = None

    def __post_init__(self) -> None:
        if self.type not in SUPPORTED_AUTH_TYPES:
            raise ValueError(f"Unsupported MCP auth type: {self.type!r}")
        if self.type != "none" and not self.env_var:
            raise ValueError("MCP auth config must declare env_var for non-none auth")


@dataclass(frozen=True)
class MCPServerSpec:
    id: str
    description: str
    transport: Literal["streamable_http", "stdio"]
    base_url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    env_vars: list[str] = field(default_factory=list)
    auth: MCPAuthSpec | None = None
    enabled: bool = True
    scopes: list[str] = field(default_factory=list)
    tool_allowlist: list[str] = field(default_factory=list)
    daily_call_limit: int = 0
    per_run_limit: int = 0
    trust_tier: Literal["official_vendor", "community", "unknown"] = "unknown"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("MCP server id must not be empty")
        if self.transport not in SUPPORTED_TRANSPORTS:
            raise ValueError(f"Unsupported MCP transport: {self.transport!r}")
        if self.transport == "streamable_http" and not self.base_url:
            raise ValueError(
                f"MCP server {self.id!r} uses streamable_http but has no base_url"
            )
        if self.transport == "stdio" and not self.command:
            raise ValueError(
                f"MCP server {self.id!r} uses stdio but has no command configured"
            )
        if self.trust_tier not in SUPPORTED_TRUST_TIERS:
            raise ValueError(
                f"MCP server {self.id!r} has unsupported trust_tier {self.trust_tier!r}"
            )
        if self.daily_call_limit < 0 or self.per_run_limit < 0:
            raise ValueError("MCP call limits must be non-negative")

    def supports_scope(self, scope: str | None) -> bool:
        if scope is None:
            return True
        return scope in self.scopes


def _load_auth(raw: dict[str, Any] | None) -> MCPAuthSpec | None:
    if not raw:
        return None
    return MCPAuthSpec(
        type=raw.get("type", "none"),
        param=raw.get("param"),
        env_var=raw.get("env_var"),
    )


def load_registry(path: Path, *, required: bool = False) -> list[MCPServerSpec]:
    if not path.exists():
        if required:
            raise ValueError(f"MCP registry file not found: {path}")
        return []

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    raw_servers = data.get("servers", [])
    if not isinstance(raw_servers, list):
        raise ValueError("MCP registry must contain a top-level 'servers' list")

    result: list[MCPServerSpec] = []
    for raw in raw_servers:
        if not isinstance(raw, dict):
            raise ValueError("Each MCP server entry must be a JSON object")
        result.append(
            MCPServerSpec(
                id=raw["id"],
                description=raw.get("description", ""),
                transport=raw.get("transport", "streamable_http"),
                base_url=raw.get("base_url"),
                command=raw.get("command"),
                args=list(raw.get("args", [])),
                cwd=raw.get("cwd"),
                env_vars=list(raw.get("env_vars", [])),
                auth=_load_auth(raw.get("auth")),
                enabled=raw.get("enabled", True),
                scopes=list(raw.get("scopes", [])),
                tool_allowlist=list(raw.get("tool_allowlist", [])),
                daily_call_limit=raw.get("daily_call_limit", 0),
                per_run_limit=raw.get("per_run_limit", 0),
                trust_tier=raw.get("trust_tier", "unknown"),
            )
        )

    if required and not result:
        raise ValueError(f"MCP registry is empty: {path}")

    if required and not any(server.enabled for server in result):
        raise ValueError(f"MCP registry contains no enabled servers: {path}")

    return result
