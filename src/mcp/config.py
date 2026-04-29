from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class MCPAuthSpec:
    type: Literal["none", "query_api_key", "header_bearer", "header_static", "oauth_client"]
    param: str | None = None
    env_var: str | None = None


@dataclass
class MCPServerSpec:
    id: str
    description: str
    transport: Literal["streamable_http", "stdio"]
    base_url: str | None = None
    auth: MCPAuthSpec | None = None
    enabled: bool = True
    scopes: list[str] = field(default_factory=list)
    tool_allowlist: list[str] = field(default_factory=list)
    daily_call_limit: int = 0
    per_run_limit: int = 0
    trust_tier: str = "unknown"

    @property
    def is_enabled(self) -> bool:
        return self.enabled


def load_registry(path: Path) -> list[MCPServerSpec]:
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    result: list[MCPServerSpec] = []
    for raw in data.get("servers", []):
        auth_raw = raw.get("auth")
        auth = (
            MCPAuthSpec(
                type=auth_raw.get("type", "none"),
                param=auth_raw.get("param"),
                env_var=auth_raw.get("env_var"),
            )
            if auth_raw
            else None
        )
        spec = MCPServerSpec(
            id=raw["id"],
            description=raw.get("description", ""),
            transport=raw.get("transport", "streamable_http"),
            base_url=raw.get("base_url"),
            auth=auth,
            enabled=raw.get("enabled", True),
            scopes=raw.get("scopes", []),
            tool_allowlist=raw.get("tool_allowlist", []),
            daily_call_limit=raw.get("daily_call_limit", 0),
            per_run_limit=raw.get("per_run_limit", 0),
            trust_tier=raw.get("trust_tier", "unknown"),
        )
        result.append(spec)

    return result
