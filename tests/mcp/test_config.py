from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.mcp.config import MCPServerSpec, load_registry


def test_load_registry_reads_valid_server(tmp_path: Path):
    path = tmp_path / "mcp_servers.json"
    path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "id": "fmp_remote",
                        "description": "FMP",
                        "transport": "streamable_http",
                        "base_url": "https://example.test/mcp",
                        "auth": {
                            "type": "query_api_key",
                            "param": "apikey",
                            "env_var": "FMP_API_KEY",
                        },
                        "enabled": True,
                        "scopes": ["consultant"],
                        "tool_allowlist": ["quote"],
                        "trust_tier": "official_vendor",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    servers = load_registry(path)

    assert len(servers) == 1
    assert servers[0].id == "fmp_remote"
    assert servers[0].supports_scope("consultant") is True


def test_load_registry_rejects_unsupported_transport(tmp_path: Path):
    path = tmp_path / "mcp_servers.json"
    path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "id": "bad",
                        "description": "bad",
                        "transport": "sse",
                        "base_url": "https://example.test/mcp",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported MCP transport"):
        load_registry(path)


def test_load_registry_required_missing_file_fails_fast(tmp_path: Path):
    path = tmp_path / "missing_mcp_servers.json"

    with pytest.raises(ValueError, match="MCP registry file not found"):
        load_registry(path, required=True)


def test_load_registry_accepts_an_all_disabled_registry(tmp_path: Path):
    path = tmp_path / "mcp_servers.json"
    path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "id": "fmp_remote",
                        "description": "FMP",
                        "transport": "streamable_http",
                        "base_url": "https://example.test/mcp",
                        "enabled": False,
                        "scopes": ["consultant"],
                        "tool_allowlist": ["quote"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # A registry whose servers are all switched off is a legitimate state:
    # MCP_ENABLED=true with nothing wired up simply exposes no MCP tools.
    # Rejecting it meant retiring the last server also required remembering to
    # flip MCP_ENABLED, and forgetting produced a warning on every run.
    servers = load_registry(path, required=True)

    assert [server.id for server in servers] == ["fmp_remote"]
    assert not any(server.enabled for server in servers)


def test_server_spec_rejects_unknown_trust_tier():
    with pytest.raises(ValueError, match="unsupported trust_tier"):
        MCPServerSpec(
            id="bad",
            description="bad",
            transport="streamable_http",
            base_url="https://example.test/mcp",
            trust_tier="vendorish",  # type: ignore[arg-type]
        )
