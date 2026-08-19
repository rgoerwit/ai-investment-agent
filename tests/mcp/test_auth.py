from __future__ import annotations

import pytest

from src.mcp.auth import MCPCredentialMissing, resolve_auth
from src.mcp.config import MCPAuthSpec, MCPServerSpec


@pytest.fixture(autouse=True)
def _clear_env_file_cache(monkeypatch: pytest.MonkeyPatch):
    """Each test starts with an empty .env-fallback so it can choose its own."""
    import src.config as config_module

    config_module._cached_env_file_values.cache_clear()
    monkeypatch.setattr(config_module, "_cached_env_file_values", lambda: {})
    yield
    config_module._cached_env_file_values.cache_clear() if hasattr(
        config_module._cached_env_file_values, "cache_clear"
    ) else None


def test_resolve_auth_uses_query_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FMP_API_KEY", "secret-token-123")
    spec = MCPServerSpec(
        id="fmp_remote",
        description="FMP",
        transport="streamable_http",
        base_url="https://example.test/mcp",
        auth=MCPAuthSpec(type="query_api_key", param="apikey", env_var="FMP_API_KEY"),
    )

    resolved = resolve_auth(spec)

    assert resolved is not None
    assert "apikey=secret-token-123" in (resolved.url or "")


def test_resolve_auth_reports_a_missing_credential_distinctly(
    monkeypatch: pytest.MonkeyPatch,
):
    """An absent key is "this server is unavailable", not "the registry is bad".

    The distinction is what lets MCPRuntime skip one vendor and keep the rest;
    before it, any missing key aborted construction of the whole runtime.
    """
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    spec = MCPServerSpec(
        id="fmp_remote",
        description="FMP",
        transport="streamable_http",
        base_url="https://example.test/mcp",
        auth=MCPAuthSpec(type="query_api_key", param="apikey", env_var="FMP_API_KEY"),
    )

    with pytest.raises(MCPCredentialMissing) as excinfo:
        resolve_auth(spec)
    assert excinfo.value.server_id == "fmp_remote"
    assert excinfo.value.env_var == "FMP_API_KEY"
    # Still a ValueError, so existing handlers keep working.
    assert isinstance(excinfo.value, ValueError)


def test_a_malformed_auth_block_is_rejected_and_is_not_a_missing_credential():
    """An auth block with no env_var is an authoring error, not a missing key.

    It is caught at spec construction rather than in resolve_auth, so it can
    never be mistaken for the "operator has no key" case and silently disable a
    server instead of surfacing a broken registry.
    """
    with pytest.raises(ValueError, match="must declare env_var") as excinfo:
        MCPAuthSpec(type="query_api_key", param="apikey", env_var=None)
    assert not isinstance(excinfo.value, MCPCredentialMissing)


def test_resolve_auth_injects_auth_env_into_stdio_process(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "secret-token-123")
    spec = MCPServerSpec(
        id="twelvedata_local",
        description="Twelve Data local MCP",
        transport="stdio",
        command="uvx",
        args=["mcp-server"],
        auth=MCPAuthSpec(type="header_static", env_var="TWELVE_DATA_API_KEY"),
    )

    resolved = resolve_auth(spec)

    assert resolved is not None
    assert resolved.stdio_env["TWELVE_DATA_API_KEY"] == "secret-token-123"


def test_resolve_auth_falls_back_to_env_file_when_shell_var_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    """`.env` keys must work even if the shell hasn't exported them.

    The repo loads `.env` via pydantic Settings; raw os.getenv does not see
    those values. resolve_auth uses get_env_value() which falls back to the
    parsed .env file. This is the regression for the "MCP fails despite
    FMP_API_KEY being in .env" bug.
    """
    import src.config as config_module

    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setattr(
        config_module,
        "_cached_env_file_values",
        lambda: {"FMP_API_KEY": "from-dotenv-789"},
    )

    spec = MCPServerSpec(
        id="fmp_remote",
        description="FMP",
        transport="streamable_http",
        base_url="https://example.test/mcp",
        auth=MCPAuthSpec(type="query_api_key", param="apikey", env_var="FMP_API_KEY"),
    )

    resolved = resolve_auth(spec)

    assert resolved is not None
    assert "apikey=from-dotenv-789" in (resolved.url or "")


def test_resolve_auth_shell_takes_precedence_over_env_file(
    monkeypatch: pytest.MonkeyPatch,
):
    """Shell-exported var wins over .env so ops/CI overrides remain effective."""
    import src.config as config_module

    monkeypatch.setenv("FMP_API_KEY", "shell-wins-456")
    monkeypatch.setattr(
        config_module,
        "_cached_env_file_values",
        lambda: {"FMP_API_KEY": "dotenv-loses-123"},
    )

    spec = MCPServerSpec(
        id="fmp_remote",
        description="FMP",
        transport="streamable_http",
        base_url="https://example.test/mcp",
        auth=MCPAuthSpec(type="query_api_key", param="apikey", env_var="FMP_API_KEY"),
    )

    resolved = resolve_auth(spec)

    assert resolved is not None
    assert "apikey=shell-wins-456" in (resolved.url or "")
