import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import MagicMock, patch

from src.prompts import AgentPrompt, PromptRegistry, config, export_prompts


def test_agentprompt_defaults_and_metadata():
    p = AgentPrompt(
        agent_key="t1",
        agent_name="T1",
        version="1.0",
        system_message="hello",
        metadata=None,
    )
    assert p.agent_key == "t1"
    assert p.metadata == {}


def test_registry_loads_defaults_and_env_override(monkeypatch):
    with TemporaryDirectory() as d:
        # Ensure PROMPTS_DIR env var is respected
        monkeypatch.setenv("PROMPTS_DIR", d)
        reg = PromptRegistry()
        assert isinstance(reg.prompts, dict)
        # default prompts expected (archive indicates market_analyst exists)
        assert "market_analyst" in reg.prompts


def test_export_and_roundtrip(tmp_path):
    reg = PromptRegistry(prompts_dir=str(tmp_path))
    # Create a sample custom prompt and export
    reg.prompts["custom_agent"] = AgentPrompt(
        agent_key="custom_agent",
        agent_name="Custom",
        version="0.1",
        system_message="hi",
    )
    export_dir = tmp_path / "exported"
    export_prompts(str(export_dir))
    # Expect files in export_dir
    assert (export_dir / "market_analyst.json").exists() or (
        export_dir / "fundamentals_analyst.json"
    ).exists()


def test_load_malformed_json_skipped(tmp_path, monkeypatch):
    # Create a malformed JSON file in prompts dir
    pdir = tmp_path / "prompts"
    pdir.mkdir()
    bad = pdir / "bad_prompt.json"
    bad.write_text("{ not: valid json }")
    # Should not crash when creating registry
    reg = PromptRegistry(prompts_dir=str(pdir))
    # bad_prompt should not be loaded
    assert "bad_prompt" not in reg.prompts


def test_registry_keeps_local_prompt_when_langfuse_fetch_disabled(monkeypatch):
    client = MagicMock()
    langfuse_module = ModuleType("langfuse")
    langfuse_module.get_client = MagicMock(return_value=client)

    monkeypatch.setitem(sys.modules, "langfuse", langfuse_module)
    with (
        patch.object(config, "langfuse_enabled", True),
        patch.object(config, "langfuse_prompt_fetch_enabled", False),
        patch.object(type(config), "get_langfuse_public_key", return_value="pk"),
        patch.object(type(config), "get_langfuse_secret_key", return_value="sk"),
    ):
        reg = PromptRegistry()
        prompt = reg.get("market_analyst")

    assert prompt is not None
    assert prompt.source == "local"
    client.get_prompt.assert_not_called()


def test_registry_fetches_prompt_from_langfuse_when_enabled(monkeypatch):
    prompt_client = MagicMock(prompt="Remote system prompt", version=17)
    client = MagicMock()
    client.get_prompt.return_value = prompt_client
    langfuse_module = ModuleType("langfuse")
    langfuse_module.get_client = MagicMock(return_value=client)

    monkeypatch.setitem(sys.modules, "langfuse", langfuse_module)
    with (
        patch.object(config, "langfuse_enabled", True),
        patch.object(config, "langfuse_prompt_fetch_enabled", True),
        patch.object(config, "langfuse_prompt_label", "production"),
        patch.object(config, "langfuse_prompt_cache_ttl_seconds", 60),
        patch.object(type(config), "get_langfuse_public_key", return_value="pk"),
        patch.object(type(config), "get_langfuse_secret_key", return_value="sk"),
    ):
        reg = PromptRegistry()
        prompt = reg.get("market_analyst")

    assert prompt is not None
    assert prompt.source == "langfuse"
    assert prompt.system_message == "Remote system prompt"
    assert prompt.langfuse_name == "market_analyst"
    assert prompt.langfuse_label == "production"
    assert prompt.langfuse_version == "17"
    assert prompt.metadata["prompt_source"] == "langfuse"
    assert prompt.metadata["prompt_name"] == "market_analyst"
    assert prompt.metadata["prompt_label"] == "production"
    client.get_prompt.assert_called_once()


def test_registry_falls_back_to_local_prompt_when_langfuse_fetch_fails(monkeypatch):
    client = MagicMock()
    client.get_prompt.side_effect = RuntimeError("boom")
    langfuse_module = ModuleType("langfuse")
    langfuse_module.get_client = MagicMock(return_value=client)

    monkeypatch.setitem(sys.modules, "langfuse", langfuse_module)
    with (
        patch.object(config, "langfuse_enabled", True),
        patch.object(config, "langfuse_prompt_fetch_enabled", True),
        patch.object(config, "langfuse_prompt_label", "production"),
        patch.object(config, "langfuse_prompt_cache_ttl_seconds", 60),
        patch.object(type(config), "get_langfuse_public_key", return_value="pk"),
        patch.object(type(config), "get_langfuse_secret_key", return_value="sk"),
    ):
        reg = PromptRegistry()
        local_prompt = reg.prompts["market_analyst"]
        prompt = reg.get("market_analyst")

    assert prompt is not None
    assert prompt.source == local_prompt.source
    assert prompt.system_message == local_prompt.system_message
    client.get_prompt.assert_called_once()
