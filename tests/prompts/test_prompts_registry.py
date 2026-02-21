# tests/test_prompts_registry.py
from tempfile import TemporaryDirectory

from src.prompts import AgentPrompt, PromptRegistry, export_prompts


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
