"""
Tests for prompts.py
Covers prompt loading, retrieval, and export.

prompts/*.json is the canonical (and only) prompt source; loading fails
loudly when the directory or a required agent prompt is missing.
"""

import json
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.prompts import (
    REQUIRED_AGENT_KEYS,
    AgentPrompt,
    PromptLoadError,
    PromptRegistry,
    get_all_prompts,
    get_prompt,
    get_registry,
)


@pytest.fixture
def prompts_copy(tmp_path):
    """Mutable copy of the canonical prompts/ directory."""
    dst = tmp_path / "prompts"
    shutil.copytree("prompts", dst)
    return dst


class TestAgentPrompt:
    """Test AgentPrompt dataclass."""

    def test_create_basic_prompt(self):
        """Test creating basic AgentPrompt."""
        prompt = AgentPrompt(
            agent_key="test_agent",
            agent_name="Test Agent",
            version="1.0",
            system_message="Test message",
        )

        assert prompt.agent_key == "test_agent"
        assert prompt.agent_name == "Test Agent"
        assert prompt.version == "1.0"
        assert prompt.system_message == "Test message"

    def test_default_category(self):
        """Test default category is 'general'."""
        prompt = AgentPrompt(
            agent_key="test", agent_name="Test", version="1.0", system_message="Test"
        )

        assert prompt.category == "general"

    def test_default_requires_tools(self):
        """Test default requires_tools is False."""
        prompt = AgentPrompt(
            agent_key="test", agent_name="Test", version="1.0", system_message="Test"
        )

        assert prompt.requires_tools is False

    def test_custom_metadata(self):
        """Test custom metadata."""
        metadata = {"last_updated": "2025-01-01", "changes": "Initial"}
        prompt = AgentPrompt(
            agent_key="test",
            agent_name="Test",
            version="1.0",
            system_message="Test",
            metadata=metadata,
        )

        assert prompt.metadata == metadata

    def test_none_metadata_initialization(self):
        """Test None metadata is converted to empty dict."""
        prompt = AgentPrompt(
            agent_key="test",
            agent_name="Test",
            version="1.0",
            system_message="Test",
            metadata=None,
        )

        assert prompt.metadata == {}


class TestPromptRegistryInit:
    """Test PromptRegistry initialization."""

    def test_init_creates_registry(self):
        """Test registry initialization from the canonical prompts dir."""
        registry = PromptRegistry()
        assert isinstance(registry.prompts, dict)

    def test_init_loads_canonical_prompts(self):
        """Test canonical JSON prompts are loaded."""
        registry = PromptRegistry()

        assert len(registry.prompts) > 0
        assert "market_analyst" in registry.prompts
        assert "fundamentals_analyst" in registry.prompts

    def test_init_with_nonexistent_dir_fails_loudly(self):
        """A missing prompts directory is a startup failure, not a fallback."""
        with pytest.raises(PromptLoadError, match="/nonexistent/path"):
            PromptRegistry(prompts_dir="/nonexistent/path")

    def test_init_with_empty_dir_names_missing_keys(self):
        """An empty prompts dir must fail and name the missing agent keys."""
        with TemporaryDirectory() as tmpdir:
            with pytest.raises(PromptLoadError, match="portfolio_manager"):
                PromptRegistry(prompts_dir=tmpdir)

    def test_init_uses_config_prompts_dir(self, prompts_copy):
        """Test prompts_dir from config Settings."""
        from unittest.mock import patch

        with patch("src.prompts.config") as mock_config:
            mock_config.prompts_dir = prompts_copy
            registry = PromptRegistry()

            assert registry.prompts_dir == prompts_copy


class TestLoadJsonPrompts:
    """Test JSON corpus loading."""

    def test_load_all_agent_types(self):
        """Test all expected agents are loaded."""
        registry = PromptRegistry()

        for agent in sorted(REQUIRED_AGENT_KEYS):
            assert agent in registry.prompts, f"{agent} not loaded"

    def test_prompts_have_required_fields(self):
        """Test all prompts have required fields."""
        registry = PromptRegistry()

        for agent_key, prompt in registry.prompts.items():
            assert prompt.agent_key == agent_key
            assert prompt.agent_name
            assert prompt.version
            assert prompt.system_message
            assert prompt.category

    def test_prompts_have_metadata(self):
        """Test prompts have metadata."""
        registry = PromptRegistry()

        for prompt in registry.prompts.values():
            assert isinstance(prompt.metadata, dict)

    def test_market_analyst_requires_tools(self):
        """Test market_analyst has requires_tools=True."""
        registry = PromptRegistry()

        prompt = registry.prompts["market_analyst"]
        assert prompt.requires_tools is True

    def test_fundamentals_analyst_system_message(self):
        """Test fundamentals_analyst has proper system message."""
        registry = PromptRegistry()

        prompt = registry.prompts["fundamentals_analyst"]
        assert "DATA_BLOCK" in prompt.system_message
        assert "ADAPTIVE SCORING" in prompt.system_message


class TestLoadFromCustomDir:
    """Test loading behavior against a mutable copy of the corpus."""

    def test_modified_prompt_file_is_authoritative(self, prompts_copy):
        """The JSON file content is the prompt — no hidden defaults."""
        modified = {
            "agent_key": "market_analyst",
            "agent_name": "Custom Market Analyst",
            "version": "99.0",
            "system_message": "Custom message",
            "category": "custom",
            "requires_tools": False,
            "metadata": {"custom": True},
        }
        (prompts_copy / "market_analyst.json").write_text(json.dumps(modified))

        registry = PromptRegistry(prompts_dir=str(prompts_copy))

        prompt = registry.prompts["market_analyst"]
        assert prompt.version == "99.0"
        assert prompt.system_message == "Custom message"
        assert prompt.metadata["custom"] is True

    def test_extra_prompt_files_load_alongside_required(self, prompts_copy):
        """Non-required experimental prompts load without ceremony."""
        for i in range(2):
            extra = {
                "agent_key": f"custom_agent_{i}",
                "agent_name": f"Custom {i}",
                "version": "1.0",
                "system_message": f"Message {i}",
            }
            (prompts_copy / f"custom_agent_{i}.json").write_text(json.dumps(extra))

        registry = PromptRegistry(prompts_dir=str(prompts_copy))

        assert "custom_agent_0" in registry.prompts
        assert "custom_agent_1" in registry.prompts

    def test_malformed_extra_json_skipped(self, prompts_copy):
        """A malformed non-required file is skipped, not fatal."""
        (prompts_copy / "bad.json").write_text("{invalid json")

        registry = PromptRegistry(prompts_dir=str(prompts_copy))

        assert "bad" not in registry.prompts
        assert "market_analyst" in registry.prompts

    def test_malformed_required_prompt_is_fatal(self, prompts_copy):
        """A required prompt file that fails to parse must fail startup."""
        (prompts_copy / "portfolio_manager.json").write_text("{invalid json")

        with pytest.raises(PromptLoadError, match="portfolio_manager"):
            PromptRegistry(prompts_dir=str(prompts_copy))

    def test_missing_agent_key_skipped(self, prompts_copy):
        """Test JSON without agent_key is skipped."""
        bad_prompt = {
            "agent_name": "Bad Agent",
            "version": "1.0",
            "system_message": "Message",
        }
        (prompts_copy / "bad.json").write_text(json.dumps(bad_prompt))

        registry = PromptRegistry(prompts_dir=str(prompts_copy))

        assert "bad_agent" not in registry.prompts


class TestGetMethod:
    """Test get() method."""

    def test_get_existing_prompt(self):
        """Test retrieving existing prompt."""
        registry = PromptRegistry()

        prompt = registry.get("market_analyst")

        assert prompt is not None
        assert prompt.agent_key == "market_analyst"

    def test_get_nonexistent_prompt(self):
        """Test retrieving non-existent prompt returns None."""
        registry = PromptRegistry()

        prompt = registry.get("nonexistent")

        assert prompt is None

    def test_get_with_env_override(self, monkeypatch):
        """Test environment variable override."""
        registry = PromptRegistry()

        # Set environment override
        monkeypatch.setenv("PROMPT_MARKET_ANALYST", "Override message")

        prompt = registry.get("market_analyst")

        assert prompt.system_message == "Override message"
        assert prompt.version.endswith("-env")
        assert prompt.metadata["source"] == "environment"

    def test_env_override_preserves_metadata(self, monkeypatch):
        """Test env override preserves other fields."""
        registry = PromptRegistry()

        monkeypatch.setenv("PROMPT_MARKET_ANALYST", "Override")

        prompt = registry.get("market_analyst")

        # Should preserve these from base prompt
        assert prompt.agent_name == "Market Analyst"
        assert prompt.category == "technical"
        assert prompt.requires_tools is True

    def test_env_override_honors_dotenv_file(self, monkeypatch):
        """Overrides defined only in .env (not shell) must be honored."""
        import src.config as config_module

        registry = PromptRegistry()

        monkeypatch.delenv("PROMPT_MARKET_ANALYST", raising=False)
        monkeypatch.setattr(
            config_module,
            "_cached_env_file_values",
            lambda: {"PROMPT_MARKET_ANALYST": "Dotenv override"},
        )

        prompt = registry.get("market_analyst")

        assert prompt.system_message == "Dotenv override"
        assert prompt.metadata["source"] == "environment"

    def test_shell_export_wins_over_dotenv(self, monkeypatch):
        """Shell-exported value takes precedence over the .env file."""
        import src.config as config_module

        registry = PromptRegistry()

        monkeypatch.setenv("PROMPT_MARKET_ANALYST", "Shell override")
        monkeypatch.setattr(
            config_module,
            "_cached_env_file_values",
            lambda: {"PROMPT_MARKET_ANALYST": "Dotenv override"},
        )

        assert registry.get("market_analyst").system_message == "Shell override"


class TestGetAllMethod:
    """Test get_all() method."""

    def test_get_all_returns_dict(self):
        """Test get_all returns dictionary."""
        registry = PromptRegistry()

        all_prompts = registry.get_all()

        assert isinstance(all_prompts, dict)

    def test_get_all_returns_copy(self):
        """Test get_all returns copy, not reference."""
        registry = PromptRegistry()

        all_prompts = registry.get_all()
        all_prompts.clear()

        # Original should still have prompts
        assert len(registry.prompts) > 0

    def test_get_all_contains_all_prompts(self):
        """Test get_all contains all loaded prompts."""
        registry = PromptRegistry()

        all_prompts = registry.get_all()

        assert len(all_prompts) == len(registry.prompts)


class TestListKeys:
    """Test list_keys() method."""

    def test_list_keys_returns_list(self):
        """Test list_keys returns list."""
        registry = PromptRegistry()

        keys = registry.list_keys()

        assert isinstance(keys, list)

    def test_list_keys_contains_expected(self):
        """Test list_keys contains expected keys."""
        registry = PromptRegistry()

        keys = registry.list_keys()

        assert "market_analyst" in keys
        assert "fundamentals_analyst" in keys


class TestExportToJson:
    """Test export_to_json() method."""

    def test_export_creates_files(self):
        """Test export creates JSON files."""
        registry = PromptRegistry()

        with TemporaryDirectory() as tmpdir:
            export_dir = f"{tmpdir}/export"
            registry.export_to_json(export_dir)

            # Check files were created
            export_path = Path(export_dir)
            assert export_path.exists()
            assert (export_path / "market_analyst.json").exists()

    def test_export_valid_json(self):
        """Test exported files are valid JSON."""
        registry = PromptRegistry()

        with TemporaryDirectory() as tmpdir:
            export_dir = f"{tmpdir}/export"
            registry.export_to_json(export_dir)

            # Load and validate JSON
            with open(f"{export_dir}/market_analyst.json") as f:
                data = json.load(f)

            assert data["agent_key"] == "market_analyst"
            assert "system_message" in data

    def test_export_preserves_all_fields(self):
        """Test export preserves all fields."""
        registry = PromptRegistry()

        with TemporaryDirectory() as tmpdir:
            export_dir = f"{tmpdir}/export"
            registry.export_to_json(export_dir)

            with open(f"{export_dir}/fundamentals_analyst.json") as f:
                data = json.load(f)

            required_fields = [
                "agent_key",
                "agent_name",
                "version",
                "system_message",
                "category",
                "requires_tools",
                "metadata",
            ]

            for field in required_fields:
                assert field in data

    def test_export_creates_directory(self):
        """Test export creates directory if not exists."""
        registry = PromptRegistry()

        with TemporaryDirectory() as tmpdir:
            export_dir = f"{tmpdir}/new/nested/dir"
            registry.export_to_json(export_dir)

            # Directory should be created
            assert Path(export_dir).exists()


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_registry_singleton(self):
        """Test get_registry returns singleton."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_get_prompt_function(self):
        """Test get_prompt convenience function."""
        prompt = get_prompt("market_analyst")

        assert prompt is not None
        assert prompt.agent_key == "market_analyst"

    def test_get_all_prompts_function(self):
        """Test get_all_prompts convenience function."""
        prompts = get_all_prompts()

        assert isinstance(prompts, dict)
        assert len(prompts) > 0


class TestEdgeCases:
    """Test edge cases."""

    def test_very_long_system_message(self):
        """Test handling of very long system message."""
        long_message = "X" * 1000000  # 1M characters

        prompt = AgentPrompt(
            agent_key="test",
            agent_name="Test",
            version="1.0",
            system_message=long_message,
        )

        assert len(prompt.system_message) == 1000000

    def test_unicode_in_system_message(self):
        """Test unicode characters in system message."""
        prompt = AgentPrompt(
            agent_key="test",
            agent_name="Test",
            version="1.0",
            system_message="测试 Test 🚀",
        )

        assert "测试" in prompt.system_message

    def test_special_characters_in_agent_key(self):
        """Test special characters in agent_key."""
        # Should handle underscores, numbers
        prompt = AgentPrompt(
            agent_key="test_agent_123",
            agent_name="Test",
            version="1.0",
            system_message="Test",
        )

        assert prompt.agent_key == "test_agent_123"

    def test_version_format_flexibility(self):
        """Test various version formats."""
        versions = ["1.0", "2.3.1", "v3.0", "latest", "2025-01-01"]

        for ver in versions:
            prompt = AgentPrompt(
                agent_key="test", agent_name="Test", version=ver, system_message="Test"
            )

            assert prompt.version == ver


class TestPromptConsistency:
    """Test prompt consistency across loads."""

    def test_reload_produces_same_prompts(self):
        """Test reloading produces identical prompts."""
        registry1 = PromptRegistry()
        registry2 = PromptRegistry()

        # Should have same keys
        assert set(registry1.list_keys()) == set(registry2.list_keys())

        # Should have same versions
        for key in registry1.list_keys():
            assert registry1.get(key).version == registry2.get(key).version

    def test_export_import_roundtrip(self):
        """Test export then import produces same data."""
        registry1 = PromptRegistry()

        with TemporaryDirectory() as tmpdir:
            # Export
            export_dir = f"{tmpdir}/export"
            registry1.export_to_json(export_dir)

            # Import
            registry2 = PromptRegistry(prompts_dir=export_dir)

            # Should have loaded from exported files
            for key in registry1.list_keys():
                original = registry1.get(key)
                reloaded = registry2.get(key)

                assert original.agent_key == reloaded.agent_key
                assert original.version == reloaded.version
                # System messages should match
                assert original.system_message == reloaded.system_message


@pytest.fixture
def temp_prompts_dir():
    """Fixture providing temporary prompts directory."""
    with TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_custom_prompt():
    """Fixture providing sample custom prompt data."""
    return {
        "agent_key": "test_agent",
        "agent_name": "Test Agent",
        "version": "1.0",
        "system_message": "Test system message",
        "category": "test",
        "requires_tools": False,
        "metadata": {"last_updated": "2025-01-01", "changes": "Initial version"},
    }


class TestNewsAnalystPromptV5:
    """news_analyst.json v5.0 structural checks."""

    @pytest.fixture
    def prompt(self):
        path = Path("prompts/news_analyst.json")
        return json.loads(path.read_text())

    def test_version_is_5_0(self, prompt):
        # Version bumped to 5.1 when fleet/capacity guardrail was added.
        assert re.match(
            r"^5\.\d+$", prompt["version"]
        ), f"Expected 5.x, got {prompt['version']}"

    def test_macro_detection_block_in_system_message(self, prompt):
        assert "MACRO_DETECTION" in prompt["system_message"]

    def test_innocent_bystander_framing_in_system_message(self, prompt):
        assert "Innocent Bystander" in prompt["system_message"]

    def test_structurally_impaired_framing_in_system_message(self, prompt):
        assert "Structurally Impaired" in prompt["system_message"]

    def test_macro_detection_in_critical_outputs(self, prompt):
        assert "macro_detection" in prompt["metadata"]["critical_outputs"]

    def test_macro_context_handling_section_present(self, prompt):
        msg = prompt["system_message"]
        assert "MACRO CONTEXT HANDLING" in msg
        assert "### PORTFOLIO MACRO EVENT" in msg
        assert "### REGIONAL MACRO CONTEXT" in msg

    def test_macro_tool_signature_is_current(self, prompt):
        assert "get_macroeconomic_news(trade_date, region)" in prompt["system_message"]

    def test_all_11_event_types_documented(self, prompt):
        """Ensure all event types from taxonomy appear in prompt."""
        msg = prompt["system_message"]
        for etype in [
            "TARIFF_TRADE",
            "LIQUIDITY_PANIC",
            "CONTAGION_SPREAD",
            "POLITICAL_EVENT",
            "MONETARY_PIVOT",
            "COMMODITY_SHOCK",
            "GEOPOLITICAL",
            "REGULATORY_SHIFT",
            "CREDIT_CONTAGION",
            "MACRO_RECESSION",
            "EXOGENOUS_SHOCK",
        ]:
            assert etype in msg, f"{etype} missing from news_analyst prompt"

    def test_word_limit_is_900(self, prompt):
        assert "900" in prompt["system_message"]

    def test_opportunity_field_in_output_block(self, prompt):
        assert "OPPORTUNITY:" in prompt["system_message"]

    def test_triggered_field_in_output_block(self, prompt):
        assert "TRIGGERED:" in prompt["system_message"]

    def test_news_prompt_mentions_ownership_change_searches(self, prompt):
        msg = prompt["system_message"]
        assert "director dealing" in msg
        assert "block trade" in msg
        assert "PDMR" in msg
        assert "股權披露" in msg


class TestMacroContextPrompt:
    def test_macro_context_prompt_exists(self):
        path = Path("prompts/macro_context_analyst.json")
        prompt = json.loads(path.read_text())
        assert prompt["agent_key"] == "macro_context_analyst"
        assert prompt["agent_name"] == "Macro Context Analyst"
        assert prompt["requires_tools"] is False

    def test_macro_context_prompt_has_required_sections(self):
        prompt = json.loads(Path("prompts/macro_context_analyst.json").read_text())
        msg = prompt["system_message"]
        assert "RATES & LIQUIDITY" in msg
        assert "FX & FLOWS" in msg
        assert "REGIME SUMMARY" in msg
        assert 'published="YYYY-MM-DD"' in msg
