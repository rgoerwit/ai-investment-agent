"""
Tests for prompts.py
Covers prompt loading, retrieval, and export.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.prompts import (
    AgentPrompt,
    PromptRegistry,
    get_all_prompts,
    get_prompt,
    get_registry,
)


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
        """Test registry initialization."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)
            assert isinstance(registry.prompts, dict)

    def test_init_loads_default_prompts(self):
        """Test default prompts are loaded."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            # Should have loaded default prompts
            assert len(registry.prompts) > 0
            assert "market_analyst" in registry.prompts
            assert "fundamentals_analyst" in registry.prompts

    def test_init_with_nonexistent_dir(self):
        """Test initialization with non-existent directory."""
        # Should not raise error
        registry = PromptRegistry(prompts_dir="/nonexistent/path")
        assert len(registry.prompts) > 0  # Still has defaults

    def test_init_uses_config_prompts_dir(self):
        """Test prompts_dir from config Settings."""
        from unittest.mock import patch

        with TemporaryDirectory() as tmpdir:
            with patch("src.prompts.config") as mock_config:
                mock_config.prompts_dir = Path(tmpdir)
                registry = PromptRegistry()

                assert str(registry.prompts_dir) == tmpdir


class TestLoadDefaultPrompts:
    """Test _load_default_prompts() method."""

    def test_load_all_agent_types(self):
        """Test all expected agents are loaded."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            expected_agents = [
                "market_analyst",
                "sentiment_analyst",
                "news_analyst",
                "fundamentals_analyst",
                "bull_researcher",
                "bear_researcher",
                "research_manager",
                "trader",
                "risky_analyst",
                "safe_analyst",
                "neutral_analyst",
                "portfolio_manager",
            ]

            for agent in expected_agents:
                assert agent in registry.prompts, f"{agent} not loaded"

    def test_prompts_have_required_fields(self):
        """Test all prompts have required fields."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            for agent_key, prompt in registry.prompts.items():
                assert prompt.agent_key == agent_key
                assert prompt.agent_name
                assert prompt.version
                assert prompt.system_message
                assert prompt.category

    def test_prompts_have_metadata(self):
        """Test prompts have metadata."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            for prompt in registry.prompts.values():
                assert isinstance(prompt.metadata, dict)

    def test_market_analyst_requires_tools(self):
        """Test market_analyst has requires_tools=True."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            prompt = registry.prompts["market_analyst"]
            assert prompt.requires_tools is True

    def test_fundamentals_analyst_system_message(self):
        """Test fundamentals_analyst has proper system message."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            prompt = registry.prompts["fundamentals_analyst"]
            assert "DATA_BLOCK" in prompt.system_message
            assert "ADAPTIVE SCORING" in prompt.system_message


class TestLoadCustomPrompts:
    """Test _load_custom_prompts() method."""

    def test_load_custom_prompt_overrides_default(self):
        """Test custom prompt overrides default."""
        with TemporaryDirectory() as tmpdir:
            # Create custom prompt file
            custom_prompt = {
                "agent_key": "market_analyst",
                "agent_name": "Custom Market Analyst",
                "version": "99.0",
                "system_message": "Custom message",
                "category": "custom",
                "requires_tools": False,
                "metadata": {"custom": True},
            }

            with open(f"{tmpdir}/market_analyst.json", "w") as f:
                json.dump(custom_prompt, f)

            registry = PromptRegistry(prompts_dir=tmpdir)

            # Should have loaded custom version
            prompt = registry.prompts["market_analyst"]
            assert prompt.version == "99.0"
            assert prompt.system_message == "Custom message"
            assert prompt.metadata["custom"] is True

    def test_load_multiple_custom_prompts(self):
        """Test loading multiple custom prompts."""
        with TemporaryDirectory() as tmpdir:
            # Create two custom prompts
            for i in range(2):
                custom = {
                    "agent_key": f"custom_agent_{i}",
                    "agent_name": f"Custom {i}",
                    "version": "1.0",
                    "system_message": f"Message {i}",
                }

                with open(f"{tmpdir}/custom_agent_{i}.json", "w") as f:
                    json.dump(custom, f)

            registry = PromptRegistry(prompts_dir=tmpdir)

            assert "custom_agent_0" in registry.prompts
            assert "custom_agent_1" in registry.prompts

    def test_malformed_json_skipped(self):
        """Test malformed JSON file is skipped."""
        with TemporaryDirectory() as tmpdir:
            # Create malformed JSON
            with open(f"{tmpdir}/bad.json", "w") as f:
                f.write("{invalid json")

            # Should not raise error
            registry = PromptRegistry(prompts_dir=tmpdir)

            # Should still have default prompts
            assert len(registry.prompts) > 0

    def test_missing_agent_key_skipped(self):
        """Test JSON without agent_key is skipped."""
        with TemporaryDirectory() as tmpdir:
            bad_prompt = {
                "agent_name": "Bad Agent",
                "version": "1.0",
                "system_message": "Message",
            }

            with open(f"{tmpdir}/bad.json", "w") as f:
                json.dump(bad_prompt, f)

            registry = PromptRegistry(prompts_dir=tmpdir)

            # Should not have loaded bad prompt
            assert "bad_agent" not in registry.prompts


class TestGetMethod:
    """Test get() method."""

    def test_get_existing_prompt(self):
        """Test retrieving existing prompt."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            prompt = registry.get("market_analyst")

            assert prompt is not None
            assert prompt.agent_key == "market_analyst"

    def test_get_nonexistent_prompt(self):
        """Test retrieving non-existent prompt returns None."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            prompt = registry.get("nonexistent")

            assert prompt is None

    def test_get_with_env_override(self, monkeypatch):
        """Test environment variable override."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            # Set environment override
            monkeypatch.setenv("PROMPT_MARKET_ANALYST", "Override message")

            prompt = registry.get("market_analyst")

            assert prompt.system_message == "Override message"
            assert prompt.version.endswith("-env")
            assert prompt.metadata["source"] == "environment"

    def test_env_override_preserves_metadata(self, monkeypatch):
        """Test env override preserves other fields."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            monkeypatch.setenv("PROMPT_MARKET_ANALYST", "Override")

            prompt = registry.get("market_analyst")

            # Should preserve these from base prompt
            assert prompt.agent_name == "Market Analyst"
            assert prompt.category == "technical"
            assert prompt.requires_tools is True


class TestGetAllMethod:
    """Test get_all() method."""

    def test_get_all_returns_dict(self):
        """Test get_all returns dictionary."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            all_prompts = registry.get_all()

            assert isinstance(all_prompts, dict)

    def test_get_all_returns_copy(self):
        """Test get_all returns copy, not reference."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            all_prompts = registry.get_all()
            all_prompts.clear()

            # Original should still have prompts
            assert len(registry.prompts) > 0

    def test_get_all_contains_all_prompts(self):
        """Test get_all contains all loaded prompts."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            all_prompts = registry.get_all()

            assert len(all_prompts) == len(registry.prompts)


class TestListKeys:
    """Test list_keys() method."""

    def test_list_keys_returns_list(self):
        """Test list_keys returns list."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            keys = registry.list_keys()

            assert isinstance(keys, list)

    def test_list_keys_contains_expected(self):
        """Test list_keys contains expected keys."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            keys = registry.list_keys()

            assert "market_analyst" in keys
            assert "fundamentals_analyst" in keys


class TestExportToJson:
    """Test export_to_json() method."""

    def test_export_creates_files(self):
        """Test export creates JSON files."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            export_dir = f"{tmpdir}/export"
            registry.export_to_json(export_dir)

            # Check files were created
            export_path = Path(export_dir)
            assert export_path.exists()
            assert (export_path / "market_analyst.json").exists()

    def test_export_valid_json(self):
        """Test exported files are valid JSON."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

            export_dir = f"{tmpdir}/export"
            registry.export_to_json(export_dir)

            # Load and validate JSON
            with open(f"{export_dir}/market_analyst.json") as f:
                data = json.load(f)

            assert data["agent_key"] == "market_analyst"
            assert "system_message" in data

    def test_export_preserves_all_fields(self):
        """Test export preserves all fields."""
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

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
        with TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(prompts_dir=tmpdir)

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
        with TemporaryDirectory() as tmpdir:
            registry1 = PromptRegistry(prompts_dir=tmpdir)
            registry2 = PromptRegistry(prompts_dir=tmpdir)

            # Should have same keys
            assert set(registry1.list_keys()) == set(registry2.list_keys())

            # Should have same versions
            for key in registry1.list_keys():
                assert registry1.get(key).version == registry2.get(key).version

    def test_export_import_roundtrip(self):
        """Test export then import produces same data."""
        with TemporaryDirectory() as tmpdir:
            registry1 = PromptRegistry(prompts_dir=tmpdir)

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
