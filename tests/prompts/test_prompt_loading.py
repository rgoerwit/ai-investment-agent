"""
Tests for prompt file loading and validation.

Ensures all JSON prompt files in prompts/ directory:
1. Are valid JSON
2. Load without errors into AgentPrompt dataclass
3. Actually override the default prompts in PromptRegistry
4. Have required fields and valid structure

This catches issues like:
- JSON syntax errors
- Schema mismatches (unexpected fields)
- Prompts not being loaded (silently using defaults)
"""

import json
from pathlib import Path

import pytest


class TestPromptFilesValid:
    """Test that all prompt JSON files are valid and parseable."""

    @pytest.fixture
    def prompts_dir(self) -> Path:
        """Get the prompts directory."""
        return Path("prompts")

    @pytest.fixture
    def prompt_files(self, prompts_dir: Path) -> list[Path]:
        """Get all JSON files in prompts directory."""
        if not prompts_dir.exists():
            pytest.skip("prompts/ directory not found")
        return list(prompts_dir.glob("*.json"))

    def test_prompts_directory_exists(self, prompts_dir: Path):
        """Verify prompts directory exists."""
        assert prompts_dir.exists(), "prompts/ directory should exist"
        assert prompts_dir.is_dir(), "prompts/ should be a directory"

    def test_prompt_files_exist(self, prompt_files: list[Path]):
        """Verify at least some prompt files exist."""
        assert len(prompt_files) > 0, "Should have at least one prompt file"

    def test_all_prompt_files_are_valid_json(self, prompt_files: list[Path]):
        """Verify all prompt files are valid JSON."""
        errors = []
        for prompt_file in prompt_files:
            try:
                with open(prompt_file) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                errors.append(f"{prompt_file.name}: {e}")

        assert not errors, "Invalid JSON files:\n" + "\n".join(errors)

    def test_all_prompt_files_have_required_fields(self, prompt_files: list[Path]):
        """Verify all prompt files have required AgentPrompt fields."""
        required_fields = ["agent_key", "agent_name", "version", "system_message"]
        errors = []

        for prompt_file in prompt_files:
            with open(prompt_file) as f:
                data = json.load(f)

            missing = [field for field in required_fields if field not in data]
            if missing:
                errors.append(f"{prompt_file.name}: missing {missing}")

        assert not errors, "Files missing required fields:\n" + "\n".join(errors)

    def test_all_prompt_files_have_valid_agent_key(self, prompt_files: list[Path]):
        """Verify agent_key is a non-empty string."""
        errors = []

        for prompt_file in prompt_files:
            with open(prompt_file) as f:
                data = json.load(f)

            agent_key = data.get("agent_key")
            if not agent_key or not isinstance(agent_key, str):
                errors.append(f"{prompt_file.name}: invalid agent_key={agent_key!r}")

        assert not errors, "Files with invalid agent_key:\n" + "\n".join(errors)


class TestPromptRegistryLoading:
    """Test that PromptRegistry loads all custom prompts without errors."""

    def test_registry_loads_without_errors(self):
        """Verify PromptRegistry initializes without exceptions."""
        from src.prompts import PromptRegistry

        # This should not raise any exceptions
        registry = PromptRegistry()

        assert registry is not None
        assert len(registry.prompts) > 0

    def test_all_prompt_files_loaded_into_registry(self):
        """Verify each JSON file's agent_key appears in the registry."""
        from src.prompts import PromptRegistry

        prompts_dir = Path("prompts")
        if not prompts_dir.exists():
            pytest.skip("prompts/ directory not found")

        registry = PromptRegistry()
        errors = []

        for prompt_file in prompts_dir.glob("*.json"):
            with open(prompt_file) as f:
                data = json.load(f)

            agent_key = data.get("agent_key")
            if agent_key and agent_key not in registry.prompts:
                errors.append(
                    f"{prompt_file.name}: agent_key={agent_key!r} not in registry"
                )

        assert not errors, "Prompts not loaded:\n" + "\n".join(errors)

    def test_custom_prompts_override_defaults(self):
        """Verify custom prompt versions match JSON files (not defaults)."""
        from src.prompts import PromptRegistry

        prompts_dir = Path("prompts")
        if not prompts_dir.exists():
            pytest.skip("prompts/ directory not found")

        registry = PromptRegistry()
        mismatches = []

        for prompt_file in prompts_dir.glob("*.json"):
            with open(prompt_file) as f:
                data = json.load(f)

            agent_key = data.get("agent_key")
            expected_version = data.get("version")

            if agent_key and agent_key in registry.prompts:
                loaded_prompt = registry.prompts[agent_key]
                if loaded_prompt.version != expected_version:
                    mismatches.append(
                        f"{prompt_file.name}: expected v{expected_version}, "
                        f"got v{loaded_prompt.version} (using default?)"
                    )

        assert not mismatches, "Custom prompts not overriding defaults:\n" + "\n".join(
            mismatches
        )

    def test_loaded_prompts_have_correct_metadata(self):
        """Verify metadata dict is preserved when loading."""
        from src.prompts import PromptRegistry

        prompts_dir = Path("prompts")
        if not prompts_dir.exists():
            pytest.skip("prompts/ directory not found")

        registry = PromptRegistry()

        for prompt_file in prompts_dir.glob("*.json"):
            with open(prompt_file) as f:
                data = json.load(f)

            agent_key = data.get("agent_key")
            if not agent_key or agent_key not in registry.prompts:
                continue

            loaded_prompt = registry.prompts[agent_key]
            expected_metadata = data.get("metadata", {})

            # Check that expected metadata keys are present
            for key in expected_metadata:
                assert (
                    key in loaded_prompt.metadata
                ), f"{prompt_file.name}: metadata key {key!r} not loaded"


class TestSpecificPromptFiles:
    """Test specific prompt files that have unique requirements."""

    def test_writer_json_has_model_config_in_metadata(self):
        """Verify writer.json has model_config nested in metadata."""
        prompt_file = Path("prompts/writer.json")
        if not prompt_file.exists():
            pytest.skip("writer.json not found")

        with open(prompt_file) as f:
            data = json.load(f)

        # model_config should be in metadata, not at top level
        assert (
            "model_config" not in data
        ), "model_config should be in metadata, not top level"
        assert "metadata" in data, "writer.json should have metadata"
        assert "model_config" in data["metadata"], "model_config should be in metadata"

    def test_writer_json_has_user_template_in_metadata(self):
        """Verify writer.json has user_template nested in metadata."""
        prompt_file = Path("prompts/writer.json")
        if not prompt_file.exists():
            pytest.skip("writer.json not found")

        with open(prompt_file) as f:
            data = json.load(f)

        # user_template should be in metadata, not at top level
        assert (
            "user_template" not in data
        ), "user_template should be in metadata, not top level"
        assert "metadata" in data, "writer.json should have metadata"
        assert (
            "user_template" in data["metadata"]
        ), "user_template should be in metadata"

    def test_fundamentals_analyst_has_data_block_instructions(self):
        """Verify fundamentals_analyst.json has DATA_BLOCK instructions."""
        prompt_file = Path("prompts/fundamentals_analyst.json")
        if not prompt_file.exists():
            pytest.skip("fundamentals_analyst.json not found")

        with open(prompt_file) as f:
            data = json.load(f)

        system_message = data.get("system_message", "")
        assert (
            "DATA_BLOCK" in system_message
        ), "fundamentals_analyst should have DATA_BLOCK instructions"

    def test_portfolio_manager_has_thesis_criteria(self):
        """Verify portfolio_manager.json has investment thesis criteria."""
        prompt_file = Path("prompts/portfolio_manager.json")
        if not prompt_file.exists():
            pytest.skip("portfolio_manager.json not found")

        with open(prompt_file) as f:
            data = json.load(f)

        system_message = data.get("system_message", "")
        # Should mention key thesis elements
        assert (
            "Financial Health" in system_message or "HEALTH" in system_message
        ), "portfolio_manager should reference Financial Health criteria"

    def test_writer_json_has_valuation_reconciliation_section(self):
        """Verify writer.json has VALUATION-DECISION RECONCILIATION section."""
        prompt_file = Path("prompts/writer.json")
        if not prompt_file.exists():
            pytest.skip("writer.json not found")

        with open(prompt_file) as f:
            data = json.load(f)

        system_message = data.get("system_message", "")
        # Check for the reconciliation section
        assert (
            "VALUATION-DECISION RECONCILIATION" in system_message
        ), "writer.json should have VALUATION-DECISION RECONCILIATION section"
        # Check for key instructions within that section
        assert (
            "narrative tension" in system_message.lower()
        ), "writer.json should mention 'narrative tension' in reconciliation section"
        assert (
            "above" in system_message.lower() and "fair value" in system_message.lower()
        ), "writer.json should describe handling price above fair value"

    def test_writer_json_has_valuation_context_placeholder(self):
        """Verify writer.json user_template includes valuation_context placeholder."""
        prompt_file = Path("prompts/writer.json")
        if not prompt_file.exists():
            pytest.skip("writer.json not found")

        with open(prompt_file) as f:
            data = json.load(f)

        user_template = data.get("metadata", {}).get("user_template", "")
        assert (
            "{valuation_context}" in user_template
        ), "writer.json user_template should include {valuation_context} placeholder"
        # Check that valuation context section exists (XML tag or header)
        assert (
            "valuation_context" in user_template.lower()
        ), "writer.json user_template should have valuation context section"

    def test_value_trap_detector_has_insider_concentration_thresholds(self):
        """Verify value_trap_detector.json has explicit insider concentration thresholds."""
        prompt_file = Path("prompts/value_trap_detector.json")
        if not prompt_file.exists():
            pytest.skip("value_trap_detector.json not found")

        with open(prompt_file) as f:
            data = json.load(f)

        system_message = data.get("system_message", "")
        # Check for INSIDER_CONCENTRATION section
        assert (
            "INSIDER/FAMILY CONCENTRATION" in system_message
        ), "value_trap_detector should have INSIDER/FAMILY CONCENTRATION section"
        # Check for explicit thresholds
        assert (
            ">50%" in system_message and "HIGH" in system_message
        ), "value_trap_detector should have >50% HIGH threshold"
        assert "30-50%" in system_message or (
            "30" in system_message and "MODERATE" in system_message
        ), "value_trap_detector should have 30-50% MODERATE threshold"
        # Check for key insight about family control
        assert (
            "does NOT equal alignment" in system_message
        ), "value_trap_detector should clarify family control != minority alignment"

    def test_value_trap_detector_has_latin_america_terminology(self):
        """Verify value_trap_detector.json has Latin America terminology section."""
        prompt_file = Path("prompts/value_trap_detector.json")
        if not prompt_file.exists():
            pytest.skip("value_trap_detector.json not found")

        with open(prompt_file) as f:
            data = json.load(f)

        system_message = data.get("system_message", "")
        # Check for LATIN AMERICA section
        assert (
            "LATIN AMERICA" in system_message
        ), "value_trap_detector should have LATIN AMERICA terminology section"
        # Check for key Spanish terms
        assert (
            "Empresa familiar" in system_message
        ), "value_trap_detector should include 'Empresa familiar' term"
        assert (
            "Accionista mayoritario" in system_message
            or "controlador" in system_message
        ), "value_trap_detector should include majority shareholder terms"

    def test_value_trap_detector_output_format_has_insider_concentration(self):
        """Verify value_trap_detector output format includes INSIDER_CONCENTRATION field."""
        prompt_file = Path("prompts/value_trap_detector.json")
        if not prompt_file.exists():
            pytest.skip("value_trap_detector.json not found")

        with open(prompt_file) as f:
            data = json.load(f)

        system_message = data.get("system_message", "")
        # Check output format includes the new field
        assert (
            "INSIDER_CONCENTRATION: [HIGH | MODERATE | LOW]" in system_message
        ), "value_trap_detector output format should include INSIDER_CONCENTRATION field"


class TestAgentPromptDataclass:
    """Test the AgentPrompt dataclass structure."""

    def test_agent_prompt_accepts_valid_data(self):
        """Verify AgentPrompt can be instantiated with valid data."""
        from src.prompts import AgentPrompt

        prompt = AgentPrompt(
            agent_key="test_agent",
            agent_name="Test Agent",
            version="1.0",
            system_message="You are a test agent.",
            category="test",
            requires_tools=False,
            metadata={"key": "value"},
        )

        assert prompt.agent_key == "test_agent"
        assert prompt.version == "1.0"
        assert prompt.metadata == {"key": "value"}

    def test_agent_prompt_rejects_unknown_fields(self):
        """Verify AgentPrompt rejects unexpected fields."""
        from src.prompts import AgentPrompt

        with pytest.raises(TypeError) as exc_info:
            AgentPrompt(
                agent_key="test_agent",
                agent_name="Test Agent",
                version="1.0",
                system_message="You are a test agent.",
                unknown_field="this should fail",  # Not in dataclass
            )

        assert "unknown_field" in str(exc_info.value)

    def test_agent_prompt_metadata_defaults_to_empty_dict(self):
        """Verify metadata defaults to empty dict if not provided."""
        from src.prompts import AgentPrompt

        prompt = AgentPrompt(
            agent_key="test_agent",
            agent_name="Test Agent",
            version="1.0",
            system_message="You are a test agent.",
        )

        assert prompt.metadata == {}


class TestPromptVersionTracking:
    """Test that prompt versions are tracked correctly."""

    def test_all_prompts_have_semantic_versions(self):
        """Verify all prompts use semantic version format (X.Y or X.Y.Z)."""
        import re

        prompts_dir = Path("prompts")
        if not prompts_dir.exists():
            pytest.skip("prompts/ directory not found")

        version_pattern = re.compile(r"^\d+\.\d+(\.\d+)?$")
        errors = []

        for prompt_file in prompts_dir.glob("*.json"):
            with open(prompt_file) as f:
                data = json.load(f)

            version = data.get("version", "")
            if not version_pattern.match(version):
                errors.append(f"{prompt_file.name}: invalid version={version!r}")

        assert not errors, "Invalid version formats:\n" + "\n".join(errors)

    def test_prompt_versions_are_documented_in_metadata(self):
        """Verify prompts with metadata document changes."""
        prompts_dir = Path("prompts")
        if not prompts_dir.exists():
            pytest.skip("prompts/ directory not found")

        # These prompts should have change documentation
        documented_prompts = []

        for prompt_file in prompts_dir.glob("*.json"):
            with open(prompt_file) as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            if "changes" in metadata or "last_updated" in metadata:
                documented_prompts.append(prompt_file.name)

        # At least some prompts should have documentation
        assert (
            len(documented_prompts) > 0
        ), "At least one prompt should have change documentation in metadata"
