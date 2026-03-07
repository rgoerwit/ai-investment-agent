"""Regression guards for research_manager prompt — TRANSIENT/STRUCTURAL instruction.

These tests fail the moment someone accidentally removes or corrupts the
TRANSIENT risk-duration instruction from the research_manager prompt.
No LLM is called; only the prompt JSON on disk is inspected.
"""

from __future__ import annotations

from src.prompts import get_prompt


class TestResearchManagerPromptContent:
    """Guard the TRANSIENT/STRUCTURAL risk-duration instruction in research_manager."""

    def test_version_is_5_2_or_higher(self):
        """Prompt version must be ≥5.2 (the version that introduced TRANSIENT tags).

        Bump the assertion when the prompt is intentionally revised to a new version.
        """
        prompt = get_prompt("research_manager")
        assert prompt is not None, "research_manager prompt not found in registry"
        major, minor = map(int, prompt.version.split(".")[:2])
        assert (major, minor) >= (5, 2), f"Expected ≥5.2, got {prompt.version}"

    def test_transient_duration_tag_present(self):
        """STEP 4 must instruct the LLM to classify risks as [TRANSIENT ...]."""
        prompt = get_prompt("research_manager")
        assert "[TRANSIENT" in prompt.system_message, (
            "TRANSIENT risk-duration tag missing from research_manager prompt. "
            "The LLM needs this to classify short-lived macro risks at 0.5× weight."
        )

    def test_half_weight_instruction_present(self):
        """The 0.5× tally reduction for TRANSIENT risks must be stated explicitly."""
        prompt = get_prompt("research_manager")
        assert "0.5" in prompt.system_message, (
            "0.5× weight instruction missing from research_manager prompt. "
            "Without it the LLM cannot downgrade transient macro risks."
        )

    def test_structural_tag_present(self):
        """STRUCTURAL tag must be present alongside TRANSIENT for contrast."""
        prompt = get_prompt("research_manager")
        assert (
            "STRUCTURAL" in prompt.system_message
        ), "STRUCTURAL risk-duration tag missing from research_manager prompt."

    def test_geopolitical_example_present(self):
        """Concrete geopolitical example helps the LLM classify risks correctly."""
        prompt = get_prompt("research_manager")
        assert "eopolitical" in prompt.system_message, (
            "Geopolitical example missing from research_manager prompt. "
            "Examples anchor the LLM's classification of TRANSIENT vs STRUCTURAL."
        )
