"""
Tests for the truncation detection utility.
"""

import pytest

from src.utils import detect_truncation


class TestCodeLevelTruncation:
    """Test detection of code-level truncation markers."""

    def test_truncated_marker_detected(self):
        """TRUNCATED marker should be detected as code truncation."""
        text = "Some content\n[...TRUNCATED 5000 chars...]\nMore content"
        result = detect_truncation(text)
        assert result["truncated"] is True
        assert result["source"] == "code"
        assert result["confidence"] == "high"

    def test_truncated_lowercase_detected(self):
        """Lowercase truncated marker should also be detected."""
        text = "Content here\n[...truncated]\nEnd"
        result = detect_truncation(text)
        assert result["truncated"] is True
        assert result["source"] == "code"

    def test_note_data_truncated_detected(self):
        """NOTE: Data truncated marker should be detected."""
        text = "Data here\n[NOTE: Data truncated due to size limits]\nMore"
        result = detect_truncation(text)
        assert result["truncated"] is True
        assert result["source"] == "code"


class TestLLMTruncation:
    """Test detection of LLM-level truncation heuristics."""

    def test_ends_mid_sentence(self):
        """Text ending mid-sentence should be flagged as LLM truncation."""
        text = "The company's revenue grew by 15% driven by"
        result = detect_truncation(text)
        assert result["truncated"] is True
        assert result["source"] == "llm"
        assert result["confidence"] == "medium"

    def test_ends_mid_word(self):
        """Text ending mid-word should be flagged."""
        text = "The analysis shows that the comp"
        result = detect_truncation(text)
        assert result["truncated"] is True
        assert result["source"] == "llm"

    def test_incomplete_pm_block(self):
        """PM_BLOCK without VERDICT should be flagged."""
        text = "PM_BLOCK:\nTICKER: TEST.X\nCONVICTION: 75%"
        result = detect_truncation(text)
        assert result["truncated"] is True
        assert result["source"] == "llm"
        assert "PM_BLOCK" in result["marker"]

    def test_incomplete_data_block(self):
        """DATA_BLOCK without scores should be flagged."""
        text = "DATA_BLOCK:\nTICKER: TEST.X\nSECTOR: Technology"
        result = detect_truncation(text)
        assert result["truncated"] is True
        assert "DATA_BLOCK" in result["marker"]


class TestCompleteOutput:
    """Test that complete outputs are not falsely flagged."""

    def test_complete_sentence(self):
        """Complete sentence should not be flagged."""
        text = "Analysis complete. VERDICT: BUY with 75% conviction."
        result = detect_truncation(text)
        assert result["truncated"] is False

    def test_complete_pm_block(self):
        """Complete PM_BLOCK should not be flagged."""
        text = "PM_BLOCK:\nTICKER: TEST.X\nVERDICT: BUY\nRISK_ZONE: LOW"
        result = detect_truncation(text)
        assert result["truncated"] is False

    def test_complete_data_block(self):
        """Complete DATA_BLOCK should not be flagged."""
        text = "DATA_BLOCK:\nHEALTH_SCORE: 75\nGROWTH_SCORE: 60"
        result = detect_truncation(text)
        assert result["truncated"] is False

    def test_ends_with_period(self):
        """Text ending with period should not be flagged."""
        text = "The investment thesis is sound."
        result = detect_truncation(text)
        assert result["truncated"] is False

    def test_ends_with_quote(self):
        """Text ending with quote should not be flagged."""
        text = 'The CEO said "We are optimistic"'
        result = detect_truncation(text)
        assert result["truncated"] is False

    def test_ends_with_newline(self):
        """Text ending with newline should not be flagged."""
        text = "Analysis complete.\n"
        result = detect_truncation(text)
        assert result["truncated"] is False


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string(self):
        """Empty string should not be flagged as truncated."""
        result = detect_truncation("")
        assert result["truncated"] is False

    def test_none_input(self):
        """None input should not crash."""
        result = detect_truncation(None)
        assert result["truncated"] is False

    def test_whitespace_only(self):
        """Whitespace-only input should not be flagged."""
        result = detect_truncation("   \n\t  ")
        assert result["truncated"] is False

    def test_very_short_text(self):
        """Very short text should still work."""
        result = detect_truncation("OK.")
        assert result["truncated"] is False

    def test_non_string_input(self):
        """Non-string input should not crash."""
        result = detect_truncation(12345)
        assert result["truncated"] is False


class TestAgentScoping:
    """Test that block checks are scoped to the producing agent."""

    def test_consultant_mentioning_data_block_not_flagged(self):
        """Consultant referencing DATA_BLOCK in prose should not trigger false positive."""
        text = (
            "CONSULTANT REVIEW: CONDITIONAL APPROVAL\n"
            "The DATA_BLOCK: shows P/E of 10.3 which is consistent with filings.\n"
            "Overall analysis is sound."
        )
        result = detect_truncation(text, agent="consultant")
        assert result["truncated"] is False

    def test_fundamentals_analyst_incomplete_data_block_flagged(self):
        """Fundamentals analyst with incomplete DATA_BLOCK should still be caught."""
        text = "DATA_BLOCK:\nTICKER: TEST.X\nSECTOR: Technology"
        result = detect_truncation(text, agent="fundamentals_analyst")
        assert result["truncated"] is True
        assert "DATA_BLOCK" in result["marker"]

    def test_no_agent_checks_all_blocks(self):
        """Without agent param, all block checks apply (backward compat)."""
        text = "DATA_BLOCK:\nTICKER: TEST.X\nSECTOR: Technology"
        result = detect_truncation(text)
        assert result["truncated"] is True
