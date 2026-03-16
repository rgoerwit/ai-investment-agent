"""
Tests for Portfolio Manager resolution utilities:
- resolve_pfic_display_status (PFIC canonical status)
- normalize_governance_terms (governance terminology correction)
"""

import pytest

from src.agents.decision_nodes import resolve_pfic_display_status
from src.report_generator import normalize_governance_terms


class TestResolvePficDisplayStatus:
    """Tests for PFIC status canonicalization."""

    def test_legal_uncertain_overrides_data_block_low(self):
        """Legal Counsel UNCERTAIN takes precedence over DATA_BLOCK LOW."""
        status, note = resolve_pfic_display_status("UNCERTAIN", "LOW")
        assert status == "UNCERTAIN"
        assert note is not None
        assert "Legal Counsel" in note
        assert "UNCERTAIN" in note

    def test_legal_probable_overrides_data_block_low(self):
        """Legal Counsel PROBABLE maps to HIGH, overrides DATA_BLOCK LOW."""
        status, note = resolve_pfic_display_status("PROBABLE", "LOW")
        assert status == "HIGH"
        assert note is not None
        assert "Legal Counsel" in note

    def test_legal_clean_defers_to_data_block(self):
        """CLEAN Legal status → return DATA_BLOCK value unchanged."""
        status, note = resolve_pfic_display_status("CLEAN", "LOW")
        assert status == "LOW"
        assert note is None

    def test_legal_clean_case_insensitive(self):
        """CLEAN check is case-insensitive."""
        status, note = resolve_pfic_display_status("clean", "MEDIUM")
        assert status == "MEDIUM"
        assert note is None

    def test_legal_none_defers_to_data_block(self):
        """None Legal status → return DATA_BLOCK value unchanged."""
        status, note = resolve_pfic_display_status(None, "MEDIUM")
        assert status == "MEDIUM"
        assert note is None

    def test_legal_na_defers_to_data_block(self):
        """N/A Legal status → return DATA_BLOCK value unchanged."""
        status, note = resolve_pfic_display_status("N/A", "MEDIUM")
        assert status == "MEDIUM"
        assert note is None

    def test_both_none_returns_na(self):
        """Both inputs None → returns 'N/A'."""
        status, note = resolve_pfic_display_status(None, None)
        assert status == "N/A"
        assert note is None

    def test_legal_probable_data_block_medium_no_note(self):
        """No note when data_block is not LOW (override only fires on LOW)."""
        status, note = resolve_pfic_display_status("PROBABLE", "MEDIUM")
        assert status == "HIGH"
        assert note is None

    def test_legal_uncertain_data_block_none_returns_uncertain(self):
        """UNCERTAIN with no DATA_BLOCK → returns UNCERTAIN, no note."""
        status, note = resolve_pfic_display_status("UNCERTAIN", None)
        assert status == "UNCERTAIN"
        assert note is None

    def test_unknown_legal_status_defers_to_data_block(self):
        """Unrecognised legal status → defers to DATA_BLOCK."""
        status, note = resolve_pfic_display_status("UNKNOWN_STATUS", "HIGH")
        assert status == "HIGH"
        assert note is None


class TestNormalizeGovernanceTerms:
    """Tests for deterministic governance terminology corrections."""

    def test_controlled_subsidiary_replaced(self):
        text = "This is a controlled subsidiary structure."
        result = normalize_governance_terms(text)
        assert "controlled company" in result
        assert "controlled subsidiary" not in result

    def test_controlled_subsidiary_case_insensitive(self):
        text = "Operates as a Controlled Subsidiary of XYZ Corp."
        result = normalize_governance_terms(text)
        assert (
            "controlled subsidiary" not in result.lower()
            or "controlled company" in result.lower()
        )

    def test_parent_subsidiary_relationship_replaced(self):
        text = "The parent subsidiary relationship creates governance risks."
        result = normalize_governance_terms(text)
        assert "controlling shareholder relationship" in result
        assert "parent subsidiary relationship" not in result

    def test_no_match_unchanged(self):
        text = "The company has strong free cash flow and low leverage."
        assert normalize_governance_terms(text) == text

    def test_empty_string(self):
        assert normalize_governance_terms("") == ""

    def test_multiple_occurrences_all_replaced(self):
        text = "A controlled subsidiary. Also a controlled subsidiary."
        result = normalize_governance_terms(text)
        assert result.count("controlled company") == 2
        assert "controlled subsidiary" not in result
