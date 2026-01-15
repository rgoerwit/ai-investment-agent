"""
Integration tests for quantitative validation changes.

Verifies that:
1. Fundamentals Analyst has moat thresholds
2. Fundamentals Analyst prioritizes OCF over FCF
3. Value Trap Detector extracts ROIC from DATA_BLOCK
4. Portfolio Manager can process the updated outputs
"""

import re

import pytest

from src.prompts import get_prompt


class TestFundamentalsAnalystMoatThresholds:
    """Test that Fundamentals Analyst has explicit moat thresholds."""

    def test_moat_threshold_section_exists(self):
        """Moat threshold section should exist."""
        prompt = get_prompt("fundamentals_analyst")
        assert "MOAT SIGNAL THRESHOLDS" in prompt.system_message

    def test_moat_margin_stability_thresholds(self):
        """Should have explicit CV thresholds for margin stability."""
        prompt = get_prompt("fundamentals_analyst")
        msg = prompt.system_message

        # Should have all three threshold levels
        assert "CV < 0.08" in msg
        assert "CV 0.08-0.15" in msg
        assert "CV > 0.15" in msg

        # Should map to categorical levels
        assert "HIGH (stable pricing power" in msg
        assert "MEDIUM (moderate stability" in msg
        assert "LOW (volatile margins" in msg

    def test_moat_cash_conversion_thresholds(self):
        """Should have explicit CFO/NI thresholds for cash conversion."""
        prompt = get_prompt("fundamentals_analyst")
        msg = prompt.system_message

        # Should have all three threshold levels
        assert "Ratio > 0.90" in msg
        assert "Ratio 0.70-0.90" in msg
        assert "Ratio < 0.70" in msg

        # Should map to categorical levels
        assert "STRONG (genuine high-quality earnings" in msg
        assert "ADEQUATE (acceptable cash conversion" in msg
        assert "WEAK (poor earnings quality" in msg

    def test_references_junior_analyst_data(self):
        """Should instruct to use Junior Analyst data."""
        prompt = get_prompt("fundamentals_analyst")
        msg = prompt.system_message

        assert "Junior Analyst provides calculated moat metrics" in msg
        assert "moat_grossMarginCV" in msg
        assert "moat_cfoToNiAvg" in msg

    def test_warns_against_inventing_thresholds(self):
        """Should explicitly warn against inventing thresholds."""
        prompt = get_prompt("fundamentals_analyst")
        msg = prompt.system_message

        assert "Do NOT invent your own thresholds" in msg


class TestFundamentalsAnalystOCFPriority:
    """Test that Fundamentals Analyst prioritizes OCF over FCF."""

    def test_ocf_is_primary_measure(self):
        """OCF should be primary measure in Cash Generation."""
        prompt = get_prompt("fundamentals_analyst")
        msg = prompt.system_message

        # Find Cash Generation section
        cash_gen_match = re.search(
            r"\*\*Cash Generation \(2 pts\)\*\*:(.+?)(?=\n\*\*|\Z)", msg, re.DOTALL
        )
        assert cash_gen_match, "Cash Generation section not found"

        cash_gen_text = cash_gen_match.group(1)

        # Should mention OCF as primary
        assert "Operating Cash Flow >0: 1 pt" in cash_gen_text
        assert "primary measure - operating health" in cash_gen_text

        # FCF should be secondary (0.5 pt)
        assert "Free Cash Flow >0: 0.5 pt additional" in cash_gen_text

    def test_ocf_fcf_note_exists(self):
        """Should have note about OCF+/FCF- being acceptable."""
        prompt = get_prompt("fundamentals_analyst")
        msg = prompt.system_message

        assert "Positive OCF with negative FCF for 1-2 years is acceptable" in msg
        assert "growth/capex cycles" in msg

    def test_ocf_sanity_check_enhanced(self):
        """OCF Sanity cross-check should distinguish OCF vs FCF issues."""
        prompt = get_prompt("fundamentals_analyst")
        msg = prompt.system_message

        # Find OCF Sanity check
        ocf_check_match = re.search(
            r"6\. \*\*OCF Sanity\*\*:(.+?)(?=\n\d+\.|\Z)", msg, re.DOTALL
        )
        assert ocf_check_match, "OCF Sanity check not found"

        ocf_check_text = ocf_check_match.group(1)

        # Should have both checks
        assert "(Net Income >0) AND (OCF <0)" in ocf_check_text
        assert "CRITICAL risk" in ocf_check_text
        assert "(OCF >0) AND (FCF <0) for 2+ years" in ocf_check_text
        assert "High capex cycle" in ocf_check_text


class TestValueTrapROICIntegration:
    """Test that Value Trap Detector integrates ROIC from DATA_BLOCK."""

    def test_roic_extraction_instruction_exists(self):
        """Should have instruction to extract ROIC from DATA_BLOCK."""
        prompt = get_prompt("value_trap_detector")
        msg = prompt.system_message

        assert "EXTRACT CAPITAL EFFICIENCY METRICS" in msg
        assert "ROIC_QUALITY from Fundamentals Analyst DATA_BLOCK" in msg
        assert "STRONG/ADEQUATE/WEAK/DESTRUCTIVE/N/A" in msg

    def test_capital_allocation_has_roic_fields(self):
        """CAPITAL_ALLOCATION block should have ROIC fields."""
        prompt = get_prompt("value_trap_detector")
        msg = prompt.system_message

        # Find CAPITAL_ALLOCATION section
        capital_alloc_match = re.search(
            r"CAPITAL_ALLOCATION:(.+?)(?=\nCATALYSTS:)", msg, re.DOTALL
        )
        assert capital_alloc_match, "CAPITAL_ALLOCATION section not found"

        capital_alloc_text = capital_alloc_match.group(1)

        # Should have ROIC fields at top
        assert "ROIC_QUALITY:" in capital_alloc_text
        assert "ROIC_PERCENT:" in capital_alloc_text

        # ROIC_QUALITY should come before RATING
        roic_pos = capital_alloc_text.index("ROIC_QUALITY:")
        rating_pos = capital_alloc_text.index("RATING:")
        assert roic_pos < rating_pos

    def test_rating_uses_roic_quality(self):
        """RATING logic should reference ROIC_QUALITY."""
        prompt = get_prompt("value_trap_detector")
        msg = prompt.system_message

        # Find CAPITAL_ALLOCATION section
        capital_alloc_match = re.search(
            r"CAPITAL_ALLOCATION:(.+?)(?=\nCATALYSTS:)", msg, re.DOTALL
        )
        assert capital_alloc_match

        capital_alloc_text = capital_alloc_match.group(1)

        # Should have ROIC-based logic
        assert "POOR if: ROIC_QUALITY = DESTRUCTIVE or WEAK" in capital_alloc_text
        assert "POOR if: Net buybacks executed while ROIC <10%" in capital_alloc_text
        assert "GOOD if: ROIC_QUALITY = STRONG" in capital_alloc_text

    def test_preserves_qualitative_context(self):
        """Should preserve M&A and buyback context fields."""
        prompt = get_prompt("value_trap_detector")
        msg = prompt.system_message

        # Find CAPITAL_ALLOCATION section
        capital_alloc_match = re.search(
            r"CAPITAL_ALLOCATION:(.+?)(?=\nCATALYSTS:)", msg, re.DOTALL
        )
        assert capital_alloc_match

        capital_alloc_text = capital_alloc_match.group(1)

        # Should still have qualitative fields
        assert "M&A_CONTEXT:" in capital_alloc_text
        assert "BUYBACK_CONTEXT:" in capital_alloc_text
        assert "PAYOUT_TREND:" in capital_alloc_text
        assert "CASH_POSITION:" in capital_alloc_text


class TestPortfolioManagerCompatibility:
    """Test that Portfolio Manager can still process updated outputs."""

    def test_moat_flags_mentioned(self):
        """PM should mention moat signals."""
        prompt = get_prompt("portfolio_manager")
        msg = prompt.system_message

        # Should reference moat signals
        assert "Moat Signal" in msg or "moat" in msg.lower()

    def test_capital_efficiency_flags_mentioned(self):
        """PM should mention capital efficiency flags."""
        prompt = get_prompt("portfolio_manager")
        msg = prompt.system_message

        # Should have capital efficiency section
        assert "Capital Efficiency" in msg
        assert "CAPITAL_VALUE_DESTRUCTION" in msg
        assert "CAPITAL_ENGINEERED_RETURNS" in msg

    def test_roic_in_qualitative_risk_tally(self):
        """PM should have ROIC-related flags in risk tally."""
        prompt = get_prompt("portfolio_manager")
        msg = prompt.system_message

        # Find qualitative risk tally section
        risk_tally_match = re.search(
            r"COUNT QUALITATIVE RISK FACTORS:(.+?)(?=\n\*\*C\)|\Z)", msg, re.DOTALL
        )
        assert risk_tally_match, "Qualitative risk tally section not found"

        risk_tally_text = risk_tally_match.group(1)

        # Should list capital efficiency penalties
        assert "CAPITAL_VALUE_DESTRUCTION" in risk_tally_text
        assert "+1.5" in risk_tally_text  # Penalty value


class TestVersionUpdates:
    """Test that version numbers were updated correctly."""

    def test_fundamentals_analyst_version(self):
        """Fundamentals Analyst should be version 7.7."""
        prompt = get_prompt("fundamentals_analyst")
        assert prompt.version == "7.7"

    def test_value_trap_detector_version(self):
        """Value Trap Detector should be version 1.1."""
        prompt = get_prompt("value_trap_detector")
        assert prompt.version == "1.1"

    def test_fundamentals_analyst_metadata(self):
        """Fundamentals Analyst metadata should document changes."""
        prompt = get_prompt("fundamentals_analyst")
        changes = prompt.metadata.get("changes", "")

        assert "7.7" in changes
        assert "moat thresholds" in changes.lower()
        assert "OCF" in changes

    def test_value_trap_detector_metadata(self):
        """Value Trap Detector metadata should document changes."""
        prompt = get_prompt("value_trap_detector")
        changes = prompt.metadata.get("changes", "")

        assert "1.1" in changes
        assert "ROIC" in changes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
