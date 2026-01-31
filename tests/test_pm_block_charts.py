"""
Tests for post-PM chart generation with PM_BLOCK extraction.

This module tests:
1. PM_BLOCK extraction from Portfolio Manager output
2. Conditional chart suppression for negative verdicts
3. Fallback to DATA_BLOCK when PM_BLOCK is missing
4. Valuation discount application based on risk zone
"""

import pytest

from src.charts.extractors.pm_block import (
    PMBlockData,
    extract_pm_block,
    extract_verdict_from_text,
)


class TestPMBlockExtraction:
    """Tests for PM_BLOCK extraction."""

    def test_extract_complete_pm_block(self):
        """Test extraction of a complete PM_BLOCK."""
        pm_output = """
### PORTFOLIO MANAGER VERDICT: BUY

Some text here...

### --- START PM_BLOCK ---
VERDICT: BUY
HEALTH_ADJ: 72
GROWTH_ADJ: 65
RISK_TALLY: 0.83
ZONE: LOW
SHOW_VALUATION_CHART: YES
VALUATION_DISCOUNT: 1.0
POSITION_SIZE: 4.5
VALUATION_CONTEXT: STANDARD
### --- END PM_BLOCK ---

More rationale here...
"""
        result = extract_pm_block(pm_output)

        assert result.verdict == "BUY"
        assert result.health_adj == 72
        assert result.growth_adj == 65
        assert result.risk_tally == 0.83
        assert result.zone == "LOW"
        assert result.show_valuation_chart is True
        assert result.valuation_discount == 1.0
        assert result.position_size == 4.5
        assert result.valuation_context == "STANDARD"

    def test_extract_do_not_initiate_verdict(self):
        """Test extraction of DO_NOT_INITIATE verdict."""
        pm_output = """
### --- START PM_BLOCK ---
VERDICT: DO_NOT_INITIATE
HEALTH_ADJ: 45
GROWTH_ADJ: 52
RISK_TALLY: 2.5
ZONE: HIGH
SHOW_VALUATION_CHART: NO
VALUATION_DISCOUNT: 0.0
POSITION_SIZE: 0.0
VALUATION_CONTEXT: N_A
### --- END PM_BLOCK ---
"""
        result = extract_pm_block(pm_output)

        assert result.verdict == "DO_NOT_INITIATE"
        assert result.show_valuation_chart is False
        assert result.valuation_discount == 0.0

    def test_extract_hold_with_moderate_risk(self):
        """Test HOLD verdict with moderate risk zone."""
        pm_output = """
### --- START PM_BLOCK ---
VERDICT: HOLD
HEALTH_ADJ: 68
GROWTH_ADJ: 55
RISK_TALLY: 1.33
ZONE: MODERATE
SHOW_VALUATION_CHART: YES
VALUATION_DISCOUNT: 0.9
POSITION_SIZE: 2.0
VALUATION_CONTEXT: CONTEXTUAL_PASS
### --- END PM_BLOCK ---
"""
        result = extract_pm_block(pm_output)

        assert result.verdict == "HOLD"
        assert result.zone == "MODERATE"
        assert result.valuation_discount == 0.9
        assert result.should_show_targets() is True

    def test_missing_pm_block_returns_defaults(self):
        """Test that missing PM_BLOCK returns default values."""
        pm_output = """
### PORTFOLIO MANAGER VERDICT: BUY

This is just plain text without a PM_BLOCK.
The decision is to buy with a 3% position.
"""
        result = extract_pm_block(pm_output)

        assert result.verdict is None
        assert result.health_adj is None
        assert result.growth_adj is None
        assert result.show_valuation_chart is True  # Default
        assert result.valuation_discount == 1.0  # Default

    def test_empty_input(self):
        """Test with empty input."""
        result = extract_pm_block("")
        assert result.verdict is None
        assert result.show_valuation_chart is True

    def test_none_input(self):
        """Test with None input."""
        result = extract_pm_block(None)
        assert result.verdict is None

    def test_uses_last_pm_block_for_self_correction(self):
        """Test that the last PM_BLOCK is used when multiple exist (self-correction)."""
        pm_output = """
### --- START PM_BLOCK ---
VERDICT: HOLD
HEALTH_ADJ: 60
RISK_TALLY: 1.5
ZONE: MODERATE
### --- END PM_BLOCK ---

Wait, I need to reconsider...

### --- START PM_BLOCK ---
VERDICT: BUY
HEALTH_ADJ: 72
RISK_TALLY: 0.8
ZONE: LOW
### --- END PM_BLOCK ---
"""
        result = extract_pm_block(pm_output)

        # Should use the last (corrected) block
        assert result.verdict == "BUY"
        assert result.health_adj == 72
        assert result.risk_tally == 0.8
        assert result.zone == "LOW"


class TestVerdictFallbackExtraction:
    """Tests for fallback verdict extraction from text."""

    def test_extract_verdict_from_header(self):
        """Test extraction from PORTFOLIO MANAGER VERDICT header."""
        pm_output = "### PORTFOLIO MANAGER VERDICT: BUY\n\nSome rationale..."
        verdict = extract_verdict_from_text(pm_output)
        assert verdict == "BUY"

    def test_extract_do_not_initiate_from_header(self):
        """Test extraction of DO NOT INITIATE from header."""
        pm_output = "### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE\n\nFails thesis..."
        verdict = extract_verdict_from_text(pm_output)
        assert verdict == "DO_NOT_INITIATE"

    def test_extract_from_action_field(self):
        """Test extraction from **Action**: field."""
        pm_output = "Some text\n\n**Action**: SELL\n\nMore text"
        verdict = extract_verdict_from_text(pm_output)
        assert verdict == "SELL"

    def test_no_verdict_found(self):
        """Test returns None when no verdict pattern found."""
        pm_output = "This is just some random text without a verdict."
        verdict = extract_verdict_from_text(pm_output)
        assert verdict is None


class TestChartSuppression:
    """Tests for conditional chart suppression based on verdict."""

    def test_should_show_targets_for_buy(self):
        """BUY verdict should show valuation targets."""
        data = PMBlockData(
            verdict="BUY",
            show_valuation_chart=True,
        )
        assert data.should_show_targets() is True

    def test_should_show_targets_for_hold(self):
        """HOLD verdict should show valuation targets."""
        data = PMBlockData(
            verdict="HOLD",
            show_valuation_chart=True,
        )
        assert data.should_show_targets() is True

    def test_should_not_show_targets_for_do_not_initiate(self):
        """DO_NOT_INITIATE verdict should not show valuation targets."""
        data = PMBlockData(
            verdict="DO_NOT_INITIATE",
            show_valuation_chart=False,
        )
        assert data.should_show_targets() is False

    def test_should_not_show_targets_for_sell(self):
        """SELL verdict should not show valuation targets."""
        data = PMBlockData(
            verdict="SELL",
            show_valuation_chart=False,
        )
        assert data.should_show_targets() is False

    def test_explicit_no_chart_overrides_verdict(self):
        """Explicit SHOW_VALUATION_CHART: NO overrides even BUY verdict."""
        data = PMBlockData(
            verdict="BUY",
            show_valuation_chart=False,
        )
        assert data.should_show_targets() is False


class TestValuationDiscount:
    """Tests for valuation discount based on risk zone."""

    def test_low_risk_full_targets(self):
        """LOW risk zone should have no discount (1.0)."""
        pm_output = """
### --- START PM_BLOCK ---
VERDICT: BUY
ZONE: LOW
VALUATION_DISCOUNT: 1.0
### --- END PM_BLOCK ---
"""
        result = extract_pm_block(pm_output)
        assert result.valuation_discount == 1.0

    def test_moderate_risk_discount(self):
        """MODERATE risk zone should have 0.9 discount."""
        pm_output = """
### --- START PM_BLOCK ---
VERDICT: HOLD
ZONE: MODERATE
VALUATION_DISCOUNT: 0.9
### --- END PM_BLOCK ---
"""
        result = extract_pm_block(pm_output)
        assert result.valuation_discount == 0.9

    def test_high_risk_discount(self):
        """HIGH risk zone should have 0.8 discount."""
        pm_output = """
### --- START PM_BLOCK ---
VERDICT: HOLD
ZONE: HIGH
VALUATION_DISCOUNT: 0.8
### --- END PM_BLOCK ---
"""
        result = extract_pm_block(pm_output)
        assert result.valuation_discount == 0.8

    def test_default_discount_based_on_zone(self):
        """Test default discount calculation when not explicitly provided."""
        pm_output = """
### --- START PM_BLOCK ---
VERDICT: HOLD
ZONE: MODERATE
### --- END PM_BLOCK ---
"""
        result = extract_pm_block(pm_output)
        # Should default to 0.9 for MODERATE zone
        assert result.valuation_discount == 0.9

    def test_negative_verdict_forces_zero_discount(self):
        """DO_NOT_INITIATE should force discount to 0.0."""
        pm_output = """
### --- START PM_BLOCK ---
VERDICT: DO_NOT_INITIATE
ZONE: HIGH
VALUATION_DISCOUNT: 0.8
### --- END PM_BLOCK ---
"""
        result = extract_pm_block(pm_output)
        # Should be forced to 0.0 regardless of what's specified
        assert result.valuation_discount == 0.0


class TestPMBlockPromptCompliance:
    """Tests to verify PM_BLOCK is in the Portfolio Manager prompt."""

    def test_pm_prompt_contains_pm_block_instruction(self):
        """Verify PM prompt contains PM_BLOCK output instruction."""
        import json

        with open("prompts/portfolio_manager.json") as f:
            prompt = json.load(f)

        message = prompt["system_message"]
        assert "PM_BLOCK" in message
        assert "### --- START PM_BLOCK ---" in message
        assert "### --- END PM_BLOCK ---" in message
        assert "VERDICT:" in message
        assert "VALUATION_DISCOUNT:" in message

    def test_pm_prompt_version_updated(self):
        """Verify PM prompt version is updated for PM_BLOCK."""
        import json

        with open("prompts/portfolio_manager.json") as f:
            prompt = json.load(f)

        # Should be at least 7.7 (the version that added PM_BLOCK)
        version = float(prompt["version"])
        assert version >= 7.7


class TestChartNodeIntegration:
    """Tests for chart generator node integration."""

    def test_chart_node_import(self):
        """Test that chart node can be imported."""
        from src.charts.chart_node import create_chart_generator_node

        node = create_chart_generator_node()
        assert callable(node)

    def test_chart_node_with_skip(self):
        """Test chart node with skip_charts=True returns empty paths."""
        import asyncio

        from src.charts.chart_node import create_chart_generator_node

        node = create_chart_generator_node(skip_charts=True)

        state = {
            "company_of_interest": "TEST",
            "final_trade_decision": "### PORTFOLIO MANAGER VERDICT: BUY",
            "fundamentals_report": "",
            "valuation_params": "",
            "red_flags": [],
        }

        result = asyncio.get_event_loop().run_until_complete(node(state, {}))
        assert result == {"chart_paths": {}}

    def test_chart_node_with_quick_mode(self):
        """Test chart node with quick_mode=True returns empty paths."""
        import asyncio

        from src.charts.chart_node import create_chart_generator_node

        node = create_chart_generator_node(quick_mode=True)

        state = {
            "company_of_interest": "TEST",
            "final_trade_decision": "",
            "fundamentals_report": "",
            "valuation_params": "",
            "red_flags": [],
        }

        result = asyncio.get_event_loop().run_until_complete(node(state, {}))
        assert result == {"chart_paths": {}}
