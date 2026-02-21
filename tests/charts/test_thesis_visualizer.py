"""Tests for thesis compliance visualizer."""

import pytest

from src.thesis_visualizer import (
    AnsiColors,
    ThesisMetrics,
    ThesisVisualizer,
    generate_thesis_visual,
)

# Sample Portfolio Manager output for testing
SAMPLE_PM_OUTPUT = """
### FINAL DECISION: BUY

### THESIS COMPLIANCE SUMMARY

**Hard Fail Checks:**
- **Financial Health**: 71% (Adjusted) - **PASS**
- **Growth Transition**: 100% (Adjusted) - **PASS**
- **Liquidity**: $86.9M Daily Avg - **PASS**
- **Analyst Coverage**: 8 - **PASS**
- **US Revenue**: ~15-20% - **PASS**
- **P/E Ratio**: 11.80 (PEG: 0.16) - **PASS**

**Hard Fail Result**: **PASS**

**Qualitative Risk Tally:**
- **ADR (MODERATE_CONCERN)**: **+0.33**
- **Qualitative Risks**: **+1.0**
- **TOTAL RISK COUNT**: **1.33**

**Action**: **BUY**
"""

SAMPLE_SELL_OUTPUT = """
### FINAL DECISION: SELL

**Hard Fail Checks:**
- **Financial Health**: 35% (Adjusted) - **FAIL**
- **Growth Transition**: 40% (Adjusted) - **FAIL**
- **Liquidity**: $50K Daily Avg - **FAIL**
- **Analyst Coverage**: 25 - **FAIL**
- **US Revenue**: 45% - **FAIL**
- **P/E Ratio**: 28.5 (PEG: 2.1) - **FAIL**

**TOTAL RISK COUNT**: **3.5**

**Action**: **SELL**
"""

MINIMAL_OUTPUT = """
### FINAL DECISION: HOLD
Financial Health: 55%
P/E Ratio: 15.2
"""


class TestThesisMetricsExtraction:
    """Test metric extraction from Portfolio Manager output."""

    def test_extract_health_score(self):
        """Test Financial Health score extraction."""
        vis = ThesisVisualizer(SAMPLE_PM_OUTPUT)
        assert vis.metrics.health_score == 71.0

    def test_extract_growth_score(self):
        """Test Growth Transition score extraction."""
        vis = ThesisVisualizer(SAMPLE_PM_OUTPUT)
        assert vis.metrics.growth_score == 100.0

    def test_extract_pe_ratio(self):
        """Test P/E Ratio extraction."""
        vis = ThesisVisualizer(SAMPLE_PM_OUTPUT)
        assert vis.metrics.pe_ratio == 11.80

    def test_extract_peg_ratio(self):
        """Test PEG Ratio extraction."""
        vis = ThesisVisualizer(SAMPLE_PM_OUTPUT)
        assert vis.metrics.peg_ratio == 0.16

    def test_extract_liquidity(self):
        """Test Liquidity extraction."""
        vis = ThesisVisualizer(SAMPLE_PM_OUTPUT)
        assert vis.metrics.liquidity_pass is True
        assert vis.metrics.liquidity_value == "$86.9M"

    def test_extract_analyst_coverage(self):
        """Test Analyst Coverage extraction."""
        vis = ThesisVisualizer(SAMPLE_PM_OUTPUT)
        assert vis.metrics.analyst_coverage == 8
        assert vis.metrics.analyst_pass is True

    def test_extract_us_revenue(self):
        """Test US Revenue extraction."""
        vis = ThesisVisualizer(SAMPLE_PM_OUTPUT)
        assert vis.metrics.us_revenue == "~15-20%"
        assert vis.metrics.us_revenue_pass is True

    def test_extract_total_risk(self):
        """Test Total Risk Count extraction."""
        vis = ThesisVisualizer(SAMPLE_PM_OUTPUT)
        assert vis.metrics.total_risk == 1.33

    def test_extract_decision(self):
        """Test Decision extraction."""
        vis = ThesisVisualizer(SAMPLE_PM_OUTPUT)
        assert vis.metrics.decision == "BUY"

    def test_extract_sell_decision(self):
        """Test SELL decision extraction with failing metrics."""
        vis = ThesisVisualizer(SAMPLE_SELL_OUTPUT)
        assert vis.metrics.decision == "SELL"
        assert vis.metrics.health_score == 35.0
        assert vis.metrics.pe_ratio == 28.5
        assert vis.metrics.total_risk == 3.5


class TestBarChartGeneration:
    """Test bar chart string generation."""

    def test_bar_full(self):
        """Test fully filled bar."""
        vis = ThesisVisualizer("")
        bar = vis._bar(100.0, 100.0)
        assert bar == "▓" * 20

    def test_bar_empty(self):
        """Test empty bar."""
        vis = ThesisVisualizer("")
        bar = vis._bar(0.0, 100.0)
        assert bar == "░" * 20

    def test_bar_half(self):
        """Test half-filled bar."""
        vis = ThesisVisualizer("")
        bar = vis._bar(50.0, 100.0)
        assert bar == "▓" * 10 + "░" * 10

    def test_bar_over_max(self):
        """Test value over max is capped."""
        vis = ThesisVisualizer("")
        bar = vis._bar(150.0, 100.0)
        assert bar == "▓" * 20  # Capped at 100%

    def test_bar_zero_max(self):
        """Test zero max value returns empty bar."""
        vis = ThesisVisualizer("")
        bar = vis._bar(50.0, 0.0)
        assert bar == "░" * 20


class TestCheckmarkGeneration:
    """Test pass/fail checkmark generation."""

    def test_check_pass(self):
        """Test passing check."""
        vis = ThesisVisualizer("")
        assert vis._check(True) == "✓"

    def test_check_fail(self):
        """Test failing check."""
        vis = ThesisVisualizer("")
        assert vis._check(False) == "✗"

    def test_check_none(self):
        """Test unknown check."""
        vis = ThesisVisualizer("")
        assert vis._check(None) == "?"


class TestRiskZone:
    """Test risk zone determination."""

    def test_zone_low(self):
        """Test low risk zone."""
        vis = ThesisVisualizer("")
        assert vis._zone(0.5) == "LOW"
        assert vis._zone(0.99) == "LOW"

    def test_zone_moderate(self):
        """Test moderate risk zone."""
        vis = ThesisVisualizer("")
        assert vis._zone(1.0) == "MODERATE"
        assert vis._zone(1.5) == "MODERATE"
        assert vis._zone(1.99) == "MODERATE"

    def test_zone_high(self):
        """Test high risk zone."""
        vis = ThesisVisualizer("")
        assert vis._zone(2.0) == "HIGH"
        assert vis._zone(3.5) == "HIGH"


class TestVisualizationOutput:
    """Test complete visualization output."""

    def test_generate_buy_visual(self):
        """Test complete visualization for BUY decision."""
        visual = generate_thesis_visual(SAMPLE_PM_OUTPUT)

        # Should contain key sections
        assert "THESIS COMPLIANCE VISUAL" in visual
        assert "CORE SCORES" in visual
        assert "VALUATION" in visual
        assert "HARD FAIL CHECKS" in visual
        assert "RISK TALLY" in visual
        assert "DECISION: BUY" in visual

        # Should contain metric values
        assert "71.0%" in visual  # Health score
        assert "100.0%" in visual  # Growth score
        assert "11.8" in visual  # P/E
        assert "0.16" in visual  # PEG
        assert "1.33" in visual  # Risk score

        # Should contain checkmarks for passing metrics
        assert "✓" in visual

    def test_generate_sell_visual(self):
        """Test complete visualization for SELL decision."""
        visual = generate_thesis_visual(SAMPLE_SELL_OUTPUT)

        assert "DECISION: SELL" in visual
        assert "35.0%" in visual  # Failing health score
        assert "28.5" in visual  # Failing P/E
        assert "HIGH" in visual  # High risk zone

        # Should contain X marks for failing metrics
        assert "✗" in visual

    def test_generate_minimal_visual(self):
        """Test visualization with minimal data."""
        visual = generate_thesis_visual(MINIMAL_OUTPUT)

        # Should still generate output with available data
        assert "THESIS COMPLIANCE VISUAL" in visual
        assert "55.0%" in visual
        assert "15.2" in visual
        assert "DECISION: HOLD" in visual

    def test_generate_empty_returns_empty(self):
        """Test empty input returns empty string."""
        visual = generate_thesis_visual("")
        assert visual == ""

    def test_generate_none_returns_empty(self):
        """Test None-like input is handled."""
        visual = generate_thesis_visual("No thesis data here at all.")
        assert visual == ""  # No parseable metrics


class TestColorSupport:
    """Test colorblind-safe ANSI color support."""

    def test_ansi_colors_defined(self):
        """Test that AnsiColors has all required color codes."""
        c = AnsiColors()
        assert c.PASS.startswith("\033[")
        assert c.FAIL.startswith("\033[")
        assert c.WARN.startswith("\033[")
        assert c.BUY.startswith("\033[")
        assert c.HOLD.startswith("\033[")
        assert c.SELL.startswith("\033[")
        assert c.RESET == "\033[0m"

    def test_disabled_colors_returns_empty_strings(self):
        """Test that disabled colors returns empty strings."""
        no_colors = AnsiColors.disabled()
        assert no_colors.PASS == ""
        assert no_colors.FAIL == ""
        assert no_colors.RESET == ""

    def test_generate_with_color_includes_ansi_codes(self):
        """Test that use_color=True includes ANSI escape codes."""
        visual = generate_thesis_visual(SAMPLE_PM_OUTPUT, use_color=True)

        # Should contain ANSI escape codes
        assert "\033[" in visual
        # Should contain reset code
        assert "\033[0m" in visual

    def test_generate_without_color_no_ansi_codes(self):
        """Test that use_color=False (default) has no ANSI codes."""
        visual = generate_thesis_visual(SAMPLE_PM_OUTPUT, use_color=False)

        # Should NOT contain ANSI escape codes
        assert "\033[" not in visual

    def test_color_mode_no_markdown_fences(self):
        """Test that color mode doesn't include markdown code fences."""
        visual = generate_thesis_visual(SAMPLE_PM_OUTPUT, use_color=True)

        # Should NOT have markdown code fences (they'd break terminal display)
        assert not visual.startswith("```")
        assert not visual.endswith("```")

    def test_no_color_mode_has_markdown_fences(self):
        """Test that no-color mode includes markdown code fences."""
        visual = generate_thesis_visual(SAMPLE_PM_OUTPUT, use_color=False)

        # Should have markdown code fences
        assert visual.startswith("```")
        assert visual.endswith("```")

    def test_colorblind_safe_palette(self):
        """Test that colors use colorblind-safe palette (no red/green)."""
        c = AnsiColors()

        # Extract color codes (ANSI format: \033[XXm where XX is color code)
        # Red is 31, Green is 32 - these should NOT be used
        red_code = "\033[31m"
        green_code = "\033[32m"

        # None of our main colors should be pure red or green
        assert c.PASS != red_code and c.PASS != green_code
        assert c.FAIL != red_code and c.FAIL != green_code
        assert c.WARN != red_code and c.WARN != green_code
        assert c.BUY != red_code and c.BUY != green_code
        assert c.SELL != red_code and c.SELL != green_code

    def test_decision_color_buy(self):
        """Test BUY decision gets correct color."""
        visual = generate_thesis_visual(SAMPLE_PM_OUTPUT, use_color=True)
        c = AnsiColors()

        # BUY decision should be colored with BUY color
        assert f"{c.BUY}BUY{c.RESET}" in visual or c.BUY in visual

    def test_decision_color_sell(self):
        """Test SELL decision gets correct color."""
        visual = generate_thesis_visual(SAMPLE_SELL_OUTPUT, use_color=True)
        c = AnsiColors()

        # SELL decision should be colored with SELL color
        assert f"{c.SELL}SELL{c.RESET}" in visual or c.SELL in visual


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_malformed_percentages(self):
        """Test handling of unusual percentage formats."""
        text = "Financial Health: 75.5% - PASS"
        vis = ThesisVisualizer(text)
        assert vis.metrics.health_score == 75.5

    def test_us_revenue_not_disclosed(self):
        """Test US Revenue 'Not disclosed' handling."""
        text = "US Revenue: Not disclosed - N/A"
        vis = ThesisVisualizer(text)
        assert vis.metrics.us_revenue == "Not disclosed"
        assert vis.metrics.us_revenue_pass is True  # N/A is treated as pass

    def test_liquidity_billion(self):
        """Test Liquidity in billions."""
        text = "Liquidity: $1.2B Daily Avg - PASS"
        vis = ThesisVisualizer(text)
        assert vis.metrics.liquidity_value == "$1.2B"
        assert vis.metrics.liquidity_pass is True

    def test_liquidity_thousand(self):
        """Test Liquidity in thousands."""
        text = "Liquidity: $500K Daily Avg - MARGINAL"
        vis = ThesisVisualizer(text)
        assert vis.metrics.liquidity_value == "$500K"
        assert vis.metrics.liquidity_pass is True  # MARGINAL is still a pass

    def test_markdown_formatting_preserved(self):
        """Test output is properly formatted for markdown."""
        visual = generate_thesis_visual(SAMPLE_PM_OUTPUT)

        # Should be wrapped in code blocks
        assert visual.startswith("```")
        assert visual.endswith("```")

    def test_bar_width_consistent(self):
        """Test all bars have consistent width."""
        visual = generate_thesis_visual(SAMPLE_PM_OUTPUT)
        lines = visual.split("\n")

        bar_lines = [line for line in lines if "▓" in line or "░" in line]
        for line in bar_lines:
            # Count bar characters
            bar_chars = sum(1 for c in line if c in "▓░")
            assert bar_chars == 20, f"Inconsistent bar width in: {line}"
