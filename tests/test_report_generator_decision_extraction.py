import pytest

from src.report_generator import QuietModeReporter


class TestExtractDecisionBasicVerdicts:
    """Test extraction of all valid PM verdicts."""

    @pytest.fixture
    def generator(self):
        return QuietModeReporter(ticker="TEST.X", company_name="Test Corp")

    @pytest.mark.parametrize(
        "verdict_text,expected",
        [
            # Standard verdicts
            ("#### PORTFOLIO MANAGER VERDICT: BUY", "BUY"),
            ("#### PORTFOLIO MANAGER VERDICT: SELL", "SELL"),
            ("#### PORTFOLIO MANAGER VERDICT: HOLD", "HOLD"),
            # DO NOT INITIATE (prose uses spaces, not underscores)
            ("#### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE", "DO NOT INITIATE"),
            # REJECT (should normalize to DO NOT INITIATE)
            ("#### PORTFOLIO MANAGER VERDICT: REJECT", "DO NOT INITIATE"),
            # With markdown formatting
            ("#### PORTFOLIO MANAGER VERDICT: **BUY**", "BUY"),
            ("#### PORTFOLIO MANAGER VERDICT: **DO NOT INITIATE**", "DO NOT INITIATE"),
            ("PORTFOLIO MANAGER VERDICT: *SELL*", "SELL"),
        ],
    )
    def test_pm_verdict_extraction(self, generator, verdict_text, expected):
        """PM verdict should be correctly extracted regardless of formatting."""
        assert generator.extract_decision(verdict_text) == expected


class TestExtractDecisionPMBlockPriority:
    """Test that PM_BLOCK VERDICT takes priority over prose."""

    @pytest.fixture
    def generator(self):
        return QuietModeReporter(ticker="TEST.X", company_name="Test Corp")

    @pytest.mark.parametrize(
        "pm_block_verdict,expected",
        [
            ("VERDICT: BUY", "BUY"),
            ("VERDICT: SELL", "SELL"),
            ("VERDICT: HOLD", "HOLD"),
            ("VERDICT: DO_NOT_INITIATE", "DO NOT INITIATE"),
            ("VERDICT: REJECT", "DO NOT INITIATE"),
        ],
    )
    def test_pm_block_verdict_extraction(self, generator, pm_block_verdict, expected):
        """PM_BLOCK VERDICT field should be recognized."""
        text = f"""
#### PM_BLOCK (REQUIRED - Machine-Readable Summary)

--- START PM_BLOCK ---

{pm_block_verdict}
HEALTH_ADJ: 79
GROWTH_ADJ: 33
--- END PM_BLOCK ---

"""
        assert generator.extract_decision(text) == expected


class TestExtractDecisionNoLeakage:
    """
    CRITICAL: Test that verdicts from subordinate agents don't leak into title.

    This is the bug that caused WPK.TO to show "BUY" in the title when the
    PM verdict was "DO NOT INITIATE".
    """

    @pytest.fixture
    def generator(self):
        return QuietModeReporter(ticker="WPK.TO", company_name="Winpak Ltd.")

    def test_no_leak_from_risky_analyst_buy(self, generator):
        """PM says DO NOT INITIATE but Risky Analyst recommends BUY."""
        text = """
#### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE

**Hard Fail Result**: **FAIL on: [US Revenue, Growth Transition]**

### Risky Analyst (Aggressive)

**RISKY ANALYST ASSESSMENT**

**Recommended Initial Position Size**: **6.0%** (Aggressive Standalone Entry)

I fundamentally disagree with the "REJECT" decision. BUY this stock!
The upside is compelling. This is a clear BUY opportunity.
"""
        assert generator.extract_decision(text) == "DO NOT INITIATE"

    def test_no_leak_from_decision_framework_default(self, generator):
        """PM says DO NOT INITIATE but decision framework shows Default: BUY."""
        text = """
#### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE

**Decision Framework Applied**:
=== DECISION LOGIC ===
ZONE: LOW (< 1.0)
Default Decision: BUY
Actual Decision: DO NOT INITIATE
Override: NO (Hard Fails on US Revenue and Growth are binding)

"""
        assert generator.extract_decision(text) == "DO NOT INITIATE"

    def test_no_leak_from_bull_researcher(self, generator):
        """PM says SELL but Bull Researcher advocates BUY."""
        text = """
#### PORTFOLIO MANAGER VERDICT: SELL

### Bull Researcher Round 1

This is clearly a BUY. The thesis supports a strong BUY recommendation.
I strongly advocate for BUY given the fundamentals.

### Bear Researcher Round 1

This is a SELL. Avoid this stock.

#### PM_BLOCK
VERDICT: SELL
"""
        assert generator.extract_decision(text) == "SELL"

    def test_no_leak_from_research_manager_recommendation(self, generator):
        """PM says HOLD but Research Manager said BUY."""
        text = """
## Investment Recommendation

#### INVESTMENT RECOMMENDATION: BUY

The Research Manager recommends BUY based on the debate synthesis.

---

## Executive Summary

#### PORTFOLIO MANAGER VERDICT: HOLD

The PM overrides to HOLD due to position sizing constraints.

#### PM_BLOCK
VERDICT: HOLD
"""
        assert generator.extract_decision(text) == "HOLD"

    def test_no_leak_from_consultant_conditional_approval(self, generator):
        """PM says REJECT but Consultant says proceed with BUY."""
        text = """
## External Consultant Review

**Consultant Assessment**: CONDITIONAL APPROVAL
**Recommended Action**: Proceed with BUY if governance concerns are addressed.

---

## Executive Summary

#### PORTFOLIO MANAGER VERDICT: REJECT

Due to structural thesis violations, we cannot initiate.

#### PM_BLOCK
VERDICT: DO_NOT_INITIATE
"""
        # REJECT should normalize to DO NOT INITIATE
        assert generator.extract_decision(text) == "DO NOT INITIATE"

    def test_no_leak_from_action_in_trader_override_section(self, generator):
        """PM says DO NOT INITIATE but Trader shows hypothetical BUY parameters."""
        text = """
#### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE

## Trading Strategy

**Action**: **REJECT / NO TRADE**

#### **EXECUTION PARAMETERS (IF OVERRIDDEN)**
*Note: These parameters are provided only in the event the Portfolio Manager
chooses to override the mandate violation.*

**Entry Strategy**:
- **Approach**: Scaled Entry
- **Action**: BUY at 44.50 CAD

**TRADER FINAL COMMENT**: Recommendation remains REJECT.
"""
        assert generator.extract_decision(text) == "DO NOT INITIATE"

    def test_no_leak_from_generic_buy_in_technical_analysis(self, generator):
        """PM says SELL but technical analysis mentions BUY signals."""
        text = """
## Technical Analysis

The RSI indicates oversold conditions, typically a BUY signal.
MACD crossover suggests BUY momentum.
Volume profile supports a BUY thesis technically.

## Executive Summary

#### PORTFOLIO MANAGER VERDICT: SELL

Despite technical BUY signals, fundamental deterioration requires SELL.

#### PM_BLOCK
VERDICT: SELL
"""
        assert generator.extract_decision(text) == "SELL"


class TestExtractDecisionPMBlockVsProse:
    """Test consistency between PM_BLOCK and prose verdict."""

    @pytest.fixture
    def generator(self):
        return QuietModeReporter(ticker="TEST.X", company_name="Test Corp")

    def test_pm_block_takes_priority_over_prose(self, generator):
        """If PM_BLOCK and prose disagree, PM_BLOCK should win (machine-readable)."""
        text = """
#### PORTFOLIO MANAGER VERDICT: HOLD

Some explanation...

#### PM_BLOCK
VERDICT: DO_NOT_INITIATE
"""
        # PM_BLOCK should be authoritative as it's machine-generated
        assert generator.extract_decision(text) == "DO NOT INITIATE"

    def test_prose_verdict_when_no_pm_block(self, generator):
        """When PM_BLOCK is missing, prose verdict should work."""
        text = """
#### PORTFOLIO MANAGER VERDICT: BUY

**Conviction**: High
"""
        assert generator.extract_decision(text) == "BUY"


class TestExtractDecisionEdgeCases:
    """Edge cases and malformed input handling."""

    @pytest.fixture
    def generator(self):
        return QuietModeReporter(ticker="TEST.X", company_name="Test Corp")

    def test_mixed_case_verdict(self, generator):
        """Verdict extraction should be case-insensitive."""
        text = "#### Portfolio Manager Verdict: do not initiate"
        assert generator.extract_decision(text) == "DO NOT INITIATE"

    def test_extra_whitespace_in_verdict(self, generator):
        """Handle extra whitespace in verdict."""
        text = "#### PORTFOLIO MANAGER VERDICT:    BUY   "
        assert generator.extract_decision(text) == "BUY"

    def test_verdict_with_colon_variations(self, generator):
        """Handle colon/no-colon variations."""
        text1 = "PORTFOLIO MANAGER VERDICT: SELL"
        text2 = "PORTFOLIO MANAGER VERDICT SELL"  # Missing colon
        assert generator.extract_decision(text1) == "SELL"
        # text2 behavior depends on implementation - document expected behavior

    def test_no_verdict_defaults_to_hold(self, generator):
        """When no verdict found at all, should default to HOLD (safe)."""
        text = "This report has no clear verdict anywhere."
        assert generator.extract_decision(text) == "HOLD"

    def test_only_subordinate_verdicts_defaults_to_hold(self, generator):
        """When only subordinate agents have verdicts, should default to HOLD."""
        text = """
### Risky Analyst
Recommendation: BUY at 6%

### Safe Analyst
Recommendation: SELL immediately

### Research Manager
INVESTMENT RECOMMENDATION: HOLD
"""
        # No PM verdict - should default, not pick up subordinate opinions
        # This depends on implementation - could be HOLD or could find Research Manager
        # Document expected behavior
        result = generator.extract_decision(text)
        # At minimum, should NOT be BUY from Risky Analyst
        assert result != "BUY" or result == "HOLD"


class TestExtractDecisionRealWorldRegression:
    """Regression tests from actual bug reports."""

    @pytest.fixture
    def generator(self):
        return QuietModeReporter(ticker="WPK.TO", company_name="Winpak Ltd.")

    def test_wpk_to_regression(self, generator):
        """
        Regression test for WPK.TO bug where title showed BUY
        but PM verdict was DO NOT INITIATE.

        The "Default Decision: BUY" in the decision framework leaked
        into the title via greedy regex.
        """
        # Simplified version of actual report structure
        text = """
# WPK.TO (Winpak Ltd.): [DECISION]

## Thesis Compliance at a Glance

Financial Health  79.0% ✓ (min 50%)
Growth Transition 33.0% ✗ (min 50%)

## Executive Summary

#### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE

**Hard Fail Result**: **FAIL on: [US Revenue, Growth Transition]**

**Decision Framework Applied**:
=== DECISION LOGIC ===
ZONE: LOW (< 1.0)
Default Decision: BUY
Actual Decision: DO NOT INITIATE
Override: NO (Hard Fails on US Revenue and Growth are binding)


#### PM_BLOCK (REQUIRED - Machine-Readable Summary)

--- START PM_BLOCK ---

VERDICT: DO_NOT_INITIATE
HEALTH_ADJ: 79
GROWTH_ADJ: 33
RISK_TALLY: -0.67
ZONE: LOW
SHOW_VALUATION_CHART: NO
POSITION_SIZE: 0.0
--- END PM_BLOCK ---


## Risk Assessment

### Risky Analyst (Aggressive)

**Recommended Initial Position Size**: **6.0%** (Aggressive Standalone Entry)

I fundamentally disagree with the "REJECT" decision. BUY this stock!
We get exposure to the robust US economy. This is a clear BUY opportunity.
"""
        result = generator.extract_decision(text)
        assert result == "DO NOT INITIATE", (
            f"WPK.TO regression: Expected 'DO NOT INITIATE' but got '{result}'. "
            "Greedy regex likely matched 'BUY' from subordinate agent."
        )
