"""
Tests for extract_decision() to prevent verdict leakage from subordinate agents.

The critical bug (WPK.TO): When PM said "DO NOT INITIATE", a greedy regex
matched "BUY" from the decision framework's "Default Decision: BUY" or from
the Risky Analyst's recommendation, causing the report title to be wrong.
"""

import pytest

from src.report_generator import QuietModeReporter


class TestExtractDecisionPMBlockPriority:
    """PM_BLOCK VERDICT is machine-readable and should be highest priority."""

    @pytest.fixture
    def generator(self):
        return QuietModeReporter(ticker="TEST.X", company_name="Test Corp")

    @pytest.mark.parametrize(
        "verdict,expected",
        [
            ("VERDICT: BUY", "BUY"),
            ("VERDICT: SELL", "SELL"),
            ("VERDICT: HOLD", "HOLD"),
            ("VERDICT: DO_NOT_INITIATE", "DO NOT INITIATE"),
            ("VERDICT: DO NOT INITIATE", "DO NOT INITIATE"),
            ("VERDICT: REJECT", "DO NOT INITIATE"),
        ],
    )
    def test_pm_block_verdict_extraction(self, generator, verdict, expected):
        """PM_BLOCK VERDICT field should be correctly extracted."""
        text = f"""
#### --- START PM_BLOCK ---
{verdict}
HEALTH_ADJ: 79
#### --- END PM_BLOCK ---
"""
        assert generator.extract_decision(text) == expected


class TestExtractDecisionProseVerdict:
    """PORTFOLIO MANAGER VERDICT in prose should work when PM_BLOCK missing."""

    @pytest.fixture
    def generator(self):
        return QuietModeReporter(ticker="TEST.X", company_name="Test Corp")

    @pytest.mark.parametrize(
        "verdict_text,expected",
        [
            ("#### PORTFOLIO MANAGER VERDICT: BUY", "BUY"),
            ("#### PORTFOLIO MANAGER VERDICT: SELL", "SELL"),
            ("#### PORTFOLIO MANAGER VERDICT: HOLD", "HOLD"),
            ("#### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE", "DO NOT INITIATE"),
            ("#### PORTFOLIO MANAGER VERDICT: REJECT", "DO NOT INITIATE"),
            # With markdown formatting
            ("#### PORTFOLIO MANAGER VERDICT: **BUY**", "BUY"),
            ("PORTFOLIO MANAGER VERDICT: **DO NOT INITIATE**", "DO NOT INITIATE"),
        ],
    )
    def test_pm_prose_verdict_extraction(self, generator, verdict_text, expected):
        """PM prose verdict should be correctly extracted."""
        assert generator.extract_decision(verdict_text) == expected


class TestExtractDecisionNoLeakage:
    """
    CRITICAL: Verdicts from subordinate agents must NOT leak into title.

    This is the bug that caused WPK.TO to show "BUY" when PM said "DO NOT INITIATE".
    """

    @pytest.fixture
    def generator(self):
        return QuietModeReporter(ticker="WPK.TO", company_name="Winpak Ltd.")

    def test_no_leak_from_decision_framework_default(self, generator):
        """PM says DO NOT INITIATE but decision framework shows Default: BUY."""
        text = """
#### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE

**Decision Framework Applied**:
```
=== DECISION LOGIC ===
ZONE: LOW (< 1.0)
Default Decision: BUY
Actual Decision: DO NOT INITIATE
Override: NO (Hard Fails binding)
======================
```
"""
        assert generator.extract_decision(text) == "DO NOT INITIATE"

    def test_no_leak_from_risky_analyst(self, generator):
        """PM says DO NOT INITIATE but Risky Analyst recommends BUY."""
        text = """
#### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE

### Risky Analyst (Aggressive)

**Recommended Initial Position Size**: **6.0%**

I disagree with the REJECT decision. BUY this stock!
This is a clear BUY opportunity. BUY BUY BUY!
"""
        assert generator.extract_decision(text) == "DO NOT INITIATE"

    def test_no_leak_from_bull_researcher(self, generator):
        """PM says SELL but Bull Researcher advocates BUY."""
        text = """
#### PORTFOLIO MANAGER VERDICT: SELL

### Bull Researcher Round 1

This is clearly a BUY. Strong BUY recommendation.

### Bear Researcher Round 1

This is a SELL.
"""
        assert generator.extract_decision(text) == "SELL"

    def test_no_leak_from_research_manager(self, generator):
        """PM says HOLD but Research Manager said BUY."""
        text = """
## Investment Recommendation

#### INVESTMENT RECOMMENDATION: BUY

---

## Executive Summary

#### PORTFOLIO MANAGER VERDICT: HOLD
"""
        assert generator.extract_decision(text) == "HOLD"

    def test_no_leak_from_trader_override_section(self, generator):
        """PM says DO NOT INITIATE but Trader shows hypothetical BUY params."""
        text = """
#### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE

## Trading Strategy

**Action**: **REJECT / NO TRADE**

#### **EXECUTION PARAMETERS (IF OVERRIDDEN)**
**Action**: BUY at 44.50 CAD
**Entry**: 44.50
"""
        assert generator.extract_decision(text) == "DO NOT INITIATE"

    def test_no_leak_from_technical_analysis(self, generator):
        """PM says SELL but technical analysis mentions BUY signals."""
        text = """
## Technical Analysis

RSI indicates oversold - typically a BUY signal.
MACD suggests BUY momentum.

## Executive Summary

#### PORTFOLIO MANAGER VERDICT: SELL
"""
        assert generator.extract_decision(text) == "SELL"

    def test_no_leak_from_consultant(self, generator):
        """PM says REJECT but Consultant says proceed with BUY."""
        text = """
## External Consultant Review

**Recommended Action**: Proceed with BUY if concerns addressed.

## Executive Summary

#### PORTFOLIO MANAGER VERDICT: REJECT
"""
        assert generator.extract_decision(text) == "DO NOT INITIATE"


class TestExtractDecisionPMBlockVsProse:
    """Test priority when both PM_BLOCK and prose exist."""

    @pytest.fixture
    def generator(self):
        return QuietModeReporter(ticker="TEST.X", company_name="Test Corp")

    def test_pm_block_wins_over_prose(self, generator):
        """PM_BLOCK is authoritative when both exist."""
        text = """
#### PORTFOLIO MANAGER VERDICT: HOLD

#### --- START PM_BLOCK ---
VERDICT: DO_NOT_INITIATE
#### --- END PM_BLOCK ---
"""
        # PM_BLOCK appears later but should be checked first (highest priority)
        assert generator.extract_decision(text) == "DO NOT INITIATE"

    def test_prose_works_when_no_pm_block(self, generator):
        """Prose verdict works when PM_BLOCK is missing."""
        text = """
#### PORTFOLIO MANAGER VERDICT: BUY

**Conviction**: High
"""
        assert generator.extract_decision(text) == "BUY"


class TestExtractDecisionEdgeCases:
    """Edge cases and malformed input."""

    @pytest.fixture
    def generator(self):
        return QuietModeReporter(ticker="TEST.X", company_name="Test Corp")

    def test_mixed_case_verdict(self, generator):
        """Verdict extraction should be case-insensitive."""
        text = "#### Portfolio Manager Verdict: do not initiate"
        assert generator.extract_decision(text) == "DO NOT INITIATE"

    def test_extra_whitespace(self, generator):
        """Handle extra whitespace in verdict."""
        text = "#### PORTFOLIO MANAGER VERDICT:    BUY   "
        assert generator.extract_decision(text) == "BUY"

    def test_no_verdict_defaults_to_hold(self, generator):
        """When no PM verdict found, default to HOLD (safe)."""
        text = "This report has no clear verdict anywhere."
        assert generator.extract_decision(text) == "HOLD"

    def test_only_subordinate_verdicts_defaults_to_hold(self, generator):
        """When only subordinate agents have verdicts, default to HOLD."""
        text = """
### Risky Analyst
Recommendation: BUY at 6%

### Safe Analyst
Recommendation: SELL immediately

### Research Manager
INVESTMENT RECOMMENDATION: BUY
"""
        # No PM verdict - must NOT pick up subordinate opinions
        assert generator.extract_decision(text) == "HOLD"

    def test_empty_string(self, generator):
        """Empty input defaults to HOLD."""
        assert generator.extract_decision("") == "HOLD"

    def test_none_like_content(self, generator):
        """Handles None-like content gracefully."""
        assert generator.extract_decision("None") == "HOLD"


class TestExtractDecisionRealWorldRegression:
    """Regression test from actual WPK.TO bug."""

    @pytest.fixture
    def generator(self):
        return QuietModeReporter(ticker="WPK.TO", company_name="Winpak Ltd.")

    def test_wpk_to_regression(self, generator):
        """
        WPK.TO showed "BUY" in title but PM said "DO NOT INITIATE".

        Root cause: Greedy regex matched "BUY" from "Default Decision: BUY"
        in the decision framework section.
        """
        text = """
# WPK.TO (Winpak Ltd.): [DECISION]

## Thesis Compliance at a Glance

Financial Health  79.0% (min 50%)
Growth Transition 33.0% (min 50%)

## Executive Summary

#### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE

**Hard Fail Result**: **FAIL on: [US Revenue, Growth Transition]**

**Decision Framework Applied**:
```
=== DECISION LOGIC ===
ZONE: LOW (< 1.0)
Default Decision: BUY
Actual Decision: DO NOT INITIATE
Override: NO (Hard Fails on US Revenue and Growth are binding)
======================
```

#### PM_BLOCK (REQUIRED - Machine-Readable Summary)

```
#### --- START PM_BLOCK ---
VERDICT: DO_NOT_INITIATE
HEALTH_ADJ: 79
GROWTH_ADJ: 33
RISK_TALLY: -0.67
ZONE: LOW
SHOW_VALUATION_CHART: NO
POSITION_SIZE: 0.0
#### --- END PM_BLOCK ---
```

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


class TestVerdictNormalization:
    """Test that verdict variants normalize correctly."""

    @pytest.fixture
    def generator(self):
        return QuietModeReporter(ticker="TEST.X", company_name="Test Corp")

    def test_normalization_mapping_exists(self, generator):
        """VERDICT_NORMALIZATION dict should have expected entries."""
        expected_keys = {
            "BUY",
            "SELL",
            "HOLD",
            "DO_NOT_INITIATE",
            "DO NOT INITIATE",
            "REJECT",
        }
        assert expected_keys.issubset(set(generator.VERDICT_NORMALIZATION.keys()))

    def test_reject_normalizes_to_do_not_initiate(self, generator):
        """REJECT should display as DO NOT INITIATE."""
        text = "VERDICT: REJECT"
        assert generator.extract_decision(text) == "DO NOT INITIATE"

    def test_underscored_normalizes_to_spaced(self, generator):
        """DO_NOT_INITIATE should display as DO NOT INITIATE."""
        text = "VERDICT: DO_NOT_INITIATE"
        assert generator.extract_decision(text) == "DO NOT INITIATE"
