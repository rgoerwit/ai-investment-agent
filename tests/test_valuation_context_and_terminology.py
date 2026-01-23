"""
Tests for valuation context, data integrity tiers, and action terminology (Jan 2026).

Verifies:
1. Valuation context exception logic in Portfolio Manager
2. Data integrity tiers (C1 structural vs C2 financial)
3. DO NOT INITIATE vs SELL terminology distinction
4. VALUATION_CONTEXT field in Fundamentals Analyst
"""

import json
from pathlib import Path

import pytest


class TestValuationContextException:
    """Verify valuation context exception is properly documented."""

    @pytest.fixture
    def pm_prompt(self):
        """Load Portfolio Manager prompt."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        with open(prompts_dir / "portfolio_manager.json") as f:
            return json.load(f)

    def test_valuation_context_exception_exists(self, pm_prompt):
        """Portfolio Manager should have valuation context exception."""
        content = pm_prompt["system_message"]

        assert (
            "VALUATION CONTEXT EXCEPTION" in content
        ), "PM must have valuation context exception for justified high multiples"

    def test_contractual_revenue_mentioned(self, pm_prompt):
        """Contractual revenue should be a valid exception."""
        content = pm_prompt["system_message"]

        assert (
            "Contractual Revenue" in content or "contractual" in content.lower()
        ), "Contractual/recurring revenue should justify higher P/E"

    def test_moat_protection_mentioned(self, pm_prompt):
        """Moat protection should be a valid exception."""
        content = pm_prompt["system_message"]

        assert (
            "MOAT_DURABLE_ADVANTAGE" in content or "Durable Moat" in content
        ), "Durable moat should justify higher P/E"

    def test_capital_efficiency_improving_mentioned(self, pm_prompt):
        """Improving capital efficiency should be a valid exception."""
        content = pm_prompt["system_message"]

        assert (
            "ROIC trending" in content or "Capital Efficiency Improving" in content
        ), "Improving ROIC should justify higher P/E"


class TestDataIntegrityTiers:
    """Verify data integrity tier system is documented."""

    @pytest.fixture
    def pm_prompt(self):
        """Load Portfolio Manager prompt."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        with open(prompts_dir / "portfolio_manager.json") as f:
            return json.load(f)

    def test_data_tiers_exist(self, pm_prompt):
        """Data integrity tiers should exist."""
        content = pm_prompt["system_message"]

        assert (
            "DATA INTEGRITY TIERS" in content
        ), "PM must document data integrity tier system"

    def test_tier_c1_structural(self, pm_prompt):
        """Tier C1 should cover structural issues."""
        content = pm_prompt["system_message"]

        assert (
            "TIER C1" in content or "C1 - Structural" in content
        ), "C1 tier for structural confusion must exist"
        # Should mention low penalty
        assert (
            "+0.25" in content or "MAX +0.5" in content
        ), "C1 should have lower penalty than C2"

    def test_tier_c2_financial(self, pm_prompt):
        """Tier C2 should cover financial metric conflicts."""
        content = pm_prompt["system_message"]

        assert (
            "TIER C2" in content or "C2 - Financial" in content
        ), "C2 tier for financial conflicts must exist"
        # Should mention ROIC, OCF, margins
        assert (
            "ROIC" in content or "OCF" in content
        ), "C2 should cover core financial metrics"

    def test_c2_higher_penalty_than_c1(self, pm_prompt):
        """C2 should have higher penalty than C1."""
        content = pm_prompt["system_message"]

        # C1 is +0.25, C2 is +0.75
        assert "+0.75" in content, "C2 should have significantly higher penalty (+0.75)"


class TestActionTerminology:
    """Verify DO NOT INITIATE vs SELL distinction."""

    @pytest.fixture
    def pm_prompt(self):
        """Load Portfolio Manager prompt."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        with open(prompts_dir / "portfolio_manager.json") as f:
            return json.load(f)

    def test_do_not_initiate_exists(self, pm_prompt):
        """DO NOT INITIATE action should exist."""
        content = pm_prompt["system_message"]

        assert (
            "DO NOT INITIATE" in content
        ), "PM must support DO NOT INITIATE as distinct action"

    def test_sell_reserved_for_deteriorating(self, pm_prompt):
        """SELL should be reserved for deteriorating fundamentals."""
        content = pm_prompt["system_message"]

        # Should mention SELL is for deteriorating/held positions
        assert (
            "deteriorating" in content.lower() or "Deteriorating" in content
        ), "SELL should be reserved for deteriorating fundamentals"

    def test_verdict_options_include_do_not_initiate(self, pm_prompt):
        """Verdict options should include DO NOT INITIATE."""
        content = pm_prompt["system_message"]

        assert (
            "BUY / HOLD / DO NOT INITIATE" in content
        ), "Verdict options should include DO NOT INITIATE"

    def test_action_terminology_section_exists(self, pm_prompt):
        """Action terminology section should explain the distinction."""
        content = pm_prompt["system_message"]

        assert (
            "ACTION TERMINOLOGY" in content
        ), "PM must have ACTION TERMINOLOGY section explaining distinctions"

    def test_semantic_precision_explained(self, pm_prompt):
        """Semantic distinction should be explained."""
        content = pm_prompt["system_message"]

        # Should explain that DO NOT INITIATE is not negative alpha
        assert (
            "opportunity cost" in content.lower() or "capital loss" in content.lower()
        ), "Should distinguish between opportunity cost and capital loss"


class TestFundamentalsValuationContext:
    """Verify Fundamentals Analyst has VALUATION_CONTEXT field."""

    @pytest.fixture
    def fund_prompt(self):
        """Load Fundamentals Analyst prompt."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        with open(prompts_dir / "fundamentals_analyst.json") as f:
            return json.load(f)

    def test_valuation_context_in_data_block(self, fund_prompt):
        """VALUATION_CONTEXT should be in DATA_BLOCK."""
        content = fund_prompt["system_message"]

        assert (
            "VALUATION_CONTEXT:" in content
        ), "DATA_BLOCK must include VALUATION_CONTEXT field"

    def test_valuation_context_options(self, fund_prompt):
        """VALUATION_CONTEXT should have defined options."""
        content = fund_prompt["system_message"]

        assert "CONTRACTUAL" in content, "Should support CONTRACTUAL context"
        assert (
            "IMPROVING_EFFICIENCY" in content
        ), "Should support IMPROVING_EFFICIENCY context"
        assert "MOAT_PROTECTED" in content, "Should support MOAT_PROTECTED context"
        assert "STANDARD" in content, "Should support STANDARD (default) context"

    def test_valuation_context_classification_explained(self, fund_prompt):
        """VALUATION_CONTEXT classification should be explained."""
        content = fund_prompt["system_message"]

        assert (
            "VALUATION CONTEXT CLASSIFICATION" in content
        ), "Must explain how to classify VALUATION_CONTEXT"


class TestResearchManagerTerminology:
    """Verify Research Manager has terminology clarification."""

    @pytest.fixture
    def rm_prompt(self):
        """Load Research Manager prompt."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        with open(prompts_dir / "research_manager.json") as f:
            return json.load(f)

    def test_terminology_note_exists(self, rm_prompt):
        """Research Manager should clarify REJECT terminology."""
        content = rm_prompt["system_message"]

        assert (
            "Terminology Note" in content or "DO NOT INITIATE" in content
        ), "RM should clarify how REJECT maps to PM actions"


class TestPromptVersionsUpdated:
    """Verify prompt versions have been updated."""

    @pytest.fixture
    def prompts_dir(self):
        """Get prompts directory."""
        return Path(__file__).parent.parent / "prompts"

    def test_pm_version_updated(self, prompts_dir):
        """Portfolio Manager should be v7.6+."""
        with open(prompts_dir / "portfolio_manager.json") as f:
            data = json.load(f)
        version = float(data["version"])
        assert version >= 7.6, f"PM version should be >= 7.6, got {data['version']}"

    def test_fund_version_updated(self, prompts_dir):
        """Fundamentals Analyst should be v8.3+."""
        with open(prompts_dir / "fundamentals_analyst.json") as f:
            data = json.load(f)
        version = float(data["version"])
        assert version >= 8.3, f"Fund version should be >= 8.3, got {data['version']}"

    def test_rm_version_updated(self, prompts_dir):
        """Research Manager should be v4.7+."""
        with open(prompts_dir / "research_manager.json") as f:
            data = json.load(f)
        version = float(data["version"])
        assert version >= 4.7, f"RM version should be >= 4.7, got {data['version']}"
