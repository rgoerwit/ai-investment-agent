"""
Tests for data quality improvements (Jan 2026).

Verifies:
1. PEG fallback calculation is disabled
2. sector/industry are in IMPORTANT_FIELDS (prevents classification hallucination)
3. Prompt content includes precision guidelines and conservative counting rules
"""

import json
from pathlib import Path

import pytest

from src.data.fetcher import SmartMarketDataFetcher


class TestPEGCalculationDisabled:
    """Verify PEG fallback calculation is disabled."""

    def test_peg_not_calculated_when_missing(self):
        """PEG should remain None when provider doesn't supply it."""
        fetcher = SmartMarketDataFetcher()

        # Simulate data with PE and growth but no PEG
        test_data = {
            "trailingPE": 18.61,
            "earningsGrowth": 0.195,  # 19.5%
            "pegRatio": None,
        }

        # Call the internal calculation method
        calculated = fetcher._calculate_derived_metrics(test_data, "TEST")

        # PEG should NOT be calculated (disabled)
        assert "pegRatio" not in calculated, (
            "PEG fallback calculation should be disabled. "
            "Missing PEG is more honest than a calculated value with time horizon mismatch."
        )

    def test_peg_not_calculated_source_tag(self):
        """No _pegRatio_source tag should exist for calculated PEG."""
        fetcher = SmartMarketDataFetcher()

        test_data = {
            "trailingPE": 15.0,
            "earningsGrowth": 0.20,
            "pegRatio": None,
        }

        calculated = fetcher._calculate_derived_metrics(test_data, "TEST")

        assert (
            "_pegRatio_source" not in calculated
        ), "No source tag should exist since PEG calculation is disabled"


class TestSectorInImportantFields:
    """Verify sector/industry are tracked as important fields."""

    def test_sector_in_important_fields(self):
        """sector should be in IMPORTANT_FIELDS to prevent hallucination."""
        fetcher = SmartMarketDataFetcher()

        assert "sector" in fetcher.IMPORTANT_FIELDS, (
            "sector must be in IMPORTANT_FIELDS to trigger gap-filling "
            "and prevent downstream sector hallucination (e.g., industrial → tech)"
        )

    def test_industry_in_important_fields(self):
        """industry should be in IMPORTANT_FIELDS for sub-classification."""
        fetcher = SmartMarketDataFetcher()

        assert (
            "industry" in fetcher.IMPORTANT_FIELDS
        ), "industry must be in IMPORTANT_FIELDS for sector-specific threshold logic"

    def test_sector_industry_position(self):
        """sector/industry should be early in the list (high priority)."""
        fetcher = SmartMarketDataFetcher()

        sector_idx = fetcher.IMPORTANT_FIELDS.index("sector")
        industry_idx = fetcher.IMPORTANT_FIELDS.index("industry")

        # Should be in first 5 fields (high priority)
        assert sector_idx < 5, "sector should be near the top of IMPORTANT_FIELDS"
        assert industry_idx < 5, "industry should be near the top of IMPORTANT_FIELDS"


class TestPromptContentUpdates:
    """Verify prompt content includes required improvements."""

    @pytest.fixture
    def prompts_dir(self):
        """Get prompts directory path."""
        return Path(__file__).parent.parent / "prompts"

    def test_news_analyst_precision_guidelines(self, prompts_dir):
        """News analyst should have precision guidelines for US revenue reporting."""
        prompt_file = prompts_dir / "news_analyst.json"

        with open(prompt_file) as f:
            data = json.load(f)

        content = data["system_message"]

        # Check for precision guidelines
        assert (
            "PRECISION GUIDELINES" in content
        ), "News analyst must include precision guidelines section"
        assert (
            "tilde" in content.lower() or "~" in content
        ), "Precision guidelines must mention tilde (~) for estimates"
        assert (
            "whole number" in content.lower() or "nearest whole" in content.lower()
        ), "Precision guidelines must mention rounding to whole numbers"

    def test_fundamentals_analyst_conservative_count_rule(self, prompts_dir):
        """Fundamentals analyst should have conservative analyst count rule."""
        prompt_file = prompts_dir / "fundamentals_analyst.json"

        with open(prompt_file) as f:
            data = json.load(f)

        content = data["system_message"]

        # Check for conservative count rule
        assert (
            "DATA SOURCE PRIORITY" in content or "LOWER count" in content
        ), "Fundamentals analyst must include conservative count rule"
        assert (
            "numberOfAnalystOpinions" in content
        ), "Must reference structured field numberOfAnalystOpinions"
        assert (
            "LOWER" in content
        ), "Must specify using lower count when sources conflict"

    def test_junior_analyst_sector_reinforcement(self, prompts_dir):
        """Junior analyst should reinforce sector/industry output requirement."""
        prompt_file = prompts_dir / "junior_fundamentals_analyst.json"

        with open(prompt_file) as f:
            data = json.load(f)

        content = data["system_message"]

        # Check for sector reinforcement
        assert (
            "sector" in content.lower() and "industry" in content.lower()
        ), "Junior analyst must mention sector and industry"
        # Check for emphasis on including these fields
        assert (
            "MUST" in content or "CRITICAL" in content
        ), "Junior analyst must emphasize sector/industry are required"

    def test_prompt_versions_updated(self, prompts_dir):
        """Verify prompt versions have been updated."""
        # News analyst
        with open(prompts_dir / "news_analyst.json") as f:
            news = json.load(f)
        assert (
            news["version"] >= "4.8"
        ), f"News analyst version should be >= 4.8, got {news['version']}"

        # Fundamentals analyst
        with open(prompts_dir / "fundamentals_analyst.json") as f:
            fund = json.load(f)
        assert (
            fund["version"] >= "8.2"
        ), f"Fundamentals analyst version should be >= 8.2, got {fund['version']}"

        # Junior analyst
        with open(prompts_dir / "junior_fundamentals_analyst.json") as f:
            junior = json.load(f)
        assert (
            junior["version"] >= "1.2"
        ), f"Junior analyst version should be >= 1.2, got {junior['version']}"


class TestCalculatedMetricsStillWork:
    """Ensure other calculated metrics (ROE, marketCap) still work."""

    def test_roe_still_calculated(self):
        """ROE calculation from ROA and D/E should still work."""
        fetcher = SmartMarketDataFetcher()

        test_data = {
            "returnOnAssets": 0.05,  # 5%
            "debtToEquity": 1.0,  # 100%
            "returnOnEquity": None,
        }

        calculated = fetcher._calculate_derived_metrics(test_data, "TEST")

        # ROE = ROA * (1 + D/E) = 0.05 * (1 + 1.0) = 0.10
        assert "returnOnEquity" in calculated
        assert abs(calculated["returnOnEquity"] - 0.10) < 0.001

    def test_market_cap_still_calculated(self):
        """Market cap calculation from price and shares should still work."""
        fetcher = SmartMarketDataFetcher()

        test_data = {
            "currentPrice": 100.0,
            "sharesOutstanding": 1_000_000,
            "marketCap": None,
        }

        calculated = fetcher._calculate_derived_metrics(test_data, "TEST")

        assert "marketCap" in calculated
        assert calculated["marketCap"] == 100_000_000
