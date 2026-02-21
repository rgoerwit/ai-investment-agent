"""
Tests for Graham consecutive positive earnings calculation.

Benjamin Graham's quality filter: requires consecutive years of positive
earnings as evidence of business stability and durability.
"""

import pandas as pd
import pytest

from src.data.fetcher import SmartMarketDataFetcher


class TestGrahamEarningsCalculation:
    """Test the Graham earnings test calculation."""

    @pytest.fixture
    def fetcher(self):
        return SmartMarketDataFetcher()

    def test_all_positive_earnings_pass(self, fetcher):
        """5 years of positive earnings should pass."""
        financials = pd.DataFrame(
            {
                "2024": [100e6],
                "2023": [90e6],
                "2022": [80e6],
                "2021": [70e6],
                "2020": [60e6],
            },
            index=["Net Income"],
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        assert result["graham_consecutive_positive_years"] == 5
        assert result["graham_test"] == "PASS"

    def test_four_positive_years_pass(self, fetcher):
        """4 years of positive earnings should pass."""
        financials = pd.DataFrame(
            {"2024": [100e6], "2023": [90e6], "2022": [80e6], "2021": [70e6]},
            index=["Net Income"],
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        assert result["graham_consecutive_positive_years"] == 4
        assert result["graham_test"] == "PASS"

    def test_recent_loss_breaks_streak(self, fetcher):
        """Loss in most recent year should result in 0 consecutive years."""
        financials = pd.DataFrame(
            {"2024": [-10e6], "2023": [90e6], "2022": [80e6], "2021": [70e6]},
            index=["Net Income"],
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        assert result["graham_consecutive_positive_years"] == 0
        assert result["graham_test"] == "FAIL"

    def test_mid_period_loss_stops_count(self, fetcher):
        """Loss in middle of period stops the consecutive count."""
        financials = pd.DataFrame(
            {"2024": [100e6], "2023": [90e6], "2022": [-5e6], "2021": [70e6]},
            index=["Net Income"],
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        # Only 2024 and 2023 are consecutive from most recent
        assert result["graham_consecutive_positive_years"] == 2
        assert result["graham_test"] == "FAIL"

    def test_three_positive_one_loss_fails(self, fetcher):
        """3 positive years with a loss should fail."""
        financials = pd.DataFrame(
            {"2024": [100e6], "2023": [90e6], "2022": [80e6], "2021": [-10e6]},
            index=["Net Income"],
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        assert result["graham_consecutive_positive_years"] == 3
        assert result["graham_test"] == "FAIL"

    def test_empty_dataframe_insufficient_data(self, fetcher):
        """Empty income statement returns INSUFFICIENT_DATA."""
        financials = pd.DataFrame()
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        assert result["graham_test"] == "INSUFFICIENT_DATA"
        assert result["graham_consecutive_positive_years"] is None

    def test_missing_net_income_row(self, fetcher):
        """Missing Net Income row returns INSUFFICIENT_DATA."""
        financials = pd.DataFrame(
            {"2024": [100e6], "2023": [90e6]},
            index=["Revenue"],  # Wrong row name
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        assert result["graham_test"] == "INSUFFICIENT_DATA"
        assert result["graham_consecutive_positive_years"] is None

    def test_nan_values_break_streak(self, fetcher):
        """NaN values should stop the consecutive count."""
        financials = pd.DataFrame(
            {"2024": [100e6], "2023": [float("nan")], "2022": [80e6]},
            index=["Net Income"],
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        # NaN breaks streak at position 1
        assert result["graham_consecutive_positive_years"] == 1

    def test_zero_earnings_breaks_streak(self, fetcher):
        """Zero earnings (exactly) breaks the streak (not strictly positive)."""
        financials = pd.DataFrame(
            {"2024": [100e6], "2023": [0], "2022": [80e6]},
            index=["Net Income"],
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        # Zero is not > 0, so streak breaks
        assert result["graham_consecutive_positive_years"] == 1

    def test_only_two_years_insufficient(self, fetcher):
        """Only 2 years of data is insufficient for meaningful test."""
        financials = pd.DataFrame(
            {"2024": [100e6], "2023": [90e6]},
            index=["Net Income"],
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        assert result["graham_consecutive_positive_years"] == 2
        # Less than 3 years available
        assert result["graham_test"] == "INSUFFICIENT_DATA"

    def test_years_available_tracked(self, fetcher):
        """Should track how many years of data were available."""
        financials = pd.DataFrame(
            {"2024": [100e6], "2023": [90e6], "2022": [80e6], "2021": [70e6]},
            index=["Net Income"],
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        assert result["_graham_years_available"] == 4


class TestGrahamTestEdgeCases:
    """Edge cases for Graham earnings test."""

    @pytest.fixture
    def fetcher(self):
        return SmartMarketDataFetcher()

    def test_single_year_insufficient(self, fetcher):
        """Single year of data is insufficient."""
        financials = pd.DataFrame(
            {"2024": [100e6]},
            index=["Net Income"],
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        assert result["graham_test"] == "INSUFFICIENT_DATA"

    def test_all_negative_years_fail(self, fetcher):
        """All negative years should definitely fail."""
        financials = pd.DataFrame(
            {"2024": [-10e6], "2023": [-20e6], "2022": [-30e6], "2021": [-40e6]},
            index=["Net Income"],
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        assert result["graham_consecutive_positive_years"] == 0
        assert result["graham_test"] == "FAIL"

    def test_large_numbers_handled(self, fetcher):
        """Large numbers (billions) should be handled correctly."""
        financials = pd.DataFrame(
            {
                "2024": [50e9],  # $50B
                "2023": [45e9],
                "2022": [40e9],
                "2021": [35e9],
                "2020": [30e9],
            },
            index=["Net Income"],
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        assert result["graham_consecutive_positive_years"] == 5
        assert result["graham_test"] == "PASS"

    def test_small_positive_numbers(self, fetcher):
        """Small positive numbers (barely profitable) should still count."""
        financials = pd.DataFrame(
            {"2024": [1000], "2023": [500], "2022": [100], "2021": [50]},
            index=["Net Income"],
        )
        result = fetcher._calculate_graham_earnings_test(financials, "TEST")
        assert result["graham_consecutive_positive_years"] == 4
        assert result["graham_test"] == "PASS"
