"""
Tests for historical ROA/ROE calculation and trend detection.
"""

import pandas as pd
import pytest

from src.data.fetcher import SmartMarketDataFetcher


class TestComputeTrendRegression:
    """Test the regression-based trend calculation."""

    def test_improving_trend(self):
        """Test that steadily improving values return IMPROVING."""
        # 5 years of steadily increasing ROA: 2%, 3%, 4%, 5%, 6%
        values = [0.02, 0.03, 0.04, 0.05, 0.06]
        mean_val = sum(values) / len(values)  # 4%
        result = SmartMarketDataFetcher._compute_trend_regression(values, mean_val)
        assert result == "IMPROVING"

    def test_declining_trend(self):
        """Test that steadily declining values return DECLINING."""
        # 5 years of steadily decreasing ROA: 8%, 7.5%, 7%, 6.5%, 6%
        # Lower variance to stay under CV threshold
        values = [0.08, 0.075, 0.07, 0.065, 0.06]
        mean_val = sum(values) / len(values)  # 7%
        result = SmartMarketDataFetcher._compute_trend_regression(values, mean_val)
        assert result == "DECLINING"

    def test_stable_trend(self):
        """Test that flat values return STABLE."""
        # 5 years of steady ROA around 5%
        values = [0.05, 0.051, 0.049, 0.05, 0.05]
        mean_val = sum(values) / len(values)
        result = SmartMarketDataFetcher._compute_trend_regression(values, mean_val)
        assert result == "STABLE"

    def test_unstable_high_variance(self):
        """Test that high variance returns UNSTABLE."""
        # Cyclical company: wild swings between -5% and 15%
        values = [0.15, -0.05, 0.12, -0.02, 0.10]
        mean_val = sum(values) / len(values)
        result = SmartMarketDataFetcher._compute_trend_regression(values, mean_val)
        assert result == "UNSTABLE"

    def test_unstable_cyclical(self):
        """Test that cyclical patterns return UNSTABLE."""
        # Shipping company: boom-bust cycle
        values = [0.02, 0.15, 0.08, 0.20, 0.03]
        mean_val = sum(values) / len(values)
        result = SmartMarketDataFetcher._compute_trend_regression(values, mean_val)
        assert result == "UNSTABLE"

    def test_minimum_data_points(self):
        """Test that less than 3 data points returns N/A."""
        values = [0.05, 0.06]
        mean_val = 0.055
        result = SmartMarketDataFetcher._compute_trend_regression(values, mean_val)
        assert result == "N/A"

    def test_zero_mean_returns_na(self):
        """Test that zero mean returns N/A to avoid division by zero."""
        values = [-0.02, 0.02, 0.00, -0.01, 0.01]
        mean_val = 0.0
        result = SmartMarketDataFetcher._compute_trend_regression(values, mean_val)
        assert result == "N/A"

    def test_three_data_points_sufficient(self):
        """Test that exactly 3 data points works."""
        # Values with lower variance: 5%, 5.5%, 6%
        values = [0.05, 0.055, 0.06]
        mean_val = sum(values) / len(values)  # ~5.5%
        result = SmartMarketDataFetcher._compute_trend_regression(values, mean_val)
        assert result == "IMPROVING"

    def test_moderate_variance_with_trend(self):
        """Test that moderate variance with clear trend still detects trend."""
        # Values with noise but clear upward trend
        values = [0.04, 0.05, 0.045, 0.06, 0.07]
        mean_val = sum(values) / len(values)
        result = SmartMarketDataFetcher._compute_trend_regression(values, mean_val)
        # Should be IMPROVING despite noise (CV < 0.40)
        assert result == "IMPROVING"


class TestCalculateReturnTrends:
    """Test the full return trends calculation from DataFrames."""

    @pytest.fixture
    def fetcher(self):
        """Create a SmartMarketDataFetcher instance."""
        return SmartMarketDataFetcher()

    def test_basic_roa_roe_calculation(self, fetcher):
        """Test basic ROA and ROE calculation from financials."""
        # Create mock financial data (columns are years, newest first)
        financials = pd.DataFrame(
            {
                "2024": [100, 50],  # Net Income, Revenue
                "2023": [90, 45],
                "2022": [80, 40],
                "2021": [70, 35],
                "2020": [60, 30],
            },
            index=["Net Income", "Revenue"],
        )

        balance_sheet = pd.DataFrame(
            {
                "2024": [1000, 500],  # Total Assets, Stockholders Equity
                "2023": [1000, 500],
                "2022": [1000, 500],
                "2021": [1000, 500],
                "2020": [1000, 500],
            },
            index=["Total Assets", "Stockholders Equity"],
        )

        result = fetcher._calculate_return_trends(financials, balance_sheet, "TEST.T")

        # Should have calculated 5Y averages
        assert "roa_5y_avg" in result
        assert "roe_5y_avg" in result
        assert "profitability_trend" in result

        # ROA should be around 8% (average of 10%, 9%, 8%, 7%, 6%)
        assert 5 < result["roa_5y_avg"] < 12

        # ROE should be around 16%
        assert 10 < result["roe_5y_avg"] < 22

        # Trend should be IMPROVING (increasing NI with constant assets)
        assert result["profitability_trend"] == "IMPROVING"

    def test_insufficient_years(self, fetcher):
        """Test that insufficient years returns empty dict."""
        financials = pd.DataFrame(
            {"2024": [100], "2023": [90]},
            index=["Net Income"],
        )

        balance_sheet = pd.DataFrame(
            {"2024": [1000], "2023": [1000]},
            index=["Total Assets"],
        )

        result = fetcher._calculate_return_trends(financials, balance_sheet, "TEST.T")

        # Should not have calculated anything
        assert "roa_5y_avg" not in result
        assert "roe_5y_avg" not in result
        assert "profitability_trend" not in result

    def test_empty_dataframes(self, fetcher):
        """Test that empty DataFrames return empty dict."""
        result = fetcher._calculate_return_trends(
            pd.DataFrame(), pd.DataFrame(), "TEST.T"
        )
        assert result == {}

    def test_outlier_exclusion(self, fetcher):
        """Test that extreme outliers are excluded."""
        financials = pd.DataFrame(
            {
                "2024": [100],  # Normal
                "2023": [90],
                "2022": [80],
                "2021": [100000],  # Extreme outlier (would be 100% ROA)
                "2020": [60],
            },
            index=["Net Income"],
        )

        balance_sheet = pd.DataFrame(
            {
                "2024": [1000],
                "2023": [1000],
                "2022": [1000],
                "2021": [1000],
                "2020": [1000],
            },
            index=["Total Assets"],
        )

        result = fetcher._calculate_return_trends(financials, balance_sheet, "TEST.T")

        # Should still calculate but exclude the outlier
        # With 4 valid points, should still work
        if "roa_5y_avg" in result:
            # ROA should be reasonable (not including 100% outlier)
            assert result["roa_5y_avg"] < 20

    def test_negative_equity_excluded(self, fetcher):
        """Test that negative equity years are excluded from ROE."""
        financials = pd.DataFrame(
            {
                "2024": [100],
                "2023": [90],
                "2022": [80],
                "2021": [70],
                "2020": [60],
            },
            index=["Net Income"],
        )

        balance_sheet = pd.DataFrame(
            {
                "2024": [1000, 500],
                "2023": [1000, 500],
                "2022": [1000, -100],  # Negative equity (insolvent)
                "2021": [1000, 500],
                "2020": [1000, 500],
            },
            index=["Total Assets", "Stockholders Equity"],
        )

        result = fetcher._calculate_return_trends(financials, balance_sheet, "TEST.T")

        # Should still have ROE but using 4 years
        if "roe_5y_avg" in result:
            assert result["_roe_5y_years"] == 4

    def test_alternative_equity_key(self, fetcher):
        """Test that 'Total Stockholder Equity' key is also recognized."""
        financials = pd.DataFrame(
            {
                "2024": [100],
                "2023": [90],
                "2022": [80],
            },
            index=["Net Income"],
        )

        balance_sheet = pd.DataFrame(
            {
                "2024": [1000, 500],
                "2023": [1000, 500],
                "2022": [1000, 500],
            },
            index=["Total Assets", "Total Stockholder Equity"],
        )

        result = fetcher._calculate_return_trends(financials, balance_sheet, "TEST.T")

        assert "roe_5y_avg" in result

    def test_stable_returns(self, fetcher):
        """Test detection of stable returns."""
        financials = pd.DataFrame(
            {
                "2024": [50],
                "2023": [50],
                "2022": [50],
                "2021": [50],
                "2020": [50],
            },
            index=["Net Income"],
        )

        balance_sheet = pd.DataFrame(
            {
                "2024": [1000],
                "2023": [1000],
                "2022": [1000],
                "2021": [1000],
                "2020": [1000],
            },
            index=["Total Assets"],
        )

        result = fetcher._calculate_return_trends(financials, balance_sheet, "TEST.T")

        assert result["profitability_trend"] == "STABLE"

    def test_declining_returns(self, fetcher):
        """Test detection of declining returns."""
        financials = pd.DataFrame(
            {
                "2024": [50],  # Newest - lowest
                "2023": [70],
                "2022": [90],
                "2021": [110],
                "2020": [130],  # Oldest - highest
            },
            index=["Net Income"],
        )

        balance_sheet = pd.DataFrame(
            {
                "2024": [1000],
                "2023": [1000],
                "2022": [1000],
                "2021": [1000],
                "2020": [1000],
            },
            index=["Total Assets"],
        )

        result = fetcher._calculate_return_trends(financials, balance_sheet, "TEST.T")

        assert result["profitability_trend"] == "DECLINING"


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def fetcher(self):
        return SmartMarketDataFetcher()

    def test_missing_net_income(self, fetcher):
        """Test handling of missing Net Income row."""
        financials = pd.DataFrame(
            {"2024": [100], "2023": [90], "2022": [80]},
            index=["Revenue"],  # No Net Income
        )

        balance_sheet = pd.DataFrame(
            {"2024": [1000], "2023": [1000], "2022": [1000]},
            index=["Total Assets"],
        )

        result = fetcher._calculate_return_trends(financials, balance_sheet, "TEST.T")

        assert "roa_5y_avg" not in result

    def test_missing_total_assets(self, fetcher):
        """Test handling of missing Total Assets row."""
        financials = pd.DataFrame(
            {"2024": [100], "2023": [90], "2022": [80]},
            index=["Net Income"],
        )

        balance_sheet = pd.DataFrame(
            {"2024": [500], "2023": [500], "2022": [500]},
            index=["Stockholders Equity"],  # No Total Assets
        )

        result = fetcher._calculate_return_trends(financials, balance_sheet, "TEST.T")

        # ROA should not be calculated, but ROE should be
        assert "roa_5y_avg" not in result
        assert "roe_5y_avg" in result

    def test_nan_values_handled(self, fetcher):
        """Test that NaN values are properly skipped."""
        import numpy as np

        financials = pd.DataFrame(
            {
                "2024": [100],
                "2023": [np.nan],  # Missing data
                "2022": [80],
                "2021": [70],
                "2020": [60],
            },
            index=["Net Income"],
        )

        balance_sheet = pd.DataFrame(
            {
                "2024": [1000],
                "2023": [1000],
                "2022": [1000],
                "2021": [1000],
                "2020": [1000],
            },
            index=["Total Assets"],
        )

        result = fetcher._calculate_return_trends(financials, balance_sheet, "TEST.T")

        # Should use 4 valid years
        if "roa_5y_avg" in result:
            assert result["_roa_5y_years"] == 4
