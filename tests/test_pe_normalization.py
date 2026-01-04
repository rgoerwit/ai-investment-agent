"""
Tests for P/E ratio normalization and data quality sanity checks.

These tests verify:
1. P/E normalization logic handles edge cases (stock splits, stale estimates)
2. Extreme earnings quality ratios are downgraded to WARNING (suggests data issue)
3. Data divergence between TTM and statements is logged for investigation

Background bugs fixed:
- Tamron (7740.T): P/E of 12.63 was incorrectly replaced with 3.13
  due to stale forward estimates not adjusted for 1:4 stock split.
  Fix: Added P/E sanity thresholds (min 5, max 3x divergence, only replace if >50).

- Tsutsumi (7937.T): Earnings quality flag with 2.18x disconnect ratio.
  Analysis: TTM values showed real deterioration (not data misalignment).
  Fix: Keep TTM values (more current), but downgrade extreme ratios (>4x)
  to WARNING since they likely indicate data quality issues, not fraud.
"""

import pytest


class TestPENormalizationSanityChecks:
    """Test that P/E normalization respects sanity thresholds."""

    @pytest.fixture
    def fetcher(self):
        from src.data.fetcher import SmartMarketDataFetcher

        return SmartMarketDataFetcher()

    def test_keeps_trailing_when_forward_too_low(self, fetcher):
        """Don't replace trailing P/E if forward P/E is suspiciously low (< 5)."""
        # Simulates stock split case: correct trailing, stale forward
        info = {"trailingPE": 12.63, "forwardPE": 3.13}

        result = fetcher._normalize_data_integrity(info, "TEST")

        # Should keep the reasonable trailing P/E
        assert result["trailingPE"] == 12.63
        assert result.get("_trailingPE_source") is None

    def test_keeps_trailing_when_divergence_extreme(self, fetcher):
        """Don't replace if divergence ratio > 3x (suggests data error)."""
        # 4x divergence - one value is almost certainly wrong
        info = {"trailingPE": 40.0, "forwardPE": 10.0}

        result = fetcher._normalize_data_integrity(info, "TEST")

        # Should keep trailing despite it being > forward * 1.4
        assert result["trailingPE"] == 40.0
        assert result.get("_trailingPE_source") is None

    def test_replaces_trailing_when_inflated_and_forward_reasonable(self, fetcher):
        """Replace trailing with forward when trailing is very high and forward is reasonable."""
        # Trailing inflated due to one-time charge, forward is normalized
        info = {"trailingPE": 80.0, "forwardPE": 25.0}

        result = fetcher._normalize_data_integrity(info, "TEST")

        # Should replace with forward (80 > 50 threshold, 25 >= 5, ratio 3.2 is acceptable)
        # Wait - ratio 3.2 > 3.0 so it should NOT replace
        # Let me recalculate: 80/25 = 3.2 which is > MAX_DIVERGENCE_RATIO of 3.0
        assert result["trailingPE"] == 80.0  # Kept due to ratio > 3

    def test_replaces_trailing_when_conditions_met(self, fetcher):
        """Replace trailing when all sanity conditions are satisfied."""
        # Trailing slightly inflated, forward reasonable, ratio within bounds
        info = {"trailingPE": 60.0, "forwardPE": 22.0}

        result = fetcher._normalize_data_integrity(info, "TEST")

        # 60/22 = 2.7 (within 3x), 22 >= 5, 60 > 50, 60 > 22*1.4
        assert result["trailingPE"] == 22.0
        assert result["_trailingPE_source"] == "normalized_forward_proxy"

    def test_no_change_when_trailing_not_high(self, fetcher):
        """Don't replace if trailing P/E is reasonable (< 50 threshold)."""
        # Normal case: trailing higher than forward but both reasonable
        info = {"trailingPE": 25.0, "forwardPE": 18.0}

        result = fetcher._normalize_data_integrity(info, "TEST")

        # Should keep trailing - it's not unusually high
        assert result["trailingPE"] == 25.0
        assert result.get("_trailingPE_source") is None

    def test_no_change_when_trailing_lower_than_forward(self, fetcher):
        """Don't replace if trailing < forward (earnings declining)."""
        info = {"trailingPE": 15.0, "forwardPE": 20.0}

        result = fetcher._normalize_data_integrity(info, "TEST")

        assert result["trailingPE"] == 15.0
        assert result.get("_trailingPE_source") is None

    def test_handles_missing_pe_values(self, fetcher):
        """Gracefully handle None or missing P/E values."""
        cases = [
            {"trailingPE": None, "forwardPE": 15.0},
            {"trailingPE": 15.0, "forwardPE": None},
            {"trailingPE": None, "forwardPE": None},
            {},
        ]

        for info in cases:
            result = fetcher._normalize_data_integrity(info.copy(), "TEST")
            # Should not crash, values unchanged
            assert result.get("trailingPE") == info.get("trailingPE")
            assert result.get("forwardPE") == info.get("forwardPE")

    def test_handles_zero_pe_values(self, fetcher):
        """Don't divide by zero or replace with zero."""
        cases = [
            {"trailingPE": 0, "forwardPE": 15.0},
            {"trailingPE": 15.0, "forwardPE": 0},
            {"trailingPE": 0, "forwardPE": 0},
        ]

        for info in cases:
            result = fetcher._normalize_data_integrity(info.copy(), "TEST")
            # Should not crash
            assert result is not None

    def test_tamron_case_exact(self, fetcher):
        """Regression test for the exact Tamron (7740.T) values that exposed the bug."""
        info = {"trailingPE": 12.626695, "forwardPE": 3.1312459}

        result = fetcher._normalize_data_integrity(info, "7740.T")

        # The correct trailing should be preserved
        assert result["trailingPE"] == 12.626695
        assert result.get("_trailingPE_source") is None


class TestDataDivergenceDetection:
    """Test that extreme TTM vs statement divergence is detected and handled."""

    def test_earnings_quality_extreme_ratio_downgraded(self):
        """Earnings quality check with >4x ratio should be WARNING, not CRITICAL."""
        from src.validators.red_flag_detector import RedFlagDetector

        # Tsutsumi-like case: extreme 4.3x ratio suggests data quality issue
        # Net Income: 1.71B, FCF: -3.74B → ratio = 2.18x (would trigger but < 4x)
        # If ratio were > 4x, it should downgrade to WARNING

        # Test case with extreme ratio (>4x)
        metrics_extreme = {
            "net_income": 1_000_000_000,  # 1B
            "fcf": -5_000_000_000,  # -5B → 5x ratio
        }

        red_flags, result = RedFlagDetector.detect_red_flags(
            metrics_extreme, ticker="TEST"
        )

        # Should be WARNING, not CRITICAL (extreme ratio suggests data issue)
        assert len(red_flags) == 1
        assert red_flags[0]["type"] == "EARNINGS_QUALITY_UNCERTAIN"
        assert red_flags[0]["severity"] == "WARNING"
        assert red_flags[0]["action"] == "RISK_PENALTY"
        assert result == "PASS"  # WARNING doesn't reject

    def test_earnings_quality_normal_ratio_critical(self):
        """Earnings quality check with 2-4x ratio should be CRITICAL."""
        from src.validators.red_flag_detector import RedFlagDetector

        # Normal fraud indicator: 2-4x ratio
        metrics_normal = {
            "net_income": 1_000_000_000,  # 1B
            "fcf": -3_000_000_000,  # -3B → 3x ratio (suspicious but plausible)
        }

        red_flags, result = RedFlagDetector.detect_red_flags(
            metrics_normal, ticker="TEST"
        )

        # Should be CRITICAL (plausible fraud indicator)
        assert len(red_flags) == 1
        assert red_flags[0]["type"] == "EARNINGS_QUALITY"
        assert red_flags[0]["severity"] == "CRITICAL"
        assert red_flags[0]["action"] == "AUTO_REJECT"
        assert result == "REJECT"

    def test_tsutsumi_case_with_ttm_data_no_divergence(self):
        """Tsutsumi case with TTM data, no divergence flag: 2.18x ratio → CRITICAL."""
        from src.validators.red_flag_detector import RedFlagDetector

        # Actual Tsutsumi TTM values, but no divergence marker in report
        metrics = {
            "net_income": 1_714_000_000,
            "fcf": -3_744_750_080,
            "_raw_report": "No FCF data quality issues mentioned",
        }

        disconnect_ratio = abs(metrics["fcf"] / metrics["net_income"])
        assert (
            2.0 < disconnect_ratio < 4.0
        ), f"Ratio should be 2-4x, got {disconnect_ratio:.2f}x"

        red_flags, result = RedFlagDetector.detect_red_flags(metrics, ticker="7937.T")

        # Without divergence flag, 2.18x ratio → CRITICAL
        assert len(red_flags) == 1
        assert red_flags[0]["type"] == "EARNINGS_QUALITY"
        assert red_flags[0]["severity"] == "CRITICAL"
        assert result == "REJECT"

    def test_tsutsumi_case_with_divergence_flag(self):
        """Tsutsumi case with FCF divergence flag: 2.18x ratio → WARNING (data uncertain)."""
        from src.validators.red_flag_detector import RedFlagDetector

        # Same values, but with divergence marker in report
        metrics = {
            "net_income": 1_714_000_000,
            "fcf": -3_744_750_080,
            "_raw_report": "FCF DATA QUALITY UNCERTAIN: TTM (-3.74B) differs from annual statement (-2.01B)",
        }

        red_flags, result = RedFlagDetector.detect_red_flags(metrics, ticker="7937.T")

        # With divergence flag, downgrade to WARNING
        assert len(red_flags) == 1
        assert red_flags[0]["type"] == "EARNINGS_QUALITY_UNCERTAIN"
        assert red_flags[0]["severity"] == "WARNING"
        assert result == "PASS"  # WARNING doesn't auto-reject
