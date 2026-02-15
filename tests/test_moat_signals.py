"""
Tests for Economic Moat Signal Detection.

Tests cover:
1. Moat signal calculation in fetcher (_calculate_moat_signals)
2. DATA_BLOCK extraction (extract_moat_signals)
3. Flag detection (detect_moat_flags)
4. Edge cases and failure modes
5. Integration with Portfolio Manager risk tally

Run with: pytest tests/test_moat_signals.py -v
"""

import statistics

import pandas as pd
import pytest

from src.validators.red_flag_detector import RedFlagDetector


class TestMoatSignalCalculation:
    """Test moat signal calculation in fetcher."""

    def test_gross_margin_cv_calculation_stable(self):
        """Test CV calculation with stable margins results in HIGH signal."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        # Stable margins: [30%, 32%, 31%, 29%, 30%]
        # Mean ~ 0.304, sample stdev ~ 0.0114, CV ~ 0.037 -> HIGH
        financials = pd.DataFrame(
            {
                "2024": [100, 30],
                "2023": [100, 32],
                "2022": [100, 31],
                "2021": [100, 29],
                "2020": [100, 30],
            },
            index=["Total Revenue", "Gross Profit"],
        )
        cashflow = pd.DataFrame()

        signals = fetcher._calculate_moat_signals(financials, cashflow, "TEST")

        assert "moat_marginStability" in signals
        assert signals["moat_marginStability"] == "HIGH"
        assert signals["moat_grossMarginCV"] < 0.08
        assert signals["moat_grossMarginYears"] == 5

    def test_uses_sample_stdev_not_population(self):
        """Test that calculation uses sample stdev (N-1), not population (N)."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        margins_raw = [0.30, 0.32, 0.31, 0.29, 0.30]
        expected_stdev = statistics.stdev(margins_raw)  # N-1

        financials = pd.DataFrame(
            {
                "2024": [100, 30],
                "2023": [100, 32],
                "2022": [100, 31],
                "2021": [100, 29],
                "2020": [100, 30],
            },
            index=["Total Revenue", "Gross Profit"],
        )
        cashflow = pd.DataFrame()

        signals = fetcher._calculate_moat_signals(financials, cashflow, "TEST")

        # CV = stdev / mean
        mean_margin = statistics.mean(margins_raw)
        expected_cv = expected_stdev / mean_margin

        assert signals["moat_grossMarginCV"] == pytest.approx(expected_cv, rel=0.01)

    def test_volatile_margins_get_low_stability(self):
        """Test that volatile margins get LOW stability rating."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        # Volatile margins: [40%, 25%, 35%, 20%, 30%]
        financials = pd.DataFrame(
            {
                "2024": [100, 40],
                "2023": [100, 25],
                "2022": [100, 35],
                "2021": [100, 20],
                "2020": [100, 30],
            },
            index=["Total Revenue", "Gross Profit"],
        )
        cashflow = pd.DataFrame()

        signals = fetcher._calculate_moat_signals(financials, cashflow, "TEST")

        assert signals.get("moat_marginStability") == "LOW"
        assert signals["moat_grossMarginCV"] > 0.15

    def test_cfo_ni_ratio_strong(self):
        """Test CFO/NI ratio > 0.90 results in STRONG signal."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        financials = pd.DataFrame(
            {"2024": [100], "2023": [100], "2022": [100]},
            index=["Net Income"],
        )
        cashflow = pd.DataFrame(
            {"2024": [95], "2023": [98], "2022": [92]},
            index=["Operating Cash Flow"],
        )

        signals = fetcher._calculate_moat_signals(financials, cashflow, "TEST")

        assert signals["moat_cashConversion"] == "STRONG"
        assert signals["moat_cfoToNiAvg"] > 0.90

    def test_cfo_ni_ratio_above_one_is_valid(self):
        """Test that CFO/NI ratio > 1.0 is valid (not treated as error)."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        # CFO exceeds NI (common with high D&A)
        financials = pd.DataFrame(
            {"2024": [100], "2023": [100], "2022": [100]},
            index=["Net Income"],
        )
        cashflow = pd.DataFrame(
            {"2024": [150], "2023": [140], "2022": [145]},  # CFO > NI
            index=["Operating Cash Flow"],
        )

        signals = fetcher._calculate_moat_signals(financials, cashflow, "TEST")

        assert signals["moat_cashConversion"] == "STRONG"
        assert signals["moat_cfoToNiAvg"] > 1.0  # Should be ~1.45

    def test_weak_cash_conversion(self):
        """Test weak cash conversion detection."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        financials = pd.DataFrame(
            {"2024": [100], "2023": [100], "2022": [100]},
            index=["Net Income"],
        )
        cashflow = pd.DataFrame(
            {"2024": [50], "2023": [60], "2022": [55]},
            index=["Operating Cash Flow"],
        )

        signals = fetcher._calculate_moat_signals(financials, cashflow, "TEST")

        assert signals["moat_cashConversion"] == "WEAK"
        assert signals["moat_cfoToNiAvg"] < 0.70

    def test_insufficient_data_returns_empty(self):
        """Test that insufficient data returns empty signals."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        # Only 2 years - not enough for margin calculation
        financials = pd.DataFrame(
            {"2024": [100, 30], "2023": [100, 32]},
            index=["Total Revenue", "Gross Profit"],
        )
        cashflow = pd.DataFrame()

        signals = fetcher._calculate_moat_signals(financials, cashflow, "TEST")

        assert "moat_marginStability" not in signals


class TestMoatSignalExtraction:
    """Test extraction of moat signals from DATA_BLOCK."""

    def test_extract_complete_moat_signals(self):
        """Test extraction from complete DATA_BLOCK with moat signals."""
        report = """
### --- START DATA_BLOCK ---
SECTOR: Information Technology
ADJUSTED_HEALTH_SCORE: 75%
MOAT_MARGIN_STABILITY: HIGH
MOAT_MARGIN_CV: 0.0456
MOAT_GROSS_MARGIN_AVG: 35.2%
MOAT_CASH_CONVERSION: STRONG
MOAT_CFO_NI_AVG: 0.95
### --- END DATA_BLOCK ---
"""
        metrics = RedFlagDetector.extract_moat_signals(report)

        assert metrics["margin_stability"] == "HIGH"
        assert metrics["margin_cv"] == pytest.approx(0.0456, rel=0.01)
        assert metrics["margin_avg"] == pytest.approx(0.352, rel=0.01)
        assert metrics["cash_conversion"] == "STRONG"
        assert metrics["cfo_ni_avg"] == pytest.approx(0.95, rel=0.01)

    def test_extract_cfo_ni_above_one(self):
        """Test extraction handles CFO/NI ratio > 1.0 correctly (no percentage conversion)."""
        report = """
### --- START DATA_BLOCK ---
MOAT_CASH_CONVERSION: STRONG
MOAT_CFO_NI_AVG: 1.45
### --- END DATA_BLOCK ---
"""
        metrics = RedFlagDetector.extract_moat_signals(report)

        # Should NOT convert 1.45 to 0.0145
        assert metrics["cfo_ni_avg"] == pytest.approx(1.45, rel=0.01)
        assert metrics["cash_conversion"] == "STRONG"

    def test_extract_handles_trailing_punctuation(self):
        """Test regex handles trailing commas and spaces."""
        report = """
### --- START DATA_BLOCK ---
MOAT_MARGIN_CV: 0.045,
MOAT_CFO_NI_AVG: 0.92
### --- END DATA_BLOCK ---
"""
        metrics = RedFlagDetector.extract_moat_signals(report)

        assert metrics["margin_cv"] == pytest.approx(0.045, rel=0.01)
        assert metrics["cfo_ni_avg"] == pytest.approx(0.92, rel=0.01)

    def test_extract_handles_missing_signals(self):
        """Test extraction when moat signals are N/A."""
        report = """
### --- START DATA_BLOCK ---
MOAT_MARGIN_STABILITY: N/A
MOAT_CASH_CONVERSION: N/A
### --- END DATA_BLOCK ---
"""
        metrics = RedFlagDetector.extract_moat_signals(report)

        assert metrics["margin_stability"] is None
        assert metrics["cash_conversion"] is None

    def test_extract_handles_empty_report(self):
        """Test extraction from empty report."""
        metrics = RedFlagDetector.extract_moat_signals("")
        assert metrics["margin_stability"] is None
        assert metrics["cash_conversion"] is None

    def test_extract_handles_none_report(self):
        """Test extraction from None report."""
        metrics = RedFlagDetector.extract_moat_signals(None)
        assert metrics["margin_stability"] is None

    def test_extract_handles_non_string_report(self):
        """Test extraction handles non-string input gracefully."""
        metrics = RedFlagDetector.extract_moat_signals(["not", "a", "string"])
        # Should not raise, should return defaults
        assert metrics["margin_stability"] is None

    def test_extract_uses_last_data_block(self):
        """Test that extraction uses the LAST DATA_BLOCK (self-correction pattern)."""
        report = """
### --- START DATA_BLOCK ---
MOAT_MARGIN_STABILITY: LOW
### --- END DATA_BLOCK ---

[Agent recalculates...]

### --- START DATA_BLOCK ---
MOAT_MARGIN_STABILITY: HIGH
### --- END DATA_BLOCK ---
"""
        metrics = RedFlagDetector.extract_moat_signals(report)
        assert metrics["margin_stability"] == "HIGH"


class TestMoatFlagDetection:
    """Test moat flag (bonus) detection."""

    def test_durable_advantage_flag(self):
        """Test MOAT_DURABLE_ADVANTAGE flag when both signals strong."""
        report = """
### --- START DATA_BLOCK ---
MOAT_MARGIN_STABILITY: HIGH
MOAT_MARGIN_CV: 0.05
MOAT_CASH_CONVERSION: STRONG
MOAT_CFO_NI_AVG: 0.95
### --- END DATA_BLOCK ---
"""
        flags = RedFlagDetector.detect_moat_flags(report, "TEST.T")

        assert len(flags) == 1
        assert flags[0]["type"] == "MOAT_DURABLE_ADVANTAGE"
        assert flags[0]["risk_penalty"] == -1.0
        assert flags[0]["action"] == "RISK_BONUS"

    def test_durable_advantage_does_not_stack(self):
        """Test that durable advantage doesn't stack with individual bonuses."""
        report = """
### --- START DATA_BLOCK ---
MOAT_MARGIN_STABILITY: HIGH
MOAT_CASH_CONVERSION: STRONG
### --- END DATA_BLOCK ---
"""
        flags = RedFlagDetector.detect_moat_flags(report, "TEST.T")

        # Should only get durable advantage, not durable + pricing + quality
        assert len(flags) == 1
        assert flags[0]["type"] == "MOAT_DURABLE_ADVANTAGE"
        total = sum(f["risk_penalty"] for f in flags)
        assert total == -1.0  # Not -2.0

    def test_pricing_power_flag_alone(self):
        """Test MOAT_PRICING_POWER flag when only margin stability is HIGH."""
        report = """
### --- START DATA_BLOCK ---
MOAT_MARGIN_STABILITY: HIGH
MOAT_CASH_CONVERSION: ADEQUATE
### --- END DATA_BLOCK ---
"""
        flags = RedFlagDetector.detect_moat_flags(report, "TEST.T")

        assert len(flags) == 1
        assert flags[0]["type"] == "MOAT_PRICING_POWER"
        assert flags[0]["risk_penalty"] == -0.5

    def test_earnings_quality_flag_alone(self):
        """Test MOAT_EARNINGS_QUALITY flag when only cash conversion is STRONG."""
        report = """
### --- START DATA_BLOCK ---
MOAT_MARGIN_STABILITY: MEDIUM
MOAT_CASH_CONVERSION: STRONG
### --- END DATA_BLOCK ---
"""
        flags = RedFlagDetector.detect_moat_flags(report, "TEST.T")

        assert len(flags) == 1
        assert flags[0]["type"] == "MOAT_EARNINGS_QUALITY"
        assert flags[0]["risk_penalty"] == -0.5

    def test_both_individual_bonuses(self):
        """Test individual bonus when one signal strong but not both."""
        report_pricing = """
### --- START DATA_BLOCK ---
MOAT_MARGIN_STABILITY: HIGH
MOAT_CASH_CONVERSION: WEAK
### --- END DATA_BLOCK ---
"""
        flags = RedFlagDetector.detect_moat_flags(report_pricing, "TEST.T")
        assert len(flags) == 1
        assert flags[0]["type"] == "MOAT_PRICING_POWER"

    def test_no_flags_for_weak_signals(self):
        """Test no flags when signals are weak."""
        report = """
### --- START DATA_BLOCK ---
MOAT_MARGIN_STABILITY: LOW
MOAT_CASH_CONVERSION: WEAK
### --- END DATA_BLOCK ---
"""
        flags = RedFlagDetector.detect_moat_flags(report, "TEST.T")
        assert len(flags) == 0

    def test_no_flags_for_medium_signals(self):
        """Test no flags when signals are only MEDIUM/ADEQUATE."""
        report = """
### --- START DATA_BLOCK ---
MOAT_MARGIN_STABILITY: MEDIUM
MOAT_CASH_CONVERSION: ADEQUATE
### --- END DATA_BLOCK ---
"""
        flags = RedFlagDetector.detect_moat_flags(report, "TEST.T")
        assert len(flags) == 0

    def test_flag_detection_uses_categorical_not_numeric(self):
        """Test that flag detection uses categorical signals, not numeric values."""
        # Even if numeric parsing fails, categorical should work
        report = """
### --- START DATA_BLOCK ---
MOAT_MARGIN_STABILITY: HIGH
MOAT_MARGIN_CV: invalid_number
MOAT_CASH_CONVERSION: STRONG
MOAT_CFO_NI_AVG: also_invalid
### --- END DATA_BLOCK ---
"""
        flags = RedFlagDetector.detect_moat_flags(report, "TEST.T")

        # Should still detect durable advantage based on categorical signals
        assert len(flags) == 1
        assert flags[0]["type"] == "MOAT_DURABLE_ADVANTAGE"

    def test_handles_empty_report(self):
        """Test graceful handling of empty report."""
        flags = RedFlagDetector.detect_moat_flags("", "TEST")
        assert flags == []

    def test_handles_none_report(self):
        """Test graceful handling of None report."""
        flags = RedFlagDetector.detect_moat_flags(None, "TEST")
        assert flags == []


class TestMoatRiskTallyIntegration:
    """Test integration with Portfolio Manager risk tally."""

    def test_moat_bonus_offsets_risk(self):
        """Test that moat bonuses offset other risks in tally."""
        risk_flags = [
            {"type": "ADR_MODERATE_CONCERN", "risk_penalty": 0.33},
            {"type": "VALUE_TRAP_MODERATE_RISK", "risk_penalty": 0.5},
        ]

        moat_report = """
### --- START DATA_BLOCK ---
MOAT_MARGIN_STABILITY: HIGH
MOAT_CASH_CONVERSION: STRONG
### --- END DATA_BLOCK ---
"""
        moat_flags = RedFlagDetector.detect_moat_flags(moat_report, "TEST")
        all_flags = risk_flags + moat_flags

        total_risk = sum(f.get("risk_penalty", 0) for f in all_flags)
        # 0.33 + 0.5 - 1.0 = -0.17
        assert total_risk == pytest.approx(-0.17, abs=0.01)

    def test_moat_can_make_net_risk_negative(self):
        """Test that moat bonus can result in negative net risk (good)."""
        report = """
### --- START DATA_BLOCK ---
MOAT_MARGIN_STABILITY: HIGH
MOAT_CASH_CONVERSION: STRONG
### --- END DATA_BLOCK ---
"""
        flags = RedFlagDetector.detect_moat_flags(report, "TEST")
        total = sum(f.get("risk_penalty", 0) for f in flags)

        assert total < 0  # Negative risk = bonus


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_negative_net_income_excluded(self):
        """Test that loss-making years are excluded from CFO/NI calculation."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        financials = pd.DataFrame(
            {"2024": [100], "2023": [-50], "2022": [100]},  # Loss year
            index=["Net Income"],
        )
        cashflow = pd.DataFrame(
            {"2024": [95], "2023": [30], "2022": [92]},
            index=["Operating Cash Flow"],
        )

        signals = fetcher._calculate_moat_signals(financials, cashflow, "TEST")

        # Should only use profitable years (2024, 2022)
        assert signals.get("moat_cfoToNiYears", 0) == 2

    def test_handles_empty_dataframes(self):
        """Test handling of empty DataFrames."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        signals = fetcher._calculate_moat_signals(
            pd.DataFrame(), pd.DataFrame(), "TEST"
        )
        assert signals == {}

    def test_handles_missing_rows(self):
        """Test graceful handling of missing financial statement rows."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        # Missing Gross Profit row
        financials = pd.DataFrame(
            {"2024": [100], "2023": [100], "2022": [100]},
            index=["Total Revenue"],
        )
        cashflow = pd.DataFrame()

        # Should not raise
        signals = fetcher._calculate_moat_signals(financials, cashflow, "TEST")
        assert "moat_marginStability" not in signals

    def test_handles_zero_revenue(self):
        """Test handling of zero revenue (division by zero)."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        financials = pd.DataFrame(
            {
                "2024": [0, 30],  # Zero revenue
                "2023": [100, 32],
                "2022": [100, 31],
                "2021": [100, 29],
            },
            index=["Total Revenue", "Gross Profit"],
        )
        cashflow = pd.DataFrame()

        # Should not raise, should skip zero revenue year
        signals = fetcher._calculate_moat_signals(financials, cashflow, "TEST")
        # Should still calculate from valid years (only 3 valid years)
        assert signals.get("moat_grossMarginYears", 0) == 3

    def test_handles_nan_values(self):
        """Test handling of NaN values in financial data."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        financials = pd.DataFrame(
            {
                "2024": [100, 30],
                "2023": [float("nan"), float("nan")],
                "2022": [100, 31],
                "2021": [100, 29],
            },
            index=["Total Revenue", "Gross Profit"],
        )
        cashflow = pd.DataFrame()

        # Should not raise
        signals = fetcher._calculate_moat_signals(financials, cashflow, "TEST")
        # Should calculate from valid years
        assert signals.get("moat_grossMarginYears", 0) == 3

    def test_extreme_margin_values_filtered(self):
        """Test that unrealistic margin values are filtered."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        financials = pd.DataFrame(
            {
                "2024": [100, 30],
                "2023": [100, 150],  # 150% margin - unrealistic
                "2022": [100, 31],
                "2021": [100, 29],
            },
            index=["Total Revenue", "Gross Profit"],
        )
        cashflow = pd.DataFrame()

        signals = fetcher._calculate_moat_signals(financials, cashflow, "TEST")

        # Unrealistic margin (1.5) should be filtered, so only 3 valid years
        assert signals.get("moat_grossMarginYears", 0) == 3

    def test_extreme_cfo_ni_ratio_filtered(self):
        """Test that extreme CFO/NI ratios are filtered."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        financials = pd.DataFrame(
            {"2024": [100], "2023": [1], "2022": [100]},  # Very small NI
            index=["Net Income"],
        )
        cashflow = pd.DataFrame(
            {"2024": [95], "2023": [100], "2022": [92]},  # Would be 100x ratio
            index=["Operating Cash Flow"],
        )

        signals = fetcher._calculate_moat_signals(financials, cashflow, "TEST")

        # 100x ratio should be filtered, only 2 valid years
        assert signals.get("moat_cfoToNiYears", 0) == 2
