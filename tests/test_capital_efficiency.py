"""
Tests for Capital Efficiency Detection (ROIC/Leverage Quality).

Separate from moat signals - these checks detect:
1. Value destruction (negative ROIC with positive ROE)
2. Leverage-engineered returns (ROE >> ROIC)
3. Below-hurdle ROIC (likely destroying value)

Run with: pytest tests/test_capital_efficiency.py -v
"""

import pandas as pd
import pytest

from src.validators.red_flag_detector import RedFlagDetector


class TestCapitalEfficiencyCalculation:
    """Test ROIC calculation in fetcher."""

    def test_roic_calculation_positive(self):
        """Test ROIC calculation with normal positive values."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        # EBIT: 100M, Tax Rate: 20%, Invested Capital: 500M
        # NOPAT = 100 * 0.8 = 80M, ROIC = 80/500 = 16%
        income_stmt = pd.DataFrame(
            {"2024": [100_000_000, 0.20]},
            index=["EBIT", "Tax Rate For Calcs"],
        )
        balance_sheet = pd.DataFrame(
            {"2024": [500_000_000]},
            index=["Invested Capital"],
        )
        info = {"returnOnAssets": 0.10, "returnOnEquity": 0.18}

        signals = fetcher._calculate_capital_efficiency_signals(
            income_stmt, balance_sheet, info, "TEST"
        )

        assert signals["capital_roic"] == pytest.approx(0.16, rel=0.01)
        assert signals["capital_roicQuality"] == "STRONG"
        assert signals["capital_leverageQuality"] == "GENUINE"

    def test_roic_calculation_negative(self):
        """Test ROIC calculation with negative EBIT."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        income_stmt = pd.DataFrame(
            {"2024": [-50_000_000, 0.21]},
            index=["EBIT", "Tax Rate For Calcs"],
        )
        balance_sheet = pd.DataFrame(
            {"2024": [500_000_000]},
            index=["Invested Capital"],
        )
        info = {"returnOnAssets": -0.05, "returnOnEquity": 0.10}

        signals = fetcher._calculate_capital_efficiency_signals(
            income_stmt, balance_sheet, info, "TEST"
        )

        assert signals["capital_roic"] < 0
        assert signals["capital_roicQuality"] == "DESTRUCTIVE"
        assert signals["capital_leverageQuality"] == "VALUE_DESTRUCTION"

    def test_roic_weak_below_hurdle(self):
        """Test ROIC below 8% hurdle rate classified as WEAK."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        # ROIC = 5%
        income_stmt = pd.DataFrame(
            {"2024": [31_250_000, 0.20]},
            index=["EBIT", "Tax Rate For Calcs"],
        )
        balance_sheet = pd.DataFrame(
            {"2024": [500_000_000]},
            index=["Invested Capital"],
        )
        info = {"returnOnAssets": 0.03, "returnOnEquity": 0.06}

        signals = fetcher._calculate_capital_efficiency_signals(
            income_stmt, balance_sheet, info, "TEST"
        )

        assert signals["capital_roic"] == pytest.approx(0.05, rel=0.01)
        assert signals["capital_roicQuality"] == "WEAK"
        assert signals["capital_hurdleSpread"] < 0

    def test_leverage_engineered_high_ratio(self):
        """Test ROE/ROIC > 3 classified as ENGINEERED."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        # ROIC = 8%, ROE = 30% -> ratio = 3.75
        income_stmt = pd.DataFrame(
            {"2024": [50_000_000, 0.20]},
            index=["EBIT", "Tax Rate For Calcs"],
        )
        balance_sheet = pd.DataFrame(
            {"2024": [500_000_000]},
            index=["Invested Capital"],
        )
        info = {"returnOnAssets": 0.05, "returnOnEquity": 0.30}

        signals = fetcher._calculate_capital_efficiency_signals(
            income_stmt, balance_sheet, info, "TEST"
        )

        assert signals["capital_leverageQuality"] == "ENGINEERED"
        assert signals["capital_roeRoicRatio"] > 3.0

    def test_leverage_suspect_moderate_ratio(self):
        """Test ROE/ROIC 2-3 classified as SUSPECT."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        # ROIC = 10%, ROE = 24% -> ratio = 2.4
        income_stmt = pd.DataFrame(
            {"2024": [62_500_000, 0.20]},
            index=["EBIT", "Tax Rate For Calcs"],
        )
        balance_sheet = pd.DataFrame(
            {"2024": [500_000_000]},
            index=["Invested Capital"],
        )
        info = {"returnOnAssets": 0.06, "returnOnEquity": 0.24}

        signals = fetcher._calculate_capital_efficiency_signals(
            income_stmt, balance_sheet, info, "TEST"
        )

        assert signals["capital_leverageQuality"] == "SUSPECT"
        assert 2.0 < signals["capital_roeRoicRatio"] < 3.0

    def test_conservative_roic_exceeds_roe(self):
        """Test ROIC > ROE classified as CONSERVATIVE."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        # ROIC = 20%, ROE = 15% -> under-leveraged
        income_stmt = pd.DataFrame(
            {"2024": [125_000_000, 0.20]},
            index=["EBIT", "Tax Rate For Calcs"],
        )
        balance_sheet = pd.DataFrame(
            {"2024": [500_000_000]},
            index=["Invested Capital"],
        )
        info = {"returnOnAssets": 0.12, "returnOnEquity": 0.15}

        signals = fetcher._calculate_capital_efficiency_signals(
            income_stmt, balance_sheet, info, "TEST"
        )

        assert signals["capital_leverageQuality"] == "CONSERVATIVE"
        assert signals["capital_roeRoicRatio"] < 1.0

    def test_default_tax_rate_when_missing(self):
        """Test 21% default tax rate when not available."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        income_stmt = pd.DataFrame(
            {"2024": [100_000_000]},
            index=["EBIT"],
        )
        balance_sheet = pd.DataFrame(
            {"2024": [500_000_000]},
            index=["Invested Capital"],
        )
        info = {"returnOnAssets": 0.10, "returnOnEquity": 0.16}

        signals = fetcher._calculate_capital_efficiency_signals(
            income_stmt, balance_sheet, info, "TEST"
        )

        # NOPAT = 100M * 0.79 = 79M, ROIC = 79/500 = 15.8%
        assert signals["capital_roic"] == pytest.approx(0.158, rel=0.01)

    def test_empty_dataframes_return_empty(self):
        """Test empty DataFrames return empty signals."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        signals = fetcher._calculate_capital_efficiency_signals(
            pd.DataFrame(), pd.DataFrame(), {}, "TEST"
        )

        assert signals == {}


class TestCapitalEfficiencyExtraction:
    """Test DATA_BLOCK extraction for capital efficiency."""

    def test_extract_complete_signals(self):
        """Test extraction of all capital efficiency fields."""
        report = """
        ### --- START DATA_BLOCK ---
        ROIC_PERCENT: 12.50
        ROIC_QUALITY: ADEQUATE
        LEVERAGE_QUALITY: GENUINE
        ROE_ROIC_RATIO: 1.45
        ### --- END DATA_BLOCK ---
        """

        metrics = RedFlagDetector.extract_capital_efficiency_signals(report)

        assert metrics["roic"] == pytest.approx(0.125, rel=0.01)
        assert metrics["roic_quality"] == "ADEQUATE"
        assert metrics["leverage_quality"] == "GENUINE"
        assert metrics["roe_roic_ratio"] == pytest.approx(1.45, rel=0.01)

    def test_extract_value_destruction(self):
        """Test extraction of VALUE_DESTRUCTION leverage quality."""
        report = """
        ### --- START DATA_BLOCK ---
        ROIC_PERCENT: -4.10
        ROIC_QUALITY: DESTRUCTIVE
        LEVERAGE_QUALITY: VALUE_DESTRUCTION
        ### --- END DATA_BLOCK ---
        """

        metrics = RedFlagDetector.extract_capital_efficiency_signals(report)

        assert metrics["roic"] < 0
        assert metrics["roic_quality"] == "DESTRUCTIVE"
        assert metrics["leverage_quality"] == "VALUE_DESTRUCTION"

    def test_extract_handles_percentage_format(self):
        """Test extraction handles both 12.5 and 12.5% formats."""
        report = """
        ### --- START DATA_BLOCK ---
        ROIC_PERCENT: 18.5%
        ROIC_QUALITY: STRONG
        ### --- END DATA_BLOCK ---
        """

        metrics = RedFlagDetector.extract_capital_efficiency_signals(report)

        assert metrics["roic"] == pytest.approx(0.185, rel=0.01)

    def test_extract_handles_normalization_heuristics(self):
        """Test the heuristic normalization rules for ROIC."""
        # Case 1: Implied percentage (val >= 2.0)
        report_percent = "### --- START DATA_BLOCK ---\nROIC_PERCENT: 15.0\n### --- END DATA_BLOCK ---"
        metrics = RedFlagDetector.extract_capital_efficiency_signals(report_percent)
        assert metrics["roic"] == pytest.approx(0.15, rel=0.01)

        # Case 2: High ROIC decimal (val < 2.0)
        report_decimal = "### --- START DATA_BLOCK ---\nROIC_PERCENT: 1.5\n### --- END DATA_BLOCK ---"
        metrics = RedFlagDetector.extract_capital_efficiency_signals(report_decimal)
        assert metrics["roic"] == pytest.approx(1.5, rel=0.01)

        # Case 3: Explicit percentage (5.0%)
        report_explicit = "### --- START DATA_BLOCK ---\nROIC_PERCENT: 5.0%\n### --- END DATA_BLOCK ---"
        metrics = RedFlagDetector.extract_capital_efficiency_signals(report_explicit)
        assert metrics["roic"] == pytest.approx(0.05, rel=0.01)

        # Case 4: Small decimal (0.05)
        report_small = "### --- START DATA_BLOCK ---\nROIC_PERCENT: 0.05\n### --- END DATA_BLOCK ---"
        metrics = RedFlagDetector.extract_capital_efficiency_signals(report_small)
        assert metrics["roic"] == pytest.approx(0.05, rel=0.01)

    def test_extract_handles_na_values(self):
        """Test extraction skips N/A values."""
        report = """
        ### --- START DATA_BLOCK ---
        ROIC_PERCENT: N/A
        ROIC_QUALITY: N/A
        LEVERAGE_QUALITY: N/A
        ### --- END DATA_BLOCK ---
        """

        metrics = RedFlagDetector.extract_capital_efficiency_signals(report)

        assert "roic" not in metrics
        assert "roic_quality" not in metrics
        assert "leverage_quality" not in metrics

    def test_extract_empty_report(self):
        """Test empty report returns empty dict."""
        assert RedFlagDetector.extract_capital_efficiency_signals("") == {}
        assert RedFlagDetector.extract_capital_efficiency_signals(None) == {}

    def test_extract_uses_last_data_block(self):
        """Test extraction uses last DATA_BLOCK if multiple present."""
        report = """
        ### --- START DATA_BLOCK ---
        ROIC_QUALITY: WEAK
        ### --- END DATA_BLOCK ---

        Updated analysis:

        ### --- START DATA_BLOCK ---
        ROIC_QUALITY: STRONG
        ### --- END DATA_BLOCK ---
        """

        metrics = RedFlagDetector.extract_capital_efficiency_signals(report)

        assert metrics["roic_quality"] == "STRONG"


class TestCapitalEfficiencyFlagDetection:
    """Test flag detection for capital efficiency patterns."""

    def test_value_destruction_flag(self):
        """Test VALUE_DESTRUCTION pattern triggers critical flag."""
        report = """
        ### --- START DATA_BLOCK ---
        ROIC_PERCENT: -4.10
        ROIC_QUALITY: DESTRUCTIVE
        LEVERAGE_QUALITY: VALUE_DESTRUCTION
        ### --- END DATA_BLOCK ---
        """

        flags = RedFlagDetector.detect_capital_efficiency_flags(report, "SUZ")

        assert len(flags) == 1
        assert flags[0]["type"] == "CAPITAL_VALUE_DESTRUCTION"
        assert flags[0]["risk_penalty"] == 1.5
        assert flags[0]["severity"] == "CRITICAL"

    def test_value_destruction_blocks_other_flags(self):
        """Test VALUE_DESTRUCTION doesn't stack with other flags."""
        report = """
        ### --- START DATA_BLOCK ---
        ROIC_PERCENT: -4.10
        ROIC_QUALITY: DESTRUCTIVE
        LEVERAGE_QUALITY: VALUE_DESTRUCTION
        ### --- END DATA_BLOCK ---
        """

        flags = RedFlagDetector.detect_capital_efficiency_flags(report, "TEST")

        # Should only have VALUE_DESTRUCTION, not WEAK ROIC
        assert len(flags) == 1
        assert flags[0]["type"] == "CAPITAL_VALUE_DESTRUCTION"

    def test_engineered_returns_flag(self):
        """Test ENGINEERED leverage quality triggers flag."""
        report = """
        ### --- START DATA_BLOCK ---
        ROIC_PERCENT: 7.30
        ROIC_QUALITY: WEAK
        LEVERAGE_QUALITY: ENGINEERED
        ROE_ROIC_RATIO: 3.40
        ### --- END DATA_BLOCK ---
        """

        flags = RedFlagDetector.detect_capital_efficiency_flags(report, "9984.T")

        flag_types = [f["type"] for f in flags]
        assert "CAPITAL_ENGINEERED_RETURNS" in flag_types
        assert "CAPITAL_BELOW_HURDLE" in flag_types

        engineered = next(f for f in flags if f["type"] == "CAPITAL_ENGINEERED_RETURNS")
        assert engineered["risk_penalty"] == 1.0

    def test_suspect_returns_flag(self):
        """Test SUSPECT leverage quality triggers flag."""
        report = """
        ### --- START DATA_BLOCK ---
        ROIC_PERCENT: 10.00
        ROIC_QUALITY: ADEQUATE
        LEVERAGE_QUALITY: SUSPECT
        ROE_ROIC_RATIO: 2.40
        ### --- END DATA_BLOCK ---
        """

        flags = RedFlagDetector.detect_capital_efficiency_flags(report, "TEST")

        assert len(flags) == 1
        assert flags[0]["type"] == "CAPITAL_SUSPECT_RETURNS"
        assert flags[0]["risk_penalty"] == 0.5

    def test_below_hurdle_flag(self):
        """Test WEAK ROIC triggers below-hurdle flag."""
        report = """
        ### --- START DATA_BLOCK ---
        ROIC_PERCENT: 5.00
        ROIC_QUALITY: WEAK
        LEVERAGE_QUALITY: GENUINE
        ROE_ROIC_RATIO: 1.20
        ### --- END DATA_BLOCK ---
        """

        flags = RedFlagDetector.detect_capital_efficiency_flags(report, "TEST")

        assert len(flags) == 1
        assert flags[0]["type"] == "CAPITAL_BELOW_HURDLE"
        assert flags[0]["risk_penalty"] == 0.5

    def test_capital_efficient_bonus(self):
        """Test STRONG + GENUINE triggers bonus flag."""
        report = """
        ### --- START DATA_BLOCK ---
        ROIC_PERCENT: 20.00
        ROIC_QUALITY: STRONG
        LEVERAGE_QUALITY: GENUINE
        ROE_ROIC_RATIO: 1.30
        ### --- END DATA_BLOCK ---
        """

        flags = RedFlagDetector.detect_capital_efficiency_flags(report, "META")

        assert len(flags) == 1
        assert flags[0]["type"] == "CAPITAL_EFFICIENT"
        assert flags[0]["risk_penalty"] == -0.5
        assert flags[0]["severity"] == "POSITIVE"

    def test_capital_efficient_with_conservative(self):
        """Test STRONG + CONSERVATIVE also triggers bonus."""
        report = """
        ### --- START DATA_BLOCK ---
        ROIC_PERCENT: 22.00
        ROIC_QUALITY: STRONG
        LEVERAGE_QUALITY: CONSERVATIVE
        ROE_ROIC_RATIO: 0.85
        ### --- END DATA_BLOCK ---
        """

        flags = RedFlagDetector.detect_capital_efficiency_flags(report, "TEST")

        assert len(flags) == 1
        assert flags[0]["type"] == "CAPITAL_EFFICIENT"

    def test_no_flags_for_adequate_genuine(self):
        """Test ADEQUATE + GENUINE triggers no flags."""
        report = """
        ### --- START DATA_BLOCK ---
        ROIC_PERCENT: 12.00
        ROIC_QUALITY: ADEQUATE
        LEVERAGE_QUALITY: GENUINE
        ROE_ROIC_RATIO: 1.50
        ### --- END DATA_BLOCK ---
        """

        flags = RedFlagDetector.detect_capital_efficiency_flags(report, "TEST")

        assert len(flags) == 0

    def test_empty_report_no_flags(self):
        """Test empty report returns no flags."""
        assert RedFlagDetector.detect_capital_efficiency_flags("", "TEST") == []
        assert RedFlagDetector.detect_capital_efficiency_flags(None, "TEST") == []


class TestRealWorldScenarios:
    """Test with real-world-like scenarios."""

    def test_suzano_value_destruction(self):
        """Test SUZ-like pattern: negative ROIC, positive ROE."""
        report = """
        Analysis for SUZ (Suzano S.A.)

        ### --- START DATA_BLOCK ---
        SECTOR: Shipping/Commodities
        ROIC_PERCENT: -4.10
        ROIC_QUALITY: DESTRUCTIVE
        LEVERAGE_QUALITY: VALUE_DESTRUCTION
        DE_RATIO: 220.50
        ROA_PERCENT: 4.50
        ### --- END DATA_BLOCK ---
        """

        flags = RedFlagDetector.detect_capital_efficiency_flags(report, "SUZ")

        assert len(flags) == 1
        assert flags[0]["type"] == "CAPITAL_VALUE_DESTRUCTION"
        assert flags[0]["risk_penalty"] == 1.5

    def test_softbank_leverage_engineering(self):
        """Test SoftBank-like pattern: ROE/ROIC > 3."""
        report = """
        Analysis for 9984.T (SoftBank Group)

        ### --- START DATA_BLOCK ---
        SECTOR: General/Diversified
        ROIC_PERCENT: 7.30
        ROIC_QUALITY: WEAK
        LEVERAGE_QUALITY: ENGINEERED
        ROE_ROIC_RATIO: 3.41
        DE_RATIO: 119.30
        ### --- END DATA_BLOCK ---
        """

        flags = RedFlagDetector.detect_capital_efficiency_flags(report, "9984.T")

        flag_types = [f["type"] for f in flags]
        assert "CAPITAL_ENGINEERED_RETURNS" in flag_types
        assert "CAPITAL_BELOW_HURDLE" in flag_types

        total_penalty = sum(f["risk_penalty"] for f in flags)
        assert total_penalty == 1.5  # 1.0 + 0.5

    def test_meta_capital_efficient(self):
        """Test META-like pattern: high ROIC, low leverage."""
        report = """
        Analysis for META (Meta Platforms)

        ### --- START DATA_BLOCK ---
        SECTOR: Technology/Software
        ROIC_PERCENT: 29.80
        ROIC_QUALITY: STRONG
        LEVERAGE_QUALITY: GENUINE
        ROE_ROIC_RATIO: 1.09
        DE_RATIO: 26.30
        ### --- END DATA_BLOCK ---
        """

        flags = RedFlagDetector.detect_capital_efficiency_flags(report, "META")

        assert len(flags) == 1
        assert flags[0]["type"] == "CAPITAL_EFFICIENT"
        assert flags[0]["risk_penalty"] == -0.5

    def test_toyota_normal_capital_structure(self):
        """Test Toyota-like pattern: normal, no flags."""
        report = """
        Analysis for 7203.T (Toyota Motor)

        ### --- START DATA_BLOCK ---
        SECTOR: General/Diversified
        ROIC_PERCENT: 6.50
        ROIC_QUALITY: WEAK
        LEVERAGE_QUALITY: GENUINE
        ROE_ROIC_RATIO: 1.98
        DE_RATIO: 105.00
        ### --- END DATA_BLOCK ---
        """

        flags = RedFlagDetector.detect_capital_efficiency_flags(report, "7203.T")

        # Only below-hurdle flag (ROIC 6.5% < 8%)
        assert len(flags) == 1
        assert flags[0]["type"] == "CAPITAL_BELOW_HURDLE"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_invested_capital(self):
        """Test zero invested capital doesn't cause division error."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        income_stmt = pd.DataFrame(
            {"2024": [100_000_000, 0.20]},
            index=["EBIT", "Tax Rate For Calcs"],
        )
        balance_sheet = pd.DataFrame(
            {"2024": [0]},
            index=["Invested Capital"],
        )
        info = {"returnOnAssets": 0.10, "returnOnEquity": 0.15}

        signals = fetcher._calculate_capital_efficiency_signals(
            income_stmt, balance_sheet, info, "TEST"
        )

        # Should not crash, should return empty or partial
        assert "capital_roic" not in signals

    def test_negative_invested_capital(self):
        """Test negative invested capital handled gracefully."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        income_stmt = pd.DataFrame(
            {"2024": [100_000_000, 0.20]},
            index=["EBIT", "Tax Rate For Calcs"],
        )
        balance_sheet = pd.DataFrame(
            {"2024": [-500_000_000]},
            index=["Invested Capital"],
        )
        info = {"returnOnAssets": 0.10, "returnOnEquity": 0.15}

        signals = fetcher._calculate_capital_efficiency_signals(
            income_stmt, balance_sheet, info, "TEST"
        )

        assert "capital_roic" not in signals

    def test_extreme_tax_rate_clamped(self):
        """Test extreme tax rates are clamped to reasonable range."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        # Tax rate of 150% (invalid) should be clamped to 50%
        income_stmt = pd.DataFrame(
            {"2024": [100_000_000, 1.50]},
            index=["EBIT", "Tax Rate For Calcs"],
        )
        balance_sheet = pd.DataFrame(
            {"2024": [500_000_000]},
            index=["Invested Capital"],
        )
        info = {"returnOnAssets": 0.10, "returnOnEquity": 0.15}

        signals = fetcher._calculate_capital_efficiency_signals(
            income_stmt, balance_sheet, info, "TEST"
        )

        # ROIC = 100M * 0.5 / 500M = 10% (clamped at 50% tax)
        assert signals["capital_roic"] == pytest.approx(0.10, rel=0.01)

    def test_missing_roe_no_leverage_quality(self):
        """Test missing ROE means no leverage quality calculation."""
        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        income_stmt = pd.DataFrame(
            {"2024": [100_000_000, 0.20]},
            index=["EBIT", "Tax Rate For Calcs"],
        )
        balance_sheet = pd.DataFrame(
            {"2024": [500_000_000]},
            index=["Invested Capital"],
        )
        info = {"returnOnAssets": 0.10}  # No ROE

        signals = fetcher._calculate_capital_efficiency_signals(
            income_stmt, balance_sheet, info, "TEST"
        )

        assert "capital_roic" in signals
        assert "capital_leverageQuality" not in signals

    def test_nan_values_handled(self):
        """Test NaN values don't crash calculation."""
        import numpy as np

        from src.data.fetcher import SmartMarketDataFetcher

        fetcher = SmartMarketDataFetcher()

        income_stmt = pd.DataFrame(
            {"2024": [np.nan, 0.20]},
            index=["EBIT", "Tax Rate For Calcs"],
        )
        balance_sheet = pd.DataFrame(
            {"2024": [500_000_000]},
            index=["Invested Capital"],
        )
        info = {"returnOnAssets": 0.10, "returnOnEquity": 0.15}

        signals = fetcher._calculate_capital_efficiency_signals(
            income_stmt, balance_sheet, info, "TEST"
        )

        assert "capital_roic" not in signals


class TestConfigThresholds:
    """Test that config thresholds are properly used."""

    def test_uses_config_hurdle_rate(self):
        """Test calculation uses config hurdle rate."""
        from src.config import config

        # Default hurdle is 8%
        assert config.roic_hurdle_rate == 0.08

    def test_uses_config_strong_threshold(self):
        """Test calculation uses config strong threshold."""
        from src.config import config

        # Default strong threshold is 15%
        assert config.roic_strong_threshold == 0.15

    def test_uses_config_leverage_ratios(self):
        """Test calculation uses config leverage ratios."""
        from src.config import config

        assert config.leverage_suspect_ratio == 2.0
        assert config.leverage_engineered_ratio == 3.0
