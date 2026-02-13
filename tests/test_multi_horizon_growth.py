"""
Tests for Multi-Horizon Growth & TTM Data Pipeline.

Tests the quarterly-derived growth metrics (TTM, MRQ) that mitigate
"rearview mirror" bias by providing FY/TTM/MRQ time horizons.

Run with: pytest tests/test_multi_horizon_growth.py -v
"""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.fetcher import SmartMarketDataFetcher
from src.validators.red_flag_detector import RedFlagDetector

# --- Helper to build quarterly DataFrames ---


def _make_quarterly_income(
    revenues: list[float | None],
    net_incomes: list[float | None] | None = None,
    start_date: str = "2025-12-31",
    freq_months: int = 3,
) -> pd.DataFrame:
    """Build a quarterly income statement DataFrame mimicking yfinance format.

    Columns are DatetimeIndex (newest first), rows are metric names.
    """
    n = len(revenues)
    dates = pd.date_range(end=start_date, periods=n, freq=f"{freq_months}ME")[::-1]
    data = {"Total Revenue": revenues}
    if net_incomes is not None:
        data["Net Income"] = net_incomes
    return pd.DataFrame(data, index=dates).T


def _make_quarterly_cashflow(
    ocf: list[float | None],
    capex: list[float | None] | None = None,
    start_date: str = "2025-12-31",
    freq_months: int = 3,
) -> pd.DataFrame:
    """Build a quarterly cash-flow DataFrame mimicking yfinance format."""
    n = len(ocf)
    dates = pd.date_range(end=start_date, periods=n, freq=f"{freq_months}ME")[::-1]
    data = {"Operating Cash Flow": ocf}
    if capex is not None:
        data["Capital Expenditure"] = capex
    return pd.DataFrame(data, index=dates).T


def _make_mock_ticker(
    qt_inc: pd.DataFrame | None = None,
    qt_cf: pd.DataFrame | None = None,
) -> MagicMock:
    """Create a mock yfinance Ticker with quarterly properties."""
    mock = MagicMock()
    type(mock).quarterly_financials = PropertyMock(
        return_value=qt_inc if qt_inc is not None else pd.DataFrame()
    )
    type(mock).quarterly_cashflow = PropertyMock(
        return_value=qt_cf if qt_cf is not None else pd.DataFrame()
    )
    return mock


class TestQuarterlyHorizonExtraction:
    """Test _extract_quarterly_horizons directly."""

    @pytest.fixture
    def fetcher(self):
        return SmartMarketDataFetcher()

    # ---- Scenario tests ----

    def test_acceleration_scenario(self, fetcher):
        """Yamax-like: FY growth ~1% but recent quarter surging much more than TTM.

        For ACCELERATING, MRQ must exceed TTM by >10 percentage points.
        """
        # 8 quarters: recent quarter surges far ahead of trailing average
        # Q0..Q3 = 160, 112, 112, 112 (sum=496)
        # Q4..Q7 = 100, 100, 100, 100 (sum=400)
        # TTM growth = (496-400)/400 = 24%
        # MRQ YoY: Q0=160 vs Q4=100 → 60%
        # Delta = 60% - 24% = 36pp > 10pp → ACCELERATING
        revenues = [160, 112, 112, 112, 100, 100, 100, 100]
        qt_inc = _make_quarterly_income(revenues)
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        assert result.get("revenueGrowth_MRQ") == pytest.approx(0.60, abs=0.01)
        assert result.get("revenueGrowth_TTM") == pytest.approx(0.24, abs=0.01)
        assert result["growth_trajectory"] == "ACCELERATING"
        assert "latest_quarter_date" in result

    def test_deceleration_scenario(self, fetcher):
        """Strong FY but recent quarter drops."""
        # Prior 4 were strong, recent 4 are weak
        # Q0..Q3 = 90, 95, 100, 105 (sum=390)
        # Q4..Q7 = 110, 112, 115, 120 (sum=457)
        # TTM growth = (390-457)/457 ≈ -14.7%
        # MRQ YoY: Q0=90 vs Q4=110 → -18.2%
        revenues = [90, 95, 100, 105, 110, 112, 115, 120]
        qt_inc = _make_quarterly_income(revenues)
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        assert result.get("revenueGrowth_MRQ") is not None
        assert result["revenueGrowth_MRQ"] < 0  # Declining
        assert result.get("revenueGrowth_TTM") is not None
        assert result["revenueGrowth_TTM"] < 0
        # Delta between MRQ (-18%) and TTM (-15%) is only ~3pp → STABLE
        # (both are declining but MRQ is not >10pp different from TTM)
        assert result["growth_trajectory"] in ("STABLE", "DECELERATING")

    def test_stable_growth(self, fetcher):
        """Consistent 12% growth across all quarters."""
        # Each quarter is 12% above same quarter last year
        # Q0..Q3 = 112, 112, 112, 112 (sum=448)
        # Q4..Q7 = 100, 100, 100, 100 (sum=400)
        revenues = [112, 112, 112, 112, 100, 100, 100, 100]
        qt_inc = _make_quarterly_income(revenues)
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        assert result.get("revenueGrowth_MRQ") == pytest.approx(0.12, abs=0.01)
        assert result.get("revenueGrowth_TTM") == pytest.approx(0.12, abs=0.01)
        assert result["growth_trajectory"] == "STABLE"


class TestEdgeCases:
    """Test data gap and boundary conditions."""

    @pytest.fixture
    def fetcher(self):
        return SmartMarketDataFetcher()

    def test_insufficient_quarters_for_mrq(self, fetcher):
        """< 5 quarters: cannot compute MRQ YoY."""
        revenues = [100, 110, 120]  # Only 3 quarters
        qt_inc = _make_quarterly_income(revenues)
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        assert result.get("revenueGrowth_MRQ") is None
        assert result.get("revenueGrowth_TTM") is None
        # Date metadata should still be present
        assert "latest_quarter_date" in result

    def test_five_quarters_mrq_only(self, fetcher):
        """5 quarters: MRQ available but TTM needs 8."""
        revenues = [120, 110, 105, 100, 95]
        qt_inc = _make_quarterly_income(revenues)
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        assert result.get("revenueGrowth_MRQ") is not None
        assert result.get("revenueGrowth_TTM") is None  # Need 8 quarters
        assert result.get("growth_trajectory") is None  # Needs both MRQ and TTM

    def test_empty_quarterly_dataframe(self, fetcher):
        """Empty DataFrame: no crash, empty result."""
        mock_ticker = _make_mock_ticker()  # Defaults to empty DFs

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        assert result.get("revenueGrowth_MRQ") is None
        assert result.get("revenueGrowth_TTM") is None
        assert result.get("growth_trajectory") is None

    def test_missing_total_revenue_row(self, fetcher):
        """DataFrame has data but no 'Total Revenue' index label."""
        dates = pd.date_range(end="2025-12-31", periods=8, freq="3ME")[::-1]
        # Only has "Net Income", not "Total Revenue"
        qt_inc = pd.DataFrame(
            {"Net Income": [50, 45, 40, 35, 30, 28, 25, 22]},
            index=dates,
        ).T
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        # Revenue metrics should be None, but NI metrics should populate
        assert result.get("revenueGrowth_MRQ") is None
        assert result.get("revenueGrowth_TTM") is None
        assert result.get("netIncome_TTM") is not None

    def test_zero_denominator(self, fetcher):
        """Prior-year quarter revenue = 0: avoid division by zero."""
        # Q4 (the YoY comparison quarter) has zero revenue
        revenues = [100, 90, 80, 70, 0, 50, 40, 30]
        qt_inc = _make_quarterly_income(revenues)
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        # MRQ should be None (can't divide by zero)
        assert result.get("revenueGrowth_MRQ") is None
        # TTM denominator (sum of Q4-Q7) = 0+50+40+30 = 120 → should still work
        assert result.get("revenueGrowth_TTM") is not None

    def test_negative_denominator(self, fetcher):
        """Prior-year quarter revenue is negative: skip calculation."""
        revenues = [100, 90, 80, 70, -10, 50, 40, 30]
        qt_inc = _make_quarterly_income(revenues)
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        # MRQ skipped (negative prior quarter)
        assert result.get("revenueGrowth_MRQ") is None

    def test_nan_in_ttm_window(self, fetcher):
        """NaN in one of the 4 TTM quarters: TTM sum should be None."""
        revenues = [100, 90, float("nan"), 70, 95, 85, 75, 65]
        qt_inc = _make_quarterly_income(revenues)
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        # min_count=4 means NaN propagates
        assert result.get("revenueGrowth_TTM") is None
        assert result.get("revenue_TTM") is None

    def test_extreme_growth_capped(self, fetcher):
        """Revenue from 1M to 100M (100x growth) exceeds sanity bound."""
        revenues = [100e6, 90e6, 80e6, 70e6, 1e6, 1e6, 1e6, 1e6]
        qt_inc = _make_quarterly_income(revenues)
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        # TTM growth = (340M - 4M) / 4M = 84x → exceeds 10.0 bound
        assert result.get("revenueGrowth_TTM") is None
        # MRQ = (100M - 1M) / 1M = 99x → exceeds 10.0 bound
        assert result.get("revenueGrowth_MRQ") is None


class TestDateAlignment:
    """Test fiscal calendar and date-matching logic."""

    @pytest.fixture
    def fetcher(self):
        return SmartMarketDataFetcher()

    def test_march_fy_end_japanese_company(self, fetcher):
        """Japanese company with FY ending March 31."""
        # Quarters: Dec 2025, Sep 2025, Jun 2025, Mar 2025,
        #           Dec 2024, Sep 2024, Jun 2024, Mar 2024
        dates = pd.DatetimeIndex(
            [
                "2025-12-31",
                "2025-09-30",
                "2025-06-30",
                "2025-03-31",
                "2024-12-31",
                "2024-09-30",
                "2024-06-30",
                "2024-03-31",
            ]
        )
        qt_inc = pd.DataFrame(
            {"Total Revenue": [130, 120, 115, 110, 100, 95, 90, 85]},
            index=dates,
        ).T

        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)
        result = fetcher._extract_quarterly_horizons(mock_ticker, "7203.T")

        # MRQ: Dec 2025 (130) vs Dec 2024 (100) → 30%
        assert result.get("revenueGrowth_MRQ") == pytest.approx(0.30, abs=0.01)
        assert result["latest_quarter_date"] == "2025-12-31"

    def test_quarter_gap_in_data(self, fetcher):
        """Missing quarter — Sep 2025 absent."""
        dates = pd.DatetimeIndex(
            [
                "2025-12-31",
                "2025-06-30",
                "2025-03-31",
                "2024-12-31",
                "2024-09-30",
                "2024-06-30",
                "2024-03-31",
                "2023-12-31",
            ]
        )
        qt_inc = pd.DataFrame(
            {"Total Revenue": [130, 115, 110, 100, 95, 90, 85, 80]},
            index=dates,
        ).T

        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)
        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        # MRQ should still find Dec 2024 (12 months from Dec 2025)
        assert result.get("revenueGrowth_MRQ") == pytest.approx(0.30, abs=0.01)

    def test_latest_quarter_date_reported(self, fetcher):
        """Verify latest_quarter_date is the actual calendar date."""
        dates = pd.DatetimeIndex(["2025-09-30", "2025-06-30", "2025-03-31"])
        qt_inc = pd.DataFrame(
            {"Total Revenue": [100, 95, 90]},
            index=dates,
        ).T

        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)
        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        assert result["latest_quarter_date"] == "2025-09-30"


class TestTTMAggregates:
    """Test TTM cash flow and net income aggregation."""

    @pytest.fixture
    def fetcher(self):
        return SmartMarketDataFetcher()

    def test_ocf_ttm_calculation(self, fetcher):
        """TTM OCF sums last 4 quarters correctly."""
        ocf_values = [500, 400, 300, 200, 150, 100, 80, 60]
        qt_cf = _make_quarterly_cashflow(ocf_values)
        mock_ticker = _make_mock_ticker(qt_cf=qt_cf)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        # Sum of Q0..Q3 = 500+400+300+200 = 1400
        assert result.get("operatingCashflow_TTM") == pytest.approx(1400)

    def test_fcf_ttm_calculation(self, fetcher):
        """TTM FCF = OCF + CapEx (CapEx is negative)."""
        ocf_values = [500, 400, 300, 200]
        capex_values = [-100, -80, -70, -50]
        qt_cf = _make_quarterly_cashflow(ocf_values, capex_values)
        mock_ticker = _make_mock_ticker(qt_cf=qt_cf)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        # FCF = (500+400+300+200) + (-100-80-70-50) = 1400 - 300 = 1100
        assert result.get("freeCashflow_TTM") == pytest.approx(1100)

    def test_net_income_ttm(self, fetcher):
        """TTM Net Income sums correctly."""
        ni_values = [50, 45, 40, 35, 30, 28, 25, 22]
        qt_inc = _make_quarterly_income(
            revenues=[100] * 8,  # Revenue doesn't matter here
            net_incomes=ni_values,
        )
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        assert result.get("netIncome_TTM") == pytest.approx(170)  # 50+45+40+35

    def test_earnings_growth_ttm(self, fetcher):
        """TTM earnings growth: last 4Q NI vs prior 4Q NI."""
        # Q0..Q3 = 50, 45, 40, 35 (sum=170)
        # Q4..Q7 = 30, 28, 25, 22 (sum=105)
        # Growth = (170-105)/105 ≈ 61.9%
        ni_values = [50, 45, 40, 35, 30, 28, 25, 22]
        qt_inc = _make_quarterly_income(
            revenues=[100] * 8,
            net_incomes=ni_values,
        )
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        assert result.get("earningsGrowth_TTM") == pytest.approx(0.619, abs=0.01)

    def test_ocf_ttm_nan_propagation(self, fetcher):
        """NaN in OCF quarter prevents partial TTM sum."""
        ocf_values = [500, float("nan"), 300, 200]
        qt_cf = _make_quarterly_cashflow(ocf_values)
        mock_ticker = _make_mock_ticker(qt_cf=qt_cf)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        # Should be None due to min_count=4 with a NaN
        assert result.get("operatingCashflow_TTM") is None


class TestGrowthTrajectory:
    """Test growth trajectory determination."""

    @pytest.fixture
    def fetcher(self):
        return SmartMarketDataFetcher()

    def test_trajectory_accelerating(self, fetcher):
        """MRQ much higher than TTM → ACCELERATING."""
        # MRQ ~50%, TTM ~20% → delta 30pp > 10pp
        revenues = [150, 120, 115, 110, 100, 100, 100, 100]
        qt_inc = _make_quarterly_income(revenues)
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        assert result["growth_trajectory"] == "ACCELERATING"

    def test_trajectory_decelerating(self, fetcher):
        """MRQ much lower than TTM → DECELERATING."""
        # Recent quarter barely grew while prior quarters were strong
        # Q0..Q3 = 101, 130, 125, 120 (sum=476)
        # Q4..Q7 = 100, 100, 100, 100 (sum=400)
        # TTM = (476-400)/400 = 19%
        # MRQ = (101-100)/100 = 1%
        # Delta = 1% - 19% = -18pp < -10pp → DECELERATING
        revenues = [101, 130, 125, 120, 100, 100, 100, 100]
        qt_inc = _make_quarterly_income(revenues)
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        assert result["growth_trajectory"] == "DECELERATING"

    def test_trajectory_stable_boundary(self, fetcher):
        """Delta exactly at boundary (<=10pp) → STABLE."""
        # Q0..Q3 = 122, 112, 112, 112 (sum=458)
        # Q4..Q7 = 100, 100, 100, 100 (sum=400)
        # TTM = (458-400)/400 = 14.5%
        # MRQ = (122-100)/100 = 22%
        # Delta = 22% - 14.5% = 7.5pp < 10pp → STABLE
        revenues = [122, 112, 112, 112, 100, 100, 100, 100]
        qt_inc = _make_quarterly_income(revenues)
        mock_ticker = _make_mock_ticker(qt_inc=qt_inc)

        result = fetcher._extract_quarterly_horizons(mock_ticker, "TEST")

        assert result["growth_trajectory"] == "STABLE"

    def test_trajectory_fallback_mrq_vs_fy(self):
        """When TTM unavailable, _calculate_derived_metrics uses MRQ vs FY."""
        fetcher = SmartMarketDataFetcher()

        # Simulate data dict with MRQ but no TTM or trajectory
        data = {
            "revenueGrowth_MRQ": 0.30,  # 30% MRQ
            "revenueGrowth": 0.05,  # 5% FY
            # No growth_trajectory, no revenueGrowth_TTM
        }
        calculated = fetcher._calculate_derived_metrics(data, "TEST")

        assert calculated["growth_trajectory"] == "ACCELERATING"
        assert calculated["_growth_trajectory_source"] == "calculated_mrq_vs_fy"

    def test_trajectory_fallback_not_needed(self):
        """When trajectory already set, fallback doesn't override."""
        fetcher = SmartMarketDataFetcher()
        data = {
            "growth_trajectory": "STABLE",
            "revenueGrowth_MRQ": 0.30,
            "revenueGrowth": 0.05,
        }
        calculated = fetcher._calculate_derived_metrics(data, "TEST")

        assert "growth_trajectory" not in calculated  # Not overridden


class TestPEGReEnablement:
    """Test PEG calculation with TTM-aligned earnings growth."""

    def test_peg_from_ttm_earnings_growth(self):
        """PEG calculated from trailingPE / (earningsGrowth_TTM * 100)."""
        fetcher = SmartMarketDataFetcher()
        data = {
            "trailingPE": 15.0,
            "earningsGrowth_TTM": 0.20,  # 20% TTM growth
            # pegRatio not set
        }
        calculated = fetcher._calculate_derived_metrics(data, "TEST")

        assert calculated["pegRatio"] == pytest.approx(0.75)
        assert calculated["_pegRatio_source"] == "calculated_from_ttm_aligned"

    def test_peg_not_overridden_when_present(self):
        """If pegRatio already present, don't recalculate."""
        fetcher = SmartMarketDataFetcher()
        data = {
            "pegRatio": 1.2,
            "trailingPE": 15.0,
            "earningsGrowth_TTM": 0.20,
        }
        calculated = fetcher._calculate_derived_metrics(data, "TEST")

        assert "pegRatio" not in calculated  # Not overridden

    def test_peg_skipped_when_growth_too_low(self):
        """Don't calculate PEG when growth is near zero."""
        fetcher = SmartMarketDataFetcher()
        data = {
            "trailingPE": 15.0,
            "earningsGrowth_TTM": 0.005,  # 0.5% → below 1% threshold
        }
        calculated = fetcher._calculate_derived_metrics(data, "TEST")

        assert "pegRatio" not in calculated

    def test_peg_skipped_when_negative_growth(self):
        """Don't calculate PEG when earnings declining."""
        fetcher = SmartMarketDataFetcher()
        data = {
            "trailingPE": 15.0,
            "earningsGrowth_TTM": -0.10,
        }
        calculated = fetcher._calculate_derived_metrics(data, "TEST")

        assert "pegRatio" not in calculated

    def test_peg_sanity_cap(self):
        """PEG > 10 is rejected (unreliable)."""
        fetcher = SmartMarketDataFetcher()
        data = {
            "trailingPE": 50.0,
            "earningsGrowth_TTM": 0.02,  # 2% → PEG = 50/2 = 25 → rejected
        }
        calculated = fetcher._calculate_derived_metrics(data, "TEST")

        assert "pegRatio" not in calculated


class TestRedFlagDetectorIntegration:
    """Test DATA_BLOCK parsing of new fields and GROWTH_CLIFF flag."""

    def test_parse_new_growth_fields(self):
        """Verify extract_metrics parses GROWTH_TRAJECTORY and REVENUE_GROWTH_TTM."""
        report = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 65% (8/12 available)
PE_RATIO_TTM: 14.50
REVENUE_GROWTH_FY: 5.2%
REVENUE_GROWTH_TTM: 18.5%
REVENUE_GROWTH_MRQ: 25.3% (as of 2025-12-31)
EARNINGS_GROWTH_TTM: 22.0%
EARNINGS_GROWTH_MRQ: 30.1%
GROWTH_TRAJECTORY: ACCELERATING
LATEST_QUARTER_DATE: 2025-12-31
### --- END DATA_BLOCK ---
"""
        metrics = RedFlagDetector.extract_metrics(report)

        assert metrics["growth_trajectory"] == "ACCELERATING"
        assert metrics["revenue_growth_ttm"] == 18.5
        assert metrics["latest_quarter_date"] == "2025-12-31"

    def test_parse_decelerating_trajectory(self):
        """Parse DECELERATING trajectory."""
        report = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 55%
GROWTH_TRAJECTORY: DECELERATING
REVENUE_GROWTH_TTM: -8.3%
LATEST_QUARTER_DATE: 2025-09-30
### --- END DATA_BLOCK ---
"""
        metrics = RedFlagDetector.extract_metrics(report)

        assert metrics["growth_trajectory"] == "DECELERATING"
        assert metrics["revenue_growth_ttm"] == -8.3
        assert metrics["latest_quarter_date"] == "2025-09-30"

    def test_growth_cliff_flag_triggered(self):
        """TTM growth < -15% triggers GROWTH_CLIFF warning."""
        report = """
### --- START DATA_BLOCK ---
SECTOR: General/Diversified
ADJUSTED_HEALTH_SCORE: 60%
REVENUE_GROWTH_TTM: -18.5%
GROWTH_TRAJECTORY: DECELERATING
### --- END DATA_BLOCK ---

### KEY METRICS FOR RISK SCREENING
**Interest Coverage**: 5.0x
**Free Cash Flow**: $200M
**Net Income**: $150M
"""
        metrics = RedFlagDetector.extract_metrics(report)
        sector = RedFlagDetector.detect_sector(report)
        flags, result = RedFlagDetector.detect_red_flags(
            metrics, ticker="TEST", sector=sector
        )

        growth_cliff_flags = [f for f in flags if f["type"] == "GROWTH_CLIFF"]
        assert len(growth_cliff_flags) == 1
        assert growth_cliff_flags[0]["severity"] == "WARNING"
        assert growth_cliff_flags[0]["risk_penalty"] == 0.5
        assert result == "PASS"  # WARNING, not AUTO_REJECT

    def test_growth_cliff_not_triggered_mild_decline(self):
        """TTM growth of -8% should NOT trigger GROWTH_CLIFF."""
        report = """
### --- START DATA_BLOCK ---
SECTOR: General/Diversified
ADJUSTED_HEALTH_SCORE: 60%
REVENUE_GROWTH_TTM: -8.0%
GROWTH_TRAJECTORY: DECELERATING
### --- END DATA_BLOCK ---

### KEY METRICS FOR RISK SCREENING
**Interest Coverage**: 5.0x
**Free Cash Flow**: $200M
**Net Income**: $150M
"""
        metrics = RedFlagDetector.extract_metrics(report)
        sector = RedFlagDetector.detect_sector(report)
        flags, result = RedFlagDetector.detect_red_flags(
            metrics, ticker="TEST", sector=sector
        )

        growth_cliff_flags = [f for f in flags if f["type"] == "GROWTH_CLIFF"]
        assert len(growth_cliff_flags) == 0

    def test_backward_compatibility_no_new_fields(self):
        """Old DATA_BLOCK format (no growth fields) still parses correctly."""
        report = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 58% (7/12 available)
PE_RATIO_TTM: 12.34
PEG_RATIO: 0.85
### --- END DATA_BLOCK ---
"""
        metrics = RedFlagDetector.extract_metrics(report)

        assert metrics["adjusted_health_score"] == 58.0
        assert metrics["pe_ratio"] == 12.34
        assert metrics["growth_trajectory"] is None
        assert metrics["revenue_growth_ttm"] is None
        assert metrics["latest_quarter_date"] is None

    def test_growth_fields_with_na_values(self):
        """N/A values for new fields should not populate metrics."""
        report = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 55%
GROWTH_TRAJECTORY: N/A
REVENUE_GROWTH_TTM: N/A
LATEST_QUARTER_DATE: N/A
### --- END DATA_BLOCK ---
"""
        metrics = RedFlagDetector.extract_metrics(report)

        assert metrics["growth_trajectory"] is None
        assert metrics["revenue_growth_ttm"] is None
        # LATEST_QUARTER_DATE regex requires YYYY-MM-DD format, "N/A" won't match
        assert metrics["latest_quarter_date"] is None


class TestHistoricalPriceResolution:
    """Test ticker resolution fallback in get_price_history()."""

    @pytest.fixture
    def fetcher(self):
        return SmartMarketDataFetcher()

    @pytest.mark.asyncio
    async def test_resolution_on_empty_history(self, fetcher):
        """When yfinance returns empty for alpha ticker, resolve and retry."""
        empty_hist = pd.DataFrame()
        resolved_hist = pd.DataFrame(
            {"Close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2025-01-01", periods=3),
        )

        call_count = 0

        def mock_history(period="1y"):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return empty_hist  # First call: alpha ticker fails
            return resolved_hist  # Second call: numeric ticker works

        mock_ticker = MagicMock()
        mock_ticker.history = mock_history

        with (
            patch("yfinance.Ticker", return_value=mock_ticker) as mock_yf,
            patch.object(
                fetcher,
                "_resolve_ticker_via_search",
                new_callable=AsyncMock,
                return_value="7052.KL",
            ) as mock_resolve,
        ):
            result = await fetcher.get_price_history("PADINI.KL")

        mock_resolve.assert_called_once_with("PADINI.KL")
        assert not result.empty
        assert len(result) == 3
        # yf.Ticker called twice: once for original, once for resolved
        assert mock_yf.call_count == 2

    @pytest.mark.asyncio
    async def test_no_resolution_when_data_present(self, fetcher):
        """When yfinance returns data on first try, skip resolution."""
        good_hist = pd.DataFrame(
            {"Close": [10.0, 11.0]},
            index=pd.date_range("2025-06-01", periods=2),
        )

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = good_hist

        with (
            patch("yfinance.Ticker", return_value=mock_ticker),
            patch.object(
                fetcher,
                "_resolve_ticker_via_search",
                new_callable=AsyncMock,
            ) as mock_resolve,
        ):
            result = await fetcher.get_price_history("0005.HK")

        mock_resolve.assert_not_called()
        assert not result.empty
        assert len(result) == 2
