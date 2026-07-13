from __future__ import annotations

from src.validators.red_flag_detector import RedFlagDetector


def test_extract_metrics_reads_sector_relative_and_trajectory_fields() -> None:
    report = """
### --- START DATA_BLOCK ---
SECTOR: Financials
PE_RATIO_TTM: 14.0
SECTOR_MEDIAN_PE: 12.0
PE_VS_SECTOR: 1.17
REVENUE_CAGR_3Y: 8.5%
FCF_CAGR_3Y: -2.5%
CYCLE_POSITION: PEAK
### --- END DATA_BLOCK ---
"""
    metrics = RedFlagDetector.extract_metrics(report)

    assert metrics["sector_median_pe"] == 12.0
    assert metrics["pe_vs_sector"] == 1.17
    assert metrics["revenue_cagr_3y"] == 8.5
    assert metrics["fcf_cagr_3y"] == -2.5
    assert metrics["cycle_position"] == "PEAK"


def test_sector_relative_valuation_flags_when_absolute_pe_passes() -> None:
    flags, result = RedFlagDetector.detect_red_flags(
        {
            "pe_ratio": 14.0,
            "sector_median_pe": 10.0,
            "pe_vs_sector": 1.4,
        },
        "TEST",
    )

    relative_flags = [
        flag for flag in flags if flag["type"] == "SECTOR_RELATIVE_VALUATION_RICH"
    ]
    assert result == "PASS"
    assert len(relative_flags) == 1
    assert relative_flags[0]["risk_penalty"] == 0.5


def test_b3_style_sector_relative_valuation_flags_after_sector_alias_fix() -> None:
    flags, result = RedFlagDetector.detect_red_flags(
        {
            "pe_ratio": 16.55,
            "sector_median_pe": 12.0,
            "pe_vs_sector": 1.38,
        },
        "B3SA3.SA",
    )

    relative_flags = [
        flag for flag in flags if flag["type"] == "SECTOR_RELATIVE_VALUATION_RICH"
    ]
    assert result == "PASS"
    assert len(relative_flags) == 1
    assert relative_flags[0]["risk_penalty"] == 0.5


def test_sector_relative_valuation_does_not_double_count_high_absolute_pe() -> None:
    flags, _ = RedFlagDetector.detect_red_flags(
        {
            "pe_ratio": 22.0,
            "sector_median_pe": 10.0,
            "pe_vs_sector": 2.2,
        },
        "TEST",
    )

    assert not [
        flag for flag in flags if flag["type"] == "SECTOR_RELATIVE_VALUATION_RICH"
    ]
