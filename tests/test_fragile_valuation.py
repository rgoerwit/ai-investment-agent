import pytest

from src.validators.red_flag_detector import RedFlagDetector, Sector


def test_fragile_valuation_detection():
    """
    Test detection of 'Fragile Valuation' (Construction Trap):
    Low Margins (<5%) + High Valuation (P/B > 4x) + High Debt (D/E > 80%).
    Based on MAIRE.MI case.
    """
    # Simulate a DATA_BLOCK with the risky metrics
    report = """
    Analyzing MAIRE.MI - Maire Tecnimont S.p.A.

    ### --- START DATA_BLOCK ---
    SECTOR: Engineering & Construction
    RAW_HEALTH_SCORE: 8/12
    ADJUSTED_HEALTH_SCORE: 66% (8/12 available)
    PE_RATIO_TTM: 17.5
    PB_RATIO: 6.94
    DE_RATIO: 138.0
    NET_MARGIN: 3.69%
    ROA_PERCENT: 2.6
    ### --- END DATA_BLOCK ---
    """

    metrics = RedFlagDetector.extract_metrics(report)

    # Verify extraction
    assert metrics["net_margin"] == 3.69
    assert metrics["pb_ratio"] == 6.94
    assert metrics["debt_to_equity"] == 138.0

    # Run detection
    red_flags, result = RedFlagDetector.detect_red_flags(metrics, ticker="MAIRE.MI")

    # Verify Red Flag
    fragile_flag = next(
        (f for f in red_flags if f["type"] == "FRAGILE_VALUATION"), None
    )

    assert fragile_flag is not None, "FRAGILE_VALUATION flag not triggered"
    assert fragile_flag["severity"] == "CRITICAL"
    assert fragile_flag["action"] == "CRITICAL_WARNING"
    assert "P/B 6.9x" in fragile_flag["detail"]
    assert "3.7% margins" in fragile_flag["detail"]
    assert result == "PASS"


def test_fragile_valuation_pass_scenario():
    """Test that a healthy company (high margin or low debt) passes."""
    # Scenario A: High Margins (Software profile)
    report_software = """
    ### --- START DATA_BLOCK ---
    PB_RATIO: 8.0
    DE_RATIO: 20.0
    NET_MARGIN: 25.0%
    ### --- END DATA_BLOCK ---
    """
    metrics_sw = RedFlagDetector.extract_metrics(report_software)
    flags_sw, result_sw = RedFlagDetector.detect_red_flags(metrics_sw)
    assert not any(f["type"] == "FRAGILE_VALUATION" for f in flags_sw)

    # Scenario B: Low Valuation (Construction Value stock)
    report_value = """
    ### --- START DATA_BLOCK ---
    PB_RATIO: 1.2
    DE_RATIO: 90.0
    NET_MARGIN: 3.5%
    ### --- END DATA_BLOCK ---
    """
    metrics_val = RedFlagDetector.extract_metrics(report_value)
    flags_val, result_val = RedFlagDetector.detect_red_flags(metrics_val)
    assert not any(f["type"] == "FRAGILE_VALUATION" for f in flags_val)
