"""Regression tests for fundamentals DATA_BLOCK sanitization."""

from __future__ import annotations

import json

from src.agents.analyst_nodes import _sanitize_fundamentals_output


def test_sanitize_fundamentals_output_forces_missing_horizons_to_na() -> None:
    content = """### --- START DATA_BLOCK ---
REVENUE_GROWTH_FY: 39.8%
REVENUE_GROWTH_TTM: 39.8%
REVENUE_GROWTH_MRQ: 39.8% (as of 2025-12-31)
EARNINGS_GROWTH_TTM: 98.8%
EARNINGS_GROWTH_MRQ: 100.5%
GROWTH_TRAJECTORY: STABLE
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "revenueGrowth": 0.398,
            "revenueGrowth_TTM": None,
            "revenueGrowth_MRQ": None,
            "earningsGrowth": 0.988,
            "earningsGrowth_TTM": None,
            "earningsGrowth_MRQ": None,
            "growth_trajectory": None,
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "2173.T")

    assert "REVENUE_GROWTH_FY: 39.8%" in sanitized
    assert "REVENUE_GROWTH_TTM: N/A" in sanitized
    assert "REVENUE_GROWTH_MRQ: N/A" in sanitized
    assert "EARNINGS_GROWTH_TTM: N/A" in sanitized
    assert "EARNINGS_GROWTH_MRQ: N/A" in sanitized
    assert "GROWTH_TRAJECTORY: N/A" in sanitized


def test_sanitize_invalidates_home_ticker_adr_routing() -> None:
    content = """### --- START DATA_BLOCK ---
ADR_EXISTS: YES
ADR_TYPE: SPONSORED
ADR_TICKER: POMO4.SA
ADR_EXCHANGE: SAO
ADR_THESIS_IMPACT: MODERATE_CONCERN
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"trailingPE": 6.0})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "POMO4.SA")

    assert "ADR_EXISTS: YES" in sanitized
    assert "ADR_TYPE: SPONSORED" in sanitized
    assert "ADR_TICKER: None" in sanitized
    assert "ADR_EXCHANGE: None" in sanitized
    assert "ADR_THESIS_IMPACT: UNCERTAIN" in sanitized
    assert "ADR_DATA_QUALITY_NOTE: Invalid ADR routing fields removed" in sanitized


def test_sanitize_invalidates_non_us_adr_exchange() -> None:
    content = """### --- START DATA_BLOCK ---
ADR_EXISTS: YES
ADR_TYPE: SPONSORED
ADR_TICKER: EXAMPLE
ADR_EXCHANGE: SAO
ADR_THESIS_IMPACT: MODERATE_CONCERN
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"trailingPE": 6.0})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "EXMP3.SA")

    assert "ADR_TICKER: None" in sanitized
    assert "ADR_EXCHANGE: None" in sanitized
    assert "ADR_THESIS_IMPACT: UNCERTAIN" in sanitized


def test_sanitize_preserves_valid_adr_routing() -> None:
    content = """### --- START DATA_BLOCK ---
ADR_EXISTS: YES
ADR_TYPE: SPONSORED
ADR_TICKER: ABEV
ADR_EXCHANGE: NYSE
ADR_THESIS_IMPACT: MODERATE_CONCERN
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "trailingPE": 16.0,
            "revenueGrowth_TTM": 0.1,
            "revenueGrowth_MRQ": 0.1,
            "earningsGrowth_TTM": 0.1,
            "earningsGrowth_MRQ": 0.1,
            "growth_trajectory": "STABLE",
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "ABEV3.SA")

    assert sanitized == content


def test_sanitize_quarantined_low_pe_sets_valuation_to_na() -> None:
    content = """### --- START DATA_BLOCK ---
PE_RATIO_TTM: 1.60
PEG_RATIO: 0.20
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"_pe_low_anomaly_quarantined": True})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "SEER3.SA")

    assert "PE_RATIO_TTM: N/A" in sanitized
    assert "PEG_RATIO: N/A" in sanitized


def test_sanitize_low_pe_flag_only_keeps_valuation_lines() -> None:
    content = """### --- START DATA_BLOCK ---
PE_RATIO_TTM: 4.20
PEG_RATIO: 0.70
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "_pe_low_anomaly_flag": "LOW_PE_REQUIRES_INVESTIGATION",
            "revenueGrowth_TTM": 0.1,
            "revenueGrowth_MRQ": 0.1,
            "earningsGrowth_TTM": 0.1,
            "earningsGrowth_MRQ": 0.1,
            "growth_trajectory": "STABLE",
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "TEST.SA")

    assert sanitized == content
