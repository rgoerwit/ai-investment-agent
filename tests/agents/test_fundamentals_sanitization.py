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
