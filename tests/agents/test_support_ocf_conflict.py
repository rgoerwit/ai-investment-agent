"""Tests for sub-annual OCF period-mismatch detection in compute_data_conflicts.

Guards the fix where a single-quarter (Q1-Q4) filing OCF compared against a TTM
aggregator OCF must be labeled "PERIOD MISMATCH", not "INVESTIGATE" — the exact
defect the external consultant flagged on recurring ALV.V runs.
"""

from __future__ import annotations

from src.agents.support import compute_data_conflicts

# Junior (yfinance) OCF 40M vs filing OCF 10.6M → ratio ~3.8x (> 1.3 threshold).
_RAW = '{"operatingCashflow": 40000000}'


def _foreign(period: str | None) -> str:
    block = "Operating Cash Flow (Filing): $10.6M"
    if period is not None:
        block += f"\nPeriod: {period}"
    return block


def _ocf_line(text: str) -> str:
    return next((line for line in text.splitlines() if line.startswith("- OCF:")), "")


def test_quarterly_period_is_mismatch():
    out = compute_data_conflicts(_RAW, _foreign("Q4 2025"))
    line = _ocf_line(out)
    assert "PERIOD MISMATCH" in line
    assert "(Q4 2025)" in line


def test_half_year_period_still_mismatch():
    out = compute_data_conflicts(_RAW, _foreign("H1 2025"))
    assert "PERIOD MISMATCH" in _ocf_line(out)


def test_annual_period_is_investigate_not_mismatch():
    out = compute_data_conflicts(_RAW, _foreign("FY2025"))
    line = _ocf_line(out)
    assert "PERIOD MISMATCH" not in line
    assert "INVESTIGATE" in line


def test_bare_year_is_investigate():
    out = compute_data_conflicts(_RAW, _foreign("2025"))
    line = _ocf_line(out)
    assert "PERIOD MISMATCH" not in line
    assert "INVESTIGATE" in line


def test_missing_period_does_not_crash_or_mismatch():
    out = compute_data_conflicts(_RAW, _foreign(None))
    line = _ocf_line(out)
    # Conflict still surfaces (ratio material), but without a sub-annual label.
    assert "PERIOD MISMATCH" not in line
    assert "INVESTIGATE" in line


def test_immaterial_ratio_emits_no_ocf_conflict():
    # Filing 39M vs junior 40M → ratio ~1.03x, below the 1.3x materiality gate.
    foreign = "Operating Cash Flow (Filing): $39M\nPeriod: Q4 2025"
    out = compute_data_conflicts(_RAW, foreign)
    assert _ocf_line(out) == ""


def test_missing_foreign_data_no_ocf_conflict():
    out = compute_data_conflicts(_RAW, "")
    assert _ocf_line(out) == ""
