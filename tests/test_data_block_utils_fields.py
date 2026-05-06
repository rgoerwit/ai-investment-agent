from __future__ import annotations

from src.data_block_utils import (
    extract_block_field,
    extract_data_block_field,
    extract_data_block_number,
    find_fenced_block_spans,
)


def test_extract_data_block_field_uses_last_parseable_block():
    report = """
### --- START DATA_BLOCK ---
SECTOR: Utilities
ADJUSTED_HEALTH_SCORE: 40%
### --- END DATA_BLOCK ---

### --- START DATA_BLOCK ---
SECTOR: Industrials
ADJUSTED_HEALTH_SCORE: 82%
ADJUSTED_GROWTH_SCORE: 61%
### --- END DATA_BLOCK ---
"""

    assert extract_data_block_field(report, "SECTOR") == "Industrials"
    assert extract_data_block_number(report, "ADJUSTED_HEALTH_SCORE") == 82.0
    assert extract_data_block_number(report, "ADJUSTED_GROWTH_SCORE") == 61.0


def test_extract_data_block_number_handles_commas_and_percentages():
    report = """
### --- START DATA_BLOCK ---
CURRENT_PRICE: $1,234.56
ADJUSTED_HEALTH_SCORE: 79% (12/12 available)
### --- END DATA_BLOCK ---
"""

    assert extract_data_block_number(report, "CURRENT_PRICE") == 1234.56
    assert extract_data_block_number(report, "ADJUSTED_HEALTH_SCORE") == 79.0


def test_extract_data_block_field_treats_common_null_tokens_as_none():
    report = """
### --- START DATA_BLOCK ---
SECTOR: N/A
ADR_THESIS_IMPACT: -
PFIC_RISK: none
### --- END DATA_BLOCK ---
"""

    assert extract_data_block_field(report, "SECTOR") is None
    assert extract_data_block_field(report, "ADR_THESIS_IMPACT") is None
    assert extract_data_block_field(report, "PFIC_RISK") is None


def test_extract_block_field_is_case_insensitive():
    report = """
### --- START DATA_BLOCK ---
Sector: Health Care
### --- END DATA_BLOCK ---
"""

    assert extract_block_field(report, "DATA_BLOCK", "SECTOR") == "Health Care"


def test_find_fenced_block_spans_returns_each_parseable_block_with_markers():
    report = """
### --- START DATA_BLOCK ---
SECTOR: Utilities
### --- END DATA_BLOCK ---

### --- START DATA_BLOCK ---
SECTOR: Energy
### --- END DATA_BLOCK ---
"""

    spans = find_fenced_block_spans(report, "DATA_BLOCK", include_markers=True)

    assert len(spans) == 2
    assert spans[0][2].startswith("### --- START DATA_BLOCK ---")
    assert "SECTOR: Energy" in spans[-1][2]


def test_extract_data_block_accessors_return_none_for_missing_or_malformed_reports():
    assert extract_data_block_field("", "SECTOR") is None
    assert extract_data_block_field(None, "SECTOR") is None
    assert extract_data_block_number("not a block", "CURRENT_PRICE") is None
