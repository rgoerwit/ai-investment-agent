from src.data_block_utils import (
    extract_last_data_block,
    extract_last_fenced_block,
    has_parseable_data_block,
    has_parseable_fenced_block,
    normalize_legacy_data_block_report,
)


def test_has_parseable_data_block_requires_fenced_block():
    report = "The analyst mentions DATA_BLOCK but never emits the fenced section."

    assert has_parseable_data_block(report) is False
    assert extract_last_data_block(report) is None


def test_extract_last_data_block_accepts_descriptive_marker():
    report = """
### --- START DATA_BLOCK (INTERNAL SCORING — NOT THIRD-PARTY RATINGS) ---
ADJUSTED_HEALTH_SCORE: 82%
### --- END DATA_BLOCK ---
"""

    assert has_parseable_data_block(report) is True
    assert "ADJUSTED_HEALTH_SCORE: 82%" in extract_last_data_block(report)


def test_generic_fenced_block_helper_requires_real_markers():
    report = "VALUATION_PARAMS:\nMETHOD: P/E_NORMALIZATION"

    assert has_parseable_fenced_block(report, "VALUATION_PARAMS") is False
    assert extract_last_fenced_block(report, "VALUATION_PARAMS") is None


def test_generic_fenced_block_helper_extracts_last_matching_block():
    report = """
### --- START VALUATION_PARAMS ---
METHOD: P/E_NORMALIZATION
### --- END VALUATION_PARAMS ---

### --- START VALUATION_PARAMS ---
METHOD: PEG_BASED
### --- END VALUATION_PARAMS ---
"""

    assert has_parseable_fenced_block(report, "VALUATION_PARAMS") is True
    assert "METHOD: PEG_BASED" in extract_last_fenced_block(report, "VALUATION_PARAMS")


def test_normalize_legacy_data_block_repairs_exact_legacy_shape():
    report = """
### DATA_BLOCK
SECTOR: Industrials
RAW_HEALTH_SCORE: 5/12
ADJUSTED_HEALTH_SCORE: 41.7% (based on 12 available points)
RAW_GROWTH_SCORE: 1/6
ADJUSTED_GROWTH_SCORE: 16.7% (based on 6 available points)
US_REVENUE_PERCENT: Not disclosed

### FINANCIAL HEALTH DETAIL
Score details here.
"""

    normalized = normalize_legacy_data_block_report(report)

    assert normalized is not None
    assert "### --- START DATA_BLOCK ---" in normalized
    assert "### --- END DATA_BLOCK ---" in normalized
    assert "### FINANCIAL HEALTH DETAIL" in normalized
    assert has_parseable_data_block(normalized) is True


def test_normalize_legacy_data_block_ignores_narrative_mentions():
    report = """
The analysis references DATA_BLOCK compliance but does not emit one.

DATA_BLOCK:
- missing
"""

    normalized = normalize_legacy_data_block_report(report)

    assert normalized == report
    assert has_parseable_data_block(normalized) is False
