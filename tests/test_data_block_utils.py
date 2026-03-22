from __future__ import annotations

import json
from pathlib import Path

from src.data_block_utils import (
    extract_last_data_block,
    extract_last_fenced_block,
    has_parseable_data_block,
    has_parseable_fenced_block,
    normalize_legacy_data_block_report,
    normalize_structured_block_boundaries,
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


def test_normalize_legacy_data_block_repairs_markdown_table_shape():
    report = """
### DATA_BLOCK

| Metric | Value |
| :--- | :--- |
| SECTOR | Energy |
| RAW_HEALTH_SCORE | 9.5/12 |
| ADJUSTED_HEALTH_SCORE | 79% (12/12 available) |
| RAW_GROWTH_SCORE | 3/6 |
| ADJUSTED_GROWTH_SCORE | 50% (6/6 available) |
| US_REVENUE_PERCENT | Not disclosed |
| ANALYST_COVERAGE_ENGLISH | 2 |
| PE_RATIO_TTM | 12.35 |
| ADR_EXISTS | YES |
| IBKR_ACCESSIBILITY | Direct |
| PFIC_RISK | LOW |

### FINANCIAL HEALTH DETAIL
Score details here.
"""

    normalized = normalize_legacy_data_block_report(report)

    assert normalized is not None
    assert "### --- START DATA_BLOCK ---" in normalized
    assert "SECTOR: Energy" in normalized
    assert "RAW_HEALTH_SCORE: 9.5/12" in normalized
    assert "### FINANCIAL HEALTH DETAIL" in normalized
    assert has_parseable_data_block(normalized) is True


def test_extract_last_data_block_recovers_markdown_table_shape():
    report = """
### DATA_BLOCK
| Metric | Value |
| :--- | :--- |
| SECTOR | Energy |
| RAW_HEALTH_SCORE | 9.5/12 |
| ADJUSTED_HEALTH_SCORE | 79% (12/12 available) |
| RAW_GROWTH_SCORE | 3/6 |
| ADJUSTED_GROWTH_SCORE | 50% (6/6 available) |
| US_REVENUE_PERCENT | Not disclosed |
| ANALYST_COVERAGE_ENGLISH | 2 |
| PE_RATIO_TTM | 12.35 |
| ADR_EXISTS | YES |
| IBKR_ACCESSIBILITY | Direct |
| PFIC_RISK | LOW |
"""

    block = extract_last_data_block(report)

    assert has_parseable_data_block(report) is True
    assert block is not None
    assert "SECTOR: Energy" in block
    assert "PFIC_RISK: LOW" in block


def test_normalize_legacy_data_block_rejects_non_two_column_tables():
    report = """
### DATA_BLOCK
| Metric | Value | Extra |
| :--- | :--- | :--- |
| SECTOR | Energy | Noise |
| RAW_HEALTH_SCORE | 9.5/12 | Noise |
| ADJUSTED_HEALTH_SCORE | 79% (12/12 available) | Noise |
| RAW_GROWTH_SCORE | 3/6 | Noise |
| ADJUSTED_GROWTH_SCORE | 50% (6/6 available) | Noise |
"""

    normalized = normalize_legacy_data_block_report(report)

    assert normalized == report
    assert has_parseable_data_block(report) is False


def test_normalize_structured_block_boundaries_repairs_glued_datablock_heading():
    report = (
        "### --- START DATA_BLOCK ---\n"
        "SECTOR: Energy\n"
        "### --- END DATA_BLOCK ---### FINANCIAL HEALTH DETAIL\n"
        "**Score**: 9/12\n"
    )

    normalized = normalize_structured_block_boundaries(report)

    assert normalized is not None
    assert "### --- END DATA_BLOCK ---\n\n### FINANCIAL HEALTH DETAIL" in normalized


def test_normalize_structured_block_boundaries_repairs_glued_pm_block_heading():
    report = (
        "### --- START PM_BLOCK ---\n"
        "VERDICT: BUY\n"
        "### --- END PM_BLOCK ---### POSITION SIZING\n"
        "5%\n"
    )

    normalized = normalize_structured_block_boundaries(report)

    assert normalized is not None
    assert "### --- END PM_BLOCK ---\n\n### POSITION SIZING" in normalized


def test_normalize_structured_block_boundaries_leaves_clean_text_unchanged():
    report = (
        "### --- START DATA_BLOCK ---\n"
        "SECTOR: Energy\n"
        "### --- END DATA_BLOCK ---\n\n"
        "### FINANCIAL HEALTH DETAIL\n"
        "**Score**: 9/12\n"
    )

    assert normalize_structured_block_boundaries(report) == report


def test_normalize_legacy_data_block_accepts_subtitled_h3_block():
    report = """
### DATA_BLOCK (INTERNAL SCORING — NOT THIRD-PARTY RATINGS)
SECTOR: Consumer Defensive
RAW_HEALTH_SCORE: 9/12
ADJUSTED_HEALTH_SCORE: 75%
RAW_GROWTH_SCORE: 2/6
ADJUSTED_GROWTH_SCORE: 33%
US_REVENUE_PERCENT: Not disclosed
ANALYST_COVERAGE_ENGLISH: 4
PE_RATIO_TTM: 1.45
ADR_EXISTS: NO
IBKR_ACCESSIBILITY: Direct
PFIC_RISK: LOW

### FINANCIAL HEALTH DETAIL
Details.
"""

    normalized = normalize_legacy_data_block_report(report)

    assert normalized is not None
    assert has_parseable_data_block(normalized) is True
    assert "### --- START DATA_BLOCK ---" in normalized


def test_normalize_legacy_data_block_accepts_dashed_header_with_explicit_end():
    report = """
### --- DATA_BLOCK ---
SECTOR: Financials
RAW_HEALTH_SCORE: 10.5/12
ADJUSTED_HEALTH_SCORE: 87.5%
RAW_GROWTH_SCORE: 4.5/6
ADJUSTED_GROWTH_SCORE: 75%
US_REVENUE_PERCENT: Not disclosed
ANALYST_COVERAGE_ENGLISH: 9
PE_RATIO_TTM: 14.68
ADR_EXISTS: NO
IBKR_ACCESSIBILITY: Direct
PFIC_RISK: LOW
### --- END DATA_BLOCK ---

### FINANCIAL HEALTH DETAIL
Details.
"""

    normalized = normalize_legacy_data_block_report(report)

    assert normalized is not None
    assert has_parseable_data_block(normalized) is True
    assert normalized.count("### --- END DATA_BLOCK ---") == 1


def test_normalize_legacy_data_block_rejects_dashed_header_without_end_marker():
    report = """
### --- DATA_BLOCK ---
SECTOR: Financials
RAW_HEALTH_SCORE: 10.5/12
ADJUSTED_HEALTH_SCORE: 87.5%
RAW_GROWTH_SCORE: 4.5/6
ADJUSTED_GROWTH_SCORE: 75%
US_REVENUE_PERCENT: Not disclosed
ANALYST_COVERAGE_ENGLISH: 9
PE_RATIO_TTM: 14.68
ADR_EXISTS: NO
IBKR_ACCESSIBILITY: Direct
PFIC_RISK: LOW

### FINANCIAL HEALTH DETAIL
Details.
"""

    normalized = normalize_legacy_data_block_report(report)

    assert normalized == report
    assert has_parseable_data_block(normalized) is False


def test_normalize_legacy_data_block_rejects_indented_h3_header():
    report = """
  ### DATA_BLOCK
SECTOR: Energy
RAW_HEALTH_SCORE: 9/12
ADJUSTED_HEALTH_SCORE: 79%
RAW_GROWTH_SCORE: 3/6
ADJUSTED_GROWTH_SCORE: 50%
US_REVENUE_PERCENT: Not disclosed
ANALYST_COVERAGE_ENGLISH: 2
PE_RATIO_TTM: 12.35
ADR_EXISTS: YES
IBKR_ACCESSIBILITY: Direct
PFIC_RISK: LOW
"""

    normalized = normalize_legacy_data_block_report(report)

    assert normalized == report
    assert has_parseable_data_block(normalized) is False


def _load_fundamentals_artifact(path: str) -> str:
    payload = json.loads(Path(path).read_text())
    artifact = payload.get("artifact_statuses", {}).get("fundamentals_report", {})
    return artifact.get("content") or payload.get("reports", {}).get(
        "fundamentals_report", ""
    )


def test_real_sample_tsu_to_datablock_becomes_parseable():
    content = _load_fundamentals_artifact(
        "results/TSU.TO_20260321_230515_analysis.json"
    )

    assert has_parseable_data_block(content) is True


def test_real_sample_veto_pa_datablock_becomes_parseable():
    content = _load_fundamentals_artifact(
        "results/VETO.PA_20260321_234237_analysis.json"
    )

    assert has_parseable_data_block(content) is True


def test_real_sample_vta_as_datablock_becomes_parseable():
    content = _load_fundamentals_artifact(
        "results/VTA.AS_20260322_000857_analysis.json"
    )

    assert has_parseable_data_block(content) is True


def test_real_sample_2001_hk_datablock_becomes_parseable():
    content = _load_fundamentals_artifact(
        "results/2001.HK_20260320_112335_analysis.json"
    )

    assert has_parseable_data_block(content) is True


def test_real_sample_1616_tw_datablock_becomes_parseable():
    content = _load_fundamentals_artifact(
        "results/1616.TW_20260320_100744_analysis.json"
    )

    assert has_parseable_data_block(content) is True


def test_real_sample_3539_t_stays_non_parseable():
    content = _load_fundamentals_artifact(
        "results/3539.T_20260320_173600_analysis.json"
    )

    assert has_parseable_data_block(content) is False
