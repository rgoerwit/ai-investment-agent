"""
Extract chart-relevant data from Fundamentals Analyst DATA_BLOCK.
"""

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ChartRawData:
    """Raw data extracted from DATA_BLOCK for chart generation."""

    current_price: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    moving_avg_50: float | None = None
    moving_avg_200: float | None = None
    external_target_high: float | None = None
    external_target_low: float | None = None
    external_target_mean: float | None = None

    # Extended metrics for Radar Chart
    pe_ratio_ttm: float | None = None
    peg_ratio: float | None = None
    adjusted_health_score: float | None = None
    adjusted_growth_score: float | None = None
    analyst_coverage: int | None = None
    pfic_risk: str | None = None
    adr_impact: str | None = None
    us_revenue_percent: str | None = None


def _extract_float(pattern: str, text: str) -> float | None:
    """Extract a float value using regex pattern.

    Handles common LLM output variations:
    - Currency symbols ($)
    - Comma separators (1,234.56)
    - Markdown formatting (*bold*, _italic_, `code`)
    """
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            # Strip markdown formatting, currency symbols, commas, whitespace
            value_str = match.group(1)
            value_str = re.sub(r"[*_`$,\s%]", "", value_str)  # Added % stripping
            return float(value_str)
        except (ValueError, IndexError):
            return None
    return None


def _extract_str(pattern: str, text: str) -> str | None:
    """Extract a string value using regex pattern."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_chart_data_from_data_block(fundamentals_report: str) -> ChartRawData:
    """Extract chart-relevant data from Fundamentals Analyst report.

    Looks for the DATA_BLOCK section and extracts:
    - Price & Targets
    - P/E & PEG
    - Scores (Health/Growth)
    - Risk factors (PFIC, ADR, Analyst Coverage)

    Args:
        fundamentals_report: The full fundamentals analyst report text

    Returns:
        ChartRawData with extracted values (None for missing fields)
    """
    if not fundamentals_report:
        return ChartRawData()

    # Find the DATA_BLOCK section (take last one for self-correction pattern)
    data_block_pattern = r"### --- START DATA_BLOCK ---(.+?)### --- END DATA_BLOCK ---"
    blocks = list(re.finditer(data_block_pattern, fundamentals_report, re.DOTALL))

    if not blocks:
        logger.debug("No DATA_BLOCK found in fundamentals report")
        return ChartRawData()

    # Use the last (most corrected) block
    data_block = blocks[-1].group(1)

    # Extract each field
    # Pattern captures optional markdown/currency chars around numbers
    # e.g., "**$180.00**" or "$1,234.56" or just "180.00"
    num_pattern = r"[*_`$\s]*([\d,.]+)[*_`]*"

    result = ChartRawData(
        current_price=_extract_float(rf"CURRENT_PRICE:\s*{num_pattern}", data_block),
        fifty_two_week_high=_extract_float(
            rf"FIFTY_TWO_WEEK_HIGH:\s*{num_pattern}", data_block
        ),
        fifty_two_week_low=_extract_float(
            rf"FIFTY_TWO_WEEK_LOW:\s*{num_pattern}", data_block
        ),
        moving_avg_50=_extract_float(rf"MOVING_AVG_50:\s*{num_pattern}", data_block),
        moving_avg_200=_extract_float(rf"MOVING_AVG_200:\s*{num_pattern}", data_block),
        external_target_high=_extract_float(
            rf"EXTERNAL_ANALYST_TARGET_HIGH:\s*{num_pattern}", data_block
        ),
        external_target_low=_extract_float(
            rf"EXTERNAL_ANALYST_TARGET_LOW:\s*{num_pattern}", data_block
        ),
        external_target_mean=_extract_float(
            rf"EXTERNAL_ANALYST_TARGET_MEAN:\s*{num_pattern}", data_block
        ),
        # Extended metrics
        pe_ratio_ttm=_extract_float(rf"PE_RATIO_TTM:\s*{num_pattern}", data_block),
        peg_ratio=_extract_float(rf"PEG_RATIO:\s*{num_pattern}", data_block),
        adjusted_health_score=_extract_float(
            rf"ADJUSTED_HEALTH_SCORE:\s*{num_pattern}", data_block
        ),
        adjusted_growth_score=_extract_float(
            rf"ADJUSTED_GROWTH_SCORE:\s*{num_pattern}", data_block
        ),
        analyst_coverage=int(
            _extract_float(rf"ANALYST_COVERAGE_ENGLISH:\s*{num_pattern}", data_block)
            or 0
        ),
        pfic_risk=_extract_str(r"PFIC_RISK:\s*([A-Za-z]+)", data_block),
        adr_impact=_extract_str(r"ADR_THESIS_IMPACT:\s*([A-Za-z_]+)", data_block),
        us_revenue_percent=_extract_str(r"US_REVENUE_PERCENT:\s*(.+)", data_block),
    )

    logger.debug(
        "Extracted chart data from DATA_BLOCK",
        current_price=result.current_price,
        health=result.adjusted_health_score,
        pe=result.pe_ratio_ttm,
    )

    return result
