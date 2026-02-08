"""
Extract chart-relevant data from Portfolio Manager PM_BLOCK.

This module extracts the PM's risk-adjusted view for post-verdict chart generation.
PM_BLOCK is a structured output that captures the PM's final assessment, enabling
charts to reflect the actual investment decision rather than raw fundamentals data.
"""

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PMBlockData:
    """Data extracted from Portfolio Manager PM_BLOCK for chart generation."""

    # Core verdict
    verdict: str | None = None  # BUY, HOLD, DO_NOT_INITIATE, SELL

    # Adjusted scores (PM may apply turnaround exceptions, etc.)
    health_adj: int | None = None  # Adjusted health score (0-100)
    growth_adj: int | None = None  # Adjusted growth score (0-100)

    # Risk assessment
    risk_tally: float | None = None  # Cumulative risk tally (e.g., 1.33)
    zone: str | None = None  # LOW, MODERATE, HIGH

    # Chart control
    show_valuation_chart: bool = True  # False for SELL/DO_NOT_INITIATE
    valuation_discount: float = 1.0  # Risk-adjusted discount for target range

    # Position sizing
    position_size: float | None = None  # Recommended position size (%)

    # Valuation context
    valuation_context: str | None = None  # STANDARD, CONTEXTUAL_PASS, N_A

    def should_show_targets(self) -> bool:
        """Determine if 'Our Target' range should be shown on football field."""
        return self.show_valuation_chart and self.verdict in ("BUY", "HOLD")


def _extract_float(pattern: str, text: str) -> float | None:
    """Extract a float value using regex pattern."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            value_str = match.group(1).strip()
            if value_str.upper() in ("N/A", "NA", "NONE", "-", ""):
                return None
            return float(value_str)
        except (ValueError, IndexError):
            return None
    return None


def _extract_int(pattern: str, text: str) -> int | None:
    """Extract an integer value using regex pattern."""
    val = _extract_float(pattern, text)
    return int(val) if val is not None else None


def _extract_str(pattern: str, text: str) -> str | None:
    """Extract a string value using regex pattern."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        if value.upper() in ("N/A", "NA", "NONE", "-", ""):
            return None
        return value
    return None


def _extract_bool(pattern: str, text: str, true_value: str = "YES") -> bool:
    """Extract a boolean value using regex pattern."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper() == true_value.upper()
    return True  # Default to True (show charts) if not specified


def extract_pm_block(pm_output: str) -> PMBlockData:
    """Extract chart-relevant data from Portfolio Manager PM_BLOCK.

    Looks for the PM_BLOCK section and extracts:
    - Verdict (BUY/HOLD/DO_NOT_INITIATE/SELL)
    - Adjusted scores (Health, Growth)
    - Risk assessment (Tally, Zone)
    - Chart control flags

    Args:
        pm_output: The full Portfolio Manager output text (final_trade_decision)

    Returns:
        PMBlockData with extracted values (defaults for missing fields)
    """
    if not pm_output:
        logger.debug("No PM output provided")
        return PMBlockData()

    # Find the PM_BLOCK section (take last one for self-correction pattern)
    # Tolerates optional descriptive text after "PM_BLOCK" (future-proofing)
    pm_block_pattern = r"### --- START PM_BLOCK[^\n]*---(.+?)### --- END PM_BLOCK ---"
    blocks = list(re.finditer(pm_block_pattern, pm_output, re.DOTALL))

    if not blocks:
        logger.warning(
            "no_pm_block_found",
            message="No PM_BLOCK found in Portfolio Manager output — charts will use fallback defaults",
        )
        return PMBlockData()

    # Use the last (most corrected) block
    pm_block = blocks[-1].group(1)

    # Extract verdict (normalize underscores/spaces)
    verdict_raw = _extract_str(r"VERDICT:\s*(\S+)", pm_block)
    verdict = None
    if verdict_raw:
        verdict = verdict_raw.upper().replace(" ", "_").replace("-", "_")
        # Normalize common variations
        if verdict in ("DO_NOT_INITIATE", "DONOTINITATE", "DONOTINITIATE"):
            verdict = "DO_NOT_INITIATE"

    # Determine show_valuation_chart based on verdict
    show_chart_raw = _extract_str(r"SHOW_VALUATION_CHART:\s*(YES|NO)", pm_block)
    if show_chart_raw:
        show_valuation_chart = show_chart_raw.upper() == "YES"
    else:
        # Default: show for BUY/HOLD, hide for negative verdicts
        show_valuation_chart = verdict in ("BUY", "HOLD", None)

    # Extract valuation discount (default based on zone if not specified)
    valuation_discount = _extract_float(r"VALUATION_DISCOUNT:\s*([\d.]+)", pm_block)
    zone = _extract_str(r"ZONE:\s*(LOW|MODERATE|HIGH)", pm_block)

    if valuation_discount is None:
        # Apply default discount based on zone
        if zone == "HIGH":
            valuation_discount = 0.8
        elif zone == "MODERATE":
            valuation_discount = 0.9
        else:
            valuation_discount = 1.0

    # For negative verdicts, force discount to 0 (suppresses targets)
    if verdict in ("DO_NOT_INITIATE", "SELL"):
        valuation_discount = 0.0

    result = PMBlockData(
        verdict=verdict,
        health_adj=_extract_int(r"HEALTH_ADJ:\s*(\d+)", pm_block),
        growth_adj=_extract_int(r"GROWTH_ADJ:\s*(\d+)", pm_block),
        risk_tally=_extract_float(r"RISK_TALLY:\s*([\d.]+)", pm_block),
        zone=zone,
        show_valuation_chart=show_valuation_chart,
        valuation_discount=valuation_discount,
        position_size=_extract_float(r"POSITION_SIZE:\s*([\d.]+)", pm_block),
        valuation_context=_extract_str(r"VALUATION_CONTEXT:\s*(\S+)", pm_block),
    )

    logger.debug(
        "Extracted PM_BLOCK data",
        verdict=result.verdict,
        health_adj=result.health_adj,
        growth_adj=result.growth_adj,
        risk_tally=result.risk_tally,
        zone=result.zone,
        show_valuation_chart=result.show_valuation_chart,
        valuation_discount=result.valuation_discount,
    )

    return result


def extract_verdict_from_text(pm_output: str) -> str | None:
    """Fallback: Extract verdict from PM output text when PM_BLOCK is missing.

    Searches for common verdict patterns in the PM's narrative output.
    Less reliable than PM_BLOCK extraction.

    Args:
        pm_output: The full Portfolio Manager output text

    Returns:
        Verdict string or None if not found
    """
    if not pm_output:
        return None

    # Look for explicit verdict header
    verdict_patterns = [
        r"PORTFOLIO MANAGER VERDICT:\s*(BUY|HOLD|DO NOT INITIATE|SELL)",
        r"\*\*Action\*\*:\s*(BUY|HOLD|DO NOT INITIATE|SELL)",
        r"Action:\s*(BUY|HOLD|DO NOT INITIATE|SELL)",
    ]

    for pattern in verdict_patterns:
        match = re.search(pattern, pm_output, re.IGNORECASE)
        if match:
            verdict = match.group(1).upper().replace(" ", "_")
            return verdict

    return None
