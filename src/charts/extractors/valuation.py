"""
Extract valuation parameters from Valuation Calculator VALUATION_PARAMS block
and calculate price targets in Python (not LLM).

The Valuation Calculator agent extracts PARAMETERS; this module does the MATH.
This separation ensures deterministic calculations and avoids LLM arithmetic errors.
"""

import re
from dataclasses import dataclass

import structlog

from src.data_block_utils import extract_last_fenced_block

logger = structlog.get_logger(__name__)


@dataclass
class ValuationParams:
    """Raw parameters extracted from Valuation Calculator."""

    method: str | None = None
    sector: str | None = None
    sector_median_pe: float | None = None
    current_pe: float | None = None
    peg_ratio: float | None = None
    growth_score_pct: float | None = None
    current_price: float | None = None
    confidence: str | None = None


@dataclass
class ValuationTargets:
    """Calculated valuation targets (Python-computed, not LLM)."""

    low: float | None = None
    high: float | None = None
    methodology: str | None = None
    confidence: str | None = None


def _extract_params(valuation_params_report: str) -> ValuationParams:
    """Extract raw parameters from VALUATION_PARAMS block.

    Block format:
    ### --- START VALUATION_PARAMS ---
    METHOD: P/E_NORMALIZATION
    SECTOR: Technology
    SECTOR_MEDIAN_PE: 25
    CURRENT_PE: 18.5
    PEG_RATIO: N/A
    GROWTH_SCORE_PCT: N/A
    CURRENT_PRICE: 150.00
    CONFIDENCE: HIGH
    ### --- END VALUATION_PARAMS ---
    """
    if not valuation_params_report:
        return ValuationParams()

    block = extract_last_fenced_block(valuation_params_report, "VALUATION_PARAMS")
    if not block:
        logger.debug("No VALUATION_PARAMS block found")
        return ValuationParams()

    def extract_float(pattern: str) -> float | None:
        m = re.search(pattern, block, re.IGNORECASE)
        if m:
            try:
                value_str = m.group(1).strip()
                if value_str.upper() in ("N/A", "NA", "NONE", "-", ""):
                    return None
                # Remove currency symbols and commas
                value_str = value_str.replace(",", "").replace("$", "")
                return float(value_str)
            except (ValueError, IndexError):
                return None
        return None

    def extract_str(pattern: str) -> str | None:
        m = re.search(pattern, block, re.IGNORECASE)
        if m:
            value = m.group(1).strip()
            if value.upper() in ("N/A", "NA", "NONE", "-", ""):
                return None
            return value
        return None

    return ValuationParams(
        method=extract_str(r"METHOD:\s*(.+?)(?:\n|$)"),
        sector=extract_str(r"SECTOR:\s*(.+?)(?:\n|$)"),
        sector_median_pe=extract_float(r"SECTOR_MEDIAN_PE:\s*([\d,.]+|N/A)"),
        current_pe=extract_float(r"CURRENT_PE:\s*([\d,.]+|N/A)"),
        peg_ratio=extract_float(r"PEG_RATIO:\s*([\d,.]+|N/A)"),
        growth_score_pct=extract_float(r"GROWTH_SCORE_PCT:\s*([\d,.]+|N/A)"),
        current_price=extract_float(r"CURRENT_PRICE:\s*\$?([\d,.]+)"),
        confidence=extract_str(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)"),
    )


def _calculate_pe_normalization(params: ValuationParams) -> ValuationTargets:
    """Calculate targets using P/E normalization to sector median.

    Formula:
        Fair Value = Current Price × (Sector Median PE / Current PE)
        Target Low = Fair Value × 0.85 (conservative discount)
        Target High = Fair Value × 1.15 (growth premium)
    """
    if (
        params.current_price is None
        or params.current_pe is None
        or params.sector_median_pe is None
        or params.current_pe <= 0
    ):
        logger.warning("Insufficient data for P/E normalization calculation")
        return ValuationTargets(confidence=params.confidence)

    # Calculate fair value
    pe_ratio = params.sector_median_pe / params.current_pe
    fair_value = params.current_price * pe_ratio

    # Apply conservative discount/premium for range
    target_low = round(fair_value * 0.85, 2)
    target_high = round(fair_value * 1.15, 2)

    methodology = (
        f"P/E normalization: Current PE {params.current_pe:.1f} → "
        f"Sector median PE {params.sector_median_pe:.1f} ({params.sector or 'General'})"
    )

    logger.debug(
        "P/E normalization calculation",
        current_price=params.current_price,
        current_pe=params.current_pe,
        sector_median_pe=params.sector_median_pe,
        fair_value=fair_value,
        target_low=target_low,
        target_high=target_high,
    )

    return ValuationTargets(
        low=target_low,
        high=target_high,
        methodology=methodology,
        confidence=params.confidence,
    )


def _calculate_peg_based(params: ValuationParams) -> ValuationTargets:
    """Calculate targets using PEG ratio.

    Formula:
        If PEG < 1.0: Stock is undervalued by (1 - PEG) * 100%
        Target = Current Price × (1 / PEG) (mean reversion to fair value)
        Apply ±15% range
    """
    if (
        params.current_price is None
        or params.peg_ratio is None
        or params.peg_ratio <= 0
    ):
        logger.warning("Insufficient data for PEG-based calculation")
        return ValuationTargets(confidence=params.confidence)

    # PEG-implied fair value (assuming fair PEG = 1.0)
    # If PEG is 0.8, stock is undervalued by 20%, fair value = price * 1.25
    fair_value = params.current_price * (1 / params.peg_ratio)

    # Constrain fair value to reasonable bounds (avoid extreme values)
    max_upside = params.current_price * 2.0  # Cap at 100% upside
    fair_value = min(fair_value, max_upside)

    # Apply conservative discount/premium for range
    target_low = round(fair_value * 0.85, 2)
    target_high = round(fair_value * 1.15, 2)

    methodology = f"PEG-based: PEG ratio {params.peg_ratio:.2f} implies fair value at ${fair_value:.2f}"

    logger.debug(
        "PEG-based calculation",
        current_price=params.current_price,
        peg_ratio=params.peg_ratio,
        fair_value=fair_value,
        target_low=target_low,
        target_high=target_high,
    )

    return ValuationTargets(
        low=target_low,
        high=target_high,
        methodology=methodology,
        confidence=params.confidence,
    )


def _calculate_growth_adjusted(params: ValuationParams) -> ValuationTargets:
    """Calculate targets using growth score (fallback method).

    Formula:
        Upside % = Growth Score % × 0.5 (conservative multiplier)
        Target = Current Price × (1 + Upside %)
        Apply ±10% range (tighter for lower confidence method)
    """
    if params.current_price is None or params.growth_score_pct is None:
        logger.warning("Insufficient data for growth-adjusted calculation")
        return ValuationTargets(confidence=params.confidence)

    # Growth score as upside indicator (conservative 0.5x multiplier)
    # Growth score of 65% → 32.5% potential upside
    upside_pct = params.growth_score_pct * 0.5 / 100  # Convert to decimal

    # Cap upside at 50% for this fallback method
    upside_pct = min(upside_pct, 0.5)

    fair_value = params.current_price * (1 + upside_pct)

    # Tighter range for lower-confidence method
    target_low = round(fair_value * 0.90, 2)
    target_high = round(fair_value * 1.10, 2)

    methodology = (
        f"Growth-adjusted: {params.growth_score_pct:.0f}% growth score "
        f"implies {upside_pct * 100:.1f}% potential upside"
    )

    logger.debug(
        "Growth-adjusted calculation",
        current_price=params.current_price,
        growth_score_pct=params.growth_score_pct,
        upside_pct=upside_pct,
        fair_value=fair_value,
        target_low=target_low,
        target_high=target_high,
    )

    return ValuationTargets(
        low=target_low,
        high=target_high,
        methodology=methodology,
        confidence=params.confidence or "LOW",
    )


def calculate_valuation_targets(valuation_params_report: str) -> ValuationTargets:
    """Extract parameters and calculate targets using appropriate method.

    This is the main entry point. It:
    1. Extracts VALUATION_PARAMS from the Valuation Calculator output
    2. Dispatches to the appropriate calculation method
    3. Returns Python-calculated targets (no LLM math)

    Args:
        valuation_params_report: Output from Valuation Calculator agent

    Returns:
        ValuationTargets with calculated values (None for insufficient data)
    """
    params = _extract_params(valuation_params_report)

    if not params.method:
        logger.debug("No valuation method specified")
        return ValuationTargets()

    method = params.method.upper().replace(" ", "_").replace("-", "_")

    logger.info(
        "Calculating valuation targets",
        method=method,
        current_price=params.current_price,
        confidence=params.confidence,
    )

    if method == "P/E_NORMALIZATION" or method == "PE_NORMALIZATION":
        return _calculate_pe_normalization(params)
    elif method == "PEG_BASED":
        return _calculate_peg_based(params)
    elif method == "GROWTH_ADJUSTED":
        return _calculate_growth_adjusted(params)
    elif method == "INSUFFICIENT_DATA":
        logger.info("Valuation Calculator reported insufficient data")
        return ValuationTargets(
            methodology="Insufficient data for valuation",
            confidence="LOW",
        )
    else:
        logger.warning(f"Unknown valuation method: {method}")
        return ValuationTargets()


# Backward compatibility alias
def extract_valuation_targets(research_manager_report: str) -> ValuationTargets:
    """DEPRECATED: Use calculate_valuation_targets with valuation_params_report instead.

    This function is kept for backward compatibility but now expects
    VALUATION_PARAMS format (from Valuation Calculator agent).
    """
    return calculate_valuation_targets(research_manager_report)


# ---------------------------------------------------------------------------
# Scenario valuation (Tranche 3, Step 6)
#
# Bear / base / bull *assumptions* parsed from the VALUATION_SCENARIOS block;
# Python computes the intrinsic values. This preserves the long-standing
# "agent extracts, Python computes" contract — see the prompt at
# prompts/valuation_calculator.json (v1.3+).
# ---------------------------------------------------------------------------


@dataclass
class ScenarioAssumption:
    """One bear/base/bull scenario row emitted by the Valuation Calculator."""

    multiple: float
    growth_pct: float
    margin_delta_bps: float
    drivers: str
    probability: float


@dataclass
class ValuationScenarios:
    """Three-case scenario valuation with Python-computed intrinsic values.

    `weighted_iv` is the probability-weighted mean of the three IVs; by
    construction it lies in ``[min(bear_iv, base_iv, bull_iv), max(...)]``.
    """

    methodology: str
    data_sufficiency: str
    bear: ScenarioAssumption
    base: ScenarioAssumption
    bull: ScenarioAssumption
    bear_iv: float
    base_iv: float
    bull_iv: float
    weighted_iv: float


def _scenario_iv(eps_ttm: float, scenario: ScenarioAssumption) -> float:
    """Compute one scenario IV from the agent-emitted assumptions.

    IV = EPS_TTM × (1 + growth/100) × multiple × (1 + margin_delta_bps/10000)
    """
    growth_factor = 1.0 + scenario.growth_pct / 100.0
    margin_factor = 1.0 + scenario.margin_delta_bps / 10000.0
    return round(eps_ttm * growth_factor * scenario.multiple * margin_factor, 2)


def _parse_scenario(block: str, label: str) -> ScenarioAssumption | None:
    """Parse one labeled scenario (BEAR / BASE / BULL) out of the fenced block."""

    def _num(field: str) -> float | None:
        m = re.search(rf"{label}_{field}:\s*(-?[\d.]+)", block, re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    multiple = _num("MULTIPLE")
    growth = _num("GROWTH_PCT")
    margin = _num("MARGIN_DELTA_BPS")
    probability = _num("PROBABILITY")
    drivers_match = re.search(
        rf"{label}_DRIVERS:\s*(.+?)(?:\n|$)", block, re.IGNORECASE
    )
    if multiple is None or growth is None or margin is None or probability is None:
        return None
    drivers = drivers_match.group(1).strip() if drivers_match else ""
    return ScenarioAssumption(
        multiple=multiple,
        growth_pct=growth,
        margin_delta_bps=margin,
        drivers=drivers,
        probability=probability,
    )


def extract_valuation_scenarios(
    valuation_params_report: str,
    eps_ttm: float | None,
) -> ValuationScenarios | None:
    """Parse VALUATION_SCENARIOS and compute IVs in Python.

    Returns ``None`` (caller falls back to legacy single-range output) when
    any of the following holds:

    - block missing
    - ``DATA_SUFFICIENCY: LOW`` (agent declined to emit scenarios)
    - probabilities don't sum to ~100
    - any scenario row fails to parse
    - multiples are inverted (``bear_multiple > bull_multiple``) — fabrication signal
    - ``eps_ttm`` is missing or non-positive
    """
    if not valuation_params_report:
        return None
    block = extract_last_fenced_block(valuation_params_report, "VALUATION_SCENARIOS")
    if not block:
        return None

    suff_match = re.search(r"DATA_SUFFICIENCY:\s*(\w+)", block, re.IGNORECASE)
    data_sufficiency = suff_match.group(1).upper() if suff_match else "LOW"
    if data_sufficiency == "LOW":
        logger.debug("Valuation scenarios suppressed: DATA_SUFFICIENCY=LOW")
        return None

    if eps_ttm is None or eps_ttm <= 0:
        logger.debug("Valuation scenarios suppressed: eps_ttm missing or non-positive")
        return None

    method_match = re.search(r"METHODOLOGY:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
    methodology = method_match.group(1).strip() if method_match else "P/E"

    bear = _parse_scenario(block, "BEAR")
    base = _parse_scenario(block, "BASE")
    bull = _parse_scenario(block, "BULL")
    if bear is None or base is None or bull is None:
        logger.debug("Valuation scenarios: one or more scenario rows failed to parse")
        return None

    prob_sum = bear.probability + base.probability + bull.probability
    if abs(prob_sum - 100.0) > 1.0:
        logger.warning(
            "Valuation scenarios rejected: probabilities sum to %.1f (expected 100)",
            prob_sum,
        )
        return None

    if bear.multiple > bull.multiple:
        logger.warning(
            "Valuation scenarios rejected: bear_multiple (%.2f) > bull_multiple (%.2f)",
            bear.multiple,
            bull.multiple,
        )
        return None

    bear_iv = _scenario_iv(eps_ttm, bear)
    base_iv = _scenario_iv(eps_ttm, base)
    bull_iv = _scenario_iv(eps_ttm, bull)

    if bear_iv > bull_iv:
        # Possible after applying growth + margin deltas even with ordered multiples.
        logger.warning(
            "Valuation scenarios rejected: bear_iv (%.2f) > bull_iv (%.2f) post-compute",
            bear_iv,
            bull_iv,
        )
        return None

    weighted_iv = round(
        (
            bear_iv * bear.probability
            + base_iv * base.probability
            + bull_iv * bull.probability
        )
        / 100.0,
        2,
    )

    return ValuationScenarios(
        methodology=methodology,
        data_sufficiency=data_sufficiency,
        bear=bear,
        base=base,
        bull=bull,
        bear_iv=bear_iv,
        base_iv=base_iv,
        bull_iv=bull_iv,
        weighted_iv=weighted_iv,
    )
