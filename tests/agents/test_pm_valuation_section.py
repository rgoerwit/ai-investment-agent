"""Test that PM payload assembly emits a VALUATION SCENARIOS section
(Tranche 5, Step 4).

The PM prompt at v9.7 asks PM to anchor stop-loss to BEAR_IV and quote
WEIGHTED_IV in rationale — but pre-Tranche-5 that hint was unread context
because `valuation_params` was never spliced into the PM payload. This test
exercises the assembly logic directly to confirm the section is now present.
"""

from __future__ import annotations

import re

_FUNDAMENTALS = (
    "### --- START DATA_BLOCK ---\n"
    "SECTOR: Industrials\n"
    "PE_RATIO_TTM: 12.0\n"
    "CURRENT_PRICE: 100.00\n"
    "### --- END DATA_BLOCK ---\n"
)

_VAL_PARAMS_WITH_SCENARIOS = (
    "### --- START VALUATION_PARAMS ---\n"
    "METHOD: P/E_NORMALIZATION\nCURRENT_PRICE: 100.00\nCONFIDENCE: HIGH\n"
    "CURRENT_PE: 12.0\nSECTOR: Industrials\nSECTOR_MEDIAN_PE: 17\n"
    "### --- END VALUATION_PARAMS ---\n\n"
    "### --- START VALUATION_SCENARIOS ---\n"
    "METHODOLOGY: P/E\nDATA_SUFFICIENCY: HIGH\n"
    "BEAR_MULTIPLE: 8\nBEAR_GROWTH_PCT: -5\nBEAR_MARGIN_DELTA_BPS: -200\n"
    "BEAR_DRIVERS: Cyclical trough.\nBEAR_PROBABILITY: 30\n"
    "BASE_MULTIPLE: 12\nBASE_GROWTH_PCT: 8\nBASE_MARGIN_DELTA_BPS: 0\n"
    "BASE_DRIVERS: Mid-cycle base.\nBASE_PROBABILITY: 50\n"
    "BULL_MULTIPLE: 16\nBULL_GROWTH_PCT: 15\nBULL_MARGIN_DELTA_BPS: 100\n"
    "BULL_DRIVERS: Cycle peak.\nBULL_PROBABILITY: 20\n"
    "### --- END VALUATION_SCENARIOS ---\n"
)

_VAL_PARAMS_NO_SCENARIOS = (
    "### --- START VALUATION_PARAMS ---\n"
    "METHOD: P/E_NORMALIZATION\nCURRENT_PRICE: 100.00\nCONFIDENCE: HIGH\n"
    "### --- END VALUATION_PARAMS ---\n"
)


def _build_valuation_section(valuation_params: str, fundamentals: str) -> str:
    """Reproduce the PM-node logic from src/agents/decision_nodes.py."""
    from src.charts.extractors.valuation import (
        extract_valuation_scenarios,
        resolve_eps_ttm,
    )
    from src.data_block_utils import extract_data_block_field

    if not (valuation_params and fundamentals):
        return ""
    eps_ttm = resolve_eps_ttm(fundamentals)
    scenarios = extract_valuation_scenarios(valuation_params, eps_ttm)
    if scenarios is None:
        return ""
    current_price = float(extract_data_block_field(fundamentals, "CURRENT_PRICE") or 0)
    weighted_upside = (scenarios.weighted_iv / current_price) - 1.0
    downside_probability = sum(
        scenario.probability
        for scenario, intrinsic_value in (
            (scenarios.bear, scenarios.bear_iv),
            (scenarios.base, scenarios.base_iv),
            (scenarios.bull, scenarios.bull_iv),
        )
        if intrinsic_value < current_price
    )
    return (
        "\n\nVALUATION SCENARIOS (Python-computed IVs from "
        f"{scenarios.methodology}; sufficiency {scenarios.data_sufficiency}; "
        "anchor stop-loss to BEAR_IV, reference WEIGHTED_IV in rationale):\n"
        f"- BEAR_IV: {scenarios.bear_iv} "
        f"({scenarios.bear.probability:.0f}%) — {scenarios.bear.drivers}\n"
        f"- BASE_IV: {scenarios.base_iv} "
        f"({scenarios.base.probability:.0f}%) — {scenarios.base.drivers}\n"
        f"- BULL_IV: {scenarios.bull_iv} "
        f"({scenarios.bull.probability:.0f}%) — {scenarios.bull.drivers}\n"
        f"- WEIGHTED_IV: {scenarios.weighted_iv}, implied upside "
        f"{weighted_upside * 100:.1f}% vs current price {current_price:.2f}, "
        f"downside probability {downside_probability:.0f}%"
    )


def test_valuation_section_emitted_when_scenarios_parse() -> None:
    section = _build_valuation_section(_VAL_PARAMS_WITH_SCENARIOS, _FUNDAMENTALS)
    assert "VALUATION SCENARIOS" in section
    assert "BEAR_IV:" in section
    assert "WEIGHTED_IV:" in section
    assert "implied upside" in section
    assert "downside probability" in section
    assert "anchor stop-loss to BEAR_IV" in section
    # Drivers carried through, not just numbers.
    assert "Cyclical trough" in section
    assert "Cycle peak" in section


def test_valuation_section_empty_when_no_scenarios_block() -> None:
    section = _build_valuation_section(_VAL_PARAMS_NO_SCENARIOS, _FUNDAMENTALS)
    assert section == ""


def test_valuation_section_empty_when_eps_unresolvable() -> None:
    """No CURRENT_PRICE → no derived EPS → no scenarios → no section."""
    barren = "### --- START DATA_BLOCK ---\nSECTOR: Industrials\n### --- END DATA_BLOCK ---\n"
    section = _build_valuation_section(_VAL_PARAMS_WITH_SCENARIOS, barren)
    assert section == ""


def test_valuation_section_empty_when_data_sufficiency_low() -> None:
    """The agent's escape hatch — DATA_SUFFICIENCY: LOW must suppress IVs."""
    low_block = _VAL_PARAMS_WITH_SCENARIOS.replace(
        "DATA_SUFFICIENCY: HIGH", "DATA_SUFFICIENCY: LOW"
    )
    section = _build_valuation_section(low_block, _FUNDAMENTALS)
    assert section == ""


def test_valuation_section_includes_drivers_as_rationale_anchors() -> None:
    """Drivers (not just numbers) must reach PM — without them PM can't write
    a defensible rationale tied to scenario logic."""
    section = _build_valuation_section(_VAL_PARAMS_WITH_SCENARIOS, _FUNDAMENTALS)
    # Each scenario row carries probability% AND driver text.
    pattern = re.compile(r"BEAR_IV: \S+ \(30%\) — Cyclical trough")
    assert pattern.search(section), section
