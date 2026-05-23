"""Tests for the scenario-valuation parser + Python IV computation (Tranche 3)."""

from __future__ import annotations

import json
import pathlib

import pytest

from src.charts.extractors.valuation import (
    ScenarioAssumption,
    ValuationScenarios,
    _scenario_iv,
    extract_valuation_scenarios,
)

# Convention: EPS_TTM 10.0 keeps math easy to eyeball.
EPS_TTM = 10.0


def _block(
    *,
    methodology: str = "P/E",
    data_sufficiency: str = "HIGH",
    bear: tuple[float, float, float, int] = (8, -5, -200, 30),
    base: tuple[float, float, float, int] = (12, 8, 0, 50),
    bull: tuple[float, float, float, int] = (16, 15, 100, 20),
    omit: set[str] | None = None,
) -> str:
    """Assemble a VALUATION_SCENARIOS block. `omit` drops named fields (for error tests)."""
    omit = omit or set()
    lines = ["### --- START VALUATION_SCENARIOS ---"]
    if "METHODOLOGY" not in omit:
        lines.append(f"METHODOLOGY: {methodology}")
    if "DATA_SUFFICIENCY" not in omit:
        lines.append(f"DATA_SUFFICIENCY: {data_sufficiency}")
    for label, vals in (("BEAR", bear), ("BASE", base), ("BULL", bull)):
        mult, growth, margin, prob = vals
        if f"{label}_MULTIPLE" not in omit:
            lines.append(f"{label}_MULTIPLE: {mult}")
        if f"{label}_GROWTH_PCT" not in omit:
            lines.append(f"{label}_GROWTH_PCT: {growth}")
        if f"{label}_MARGIN_DELTA_BPS" not in omit:
            lines.append(f"{label}_MARGIN_DELTA_BPS: {margin}")
        if f"{label}_DRIVERS" not in omit:
            lines.append(f"{label}_DRIVERS: {label.title()} case driver text.")
        if f"{label}_PROBABILITY" not in omit:
            lines.append(f"{label}_PROBABILITY: {prob}")
    lines.append("### --- END VALUATION_SCENARIOS ---")
    return "\n".join(lines)


# ---------- prompt regression ----------


def test_valuation_calculator_prompt_extended() -> None:
    data = json.loads(
        pathlib.Path("prompts/valuation_calculator.json").read_text(encoding="utf-8")
    )
    assert data["version"] == "1.3"
    msg = data["system_message"]
    assert "VALUATION_SCENARIOS" in msg
    assert "BEAR_MULTIPLE" in msg
    assert "BULL_PROBABILITY" in msg
    assert "DATA_SUFFICIENCY" in msg
    assert "DO NOT COMPUTE IVs" in msg
    # Original VALUATION_PARAMS contract preserved.
    assert "VALUATION_PARAMS" in msg
    assert "DO NOT CALCULATE" in msg


# ---------- _scenario_iv math ----------


def test_scenario_iv_baseline_growth_zero_margin_zero() -> None:
    """IV = EPS × (1+0) × multiple × (1+0) = EPS × multiple."""
    s = ScenarioAssumption(
        multiple=12.0, growth_pct=0.0, margin_delta_bps=0.0, drivers="", probability=50
    )
    assert _scenario_iv(10.0, s) == 120.0


def test_scenario_iv_applies_growth_and_margin_deltas() -> None:
    s = ScenarioAssumption(
        multiple=10.0,
        growth_pct=10.0,
        margin_delta_bps=200.0,  # +2%
        drivers="",
        probability=50,
    )
    # 10 × 1.10 × 10 × 1.02 = 112.20
    assert _scenario_iv(10.0, s) == 112.20


def test_scenario_iv_handles_negative_growth_and_margin() -> None:
    s = ScenarioAssumption(
        multiple=8.0,
        growth_pct=-10.0,
        margin_delta_bps=-300.0,  # -3%
        drivers="",
        probability=30,
    )
    # 10 × 0.90 × 8 × 0.97 = 69.84
    assert _scenario_iv(10.0, s) == 69.84


# ---------- extract_valuation_scenarios: happy path ----------


def test_extract_happy_path_returns_full_scenarios() -> None:
    scen = extract_valuation_scenarios(_block(), eps_ttm=EPS_TTM)
    assert isinstance(scen, ValuationScenarios)
    assert scen.methodology == "P/E"
    assert scen.data_sufficiency == "HIGH"
    assert scen.bear.probability == 30
    assert scen.base.probability == 50
    assert scen.bull.probability == 20
    assert scen.bear_iv == _scenario_iv(EPS_TTM, scen.bear)
    assert scen.base_iv == _scenario_iv(EPS_TTM, scen.base)
    assert scen.bull_iv == _scenario_iv(EPS_TTM, scen.bull)


def test_weighted_iv_lies_in_min_max_envelope() -> None:
    """Critical correctness check (reviewer correction #7):
    Σ p_i × iv_i / 100 must lie in [min iv, max iv] — not just between base and bull.
    """
    # Bear-heavy probability so weighted should sit below base.
    scen = extract_valuation_scenarios(
        _block(
            bear=(8, -10, -300, 70),  # bear IV pulled lower, high probability
            base=(12, 5, 0, 20),
            bull=(16, 15, 100, 10),
        ),
        eps_ttm=EPS_TTM,
    )
    assert scen is not None
    ivs = [scen.bear_iv, scen.base_iv, scen.bull_iv]
    assert min(ivs) <= scen.weighted_iv <= max(ivs)
    # And specifically below base_iv given the heavy bear weight.
    assert scen.weighted_iv < scen.base_iv


def test_weighted_iv_arithmetic_matches_hand_calc() -> None:
    """Manual cross-check of the weighted-mean formula."""
    scen = extract_valuation_scenarios(_block(), eps_ttm=EPS_TTM)
    assert scen is not None
    expected = round(
        (scen.bear_iv * 30 + scen.base_iv * 50 + scen.bull_iv * 20) / 100.0, 2
    )
    assert scen.weighted_iv == expected


def test_drivers_captured() -> None:
    scen = extract_valuation_scenarios(_block(), eps_ttm=EPS_TTM)
    assert scen is not None
    assert scen.bear.drivers == "Bear case driver text."
    assert scen.bull.drivers == "Bull case driver text."


# ---------- extract_valuation_scenarios: edge cases ----------


def test_data_sufficiency_low_returns_none() -> None:
    assert (
        extract_valuation_scenarios(_block(data_sufficiency="LOW"), eps_ttm=EPS_TTM)
        is None
    )


def test_missing_block_returns_none() -> None:
    assert (
        extract_valuation_scenarios("no scenario block here", eps_ttm=EPS_TTM) is None
    )
    assert extract_valuation_scenarios("", eps_ttm=EPS_TTM) is None


def test_eps_missing_or_nonpositive_returns_none() -> None:
    block = _block()
    assert extract_valuation_scenarios(block, eps_ttm=None) is None
    assert extract_valuation_scenarios(block, eps_ttm=0.0) is None
    assert extract_valuation_scenarios(block, eps_ttm=-2.5) is None


# ---------- extract_valuation_scenarios: error / sanity checks ----------


def test_probabilities_must_sum_to_100() -> None:
    bad = _block(bear=(8, -5, -200, 30), base=(12, 8, 0, 40), bull=(16, 15, 100, 20))
    assert extract_valuation_scenarios(bad, eps_ttm=EPS_TTM) is None


def test_probability_sum_tolerates_rounding() -> None:
    """Sum of 99 or 101 still accepted (±1 tolerance)."""
    almost = _block(bear=(8, -5, -200, 29), base=(12, 8, 0, 50), bull=(16, 15, 100, 20))
    assert extract_valuation_scenarios(almost, eps_ttm=EPS_TTM) is not None


def test_inverted_multiples_rejected_as_fabrication() -> None:
    bad = _block(
        bear=(20, -5, -200, 30),  # bear multiple > bull multiple
        base=(12, 8, 0, 50),
        bull=(8, 15, 100, 20),
    )
    assert extract_valuation_scenarios(bad, eps_ttm=EPS_TTM) is None


def test_post_compute_inverted_iv_rejected() -> None:
    """Ordered multiples but extreme growth/margin deltas flip the IV order — also rejected."""
    bad = _block(
        bear=(10, 100, 500, 30),  # huge upside in 'bear'
        base=(12, 0, 0, 50),
        bull=(14, -50, -500, 20),  # crashed bull
    )
    assert extract_valuation_scenarios(bad, eps_ttm=EPS_TTM) is None


@pytest.mark.parametrize(
    "missing", ["BEAR_MULTIPLE", "BASE_GROWTH_PCT", "BULL_PROBABILITY"]
)
def test_missing_required_field_returns_none(missing: str) -> None:
    block = _block(omit={missing})
    assert extract_valuation_scenarios(block, eps_ttm=EPS_TTM) is None


def test_unparseable_block_returns_none() -> None:
    block = (
        "### --- START VALUATION_SCENARIOS ---\n"
        "Garbage text with no recognizable fields.\n"
        "### --- END VALUATION_SCENARIOS ---"
    )
    assert extract_valuation_scenarios(block, eps_ttm=EPS_TTM) is None
