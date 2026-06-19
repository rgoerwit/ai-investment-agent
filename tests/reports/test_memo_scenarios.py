"""Tests for memo wiring of scenario valuation (Tranche 3, Step 6c)."""

from __future__ import annotations

import json
import pathlib
import re

from src.reporting.memo import (
    build_memo,
    extract_legacy_target_range,
    format_scenario_summary,
    render_memo_markdown,
)

_FUNDAMENTALS = (
    "## Fundamentals\n\n"
    "### --- START DATA_BLOCK ---\n"
    "SECTOR: Industrials\n"
    "PE_RATIO_TTM: 12.0\n"
    "EPS_TTM: 10.0\n"
    "CURRENT_PRICE: 100.00\n"
    "### --- END DATA_BLOCK ---\n"
)

_VALUATION_PARAMS = (
    "### --- START VALUATION_PARAMS ---\n"
    "METHOD: P/E_NORMALIZATION\n"
    "SECTOR: Industrials\n"
    "SECTOR_MEDIAN_PE: 17\n"
    "CURRENT_PE: 12.0\n"
    "PEG_RATIO: N/A\n"
    "GROWTH_SCORE_PCT: 65\n"
    "CURRENT_PRICE: 100.00\n"
    "CONFIDENCE: HIGH\n"
    "### --- END VALUATION_PARAMS ---\n\n"
    "### --- START VALUATION_SCENARIOS ---\n"
    "METHODOLOGY: P/E\n"
    "DATA_SUFFICIENCY: HIGH\n"
    "BEAR_MULTIPLE: 8\n"
    "BEAR_GROWTH_PCT: -5\n"
    "BEAR_MARGIN_DELTA_BPS: -200\n"
    "BEAR_DRIVERS: Cyclical trough, margin compression.\n"
    "BEAR_PROBABILITY: 30\n"
    "BASE_MULTIPLE: 12\n"
    "BASE_GROWTH_PCT: 8\n"
    "BASE_MARGIN_DELTA_BPS: 0\n"
    "BASE_DRIVERS: Mid-cycle base case, margins held.\n"
    "BASE_PROBABILITY: 50\n"
    "BULL_MULTIPLE: 16\n"
    "BULL_GROWTH_PCT: 15\n"
    "BULL_MARGIN_DELTA_BPS: 100\n"
    "BULL_DRIVERS: Cycle peak with 100bps margin expansion.\n"
    "BULL_PROBABILITY: 20\n"
    "### --- END VALUATION_SCENARIOS ---\n"
)


# ---------- format_scenario_summary ----------


def test_scenario_summary_when_block_parses() -> None:
    state = {
        "fundamentals_report": _FUNDAMENTALS,
        "valuation_params": _VALUATION_PARAMS,
    }
    out = format_scenario_summary(state)
    assert out is not None
    assert "Bear" in out and "Base" in out and "Bull" in out
    assert "weighted" in out
    assert "P/E" in out
    assert "HIGH" in out  # sufficiency tag


def test_scenario_summary_reads_saved_json_shape() -> None:
    saved = {
        "reports": {
            "fundamentals_report": _FUNDAMENTALS,
            "valuation_params": _VALUATION_PARAMS,
        }
    }
    assert format_scenario_summary(saved) is not None


def test_scenario_summary_none_when_block_missing() -> None:
    state = {
        "fundamentals_report": _FUNDAMENTALS,
        "valuation_params": "### --- START VALUATION_PARAMS ---\nMETHOD: P/E_NORMALIZATION\n### --- END VALUATION_PARAMS ---",
    }
    assert format_scenario_summary(state) is None


def test_scenario_summary_derives_eps_when_field_absent() -> None:
    """Tier 1 / Step 3 update: when EPS_TTM is absent but CURRENT_PRICE and
    PE_RATIO_TTM are present, the resolver derives EPS so scenarios still fire.

    (Pre-Tranche-5 this case fell back to legacy single-range — that was the
    bug, since the vast majority of real DATA_BLOCKs don't carry EPS_TTM.)
    """
    fundamentals_no_eps_field = (
        "### --- START DATA_BLOCK ---\n"
        "SECTOR: Industrials\n"
        "PE_RATIO_TTM: 12.0\n"
        "CURRENT_PRICE: 100.00\n"
        "### --- END DATA_BLOCK ---\n"
    )
    state = {
        "fundamentals_report": fundamentals_no_eps_field,
        "valuation_params": _VALUATION_PARAMS,
    }
    out = format_scenario_summary(state)
    assert out is not None
    assert "Bear" in out and "Base" in out and "Bull" in out


def test_scenario_summary_warns_when_valuation_is_conditional() -> None:
    fundamentals = (
        "#### CROSS-CHECK FLAGS\n"
        "- [CYCLICAL PEAK — LOW P/E MAY BE PEAK-DISTORTED]: returns above history.\n"
        "### --- START DATA_BLOCK ---\n"
        "SECTOR: Industrials\n"
        "PE_RATIO_TTM: 10.0\n"
        "PE_RATIO_FORWARD: 8.0\n"
        "CURRENT_PRICE: 100.00\n"
        "### --- END DATA_BLOCK ---\n"
    )
    out = format_scenario_summary(
        {"fundamentals_report": fundamentals, "valuation_params": _VALUATION_PARAMS}
    )
    assert out is not None
    assert "Warning: peak/distorted earnings flagged" in out
    assert "conditional, not normalized fair value" in out


def test_scenario_summary_drops_cents_on_large_nominal_values() -> None:
    """KRW/JPY-scale IVs run to thousands; two decimals there is false precision."""
    fundamentals = (
        "### --- START DATA_BLOCK ---\n"
        "SECTOR: Industrials\n"
        "REPORTING_CURRENCY: KRW\n"
        "PE_RATIO_TTM: 12.0\n"
        "EPS_TTM: 3000.0\n"
        "CURRENT_PRICE: 36000.00\n"
        "### --- END DATA_BLOCK ---\n"
    )
    out = format_scenario_summary(
        {"fundamentals_report": fundamentals, "valuation_params": _VALUATION_PARAMS}
    )
    assert out is not None
    assert "KRW" in out
    # No thousands-grouped value carries cents (e.g. "38,880.00").
    assert not re.search(r"\d,\d{3}\.\d", out)


def test_scenario_summary_keeps_cents_on_small_nominal_values() -> None:
    """USD-scale IVs (< 1000) keep two decimals — cents are meaningful there."""
    out = format_scenario_summary(
        {"fundamentals_report": _FUNDAMENTALS, "valuation_params": _VALUATION_PARAMS}
    )
    assert out is not None
    assert re.search(r"\d\.\d{2}", out)


def test_scenario_summary_none_when_eps_unresolvable() -> None:
    """Without EPS_TTM AND without (CURRENT_PRICE + PE_RATIO_TTM) → fallback."""
    fundamentals_no_inputs = (
        "### --- START DATA_BLOCK ---\n"
        "SECTOR: Industrials\n"
        "### --- END DATA_BLOCK ---\n"
    )
    state = {
        "fundamentals_report": fundamentals_no_inputs,
        "valuation_params": _VALUATION_PARAMS,
    }
    assert format_scenario_summary(state) is None


def test_scenario_summary_none_when_sufficiency_low() -> None:
    low_block = _VALUATION_PARAMS.replace(
        "DATA_SUFFICIENCY: HIGH", "DATA_SUFFICIENCY: LOW"
    )
    state = {
        "fundamentals_report": _FUNDAMENTALS,
        "valuation_params": low_block,
    }
    assert format_scenario_summary(state) is None


# ---------- end-to-end memo render ----------


def test_memo_uses_scenarios_when_available() -> None:
    state = {
        "final_trade_decision": (
            "#### PORTFOLIO MANAGER VERDICT: BUY\n\n"
            "### DECISION RATIONALE\n\n"
            "Cyclical mid-cycle entry with bear-case IV providing a stop anchor.\n\n"
            "### --- START PM_BLOCK ---\nVERDICT: BUY\n### --- END PM_BLOCK ---\n"
        ),
        "fundamentals_report": _FUNDAMENTALS,
        "valuation_params": _VALUATION_PARAMS,
    }
    memo = build_memo(state)
    assert "Bear" in memo.valuation
    assert "weighted" in memo.valuation
    md = render_memo_markdown(memo)
    assert "Bear" in md and "Bull" in md


def test_memo_falls_back_when_scenarios_unparseable() -> None:
    """Without a scenarios block the memo must still render — using legacy valuation."""
    state = {
        "final_trade_decision": (
            "### DECISION RATIONALE\n\nLegacy fallback.\n\n"
            "### --- START PM_BLOCK ---\nVERDICT: HOLD\n### --- END PM_BLOCK ---\n"
        ),
        "fundamentals_report": _FUNDAMENTALS,
        "valuation_context": (
            "VALUATION DATA (from Football Field Chart):\n"
            "- Methodology: P/E Normalization\n"
            "- Target Range: $90.00 - $110.00\n"
            "Fair Value (midpoint): $100.00\n"
            "- Current Price: $100.00\n"
        ),
    }
    memo = build_memo(state)
    # Legacy single-range string is what we get when scenarios are absent.
    assert "Target range" in memo.valuation
    assert "Bear" not in memo.valuation


def test_legacy_target_range_reads_saved_json_fundamentals() -> None:
    saved = {"reports": {"fundamentals_report": _FUNDAMENTALS}}
    assert (
        extract_legacy_target_range(saved)
        == "Current price 100.00; target range unavailable."
    )


# ---------- PM prompt regression ----------


def test_pm_prompt_includes_scenario_rationale_hint() -> None:
    data = json.loads(
        pathlib.Path("prompts/portfolio_manager.json").read_text(encoding="utf-8")
    )
    # Pin format, not value — version bumps are routine.
    assert re.match(r"^\d+\.\d+$", data["version"])
    msg = data["system_message"]
    assert "SCENARIO VALUATION HINT" in msg
    assert "BEAR_IV" in msg
    assert "WEIGHTED_IV" in msg
    # Prior reconciliation blocks still intact.
    assert "CONSULTANT_RESOLUTION" in msg
    assert "APAC_RESOLUTION" in msg
    assert "AUDITOR_RESOLUTION" in msg


def test_pm_growth_transition_exception_is_not_single_pe_cliff() -> None:
    data = json.loads(
        pathlib.Path("prompts/portfolio_manager.json").read_text(encoding="utf-8")
    )
    msg = data["system_message"]
    match = re.search(
        r"2\. \*\*Growth Transition Score\*\*:(?P<section>.*?)"
        r"\n3\. \*\*Liquidity FAIL\*\*",
        msg,
        flags=re.S,
    )
    assert match is not None
    section = match.group("section")

    assert "P/E <= 13.0" in section or "P/E ≤ 13.0" in section
    assert "Data-Vacuum Exception" in section
    assert "P/E < 12.0" not in section


def test_senior_prompt_requires_missing_growth_input_note() -> None:
    data = json.loads(
        pathlib.Path("prompts/fundamentals_analyst.json").read_text(encoding="utf-8")
    )
    msg = data["system_message"]

    assert "Growth Transition Detail" in msg
    assert "Missing growth inputs: <fields or NONE>" in msg
