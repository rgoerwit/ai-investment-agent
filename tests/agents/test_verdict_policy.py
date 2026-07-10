"""Step 6: deterministic Data-Vacuum / Marginal-Turnaround verdict floor."""

from __future__ import annotations

from src.agents.verdict_policy import (
    maybe_demote_buy_on_blocking_flags,
    maybe_floor_verdict_to_hold,
    maybe_qualify_buy_in_quick_mode,
)
from src.charts.extractors.pm_block import extract_pm_block


def _pm_output(verdict_display: str, verdict_block: str) -> str:
    return (
        f"### PORTFOLIO MANAGER VERDICT: {verdict_display}\n\n"
        "Rationale text here.\n\n"
        "### CONSULTANT DISAGREEMENT RESOLUTION\n"
        "- VERDICT: CONFIRMED_RISK (+0.25, Tier C1 conflict)\n\n"
        "```\nPM_BLOCK\n"
        f"VERDICT: {verdict_block}\n"
        "POSITION_SIZE: 0.0%\n"
        "```\n"
    )


def _full_pm_output(verdict_display: str, verdict_block: str) -> str:
    """A realistic DO_NOT_INITIATE PM_BLOCK with the verdict-coupled chart fields."""
    return (
        f"### PORTFOLIO MANAGER VERDICT: {verdict_display}\n\n"
        "```\n"
        "=== DECISION LOGIC ===\n"
        "Default Decision: DO NOT INITIATE\n"
        f"Actual Decision: {verdict_display}\n"
        "======================\n"
        "```\n\n"
        "### FINAL EXECUTION PARAMETERS\n\n"
        f"**Action**: {verdict_display}\n"
        "TRADE_BLOCK:\n"
        "ACTION: BUY\n\n"
        "**Action Required**:\n"
        "Re-run analysis with verbose logging.\n\n"
        "### --- START PM_BLOCK ---\n"
        f"VERDICT: {verdict_block}\n"
        "HEALTH_ADJ: 75\nGROWTH_ADJ: 33\nRISK_TALLY: 1.0\nZONE: HIGH\n"
        "SHOW_VALUATION_CHART: NO\n"
        "VALUATION_DISCOUNT: 0.0\n"
        "POSITION_SIZE: 0.0%\n"
        "### --- END PM_BLOCK ---\n"
    )


def _data_block(health, pe, growth, cagr) -> str:
    return (
        "### --- START DATA_BLOCK ---\n"
        f"ADJUSTED_HEALTH_SCORE: {health}% (based on 12 available points)\n"
        f"ADJUSTED_GROWTH_SCORE: {growth}% (based on 6 available points)\n"
        f"PE_RATIO_TTM: {pe}\n"
        f"REVENUE_CAGR_3Y: {cagr}%\n"
        "### --- END DATA_BLOCK ---"
    )


# APR.WA profile: healthy, cheap, growth-data-vacuum, positive 3Y CAGR.
APR_BLOCK = _data_block(75, 14.88, 33, 16.0)
# KTY.WA profile: P/E > 18 AND negative 3Y CAGR — a true ex-growth name.
KTY_BLOCK = _data_block(67, 20.54, 33, -2.5)


def test_floor_normalizes_chart_control_fields():
    out, floored = maybe_floor_verdict_to_hold(
        _full_pm_output("DO NOT INITIATE", "DO_NOT_INITIATE"),
        fundamentals_report=APR_BLOCK,
        red_flags=[],
        code_subtotal=1.0,
        pre_screening_result="PASS",
        ticker="APR.WA",
    )
    assert floored is True
    assert "SHOW_VALUATION_CHART: YES" in out
    assert "SHOW_VALUATION_CHART: NO" not in out
    assert "VALUATION_DISCOUNT: 0.8" in out
    assert "VALUATION_DISCOUNT: 0.0" not in out
    assert "Actual Decision: HOLD" in out
    assert "**Action**: HOLD" in out
    assert "ACTION: BUY" in out
    assert "**Action Required**:" in out
    # Downstream chart extractor must now agree the verdict is non-negative.
    block = extract_pm_block(out)
    assert block.verdict == "HOLD"
    assert block.show_valuation_chart is True
    assert block.should_show_targets() is True
    assert block.valuation_discount == 0.8


def test_apr_floored_to_hold():
    out, floored = maybe_floor_verdict_to_hold(
        _pm_output("DO NOT INITIATE", "DO_NOT_INITIATE"),
        fundamentals_report=APR_BLOCK,
        red_flags=[{"type": "VALUE_TRAP_MODERATE_RISK", "risk_penalty": 0.5}],
        code_subtotal=1.0,
        pre_screening_result="PASS",
        ticker="APR.WA",
    )
    assert floored is True
    assert "VERDICT: HOLD" in out
    assert "VERDICT: DO_NOT_INITIATE" not in out
    assert "PORTFOLIO MANAGER VERDICT: HOLD" in out
    assert "VERDICT FLOOR APPLIED" in out
    # consultant-resolution VERDICT lines must be untouched
    assert "VERDICT: CONFIRMED_RISK" in out


def test_kty_not_floored_pe_and_negative_cagr():
    out, floored = maybe_floor_verdict_to_hold(
        _pm_output("DO NOT INITIATE", "DO_NOT_INITIATE"),
        fundamentals_report=KTY_BLOCK,
        red_flags=[],
        code_subtotal=1.0,
        pre_screening_result="PASS",
        ticker="KTY.WA",
    )
    assert floored is False
    assert out == _pm_output("DO NOT INITIATE", "DO_NOT_INITIATE")


def test_not_floored_when_subtotal_at_zone1():
    _, floored = maybe_floor_verdict_to_hold(
        _pm_output("DO NOT INITIATE", "DO_NOT_INITIATE"),
        fundamentals_report=APR_BLOCK,
        red_flags=[],
        code_subtotal=2.0,
        pre_screening_result="PASS",
        ticker="X",
    )
    assert floored is False


def test_not_floored_with_auto_reject_flag():
    _, floored = maybe_floor_verdict_to_hold(
        _pm_output("DO NOT INITIATE", "DO_NOT_INITIATE"),
        fundamentals_report=APR_BLOCK,
        red_flags=[{"type": "EXTREME_LEVERAGE", "action": "AUTO_REJECT"}],
        code_subtotal=1.0,
        pre_screening_result="PASS",
        ticker="X",
    )
    assert floored is False


def test_not_floored_when_prescreen_reject():
    _, floored = maybe_floor_verdict_to_hold(
        _pm_output("DO NOT INITIATE", "DO_NOT_INITIATE"),
        fundamentals_report=APR_BLOCK,
        red_flags=[],
        code_subtotal=1.0,
        pre_screening_result="REJECT",
        ticker="X",
    )
    assert floored is False


def test_not_floored_when_low_health():
    _, floored = maybe_floor_verdict_to_hold(
        _pm_output("DO NOT INITIATE", "DO_NOT_INITIATE"),
        fundamentals_report=_data_block(55, 14.0, 33, 16.0),
        red_flags=[],
        code_subtotal=1.0,
        pre_screening_result="PASS",
        ticker="X",
    )
    assert floored is False


def test_non_dni_verdict_untouched():
    src = _pm_output("HOLD", "HOLD")
    out, floored = maybe_floor_verdict_to_hold(
        src,
        fundamentals_report=APR_BLOCK,
        red_flags=[],
        code_subtotal=1.0,
        pre_screening_result="PASS",
        ticker="X",
    )
    assert floored is False
    assert out == src


def test_sell_verdict_not_upgraded():
    out, floored = maybe_floor_verdict_to_hold(
        _pm_output("SELL", "SELL"),
        fundamentals_report=APR_BLOCK,
        red_flags=[],
        code_subtotal=1.0,
        pre_screening_result="PASS",
        ticker="X",
    )
    assert floored is False
    assert "VERDICT: SELL" in out


def test_reject_synonym_is_floored():
    # REJECT canonicalizes to DO_NOT_INITIATE; the floor must rewrite it too,
    # not silently no-op (the gate accepts it).
    # Use the canonical-marker PM_BLOCK (the production format extract_pm_block parses).
    out, floored = maybe_floor_verdict_to_hold(
        _full_pm_output("REJECT", "REJECT"),
        fundamentals_report=APR_BLOCK,
        red_flags=[],
        code_subtotal=1.0,
        pre_screening_result="PASS",
        ticker="APR.WA",
    )
    assert floored is True
    assert "VERDICT: HOLD" in out
    assert "VERDICT: REJECT" not in out
    assert "PORTFOLIO MANAGER VERDICT: HOLD" in out


def test_not_floored_when_growth_score_unparseable():
    # Conservative: absent ADJUSTED_GROWTH_SCORE cannot confirm a data-driven failure.
    db = (
        "### --- START DATA_BLOCK ---\n"
        "ADJUSTED_HEALTH_SCORE: 75% (based on 12 available points)\n"
        "PE_RATIO_TTM: 14.88\n"
        "REVENUE_CAGR_3Y: 16.0%\n"
        "### --- END DATA_BLOCK ---"
    )
    _, floored = maybe_floor_verdict_to_hold(
        _pm_output("DO NOT INITIATE", "DO_NOT_INITIATE"),
        fundamentals_report=db,
        red_flags=[],
        code_subtotal=1.0,
        pre_screening_result="PASS",
        ticker="X",
    )
    assert floored is False


# --------------------------------------------------------------------------- #
# maybe_demote_buy_on_blocking_flags
# --------------------------------------------------------------------------- #
_UNRELIABLE = {
    "type": "HEALTH_SCORE_UNRELIABLE",
    "severity": "WARNING",
    "risk_penalty": 0.0,
    "blocks_buy": True,
}


def _buy_output() -> str:
    return (
        "### PORTFOLIO MANAGER VERDICT: BUY\n\n"
        "### --- START PM_BLOCK ---\n"
        "VERDICT: BUY\n"
        "HEALTH_ADJ: 56\nGROWTH_ADJ: 67\nRISK_TALLY: 0.75\nZONE: LOW\n"
        "POSITION_SIZE: 3.0%\n"
        "### --- END PM_BLOCK ---\n"
    )


def test_buy_demoted_when_health_score_unreliable():
    """AGS.BR 2026-07-02 shape: unreliable 56% health must not back a BUY."""
    out, demoted = maybe_demote_buy_on_blocking_flags(
        _buy_output()
        + "\nActual Decision: BUY\n**Action**: BUY\nTRADE_BLOCK:\nACTION: BUY\n",
        red_flags=[_UNRELIABLE],
        ticker="AGS.BR",
    )
    assert demoted
    assert extract_pm_block(out).verdict == "HOLD"
    assert "### PORTFOLIO MANAGER VERDICT: HOLD" in out
    assert "Actual Decision: HOLD" in out
    assert "**Action**: HOLD" in out
    assert "ACTION: BUY" in out
    assert "DETERMINISTIC VERDICT DEMOTION APPLIED" in out
    assert "HEALTH_SCORE_UNRELIABLE" in out


def test_material_unverified_signal_also_blocks_buy():
    """blocks_buy is the contract: the material-op-signal flag's documented
    BUY-block is now enforced deterministically, not via unrendered rationale."""
    out, demoted = maybe_demote_buy_on_blocking_flags(
        _buy_output(),
        red_flags=[
            {
                "type": "MATERIAL_UNVERIFIED_OPERATING_SIGNAL",
                "risk_penalty": 0.0,
                "blocks_buy": True,
            }
        ],
        ticker="T",
    )
    assert demoted
    assert extract_pm_block(out).verdict == "HOLD"


def test_boolean_true_required_for_blocks_buy():
    """Truthy-but-not-True values (e.g. a stray string) must not demote."""
    out, demoted = maybe_demote_buy_on_blocking_flags(
        _buy_output(),
        red_flags=[{"type": "X", "blocks_buy": "yes"}],
        ticker="T",
    )
    assert not demoted


def test_buy_kept_without_unreliable_flag():
    out, demoted = maybe_demote_buy_on_blocking_flags(
        _buy_output(),
        red_flags=[{"type": "VIE_STRUCTURE", "risk_penalty": 0.5}],
        ticker="T",
    )
    assert not demoted
    assert extract_pm_block(out).verdict == "BUY"


def test_non_buy_verdicts_untouched_by_demotion():
    for display, block in (("HOLD", "HOLD"), ("DO NOT INITIATE", "DO_NOT_INITIATE")):
        content = _pm_output(display, block)
        out, demoted = maybe_demote_buy_on_blocking_flags(
            content, red_flags=[_UNRELIABLE], ticker="T"
        )
        assert not demoted
        assert out == content


def test_demotion_skipped_when_pm_block_verdict_missing():
    content = "### PORTFOLIO MANAGER VERDICT: BUY\n\nProse only, no PM_BLOCK."
    out, demoted = maybe_demote_buy_on_blocking_flags(
        content, red_flags=[_UNRELIABLE], ticker="T"
    )
    assert not demoted
    assert out == content


class TestQuickModeBuyQualification:
    """Quick-mode BUY gets a 'candidate, not investable' caveat; token stays BUY."""

    def _buy(self) -> str:
        return _pm_output("BUY", "BUY")

    def test_quick_buy_gets_caveat_and_token_unchanged(self):
        out, qualified = maybe_qualify_buy_in_quick_mode(
            self._buy(), quick_mode=True, ticker="X"
        )
        assert qualified is True
        assert "QUICK-MODE QUALIFICATION" in out
        # Verdict token deliberately unchanged so no downstream parser breaks.
        assert "VERDICT: BUY" in out

    def test_full_mode_no_caveat(self):
        pm = self._buy()
        out, qualified = maybe_qualify_buy_in_quick_mode(pm, quick_mode=False)
        assert qualified is False
        assert out == pm

    def test_quick_non_buy_untouched(self):
        pm = _pm_output("HOLD", "HOLD")
        out, qualified = maybe_qualify_buy_in_quick_mode(pm, quick_mode=True)
        assert qualified is False
        assert out == pm

    def test_idempotent(self):
        out1, _ = maybe_qualify_buy_in_quick_mode(self._buy(), quick_mode=True)
        out2, qualified2 = maybe_qualify_buy_in_quick_mode(out1, quick_mode=True)
        assert qualified2 is True
        assert out2 == out1
        assert out2.count("QUICK-MODE QUALIFICATION") == 1
