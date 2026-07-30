"""Step 6: deterministic Data-Vacuum / Marginal-Turnaround verdict floor."""

from __future__ import annotations

import re

import pytest

from src.agents.verdict_policy import (
    DNI_REVIEW_CANDIDATE_MARKER,
    maybe_demote_buy_on_blocking_flags,
    maybe_floor_verdict_to_hold,
    maybe_qualify_buy_in_quick_mode,
    maybe_qualify_weak_asymmetry_buy,
    maybe_tag_dni_review_candidate,
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


def test_off_balance_sheet_recourse_blocks_buy():
    out, demoted = maybe_demote_buy_on_blocking_flags(
        _buy_output(),
        red_flags=[
            {
                "type": "OFF_BALANCE_SHEET_RECOURSE",
                "risk_penalty": 0.0,
                "blocks_buy": True,
            }
        ],
        ticker="CRWV",
    )

    assert demoted is True
    assert extract_pm_block(out).verdict == "HOLD"
    assert "OFF_BALANCE_SHEET_RECOURSE" in out


def test_missing_normalized_earnings_bridge_blocks_buy():
    out, demoted = maybe_demote_buy_on_blocking_flags(
        _buy_output(),
        red_flags=[
            {
                "type": "NORMALIZED_EARNINGS_REQUIRED",
                "risk_penalty": 0.0,
                "blocks_buy": True,
            }
        ],
        ticker="6745.T",
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


class TestWeakAsymmetryBuyQualification:
    """A BUY with thin weighted-IV upside / high downside prob is qualified."""

    def _buy(self) -> str:
        return _pm_output("BUY", "BUY")

    def test_weak_upside_gets_caveat_token_unchanged(self):
        out, qualified = maybe_qualify_weak_asymmetry_buy(
            self._buy(), weighted_upside=0.043, downside_probability=20.0, ticker="X"
        )
        assert qualified is True
        assert "WEAK VALUATION ASYMMETRY" in out
        assert "4.3%" in out
        # Verdict token deliberately unchanged so no downstream parser breaks.
        assert "VERDICT: BUY" in out

    def test_high_downside_probability_triggers(self):
        out, qualified = maybe_qualify_weak_asymmetry_buy(
            self._buy(), weighted_upside=0.30, downside_probability=55.0
        )
        assert qualified is True
        assert "WEAK VALUATION ASYMMETRY" in out

    def test_strong_asymmetry_untouched(self):
        pm = self._buy()
        out, qualified = maybe_qualify_weak_asymmetry_buy(
            pm, weighted_upside=0.30, downside_probability=20.0
        )
        assert qualified is False
        assert out == pm

    def test_non_buy_untouched(self):
        pm = _pm_output("HOLD", "HOLD")
        out, qualified = maybe_qualify_weak_asymmetry_buy(
            pm, weighted_upside=0.01, downside_probability=80.0
        )
        assert qualified is False
        assert out == pm

    def test_none_upside_no_crash(self):
        pm = self._buy()
        out, qualified = maybe_qualify_weak_asymmetry_buy(
            pm, weighted_upside=None, downside_probability=None
        )
        assert qualified is False
        assert out == pm

    def test_idempotent(self):
        out1, _ = maybe_qualify_weak_asymmetry_buy(
            self._buy(), weighted_upside=0.043, downside_probability=20.0
        )
        out2, qualified2 = maybe_qualify_weak_asymmetry_buy(
            out1, weighted_upside=0.043, downside_probability=20.0
        )
        assert qualified2 is True
        assert out2 == out1
        assert out2.count("WEAK VALUATION ASYMMETRY") == 1


def _dni_output(health: str = "60", growth: str = "70") -> str:
    """A gate-passing DO_NOT_INITIATE PM output with configurable adjusted scores."""
    return (
        "### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE\n\n"
        "Rationale: risk tally crossed the Zone-1 threshold.\n\n"
        "### --- START PM_BLOCK ---\n"
        "VERDICT: DO_NOT_INITIATE\n"
        f"HEALTH_ADJ: {health}\n"
        f"GROWTH_ADJ: {growth}\n"
        "RISK_TALLY: 2.5\n"
        "ZONE: HIGH\n"
        "POSITION_SIZE: 0.0%\n"
        "### --- END PM_BLOCK ---\n"
    )


class TestDniReviewCandidate:
    """A gate-passing DNI is tagged as a review candidate; token stays DNI.

    The tag deliberately does NOT assert why the PM declined (liquidity,
    coverage, valuation, US-revenue, or data gaps may bind) — it is a
    re-review signal, never an entry-timing claim.
    """

    def test_gate_passing_dni_tagged_token_unchanged(self):
        pm = _dni_output()
        out, tagged = maybe_tag_dni_review_candidate(pm, red_flags=[], ticker="X")
        assert tagged is True
        assert DNI_REVIEW_CANDIDATE_MARKER in out
        # Verdict token deliberately unchanged so no downstream parser breaks.
        assert "VERDICT: DO_NOT_INITIATE" in out
        assert "does not assert why" in out
        assert "Not an entry-timing signal" in out

    def test_note_is_regex_inert(self):
        # Ripple-audit contract: the appended note must contain no parentheses
        # (PM-claim/article citation audits scan bare parentheticals), no
        # uppercase KEY: tokens, and no standalone PASS/FAIL tokens
        # (thesis_visualizer / score-parser / discipline-check regexes key on
        # word-boundary tokens — "GATE-PASSING" is a different word).
        pm = _dni_output()
        out, _ = maybe_tag_dni_review_candidate(pm, red_flags=[])
        note = out[len(pm.rstrip()) :]
        assert "(" not in note and ")" not in note
        assert not re.search(r"\bPASS\b|\bFAIL\b", note)
        assert not re.search(r"\b[A-Z][A-Z0-9_]{2,}:", note)

    def test_liquidity_or_data_vacuum_dni_gets_non_assertive_note(self):
        # A high-score DNI with no red flags (e.g. liquidity fail, coverage
        # fail, or strict-mode data-vacuum conversion) IS tagged — but the note
        # must not claim to know the binding constraint.
        out, tagged = maybe_tag_dni_review_candidate(_dni_output(), red_flags=[])
        assert tagged is True
        assert "liquidity, coverage, valuation, US-revenue" in out
        assert "entry-timing hold" not in out
        assert "Monitor for a better entry" not in out

    def test_failing_health_gate_not_tagged(self):
        pm = _dni_output(health="45")
        out, tagged = maybe_tag_dni_review_candidate(pm, red_flags=[])
        assert tagged is False
        assert out == pm

    def test_failing_growth_gate_not_tagged(self):
        pm = _dni_output(growth="33")
        out, tagged = maybe_tag_dni_review_candidate(pm, red_flags=[])
        assert tagged is False
        assert out == pm

    def test_missing_scores_no_tag_no_raise(self):
        pm = (
            "### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE\n\n"
            "### --- START PM_BLOCK ---\n"
            "VERDICT: DO_NOT_INITIATE\n"
            "POSITION_SIZE: 0.0%\n"
            "### --- END PM_BLOCK ---\n"
        )
        out, tagged = maybe_tag_dni_review_candidate(pm, red_flags=[])
        assert tagged is False
        assert out == pm

    def test_non_dni_verdicts_untouched(self):
        for display, block in (("BUY", "BUY"), ("HOLD", "HOLD"), ("SELL", "SELL")):
            pm = _pm_output(display, block)
            out, tagged = maybe_tag_dni_review_candidate(pm, red_flags=[])
            assert tagged is False
            assert out == pm

    def test_auto_reject_flag_disqualifies(self):
        out, tagged = maybe_tag_dni_review_candidate(
            _dni_output(),
            red_flags=[{"type": "EXTREME_LEVERAGE", "action": "AUTO_REJECT"}],
        )
        assert tagged is False
        assert DNI_REVIEW_CANDIDATE_MARKER not in out

    def test_blocks_buy_flag_disqualifies(self):
        # An unreliable score makes the gate-pass claim itself indeterminate.
        out, tagged = maybe_tag_dni_review_candidate(
            _dni_output(),
            red_flags=[
                {
                    "type": "HEALTH_SCORE_UNRELIABLE",
                    "action": "REVIEW",
                    "risk_penalty": 0.0,
                    "blocks_buy": True,
                }
            ],
        )
        assert tagged is False

    def test_material_penalty_with_novel_name_disqualifies(self):
        # Structural threshold: any single flag >= 1.0 penalty disqualifies
        # regardless of name — future flags need no denylist maintenance.
        out, tagged = maybe_tag_dni_review_candidate(
            _dni_output(),
            red_flags=[{"type": "SOME_FUTURE_FLAG", "risk_penalty": 1.5}],
        )
        assert tagged is False

    @pytest.mark.parametrize(
        "flag_type, penalty",
        [
            ("PFIC_PROBABLE", 1.0),
            ("PFIC_UNCERTAIN", 0.5),
            ("VIE_STRUCTURE", 0.5),
            ("CMIC_FLAGGED", 2.0),
            ("CMIC_UNCERTAIN", 1.0),
            # Dynamically suffixed by Legal Counsel — prefix match required.
            ("REGULATORY_DELISTING", 0.5),
            ("CONSULTANT_MANDATE_BREACH", 2.0),
            ("CONSULTANT_MAJOR_CONCERNS", 1.5),
            # Category prefix disqualifies even below the 1.0 threshold.
            ("CONSULTANT_TRANSIENT_STRENGTH", 0.5),
            ("VALUE_TRAP_HIGH_RISK", 1.0),
            ("VALUE_TRAP_VERDICT", 1.0),
        ],
    )
    def test_category_disqualifiers(self, flag_type, penalty):
        out, tagged = maybe_tag_dni_review_candidate(
            _dni_output(),
            red_flags=[
                {"type": flag_type, "action": "RISK_PENALTY", "risk_penalty": penalty}
            ],
        )
        assert tagged is False

    def test_value_trap_moderate_is_deliberate_carve_out(self):
        # A moderate governance signal (0.5, no category prefix) is compatible
        # with "review" — documented carve-out, not an oversight.
        out, tagged = maybe_tag_dni_review_candidate(
            _dni_output(),
            red_flags=[
                {
                    "type": "VALUE_TRAP_MODERATE_RISK",
                    "action": "RISK_PENALTY",
                    "risk_penalty": 0.5,
                }
            ],
        )
        assert tagged is True

    def test_idempotent(self):
        out1, _ = maybe_tag_dni_review_candidate(_dni_output(), red_flags=[])
        out2, tagged2 = maybe_tag_dni_review_candidate(out1, red_flags=[])
        assert tagged2 is True
        assert out2 == out1
        assert out2.count(DNI_REVIEW_CANDIDATE_MARKER) == 1

    def test_floored_dni_not_also_tagged(self):
        # Sequencing contract (mirrors the pm_node tail): a DNI the floor hook
        # converts to HOLD must not then be tagged as a review candidate.
        pm = _full_pm_output("DO NOT INITIATE", "DO_NOT_INITIATE")
        floored, did_floor = maybe_floor_verdict_to_hold(
            pm,
            fundamentals_report=APR_BLOCK,
            red_flags=[],
            code_subtotal=1.0,
            pre_screening_result="PASS",
        )
        assert did_floor is True
        out, tagged = maybe_tag_dni_review_candidate(floored, red_flags=[])
        assert tagged is False
        assert out == floored
