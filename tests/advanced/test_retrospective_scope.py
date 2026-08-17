"""Step 5: carry the pre-commitment, refuse to adjudicate it, grade the claim.

Two things the retrospective must stop asserting.

**That a thesis-break trigger fired.** It holds price and macro data only. Whether
a company-specific pre-registered trigger actually occurred is a question about
filings and news published *after* the decision, which this code path never
fetches. So ``kill_criteria`` is recorded and ``thesis_validation_status`` is
permanently ``NOT_EVALUATED``.

**That a stock-specific move means the analysis was wrong.** A residual can be an
earnings surprise, a fraud disclosure, a takeover rumour, sector rotation, a data
error, or a mistaken thesis. ``UNRESOLVED`` says what is actually known.

Both guards below are **AST scans, not text scans**: a comment explaining why a
value is never written necessarily contains that value, and this repository has
already been burned by a regex guard that false-positived on its own docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.retrospective import (
    DRIVER_MARKET,
    DRIVER_MIXED,
    DRIVER_RESIDUAL,
    DRIVER_UNKNOWN,
    LESSON_ELIGIBILITIES,
    LESSON_ELIGIBILITY_INJECTABLE,
    LESSON_ELIGIBILITY_REVIEW_ONLY,
    LESSON_SCOPES,
    RESERVED_UNOBSERVED_SCOPES,
    SCOPE_CONTEXTUAL,
    SCOPE_UNRESOLVED,
    SCOPE_VALIDATED,
    THESIS_NOT_EVALUATED,
    extract_snapshot,
    lesson_eligibility,
    lesson_scope_for,
)

_RETROSPECTIVE_SOURCE = Path("src/retrospective.py")


def _result_with_bear(bear_text: str) -> dict:
    return {
        "final_trade_decision": "PORTFOLIO MANAGER VERDICT: HOLD",
        "fundamentals_report": "",
        "investment_debate_state": {"bear_history": bear_text},
    }


# Marker form copied verbatim from a real artifact (001060.KS). The block is not
# a triple-backtick fence — an invented fixture shape here would have tested the
# fixture rather than the extractor.
_BEAR_WITH_TRIGGERS = """
The bear case rests on cyclical exposure.

**KILL CRITERIA**

### --- START KILL_CRITERIA ---
TRIGGER_1: Operating margin falls below 8% for two consecutive quarters
TRIGGER_2: Net debt/EBITDA exceeds 3.5x
### --- END KILL_CRITERIA ---
"""


# ══════════════════════════════════════════════════════════════════════════════
# The pre-commitment
# ══════════════════════════════════════════════════════════════════════════════


class TestKillCriteriaReachTheSnapshot:
    def test_triggers_are_captured(self):
        snapshot = extract_snapshot(_result_with_bear(_BEAR_WITH_TRIGGERS), "2767.T")
        assert snapshot["kill_criteria"] == [
            "Operating margin falls below 8% for two consecutive quarters",
            "Net debt/EBITDA exceeds 3.5x",
        ]

    def test_a_missing_block_yields_an_empty_list_not_none(self):
        snapshot = extract_snapshot(
            _result_with_bear("Plain bear prose with no block."), "2767.T"
        )
        assert snapshot["kill_criteria"] == []

    def test_a_malformed_block_yields_an_empty_list(self):
        bear = (
            "### --- START KILL_CRITERIA ---\n"
            "not a trigger line\n"
            "### --- END KILL_CRITERIA ---"
        )
        snapshot = extract_snapshot(_result_with_bear(bear), "2767.T")
        assert snapshot["kill_criteria"] == []

    def test_more_than_three_triggers_are_truncated(self):
        bear = (
            "### --- START KILL_CRITERIA ---\n"
            + "\n".join(f"TRIGGER_{i}: condition {i}" for i in range(1, 6))
            + "\n### --- END KILL_CRITERIA ---"
        )
        snapshot = extract_snapshot(_result_with_bear(bear), "2767.T")
        assert len(snapshot["kill_criteria"]) == 3

    def test_an_absent_debate_state_does_not_raise(self):
        snapshot = extract_snapshot(
            {"final_trade_decision": "", "fundamentals_report": ""}, "2767.T"
        )
        assert snapshot["kill_criteria"] == []

    def test_a_non_mapping_debate_state_does_not_raise(self):
        snapshot = extract_snapshot(
            {
                "final_trade_decision": "",
                "fundamentals_report": "",
                "investment_debate_state": "corrupted",
            },
            "2767.T",
        )
        assert snapshot["kill_criteria"] == []
        assert snapshot["bear_risks_excerpt"] == ""

    def test_the_excerpt_and_the_triggers_read_the_same_round(self):
        """One resolver, so the two consumers cannot diverge."""
        result = {
            "final_trade_decision": "",
            "fundamentals_report": "",
            "investment_debate_state": {"bear_round1": _BEAR_WITH_TRIGGERS},
        }
        snapshot = extract_snapshot(result, "2767.T")
        assert snapshot["kill_criteria"], "round1 fallback must feed kill criteria too"
        assert snapshot["bear_risks_excerpt"]


class TestTheThesisIsNeverAdjudicated:
    """Price cannot establish that a company-specific pre-commitment occurred."""

    def test_only_one_status_value_exists(self):
        assert THESIS_NOT_EVALUATED == "NOT_EVALUATED"

    def test_no_module_assigns_met_or_not_met(self):
        """AST scan: a string in a comment is not an assignment."""
        tree = ast.parse(_RETROSPECTIVE_SOURCE.read_text())
        forbidden = {"MET", "NOT_MET"}
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.value in forbidden:
                offenders.append((node.value, node.lineno))
        assert offenders == [], (
            "src/retrospective.py names a thesis-validation verdict it has no "
            f"evidence for: {offenders}"
        )

    def test_the_status_key_is_only_ever_written_with_the_constant(self):
        tree = ast.parse(_RETROSPECTIVE_SOURCE.read_text())
        written: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=False):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "thesis_validation_status"
                ):
                    written.append(ast.dump(value))
        assert written, "the field is not written anywhere — did it get dropped?"
        assert all("THESIS_NOT_EVALUATED" in entry for entry in written), (
            "thesis_validation_status must only ever be the NOT_EVALUATED "
            f"constant; found {written}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Scope
# ══════════════════════════════════════════════════════════════════════════════


class TestLessonScope:
    @pytest.mark.parametrize(
        ("driver", "expected"),
        [
            (DRIVER_MARKET, SCOPE_CONTEXTUAL),
            (DRIVER_RESIDUAL, SCOPE_UNRESOLVED),
            (DRIVER_MIXED, SCOPE_UNRESOLVED),
            (DRIVER_UNKNOWN, SCOPE_UNRESOLVED),
        ],
    )
    def test_scope_follows_the_driver(self, driver, expected):
        assert lesson_scope_for(driver) == expected

    def test_an_unattributed_move_is_not_a_regime_observation(self):
        """MIXED means neither leg dominated, i.e. we could not attribute it.

        Reading that as "market-dominated" did more than mislabel a record: the
        scope is printed into the lesson prompt, and the prompt's "unexplained,
        not diagnosed" rule keys on it, so every MIXED outcome was generated with
        that rule switched off. Measured 2026-08-16: 95 RESIDUAL, 67 MIXED, zero
        MARKET — and all 67 CONTEXTUAL lessons asserted a cause never established.
        """
        assert lesson_scope_for(DRIVER_MIXED) == SCOPE_UNRESOLVED
        assert lesson_scope_for(DRIVER_MIXED) != SCOPE_CONTEXTUAL

    def test_a_stock_specific_move_is_unresolved_not_validated(self):
        """The concession that matters: we know it was idiosyncratic, not why."""
        assert lesson_scope_for(DRIVER_RESIDUAL) == SCOPE_UNRESOLVED
        assert lesson_scope_for(DRIVER_RESIDUAL) != SCOPE_VALIDATED

    def test_no_driver_ever_yields_validated(self):
        for driver in (DRIVER_MARKET, DRIVER_RESIDUAL, DRIVER_MIXED, DRIVER_UNKNOWN):
            assert lesson_scope_for(driver) != SCOPE_VALIDATED

    def test_an_unrecognized_driver_degrades_to_the_humbler_scope(self):
        """A driver added later must not inherit authority by falling through."""
        assert lesson_scope_for("SOMETHING_NEW") == SCOPE_UNRESOLVED


class TestValidatedIsReservedNotDead:
    """Three prior tokens in this repo silently gated decisions while unemitted.

    ``CMIC_LISTED``, ``other_legal_risks`` and ``COVERAGE_COMPLETE_NO_MATCH`` each
    cost real behaviour. VALIDATED is declared reserved rather than left to be
    rediscovered — and nothing may gate on it until a producer exists.
    """

    def test_it_is_a_member_of_the_scope_vocabulary(self):
        assert SCOPE_VALIDATED in LESSON_SCOPES

    def test_it_is_declared_reserved(self):
        assert RESERVED_UNOBSERVED_SCOPES == {SCOPE_VALIDATED}

    def test_the_other_scopes_are_not_reserved(self):
        assert SCOPE_CONTEXTUAL not in RESERVED_UNOBSERVED_SCOPES
        assert SCOPE_UNRESOLVED not in RESERVED_UNOBSERVED_SCOPES

    def test_no_comparison_operator_reads_it(self):
        """A reserved token may be *defined*, never *branched on*."""
        tree = ast.parse(_RETROSPECTIVE_SOURCE.read_text())
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                names = {
                    child.id for child in ast.walk(node) if isinstance(child, ast.Name)
                }
                if "SCOPE_VALIDATED" in names:
                    offenders.append(node.lineno)
        assert offenders == [], (
            "SCOPE_VALIDATED is compared against at "
            f"{offenders}, but nothing emits it — that is a dead gate"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Eligibility
# ══════════════════════════════════════════════════════════════════════════════


def _comparison(driver, regime, shifted, shift_reason=""):
    return {
        "attribution": {"dominant_driver": driver},
        "regime_at_decision": regime,
        "cached_regime_delta": {"shifted": shifted, "shift_reason": shift_reason},
    }


_STABLE_REGIME = {"risk_appetite": "RISK_ON", "shock_type": "NONE"}


class TestLessonEligibility:
    """Scope says how far an outcome generalizes; eligibility says whether the
    record carries the evidence needed to apply it. Both are required, and the
    second is why 67 records could be stamped CONTEXTUAL and still be inert.
    """

    def test_a_market_move_in_a_stable_recorded_regime_is_injectable(self):
        eligibility, _ = lesson_eligibility(
            _comparison(DRIVER_MARKET, _STABLE_REGIME, False)
        )
        assert eligibility == LESSON_ELIGIBILITY_INJECTABLE

    @pytest.mark.parametrize(
        "driver", [DRIVER_MIXED, DRIVER_RESIDUAL, DRIVER_UNKNOWN, "SOMETHING_NEW"]
    )
    def test_an_unattributed_move_is_review_only(self, driver):
        eligibility, reason = lesson_eligibility(
            _comparison(driver, _STABLE_REGIME, False)
        )
        assert eligibility == LESSON_ELIGIBILITY_REVIEW_ONLY
        assert driver in reason

    def test_a_market_move_with_no_recorded_regime_is_review_only(self):
        """The exact shape of all 67 records stored on 2026-08-16.

        A scope-only fix misses this: the record would still say CONTEXTUAL while
        `_regime_matches` could never fire for it, because the fields it compares
        are blank. Injectability has to be checked against the metadata that
        retrieval actually reads.
        """
        eligibility, reason = lesson_eligibility(
            _comparison(DRIVER_MARKET, {"risk_appetite": "", "shock_type": ""}, False)
        )
        assert eligibility == LESSON_ELIGIBILITY_REVIEW_ONLY
        assert "regime" in reason

    def test_a_missing_regime_key_is_treated_as_absent_not_as_an_error(self):
        eligibility, _ = lesson_eligibility(
            {"attribution": {"dominant_driver": DRIVER_MARKET}}
        )
        assert eligibility == LESSON_ELIGIBILITY_REVIEW_ONLY

    def test_a_shifted_regime_is_review_only(self):
        eligibility, reason = lesson_eligibility(
            _comparison(
                DRIVER_MARKET,
                _STABLE_REGIME,
                True,
                "risk appetite: RISK_ON -> RISK_OFF",
            )
        )
        assert eligibility == LESSON_ELIGIBILITY_REVIEW_ONLY
        assert "RISK_OFF" in reason

    def test_an_unknown_shift_is_withheld_not_permitted(self):
        """`shifted` is False only when both regimes were usable, comparable and
        equal; every degraded path returns None. "We could not establish that the
        regime held still" is not a basis for authorizing guidance about it.
        """
        eligibility, _ = lesson_eligibility(
            _comparison(DRIVER_MARKET, _STABLE_REGIME, None, "cache is 40d old")
        )
        assert eligibility == LESSON_ELIGIBILITY_REVIEW_ONLY

    def test_the_two_withholding_reasons_are_distinguishable(self):
        """REVIEW_ONLY is ambiguous on its face, so the reason is persisted.

        "the driver was MIXED" is a finding about the snapshot; "the macro cache
        was stale" is a finding about the run that evaluated it. An auditor
        reading the store must be able to tell them apart.
        """
        _, about_snapshot = lesson_eligibility(
            _comparison(DRIVER_MIXED, _STABLE_REGIME, False)
        )
        _, about_run = lesson_eligibility(
            _comparison(DRIVER_MARKET, _STABLE_REGIME, None, "cache is 40d old")
        )
        assert about_snapshot != about_run
        assert "40d old" in about_run

    def test_every_result_is_a_declared_token(self):
        for driver in (DRIVER_MARKET, DRIVER_MIXED, DRIVER_RESIDUAL, DRIVER_UNKNOWN):
            for shifted in (True, False, None):
                eligibility, reason = lesson_eligibility(
                    _comparison(driver, _STABLE_REGIME, shifted)
                )
                assert eligibility in LESSON_ELIGIBILITIES
                assert reason, "a withheld or granted record must say why"

    def test_a_malformed_comparison_does_not_raise(self):
        for bad in ({}, {"attribution": "nonsense"}, {"cached_regime_delta": 5}):
            eligibility, _ = lesson_eligibility(bad)
            assert eligibility == LESSON_ELIGIBILITY_REVIEW_ONLY


class TestAnActiveDealIsNotAMarketOutcome:
    """A live tender prices the stock against deal terms, not against its market.

    The benchmark decomposition still computes and means nothing, so a MARKET
    label there is an artefact of deal mechanics. `m_and_a_status` was already
    carried in the snapshot for the IBKR reconciler; eligibility now reads it.
    """

    def test_an_active_tender_is_review_only(self):
        comparison = _comparison(DRIVER_MARKET, _STABLE_REGIME, False)
        comparison["m_and_a_status"] = "ACTIVE_TENDER"
        eligibility, reason = lesson_eligibility(comparison)
        assert eligibility == LESSON_ELIGIBILITY_REVIEW_ONLY
        assert "tender" in reason

    def test_a_rumour_does_not_disqualify(self):
        """A rumour does not pin the price, so attribution still means something."""
        comparison = _comparison(DRIVER_MARKET, _STABLE_REGIME, False)
        comparison["m_and_a_status"] = "RUMORED"
        eligibility, _ = lesson_eligibility(comparison)
        assert eligibility == LESSON_ELIGIBILITY_INJECTABLE

    @pytest.mark.parametrize("value", ["NONE", "", None])
    def test_no_deal_is_not_disqualifying(self, value):
        comparison = _comparison(DRIVER_MARKET, _STABLE_REGIME, False)
        comparison["m_and_a_status"] = value
        eligibility, _ = lesson_eligibility(comparison)
        assert eligibility == LESSON_ELIGIBILITY_INJECTABLE
