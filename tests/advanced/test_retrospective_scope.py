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
    LESSON_SCOPES,
    RESERVED_UNOBSERVED_SCOPES,
    SCOPE_CONTEXTUAL,
    SCOPE_UNRESOLVED,
    SCOPE_VALIDATED,
    THESIS_NOT_EVALUATED,
    extract_snapshot,
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
            (DRIVER_RESIDUAL, SCOPE_UNRESOLVED),
            (DRIVER_MARKET, SCOPE_CONTEXTUAL),
            (DRIVER_MIXED, SCOPE_CONTEXTUAL),
            (DRIVER_UNKNOWN, SCOPE_CONTEXTUAL),
        ],
    )
    def test_scope_follows_the_driver(self, driver, expected):
        assert lesson_scope_for(driver) == expected

    def test_a_stock_specific_move_is_unresolved_not_validated(self):
        """The concession that matters: we know it was idiosyncratic, not why."""
        assert lesson_scope_for(DRIVER_RESIDUAL) == SCOPE_UNRESOLVED
        assert lesson_scope_for(DRIVER_RESIDUAL) != SCOPE_VALIDATED

    def test_no_driver_ever_yields_validated(self):
        for driver in (DRIVER_MARKET, DRIVER_RESIDUAL, DRIVER_MIXED, DRIVER_UNKNOWN):
            assert lesson_scope_for(driver) != SCOPE_VALIDATED

    def test_an_unrecognized_driver_degrades_to_the_humbler_scope(self):
        assert lesson_scope_for("SOMETHING_NEW") == SCOPE_CONTEXTUAL


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
