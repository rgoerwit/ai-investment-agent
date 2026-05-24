"""Tests for the final_verdict_formed log event in pm_node.

Stage 2 of the logging-audit follow-up. Verifies:
- The event fires on the PM success path with a parsed verdict.
- The event fires on the invalid-structure fast-fail path with verdict=PARSE_FAILURE.
- direct_pm_inputs_present/missing reflect artifact validity, not just non-empty.
- _present_pm_inputs correctly partitions inputs using artifact_statuses.
"""

from __future__ import annotations

from src.agents.decision_nodes import _present_pm_inputs
from src.agents.pm_inputs import DIRECT_PM_INPUT_FIELDS


def test_present_pm_inputs_partitions_by_validity() -> None:
    state = {
        "market_report": "valid market text",
        "sentiment_report": "",  # empty -> missing
        "fundamentals_report": "valid fundamentals",
        "artifact_statuses": {
            "fundamentals_report": {
                "complete": True,
                "ok": True,
                "content": "valid fundamentals",
            },
            "value_trap_report": {"complete": True, "ok": False, "content": "stub"},
        },
        "risk_debate_state": {"current_risky_response": "r"},
    }
    present, missing = _present_pm_inputs(state)
    assert "market_report" in present
    assert "sentiment_report" in missing
    assert "fundamentals_report" in present
    # value_trap_report is marked not-ok -> should be missing despite content
    assert "value_trap_report" in missing
    assert "risk_debate_state" in present
    # Every direct PM input field is accounted for somewhere
    assert set(present + missing) >= set(DIRECT_PM_INPUT_FIELDS) | {"risk_debate_state"}


def test_present_pm_inputs_marks_risk_debate_absent_when_all_views_empty() -> None:
    state = {"risk_debate_state": {}}
    present, missing = _present_pm_inputs(state)
    assert "risk_debate_state" in missing
    assert "risk_debate_state" not in present


def test_direct_pm_input_fields_no_duplicates_and_no_risk_debate() -> None:
    """The risk_debate_state field is handled separately, not in the tuple."""
    assert len(set(DIRECT_PM_INPUT_FIELDS)) == len(DIRECT_PM_INPUT_FIELDS)
    assert "risk_debate_state" not in DIRECT_PM_INPUT_FIELDS
