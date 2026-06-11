"""Tests for the Bear KILL_CRITERIA parser and PM-payload wiring (Tranche 1, Step 2)."""

from __future__ import annotations

import json
import pathlib

import pytest

from src.agents.support import extract_kill_criteria, get_bear_history

# ---------- prompt regression ----------


def test_bear_prompt_includes_fenced_kill_criteria_block() -> None:
    import re

    prompt_path = pathlib.Path("prompts/bear_researcher.json")
    data = json.loads(prompt_path.read_text(encoding="utf-8"))
    assert re.match(r"^\d+\.\d+$", data["version"])  # format, not a pinned value
    msg = data["system_message"]
    assert "### --- START KILL_CRITERIA ---" in msg
    assert "### --- END KILL_CRITERIA ---" in msg
    assert "TRIGGER_1:" in msg
    assert "TRIGGER_2:" in msg


# ---------- extract_kill_criteria: happy / edge / error ----------


def test_extract_kill_criteria_happy_two_triggers() -> None:
    bear = (
        "blah blah\n"
        "### --- START KILL_CRITERIA ---\n"
        "TRIGGER_1: D/E ratio exceeds 1.5\n"
        "TRIGGER_2: two consecutive years negative FCF\n"
        "### --- END KILL_CRITERIA ---\n"
        "trailing narrative\n"
    )
    triggers = extract_kill_criteria(bear)
    assert triggers == [
        "D/E ratio exceeds 1.5",
        "two consecutive years negative FCF",
    ]


def test_extract_kill_criteria_single_trigger_accepted() -> None:
    bear = (
        "### --- START KILL_CRITERIA ---\n"
        "TRIGGER_1: gross margin drops below 30% for two quarters\n"
        "### --- END KILL_CRITERIA ---"
    )
    assert extract_kill_criteria(bear) == [
        "gross margin drops below 30% for two quarters"
    ]


def test_extract_kill_criteria_caps_at_three() -> None:
    bear = (
        "### --- START KILL_CRITERIA ---\n"
        "TRIGGER_1: one\n"
        "TRIGGER_2: two\n"
        "TRIGGER_3: three\n"
        "TRIGGER_4: four\n"
        "### --- END KILL_CRITERIA ---"
    )
    assert extract_kill_criteria(bear) == ["one", "two", "three"]


@pytest.mark.parametrize(
    "bear",
    [
        "",
        None,
        "no fenced block here, free-form bear text",
        "### --- START KILL_CRITERIA ---\nno triggers, just prose\n### --- END KILL_CRITERIA ---",
        "### --- START KILL_CRITERIA ---\nTRIGGER_1:    \n### --- END KILL_CRITERIA ---",
        # Missing end marker → regex doesn't match → empty list (graceful).
        "### --- START KILL_CRITERIA ---\nTRIGGER_1: foo\n",
    ],
)
def test_extract_kill_criteria_malformed_returns_empty(bear: str | None) -> None:
    assert extract_kill_criteria(bear) == []


# ---------- get_bear_history: dual-shape support ----------


def test_get_bear_history_runtime_state_shape() -> None:
    state = {"investment_debate_state": {"bear_history": "runtime bear text"}}
    assert get_bear_history(state) == "runtime bear text"


def test_get_bear_history_saved_json_shape() -> None:
    saved = {
        "investment_analysis": {
            "investment_debate": {"bear_history": "saved bear text"}
        }
    }
    assert get_bear_history(saved) == "saved bear text"


def test_get_bear_history_runtime_wins_when_both_present() -> None:
    both = {
        "investment_debate_state": {"bear_history": "runtime"},
        "investment_analysis": {"investment_debate": {"bear_history": "saved"}},
    }
    assert get_bear_history(both) == "runtime"


def test_get_bear_history_missing_returns_empty_string() -> None:
    assert get_bear_history({}) == ""
    assert get_bear_history({"investment_debate_state": {}}) == ""
    assert get_bear_history(None) == ""  # type: ignore[arg-type]
