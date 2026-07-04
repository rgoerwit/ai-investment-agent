"""Writer↔editor shared-standard parity.

The editor deliberately duplicates the writer's essential standards (it holds
the writer to them and cannot see the writer's prompt at runtime). That
duplication is intentional; silent divergence is not. These tests fail CI when
a shared invariant is edited out of one prompt but not the other — the drift
mode that motivated the July 2026 prompt audit.

Writer-side text = system_message + revision_template (the CORRECTED/CONTESTED
protocol lives in the template). Editor-side text = system_message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def _load(name: str) -> dict:
    path = PROMPTS_DIR / name
    assert path.exists(), f"{name} missing — fail loud, do not skip"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def writer_text() -> str:
    data = _load("writer.json")
    return data["system_message"] + "\n" + data["metadata"]["revision_template"]


@pytest.fixture(scope="module")
def editor_text() -> str:
    return _load("editor.json")["system_message"]


# Exact tokens that must appear verbatim in BOTH prompts. Losing one from
# either side means the shared standard has drifted.
SHARED_INVARIANTS = [
    # source-report literal phrases (valuation.py / memo.py emit these)
    '"conditional"',
    '"suppressed"',
    # 'verified' discipline
    "'verified'",
    # ownership / control threshold
    ">20%",
    # cyclical-context rule
    "200%",
    "PEG < 0.2",
    # voice / copyedit standards
    "First-Person Discipline",
    "Narrative Lead",
    "Narrative Close",
    "appended scorecard",
    "rearticulate",
    "Chicago Manual of Style",
    "Strunk & White",
    # mandatory-disclosure flag vocabulary (current as of July 2026)
    "PFIC",
    "VIE",
    "CMIC",
    "SEGMENT_FLAG",
    "UNEXPLAINED_DRAWDOWN_NEWS_GAP",
    "HEALTH_SCORE_UNRELIABLE",
    "GROWTH_SCORE_UNRELIABLE",
    "_SCORE_CONSISTENCY: SUSPECT",
    "GROWTH_DATA_STALE",
    "OCF_FILING_VALUE_UNCORROBORATED",
    "ANALYST_COVERAGE_TOTAL_EST",
    "low English-language aggregator visibility",
    # revision meta-block protocol
    "EDITOR_NOTES",
]

# Banned-lexicon core: each word must be named in both prompts (either side may
# add words freely; dropping a shared one fails).
SHARED_LEXICON = [
    "optionality",
    "impactful",
    "solutioning",
    "learnings",
    "granular",
    "robust",
    "landscape",
    "game-changer",
    "hidden gem",
    "masterclass",
    "massive",
    "narrative tension",
]


@pytest.mark.parametrize("token", SHARED_INVARIANTS)
def test_shared_invariant_present_in_both(token, writer_text, editor_text):
    assert token in writer_text, f"writer.json lost shared invariant: {token!r}"
    assert token in editor_text, f"editor.json lost shared invariant: {token!r}"


@pytest.mark.parametrize("word", SHARED_LEXICON)
def test_shared_lexicon_word_in_both(word, writer_text, editor_text):
    assert word in writer_text.lower(), f"writer.json lost lexicon word: {word!r}"
    assert word in editor_text.lower(), f"editor.json lost lexicon word: {word!r}"


def test_editor_confidence_is_advisory(editor_text):
    """The confidence float must be documented as advisory, never a gate."""
    assert "advisory" in editor_text
    assert "confidence >= 0.85" not in editor_text


def test_first_person_not_banned_outright(writer_text, editor_text):
    """'I' is an allowed analytical instrument; only the opening is restricted."""
    assert "The 'I' Ban" not in writer_text
    assert "NOT a ban on 'I'" in writer_text
    assert "do NOT flag the word 'I'" in editor_text
