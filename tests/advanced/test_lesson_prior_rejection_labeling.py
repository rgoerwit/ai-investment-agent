"""Tests that prior_rejection records are labeled distinctly from learned lessons.

A `prior_rejection` record (a same-day screening artifact, failure_mode N/A) must
render as "PRIOR REJECTION (<ticker>):" so it is not mistaken for a generalizable
retrospective lesson when injected into researcher / RM prompts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import retrospective


class _FakeCollection:
    def count(self) -> int:
        return 5


def _fake_memory() -> SimpleNamespace:
    return SimpleNamespace(available=True, situation_collection=_FakeCollection())


def _result(document: str, **meta) -> dict:
    base = {
        "failure_mode": meta.pop("failure_mode", "UNKNOWN"),
        "sector": "Energy",
        "exchange": "V",
        "confidence_weight": 0.7,
    }
    base.update(meta)
    return {"document": document, "metadata": base}


@pytest.mark.asyncio
async def test_prior_rejection_and_lesson_labels(monkeypatch):
    results = [
        _result(
            "Screened out on D/E",
            lesson_type="prior_rejection",
            ticker="ALV.V",
            failure_mode="N/A",
        ),
        _result("Cyclical peak risk underweighted", failure_mode="CYCLICAL_PEAK"),
    ]

    async def fake_get_relevant_lessons(memory, sector, ticker):
        return results

    monkeypatch.setattr(
        retrospective, "get_relevant_lessons", fake_get_relevant_lessons
    )

    text = await retrospective.format_lessons_for_injection(
        _fake_memory(), "ALV.V", "Energy"
    )

    assert "PRIOR REJECTION (ALV.V): Screened out on D/E" in text
    assert "LESSON: Cyclical peak risk underweighted" in text
    # The screening record must NOT be presented as a plain lesson.
    assert "LESSON: Screened out on D/E" not in text


@pytest.mark.asyncio
async def test_normal_lesson_only_uses_lesson_label(monkeypatch):
    async def fake_get_relevant_lessons(memory, sector, ticker):
        return [_result("Governance bleed risk", failure_mode="GOVERNANCE_BLEED")]

    monkeypatch.setattr(
        retrospective, "get_relevant_lessons", fake_get_relevant_lessons
    )

    text = await retrospective.format_lessons_for_injection(
        _fake_memory(), "7203.T", "Consumer Discretionary"
    )

    assert "LESSON: Governance bleed risk" in text
    assert "PRIOR REJECTION" not in text
