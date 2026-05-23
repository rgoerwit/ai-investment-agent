from __future__ import annotations

from src.agents.structured_outputs import (
    PMVerdictMetadata,
    pm_verdict_metadata_from_text,
)


def test_pm_verdict_metadata_from_pm_block() -> None:
    metadata = pm_verdict_metadata_from_text(
        """
### --- START PM_BLOCK ---
VERDICT: BUY
HEALTH_ADJ: 90
GROWTH_ADJ: 70
RISK_TALLY: 1.0
ZONE: LOW
### --- END PM_BLOCK ---
"""
    )

    assert metadata == PMVerdictMetadata(verdict="BUY")


def test_pm_verdict_metadata_falls_back_to_text_verdict() -> None:
    metadata = pm_verdict_metadata_from_text("PORTFOLIO MANAGER VERDICT: HOLD")

    assert metadata.verdict == "HOLD"


def test_pm_verdict_metadata_marks_unparseable() -> None:
    metadata = pm_verdict_metadata_from_text("No action label here")

    assert metadata.verdict == "UNPARSEABLE"
