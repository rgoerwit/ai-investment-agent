"""Stage 5a: typed, snapshot-authoritative decision inputs."""

from __future__ import annotations

from src.decision_inputs import DecisionInputs


def _snapshot(health_pct: float, growth_pct: float, *, eligible: bool = True) -> dict:
    return {
        "contract_status": "VALID",
        "scorecards": {
            "HEALTH": {"percentage": health_pct, "decision_eligible": eligible},
            "GROWTH": {"percentage": growth_pct, "decision_eligible": eligible},
        },
    }


def test_fully_projected_metrics_are_byte_parity() -> None:
    """When the DATA_BLOCK already carries the snapshot scores (the normal
    projected case), DecisionInputs reproduces them unchanged."""
    metrics = {
        "adjusted_health_score": 66.7,
        "adjusted_growth_score": 50.0,
        "debt_to_equity": 120.0,
        "interest_coverage": 4.0,
        "net_income": 100.0,
        "fcf": 80.0,
    }
    inputs = DecisionInputs.from_metrics_and_snapshot(
        metrics, "Industrials", _snapshot(66.7, 50.0), ticker="TEST"
    )

    assert inputs.health_decision_pct == 66.7
    assert inputs.growth_decision_pct == 50.0
    assert inputs.snapshot_authoritative is True
    # decision_metrics is a no-op reconciliation on already-projected metrics.
    assert inputs.decision_metrics == metrics


def test_snapshot_wins_over_diverging_data_block() -> None:
    """If the parsed DATA_BLOCK score ever disagrees with the canonical
    scorecard, the snapshot value is authoritative."""
    metrics = {
        "adjusted_health_score": 83.3,  # stale / drifted text value
        "adjusted_growth_score": 66.7,
    }
    inputs = DecisionInputs.from_metrics_and_snapshot(
        metrics, "Industrials", _snapshot(50.0, 50.0), ticker="TEST"
    )

    assert inputs.health_decision_pct == 50.0
    assert inputs.growth_decision_pct == 50.0
    assert inputs.decision_metrics["adjusted_health_score"] == 50.0
    assert inputs.decision_metrics["adjusted_growth_score"] == 50.0


def test_legacy_artifact_falls_back_to_data_block() -> None:
    """No snapshot (legacy/frozen artifact) → the parsed DATA_BLOCK scores are
    used and never overwritten."""
    metrics = {"adjusted_health_score": 70.0, "adjusted_growth_score": 55.0}
    inputs = DecisionInputs.from_metrics_and_snapshot(
        metrics, "Industrials", None, ticker="TEST"
    )

    assert inputs.health_decision_pct == 70.0
    assert inputs.growth_decision_pct == 55.0
    assert inputs.snapshot_authoritative is False
    assert inputs.decision_metrics == metrics


def test_ineligible_scorecard_yields_none_not_the_data_block() -> None:
    """A VALID snapshot with an ineligible scorecard is authoritative: the score
    is None (canonically unusable), never the parsed DATA_BLOCK value."""
    metrics = {"adjusted_health_score": None, "adjusted_growth_score": None}
    inputs = DecisionInputs.from_metrics_and_snapshot(
        metrics, "Industrials", _snapshot(50.0, 50.0, eligible=False), ticker="TEST"
    )

    assert inputs.health_decision_pct is None
    assert inputs.growth_decision_pct is None
    # The snapshot is VALID, so it owns the field even when it supplies None.
    assert inputs.snapshot_authoritative is True


def test_valid_but_ineligible_snapshot_never_falls_back_to_stale_data_block() -> None:
    """P1 regression: a VALID snapshot that marks a score ineligible must NOT let
    a stale narrative DATA_BLOCK score leak through. The canonical layer said the
    score is unusable; DecisionInputs must return None (and drop it from the
    reconciled metrics the engine reads)."""
    metrics = {
        "adjusted_health_score": 70.0,  # stale narrative value that must be dropped
        "adjusted_growth_score": 68.0,
    }
    inputs = DecisionInputs.from_metrics_and_snapshot(
        metrics, "Industrials", _snapshot(50.0, 50.0, eligible=False), ticker="TEST"
    )

    assert inputs.health_decision_pct is None
    assert inputs.growth_decision_pct is None
    assert inputs.decision_metrics["adjusted_health_score"] is None
    assert inputs.decision_metrics["adjusted_growth_score"] is None


def test_score_consistency_suspect_marks_unreliable() -> None:
    metrics = {
        "adjusted_health_score": 60.0,
        "adjusted_growth_score": 55.0,
        "health_score_consistency": "SUSPECT",
    }
    inputs = DecisionInputs.from_metrics_and_snapshot(
        metrics, "Financials", _snapshot(60.0, 55.0), ticker="TEST"
    )

    assert inputs.health_score_reliable is False
    assert inputs.growth_score_reliable is True
    assert inputs.sector == "Financials"
