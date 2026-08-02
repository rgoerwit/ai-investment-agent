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


class TestNonValidSnapshotFailsClosed:
    """A snapshot that RAN but did not validate must not silently authorize the
    unprojected DATA_BLOCK arithmetic it failed to reconcile."""

    def test_invalid_contract_marks_both_scores_suspect(self) -> None:
        inputs = DecisionInputs.from_metrics_and_snapshot(
            {"adjusted_health_score": 70.0, "adjusted_growth_score": 68.0},
            "Industrials",
            {"contract_status": "INVALID", "contract_reason": "VALIDATOR_CRASHED"},
            ticker="TEST",
        )

        assert inputs.snapshot_authoritative is False
        assert inputs.snapshot_status == "INVALID"
        assert inputs.health_score_reliable is False
        assert inputs.growth_score_reliable is False
        assert inputs.decision_metrics["health_score_consistency"] == "SUSPECT"
        assert inputs.decision_metrics["growth_score_consistency"] == "SUSPECT"

    def test_suspect_stamp_raises_blocking_unreliable_flags(self) -> None:
        """The SUSPECT contract is the point: it routes into the existing
        *_SCORE_UNRELIABLE -> blocks_buy chain rather than inventing semantics."""
        from src.validators.financial_rules import detect_red_flags

        inputs = DecisionInputs.from_metrics_and_snapshot(
            {"adjusted_health_score": 70.0, "adjusted_growth_score": 68.0},
            "Industrials",
            {"contract_status": "INVALID"},
            ticker="TEST",
        )
        flags, _pre_screen = detect_red_flags(inputs, "TEST")
        blocking = {f["type"] for f in flags if f.get("blocks_buy")}

        assert "HEALTH_SCORE_UNRELIABLE" in blocking
        assert "GROWTH_SCORE_UNRELIABLE" in blocking

    def test_future_schema_claiming_valid_is_not_authoritative(self) -> None:
        """Two readers, one verdict: a payload the publication boundary decodes
        to DECODE_FAILED must not be authoritative here just because its raw
        contract_status string says VALID."""
        from src.runtime_diagnostics.artifact_status import _decode_snapshot_status

        payload = {
            "contract_status": "VALID",
            "schema_version": 99_999,
            "scorecards": {
                "HEALTH": {"percentage": 91.0, "decision_eligible": True},
                "GROWTH": {"percentage": 88.0, "decision_eligible": True},
            },
        }
        inputs = DecisionInputs.from_metrics_and_snapshot(
            {"adjusted_health_score": 40.0, "adjusted_growth_score": 40.0},
            "Industrials",
            payload,
            ticker="TEST",
        )

        assert _decode_snapshot_status(payload) == "DECODE_FAILED"
        assert inputs.snapshot_status == "DECODE_FAILED"
        assert inputs.snapshot_authoritative is False
        # The snapshot's 91/88 is NOT adopted, and the DATA_BLOCK 40/40 is not
        # trusted either — it is marked untrusted rather than consumed.
        assert inputs.health_decision_pct == 40.0
        assert inputs.health_score_reliable is False

    def test_degraded_contract_does_not_poison_scores(self) -> None:
        """DEGRADED is a pre-senior input-availability state (no usable analytic
        RAW_METRICS fields), not a rejection of the score arithmetic. It must not
        stamp SUSPECT — that would flag every thin-data run as untrusted."""
        inputs = DecisionInputs.from_metrics_and_snapshot(
            {"adjusted_health_score": 68.0, "adjusted_growth_score": 55.0},
            "Financials",
            {"contract_status": "DEGRADED", "contract_reason": "RAW_METRICS_NO_USABLE"},
            ticker="TEST",
        )

        assert inputs.snapshot_status == "DEGRADED"
        assert inputs.snapshot_authoritative is False
        assert inputs.health_score_reliable is True
        assert inputs.growth_score_reliable is True
        assert "health_score_consistency" not in inputs.decision_metrics

    def test_legacy_no_snapshot_is_untouched(self) -> None:
        """Absent snapshot stays fully backward compatible (no SUSPECT stamp)."""
        metrics = {"adjusted_health_score": 70.0, "adjusted_growth_score": 55.0}
        for snapshot in (None, {}):
            inputs = DecisionInputs.from_metrics_and_snapshot(
                metrics, "Industrials", snapshot, ticker="TEST"
            )
            assert inputs.snapshot_status is None
            assert inputs.snapshot_authoritative is False
            assert inputs.health_score_reliable is True
            assert inputs.decision_metrics == metrics

    def test_non_mapping_snapshot_is_treated_as_absent(self) -> None:
        for snapshot in ([], "corrupt", 3):
            inputs = DecisionInputs.from_metrics_and_snapshot(
                {"adjusted_health_score": 70.0}, "Industrials", snapshot, ticker="TEST"
            )
            assert inputs.snapshot_status is None
            assert inputs.health_score_reliable is True
