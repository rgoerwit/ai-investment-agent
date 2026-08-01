"""Wire-format + fail-closed contract for the Stage 6 provenance codecs.

The load-bearing guarantee is *wire compatibility*, not the self-validating
``from_dict(to_dict(x)) == x``: a current-schema payload must round-trip
byte-identically (modulo the additive ``schema_version``), a legacy payload must
still decode, and a future-schema or type-corrupt gate-critical payload must
fail closed (``SchemaDecodeError`` → non-publishable), never silently default a
corrupted value into an apparently-valid model.
"""

from __future__ import annotations

import pytest

from src.analysis_snapshot import AnalysisSnapshot, build_analysis_snapshot
from src.provenance_schema import (
    DecisionTrace,
    SchemaDecodeError,
    Scorecard,
    ScorecardCriterion,
    classify_schema_version,
)
from src.runtime_diagnostics import build_analysis_validity, stamp_provenance_contract


# --------------------------------------------------------------------------- #
# Schema-version classification                                               #
# --------------------------------------------------------------------------- #
def test_missing_version_is_legacy_compatible() -> None:
    status = classify_schema_version(None, 1)
    assert status.legacy and status.compatible and not status.future


def test_current_version_is_compatible() -> None:
    status = classify_schema_version(1, 1)
    assert status.compatible and not status.legacy and not status.future


def test_future_version_is_incompatible() -> None:
    status = classify_schema_version(2, 1)
    assert status.future and not status.compatible


def test_noninteger_version_is_incompatible() -> None:
    assert not classify_schema_version("1", 1).compatible
    assert not classify_schema_version(True, 1).compatible  # bool is not a version


# --------------------------------------------------------------------------- #
# Scorecard                                                                    #
# --------------------------------------------------------------------------- #
def _full_scorecard_wire(*, advisory_only: bool = False, gaps: bool = False) -> dict:
    return {
        "criteria": {
            "ROE": {"award": "1", "max_points": 1, "derived_from": ["claim:roe:x"]},
            "GLOBAL_EXPANSION": {"award": "1", "max_points": 1, "derived_from": []},
        },
        "earned": 2.0,
        "available": 2,
        "rubric_total": 12,
        "percentage": 50.0,
        "advisory_percentage": 100.0 if advisory_only else 50.0,
        "advisory_only_awards": ["GLOBAL_EXPANSION"] if advisory_only else [],
        "decision_eligible": True,
        "lineage_gaps": ["NET_DEBT_EBITDA"] if gaps else [],
        "schema_version": 1,
    }


def test_scorecard_current_wire_round_trips_byte_identical() -> None:
    wire = _full_scorecard_wire()
    assert Scorecard.from_dict(wire).to_dict() == wire


def test_scorecard_with_advisory_only_awards_round_trips() -> None:
    wire = _full_scorecard_wire(advisory_only=True)
    decoded = Scorecard.from_dict(wire)
    assert decoded.advisory_only_awards == ("GLOBAL_EXPANSION",)
    assert decoded.advisory_percentage == 100.0
    assert decoded.to_dict() == wire


def test_scorecard_with_lineage_gaps_round_trips() -> None:
    wire = _full_scorecard_wire(gaps=True)
    assert Scorecard.from_dict(wire).lineage_gaps == ("NET_DEBT_EBITDA",)
    assert Scorecard.from_dict(wire).to_dict() == wire


def test_scorecard_criterion_numeric_type_preserved() -> None:
    # int max_points stays int through the round trip (byte-identity).
    wire = _full_scorecard_wire()
    out = Scorecard.from_dict(wire).to_dict()
    assert out["criteria"]["ROE"]["max_points"] == 1
    assert isinstance(out["criteria"]["ROE"]["max_points"], int)


def test_scorecard_minimal_legacy_shape_decodes() -> None:
    # Several consumers build a minimal scorecard; it must decode (defaults) and
    # not raise, though it does not round-trip byte-identically.
    card = Scorecard.from_dict({"percentage": 66.7, "decision_eligible": True})
    assert card.percentage == 66.7 and card.decision_eligible is True
    assert card.earned == 0.0


def test_scorecard_future_schema_fails_closed() -> None:
    wire = _full_scorecard_wire()
    wire["schema_version"] = 2
    with pytest.raises(SchemaDecodeError):
        Scorecard.from_dict(wire)


def test_scorecard_malformed_percentage_fails_closed() -> None:
    wire = _full_scorecard_wire()
    wire["percentage"] = "not-a-number"
    with pytest.raises(SchemaDecodeError):
        Scorecard.from_dict(wire)


def test_scorecard_present_but_corrupt_earned_fails_closed() -> None:
    wire = _full_scorecard_wire()
    wire["earned"] = "oops"
    with pytest.raises(SchemaDecodeError):
        Scorecard.from_dict(wire)


def test_scorecard_criterion_missing_award_defaults() -> None:
    crit = ScorecardCriterion.from_dict({"max_points": 1.0})
    assert crit.award == "N/A"


# --------------------------------------------------------------------------- #
# DecisionTrace                                                                #
# --------------------------------------------------------------------------- #
def _full_trace() -> DecisionTrace:
    return DecisionTrace(
        status="VALID",
        verdict="BUY",
        decision_facts=("claim:a",),
        decision_gates=("GATE_X",),
        support_facts=("claim:a",),
        thesis_support_facts=("claim:a",),
        source_families=("GUIDANCE",),
        reason=None,
    )


def test_trace_round_trips_and_emits_both_source_family_aliases() -> None:
    wire = _full_trace().to_dict()
    assert wire["untraced_source_families"] == wire["advisory_source_families"]
    assert DecisionTrace.from_dict(wire).to_dict() == wire


def test_trace_pm_block_missing_unified_shape() -> None:
    wire = DecisionTrace(
        status="INVALID",
        verdict="HOLD",
        missing_gates=("G1",),
        reason="PM_BLOCK_MISSING",
    ).to_dict()
    # Unified: previously-omitted list keys now present (empty) + both aliases.
    assert wire["support_facts"] == []
    assert wire["thesis_support_facts"] == []
    assert wire["untraced_source_families"] == [] == wire["advisory_source_families"]
    assert wire["missing_gates"] == ["G1"]
    assert wire["schema_version"] == 1


def test_trace_unparseable_verdict_survives() -> None:
    wire = DecisionTrace(status="INVALID", verdict="UNPARSEABLE").to_dict()
    assert DecisionTrace.from_dict(wire).verdict == "UNPARSEABLE"


def test_trace_reads_legacy_untraced_only() -> None:
    # A legacy wire may carry only advisory_source_families; decode should keep it.
    decoded = DecisionTrace.from_dict(
        {"status": "VALID", "verdict": "BUY", "advisory_source_families": ["GUIDANCE"]}
    )
    assert decoded.source_families == ("GUIDANCE",)


def test_trace_future_schema_fails_closed() -> None:
    with pytest.raises(SchemaDecodeError):
        DecisionTrace.from_dict(
            {"status": "VALID", "verdict": "BUY", "schema_version": 99}
        )


def test_trace_corrupt_status_fails_closed() -> None:
    with pytest.raises(SchemaDecodeError):
        DecisionTrace.from_dict({"status": ["not", "a", "string"], "verdict": "BUY"})


def test_trace_missing_status_defaults_invalid() -> None:
    assert DecisionTrace.from_dict({"verdict": "BUY"}).status == "INVALID"


# --------------------------------------------------------------------------- #
# AnalysisSnapshot                                                             #
# --------------------------------------------------------------------------- #
def test_snapshot_full_wire_round_trips_and_versions_last() -> None:
    snap = AnalysisSnapshot(
        version=3,
        stage="POST_SENIOR_DERIVED",
        contract_status="VALID",
        contract_reason=None,
        claims={"claim:x": {"field": "PE_RATIO_TTM"}},
        conflicts=[{"field": "X", "type": "T", "detail": "d"}],
        commentary_status="NON_AUTHORITATIVE_UNLESS_CLAIM_REFERENCED",
    )
    wire = snap.to_dict()
    assert list(wire)[-1] == "schema_version"
    assert AnalysisSnapshot.from_dict(wire).to_dict() == wire


def test_snapshot_reduced_invalid_shape_preserved() -> None:
    # The DATA_BLOCK-missing form carries neither stage nor commentary_status.
    wire = build_analysis_snapshot({"fundamentals_report": "no data block here"})
    assert list(wire) == [
        "version",
        "contract_status",
        "contract_reason",
        "claims",
        "conflicts",
        "schema_version",
    ]
    assert AnalysisSnapshot.from_dict(wire).to_dict() == wire


def test_snapshot_legacy_without_schema_version_decodes() -> None:
    legacy = {
        "version": 1,
        "stage": "LEGACY_POST_SENIOR",
        "contract_status": "VALID",
        "contract_reason": None,
        "claims": {},
        "conflicts": [],
        "commentary_status": "NON_AUTHORITATIVE_UNLESS_CLAIM_REFERENCED",
    }
    assert AnalysisSnapshot.from_dict(legacy).contract_status == "VALID"


def test_snapshot_missing_contract_status_defaults_invalid() -> None:
    assert AnalysisSnapshot.from_dict({"version": 1}).contract_status == "INVALID"


def test_snapshot_corrupt_contract_status_fails_closed() -> None:
    with pytest.raises(SchemaDecodeError):
        AnalysisSnapshot.from_dict({"contract_status": 123})


def test_snapshot_future_schema_fails_closed() -> None:
    with pytest.raises(SchemaDecodeError):
        AnalysisSnapshot.from_dict({"contract_status": "VALID", "schema_version": 2})


# --------------------------------------------------------------------------- #
# Boundary integration: future/corrupt gate-critical payload => non-publishable #
# --------------------------------------------------------------------------- #
def _publishable_base() -> dict:
    result = {
        "fundamentals_report": "```DATA_BLOCK\nPE_RATIO_TTM: 10\n```",
        "final_trade_decision": "PM_BLOCK verdict",
        "pre_screening_result": "PASS",
        "analysis_snapshot": {
            "version": 1,
            "stage": "POST_SENIOR_DERIVED",
            "contract_status": "VALID",
            "contract_reason": None,
            "claims": {},
            "conflicts": [],
            "commentary_status": "NON_AUTHORITATIVE_UNLESS_CLAIM_REFERENCED",
            "schema_version": 1,
        },
        "decision_trace": DecisionTrace(status="VALID", verdict="BUY").to_dict(),
    }
    stamp_provenance_contract(result)
    return result


def test_future_schema_snapshot_makes_analysis_non_publishable() -> None:
    result = _publishable_base()
    result["analysis_snapshot"]["schema_version"] = 999
    validity = build_analysis_validity(result)
    assert validity["analysis_snapshot_status"] == "DECODE_FAILED"
    assert "analysis_snapshot" in validity["required_failures"]
    assert not validity["publishable"]


def test_corrupt_trace_makes_analysis_non_publishable() -> None:
    result = _publishable_base()
    result["decision_trace"]["status"] = 12345  # corrupt type
    validity = build_analysis_validity(result)
    assert validity["decision_trace_status"] == "DECODE_FAILED"
    assert "decision_trace" in validity["required_failures"]
    assert not validity["publishable"]
