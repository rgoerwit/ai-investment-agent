from src.agents.evidence_constraints import downstream_evidence_constraints


def _state(
    execution: str,
    *,
    corrected: bool = False,
    extra_fields: str = "",
) -> dict:
    warning = "AUTHORITATIVE_METRIC_CORRECTION\n" if corrected else ""
    report = f"""{warning}### --- START DATA_BLOCK ---
SHAREHOLDER_RETURN_EXECUTION: {execution}
{extra_fields}
### --- END DATA_BLOCK ---"""
    return {
        "fundamentals_report": report,
        "artifact_statuses": {
            "fundamentals_report": {"complete": True, "ok": True, "content": report}
        },
    }


def test_unproven_buyback_cannot_receive_governance_credit() -> None:
    constraint = downstream_evidence_constraints(_state("ANNOUNCED_ONLY"))

    assert "Shareholder-return execution is ANNOUNCED_ONLY" in constraint
    assert "proven governance mitigation" in constraint


def test_proven_buyback_does_not_add_negative_buyback_constraint() -> None:
    constraint = downstream_evidence_constraints(_state("PROVEN"))
    assert "Shareholder-return execution" not in constraint
    assert "10% of 30-day average daily turnover" in constraint


def test_unknown_execution_is_conservative_and_correction_survives_summary() -> None:
    constraint = downstream_evidence_constraints(_state("N/A", corrected=True))

    assert "AUTHORITATIVE_METRIC_CORRECTION" in constraint
    assert "Shareholder-return execution is UNKNOWN" in constraint


def test_value_trap_conflict_quarantines_raw_score_for_downstream_agents() -> None:
    state = _state("PROVEN")
    state["red_flags"] = [{"type": "VALUE_TRAP_DATA_CONFLICT"}]

    constraint = downstream_evidence_constraints(state)

    assert "raw Value Trap score and TRAP verdict are UNRECONCILED" in constraint
    assert "Do not use that raw score/verdict as a hard fail" in constraint


def test_liquidity_is_order_relative_and_unknown_notional_cannot_hard_fail() -> None:
    constraint = downstream_evidence_constraints(_state("PROVEN"))

    assert "Assess trading liquidity relative to the proposed order" in constraint
    assert "If order notional is unknown, do not infer a hard fail" in constraint


def test_minority_largest_holder_and_no_majority_are_not_a_conflict() -> None:
    state = _state("PROVEN")
    state["entity_governance_card"] = {
        "control_status": "NOT_CONTROLLED",
        "largest_shareholder": {"name": "BenQ Materials Corp.", "pct": 14.82},
    }

    constraint = downstream_evidence_constraints(state)

    assert "Do not call the issuer a controlled subsidiary" in constraint
    assert "MAJORITY_HOLDER: NONE is compatible" in constraint
    assert "14.82%" in constraint


def test_parallel_value_trap_roic_na_is_analysis_quality_not_issuer_risk() -> None:
    constraint = downstream_evidence_constraints(_state("PROVEN"))

    assert "Value Trap ROIC: N/A is expected" in constraint
    assert "Score 0.0 risk" in constraint


def test_secondary_or_undated_capacity_is_conditionally_constrained() -> None:
    state = _state(
        "PROVEN",
        extra_fields="""CAPACITY_UTILIZATION: 95%
CAPACITY_EVIDENCE_STATUS: SECONDARY
CAPACITY_UTILIZATION_AS_OF: UNKNOWN""",
    )

    constraint = downstream_evidence_constraints(state)

    assert "Retain it as a conditional reference only" in constraint
    assert "independent risk/bonus" in constraint


def test_primary_dated_capacity_does_not_receive_uncertainty_constraint() -> None:
    state = _state(
        "PROVEN",
        extra_fields="""CAPACITY_UTILIZATION: 95%
CAPACITY_EVIDENCE_STATUS: PRIMARY
CAPACITY_UTILIZATION_AS_OF: 2026-Q1""",
    )

    assert "Retain it as a conditional reference only" not in (
        downstream_evidence_constraints(state)
    )


def test_narrow_catalyst_and_aggregator_coverage_scopes_reach_pm() -> None:
    state = _state("PROVEN", extra_fields="ANALYST_COVERAGE_ENGLISH: 7")
    state["red_flags"] = [{"type": "NO_CATALYST_DETECTED"}]

    constraint = downstream_evidence_constraints(state)

    assert "limited to activist, index, tender, and restructuring" in constraint
    assert "aggregator analyst-opinion count" in constraint


def test_primary_latest_actuals_cannot_be_recast_as_forward_evidence() -> None:
    state = _state(
        "PROVEN",
        extra_fields="""LATEST_RESULTS_PERIOD: Three months ended March 31, 2026
LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY""",
    )

    constraint = downstream_evidence_constraints(state)

    assert "primary historical actuals" in constraint
    assert "do not use actual YoY growth as management guidance" in constraint


def test_uncited_value_trap_m_and_a_cannot_support_acquisition_claim() -> None:
    state = _state("PROVEN")
    value_trap = """### --- START VALUE_TRAP_BLOCK ---
M&A_CONTEXT_EVIDENCE: UNKNOWN
### --- END VALUE_TRAP_BLOCK ---"""
    state["value_trap_report"] = value_trap
    state["artifact_statuses"]["value_trap_report"] = {
        "complete": True,
        "ok": True,
        "content": value_trap,
    }

    constraint = downstream_evidence_constraints(state)

    assert "Do not name an acquisition" in constraint
    assert "infer acquisition-led growth" in constraint
