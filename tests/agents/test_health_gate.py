"""Canonical health policy, exception authority and report parity."""

from dataclasses import asdict, replace

import pytest

from src.agents.verdict_policy import (
    assess_growth_gate,
    assess_health_gate,
    assess_initiation_gates,
    assess_verdict_policy,
    format_compliance_result,
    format_initiation_gate,
    health_gate_from_policy,
    maybe_floor_verdict_to_hold,
    project_compliance_gates,
    render_initiation_gate_context,
)
from src.decision_inputs import DecisionInputs
from src.liquidity_assessment import LiquidityAssessment
from src.report_generator import QuietModeReporter
from src.score_lineage import eligible_fact_number
from src.thesis_visualizer import ThesisVisualizer


def inputs(health=45, pb="0.59", ratio="1.51", *, authoritative=True):
    snapshot = {
        "contract_status": "VALID",
        "scorecards": {
            "HEALTH": {"percentage": health, "decision_eligible": True},
            "GROWTH": {"percentage": 60, "decision_eligible": True},
        },
        "claims": {
            name: {
                "field": name,
                "kind": "FACT",
                "decision_eligible": True,
                "value": value,
            }
            for name, value in (("PB_RATIO", pb), ("CURRENT_RATIO_RAW", ratio))
        },
    }
    result = DecisionInputs.from_metrics_and_snapshot(
        {"adjusted_health_score": health, "adjusted_growth_score": 60},
        None,
        snapshot if authoritative else None,
    )
    return replace(
        result,
        english_analyst_count=4,
        us_revenue_pct=10,
        liquidity=LiquidityAssessment("PASS", average_daily_turnover_usd=500_000),
    )


def pm(verdict="HOLD"):
    return (
        f"### PORTFOLIO MANAGER VERDICT: {verdict}\n"
        "### THESIS COMPLIANCE SUMMARY\n**Hard Fail Checks:**\n"
        "- **Financial Health**: 99% - PASS\n"
        "Unowned economic risk prose.\n"
        f"### --- START PM_BLOCK ---\nVERDICT: {verdict}\nPOSITION_SIZE: 0.0\n"
        "### --- END PM_BLOCK ---"
    )


@pytest.mark.parametrize(
    "health,pb,ratio,status",
    [
        (39.9, "0.1", "2", "FAIL"),
        (40, "0.59", "1.51", "PASS"),
        (49.9, "0.59", "1.51", "PASS"),
        (50, None, None, "PASS"),
        (49.9, "0.6", "1.51", "FAIL"),
        (49.9, "0.59", "1.5", "FAIL"),
        (45, None, "2", "FAIL"),
        (45, "0.5", None, "FAIL"),
        (45, "1e-05", "2e0", "PASS"),
        (45, "-0.5", "2", "FAIL"),
    ],
)
def test_health_threshold_and_exception_boundaries(health, pb, ratio, status):
    resolved = inputs(health, pb, ratio)
    gate = assess_health_gate(resolved)
    assert gate.status == status and gate.value == health
    assert gate.exception == (
        "deep_value_liquidity" if status == "PASS" and health < 50 else None
    )
    assert gate.supporting_ids == (
        ("PB_RATIO", "CURRENT_RATIO_RAW") if gate.exception else ()
    )
    violation = assess_verdict_policy(pm(), decision_inputs=resolved)
    assert (violation.rule if violation else None) == (
        "financial_health_hard_fail" if status == "FAIL" else None
    )


@pytest.mark.parametrize(
    "value",
    ["0,5", "0.5 / 2", "~0.5", "0.5x", "1-2", "NaN", "inf", True, [], {}, "1e9999"],
)
def test_strict_ratio_never_promotes_partial_or_nonfinite_scalar(value):
    resolved = inputs(pb=value)
    assert resolved.health_exception_pb_ratio is None
    assert assess_health_gate(resolved).status == "FAIL"


def test_ratio_conflict_and_legacy_lineage_parsing_are_distinct_contracts():
    claims = {
        "a": {
            "field": "PB_RATIO",
            "kind": "FACT",
            "decision_eligible": True,
            "value": "0.5",
        },
        "b": {
            "field": "PB_RATIO",
            "kind": "FACT",
            "decision_eligible": True,
            "value": "0.6",
        },
    }
    assert eligible_fact_number(claims, "PB_RATIO", strict=True) == (None, None)
    claims["b"]["value"] = "0.5"
    assert eligible_fact_number(claims, "PB_RATIO", strict=True) == (0.5, "a")
    claims["b"]["decision_eligible"] = False
    claims["a"]["value"] = "$1,000 million"
    assert eligible_fact_number(claims, "PB_RATIO") == (1000, "a")
    assert eligible_fact_number(claims, "PB_RATIO", strict=True) == (None, None)


def test_no_split_raw_narrative_or_non_authoritative_exception():
    resolved = inputs(authoritative=False)
    assert resolved.health_exception_pb_ratio is None
    assert assess_health_gate(resolved).status == "FAIL"
    assert (
        assess_health_gate(replace(inputs(), snapshot_authoritative=False)).status
        == "FAIL"
    )


@pytest.mark.parametrize(
    "value", [None, True, "40", float("inf"), float("nan"), -1, 101, 10**1000]
)
def test_malformed_or_unreliable_health_is_uncertainty(value):
    resolved = replace(inputs(), health_decision_pct=value)
    assert assess_health_gate(resolved).status == "INDETERMINATE"
    assert assess_verdict_policy(pm(), decision_inputs=resolved) is None
    assert (
        assess_health_gate(replace(inputs(), health_score_reliable=False)).status
        == "INDETERMINATE"
    )


@pytest.mark.parametrize("tender", [False, True])
@pytest.mark.parametrize("health", [39.9, None, 45, 80])
def test_real_context_policy_visual_report_agree(health, tender):
    resolved = inputs(80)
    resolved = replace(
        resolved,
        health_decision_pct=health,
        growth_decision_pct=40,
        health_exception_pb_ratio=None,
        decision_metrics={
            **resolved.decision_metrics,
            "m_and_a_status": "ACTIVE_TENDER" if tender else "NONE",
        },
    )
    gate = assess_health_gate(resolved)
    growth = assess_growth_gate(resolved)
    gates = assess_initiation_gates(resolved)
    policy = {
        "schema_version": 1,
        "health_gate": asdict(gate),
        "growth_gate": {"status": growth.status},
        "initiation_gates": [asdict(g) for g in gates],
    }
    assert health_gate_from_policy(policy) == gate
    projected = project_compliance_gates(pm(), resolved)
    assert project_compliance_gates(projected, resolved) == projected
    visual = ThesisVisualizer(projected, policy).generate()
    report = QuietModeReporter("TEST").generate_report(
        {
            "final_trade_decision": projected,
            "decision_policy": policy,
            "chart_paths": {"mock": "no writes"},
        }
    )
    text = format_initiation_gate(gate)
    result = format_compliance_result(gates, growth.status, gate)
    for output in (render_initiation_gate_context(resolved), visual, report):
        assert text in output
    assert result in projected and result in visual and result in report
    assert "Unowned economic risk prose." in projected
    assert "99.0%" not in visual
    if tender:
        assert "Exempt under ACTIVE_TENDER" in visual
        assert result == "PASS"
        assert assess_verdict_policy(pm(), decision_inputs=resolved) is None
        assert (
            assess_verdict_policy(
                pm(), decision_inputs=replace(resolved, english_analyst_count=15)
            ).rule
            == "analyst_coverage"
        )


@pytest.mark.parametrize(
    "change",
    [
        {"status": []},
        {"status": {}},
        {"status": "PASS", "value": None},
        {"status": "FAIL", "value": None},
        {"exception": []},
        {"exception": {}},
        {"value": "45"},
        {"applicable": "false"},
    ],
)
def test_malformed_saved_health_cannot_borrow_pm_health(change):
    policy = {
        "schema_version": 1,
        "health_gate": {**asdict(assess_health_gate(inputs())), **change},
    }
    visual = ThesisVisualizer(pm(), policy).generate()
    assert "Financial Health: Unknown - INDETERMINATE" in visual
    assert "99.0%" not in visual


@pytest.mark.parametrize("version", [True, "1", 1.0, 2, None, {}, []])
def test_future_or_malformed_schema_health_is_unknown(version):
    policy = {
        "schema_version": version,
        "health_gate": asdict(assess_health_gate(inputs(80))),
    }
    assert health_gate_from_policy(policy).status == "INDETERMINATE"


def test_legacy_saved_policy_has_unknown_health_but_no_policy_retains_legacy():
    assert (
        "Financial Health: Unknown - INDETERMINATE"
        in ThesisVisualizer(pm(), {"schema_version": 1}).generate()
    )
    assert ThesisVisualizer(pm()).metrics.health_score == 99


def test_unknown_count_and_verified_failure_precedence():
    resolved = replace(
        inputs(),
        health_score_reliable=False,
        growth_score_reliable=False,
        english_analyst_count=None,
        us_revenue_pct=None,
    )
    result = format_compliance_result(
        assess_initiation_gates(resolved),
        assess_growth_gate(resolved).status,
        assess_health_gate(resolved),
    )
    assert (
        result
        == "INDETERMINATE — No verified hard failure among these gates; 4 gates unverified"
    )
    resolved = replace(resolved, english_analyst_count=15)
    assert (
        format_compliance_result(
            assess_initiation_gates(resolved),
            "INDETERMINATE",
            assess_health_gate(resolved),
        )
        == "FAIL"
    )


def test_dni_floor_cannot_rescue_health_fail():
    resolved = inputs(39.9)
    assert (
        maybe_floor_verdict_to_hold(
            pm("DO_NOT_INITIATE"),
            decision_inputs=resolved,
            red_flags=[],
            code_subtotal=0,
            pre_screening_result="PASS",
        )[1]
        is False
    )


@pytest.mark.parametrize(
    "pb,ratio,retained,status",
    [(0.596, 1.504, (0.6, 1.5), "FAIL"), (0.594, 1.506, (0.59, 1.51), "PASS")],
)
def test_real_canonical_ratio_producer_rounding_is_used_without_epsilon(
    pb, ratio, retained, status
):
    from src.analysis_snapshot import build_pre_senior_snapshot
    from src.tooling.structured_ingress import build_structured_ingress_record

    snapshot = build_pre_senior_snapshot(
        {
            "structured_inputs": {
                "raw_financial_metrics": build_structured_ingress_record(
                    {"priceToBook": pb, "currentRatio": ratio},
                    agent_key="junior_fundamentals_analyst",
                    tool_name="get_financial_metrics",
                )
            }
        }
    )
    snapshot["scorecards"] = {
        "HEALTH": {"percentage": 45, "decision_eligible": True},
        "GROWTH": {"percentage": 60, "decision_eligible": True},
    }
    resolved = DecisionInputs.from_metrics_and_snapshot({}, None, snapshot)
    assert (
        resolved.health_exception_pb_ratio,
        resolved.health_exception_current_ratio,
    ) == retained
    assert assess_health_gate(resolved).status == status


@pytest.mark.parametrize(
    "snapshot",
    [
        {"schema_version": 999, "contract_status": "VALID"},
        {"contract_status": "INVALID"},
        {"claims": {}, "contract_status": "DECODE_FAILED"},
    ],
)
def test_exception_never_uses_ratios_from_rejected_snapshot(snapshot):
    resolved = DecisionInputs.from_metrics_and_snapshot(
        {"adjusted_health_score": 45, "pb_ratio": 0.5, "current_ratio": 2},
        None,
        snapshot,
    )
    assert not resolved.snapshot_authoritative
    assert (
        resolved.health_exception_pb_ratio is None
        and resolved.health_exception_current_ratio is None
    )


def test_negative_zero_ratio_cannot_qualify_exception():
    import math

    resolved = inputs(pb="-0", ratio="2")
    assert math.copysign(1, resolved.health_exception_pb_ratio) == -1
    assert assess_health_gate(resolved).status == "FAIL"
    assert assess_health_gate(inputs(pb="0", ratio="2")).status == "PASS"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "health,status", [(39.9, "FAIL"), (80, "PASS"), (None, "INDETERMINATE")]
)
async def test_deterministic_screen_stores_same_health_without_model_or_narrative_fallback(
    health, status
):
    from src.agents.decision_nodes import create_screen_rejection_node

    result = await create_screen_rejection_node()(
        {
            "company_of_interest": "TEST",
            "analysis_outcome": {
                "schema_version": 1,
                "eligibility": "REJECTED",
                "run_status": "COMPLETED",
                "reason_codes": ["LIQUIDITY_HARD_FAIL"],
            },
            "pre_screening_result": "REJECT",
            "red_flags": [
                {
                    "type": "LIQUIDITY_HARD_FAIL",
                    "severity": "CRITICAL",
                    "action": "AUTO_REJECT",
                }
            ],
            "analysis_snapshot": {
                "contract_status": "VALID",
                "claims": {},
                "scorecards": {
                    "HEALTH": {
                        "percentage": health,
                        "decision_eligible": health is not None,
                    }
                },
            },
        },
        {},
    )
    assert (
        result["artifact_statuses"]["final_trade_decision"]["provider"]
        == "deterministic"
    )
    assert result["decision_policy"]["health_gate"]["status"] == status
    assert result["decision_policy"]["health_gate"]["value"] == health
    assert result["decision_trace"]["status"] == "VALID"
    assert result["decision_trace"]["decision_gates"] == ["LIQUIDITY_HARD_FAIL"]
    assert "LIQUIDITY_HARD_FAIL" in result["final_trade_decision"]
    visual = ThesisVisualizer(
        result["final_trade_decision"], result["decision_policy"]
    ).generate()
    assert (
        f"Financial Health: {'Unknown' if health is None else f'{health:g}%'} - {status}"
        in visual
    )
    if health == 80:
        assert (
            "No verified hard failure among these gates; 4 gates unverified" in visual
        )
        assert result["decision_policy"]["final_verdict"] == "DO_NOT_INITIATE"


@pytest.mark.parametrize("prepared", [False, True])
def test_shared_decision_inputs_preserve_authority_and_runtime_context(
    monkeypatch, prepared
):
    from src.agents.decision_nodes import _decision_inputs_from_state
    from src.validators.red_flag_detector import RedFlagDetector

    snapshot = {
        "contract_status": "VALID",
        "scorecards": {
            "HEALTH": {"percentage": 45, "decision_eligible": True},
            "GROWTH": {"percentage": 60, "decision_eligible": True},
        },
        "claims": {
            field: {
                "field": field,
                "kind": "FACT",
                "decision_eligible": True,
                "value": value,
            }
            for field, value in (("PB_RATIO", "0.59"), ("CURRENT_RATIO_RAW", "1.51"))
        },
    }
    metrics = {"adjusted_health_score": 99, "adjusted_growth_score": 99}
    liquidity = LiquidityAssessment(
        "FAIL_INSUFFICIENT_LIQUIDITY", average_daily_turnover_usd=10_000
    )
    state = {
        "fundamentals_report": "Sector: Industrials",
        "analysis_snapshot": {"contract_status": "INVALID"} if prepared else snapshot,
        "liquidity_assessment": liquidity.to_dict(),
    }
    calls = []

    def extract(report, *, ticker):
        calls.append((report, ticker))
        return metrics

    monkeypatch.setattr(RedFlagDetector, "extract_metrics", extract)
    monkeypatch.setattr(RedFlagDetector, "detect_sector", lambda report: "Industrials")
    result = _decision_inputs_from_state(
        state,
        "TEST",
        **({"metrics": metrics, "snapshot": snapshot} if prepared else {}),
    )
    assert result.snapshot_authoritative
    assert result.health_decision_pct == 45
    assert result.growth_decision_pct == 60
    assert result.sector == "Industrials"
    assert result.liquidity == liquidity
    assert assess_health_gate(result).exception == "deep_value_liquidity"
    assert assess_verdict_policy(pm("BUY"), decision_inputs=result).rule == "liquidity"
    assert calls == ([] if prepared else [("Sector: Industrials", "TEST")])


def test_shared_decision_inputs_preserve_empty_prepared_inputs(monkeypatch):
    from src.agents.decision_nodes import _decision_inputs_from_state
    from src.validators.red_flag_detector import RedFlagDetector

    def unexpected_extract(*args, **kwargs):
        pytest.fail("Prepared metrics must not be reparsed")

    monkeypatch.setattr(RedFlagDetector, "extract_metrics", unexpected_extract)
    result = _decision_inputs_from_state(
        {"analysis_snapshot": {"contract_status": "VALID"}},
        "TEST",
        metrics={},
        snapshot={},
    )
    assert not result.snapshot_authoritative
    assert result.health_decision_pct is None
    assert assess_health_gate(result).status == "INDETERMINATE"


def test_shared_decision_inputs_do_not_parse_failed_artifact(monkeypatch):
    from src.agents.decision_nodes import _decision_inputs_from_state
    from src.runtime_diagnostics.artifact_status import failure_artifact
    from src.validators.red_flag_detector import RedFlagDetector

    def extract(report, *, ticker):
        assert report == "" and ticker == "TEST"
        return {}

    monkeypatch.setattr(RedFlagDetector, "extract_metrics", extract)
    result = _decision_inputs_from_state(
        {
            **failure_artifact("fundamentals_report", "Unavailable"),
            "fundamentals_report": "Untrusted health score: 99%",
        },
        "TEST",
    )
    assert result.sector is None
    assert result.health_decision_pct is None
    assert assess_health_gate(result).status == "INDETERMINATE"
