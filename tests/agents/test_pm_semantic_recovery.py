"""An inspected Oct-5 PM fragment must pass semantics before publication."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.decision_nodes import create_portfolio_manager_node
from src.agents.output_validation import validate_required_output
from src.pm_claim_audit import (
    normalize_decision_trace_citations,
    validate_decision_trace,
)

_EMPTY = (Path(__file__).parents[1] / "fixtures/pm_empty_trace.txt").read_text()
_FUNDAMENTALS = "### --- START DATA_BLOCK ---\nPE_RATIO_TTM: 12.0\nVALUATION_INPUT_RELIABILITY: USABLE\n### --- END DATA_BLOCK ---"
_SNAPSHOT = {
    "contract_status": "VALID",
    "claims": {
        "claim:pe": {
            "id": "claim:pe",
            "field": "PE_RATIO_TTM",
            "value": "12.0",
            "period": None,
            "authority": "AGGREGATOR",
            "coverage": "FOUND",
            "decision_eligible": True,
            "decision_role": "SUPPORT",
        }
    },
}


def test_real_empty_trace_is_an_output_contract_failure():
    assert not validate_required_output("portfolio_manager", _EMPTY)["ok"]


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["BUY", "HOLD", "DO_NOT_INITIATE"])
@pytest.mark.parametrize(
    "recovered_facts,accepted",
    [("claim:pe", True), ("NONE", False), ("claim:invented", False)],
)
async def test_semantic_recovery_is_once_and_records_final_validity(
    verdict, recovered_facts, accepted
):
    initial = _EMPTY.replace("DO NOT INITIATE", verdict).replace(
        "DO_NOT_INITIATE", verdict
    )
    recovery = initial.replace(
        "DECISION_FACTS: NONE", f"DECISION_FACTS: {recovered_facts}"
    )
    model = SimpleNamespace(model_name="initial")
    repair = SimpleNamespace(model_name="repair")
    invoke = AsyncMock(
        side_effect=[
            SimpleNamespace(
                content=initial, response_metadata={"model_name": "initial-model"}
            ),
            SimpleNamespace(
                content=recovery,
                response_metadata={"model_name": "accepted-recovery-model"},
            ),
        ]
    )
    state = {
        "company_of_interest": "TEST",
        "pre_screening_result": "PASS",
        "analysis_snapshot": _SNAPSHOT,
        "red_flags": [],
        "fundamentals_report": _FUNDAMENTALS,
    }
    with (
        patch(
            "src.prompts.get_prompt",
            return_value=SimpleNamespace(
                system_message="PM", agent_name="Portfolio Manager"
            ),
        ),
        patch(
            "src.agents.decision_nodes.agent_runtime.invoke_with_rate_limit_handling",
            invoke,
        ),
    ):
        result = await create_portfolio_manager_node(model, None, recovery_llm=repair)(
            state, {}
        )
    assert invoke.await_count == 2
    event = result["structural_recovery_events"][0]
    assert event["final_output_valid"] is accepted
    assert event["outcome"] == ("accepted_text" if accepted else "failed")
    assert result["artifact_statuses"]["final_trade_decision"]["ok"] is accepted
    assert result["artifact_statuses"]["final_trade_decision"].get("model") == (
        "accepted-recovery-model" if accepted else None
    )
    assert result["decision_trace"]["status"] == ("VALID" if accepted else "INVALID")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["missing", "ineligible", "invalid_contract", "context_only"]
)
async def test_no_eligible_evidence_stays_invalid_and_does_not_invent_facts(case):
    response = SimpleNamespace(content=_EMPTY)
    invoke = AsyncMock(return_value=response)
    state = {
        "company_of_interest": "TEST",
        "red_flags": [],
        "fundamentals_report": _FUNDAMENTALS,
    }
    if case != "missing":
        snapshot = deepcopy(_SNAPSHOT)
        if case == "ineligible":
            snapshot["claims"]["claim:pe"]["decision_eligible"] = False
        elif case == "invalid_contract":
            snapshot["contract_status"] = "INVALID"
        else:
            snapshot["claims"]["claim:pe"]["decision_role"] = "CONTEXT"
        state["analysis_snapshot"] = snapshot
    with (
        patch(
            "src.prompts.get_prompt",
            return_value=SimpleNamespace(
                system_message="PM", agent_name="Portfolio Manager"
            ),
        ),
        patch(
            "src.agents.decision_nodes.agent_runtime.invoke_with_rate_limit_handling",
            invoke,
        ),
    ):
        result = await create_portfolio_manager_node(
            SimpleNamespace(model_name="base"), None
        )(state, {})
    assert invoke.await_count == 1
    assert not result["artifact_statuses"]["final_trade_decision"]["ok"]
    assert result["decision_trace"]["decision_facts"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["gate_only", "unknown_fact", "inactive_gate", "tool_calls", "timeout"]
)
async def test_pm_gate_authority_and_recovery_failures(case):
    flag = {"type": "COVERAGE_GAP", "blocks_buy": True, "risk_penalty": 0.0}
    valid = _EMPTY.replace("DECISION_GATES: NONE", "DECISION_GATES: COVERAGE_GAP")
    if case == "gate_only":
        initial = valid
    elif case == "unknown_fact":
        initial = valid.replace(
            "DECISION_FACTS: NONE", "DECISION_FACTS: claim:invented"
        )
    elif case == "inactive_gate":
        initial = valid.replace("COVERAGE_GAP", "INACTIVE_GATE")
    else:
        # An active gate can repair HOLD/DNI deterministically. BUY without a
        # thesis-bearing fact still needs real semantic recovery.
        initial = _EMPTY.replace("DO_NOT_INITIATE", "BUY").replace(
            "DO NOT INITIATE", "BUY"
        )
    recovered = (
        TimeoutError("bounded timeout")
        if case == "timeout"
        else SimpleNamespace(
            content=valid,
            tool_calls=[{"name": "web_search"}] if case == "tool_calls" else [],
        )
    )
    invoke = AsyncMock(side_effect=[SimpleNamespace(content=initial), recovered])
    state = {
        "company_of_interest": "TEST",
        "pre_screening_result": "PASS",
        "fundamentals_report": _FUNDAMENTALS,
        "red_flags": [flag],
    }
    with (
        patch(
            "src.prompts.get_prompt",
            return_value=SimpleNamespace(
                system_message="PM", agent_name="Portfolio Manager"
            ),
        ),
        patch(
            "src.agents.decision_nodes.agent_runtime.invoke_with_rate_limit_handling",
            invoke,
        ),
    ):
        result = await create_portfolio_manager_node(
            SimpleNamespace(model_name="base"),
            None,
            recovery_llm=SimpleNamespace(model_name="repair"),
        )(state, {})
    assert invoke.await_count == (2 if case in {"tool_calls", "timeout"} else 1)
    if case in {"gate_only", "unknown_fact", "inactive_gate"}:
        assert not result.get("structural_recovery_events")
    valid_outcome = case in {"gate_only", "unknown_fact", "inactive_gate"}
    assert result["artifact_statuses"]["final_trade_decision"]["ok"] is valid_outcome
    if valid_outcome:
        assert result["decision_trace"]["status"] == "VALID"
        assert result["decision_trace"]["decision_gates"] == ["COVERAGE_GAP"]
    else:
        assert result["structural_recovery_events"][0]["final_output_valid"] is False


@pytest.mark.parametrize("verdict", ["BUY", "HOLD", "DO_NOT_INITIATE"])
def test_citation_cleanup_preserves_scores_verdict_prose_and_valid_facts(verdict):
    text = _EMPTY.replace("DO_NOT_INITIATE", verdict).replace(
        "DO NOT INITIATE", verdict
    )
    text = text.replace(
        "DECISION_FACTS: NONE", "DECISION_FACTS: claim:pe, claim:invented"
    )
    text = text.replace("DECISION_GATES: NONE", "DECISION_GATES: INACTIVE")
    flag = {"type": "COVERAGE_GAP", "blocks_buy": True}
    clean = normalize_decision_trace_citations(text, _SNAPSHOT, [flag])
    expected = text.replace("claim:pe, claim:invented", "claim:pe").replace(
        "DECISION_GATES: INACTIVE", "DECISION_GATES: COVERAGE_GAP"
    )
    assert clean.rstrip() == expected.rstrip()
    assert normalize_decision_trace_citations(clean, _SNAPSHOT, [flag]) == clean
    assert validate_decision_trace(clean, _SNAPSHOT, [flag])["status"] == "VALID"


@pytest.mark.parametrize("missing", ["block", "facts", "gates"])
def test_citation_cleanup_does_not_invent_missing_structural_fields(missing):
    text = _EMPTY
    if missing == "block":
        text = "HOLD pending verified evidence."
    else:
        field = "DECISION_FACTS" if missing == "facts" else "DECISION_GATES"
        text = text.replace(f"{field}: NONE\n", "")
    clean = normalize_decision_trace_citations(text, _SNAPSHOT, [])
    assert clean.rstrip() == text.rstrip()
    assert validate_decision_trace(clean, _SNAPSHOT, [])["status"] == "INVALID"


def test_cleanup_cannot_make_an_incidental_fact_support_buy():
    snapshot = deepcopy(_SNAPSHOT)
    snapshot["claims"]["claim:pe"]["field"] = "CURRENT_PRICE"
    text = _EMPTY.replace("DO_NOT_INITIATE", "BUY").replace(
        "DECISION_FACTS: NONE", "DECISION_FACTS: claim:pe, invented"
    )
    clean = normalize_decision_trace_citations(text, snapshot, [])
    assert validate_decision_trace(clean, snapshot, [])["status"] == "INVALID"


@pytest.mark.parametrize(
    "boundary", ["glued_start", "glued_end", "duplicate", "truncated"]
)
def test_cleanup_uses_the_shared_normalized_block_boundary_contract(boundary):
    text = _EMPTY.replace("DECISION_FACTS: NONE", "DECISION_FACTS: claim:pe, invented")
    if boundary == "glued_start":
        text = text.replace("\n### --- START PM_BLOCK", " prose.### --- START PM_BLOCK")
    elif boundary == "glued_end":
        text = text.replace(
            "### --- END PM_BLOCK ---",
            "### --- END PM_BLOCK ---### AFTERWORD\nMonitoring only.",
        )
    elif boundary == "duplicate":
        text = _EMPTY + "\n" + text
    else:
        text = text.replace("### --- END PM_BLOCK ---", "")
    clean = normalize_decision_trace_citations(text, _SNAPSHOT, [])
    if boundary == "truncated":
        assert clean == text
    else:
        assert "DECISION_FACTS: claim:pe, invented" not in clean
        assert validate_decision_trace(clean, _SNAPSHOT, [])["status"] == "VALID"
        if boundary == "duplicate":
            assert clean.startswith(_EMPTY)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "coverage,initial,final",
    [
        (None, "DO_NOT_INITIATE", "DO_NOT_INITIATE"),
        (14, "DO_NOT_INITIATE", "HOLD"),
        (15, "HOLD", "DO_NOT_INITIATE"),
        (16, "DO_NOT_INITIATE", "DO_NOT_INITIATE"),
    ],
)
async def test_independent_gate_context_floor_and_saved_verdict_agree(
    coverage, initial, final
):
    from src.liquidity_assessment import LiquidityAssessment

    snapshot = deepcopy(_SNAPSHOT)
    snapshot["scorecards"] = {
        "HEALTH": {"percentage": 75, "decision_eligible": True},
        "GROWTH": {"percentage": 40, "decision_eligible": True},
    }
    if coverage is not None:
        snapshot["claims"]["claim:coverage"] = {
            "id": "claim:coverage",
            "field": "ANALYST_COVERAGE_ENGLISH",
            "value": str(coverage),
            "authority": "PRIMARY",
            "exactness": "EXACT",
            "coverage": "FOUND",
            "decision_eligible": True,
            "decision_role": "GATE_INPUT",
            "source_url": "https://issuer.example/coverage",
            "evidence_id": "evidence:1:abc",
        }
    initial_text = (
        _EMPTY.replace("DECISION_FACTS: NONE", "DECISION_FACTS: claim:pe")
        .replace("DO NOT INITIATE", initial.replace("_", " "))
        .replace("DO_NOT_INITIATE", initial)
    )
    invoke = AsyncMock(return_value=SimpleNamespace(content=initial_text))
    state = {
        "company_of_interest": "TEST",
        "pre_screening_result": "PASS",
        "red_flags": [],
        "analysis_snapshot": snapshot,
        "fundamentals_report": _FUNDAMENTALS.replace(
            "PE_RATIO_TTM: 12.0", "PE_RATIO_TTM: 12.0\nREVENUE_CAGR_3Y: 16%"
        ),
        "liquidity_assessment": LiquidityAssessment(
            "PASS", average_daily_turnover_usd=500_000
        ).to_dict(),
    }
    with (
        patch(
            "src.prompts.get_prompt",
            return_value=SimpleNamespace(
                system_message="PM", agent_name="Portfolio Manager"
            ),
        ),
        patch(
            "src.agents.decision_nodes.agent_runtime.invoke_with_rate_limit_handling",
            invoke,
        ),
    ):
        result = await create_portfolio_manager_node(
            SimpleNamespace(model_name="initial"), None
        )(state, {})
    assert result["artifact_statuses"]["final_trade_decision"]["ok"]
    assert result["decision_policy"]["final_verdict"] == final
    assert result["decision_policy"]["original_verdict"] == initial
    assert f"VERDICT: {final}" in result["final_trade_decision"]
    gate = result["decision_policy"]["initiation_gates"][0]
    status = (
        "INDETERMINATE" if coverage is None else "PASS" if coverage < 15 else "FAIL"
    )
    assert gate["status"] == status
    prompt = invoke.call_args.args[1][0].content
    assert gate["value"] == coverage
    value = "Unknown" if coverage is None else f"{coverage} English analysts"
    assert f"Analyst Coverage: {value} - {status};" in prompt
    assert (
        "US Revenue: Unknown - INDETERMINATE; Verified US revenue unavailable" in prompt
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["unavailable", "unreliable"])
async def test_indeterminate_growth_status_agrees_in_prompt_and_saved_policy(case):
    snapshot = deepcopy(_SNAPSHOT)
    metrics = {"adjusted_health_score": 75, "pe_ratio": 12, "revenue_cagr_3y": 16}
    if case == "unreliable":
        snapshot["scorecards"] = {
            "HEALTH": {"percentage": 75, "decision_eligible": True},
            "GROWTH": {"percentage": 40, "decision_eligible": True},
        }
        metrics["growth_score_consistency"] = "SUSPECT"
    initial = _EMPTY.replace("DECISION_FACTS: NONE", "DECISION_FACTS: claim:pe")
    invoke = AsyncMock(return_value=SimpleNamespace(content=initial))
    with (
        patch(
            "src.prompts.get_prompt",
            return_value=SimpleNamespace(
                system_message="PM", agent_name="Portfolio Manager"
            ),
        ),
        patch(
            "src.agents.decision_nodes.agent_runtime.invoke_with_rate_limit_handling",
            invoke,
        ),
        patch(
            "src.validators.red_flag_detector.RedFlagDetector.extract_metrics",
            return_value=metrics,
        ),
    ):
        result = await create_portfolio_manager_node(
            SimpleNamespace(model_name="initial"), None
        )(
            {
                "company_of_interest": "TEST",
                "pre_screening_result": "PASS",
                "red_flags": [],
                "analysis_snapshot": snapshot,
                "fundamentals_report": _FUNDAMENTALS,
            },
            {},
        )
    assert result["decision_policy"]["growth_gate"]["status"] == "INDETERMINATE"
    assert result["decision_policy"]["growth_gate"]["exception"] is None
    assert "STATUS: INDETERMINATE" in invoke.call_args.args[1][0].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pe,tender,final",
    [(12.52, False, "HOLD"), (14.0, False, "HOLD"), (12.52, True, "BUY")],
)
async def test_exception_ceiling_is_saved_before_quick_qualification_and_trace(
    pe, tender, final
):
    from src.liquidity_assessment import LiquidityAssessment

    snapshot = deepcopy(_SNAPSHOT)
    snapshot["scorecards"] = {
        "HEALTH": {"percentage": 90.9, "decision_eligible": True},
        "GROWTH": {"percentage": 0.0, "decision_eligible": True},
    }
    for name, field_name, value in (
        ("coverage", "ANALYST_COVERAGE_ENGLISH", "4"),
        ("us", "US_REVENUE_PERCENT", "10%"),
    ):
        snapshot["claims"][f"claim:{name}"] = {
            "id": f"claim:{name}",
            "field": field_name,
            "value": value,
            "authority": "PRIMARY",
            "exactness": "EXACT",
            "coverage": "FOUND",
            "decision_eligible": True,
            "decision_role": "GATE_INPUT",
            "source_url": f"https://issuer.example/{name}",
            "evidence_id": "evidence:1:abc",
        }
    initial = (
        _EMPTY.replace("DECISION_FACTS: NONE", "DECISION_FACTS: claim:pe")
        .replace("DO NOT INITIATE", "BUY")
        .replace("DO_NOT_INITIATE", "BUY")
    )
    invoke = AsyncMock(return_value=SimpleNamespace(content=initial))
    state = {
        "company_of_interest": "TEST",
        "pre_screening_result": "PASS",
        "red_flags": [],
        "analysis_snapshot": snapshot,
        "fundamentals_report": _FUNDAMENTALS.replace(
            "PE_RATIO_TTM: 12.0",
            f"PE_RATIO_TTM: {pe}\nREVENUE_CAGR_3Y: 5.1%\nM_AND_A_STATUS: {'ACTIVE_TENDER' if tender else 'NONE'}\nM_AND_A_TENDER_PRICE: {'USD 15.00' if tender else 'N/A'}\nM_AND_A_SUBJECT_ROLE: {'TARGET' if tender else 'N/A'}",
        ),
        "liquidity_assessment": LiquidityAssessment(
            "PASS", average_daily_turnover_usd=500_000
        ).to_dict(),
    }
    context = SimpleNamespace(quick_mode=True, ticker="TEST", trade_date="2026-10-06")
    with (
        patch(
            "src.prompts.get_prompt",
            return_value=SimpleNamespace(
                system_message="PM", agent_name="Portfolio Manager"
            ),
        ),
        patch(
            "src.agents.decision_nodes.agent_runtime.invoke_with_rate_limit_handling",
            invoke,
        ),
    ):
        result = await create_portfolio_manager_node(
            SimpleNamespace(model_name="initial"), None
        )(state, {"configurable": {"context": context}})
    assert result["artifact_statuses"]["final_trade_decision"]["ok"]
    assert result["decision_policy"]["final_verdict"] == final
    assert f"VERDICT: {final}" in result["final_trade_decision"]
    assert result["decision_trace"]["verdict"] == final
    adjustments = result["decision_policy"]["adjustments"]
    if tender:
        assert not adjustments
        assert "MAX_VERDICT: HOLD" not in invoke.call_args.args[1][0].content
    else:
        assert (
            len(adjustments) == 1
            and adjustments[0]["kind"] == "growth_exception_ceiling"
        )
        assert "MAX_VERDICT: HOLD" in invoke.call_args.args[1][0].content
        assert not result["decision_policy"]["qualifications"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["normal", "semantic_repair", "structural_repair", "repair_fails"]
)
@pytest.mark.parametrize("initial_verdict", ["BUY", "HOLD"])
async def test_health_policy_uses_existing_bounded_pm_path_and_preserves_producer(
    path, initial_verdict
):
    from src.report_generator import QuietModeReporter

    snapshot = deepcopy(_SNAPSHOT)
    snapshot["scorecards"] = {
        "HEALTH": {"percentage": 39.9, "decision_eligible": True},
        "GROWTH": {"percentage": 60, "decision_eligible": True},
    }
    valid = (
        _EMPTY.replace("DECISION_FACTS: NONE", "DECISION_FACTS: claim:pe")
        .replace("DO NOT INITIATE", initial_verdict)
        .replace("DO_NOT_INITIATE", initial_verdict)
    )
    valid = valid.replace(
        "RISK_TALLY: 1.00",
        "POSITION_SIZE: 2.5\nSHOW_VALUATION_CHART: YES\nVALUATION_DISCOUNT: 1.0\nRISK_TALLY: 1.00",
    )
    initial = (
        valid
        if path == "normal"
        else valid.replace("DECISION_FACTS: claim:pe", "DECISION_FACTS: NONE")
        if path == "semantic_repair"
        else "Incomplete PM output"
    )
    responses = [
        SimpleNamespace(
            content=initial, response_metadata={"model_name": "original-model"}
        )
    ]
    if path != "normal":
        responses.append(
            TimeoutError("bounded repair")
            if path == "repair_fails"
            else SimpleNamespace(
                content=valid, response_metadata={"model_name": "repaired-model"}
            )
        )
    invoke = AsyncMock(side_effect=responses)
    with (
        patch(
            "src.prompts.get_prompt",
            return_value=SimpleNamespace(
                system_message="PM", agent_name="Portfolio Manager"
            ),
        ),
        patch(
            "src.agents.decision_nodes.agent_runtime.invoke_with_rate_limit_handling",
            invoke,
        ),
    ):
        result = await create_portfolio_manager_node(
            SimpleNamespace(model_name="initial"),
            None,
            recovery_llm=SimpleNamespace(model_name="repair"),
        )(
            {
                "company_of_interest": "TEST",
                "pre_screening_result": "PASS",
                "red_flags": [],
                "analysis_snapshot": snapshot,
                "fundamentals_report": _FUNDAMENTALS,
            },
            {},
        )
    assert invoke.await_count == (1 if path == "normal" else 2)
    if path == "repair_fails":
        assert not result["artifact_statuses"]["final_trade_decision"]["ok"]
        assert not result.get("decision_policy")
        return
    policy = result["decision_policy"]
    assert (
        policy["original_verdict"] == initial_verdict
        and policy["final_verdict"] == "DO_NOT_INITIATE"
    )
    assert (
        policy["health_gate"]["status"] == "FAIL"
        and policy["health_gate"]["value"] == 39.9
    )
    assert policy["adjustments"][0]["rule"] == "financial_health_hard_fail"
    assert (
        result["decision_trace"]["status"] == "VALID"
        and result["decision_trace"]["verdict"] == "DO_NOT_INITIATE"
    )
    assert result["decision_trace"]["decision_facts"] == ["claim:pe"]
    assert result["artifact_statuses"]["final_trade_decision"]["model"] == (
        "original-model" if path == "normal" else "repaired-model"
    )
    assert (
        result["artifact_statuses"]["final_trade_decision"]["provider"]
        != "deterministic"
    )
    output = result["final_trade_decision"]
    assert (
        "VERDICT: DO_NOT_INITIATE" in output
        and "PORTFOLIO MANAGER VERDICT: DO NOT INITIATE" in output
    )
    assert "Financial Health**: 39.9% - FAIL" in output
    from src.charts.extractors.pm_block import extract_pm_block

    assert extract_pm_block(output).position_size == 0
    report = QuietModeReporter("TEST").generate_report(
        {**result, "chart_paths": {"mock": "no writes"}}
    )
    assert (
        "Financial Health: 39.9% - FAIL" in report
        and "Hard Fail Result: FAIL" in report
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["BUY", "HOLD"])
@pytest.mark.parametrize(
    "case",
    [
        "qualified",
        "tender",
        "tender_coverage_fail",
        "ineligible_ratio",
        "conflicting_ratio",
    ],
)
async def test_actual_pm_preserves_health_exception_tender_and_independent_blockers(
    verdict, case
):
    from src.liquidity_assessment import LiquidityAssessment

    tender = case.startswith("tender")
    snapshot = deepcopy(_SNAPSHOT)
    snapshot["scorecards"] = {
        "HEALTH": {"percentage": 39.9 if tender else 45, "decision_eligible": True},
        "GROWTH": {"percentage": 40 if tender else 60, "decision_eligible": True},
    }
    for field, value in (("PB_RATIO", "0.59"), ("CURRENT_RATIO_RAW", "1.51")):
        snapshot["claims"][f"claim:{field}"] = {
            "id": f"claim:{field}",
            "field": field,
            "kind": "FACT",
            "value": value,
            "authority": "AGGREGATOR",
            "coverage": "FOUND",
            "exactness": "EXACT",
            "decision_eligible": True,
            "decision_role": "SUPPORT",
        }
    for field, value in (
        ("ANALYST_COVERAGE_ENGLISH", "15" if case == "tender_coverage_fail" else "4"),
        ("US_REVENUE_PERCENT", "10%"),
    ):
        snapshot["claims"][f"claim:{field}"] = {
            "id": f"claim:{field}",
            "field": field,
            "kind": "FACT",
            "value": value,
            "authority": "PRIMARY",
            "coverage": "FOUND",
            "exactness": "EXACT",
            "decision_eligible": True,
            "decision_role": "GATE_INPUT",
            "source_url": "https://issuer.example/evidence",
            "evidence_id": "evidence:1:abc",
        }
    if case == "ineligible_ratio":
        snapshot["claims"]["claim:PB_RATIO"]["decision_eligible"] = False
    if case == "conflicting_ratio":
        snapshot["claims"]["claim:conflict"] = {
            **snapshot["claims"]["claim:PB_RATIO"],
            "id": "claim:conflict",
            "value": "0.7",
        }
    content = (
        _EMPTY.replace("DECISION_FACTS: NONE", "DECISION_FACTS: claim:pe")
        .replace("DO NOT INITIATE", verdict)
        .replace("DO_NOT_INITIATE", verdict)
    )
    invoke = AsyncMock(
        return_value=SimpleNamespace(
            content=content, response_metadata={"model_name": "accepted-fallback-model"}
        )
    )
    with (
        patch(
            "src.prompts.get_prompt",
            return_value=SimpleNamespace(
                system_message="PM", agent_name="Portfolio Manager"
            ),
        ),
        patch(
            "src.agents.decision_nodes.agent_runtime.invoke_with_rate_limit_handling",
            invoke,
        ),
    ):
        result = await create_portfolio_manager_node(
            SimpleNamespace(model_name="base"), None
        )(
            {
                "company_of_interest": "TEST",
                "pre_screening_result": "PASS",
                "red_flags": [],
                "analysis_snapshot": snapshot,
                "fundamentals_report": _FUNDAMENTALS.replace(
                    "### --- END DATA_BLOCK ---",
                    f"M_AND_A_STATUS: {'ACTIVE_TENDER' if tender else 'NONE'}\nM_AND_A_TENDER_PRICE: {'USD 15.00' if tender else 'N/A'}\nM_AND_A_SUBJECT_ROLE: {'TARGET' if tender else 'N/A'}\n### --- END DATA_BLOCK ---",
                ),
                "liquidity_assessment": LiquidityAssessment(
                    "PASS", average_daily_turnover_usd=500_000
                ).to_dict(),
            },
            {},
        )
    expected = (
        "DO_NOT_INITIATE"
        if case in {"tender_coverage_fail", "ineligible_ratio", "conflicting_ratio"}
        else verdict
    )
    assert invoke.await_count == 1
    assert result["artifact_statuses"]["final_trade_decision"]["ok"]
    assert (
        result["artifact_statuses"]["final_trade_decision"]["model"]
        == "accepted-fallback-model"
    )
    assert (
        result["decision_policy"]["final_verdict"] == expected
        and result["decision_trace"]["verdict"] == expected
    )
    if tender:
        assert not result["decision_policy"]["health_gate"]["applicable"]
        assert result["decision_policy"]["health_gate"]["status"] == "FAIL"
    if case == "qualified":
        assert (
            result["decision_policy"]["health_gate"]["exception"]
            == "deep_value_liquidity"
        )
        assert set(result["decision_policy"]["health_gate"]["supporting_ids"]) == {
            "claim:PB_RATIO",
            "claim:CURRENT_RATIO_RAW",
        }
