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
            SimpleNamespace(content=initial),
            SimpleNamespace(content=recovery),
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
