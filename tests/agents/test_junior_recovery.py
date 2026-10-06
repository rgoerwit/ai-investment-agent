"""Junior's existing text-only recovery is bounded in actual node execution."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.agents.analyst_nodes import create_analyst_node
from src.forensic_budget import GraphResearchBudgetPolicy

_VALID = """=== RAW FINANCIAL DATA FOR TEST ===
### TOOL 1: get_financial_metrics
{"sector": "Industrials", "industry": "Machinery", "operatingCashflow": 266690000, "financialCurrency": "AUD"}
### TOOL 2: get_fundamental_analysis
No ADR evidence found; analyst coverage unavailable.
=== END RAW DATA ==="""
_LONG_INVALID = _VALID.removesuffix("=== END RAW DATA ===") + " " * 500
_OBSERVED = json.loads(
    (Path(__file__).parents[1] / "fixtures/junior_incomplete_output.json").read_text()
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial", [_LONG_INVALID, "Financial data was unavailable. " * 30]
)
@pytest.mark.parametrize(
    "recovered,accepted",
    [(_VALID, True), ("Financial data unavailable. " * 30, False), ("", False)],
)
async def test_junior_contract_failure_retries_once_without_tools(
    initial, recovered, accepted
):
    primary, repair = MagicMock(), MagicMock()
    invoke = AsyncMock(
        side_effect=[AIMessage(content=initial), AIMessage(content=recovered)]
    )
    node = create_analyst_node(
        primary,
        "junior_fundamentals_analyst",
        [],
        "raw_fundamentals_data",
        retry_llm=repair,
        allow_retry=True,
    )
    with patch("src.agents.runtime.invoke_with_rate_limit_handling", invoke):
        result = await node(
            {"messages": [], "company_of_interest": "TEST"}, {"configurable": {}}
        )
    assert invoke.await_count == 2
    repair.bind_tools.assert_not_called()
    assert (
        "Use only retained tool evidence"
        in invoke.await_args.args[1]["messages"][-1].content
    )
    assert result["artifact_statuses"]["raw_fundamentals_data"]["ok"] is accepted
    assert result["structural_recovery_events"][0]["final_output_valid"] is accepted
    assert invoke.await_args.kwargs["canonical_agent"] == "Junior Fundamentals Analyst"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["valid", "no_binding", "disabled", "budget", "tool_calls", "timeout"]
)
async def test_junior_recovery_failure_and_skip_paths(case):
    repair = MagicMock() if case != "no_binding" else None
    initial = AIMessage(content=_VALID if case == "valid" else _LONG_INVALID)
    recovered = (
        AIMessage(
            content=_VALID,
            tool_calls=[{"name": "get_financial_metrics", "args": {}, "id": "new"}],
        )
        if case == "tool_calls"
        else TimeoutError("bounded recovery timeout")
    )
    invoke = AsyncMock(side_effect=[initial, recovered])
    policy = (
        GraphResearchBudgetPolicy(
            tool_limits={},
            max_tool_iterations=1,
            max_llm_calls=1,
            max_tool_calls_per_turn=1,
        )
        if case == "budget"
        else None
    )
    node = create_analyst_node(
        MagicMock(),
        "junior_fundamentals_analyst",
        [],
        "raw_fundamentals_data",
        retry_llm=repair,
        allow_retry=case != "disabled",
        research_budget_policy=policy,
    )
    with patch("src.agents.runtime.invoke_with_rate_limit_handling", invoke):
        result = await node(
            {"messages": [], "company_of_interest": "TEST"}, {"configurable": {}}
        )
    assert invoke.await_count == (2 if case in {"tool_calls", "timeout"} else 1)
    assert result["artifact_statuses"]["raw_fundamentals_data"]["ok"] is (
        case == "valid"
    )
    if case == "budget":
        assert (
            "LLM_RECOVERY_NOT_BUDGETED"
            in result["research_budgets"]["junior_fundamentals_analyst"]["outcomes"]
        )
    if case == "tool_calls":
        assert (
            result["structural_recovery_events"][0]["outcome"] == "rejected_tool_calls"
        )


@pytest.mark.asyncio
async def test_retained_incomplete_junior_repair_sees_evidence_without_replacing_canonical_metrics():
    payload = _OBSERVED["metrics_payload"]
    messages = [
        AIMessage(
            content="",
            name="junior_fundamentals_analyst",
            tool_calls=[
                {
                    "id": "metrics",
                    "name": "get_financial_metrics",
                    "args": {"ticker": _OBSERVED["ticker"]},
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(payload),
            tool_call_id="metrics",
            name="get_financial_metrics",
            additional_kwargs={"agent_key": "junior_fundamentals_analyst"},
        ),
    ]
    state = {
        "company_of_interest": _OBSERVED["ticker"],
        "messages": messages,
        "structured_inputs": {
            "raw_financial_metrics": {
                "status": "VALID",
                "agent_key": "junior_fundamentals_analyst",
                "tool_name": "get_financial_metrics",
                "payload": payload,
            }
        },
        "research_budgets": {
            "junior_fundamentals_analyst": _OBSERVED["research_budget"]
        },
    }
    recovered = (
        _OBSERVED["incomplete_output"].replace(str(payload["operatingCashflow"]), "1")
        + "=== END RAW DATA ==="
    )
    invoke = AsyncMock(
        side_effect=[
            AIMessage(content=_OBSERVED["incomplete_output"]),
            AIMessage(content=recovered),
        ]
    )
    policy = GraphResearchBudgetPolicy(
        tool_limits={},
        max_tool_iterations=1,
        max_llm_calls=3,
        max_tool_calls_per_turn=1,
    )
    with patch("src.agents.runtime.invoke_with_rate_limit_handling", invoke):
        result = await create_analyst_node(
            MagicMock(),
            "junior_fundamentals_analyst",
            [],
            "raw_fundamentals_data",
            retry_llm=MagicMock(),
            allow_retry=True,
            research_budget_policy=policy,
        )(state, {"configurable": {}})
    assert invoke.await_count == 2
    retained_tools = [
        message
        for message in invoke.await_args.args[1]["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert json.loads(retained_tools[0].content) == payload
    assert result["artifact_statuses"]["raw_fundamentals_data"]["ok"] is True
    merged = {**state, **result}
    assert (
        merged["structured_inputs"]["raw_financial_metrics"]["payload"][
            "operatingCashflow"
        ]
        == 5534067458048
    )
    assert merged["research_budgets"]["junior_fundamentals_analyst"]["llm_calls"] == 3
