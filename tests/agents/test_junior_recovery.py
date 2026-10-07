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


def _capped_case():
    from src.graph.tool_nodes import _cap_tool_output
    from src.tooling.structured_ingress import build_structured_ingress_record

    payload = {
        "sector": "Industrials",
        "operatingCashflow": 123,
        "public_statement": "x" * 21000,
    }
    text = json.dumps(payload)
    capped = _cap_tool_output(text, "get_financial_metrics")
    output = _VALID.replace(_VALID.splitlines()[2], capped)
    state = {
        "messages": [],
        "company_of_interest": "G.MI",
        "structured_inputs": {
            "raw_financial_metrics": build_structured_ingress_record(
                text,
                agent_key="junior_fundamentals_analyst",
                tool_name="get_financial_metrics",
            )
        },
    }
    return payload, capped, output, state


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_exhausted", [False, True])
async def test_copied_tool_cap_resplice_restores_exact_ingress_without_llm_recovery(
    budget_exhausted,
):
    from src.agents.analyst_nodes import _should_retry_output
    from src.agents.output_validation import validate_required_output
    from src.tooling.structured_ingress import render_structured_ingress_payload

    payload, _, initial, state = _capped_case()
    assert validate_required_output("junior_fundamentals_analyst", initial)["ok"]
    assert _should_retry_output(initial, "junior_fundamentals_analyst")
    invoke = AsyncMock(return_value=AIMessage(content=initial))
    policy = (
        GraphResearchBudgetPolicy(
            tool_limits={},
            max_tool_iterations=1,
            max_llm_calls=1,
            max_tool_calls_per_turn=1,
        )
        if budget_exhausted
        else None
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
    assert invoke.await_count == 1
    assert result["artifact_statuses"]["raw_fundamentals_data"]["ok"]
    body = (
        result["raw_fundamentals_data"]
        .split("### TOOL 1: get_financial_metrics\n")[1]
        .split("\n### TOOL 2:")[0]
    )
    assert body == render_structured_ingress_payload(state, "raw_financial_metrics")
    assert json.loads(body) == payload
    (event,) = result["structural_recovery_events"]
    assert event["repair"] == "structured_ingress_resplice"
    assert event["outcome"] == "accepted_text" and event["final_output_valid"]
    assert event["recovery_model"] is None
    if budget_exhausted:
        assert (
            result["research_budgets"]["junior_fundamentals_analyst"]["llm_calls"] == 1
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "invalid",
        "error_payload",
        "wrong_owner",
        "wrong_tool",
        "missing_wrapper",
        "multiple_sections",
        "outside_metrics",
        "missing_end",
        "malformed_registry",
    ],
)
async def test_resplice_fails_closed_and_invalid_retry_keeps_original(case):
    payload, _, initial, state = _capped_case()
    record = state["structured_inputs"]["raw_financial_metrics"]
    if case == "missing":
        state.pop("structured_inputs")
    elif case == "malformed_registry":
        state["structured_inputs"] = ["malformed"]
    elif case == "invalid":
        record["status"] = "INVALID"
    elif case == "error_payload":
        record["payload"] = {"error": "unavailable"}
    elif case == "wrong_owner":
        record["agent_key"] = "other"
    elif case == "wrong_tool":
        record["tool_name"] = "other_tool"
    elif case == "missing_wrapper":
        initial = initial.removeprefix("=== RAW FINANCIAL DATA FOR TEST ===")
    elif case == "multiple_sections":
        initial += "\n### TOOL 1: get_financial_metrics\n{}"
    elif case == "outside_metrics":
        initial = (
            _VALID
            + "\n[...TRUNCATED 403 chars of tool output 'get_financial_metrics'...]"
        )
    elif case == "missing_end":
        initial = initial.removesuffix("=== END RAW DATA ===")
    invoke = AsyncMock(
        side_effect=[AIMessage(content=initial), AIMessage(content=initial)]
    )
    with patch("src.agents.runtime.invoke_with_rate_limit_handling", invoke):
        result = await create_analyst_node(
            MagicMock(),
            "junior_fundamentals_analyst",
            [],
            "raw_fundamentals_data",
            retry_llm=MagicMock(),
            allow_retry=True,
        )(state, {"configurable": {}})
    assert invoke.await_count == 2
    assert result["artifact_statuses"]["raw_fundamentals_data"]["ok"] is False
    assert result["structural_recovery_events"][0]["outcome"] == "failed"
    assert "repair" not in result["structural_recovery_events"][0]


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "other_tool",
        "other_owner",
        "blocked",
        "error",
        "no_marker",
        "invalid_ingress",
    ],
)
def test_retry_only_restores_valid_capped_metrics_tool_messages(case):
    from src.agents.analyst_nodes import _build_retry_invocation_messages

    payload, capped, _, state = _capped_case()
    msg = ToolMessage(
        content=capped,
        name="get_financial_metrics",
        tool_call_id="m",
        additional_kwargs={"agent_key": "junior_fundamentals_analyst"},
    )
    if case == "other_tool":
        msg.name = "get_fundamental_analysis"
    elif case == "other_owner":
        msg.additional_kwargs["agent_key"] = "other"
    elif case == "blocked":
        msg.additional_kwargs["blocked"] = True
    elif case == "error":
        msg.status = "error"
    elif case == "no_marker":
        msg.content = "{}"
    elif case == "invalid_ingress":
        state["structured_inputs"]["raw_financial_metrics"]["status"] = "INVALID"
    retry = _build_retry_invocation_messages(
        [msg], "junior_fundamentals_analyst", "invalid", state=state
    )
    if case == "valid":
        assert json.loads(retry[0].content) == payload
        assert retry[0].tool_call_id == msg.tool_call_id
    else:
        assert retry[0] is msg
    assert msg.content == ("{}" if case == "no_marker" else capped)


@pytest.mark.parametrize(
    "agent",
    ["junior_fundamentals_analyst", "fundamentals_analyst", "foreign_language_analyst"],
)
@pytest.mark.parametrize(
    "content",
    [_VALID, _LONG_INVALID, "refused", _VALID + "\n[...TRUNCATED 10 chars...]"],
)
def test_recovery_uses_same_structural_validity_predicate(agent, content):
    from src.agents import support
    from src.agents.analyst_nodes import _final_output_invalid, _should_retry_output

    assert _should_retry_output(content, agent) == (
        support._is_output_insufficient(content, agent)
        or _final_output_invalid(content, agent)
    )


@pytest.mark.parametrize("registry", [["bad"], "bad", None])
def test_malformed_ingress_registry_never_raises_or_resplice(registry):
    from src.agents.analyst_nodes import _resplice_junior_metrics

    _, _, output, state = _capped_case()
    state["structured_inputs"] = registry
    assert _resplice_junior_metrics(output, state) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("delta", [-1, 0, 1])
async def test_junior_ingress_respects_persisted_artifact_budget_boundary(delta):
    from src.agents.output_limits import _MAX_STATE_FIELD_CHARS
    from src.graph.tool_nodes import _cap_tool_output
    from src.tooling.structured_ingress import build_structured_ingress_record

    payload = {
        "sector": "Industrials",
        "public_statement": "",
    }
    base_size = len(
        _VALID.replace(_VALID.splitlines()[2], json.dumps(payload, sort_keys=True))
    )
    payload["public_statement"] = "x" * (_MAX_STATE_FIELD_CHARS + delta - base_size)
    full = _VALID.replace(_VALID.splitlines()[2], json.dumps(payload, sort_keys=True))
    assert len(full) == _MAX_STATE_FIELD_CHARS + delta
    capped = _VALID.replace(
        _VALID.splitlines()[2],
        _cap_tool_output(json.dumps(payload), "get_financial_metrics"),
    )
    state = {
        "messages": [],
        "company_of_interest": "TEST",
        "structured_inputs": {
            "raw_financial_metrics": build_structured_ingress_record(
                payload,
                agent_key="junior_fundamentals_analyst",
                tool_name="get_financial_metrics",
            )
        },
    }
    invoke = AsyncMock(side_effect=[AIMessage(content=capped), AIMessage(content=full)])
    with patch("src.agents.runtime.invoke_with_rate_limit_handling", invoke):
        result = await create_analyst_node(
            MagicMock(),
            "junior_fundamentals_analyst",
            [],
            "raw_fundamentals_data",
            retry_llm=MagicMock(),
            allow_retry=True,
        )(state, {"configurable": {}})
    assert invoke.await_count == (2 if delta > 0 else 1)
    assert result["artifact_statuses"]["raw_fundamentals_data"]["ok"] is (delta <= 0)
    assert result["structural_recovery_events"][0]["final_output_valid"] is (delta <= 0)


@pytest.mark.parametrize("size", [20000, 20001, 23456, 35000])
@pytest.mark.parametrize(
    "tool_name", ["get_financial_metrics", "get_fundamental_analysis"]
)
def test_shared_tool_cap_contract_roundtrips_actual_producer(size, tool_name):
    from src.graph.tool_nodes import _cap_tool_output
    from src.utils import format_tool_output_cap, has_tool_output_cap

    content = "a" * 17000 + "b" * (size - 20000) + "c" * 3000
    capped = _cap_tool_output(content, tool_name)
    assert has_tool_output_cap(capped, tool_name) is (size > 20000)
    assert not has_tool_output_cap(capped, tool_name + "_other")
    if size > 20000:
        assert (
            capped
            == "a" * 17000
            + f"\n\n{format_tool_output_cap(size - 20000, tool_name)}\n\n"
            + "c" * 3000
        )


@pytest.mark.parametrize(
    "marker",
    [
        "[...TRUNCATED 1,000 chars of tool output 'get_financial_metrics_other'...]",
        "[...TRUNCATED 1,000 chars of tool output 'GET_FINANCIAL_METRICS'...]",
        "[...TRUNCATED 1,000 chars of tool output 'get_financial_metrics'..]",
        "[...TRUNCATED 1,000 chars...]",
    ],
)
def test_similar_markers_never_authorize_metrics_resplice(marker):
    from src.agents.analyst_nodes import _resplice_junior_metrics

    _, capped, initial, state = _capped_case()
    assert _resplice_junior_metrics(initial.replace(capped, marker), state) is None
