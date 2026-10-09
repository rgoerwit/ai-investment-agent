"""Code relays only the current Junior acquisition before any display caps."""

import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from src.agents.analyst_nodes import create_analyst_node
from src.agents.state import AgentState
from src.forensic_budget import graph_research_budget_policies
from src.graph.routing import (
    FUNDAMENTALS_BARRIER,
    fundamentals_sync_router,
    route_analyst_tools,
)
from src.graph.tool_nodes import create_agent_tool_node
from src.runtime_services import RuntimeServices, use_runtime_services
from src.tooling.inspection_service import InspectionService
from src.tooling.runtime import ToolCallBlocked, ToolExecutionService, ToolResult
from src.tooling.structured_ingress import get_structured_ingress_payload

JUNIOR = "junior_fundamentals_analyst"
METRICS = "get_financial_metrics"
QUAL = "get_fundamental_analysis"
FIELD = "raw_fundamentals_data"
PAYLOAD = {
    "sector": "Industrials",
    "industry": "Machinery",
    "operatingCashflow": 123,
    "financialCurrency": "AUD",
}
POLICY = graph_research_budget_policies(quick_mode=True)[JUNIOR]
OBSERVED = json.loads(
    (Path(__file__).parents[1] / "fixtures/junior_incomplete_output.json").read_text()
)


def calls(names=(METRICS, QUAL), ticker="TEST"):
    return [
        {"name": name, "args": {"ticker": ticker}, "id": f"call-{index}"}
        for index, name in enumerate(names)
    ]


def tools(
    metrics=None, qualitative="ADR unavailable; see https://example.com/exact?a=1&b=2"
):
    result = []
    for name, value in (
        (METRICS, json.dumps(PAYLOAD) if metrics is None else metrics),
        (QUAL, qualitative),
    ):
        tool = MagicMock()
        tool.name = name
        tool.ainvoke = AsyncMock(
            side_effect=value if isinstance(value, BaseException) else None,
            return_value=value,
        )
        result.append(tool)
    return result


async def relay(
    tool_list, requested=None, *, ticker="TEST", policy=POLICY, messages=()
):
    state = {
        "company_of_interest": ticker,
        "messages": [
            *messages,
            AIMessage(
                content="",
                name=JUNIOR,
                tool_calls=calls() if requested is None else requested,
            ),
        ],
        "research_budgets": {JUNIOR: {"llm_calls": 1}},
    }
    result = await create_agent_tool_node(tool_list, JUNIOR, budget_policy=policy)(
        state, {}
    )
    return state, result


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [0, 19_999, 20_000, 20_001, 35_000])
async def test_full_inspected_outputs_survive_message_caps_and_one_model_turn(size):
    metrics = json.dumps({**PAYLOAD, "summary": "x" * size})
    qualitative = (
        '<search_results source="tavily">https://example.com/exact?a=1&b=2\n[NOTE: Data truncated\n'
        + "y" * size
        + "</search_results>"
    )
    chosen = tools(metrics, qualitative)
    invoke = AsyncMock(return_value=AIMessage(content="", tool_calls=calls()))
    analyst = create_analyst_node(
        MagicMock(),
        JUNIOR,
        chosen,
        FIELD,
        retry_llm=MagicMock(),
        allow_retry=True,
        research_budget_policy=POLICY,
    )
    with patch("src.agents.runtime.invoke_with_rate_limit_handling", invoke):
        initial = await analyst({"messages": [], "company_of_interest": "TEST"}, {})
        acquired = await create_agent_tool_node(chosen, JUNIOR, budget_policy=POLICY)(
            {"company_of_interest": "TEST", **initial}, {}
        )
        terminal = await analyst({"company_of_interest": "TEST", **acquired}, {})
    assert invoke.await_count == 1
    assert metrics in acquired[FIELD] and qualitative in acquired[FIELD]
    assert acquired["artifact_statuses"][FIELD]["ok"]
    assert acquired["artifact_statuses"][FIELD]["provider"] == "code"
    assert get_structured_ingress_payload(acquired, "raw_financial_metrics")[
        0
    ] == json.loads(metrics)
    assert not terminal["messages"][0].tool_calls
    assert terminal[FIELD] == acquired[FIELD]
    assert terminal["artifact_statuses"][FIELD] == acquired["artifact_statuses"][FIELD]
    assert acquired["research_budgets"][JUNIOR]["llm_calls"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "duplicate",
        "wrong_ticker",
        "unknown",
        "empty",
        "malformed",
        "embedded_error",
        "failed",
        "timeout",
        "blocked",
    ],
)
async def test_bad_acquisition_is_terminal_and_never_evidence(case):
    chosen = tools()
    requested = calls()
    if case == "missing":
        requested = calls((METRICS,))
    if case == "duplicate":
        requested = calls((METRICS, METRICS))
    if case == "wrong_ticker":
        requested = calls(ticker="OTHER")
    if case == "unknown":
        requested = calls(("unknown", QUAL))
    if case == "empty":
        chosen = tools(qualitative="")
    if case == "malformed":
        chosen = tools(metrics='{"sector":')
    if case == "embedded_error":
        chosen = tools(
            qualitative='STATUS: EVIDENCE_FOUND\n{"error":"provider unavailable"}'
        )
    if case == "failed":
        chosen = tools(qualitative=ValueError("private exception text"))
    if case == "timeout":
        chosen = tools(qualitative=TimeoutError())
    service = ToolExecutionService()
    if case == "blocked":
        service.execute = AsyncMock(
            side_effect=[
                ToolResult(json.dumps(PAYLOAD)),
                ToolResult("must not relay", blocked=True),
            ]
        )
    with use_runtime_services(
        RuntimeServices(tool_service=service, inspection_service=InspectionService())
    ):
        state, result = await relay(chosen, requested)
    assert not result["artifact_statuses"][FIELD]["ok"]
    assert result["artifact_statuses"][FIELD]["complete"]
    assert "STATUS: UNAVAILABLE" in result[FIELD]
    assert "private exception text" not in result[FIELD]
    assert "must not relay" not in result[FIELD]
    assert result["research_budgets"][JUNIOR]["stop_reason"] == "ACQUISITION_INCOMPLETE"
    if case in {
        "empty",
        "embedded_error",
        "failed",
        "timeout",
        "blocked",
        "missing",
    }:
        assert (
            get_structured_ingress_payload(result, "raw_financial_metrics")[0]
            == PAYLOAD
        )
    invoke = AsyncMock()
    with patch("src.agents.runtime.invoke_with_rate_limit_handling", invoke):
        terminal = await create_analyst_node(MagicMock(), JUNIOR, chosen, FIELD)(
            {**state, **result}, {}
        )
    invoke.assert_not_awaited()
    assert not terminal["messages"][0].tool_calls


@pytest.mark.asyncio
async def test_other_agent_and_old_round_are_not_relay_sources():
    old = AIMessage(content="", name="news_analyst", tool_calls=calls(ticker="OTHER"))
    _, result = await relay(tools(), calls((METRICS,)), messages=[old])
    assert "https://example.com" not in result[FIELD]
    assert "MISSING_OR_DUPLICATE_ACQUISITION" in result[FIELD]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "final", ["", OBSERVED["incomplete_output"], "provider returned no calls"]
)
async def test_model_text_without_acquisition_never_triggers_copy_recovery(final):
    invoke = AsyncMock(return_value=AIMessage(content=final))
    with patch("src.agents.runtime.invoke_with_rate_limit_handling", invoke):
        result = await create_analyst_node(
            MagicMock(),
            JUNIOR,
            tools(),
            FIELD,
            retry_llm=MagicMock(),
            allow_retry=True,
            research_budget_policy=POLICY,
        )({"messages": [], "company_of_interest": "TEST"}, {})
    assert invoke.await_count == 1
    assert not result["artifact_statuses"][FIELD]["ok"]
    assert not result.get("structural_recovery_events")


@pytest.mark.asyncio
@pytest.mark.parametrize("delta", [-1, 0, 1])
async def test_raw_state_budget_is_honest_and_does_not_reduce_canonical_metrics(delta):
    _, base = await relay(tools())
    padding = 200_000 - len(base[FIELD]) + delta
    qualitative = (
        "ADR unavailable; see https://example.com/exact?a=1&b=2" + "x" * padding
    )
    _, result = await relay(tools(qualitative=qualitative))
    assert result["artifact_statuses"][FIELD]["ok"] is (delta <= 0)
    assert get_structured_ingress_payload(result, "raw_financial_metrics")[0] == PAYLOAD
    if delta > 0:
        assert result[FIELD].endswith("[...truncated]")
        assert (
            result["research_budgets"][JUNIOR]["stop_reason"]
            == "ACQUISITION_INCOMPLETE"
        )


@pytest.mark.asyncio
async def test_cancellation_propagates_without_fabricated_relay():
    with pytest.raises(asyncio.CancelledError):
        await relay(tools(qualitative=asyncio.CancelledError()))


@pytest.mark.asyncio
@pytest.mark.parametrize("quick_mode", [False, True])
async def test_compiled_graph_closes_once_and_preserves_downstream_artifact(quick_mode):
    from src.eval.prompt_checks import PromptCheckCollector

    collector = PromptCheckCollector()
    chosen = tools()
    policy = graph_research_budget_policies(quick_mode=quick_mode)[JUNIOR]
    graph = StateGraph(AgentState)
    graph.add_node(
        "junior",
        collector.wrap_node(
            "Junior Fundamentals Analyst",
            create_analyst_node(
                MagicMock(), JUNIOR, chosen, FIELD, research_budget_policy=policy
            ),
        ),
    )
    graph.add_node(
        "tools", create_agent_tool_node(chosen, JUNIOR, budget_policy=policy)
    )

    async def parallel(state):
        return {"news_report": "independent"}

    graph.add_node("parallel", parallel)
    graph.add_edge(START, "junior")
    graph.add_edge(START, "parallel")
    graph.add_conditional_edges(
        "junior", route_analyst_tools(JUNIOR), {"tools": "tools", "continue": END}
    )
    graph.add_edge("tools", "junior")
    graph.add_edge("parallel", END)
    invoke = AsyncMock(return_value=AIMessage(content="", tool_calls=calls()))
    with patch("src.agents.runtime.invoke_with_rate_limit_handling", invoke):
        result = await graph.compile().ainvoke(
            {"messages": [], "company_of_interest": "TEST"}
        )
    assert invoke.await_count == 1
    assert result["news_report"] == "independent"
    assert result[FIELD].endswith("=== END RAW DATA ===")
    assert result["artifact_statuses"][FIELD]["ok"]

    assert collector.collected_outputs()[JUNIOR].artifact_text == result[FIELD]
    assert (
        collector.collected_outputs()[JUNIOR].artifact_status
        == result["artifact_statuses"][FIELD]
    )


@pytest.mark.parametrize("module", ["src.utils", "src.agents.analyst_nodes"])
def test_fresh_process_imports(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_metrics_state_overflow_preserves_full_canonical_payload():
    payload = {**PAYLOAD, "summary": "x" * 210_000}
    _, result = await relay(tools(metrics=json.dumps(payload)))
    assert not result["artifact_statuses"][FIELD]["ok"]
    assert get_structured_ingress_payload(result, "raw_financial_metrics")[0] == payload
    assert (
        result["research_budgets"][JUNIOR]["outcomes"].count("ACQUISITION_INCOMPLETE")
        == 1
    )
    assert "CODE_RELAY_COMPLETE" not in result["research_budgets"][JUNIOR]["outcomes"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_shared_factory_concurrent_tickers_keep_owned_results(reverse):
    """Force interleaving rather than relying on the scheduler to expose leakage."""
    arrivals = 0
    ready = asyncio.Event()

    async def acquire(args, *, name):
        nonlocal arrivals
        arrivals += 1
        if arrivals == 4:
            ready.set()
        await ready.wait()
        ticker = args["ticker"]
        if name == METRICS:
            return json.dumps({**PAYLOAD, "companyName": ticker})
        return f"Inspected qualitative evidence for {ticker}"

    chosen = tools()
    for tool in chosen:

        async def run(args, name=tool.name):
            return await acquire(args, name=name)

        tool.ainvoke.side_effect = run
    node = create_agent_tool_node(chosen, JUNIOR, budget_policy=POLICY)

    async def execute(ticker):
        requested = calls(ticker=ticker)
        if reverse:
            requested.reverse()
        return await node(
            {
                "company_of_interest": ticker,
                "messages": [AIMessage(content="", name=JUNIOR, tool_calls=requested)],
            },
            {},
        )

    first, second = await asyncio.wait_for(
        asyncio.gather(execute("FIRST.T"), execute("SECOND.T")), timeout=2
    )
    for result, ticker, other in (
        (first, "FIRST.T", "SECOND.T"),
        (second, "SECOND.T", "FIRST.T"),
    ):
        assert result["artifact_statuses"][FIELD]["ok"]
        assert ticker in result[FIELD] and other not in result[FIELD]
        assert (
            get_structured_ingress_payload(result, "raw_financial_metrics")[0][
                "companyName"
            ]
            == ticker
        )
        assert result[FIELD].index("### TOOL 1") < result[FIELD].index("### TOOL 2")
        assert result["research_budgets"][JUNIOR]["tool_calls"] == {METRICS: 1, QUAL: 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action", ["replace", "block_before", "block_after", "raise_after"]
)
async def test_relay_respects_real_execution_hook_chain(action):
    """Mock the provider, not the service boundary whose authority matters."""
    observed = []

    class Hook:
        async def before(self, call):
            observed.append(("before", call.name))
            if call.name == QUAL and action == "block_before":
                raise ToolCallBlocked("POLICY_REJECTED")
            return call

        async def after(self, call, result):
            observed.append(("after", call.name))
            if call.name != QUAL:
                return result
            if action == "raise_after":
                raise ValueError("untrusted-secret-body")
            return ToolResult(
                "post-inspection approved evidence",
                blocked=action == "block_after",
                findings=["INSPECTED"],
            )

    chosen = tools(qualitative="pre-inspection untrusted evidence")
    with use_runtime_services(
        RuntimeServices(
            tool_service=ToolExecutionService([Hook()]),
            inspection_service=InspectionService(),
        )
    ):
        _, result = await relay(chosen)
    assert ("before", METRICS) in observed
    assert ("after", METRICS) in observed
    assert result["artifact_statuses"][FIELD]["ok"] is (action == "replace")
    assert "pre-inspection untrusted evidence" not in result[FIELD]
    assert "untrusted-secret-body" not in result[FIELD]
    assert ("post-inspection approved evidence" in result[FIELD]) is (
        action == "replace"
    )
    assert get_structured_ingress_payload(result, "raw_financial_metrics")[0] == PAYLOAD
    if action == "block_before":
        chosen[1].ainvoke.assert_not_awaited()
        assert ("after", QUAL) not in observed


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["node", "service"])
async def test_real_tool_deadline_cancels_worker_and_retains_neighbor(
    monkeypatch, owner
):
    cancelled = asyncio.Event()
    started = asyncio.Event()

    async def stalled(args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    chosen = tools()
    chosen[1].ainvoke.side_effect = stalled
    timeout_owner = (
        "src.graph.tool_nodes.settings_config.tool_execution_with_inspection_timeout_seconds"
        if owner == "node"
        else "src.tooling.runtime.settings_config.tool_execution_timeout_seconds"
    )
    monkeypatch.setattr(timeout_owner, 0.02)
    _, result = await asyncio.wait_for(relay(chosen), timeout=2)
    assert started.is_set() and cancelled.is_set()
    assert not result["artifact_statuses"][FIELD]["ok"]
    assert get_structured_ingress_payload(result, "raw_financial_metrics")[0] == PAYLOAD
    assert "TOOL_TIMEOUT" in result["messages"][1].content


@pytest.mark.asyncio
@pytest.mark.parametrize("quick_mode", [False, True])
@pytest.mark.parametrize("missing", [None, QUAL, METRICS])
async def test_actual_fan_in_releases_senior_once_even_when_junior_fails(
    quick_mode, missing
):
    from src.reporting.state_access import get_raw_fundamentals_data
    from src.runtime_diagnostics import success_artifact

    chosen = tools()
    policy = graph_research_budget_policies(quick_mode=quick_mode)[JUNIOR]
    graph = StateGraph(AgentState)
    graph.add_node(
        "junior",
        create_analyst_node(
            MagicMock(), JUNIOR, chosen, FIELD, research_budget_policy=policy
        ),
    )
    graph.add_node(
        "tools", create_agent_tool_node(chosen, JUNIOR, budget_policy=policy)
    )
    graph.add_node("barrier", FUNDAMENTALS_BARRIER.arrival_update)

    async def foreign(state):
        return success_artifact("foreign_language_report", "native sources")

    async def legal(state):
        return success_artifact("legal_report", "legal assessment")

    seen = []

    async def senior(state):
        seen.append(state)
        if missing != METRICS:
            assert json.loads(get_raw_fundamentals_data(state)) == PAYLOAD
        else:
            assert (
                get_structured_ingress_payload(state, "raw_financial_metrics")[0]
                is None
            )
            assert "STATUS: UNAVAILABLE" in get_raw_fundamentals_data(state)
        return success_artifact("fundamentals_report", "senior completion")

    graph.add_node("foreign", foreign)
    graph.add_node("legal", legal)
    graph.add_node("Fundamentals Analyst", senior)
    for branch in ("junior", "foreign", "legal"):
        graph.add_edge(START, branch)
    graph.add_conditional_edges(
        "junior", route_analyst_tools(JUNIOR), {"tools": "tools", "continue": "barrier"}
    )
    graph.add_edge("tools", "junior")
    for branch in ("foreign", "legal"):
        graph.add_edge(branch, "barrier")
    graph.add_conditional_edges(
        "barrier",
        fundamentals_sync_router,
        {"Fundamentals Analyst": "Fundamentals Analyst", "__end__": END},
    )
    graph.add_edge("Fundamentals Analyst", END)
    requested = calls(tuple(name for name in (METRICS, QUAL) if name != missing))
    invoke = AsyncMock(return_value=AIMessage(content="", tool_calls=requested))
    with patch("src.agents.runtime.invoke_with_rate_limit_handling", invoke):
        result = await asyncio.wait_for(
            graph.compile().ainvoke({"messages": [], "company_of_interest": "TEST"}),
            timeout=2,
        )
    assert len(seen) == 1
    assert invoke.await_count == 1
    assert result["artifact_statuses"][FIELD]["ok"] is (missing is None)
    assert result["artifact_statuses"]["fundamentals_report"]["ok"]


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["round", "fanout", "tool", "missing_ticker"])
async def test_denied_acquisition_never_executes_or_promotes_tools(limit):
    policy = POLICY
    requested = calls()
    if limit == "round":
        policy = replace(POLICY, max_tool_iterations=0)
    elif limit == "fanout":
        policy = replace(POLICY, max_tool_calls_per_turn=0)
    elif limit == "tool":
        policy = replace(POLICY, tool_limits={METRICS: 0, QUAL: 0})
    else:
        for call in requested:
            call["args"] = {}
    chosen = tools()
    _, result = await relay(chosen, requested, policy=policy)
    for tool in chosen:
        tool.ainvoke.assert_not_awaited()
    assert not result["artifact_statuses"][FIELD]["ok"]
    assert get_structured_ingress_payload(result, "raw_financial_metrics")[0] is None
    assert len(result["messages"]) == len(requested)
    assert [message.tool_call_id for message in result["messages"]] == [
        call["id"] for call in requested
    ]


@pytest.mark.asyncio
async def test_cancelled_acquisition_cancels_both_pending_tools():
    started = set()
    cancelled = set()
    ready = asyncio.Event()
    chosen = tools()
    for tool in chosen:

        async def stalled(args, name=tool.name):
            started.add(name)
            if len(started) == 2:
                ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add(name)

        tool.ainvoke.side_effect = stalled
    pending = asyncio.create_task(relay(chosen))
    try:
        await asyncio.wait_for(ready.wait(), timeout=2)
    finally:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert cancelled == {METRICS, QUAL}


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [False, True])
@pytest.mark.parametrize("size", [0, 21_000])
async def test_partial_provider_results_retain_useful_inspected_neighbors(order, size):
    valid = "<result><url>https://example.com</url><content>inspected fact</content></result>"
    error = '<result>{"error":"unavailable"}</result>'
    content = "STATUS: EVIDENCE_FOUND\n" + (
        valid + "x" * size + error if order else error + "x" * size + valid
    )
    _, result = await relay(tools(qualitative=content))
    assert result["artifact_statuses"][FIELD]["ok"]
    assert content in result[FIELD]
    assert "Partial provider failure" in result[FIELD]
    assert get_structured_ingress_payload(result, "raw_financial_metrics")[0] == PAYLOAD


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "<result><content>fact</content>",
        '<result>{"error":"unavailable"}</result>',
        'STATUS: EVIDENCE_FOUND\n<result>{"error":"unavailable"}</result>',
    ],
)
async def test_unusable_qualitative_acquisition_is_data_failure_not_contract_failure(
    content,
):
    _, result = await relay(tools(qualitative=content))
    assert not result["artifact_statuses"][FIELD]["ok"]
    assert result["artifact_statuses"][FIELD]["error_kind"] == "data_unavailable"
    assert get_structured_ingress_payload(result, "raw_financial_metrics")[0] == PAYLOAD


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ticker,requested",
    [
        ("0005.HK", "5:SEHK"),
        ("TEL2.B.ST", "TEL2-B.ST"),
        ("brk.b", "BRK-B"),
        ("TEST", " test "),
    ],
)
async def test_junior_guard_accepts_existing_listing_normalization(ticker, requested):
    chosen = tools()
    _, result = await relay(chosen, calls(ticker=requested), ticker=ticker)
    assert result["artifact_statuses"][FIELD]["ok"]
    assert all(tool.ainvoke.await_count == 1 for tool in chosen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ticker,requested",
    [
        ("0005.HK", "0005.T"),
        ("TEL2.B.ST", "TEL2.A.ST"),
        ("BRK.B", "BRK.A"),
        ("TEST", ""),
        ("TEST", None),
        ("TEST", {}),
    ],
)
async def test_junior_guard_blocks_distinct_or_malformed_identity_with_ledger_accounting(
    ticker, requested
):
    chosen = tools()
    _, result = await relay(chosen, calls(ticker=requested), ticker=ticker)
    assert not result["artifact_statuses"][FIELD]["ok"]
    assert all(tool.ainvoke.await_count == 0 for tool in chosen)
    assert "JUNIOR_TICKER_MISMATCH" in str(result["research_budgets"][JUNIOR])


@pytest.mark.asyncio
async def test_terminal_rejects_artifact_from_another_ticker_without_model_call():
    _, acquired = await relay(tools())
    invoke = AsyncMock()
    with patch("src.agents.runtime.invoke_with_rate_limit_handling", invoke):
        result = await create_analyst_node(MagicMock(), JUNIOR, tools(), FIELD)(
            {**acquired, "company_of_interest": "OTHER"}, {}
        )
    invoke.assert_not_awaited()
    assert not result["artifact_statuses"][FIELD]["ok"]
    assert "TEST" not in result[FIELD]


def test_full_inspected_classifier_rejects_malformed_result_storm_under_watchdog():
    code = """
from src.tooling.evidence_recorder import classify_evidence_value
for text in ('<result>' * 16000, '</result>' * 16000, '<result>' * 16000 + '</result>' * 16000):
    execution, evidence, reason, _ = classify_evidence_value('get_fundamental_analysis', text)
    assert evidence == 'INSUFFICIENT'
    assert reason == 'MALFORMED_RESULT_ENVELOPE'
"""
    subprocess.run(
        [sys.executable, "-c", code], timeout=5, check=True, capture_output=True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("quick_mode", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "matched_failure",
        "empty_failure",
        "stale",
        "wrong_type",
        "missing",
        "bad_status",
    ],
)
async def test_all_terminal_paths_close_without_replaying_or_changing_ingress(
    quick_mode, case
):
    from src.eval.prompt_checks import PromptCheckCollector
    from src.runtime_diagnostics import failure_artifact, success_artifact

    policy = graph_research_budget_policies(quick_mode=quick_mode)[JUNIOR]
    chosen = tools()
    selected, acquired = await relay(
        chosen,
        requested=calls((METRICS,)) if case == "matched_failure" else None,
        policy=policy,
    )
    state = {
        **selected,
        **acquired,
        "messages": [
            *selected["messages"],
            *acquired["messages"],
            AIMessage(name="news_analyst", content="", tool_calls=calls()),
        ],
    }
    if case == "empty_failure":
        state.update(
            failure_artifact(
                FIELD,
                "Acquisition capped",
                provider="code",
                error_kind="output_cap_exhausted",
            )
        )
    elif case == "stale":
        state[FIELD] = state[FIELD].replace("FOR TEST", "FOR OTHER")
    elif case == "wrong_type":
        state[FIELD] = {"untrusted": "wrong type"}
    elif case == "missing":
        state.pop(FIELD)
        state.pop("artifact_statuses")
    elif case == "bad_status":
        state["artifact_statuses"] = {FIELD: []}
    original_status = (state.get("artifact_statuses") or {}).get(FIELD)
    telemetry = state["research_budgets"][JUNIOR].copy()
    ingress = get_structured_ingress_payload(state, "raw_financial_metrics")
    collector = PromptCheckCollector()
    graph = StateGraph(AgentState)
    analyst = create_analyst_node(
        MagicMock(), JUNIOR, chosen, FIELD, research_budget_policy=policy
    )
    graph.add_node(
        "junior", collector.wrap_node("Junior Fundamentals Analyst", analyst)
    )
    tool_node = AsyncMock(
        side_effect=AssertionError("Terminal path replayed tool request")
    )
    graph.add_node("tools", tool_node)
    graph.add_node("barrier", FUNDAMENTALS_BARRIER.arrival_update)
    graph.add_node(
        "foreign", lambda state: success_artifact("foreign_language_report", "native")
    )
    graph.add_node("legal", lambda state: success_artifact("legal_report", "legal"))
    senior = AsyncMock(return_value=success_artifact("fundamentals_report", "senior"))
    graph.add_node("Fundamentals Analyst", senior)
    for node in ("junior", "foreign", "legal"):
        graph.add_edge(START, node)
    graph.add_conditional_edges(
        "junior", route_analyst_tools(JUNIOR), {"tools": "tools", "continue": "barrier"}
    )
    graph.add_edge("tools", "junior")
    for node in ("foreign", "legal"):
        graph.add_edge(node, "barrier")
    graph.add_conditional_edges(
        "barrier",
        fundamentals_sync_router,
        {"Fundamentals Analyst": "Fundamentals Analyst", "__end__": END},
    )
    graph.add_edge("Fundamentals Analyst", END)
    invoke = AsyncMock(side_effect=AssertionError("Terminal path invoked model"))
    executions = sum(tool.ainvoke.await_count for tool in chosen)
    with patch("src.agents.runtime.invoke_with_rate_limit_handling", invoke):
        result = await asyncio.wait_for(graph.compile().ainvoke(state), timeout=2)
    assert (
        not invoke.await_count and not tool_node.await_count and senior.await_count == 1
    )
    assert result["research_budgets"][JUNIOR] == telemetry
    assert sum(tool.ainvoke.await_count for tool in chosen) == executions
    assert get_structured_ingress_payload(result, "raw_financial_metrics") == ingress
    assert (
        collector.collected_outputs()[JUNIOR].artifact_status
        == result["artifact_statuses"][FIELD]
    )
    assert collector.collected_outputs()[JUNIOR].artifact_text == result[FIELD]
    closing = next(
        message
        for message in reversed(result["messages"])
        if getattr(message, "name", None) == JUNIOR
    )
    assert isinstance(closing, AIMessage) and not closing.tool_calls
    assert not result["artifact_statuses"][FIELD]["ok"]
    if case in {"matched_failure", "empty_failure"}:
        assert result["artifact_statuses"][FIELD] == original_status
    else:
        assert result["artifact_statuses"][FIELD]["error_kind"] == "data_unavailable"
        assert "FOR OTHER" not in result[FIELD]


@pytest.mark.asyncio
async def test_terminal_matching_is_isolated_between_concurrent_tickers():
    _, acquired = await relay(tools())
    analyst = create_analyst_node(
        MagicMock(), JUNIOR, tools(), FIELD, research_budget_policy=POLICY
    )
    valid, stale = await asyncio.gather(
        analyst({"company_of_interest": "TEST", **acquired}, {}),
        analyst({**acquired, "company_of_interest": "OTHER"}, {}),
    )
    assert (
        valid[FIELD] == acquired[FIELD]
        and valid["artifact_statuses"] == acquired["artifact_statuses"]
    )
    assert not stale["artifact_statuses"][FIELD]["ok"]
    assert valid["sender"] == stale["sender"] == JUNIOR
    assert not valid["messages"][0].tool_calls and not stale["messages"][0].tool_calls
