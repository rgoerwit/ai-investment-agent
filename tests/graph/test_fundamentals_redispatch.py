"""Senior Fundamentals is released once per analysis.

Field incident (ORK.OL, UNVR.JK 2026-10-01; 9933.TW 2026-09-02): Junior
Fundamentals timed out. Its tool-loop router read the shared ``sender`` key,
which the exception path did not reset and which named a parallel analyst with
tool calls pending, so Junior was sent back into its tool node. The tool node
replayed Junior's already-answered calls, Junior's next transcript failed
validation (``ToolHistoryIntegrityError``), and Junior reached the Fundamentals
barrier a second time. The barrier released Senior again; the second Senior
transcript ended in its own reply (Gemini: 400 "Requests ending with a model
turn are not supported"), and that failure overwrote the valid report.

The graph tests drive real LangGraph supersteps with the production
``AgentState`` reducers, the production routers and the production barrier node.
Only the analyst, tool and Senior nodes are stubs; they reuse the production
transcript functions (``filter_messages_by_agent``, ``validate_tool_history``)
and mirror ``create_analyst_node``'s success and exception writes. The
combinations below vary tool-turn counts and fan-out order deterministically;
they are superstep orderings, not wall-clock schedules.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from src.agents.message_utils import (
    ToolHistoryIntegrityError,
    filter_messages_by_agent,
    validate_tool_history,
)
from src.agents.state import AgentState, add_counts
from src.graph.builder import fundamentals_barrier_node
from src.graph.routing import (
    FUNDAMENTALS_BARRIER,
    fundamentals_sync_router,
    route_analyst_tools,
    should_continue_analyst,
)
from src.runtime_diagnostics import (
    failure_artifact,
    is_artifact_valid,
    success_artifact,
)

JUNIOR = "junior_fundamentals_analyst"
FOREIGN = "foreign_language_analyst"
SENIOR = "fundamentals_analyst"
FIELDS = {
    JUNIOR: "raw_fundamentals_data",
    FOREIGN: "foreign_language_report",
    "news_analyst": "news_report",
    "market_analyst": "market_report",
}


def _ai(owner: str, content: str = "", tool_calls: list | None = None) -> AIMessage:
    message = AIMessage(content=content, tool_calls=tool_calls or [])
    message.name = owner
    return message


def _call(owner: str, n: int) -> dict:
    return {"name": "t", "args": {}, "id": f"{owner}-{n}", "type": "tool_call"}


def _legacy_router(owner: str) -> Callable:
    return should_continue_analyst


async def _prefix_barrier(state, config):
    """The production barrier node as it was before the release counter."""
    update = await fundamentals_barrier_node(state, config)
    update.pop(FUNDAMENTALS_BARRIER.count_field)
    return update


@dataclass
class Run:
    calls: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    final: dict = field(default_factory=dict)


def _run(
    *,
    router_for: Callable[[str], Callable],
    barrier: Callable,
    junior_tool_turns: int,
    news_tool_turns: int,
    foreign_tool_turns: int,
    reverse_fan_out: bool,
    exception_sets_sender: bool,
) -> Run:
    run = Run()
    turns = {
        JUNIOR: junior_tool_turns,
        FOREIGN: foreign_tool_turns,
        "news_analyst": news_tool_turns,
        "market_analyst": 1,
    }

    def failure(owner: str, exc: Exception) -> dict:
        # create_analyst_node's exception path.
        run.errors.append(f"{owner}:{type(exc).__name__}")
        result = failure_artifact(FIELDS[owner], exc)
        result["messages"] = [_ai(owner, f"Error: {exc}")]
        if exception_sets_sender:
            result["sender"] = owner
        return result

    def analyst(owner: str):
        async def node(state, config):
            run.calls[owner] += 1
            n = run.calls[owner]
            try:
                # prepare_messages_for_model's first two steps, unchanged.
                validate_tool_history(
                    filter_messages_by_agent(state["messages"], owner),
                    agent_key=owner,
                )
            except ToolHistoryIntegrityError as exc:
                return failure(owner, exc)
            if owner == JUNIOR and n > turns[owner]:
                return failure(owner, TimeoutError("hard timeout of 60.0s"))
            if n <= turns[owner]:
                return {
                    "sender": owner,
                    "messages": [_ai(owner, tool_calls=[_call(owner, n)])],
                }
            result = success_artifact(FIELDS[owner], f"{owner} report")
            result.update({"sender": owner, "messages": [_ai(owner, "report")]})
            return result

        return node

    def tool_node(owner: str):
        # create_agent_tool_node's selection: the owner's latest tool-calling
        # AIMessage, answered or not.
        async def node(state, config):
            run.calls[f"{owner}_tools"] += 1
            target = next(
                (
                    m
                    for m in reversed(state["messages"])
                    if isinstance(m, AIMessage) and m.tool_calls and m.name == owner
                ),
                None,
            )
            if target is None:
                return {"messages": []}
            return {
                "messages": [
                    ToolMessage(
                        content="ok",
                        tool_call_id=tc["id"],
                        additional_kwargs={"agent_key": owner},
                    )
                    for tc in target.tool_calls
                ]
            }

        return node

    async def legal(state, config):
        return success_artifact("legal_report", "legal")

    async def counted_barrier(state, config):
        run.calls["barrier"] += 1
        return await barrier(state, config)

    async def senior(state, config):
        run.calls["senior"] += 1
        transcript = filter_messages_by_agent(state["messages"], SENIOR)
        if transcript and isinstance(transcript[-1], AIMessage):
            # What Gemini returns for a request ending on a model turn.
            run.calls["senior_rejected"] += 1
            result = failure_artifact(
                "fundamentals_report",
                ValueError("400 INVALID_ARGUMENT: request ends with a model turn"),
            )
            result["messages"] = [_ai(SENIOR, "Error: 400")]
            return result
        result = success_artifact("fundamentals_report", "senior report")
        result["messages"] = [_ai(SENIOR, "senior report")]
        return result

    graph = StateGraph(AgentState)
    owners = [JUNIOR, "news_analyst", "market_analyst", FOREIGN]
    if reverse_fan_out:
        owners.reverse()
    for owner in owners:
        graph.add_node(owner, analyst(owner))
        graph.add_node(f"{owner}_tools", tool_node(owner))
        graph.add_edge(START, owner)
        graph.add_conditional_edges(
            owner,
            router_for(owner),
            {
                "tools": f"{owner}_tools",
                "continue": "barrier" if owner in (JUNIOR, FOREIGN) else END,
            },
        )
        graph.add_edge(f"{owner}_tools", owner)
    graph.add_node("legal", legal)
    graph.add_node("barrier", counted_barrier)
    graph.add_node("senior", senior)
    graph.add_edge(START, "legal")
    graph.add_edge("legal", "barrier")
    graph.add_conditional_edges(
        "barrier",
        fundamentals_sync_router,
        {"Fundamentals Analyst": "senior", "__end__": END},
    )
    graph.add_edge("senior", END)

    run.final = asyncio.run(
        graph.compile().ainvoke(
            {
                "messages": [HumanMessage(content="analyze")],
                "company_of_interest": "TEST",
            },
            {"recursion_limit": 100},
        )
    )
    return run


COMBINATIONS = list(
    itertools.product(range(1, 3), range(0, 5), range(0, 3), (False, True))
)


def _sweep(**config) -> list[tuple[tuple, Run]]:
    return [
        (
            combo,
            _run(
                junior_tool_turns=combo[0],
                news_tool_turns=combo[1],
                foreign_tool_turns=combo[2],
                reverse_fan_out=combo[3],
                **config,
            ),
        )
        for combo in COMBINATIONS
    ]


class TestFieldChain:
    def test_pre_fix_graph_reproduces_the_whole_incident(self):
        """Shared-sender routing, no sender reset, no release counter."""
        runs = _sweep(
            router_for=_legacy_router,
            barrier=_prefix_barrier,
            exception_sets_sender=False,
        )
        incidents = [
            combo
            for combo, run in runs
            if f"{JUNIOR}:ToolHistoryIntegrityError" in run.errors
            and run.calls["senior"] == 2
            and run.calls["senior_rejected"] == 1
            and not is_artifact_valid(run.final, "fundamentals_report")
        ]
        assert incidents, "pre-fix graph no longer reproduces the incident"

    def test_fixed_graph_releases_senior_once_in_every_combination(self):
        runs = _sweep(
            router_for=route_analyst_tools,
            barrier=fundamentals_barrier_node,
            exception_sets_sender=True,
        )
        for combo, run in runs:
            jt = combo[0]
            assert run.calls[JUNIOR] == jt + 1, combo  # tool turns, then the timeout
            assert run.errors == [f"{JUNIOR}:TimeoutError"], combo
            assert run.calls["senior"] == 1, combo
            assert run.calls["senior_rejected"] == 0, combo
            assert is_artifact_valid(run.final, "fundamentals_report"), combo
            assert run.final[FUNDAMENTALS_BARRIER.count_field] == 1, combo


class TestEachLayerAloneContainsTheIncident:
    """Owner-bound routing and the release counter are independent defenses."""

    def test_release_counter_contains_legacy_routing(self):
        runs = _sweep(
            router_for=_legacy_router,
            barrier=fundamentals_barrier_node,
            exception_sets_sender=False,
        )
        for combo, run in runs:
            assert run.calls["senior"] == 1, combo
            assert is_artifact_valid(run.final, "fundamentals_report"), combo
        # The counter actually engaged: some combinations re-reached the barrier.
        assert any(
            run.final[FUNDAMENTALS_BARRIER.count_field] > 1 for _, run in runs
        ), "no combination re-reaches the barrier; the guard is untested"

    def test_owner_routing_contains_a_counterless_barrier(self):
        runs = _sweep(
            router_for=route_analyst_tools,
            barrier=_prefix_barrier,
            exception_sets_sender=False,
        )
        for combo, run in runs:
            assert run.calls["senior"] == 1, combo
            assert f"{JUNIOR}:ToolHistoryIntegrityError" not in run.errors, combo


class TestRouteAnalystTools:
    def _pending(self, owner: str, n: int = 1) -> AIMessage:
        return _ai(owner, tool_calls=[_call(owner, n)])

    def test_routes_on_owner_not_on_sender(self):
        state = {
            "sender": "news_analyst",
            "messages": [
                _ai(JUNIOR, "Error: hard timeout"),
                self._pending("news_analyst"),
            ],
        }
        assert route_analyst_tools(JUNIOR)(state, {}) == "continue"
        assert route_analyst_tools("news_analyst")(state, {}) == "tools"
        # The legacy router follows sender and gets Junior wrong.
        assert should_continue_analyst(state, {}) == "tools"

    def test_owner_with_pending_calls_routes_to_tools(self):
        state = {"sender": "market_analyst", "messages": [self._pending(JUNIOR)]}
        assert route_analyst_tools(JUNIOR)(state, {}) == "tools"

    def test_newer_plain_reply_supersedes_an_older_tool_call(self):
        state = {"messages": [self._pending(JUNIOR), _ai(JUNIOR, "final")]}
        assert route_analyst_tools(JUNIOR)(state, {}) == "continue"

    def test_empty_history_continues_silently(self):
        with patch("src.graph.routing.logger") as logger:
            assert route_analyst_tools(JUNIOR)({"messages": []}, {}) == "continue"
        logger.warning.assert_not_called()

    def test_missing_owner_reply_continues_with_a_warning(self):
        state = {"messages": [HumanMessage(content="x"), self._pending("news_analyst")]}
        with patch("src.graph.routing.logger") as logger:
            assert route_analyst_tools(JUNIOR)(state, {}) == "continue"
        logger.warning.assert_called_once()
        assert logger.warning.call_args.args == (
            "analyst_routing_owner_response_missing",
        )
        assert logger.warning.call_args.kwargs["sender"] == JUNIOR

    def test_router_name_identifies_its_owner(self):
        assert route_analyst_tools(JUNIOR).__name__ == f"route_analyst_tools[{JUNIOR}]"


def _statuses(**fields: bool) -> dict:
    statuses: dict = {}
    for name, ok in fields.items():
        artifact = (
            success_artifact(name, "x")
            if ok
            else failure_artifact(name, TimeoutError("t"))
        )
        statuses.update(artifact["artifact_statuses"])
    return {"artifact_statuses": statuses}


ALL_IN = _statuses(
    raw_fundamentals_data=True, foreign_language_report=True, legal_report=True
)


class TestReleaseOnceBarrier:
    @pytest.mark.parametrize("count", [None, 1])
    def test_first_release_dispatches_senior(self, count):
        state = {**ALL_IN}
        if count is not None:
            state["fundamentals_release_count"] = count
        assert fundamentals_sync_router(state, {}) == "Fundamentals Analyst"

    @pytest.mark.parametrize("count", [2, 7])
    def test_rerelease_is_suppressed_and_logged(self, count):
        with patch("src.graph.routing.logger") as logger:
            result = fundamentals_sync_router(
                {**ALL_IN, "fundamentals_release_count": count}, {}
            )
        assert result == "__end__"
        logger.warning.assert_called_once_with(
            "barrier_redispatch_suppressed",
            barrier="fundamentals",
            release_count=count,
        )

    @pytest.mark.parametrize(
        "missing", ["raw_fundamentals_data", "foreign_language_report", "legal_report"]
    )
    def test_any_missing_input_holds_regardless_of_count(self, missing):
        present = {f: True for f in FUNDAMENTALS_BARRIER.inputs if f != missing}
        state = {**_statuses(**present), "fundamentals_release_count": 1}
        assert fundamentals_sync_router(state, {}) == "__end__"
        assert FUNDAMENTALS_BARRIER.arrival_update(state) == {
            "fundamentals_release_count": 0
        }

    def test_failed_input_counts_as_arrived(self):
        state = _statuses(
            raw_fundamentals_data=False, foreign_language_report=True, legal_report=True
        )
        assert FUNDAMENTALS_BARRIER.arrival_update(state) == {
            "fundamentals_release_count": 1
        }

    def test_production_node_writes_the_count(self):
        state = {**ALL_IN, "messages": [], "company_of_interest": "TEST"}
        update = asyncio.run(fundamentals_barrier_node(state, {}))
        assert update[FUNDAMENTALS_BARRIER.count_field] == 1

    def test_rearrival_counts_without_rolling_back_later_state(self):
        """A late arrival after release must only bump the count: rewriting the
        snapshot or the reconciled FLA report would replace what Senior and Sync
        Check wrote after the release."""
        advanced = {"version": 3, "claims": ["from sync check"]}
        state = {
            **ALL_IN,
            "messages": [],
            "company_of_interest": "TEST",
            "fundamentals_release_count": 1,
            "analysis_snapshot": advanced,
        }
        with patch("src.graph.builder._reconcile_fundamentals_evidence") as reconcile:
            update = asyncio.run(fundamentals_barrier_node(state, {}))
        assert update == {FUNDAMENTALS_BARRIER.count_field: 1}
        reconcile.assert_not_called()
        assert (
            fundamentals_sync_router({**state, FUNDAMENTALS_BARRIER.count_field: 2}, {})
            == "__end__"
        )

    def test_releasing_arrival_still_writes_the_pre_senior_snapshot(self):
        state = {**ALL_IN, "messages": [], "company_of_interest": "TEST"}
        update = asyncio.run(fundamentals_barrier_node(state, {}))
        assert update["analysis_snapshot"]["version"] == 1

    @pytest.mark.parametrize("count", [None, 0])
    def test_incomplete_or_first_arrival_is_not_a_rearrival(self, count):
        state = {**ALL_IN, "fundamentals_release_count": count}
        assert not FUNDAMENTALS_BARRIER.already_released(state)
        held = {
            **_statuses(raw_fundamentals_data=True),
            "fundamentals_release_count": 1,
        }
        assert not FUNDAMENTALS_BARRIER.already_released(held)

    @pytest.mark.parametrize(
        ("left", "right", "total"),
        [(None, None, 0), (None, 1, 1), (1, 1, 2), (2, 0, 2)],
    )
    def test_count_reducer_sums_and_tolerates_absence(self, left, right, total):
        assert add_counts(left, right) == total


def _compile_production_graph():
    import src.graph.components as components

    real_build = components.build_seat_model

    def build_or_stub(seat, **kwargs):
        model = real_build(seat, **kwargs)
        return model if model is not None else MagicMock(name=str(seat))

    with (
        patch("src.graph.components.create_memory_instances", return_value={}),
        patch("src.graph.components.cleanup_all_memories"),
        patch("src.graph.components._is_auditor_enabled", return_value=True),
        patch("src.graph.components.build_seat_model", side_effect=build_or_stub),
    ):
        from src.graph import create_trading_graph

        return create_trading_graph(
            ticker="TEST", max_debate_rounds=1, enable_memory=False
        )


@pytest.fixture(scope="module")
def graph():
    return _compile_production_graph()


class TestProductionWiring:
    @pytest.mark.parametrize(
        ("node", "owner"),
        [
            ("Market Analyst", "market_analyst"),
            ("Sentiment Analyst", "sentiment_analyst"),
            ("News Analyst", "news_analyst"),
            ("Junior Fundamentals Analyst", JUNIOR),
            ("Foreign Language Analyst", FOREIGN),
            ("Value Trap Detector", "value_trap_detector"),
            ("Auditor", "global_forensic_auditor"),
        ],
    )
    def test_every_tool_loop_edge_is_bound_to_its_owner(self, graph, node, owner):
        assert list(graph.builder.branches[node]) == [f"route_analyst_tools[{owner}]"]

    def test_no_edge_routes_by_shared_sender(self, graph):
        for node, branches in graph.builder.branches.items():
            for branch in branches.values():
                assert branch.path.func is not should_continue_analyst, node

    def test_barrier_node_is_the_counting_production_node(self, graph):
        node = graph.builder.nodes["Fundamentals Sync Check"].runnable
        assert node.afunc is fundamentals_barrier_node
        branch = graph.builder.branches["Fundamentals Sync Check"]
        assert [b.path.func for b in branch.values()] == [fundamentals_sync_router]
