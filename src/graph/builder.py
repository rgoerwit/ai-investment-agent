from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import structlog
from langgraph.graph import END, StateGraph
from langgraph.types import RunnableConfig

from src.agents import AgentState
from src.eval import BaselineCaptureManager

from .components import build_graph_components
from .routing import (
    CONSULTANT_SKIP_SENTINEL,
    consultant_gate_router,
    dispatch_destinations,
    fan_out_to_analysts,
    fundamentals_sync_router,
    post_research_sync_router,
    should_continue_analyst,
    sync_check_router,
)

logger = structlog.get_logger(__name__)


def _reconcile_fundamentals_evidence(state: AgentState) -> dict[str, Any]:
    """Revalidate the original FLA response against all barrier-complete evidence."""
    from src.agents.foreign_language_evidence import normalize_foreign_language_evidence
    from src.agents.management_guidance import normalize_management_guidance_output
    from src.agents.message_utils import (
        evidence_record_to_tool_evidence,
        latest_agent_text,
    )
    from src.runtime_diagnostics import get_valid_artifact_content
    from src.runtime_services import get_current_evidence_records

    valid_report = get_valid_artifact_content(
        state,
        "foreign_language_report",
    )
    if not valid_report:
        return {}
    raw_report = (
        latest_agent_text(
            state.get("messages", []),
            "foreign_language_analyst",
        )
        or valid_report
    )
    guidance_evidence = state.get("management_guidance_evidence", "") or ""
    source_records = [
        record
        for record in get_current_evidence_records()
        if not record.blocked
        and record.agent_key in {"foreign_language_analyst", "legal_counsel"}
    ]
    guidance_normalized = normalize_management_guidance_output(
        raw_report,
        guidance_evidence,
        source_records,
    )
    records = [evidence_record_to_tool_evidence(record) for record in source_records]
    reconciled = normalize_foreign_language_evidence(
        guidance_normalized,
        [],
        ticker=state.get("company_of_interest", "UNKNOWN"),
        supplemental_evidence=guidance_evidence,
        additional_records=records,
    )
    if reconciled == state.get("foreign_language_report", ""):
        return {}
    logger.info(
        "fundamentals_evidence_reconciled",
        ticker=state.get("company_of_interest", "UNKNOWN"),
        evidence_records=len(records),
    )
    return {"foreign_language_report": reconciled}


def create_trading_graph(
    max_debate_rounds: int = 2,
    enable_memory: bool = True,
    recursion_limit: int = 100,
    ticker: str | None = None,
    cleanup_previous: bool = False,
    quick_mode: bool = False,
    strict_mode: bool = False,
    chart_format: str = "png",
    transparent_charts: bool = False,
    image_dir: Path | None = None,
    skip_charts: bool = False,
    baseline_capture: BaselineCaptureManager | None = None,
    node_observer: Any | None = None,
):
    """
    Create the multi-agent trading analysis graph with parallel analyst execution.
    """
    components = build_graph_components(
        max_debate_rounds=max_debate_rounds,
        enable_memory=enable_memory,
        ticker=ticker,
        cleanup_previous=cleanup_previous,
        quick_mode=quick_mode,
        strict_mode=strict_mode,
        chart_format=chart_format,
        transparent_charts=transparent_charts,
        image_dir=image_dir,
        skip_charts=skip_charts,
    )

    workflow = StateGraph(AgentState)

    async def dispatcher_node(state: AgentState, config: RunnableConfig):
        """Entry point that triggers parallel analyst streams."""
        return {}

    async def sync_check_node(state: AgentState, config: RunnableConfig):
        """Refresh canonical claims once all pre-debate inputs are complete."""
        from src.analysis_snapshot import refresh_analysis_snapshot
        from src.runtime_diagnostics import is_artifact_complete
        from src.runtime_services import get_current_evidence_records

        required = (
            "market_report",
            "sentiment_report",
            "news_report",
            "value_trap_report",
        )
        ready = all(is_artifact_complete(state, field) for field in required)
        ready = ready and state.get("pre_screening_result") in {"PASS", "REJECT"}
        if components.auditor_enabled:
            ready = ready and is_artifact_complete(state, "auditor_report")
        if not ready:
            return {}
        prior = state.get("analysis_snapshot") or {}
        return {
            "analysis_snapshot": refresh_analysis_snapshot(
                prior,
                state,
                get_current_evidence_records(),
                version=max(2, int(prior.get("version", 1)) + 1),
            )
        }

    async def fundamentals_sync_node(state: AgentState, config: RunnableConfig):
        """Reconcile sanitized FLA and Legal evidence before Senior runs."""
        from src.analysis_snapshot import build_pre_senior_snapshot
        from src.runtime_services import get_current_evidence_records

        reconciled = _reconcile_fundamentals_evidence(state)
        claim_state = {**state, **reconciled}
        return {
            **reconciled,
            "analysis_snapshot": build_pre_senior_snapshot(
                claim_state,
                get_current_evidence_records(),
                version=1,
            ),
        }

    async def debate_sync_r1_node(state: AgentState, config: RunnableConfig):
        """
        Synchronization point after Round 1 of Bull/Bear debate.
        Assembles R1 outputs so R2 agents can reference opponent arguments.
        """
        debate = state.get("investment_debate_state", {})
        bull_r1 = debate.get("bull_round1", "")
        bear_r1 = debate.get("bear_round1", "")

        history = f"""=== ROUND 1 ===

BULL RESEARCHER:
{bull_r1}

BEAR RESEARCHER:
{bear_r1}
"""
        logger.debug(
            "debate_sync_r1_complete",
            bull_r1_len=len(bull_r1),
            bear_r1_len=len(bear_r1),
        )

        return {
            "investment_debate_state": {
                "history": history,
                "bull_history": bull_r1,
                "bear_history": bear_r1,
                "current_round": 2,
                "count": 2,
            }
        }

    async def debate_sync_final_node(state: AgentState, config: RunnableConfig):
        """
        Final synchronization after debate completion.
        Assembles full debate history (R1 + R2) for Research Manager.
        """
        debate = state.get("investment_debate_state", {})
        bull_r1 = debate.get("bull_round1", "")
        bear_r1 = debate.get("bear_round1", "")
        bull_r2 = debate.get("bull_round2", "")
        bear_r2 = debate.get("bear_round2", "")

        if bull_r2 or bear_r2:
            history = f"""=== ROUND 1 ===

BULL RESEARCHER:
{bull_r1}

BEAR RESEARCHER:
{bear_r1}

=== ROUND 2 ===

BULL RESEARCHER (Rebuttal):
{bull_r2}

BEAR RESEARCHER (Rebuttal):
{bear_r2}
"""
            bull_history = f"{bull_r1}\n\n{bull_r2}"
            bear_history = f"{bear_r1}\n\n{bear_r2}"
            count = 4
        else:
            history = f"""=== ROUND 1 ===

BULL RESEARCHER:
{bull_r1}

BEAR RESEARCHER:
{bear_r1}
"""
            bull_history = bull_r1
            bear_history = bear_r1
            count = 2

        logger.debug(
            "debate_sync_final_complete",
            rounds=2 if bull_r2 else 1,
            total_arguments=count,
        )

        return {
            "investment_debate_state": {
                "history": history,
                "bull_history": bull_history,
                "bear_history": bear_history,
                "count": count,
            }
        }

    def maybe_wrap(node_name: str, node):
        wrapped = node
        if baseline_capture is not None:
            wrapped = baseline_capture.wrap_node(node_name, wrapped)
        if node_observer is not None:
            wrapped = node_observer.wrap_node(node_name, wrapped)
        return wrapped

    workflow.add_node("Dispatcher", maybe_wrap("Dispatcher", dispatcher_node))
    workflow.add_node("Sync Check", maybe_wrap("Sync Check", sync_check_node))
    workflow.add_node(
        "Fundamentals Sync Check",
        maybe_wrap("Fundamentals Sync Check", fundamentals_sync_node),
    )
    workflow.add_node(
        "Debate Sync R1", maybe_wrap("Debate Sync R1", debate_sync_r1_node)
    )
    workflow.add_node(
        "Debate Sync Final",
        maybe_wrap("Debate Sync Final", debate_sync_final_node),
    )

    for node_name, node in components.nodes.items():
        workflow.add_node(node_name, maybe_wrap(node_name, node))
    for node_name, node in components.tool_nodes.items():
        workflow.add_node(node_name, maybe_wrap(node_name, node))

    workflow.set_entry_point("Dispatcher")

    workflow.add_conditional_edges(
        "Dispatcher",
        fan_out_to_analysts,
        dispatch_destinations(include_auditor=components.auditor_enabled),
    )

    workflow.add_conditional_edges(
        "Market Analyst",
        should_continue_analyst,
        {"tools": "market_tools", "continue": "Sync Check"},
    )
    workflow.add_edge("market_tools", "Market Analyst")

    workflow.add_conditional_edges(
        "Sentiment Analyst",
        should_continue_analyst,
        {"tools": "sentiment_tools", "continue": "Sync Check"},
    )
    workflow.add_edge("sentiment_tools", "Sentiment Analyst")

    workflow.add_conditional_edges(
        "News Analyst",
        should_continue_analyst,
        {"tools": "news_tools", "continue": "Sync Check"},
    )
    workflow.add_edge("news_tools", "News Analyst")

    workflow.add_conditional_edges(
        "Junior Fundamentals Analyst",
        should_continue_analyst,
        {"tools": "junior_fund_tools", "continue": "Fundamentals Sync Check"},
    )
    workflow.add_edge("junior_fund_tools", "Junior Fundamentals Analyst")

    workflow.add_conditional_edges(
        "Foreign Language Analyst",
        should_continue_analyst,
        {"tools": "foreign_tools", "continue": "Fundamentals Sync Check"},
    )
    workflow.add_edge("foreign_tools", "Foreign Language Analyst")

    workflow.add_conditional_edges(
        "Legal Counsel",
        should_continue_analyst,
        {"tools": "legal_tools", "continue": "Fundamentals Sync Check"},
    )
    workflow.add_edge("legal_tools", "Legal Counsel")

    workflow.add_conditional_edges(
        "Value Trap Detector",
        should_continue_analyst,
        {"tools": "value_trap_tools", "continue": "Sync Check"},
    )
    workflow.add_edge("value_trap_tools", "Value Trap Detector")

    if components.auditor_enabled:
        workflow.add_conditional_edges(
            "Auditor",
            should_continue_analyst,
            {"tools": "auditor_tools", "continue": "Sync Check"},
        )
        workflow.add_edge("auditor_tools", "Auditor")

    workflow.add_conditional_edges(
        "Fundamentals Sync Check",
        fundamentals_sync_router,
        {"__end__": END, "Fundamentals Analyst": "Fundamentals Analyst"},
    )

    workflow.add_edge("Fundamentals Analyst", "Financial Validator")
    workflow.add_edge("Financial Validator", "Sync Check")
    workflow.add_edge("PM Fast-Fail", "Chart Generator")

    workflow.add_conditional_edges(
        "Sync Check",
        sync_check_router,
        ["__end__", "PM Fast-Fail", "Bull Researcher R1", "Bear Researcher R1"],
    )

    workflow.add_edge("Bull Researcher R1", "Debate Sync R1")
    workflow.add_edge("Bear Researcher R1", "Debate Sync R1")

    def debate_r1_router(
        state: AgentState, config: RunnableConfig
    ) -> Literal["Debate Sync Final"] | list[str]:
        """Route after Round 1: to Round 2 or directly to final sync (quick mode)."""
        context = config.get("configurable", {}).get("context")
        max_rounds = getattr(context, "max_debate_rounds", 2) if context else 2

        if max_rounds <= 1:
            logger.debug("debate_r1_router", decision="skip_r2_quick_mode")
            return "Debate Sync Final"

        logger.debug("debate_r1_router", decision="proceed_to_r2")
        return ["Bull Researcher R2", "Bear Researcher R2"]

    workflow.add_conditional_edges(
        "Debate Sync R1",
        debate_r1_router,
        ["Debate Sync Final", "Bull Researcher R2", "Bear Researcher R2"],
    )

    workflow.add_edge("Bull Researcher R2", "Debate Sync Final")
    workflow.add_edge("Bear Researcher R2", "Debate Sync Final")
    workflow.add_edge("Debate Sync Final", "Research Manager")

    async def post_research_sync_node(state: AgentState, config: RunnableConfig):
        return {}

    workflow.add_node("Post Research Sync", post_research_sync_node)

    workflow.add_edge("Research Manager", "Valuation Calculator")
    workflow.add_edge("Valuation Calculator", "Post Research Sync")
    if components.apac_specialist_enabled:
        workflow.add_edge("Research Manager", "APAC Regional Specialist")
        consultant_gate_source = "APAC Regional Specialist"
    else:
        consultant_gate_source = "Research Manager"

    if components.consultant_enabled:
        from src.runtime_diagnostics import success_artifact

        from .routing import should_invoke_consultant

        async def consultant_skip_node(state: AgentState, config: RunnableConfig):
            """Sentinel node activated when consultant_gate_router bypasses the
            Consultant LLM in quick-mode screening. Writes a transparent
            sentinel so downstream agents and the saved report still record
            the skip rather than seeing a missing field.
            """
            _, reason = should_invoke_consultant(state, config)
            return success_artifact(
                "consultant_review",
                CONSULTANT_SKIP_SENTINEL.format(reason=reason),
                provider="bypass",
            )

        workflow.add_node("Consultant Skip", consultant_skip_node)
        workflow.add_conditional_edges(
            consultant_gate_source,
            consultant_gate_router,
            ["Consultant", "Consultant Skip"],
        )
        workflow.add_edge("Consultant", "Post Research Sync")
        workflow.add_edge("Consultant Skip", "Post Research Sync")
    elif components.apac_specialist_enabled:
        workflow.add_edge("APAC Regional Specialist", "Post Research Sync")

    def post_research_router(
        state: AgentState, config: RunnableConfig
    ) -> Literal["Trader", "__end__"]:
        return post_research_sync_router(
            state,
            config,
            apac_required=components.apac_specialist_enabled,
            consultant_required=components.consultant_enabled,
        )

    workflow.add_conditional_edges(
        "Post Research Sync",
        post_research_router,
        ["__end__", "Trader"],
    )

    workflow.add_edge("Trader", "Risky Analyst")
    workflow.add_edge("Trader", "Safe Analyst")
    workflow.add_edge("Trader", "Neutral Analyst")
    workflow.add_edge("Risky Analyst", "Portfolio Manager")
    workflow.add_edge("Safe Analyst", "Portfolio Manager")
    workflow.add_edge("Neutral Analyst", "Portfolio Manager")
    workflow.add_edge("Portfolio Manager", "Chart Generator")
    workflow.add_edge("Chart Generator", END)

    logger.debug(
        "trading_graph_created",
        ticker=ticker,
        architecture="parallel",
        parallel_streams=[
            "Market",
            "Sentiment",
            "News",
            "Junior Fundamentals",
            "Foreign Language",
            "Legal Counsel",
            "Value Trap Detector",
        ],
        fundamentals_sync="Junior + Foreign + Legal → Senior → Validator",
        debate_parallel=[
            "Bull R1 || Bear R1",
            "Sync R1",
            "Bull R2 || Bear R2 (if max_rounds > 1)",
            "Sync Final",
        ],
        post_research_parallel=(
            "Valuation Calculator || APAC Regional Specialist || Consultant"
            if components.apac_specialist_enabled and components.consultant_enabled
            else "Valuation Calculator || APAC Regional Specialist"
            if components.apac_specialist_enabled
            else "Valuation Calculator || Consultant"
            if components.consultant_enabled
            else "Valuation Calculator"
        ),
        risk_team_parallel=["Risky Analyst", "Safe Analyst", "Neutral Analyst"],
        post_pm="Chart Generator (verdict-aligned visuals)",
        chart_generation=not (skip_charts or quick_mode),
        quick_mode=quick_mode,
    )

    return workflow.compile()
