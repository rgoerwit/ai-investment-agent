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
    dispatch_destinations,
    fan_out_to_analysts,
    fundamentals_sync_router,
    should_continue_analyst,
    sync_check_router,
)

logger = structlog.get_logger(__name__)


def create_trading_graph(
    max_debate_rounds: int = 2,
    max_risk_discuss_rounds: int = 1,
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
        """Synchronization barrier - routing logic in conditional edges."""
        return {}

    async def fundamentals_sync_node(state: AgentState, config: RunnableConfig):
        """Synchronization barrier for fundamentals data streams - routing logic in conditional edges."""
        return {}

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
        logger.info(
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

        logger.info(
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
            logger.info("debate_r1_router", decision="skip_r2_quick_mode")
            return "Debate Sync Final"

        logger.info("debate_r1_router", decision="proceed_to_r2")
        return ["Bull Researcher R2", "Bear Researcher R2"]

    workflow.add_conditional_edges(
        "Debate Sync R1",
        debate_r1_router,
        ["Debate Sync Final", "Bull Researcher R2", "Bear Researcher R2"],
    )

    workflow.add_edge("Bull Researcher R2", "Debate Sync Final")
    workflow.add_edge("Bear Researcher R2", "Debate Sync Final")
    workflow.add_edge("Debate Sync Final", "Research Manager")

    workflow.add_edge("Research Manager", "Valuation Calculator")
    workflow.add_edge("Valuation Calculator", "Trader")

    if components.consultant_enabled:
        workflow.add_edge("Research Manager", "Consultant")
        workflow.add_edge("Consultant", "Trader")

    workflow.add_edge("Trader", "Risky Analyst")
    workflow.add_edge("Trader", "Safe Analyst")
    workflow.add_edge("Trader", "Neutral Analyst")
    workflow.add_edge("Risky Analyst", "Portfolio Manager")
    workflow.add_edge("Safe Analyst", "Portfolio Manager")
    workflow.add_edge("Neutral Analyst", "Portfolio Manager")
    workflow.add_edge("Portfolio Manager", "Chart Generator")
    workflow.add_edge("Chart Generator", END)

    logger.info(
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
            "Valuation Calculator || Consultant"
            if components.consultant_enabled
            else "Valuation Calculator"
        ),
        risk_team_parallel=["Risky Analyst", "Safe Analyst", "Neutral Analyst"],
        post_pm="Chart Generator (verdict-aligned visuals)",
        chart_generation=not (skip_charts or quick_mode),
        quick_mode=quick_mode,
    )

    return workflow.compile()
