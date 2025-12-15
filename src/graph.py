"""
Multi-Agent Trading System Graph
Updated for Gemini 3.0 Routing and LangGraph 1.x compatibility.
FIXED: Tool routing now tracks which agent called the tool via sender field.
FIXED: Added ticker logging to track contamination issues.
UPDATED: Added ticker-specific memory isolation to prevent cross-contamination.
"""

from typing import Literal, Dict, Optional
from dataclasses import dataclass
import structlog

from langgraph.graph import StateGraph, END
from langgraph.types import RunnableConfig
# Modern ToolNode import for LangGraph 1.x
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage

from src.agents import (
    AgentState, create_analyst_node, create_researcher_node,
    create_research_manager_node, create_trader_node,
    create_risk_debater_node, create_portfolio_manager_node,
    create_state_cleaner_node, create_financial_health_validator_node,
    create_consultant_node
)
from src.llms import create_quick_thinking_llm, create_deep_thinking_llm, get_consultant_llm
from src.toolkit import toolkit
from src.token_tracker import TokenTrackingCallback, get_tracker
from src.memory import (
    create_memory_instances, cleanup_all_memories, FinancialSituationMemory,
    sanitize_ticker_for_collection
)

logger = structlog.get_logger(__name__)

@dataclass
class TradingContext:
    """
    Context object passed to graph nodes via configuration.
    Includes parameters that control graph execution flow.
    
    UPDATED: Added ticker-specific memory management.
    """
    ticker: str
    trade_date: str
    quick_mode: bool = False
    enable_memory: bool = True
    max_debate_rounds: int = 2
    max_risk_rounds: int = 1
    # NEW: Ticker-specific memories to prevent cross-contamination
    ticker_memories: Optional[Dict[str, any]] = None
    # NEW: Whether to cleanup previous ticker memories
    cleanup_previous_memories: bool = True

def should_continue_analyst(state: AgentState, config: RunnableConfig) -> Literal["tools", "continue"]:
    """
    Determine if analyst should call tools or continue to next node.
    
    Args:
        state: Current agent state
        config: Runtime configuration
        
    Returns:
        "tools" if agent has pending tool calls, "continue" otherwise
    """
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
        return "tools"
    return "continue"

def route_tools(state: AgentState) -> str:
    """
    Route back to the agent that called the tool.
    Uses the 'sender' field from the state to determine which agent to return to.

    Args:
        state: Current agent state with sender information

    Returns:
        Name of the node to return to after tool execution
    """
    sender = state.get("sender", "")

    # Map internal agent keys to Node Names
    agent_map = {
        "market_analyst": "Market Analyst",
        "sentiment_analyst": "Social Analyst",
        "news_analyst": "News Analyst",
        "fundamentals_analyst": "Fundamentals Analyst"
    }

    node_name = agent_map.get(sender, "Market Analyst")

    logger.debug(
        "tool_routing",
        sender=sender,
        routing_to=node_name
    )

    return node_name


def validator_router(state: AgentState, config: RunnableConfig) -> Literal["Portfolio Manager", "Bull Researcher"]:
    """
    Route based on pre-screening red-flag validation results.

    If critical red flags detected (pre_screening_result == "REJECT"):
    - Skip bull/bear debate (saves tokens, time)
    - Route directly to Portfolio Manager for final SELL decision

    If no critical red flags (pre_screening_result == "PASS"):
    - Continue to normal flow: Bull Researcher → debate → Portfolio Manager

    This routing implements the "fast-fail" pattern for extreme financial risks
    that would result in automatic SELL regardless of debate outcome.

    Args:
        state: Current agent state with pre_screening_result populated
        config: Runtime configuration (not currently used)

    Returns:
        "Portfolio Manager" if REJECT, "Bull Researcher" if PASS

    Example red-flag scenarios that trigger REJECT:
    - D/E ratio > 500% (leverage bomb)
    - Positive income but negative FCF >2x income (earnings quality)
    - Interest coverage <2.0x with D/E >100% (refinancing risk)
    """
    pre_screening_result = state.get('pre_screening_result', 'PASS')

    if pre_screening_result == 'REJECT':
        logger.info(
            "validator_routing_to_pm",
            ticker=state.get('company_of_interest', 'UNKNOWN'),
            message="Red flags detected - skipping debate, routing to Portfolio Manager"
        )
        return "Portfolio Manager"

    # Normal flow - proceed to debate
    return "Bull Researcher"

def create_trading_graph(
    max_debate_rounds: int = 2,
    max_risk_discuss_rounds: int = 1,
    enable_memory: bool = True,
    recursion_limit: int = 100,
    ticker: Optional[str] = None,
    cleanup_previous: bool = False,
    quick_mode: bool = False
):
    """
    Create the multi-agent trading analysis graph with ticker-specific memory isolation.

    UPDATED: Now supports ticker-specific memories to prevent cross-contamination.

    Args:
        ticker: Stock ticker symbol (e.g., "0005.HK", "AAPL"). If provided, creates
                ticker-specific memories. If None, uses legacy global memories (NOT recommended).
        cleanup_previous: If True, deletes all previous memories before creating new ones.
                         Use this to ensure fresh analysis without contamination.
        max_debate_rounds: Maximum rounds of bull/bear debate (default: 2)
        max_risk_discuss_rounds: Maximum rounds of risk discussion (default: 1)
        enable_memory: Whether to enable agent memory (default: True)
        recursion_limit: Maximum recursion depth for graph execution (default: 100)
        quick_mode: If True, use faster/cheaper models for consultant LLM (default: False)

    Returns:
        Compiled LangGraph StateGraph ready for execution

    Example:
        # Recommended: Ticker-specific memory with cleanup
        graph = create_trading_graph(
            ticker="0005.HK",
            cleanup_previous=True,
            max_debate_rounds=2,
            quick_mode=False
        )

        # Legacy: Global memory (may cause contamination)
        graph = create_trading_graph(max_debate_rounds=2)
    """
    
    # Determine which memories to use
    if ticker and enable_memory:
        # RECOMMENDED: Create ticker-specific memories
        if cleanup_previous:
            logger.info(
                "cleaning_previous_memories",
                ticker=ticker,
                message="Deleting previous memory collections for THIS ticker to prevent contamination"
            )
            # UPDATED: Pass ticker to scoped cleanup
            cleanup_all_memories(days=0, ticker=ticker)
        
        logger.info(
            "creating_ticker_memories",
            ticker=ticker,
            message="Creating ticker-specific memory collections"
        )
        memories = create_memory_instances(ticker)

        # Extract specific memories for each agent
        # CRITICAL: Must use same sanitization as create_memory_instances()
        safe_ticker = sanitize_ticker_for_collection(ticker)
        bull_memory = memories.get(f"{safe_ticker}_bull_memory")
        bear_memory = memories.get(f"{safe_ticker}_bear_memory")
        invest_judge_memory = memories.get(f"{safe_ticker}_invest_judge_memory")
        trader_memory = memories.get(f"{safe_ticker}_trader_memory")
        risk_manager_memory = memories.get(f"{safe_ticker}_risk_manager_memory")

        # Verify all memories were successfully created
        if not all([bull_memory, bear_memory, invest_judge_memory, trader_memory, risk_manager_memory]):
            missing = []
            if not bull_memory: missing.append("bull_memory")
            if not bear_memory: missing.append("bear_memory")
            if not invest_judge_memory: missing.append("invest_judge_memory")
            if not trader_memory: missing.append("trader_memory")
            if not risk_manager_memory: missing.append("risk_manager_memory")
            raise ValueError(
                f"Failed to create memory instances for ticker {ticker}. "
                f"Missing: {', '.join(missing)}. "
                f"Available keys: {list(memories.keys())}"
            )

        logger.info(
            "ticker_memories_ready",
            ticker=ticker,
            bull_available=bull_memory.available,
            bear_available=bear_memory.available,
            judge_available=invest_judge_memory.available,
            trader_available=trader_memory.available,
            risk_available=risk_manager_memory.available
        )
    else:
        # LEGACY: Use global memories (will cause cross-contamination!)
        logger.warning(
            "using_legacy_memories",
            ticker=ticker,
            enable_memory=enable_memory,
            message="Using legacy global memories. This WILL cause cross-ticker contamination! "
                    "Use ticker-specific memories by passing ticker parameter."
        )
        # Manually create legacy instances since they are no longer global
        bull_memory = FinancialSituationMemory("legacy_bull_memory")
        bear_memory = FinancialSituationMemory("legacy_bear_memory")
        invest_judge_memory = FinancialSituationMemory("legacy_invest_judge_memory")
        trader_memory = FinancialSituationMemory("legacy_trader_memory")
        risk_manager_memory = FinancialSituationMemory("legacy_risk_manager_memory")
    
    # Log graph creation
    logger.info(
        "creating_trading_graph",
        ticker=ticker,
        max_debate_rounds=max_debate_rounds,
        enable_memory=enable_memory,
        using_ticker_specific_memory=ticker is not None
    )

    # Create LLMs with token tracking callbacks
    tracker = get_tracker()

    market_llm = create_quick_thinking_llm(callbacks=[TokenTrackingCallback("Market Analyst", tracker)])
    social_llm = create_quick_thinking_llm(callbacks=[TokenTrackingCallback("Social Analyst", tracker)])
    news_llm = create_quick_thinking_llm(callbacks=[TokenTrackingCallback("News Analyst", tracker)])
    fund_llm = create_quick_thinking_llm(callbacks=[TokenTrackingCallback("Fundamentals Analyst", tracker)])
    bull_llm = create_quick_thinking_llm(callbacks=[TokenTrackingCallback("Bull Researcher", tracker)])
    bear_llm = create_quick_thinking_llm(callbacks=[TokenTrackingCallback("Bear Researcher", tracker)])

    # Assign LLMs to thinking agents based on quick_mode.
    if quick_mode:
        # In quick mode, EVERYONE uses the quick LLM.
        logger.info("Quick mode ON: Using QUICK_MODEL for all agents, including thinking agents.")
        res_mgr_llm = create_quick_thinking_llm(callbacks=[TokenTrackingCallback("Research Manager", tracker)])
        pm_llm = create_quick_thinking_llm(callbacks=[TokenTrackingCallback("Portfolio Manager", tracker)])
    else:
        # In normal (deep) mode, thinking agents use the deep LLM.
        res_mgr_llm = create_deep_thinking_llm(callbacks=[TokenTrackingCallback("Research Manager", tracker)])
        pm_llm = create_deep_thinking_llm(callbacks=[TokenTrackingCallback("Portfolio Manager", tracker)])

    trader_llm = create_quick_thinking_llm(callbacks=[TokenTrackingCallback("Trader", tracker)])
    risky_llm = create_quick_thinking_llm(callbacks=[TokenTrackingCallback("Risky Analyst", tracker)])
    safe_llm = create_quick_thinking_llm(callbacks=[TokenTrackingCallback("Safe Analyst", tracker)])
    neutral_llm = create_quick_thinking_llm(callbacks=[TokenTrackingCallback("Neutral Analyst", tracker)])

    # Consultant LLM (OpenAI - optional, may be None if disabled/unavailable)
    consultant_llm = get_consultant_llm(callbacks=[TokenTrackingCallback("Consultant", tracker)], quick_mode=quick_mode)

    # Nodes
    market = create_analyst_node(market_llm, "market_analyst", toolkit.get_technical_tools(), "market_report")
    social = create_analyst_node(social_llm, "sentiment_analyst", toolkit.get_sentiment_tools(), "sentiment_report")
    news = create_analyst_node(news_llm, "news_analyst", toolkit.get_news_tools(), "news_report")
    fund = create_analyst_node(fund_llm, "fundamentals_analyst", toolkit.get_fundamental_tools(), "fundamentals_report")

    cleaner = create_state_cleaner_node()
    # Standard ToolNode initialized with all tools
    tool_node = ToolNode(toolkit.get_all_tools())

    # Red-flag pre-screening validator (runs after fundamentals, before debate)
    validator = create_financial_health_validator_node()

    # Research & Execution Nodes (now using ticker-specific or legacy memories)
    bull = create_researcher_node(bull_llm, bull_memory, "bull_researcher")
    bear = create_researcher_node(bear_llm, bear_memory, "bear_researcher")
    res_mgr = create_research_manager_node(res_mgr_llm, invest_judge_memory)
    trader = create_trader_node(trader_llm, trader_memory)

    # Risk Nodes
    risky = create_risk_debater_node(risky_llm, "risky_analyst")
    safe = create_risk_debater_node(safe_llm, "safe_analyst")
    neutral = create_risk_debater_node(neutral_llm, "neutral_analyst")
    pm = create_portfolio_manager_node(pm_llm, risk_manager_memory)

    # Consultant Node (optional - only if consultant_llm is available)
    consultant = None
    if consultant_llm is not None:
        consultant = create_consultant_node(consultant_llm, "consultant")
        logger.info(
            "consultant_node_enabled",
            ticker=ticker,
            message="External consultant (OpenAI) will cross-validate Gemini analysis"
        )
    else:
        logger.info(
            "consultant_node_disabled",
            ticker=ticker,
            message="Consultant node skipped (OpenAI API key not configured or disabled)"
        )

    workflow = StateGraph(AgentState)
    
    workflow.add_node("Market Analyst", market)
    workflow.add_node("Social Analyst", social)
    workflow.add_node("News Analyst", news)
    workflow.add_node("Fundamentals Analyst", fund)
    workflow.add_node("tools", tool_node)
    workflow.add_node("Cleaner", cleaner)

    # Add red-flag validator (pre-screening layer)
    workflow.add_node("Financial Validator", validator)

    # Add research and risk nodes
    workflow.add_node("Bull Researcher", bull)
    workflow.add_node("Bear Researcher", bear)
    workflow.add_node("Research Manager", res_mgr)

    # Add consultant node (optional - only if OpenAI API key configured)
    if consultant is not None:
        workflow.add_node("Consultant", consultant)

    workflow.add_node("Trader", trader)
    workflow.add_node("Risky Analyst", risky)
    workflow.add_node("Safe Analyst", safe)
    workflow.add_node("Neutral Analyst", neutral)
    workflow.add_node("Portfolio Manager", pm)

    # Flow
    workflow.set_entry_point("Market Analyst")
    
    # 1. Market Flow
    workflow.add_conditional_edges("Market Analyst", should_continue_analyst, {"tools": "tools", "continue": "Cleaner"})
    
    # 2. Social Flow (via cleaner nodes to reset history)
    workflow.add_node("Clean1", cleaner)
    workflow.add_edge("Cleaner", "Clean1")
    workflow.add_edge("Clean1", "Social Analyst")
    
    workflow.add_conditional_edges("Social Analyst", should_continue_analyst, {"tools": "tools", "continue": "News Analyst"})
    workflow.add_conditional_edges("News Analyst", should_continue_analyst, {"tools": "tools", "continue": "Fundamentals Analyst"})
    workflow.add_conditional_edges("Fundamentals Analyst", should_continue_analyst, {"tools": "tools", "continue": "Financial Validator"})

    # Tool Return Logic
    workflow.add_conditional_edges("tools", route_tools, {
        "Market Analyst": "Market Analyst",
        "Social Analyst": "Social Analyst",
        "News Analyst": "News Analyst",
        "Fundamentals Analyst": "Fundamentals Analyst"
    })

    # Validator Routing (Red-Flag Detection)
    # - If REJECT: Skip debate, go straight to Portfolio Manager
    # - If PASS: Continue to normal debate flow
    workflow.add_conditional_edges("Financial Validator", validator_router, {
        "Portfolio Manager": "Portfolio Manager",
        "Bull Researcher": "Bull Researcher"
    })

    # Debate Flow (Research Manager → Bull/Bear → Consultant → Trader)
    def debate_router(state: AgentState, config: RunnableConfig):
        """
        Route debate flow between Bull and Bear researchers.
        After debate converges, routes to Consultant (if enabled) or Trader (if disabled).
        """
        # Retrieve configuration from context
        context = config.get("configurable", {}).get("context")
        # Default to 2 rounds if context is missing or field is None
        max_rounds = getattr(context, "max_debate_rounds", 2) if context else 2
        
        # Total turns = rounds * 2 (Bull + Bear per round)
        limit = max_rounds * 2
        
        count = state.get("investment_debate_state", {}).get("count", 0)
        
        if count >= limit:
            return "Research Manager"
            
        # Alternating flow
        return "Bear Researcher" if count % 2 != 0 else "Bull Researcher"

    workflow.add_conditional_edges("Bull Researcher", debate_router, ["Bear Researcher", "Research Manager"])
    workflow.add_conditional_edges("Bear Researcher", debate_router, ["Bull Researcher", "Research Manager"])

    # Consultant routing: Research Manager → Consultant → Trader
    # If consultant node not available, route directly to Trader
    if consultant is not None:
        workflow.add_edge("Research Manager", "Consultant")
        workflow.add_edge("Consultant", "Trader")
    else:
        workflow.add_edge("Research Manager", "Trader")

    workflow.add_edge("Trader", "Risky Analyst")
    
    # Risk Flow
    workflow.add_edge("Risky Analyst", "Safe Analyst")
    workflow.add_edge("Safe Analyst", "Neutral Analyst")
    workflow.add_edge("Neutral Analyst", "Portfolio Manager")
    workflow.add_edge("Portfolio Manager", END)

    logger.info(
        "trading_graph_created",
        ticker=ticker,
        using_ticker_specific_memory=ticker is not None
    )
    return workflow.compile()