"""
Multi-Agent Trading System Graph
REFACTORED: Parallel analyst execution with synchronization barrier.

Architecture:
1. Dispatcher fans out to 4 parallel analyst streams
2. Market, Sentiment, News analysts run independently
3. Junior Fund → Senior Fund → Validator runs as sequential chain
4. Sync Check waits for all 4 streams, then routes to Research Manager or Portfolio Manager
5. Bull/Bear debate → Consultant → Trader → Risk Team → Portfolio Manager
"""

from typing import Literal, Dict, Optional, List
from dataclasses import dataclass
import structlog

from langgraph.graph import StateGraph, END
from langgraph.types import RunnableConfig
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage

from src.agents import (
    AgentState, create_analyst_node, create_researcher_node,
    create_research_manager_node, create_trader_node,
    create_risk_debater_node, create_portfolio_manager_node,
    create_financial_health_validator_node,
    create_consultant_node
)
from src.llms import (
    create_quick_thinking_llm,
    create_deep_thinking_llm,
    get_consultant_llm,
    is_gemini_v3_or_greater,
)
from src.config import config
from src.toolkit import toolkit
from src.token_tracker import TokenTrackingCallback, get_tracker
from src.memory import (
    create_memory_instances, cleanup_all_memories, FinancialSituationMemory,
    sanitize_ticker_for_collection
)

logger = structlog.get_logger(__name__)


@dataclass
class TradingContext:
    """Context object passed to graph nodes via configuration."""
    ticker: str
    trade_date: str
    quick_mode: bool = False
    enable_memory: bool = True
    max_debate_rounds: int = 2
    max_risk_rounds: int = 1
    ticker_memories: Optional[Dict[str, any]] = None
    cleanup_previous_memories: bool = True


# --- Routing Functions ---

def should_continue_analyst(
    state: AgentState, config: RunnableConfig
) -> Literal["tools", "continue"]:
    """
    Determine if analyst should call tools or continue to next node.
    Returns "tools" if agent has pending tool calls, "continue" otherwise.
    """
    messages = state.get("messages", [])
    sender = state.get("sender", "unknown")
    has_tool_calls = messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls

    result = "tools" if has_tool_calls else "continue"

    logger.info(
        "analyst_routing",
        sender=sender,
        has_tool_calls=has_tool_calls,
        result=result
    )

    return result


def route_tools(state: AgentState) -> str:
    """
    Route back to the agent that called the tool.
    Uses the 'sender' field from the state.
    """
    sender = state.get("sender", "")

    agent_map = {
        "market_analyst": "Market Analyst",
        "sentiment_analyst": "Sentiment Analyst",
        "news_analyst": "News Analyst",
        "junior_fundamentals_analyst": "Junior Fundamentals Analyst",
    }

    node_name = agent_map.get(sender, "Market Analyst")

    logger.debug(
        "tool_routing",
        sender=sender,
        routing_to=node_name
    )

    return node_name


def create_agent_tool_node(tools: List, agent_key: str):
    """
    Create a tool execution node that only processes tool_calls from a specific agent.

    CRITICAL FIX: In parallel execution, multiple agents add AIMessages to the shared
    messages list. The standard ToolNode looks at the "latest AIMessage" which might
    be from a different agent. This wrapper filters messages to find the AIMessage
    that contains tool_calls for THIS agent's tools.

    Args:
        tools: List of tools this node can execute
        agent_key: The sender key for the agent (e.g., "market_analyst")

    Returns:
        An async function that executes tools for the specific agent
    """
    tool_node = ToolNode(tools)
    tool_names = {tool.name for tool in tools}

    async def agent_tool_node(state: AgentState, config: RunnableConfig) -> Dict:
        """Execute tools for a specific agent by filtering messages."""
        messages = state.get("messages", [])

        # Find the AIMessage from THIS agent (has tool_calls for our tools)
        target_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Check if any tool_call is for one of our tools
                msg_tool_names = {tc.get('name', tc.get('function', {}).get('name', ''))
                                 for tc in msg.tool_calls}
                if msg_tool_names & tool_names:  # Intersection - has at least one of our tools
                    target_message = msg
                    break

        if target_message is None:
            logger.warning(
                "agent_tool_node_no_matching_message",
                agent_key=agent_key,
                tool_names=list(tool_names),
                message="No AIMessage found with tool_calls for this agent's tools"
            )
            return {"messages": []}

        # Create a filtered state with only the target message
        # This ensures ToolNode processes the correct tool_calls
        filtered_messages = [target_message]

        logger.debug(
            "agent_tool_node_executing",
            agent_key=agent_key,
            tool_calls=[tc.get('name') for tc in target_message.tool_calls],
            total_messages=len(messages)
        )

        # Execute the tools using the filtered messages
        result = await tool_node.ainvoke({"messages": filtered_messages}, config)

        # Return the tool messages
        return result

    return agent_tool_node


def fan_out_to_analysts(
    state: AgentState, config: RunnableConfig
) -> List[str]:
    """
    Fan-out router that triggers all 4 analyst streams in parallel.
    Returns a list of destinations for parallel execution.
    """
    return [
        "Market Analyst",
        "Sentiment Analyst",
        "News Analyst",
        "Junior Fundamentals Analyst"
    ]


def sync_check_router(
    state: AgentState, config: RunnableConfig
) -> Literal["Research Manager", "Portfolio Manager", "__end__"]:
    """
    Synchronization barrier for parallel analyst streams (fan-in pattern).

    All 4 parallel branches converge here. Each branch calls sync_check
    when it completes. The router checks if ALL required reports are present:
    - market_report (from Market Analyst)
    - sentiment_report (from Sentiment Analyst)
    - news_report (from News Analyst)
    - pre_screening_result (from Validator in Fundamentals chain)

    Routing behavior:
    - If NOT all reports present: Return __end__ to terminate THIS branch
      (other branches continue running and will hit sync_check later)
    - If all present AND pre_screening=REJECT: Route to Portfolio Manager (fast-fail)
    - If all present AND pre_screening=PASS: Route to Bull Researcher (start debate)

    The LAST branch to complete will see all reports and proceed to the next phase.
    """
    market_done = bool(state.get("market_report"))
    sentiment_done = bool(state.get("sentiment_report"))
    news_done = bool(state.get("news_report"))

    # Validator completion check
    pre_screening = state.get("pre_screening_result")
    validator_done = pre_screening in ["PASS", "REJECT"]

    all_done = all([market_done, sentiment_done, news_done, validator_done])

    logger.info(
        "sync_check_status",
        market_done=market_done,
        sentiment_done=sentiment_done,
        news_done=news_done,
        validator_done=validator_done,
        pre_screening=pre_screening,
        all_done=all_done
    )

    if not all_done:
        # This branch finished early - terminate it
        # Other branches continue executing
        return "__end__"

    # All streams complete - route based on validator result
    if pre_screening == "REJECT":
        logger.info(
            "sync_routing_to_pm",
            message="Red flags detected - skipping debate, routing to Portfolio Manager"
        )
        return "Portfolio Manager"

    logger.info(
        "sync_routing_to_research_manager",
        message="All analysts complete - proceeding to Research Manager"
    )
    return "Research Manager"


def debate_router(state: AgentState, config: RunnableConfig):
    """
    Route debate flow between Bull and Bear researchers.
    After debate converges, routes to Research Manager.
    """
    context = config.get("configurable", {}).get("context")
    max_rounds = getattr(context, "max_debate_rounds", 2) if context else 2
    limit = max_rounds * 2  # Bull + Bear per round

    count = state.get("investment_debate_state", {}).get("count", 0)

    if count >= limit:
        return "Research Manager"

    # Alternating flow
    return "Bear Researcher" if count % 2 != 0 else "Bull Researcher"


# --- Graph Creation ---

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
    Create the multi-agent trading analysis graph with parallel analyst execution.

    Architecture:
    1. Dispatcher fans out to 4 parallel analyst streams
    2. Each stream processes independently with tool calling
    3. Sync Check waits for all streams to complete
    4. Routes to Research Manager (PASS) or Portfolio Manager (REJECT)
    5. Bull/Bear debate → Consultant → Trader → Risk → Portfolio Manager

    Args:
        ticker: Stock ticker symbol (e.g., "0005.HK")
        cleanup_previous: If True, deletes previous memories
        max_debate_rounds: Maximum rounds of bull/bear debate
        max_risk_discuss_rounds: Maximum rounds of risk discussion
        enable_memory: Whether to enable agent memory
        recursion_limit: Max recursion depth
        quick_mode: If True, use faster/cheaper models

    Returns:
        Compiled LangGraph StateGraph
    """

    # --- Memory Setup ---
    if ticker and enable_memory:
        if cleanup_previous:
            logger.info("cleaning_previous_memories", ticker=ticker)
            cleanup_all_memories(days=0, ticker=ticker)

        logger.info("creating_ticker_memories", ticker=ticker)
        memories = create_memory_instances(ticker)

        safe_ticker = sanitize_ticker_for_collection(ticker)
        bull_memory = memories.get(f"{safe_ticker}_bull_memory")
        bear_memory = memories.get(f"{safe_ticker}_bear_memory")
        invest_judge_memory = memories.get(f"{safe_ticker}_invest_judge_memory")
        trader_memory = memories.get(f"{safe_ticker}_trader_memory")
        risk_manager_memory = memories.get(f"{safe_ticker}_risk_manager_memory")

        all_memories = [
            bull_memory, bear_memory, invest_judge_memory,
            trader_memory, risk_manager_memory
        ]
        if not all(all_memories):
            missing = []
            if not bull_memory: missing.append("bull_memory")
            if not bear_memory: missing.append("bear_memory")
            if not invest_judge_memory: missing.append("invest_judge_memory")
            if not trader_memory: missing.append("trader_memory")
            if not risk_manager_memory: missing.append("risk_manager_memory")
            raise ValueError(
                f"Failed to create memory instances for {ticker}. Missing: {', '.join(missing)}"
            )

        logger.info(
            "ticker_memories_ready",
            ticker=ticker,
            bull_available=bull_memory.available,
            bear_available=bear_memory.available
        )
    else:
        logger.warning("using_legacy_memories", ticker=ticker)
        bull_memory = FinancialSituationMemory("legacy_bull_memory")
        bear_memory = FinancialSituationMemory("legacy_bear_memory")
        invest_judge_memory = FinancialSituationMemory("legacy_invest_judge_memory")
        trader_memory = FinancialSituationMemory("legacy_trader_memory")
        risk_manager_memory = FinancialSituationMemory("legacy_risk_manager_memory")

    logger.info(
        "creating_trading_graph",
        ticker=ticker,
        max_debate_rounds=max_debate_rounds,
        enable_memory=enable_memory,
        architecture="parallel"
    )

    # --- LLM Setup ---
    tracker = get_tracker()

    # Data gathering agents: Always LOW thinking
    market_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Market Analyst", tracker)]
    )
    social_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Sentiment Analyst", tracker)]
    )
    news_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("News Analyst", tracker)]
    )
    junior_fund_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Junior Fundamentals Analyst", tracker)]
    )
    senior_fund_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Fundamentals Analyst", tracker)]
    )

    # Retry LLM for data gathering agents
    retry_llm = None
    allow_retry = False
    if not quick_mode and is_gemini_v3_or_greater(config.quick_think_llm):
        retry_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Retry Agent (Deep)", tracker)]
        )
        allow_retry = True
        logger.info("retry_llm_enabled", ticker=ticker)
    elif quick_mode:
        logger.info("retry_llm_disabled_quick_mode", ticker=ticker)

    # Synthesis agents: Mode-dependent thinking
    if quick_mode:
        logger.info("Quick mode: LOW thinking for synthesis agents")
        bull_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Bull Researcher", tracker)]
        )
        bear_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Bear Researcher", tracker)]
        )
        res_mgr_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Research Manager", tracker)]
        )
        pm_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Portfolio Manager", tracker)]
        )
        risky_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Risky Analyst", tracker)]
        )
        safe_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Safe Analyst", tracker)]
        )
        neutral_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Neutral Analyst", tracker)]
        )
    else:
        logger.info("Normal mode: HIGH thinking for synthesis agents")
        bull_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Bull Researcher", tracker)]
        )
        bear_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Bear Researcher", tracker)]
        )
        res_mgr_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Research Manager", tracker)]
        )
        pm_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Portfolio Manager", tracker)]
        )
        risky_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Risky Analyst", tracker)]
        )
        safe_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Safe Analyst", tracker)]
        )
        neutral_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Neutral Analyst", tracker)]
        )

    trader_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Trader", tracker)]
    )

    consultant_llm = get_consultant_llm(
        callbacks=[TokenTrackingCallback("Consultant", tracker)],
        quick_mode=quick_mode
    )

    # --- Node Creation ---

    # Data gathering analysts (parallel)
    market = create_analyst_node(
        market_llm, "market_analyst",
        toolkit.get_technical_tools(), "market_report",
        retry_llm=retry_llm, allow_retry=allow_retry
    )
    sentiment = create_analyst_node(
        social_llm, "sentiment_analyst",
        toolkit.get_sentiment_tools(), "sentiment_report",
        retry_llm=retry_llm, allow_retry=allow_retry
    )
    news = create_analyst_node(
        news_llm, "news_analyst",
        toolkit.get_news_tools(), "news_report",
        retry_llm=retry_llm, allow_retry=allow_retry
    )

    # Fundamentals chain (sequential)
    junior_fund = create_analyst_node(
        junior_fund_llm, "junior_fundamentals_analyst",
        toolkit.get_junior_fundamental_tools(), "raw_fundamentals_data",
        retry_llm=retry_llm, allow_retry=allow_retry
    )
    senior_fund = create_analyst_node(
        senior_fund_llm, "fundamentals_analyst",
        toolkit.get_senior_fundamental_tools(), "fundamentals_report",
        retry_llm=retry_llm, allow_retry=allow_retry
    )
    validator = create_financial_health_validator_node()

    # Agent-specific tool nodes for parallel execution
    # CRITICAL: Uses create_agent_tool_node to filter messages by agent's tools
    # This prevents parallel agents from executing each other's tool_calls
    market_tools = create_agent_tool_node(
        toolkit.get_market_tools(), "market_analyst"
    )
    sentiment_tools = create_agent_tool_node(
        toolkit.get_sentiment_tools(), "sentiment_analyst"
    )
    news_tools = create_agent_tool_node(
        toolkit.get_news_tools(), "news_analyst"
    )
    junior_fund_tools = create_agent_tool_node(
        toolkit.get_junior_fundamental_tools(), "junior_fundamentals_analyst"
    )

    # Research & Decision nodes
    bull = create_researcher_node(bull_llm, bull_memory, "bull_researcher")
    bear = create_researcher_node(bear_llm, bear_memory, "bear_researcher")
    res_mgr = create_research_manager_node(res_mgr_llm, invest_judge_memory)
    trader = create_trader_node(trader_llm, trader_memory)

    # Risk nodes
    risky = create_risk_debater_node(risky_llm, "risky_analyst")
    safe = create_risk_debater_node(safe_llm, "safe_analyst")
    neutral = create_risk_debater_node(neutral_llm, "neutral_analyst")
    pm = create_portfolio_manager_node(pm_llm, risk_manager_memory)

    # Consultant node (optional)
    consultant = None
    if consultant_llm is not None:
        consultant = create_consultant_node(consultant_llm, "consultant")
        logger.info("consultant_node_enabled", ticker=ticker)
    else:
        logger.info("consultant_node_disabled", ticker=ticker)

    # --- Graph Construction ---
    workflow = StateGraph(AgentState)

    # Dispatcher node (no-op, just for fan-out)
    async def dispatcher_node(state: AgentState, config: RunnableConfig):
        """Entry point that triggers parallel analyst streams."""
        return {}

    workflow.add_node("Dispatcher", dispatcher_node)

    # Sync Check node (no-op, routing logic is in conditional edges)
    async def sync_check_node(state: AgentState, config: RunnableConfig):
        """Synchronization barrier - routing logic in conditional edges."""
        return {}

    workflow.add_node("Sync Check", sync_check_node)

    # Add analyst nodes
    workflow.add_node("Market Analyst", market)
    workflow.add_node("Sentiment Analyst", sentiment)
    workflow.add_node("News Analyst", news)
    workflow.add_node("Junior Fundamentals Analyst", junior_fund)
    workflow.add_node("Fundamentals Analyst", senior_fund)
    workflow.add_node("Financial Validator", validator)
    # Separate tool nodes for parallel execution (avoids sender race condition)
    workflow.add_node("market_tools", market_tools)
    workflow.add_node("sentiment_tools", sentiment_tools)
    workflow.add_node("news_tools", news_tools)
    workflow.add_node("junior_fund_tools", junior_fund_tools)

    # Add research and risk nodes
    workflow.add_node("Bull Researcher", bull)
    workflow.add_node("Bear Researcher", bear)
    workflow.add_node("Research Manager", res_mgr)

    if consultant is not None:
        workflow.add_node("Consultant", consultant)

    workflow.add_node("Trader", trader)
    workflow.add_node("Risky Analyst", risky)
    workflow.add_node("Safe Analyst", safe)
    workflow.add_node("Neutral Analyst", neutral)
    workflow.add_node("Portfolio Manager", pm)

    # --- Edge Configuration ---

    # Entry point
    workflow.set_entry_point("Dispatcher")

    # Fan-out from Dispatcher to all 4 parallel analyst streams
    workflow.add_conditional_edges(
        "Dispatcher",
        fan_out_to_analysts,
        ["Market Analyst", "Sentiment Analyst", "News Analyst", "Junior Fundamentals Analyst"]
    )

    # Market, Sentiment, News: tools loop → Sync Check
    # All branches converge at Sync Check for proper synchronization
    # Each analyst has its own tool node to avoid parallel routing conflicts
    workflow.add_conditional_edges(
        "Market Analyst", should_continue_analyst,
        {"tools": "market_tools", "continue": "Sync Check"}
    )
    workflow.add_edge("market_tools", "Market Analyst")

    workflow.add_conditional_edges(
        "Sentiment Analyst", should_continue_analyst,
        {"tools": "sentiment_tools", "continue": "Sync Check"}
    )
    workflow.add_edge("sentiment_tools", "Sentiment Analyst")

    workflow.add_conditional_edges(
        "News Analyst", should_continue_analyst,
        {"tools": "news_tools", "continue": "Sync Check"}
    )
    workflow.add_edge("news_tools", "News Analyst")

    # Junior Fundamentals flow: tools loop → Senior Fundamentals
    workflow.add_conditional_edges(
        "Junior Fundamentals Analyst", should_continue_analyst,
        {"tools": "junior_fund_tools", "continue": "Fundamentals Analyst"}
    )
    workflow.add_edge("junior_fund_tools", "Junior Fundamentals Analyst")

    # Senior Fundamentals → Validator → Sync Check
    workflow.add_edge("Fundamentals Analyst", "Financial Validator")
    workflow.add_edge("Financial Validator", "Sync Check")

    # Sync Check: wait for all streams, then route
    workflow.add_conditional_edges(
        "Sync Check", sync_check_router,
        {
            "__end__": END,
            "Portfolio Manager": "Portfolio Manager",
            "Research Manager": "Bull Researcher"  # Start debate with Bull
        }
    )

    # Bull/Bear Debate Flow
    workflow.add_conditional_edges(
        "Bull Researcher", debate_router,
        ["Bear Researcher", "Research Manager"]
    )
    workflow.add_conditional_edges(
        "Bear Researcher", debate_router,
        ["Bull Researcher", "Research Manager"]
    )

    # Research Manager → Consultant → Trader (or direct to Trader)
    if consultant is not None:
        workflow.add_edge("Research Manager", "Consultant")
        workflow.add_edge("Consultant", "Trader")
    else:
        workflow.add_edge("Research Manager", "Trader")

    # Trader → Risk Team → Portfolio Manager
    workflow.add_edge("Trader", "Risky Analyst")
    workflow.add_edge("Risky Analyst", "Safe Analyst")
    workflow.add_edge("Safe Analyst", "Neutral Analyst")
    workflow.add_edge("Neutral Analyst", "Portfolio Manager")
    workflow.add_edge("Portfolio Manager", END)

    logger.info(
        "trading_graph_created",
        ticker=ticker,
        architecture="parallel",
        parallel_streams=["Market", "Sentiment", "News", "Fundamentals"]
    )

    return workflow.compile()
