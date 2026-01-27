"""
Multi-Agent Trading System Graph
REFACTORED: Parallel analyst execution with synchronization barrier.

Architecture:
1. Dispatcher fans out to 6 parallel analyst streams
2. Market, Sentiment, News analysts run independently → Sync Check
3. Junior Fundamentals, Foreign Language, Legal Counsel run in parallel → Fundamentals Sync Check
4. Fundamentals Sync waits for all 3, then Senior Fund → Validator → Sync Check
5. Sync Check waits for all streams, then routes to:
   - PASS: Bull/Bear Researcher R1 (parallel debate)
   - REJECT: PM Fast-Fail (separate node to avoid edge conflict with normal PM)
6. Bull/Bear debate → Consultant → Trader → Risk Team → Portfolio Manager
7. Portfolio Manager → Chart Generator → END (charts reflect PM verdict)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import structlog
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig

from src.agents import (
    AgentState,
    create_analyst_node,
    create_auditor_node,
    create_consultant_node,
    create_financial_health_validator_node,
    create_legal_counsel_node,
    create_portfolio_manager_node,
    create_research_manager_node,
    create_researcher_node,
    create_risk_debater_node,
    create_trader_node,
    create_valuation_calculator_node,
)
from src.charts.chart_node import create_chart_generator_node
from src.config import config
from src.llms import (
    create_auditor_llm,
    create_deep_thinking_llm,
    create_quick_thinking_llm,
    get_consultant_llm,
    is_gemini_v3_or_greater,
)
from src.memory import (
    FinancialSituationMemory,
    cleanup_all_memories,
    create_memory_instances,
    sanitize_ticker_for_collection,
)
from src.token_tracker import TokenTrackingCallback, get_tracker
from src.toolkit import toolkit

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
    ticker_memories: dict[str, Any] | None = None
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
    has_tool_calls = (
        messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls
    )

    result = "tools" if has_tool_calls else "continue"

    logger.info(
        "analyst_routing", sender=sender, has_tool_calls=has_tool_calls, result=result
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
        "foreign_language_analyst": "Foreign Language Analyst",
        "legal_counsel": "Legal Counsel",
        "global_forensic_auditor": "Auditor",
        "value_trap_detector": "Value Trap Detector",
    }

    node_name = agent_map.get(sender, "Market Analyst")

    logger.debug("tool_routing", sender=sender, routing_to=node_name)

    return node_name


def create_agent_tool_node(tools: list, agent_key: str):
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
    tool_node = ToolNode(tools, handle_tool_errors=True)
    tool_names = {tool.name for tool in tools}

    async def agent_tool_node(state: AgentState, config: RunnableConfig) -> dict:
        """Execute tools for a specific agent by filtering messages."""
        messages = state.get("messages", [])

        # Find the AIMessage from THIS agent (has tool_calls for our tools)
        # CRITICAL FIX: Also check msg.name == agent_key to avoid picking up
        # tool_calls from a different agent that happens to use the same tool
        target_message = None
        for msg in reversed(messages):
            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                # CRITICAL: Check that this AIMessage is from THIS agent
                # Multiple agents may call the same tool (e.g., get_news)
                if getattr(msg, "name", None) != agent_key:
                    continue

                # Check if any tool_call is for one of our tools
                msg_tool_names = {
                    tc.get("name", tc.get("function", {}).get("name", ""))
                    for tc in msg.tool_calls
                }
                if (
                    msg_tool_names & tool_names
                ):  # Intersection - has at least one of our tools
                    target_message = msg
                    break

        if target_message is None:
            logger.warning(
                "agent_tool_node_no_matching_message",
                agent_key=agent_key,
                tool_names=list(tool_names),
                message="No AIMessage found with tool_calls for this agent's tools",
            )
            return {"messages": []}

        # Create a filtered state with only the target message
        # This ensures ToolNode processes the correct tool_calls
        filtered_messages = [target_message]

        logger.debug(
            "agent_tool_node_executing",
            agent_key=agent_key,
            tool_calls=[tc.get("name") for tc in target_message.tool_calls],
            tool_call_count=len(target_message.tool_calls),
            total_messages=len(messages),
        )

        # Execute the tools using the filtered messages
        result = await tool_node.ainvoke({"messages": filtered_messages}, config)

        # CRITICAL: Tag ToolMessages with agent_key for parallel execution filtering
        # This allows the analyst to identify its own tool results
        result_msg_count = len(result.get("messages", []))
        expected_count = len(target_message.tool_calls)
        logger.debug(
            "agent_tool_node_results",
            agent_key=agent_key,
            result_message_count=result_msg_count,
            tool_call_count=expected_count,
        )

        # Graceful error recovery: Log error if tool call count doesn't match results
        # This can happen if a tool fails to execute or returns no result
        if result_msg_count != expected_count:
            logger.error(
                "agent_tool_node_message_mismatch",
                agent_key=agent_key,
                expected_tool_calls=expected_count,
                received_results=result_msg_count,
                tool_calls_requested=[
                    tc.get("name") for tc in target_message.tool_calls
                ],
                message="Not all tool calls resulted in ToolMessages. Agent may receive incomplete data.",
            )

        if "messages" in result:
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    # Preserve tool name, add agent_key to additional_kwargs
                    if msg.additional_kwargs is None:
                        msg.additional_kwargs = {}
                    msg.additional_kwargs["agent_key"] = agent_key

        return result

    return agent_tool_node


def _is_auditor_enabled() -> bool:
    """
    Check if auditor node should be enabled.

    Must match the logic in create_auditor_llm() to avoid graph/router mismatch.
    Returns True only if:
    - ENABLE_CONSULTANT is True
    - OPENAI_API_KEY is available
    """
    if not config.enable_consultant:
        return False
    # Check if OpenAI API key exists (same check as create_auditor_llm)
    return bool(config.get_openai_api_key())


def fan_out_to_analysts(state: AgentState, config: RunnableConfig) -> list[str]:
    """
    Fan-out router that triggers all parallel analyst streams.
    Returns a list of destinations for parallel execution.

    Architecture:
    - Market, Sentiment, News → Sync Check (direct)
    - Junior Fundamentals, Foreign Language, Legal Counsel → Fundamentals Sync → Senior → Validator → Sync Check
    - Value Trap Detector → Sync Check (governance analysis)
    - Auditor (if enabled) → Sync Check (independent forensic track)
    """
    destinations = [
        "Market Analyst",
        "Sentiment Analyst",
        "News Analyst",
        "Junior Fundamentals Analyst",
        "Foreign Language Analyst",
        "Legal Counsel",
        "Value Trap Detector",
    ]
    if _is_auditor_enabled():
        destinations.append("Auditor")
    return destinations


def fundamentals_sync_router(
    state: AgentState, config: RunnableConfig
) -> Literal["Fundamentals Analyst", "__end__"]:
    """
    Synchronization barrier for Junior Fundamentals, Foreign Language, and Legal Counsel.

    All three analysts run in parallel gathering financial/legal data. This barrier waits
    for all to complete before allowing the Senior Fundamentals Analyst to process
    their combined output.

    Routing behavior:
    - If ALL THREE complete (raw_fundamentals_data, foreign_language_report, legal_report):
      Route to Senior Fundamentals Analyst
    - If not all complete: Return __end__ to terminate THIS branch
      (the other branches will eventually complete and trigger the router)
    """
    junior_done = bool(state.get("raw_fundamentals_data"))
    foreign_done = bool(state.get("foreign_language_report"))
    legal_done = bool(state.get("legal_report"))

    logger.info(
        "fundamentals_sync_status",
        junior_done=junior_done,
        foreign_done=foreign_done,
        legal_done=legal_done,
    )

    if junior_done and foreign_done and legal_done:
        logger.info(
            "fundamentals_sync_complete",
            message="Junior, Foreign Language, and Legal Counsel complete - proceeding to Senior Fundamentals",
        )
        return "Fundamentals Analyst"

    # Not all fundamentals streams complete - terminate this branch
    return "__end__"


def sync_check_router(
    state: AgentState, config: RunnableConfig
) -> Literal["PM Fast-Fail", "__end__"] | list[str]:
    """
    Synchronization barrier for parallel analyst streams (fan-in pattern).

    All parallel branches converge here. Each branch calls sync_check
    when it completes. The router checks if ALL required reports are present:
    - market_report (from Market Analyst)
    - sentiment_report (from Sentiment Analyst)
    - news_report (from News Analyst)
    - value_trap_report (from Value Trap Detector)
    - pre_screening_result (from Validator in Fundamentals chain)

    Routing behavior:
    - If NOT all reports present: Return __end__ to terminate THIS branch
      (other branches continue running and will hit sync_check later)
    - If all present AND pre_screening=REJECT: Route to PM Fast-Fail (separate node to avoid edge conflicts)
    - If all present AND pre_screening=PASS: Fan-out to Bull/Bear Researcher R1 (parallel)

    The LAST branch to complete will see all reports and proceed to the next phase.
    """
    market_done = bool(state.get("market_report"))
    sentiment_done = bool(state.get("sentiment_report"))
    news_done = bool(state.get("news_report"))
    value_trap_done = bool(state.get("value_trap_report"))

    # Validator completion check
    pre_screening = state.get("pre_screening_result")
    validator_done = pre_screening in ["PASS", "REJECT"]

    # Auditor check (if enabled)
    # Uses same helper as fan_out_to_analysts for consistency
    auditor_done = True
    if _is_auditor_enabled():
        auditor_done = bool(state.get("auditor_report"))

    all_done = all(
        [
            market_done,
            sentiment_done,
            news_done,
            value_trap_done,
            validator_done,
            auditor_done,
        ]
    )

    logger.info(
        "sync_check_status",
        market_done=market_done,
        sentiment_done=sentiment_done,
        news_done=news_done,
        value_trap_done=value_trap_done,
        validator_done=validator_done,
        auditor_done=auditor_done,
        pre_screening=pre_screening,
        all_done=all_done,
    )

    if not all_done:
        # This branch finished early - terminate it
        # Other branches continue executing
        return "__end__"

    # All streams complete - route based on validator result
    if pre_screening == "REJECT":
        logger.info(
            "sync_routing_to_pm_reject",
            message="Red flags detected - skipping debate, routing to PM Fast-Fail",
        )
        # CRITICAL: Use separate "PM Fast-Fail" node to avoid edge conflict
        # with the normal path (Risk Team → Portfolio Manager)
        return "PM Fast-Fail"

    logger.info(
        "sync_routing_to_debate",
        message="All analysts complete - proceeding to Bull/Bear Debate Round 1",
    )
    # Fan-out to parallel Bull/Bear R1 by returning list
    return ["Bull Researcher R1", "Bear Researcher R1"]


# --- Graph Creation ---


def create_trading_graph(
    max_debate_rounds: int = 2,
    max_risk_discuss_rounds: int = 1,
    enable_memory: bool = True,
    recursion_limit: int = 100,
    ticker: str | None = None,
    cleanup_previous: bool = False,
    quick_mode: bool = False,
    # Chart generation parameters (post-PM)
    chart_format: str = "png",
    transparent_charts: bool = False,
    image_dir: Path | None = None,
    skip_charts: bool = False,
):
    """
    Create the multi-agent trading analysis graph with parallel analyst execution.

    Architecture:
    1. Dispatcher fans out to 4 parallel analyst streams
    2. Each stream processes independently with tool calling
    3. Sync Check waits for all streams to complete
    4. Routes to Research Manager (PASS) or Portfolio Manager (REJECT)
    5. Bull/Bear debate → Consultant → Trader → Risk → Portfolio Manager
    6. Portfolio Manager → Chart Generator → END (charts reflect PM verdict)

    Args:
        ticker: Stock ticker symbol (e.g., "0005.HK")
        cleanup_previous: If True, deletes previous memories
        max_debate_rounds: Maximum rounds of bull/bear debate
        max_risk_discuss_rounds: Maximum rounds of risk discussion
        enable_memory: Whether to enable agent memory
        recursion_limit: Max recursion depth
        quick_mode: If True, use faster/cheaper models
        chart_format: Chart output format ('png' or 'svg')
        transparent_charts: Whether to use transparent chart backgrounds
        image_dir: Directory for chart output (None = use config default)
        skip_charts: If True, skip chart generation entirely

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
            bull_memory,
            bear_memory,
            invest_judge_memory,
            trader_memory,
            risk_manager_memory,
        ]
        if not all(all_memories):
            missing = []
            if not bull_memory:
                missing.append("bull_memory")
            if not bear_memory:
                missing.append("bear_memory")
            if not invest_judge_memory:
                missing.append("invest_judge_memory")
            if not trader_memory:
                missing.append("trader_memory")
            if not risk_manager_memory:
                missing.append("risk_manager_memory")
            raise ValueError(
                f"Failed to create memory instances for {ticker}. Missing: {', '.join(missing)}"
            )

        logger.info(
            "ticker_memories_ready",
            ticker=ticker,
            bull_available=bull_memory.available,
            bear_available=bear_memory.available,
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
        architecture="parallel",
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

    # Retry LLM for data gathering agents (Junior Fundamentals, NOT Senior)
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
    # Senior Fundamentals Analyst is a synthesis agent (parses JSON, calculates scores, produces DATA_BLOCK)
    if quick_mode:
        logger.info("Quick mode: LOW thinking for synthesis agents")
        senior_fund_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Fundamentals Analyst", tracker)]
        )
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
        # Senior Fundamentals uses QUICK thinking even in normal mode:
        # - It does structured scoring (rule-based), not creative synthesis
        # - Large input context (48k tokens) + HIGH thinking causes 504 timeouts
        # - Prompt v7.2 EFFICIENCY DIRECTIVE further reduces verbosity
        senior_fund_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Fundamentals Analyst", tracker)]
        )
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

    # Valuation Calculator: Always quick thinking (simple parameter extraction)
    valuation_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Valuation Calculator", tracker)]
    )

    consultant_llm = get_consultant_llm(
        callbacks=[TokenTrackingCallback("Consultant", tracker)], quick_mode=quick_mode
    )

    auditor_llm = create_auditor_llm(
        callbacks=[TokenTrackingCallback("Global Forensic Auditor", tracker)]
    )

    # --- Node Creation ---

    # Data gathering analysts (parallel)
    market = create_analyst_node(
        market_llm,
        "market_analyst",
        toolkit.get_technical_tools(),
        "market_report",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )
    sentiment = create_analyst_node(
        social_llm,
        "sentiment_analyst",
        toolkit.get_sentiment_tools(),
        "sentiment_report",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )
    news = create_analyst_node(
        news_llm,
        "news_analyst",
        toolkit.get_news_tools(),
        "news_report",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )

    # Foreign Language Analyst (parallel with Junior Fundamentals and Legal Counsel)
    foreign_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Foreign Language Analyst", tracker)]
    )
    foreign_analyst = create_analyst_node(
        foreign_llm,
        "foreign_language_analyst",
        toolkit.get_foreign_language_tools(),
        "foreign_language_report",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )

    # Legal Counsel (parallel with Junior Fundamentals and Foreign Language)
    legal_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Legal Counsel", tracker)]
    )
    legal_counsel = create_legal_counsel_node(legal_llm, toolkit.get_legal_tools())

    # Value Trap Detector (parallel governance analysis)
    value_trap_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Value Trap Detector", tracker)]
    )
    value_trap_detector = create_analyst_node(
        value_trap_llm,
        "value_trap_detector",
        toolkit.get_value_trap_tools(),
        "value_trap_report",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )

    # Auditor (Independent Forensic Track)
    # Runs parallel to other analysts, provides independent forensic validation
    # Output is consumed by Consultant for cross-validation
    auditor = None
    auditor_tools = None
    if auditor_llm:
        # Auditor uses: foreign sources (native filings), financial metrics, and news
        # These are the same tools other analysts use, but auditor works independently
        _aud_tools = (
            toolkit.get_foreign_language_tools()
            + toolkit.get_junior_fundamental_tools()
            + toolkit.get_news_tools()
        )
        auditor = create_auditor_node(auditor_llm, _aud_tools)
        auditor_tools = create_agent_tool_node(_aud_tools, "global_forensic_auditor")
        logger.info("auditor_node_enabled", ticker=ticker)

    # Fundamentals chain (Junior + Foreign → Senior → Validator)
    # Junior uses retry logic (data gathering agent with quick thinking)
    junior_fund = create_analyst_node(
        junior_fund_llm,
        "junior_fundamentals_analyst",
        toolkit.get_junior_fundamental_tools(),
        "raw_fundamentals_data",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )
    # Senior is a synthesis agent (mode-dependent thinking, no retry needed)
    senior_fund = create_analyst_node(
        senior_fund_llm,
        "fundamentals_analyst",
        toolkit.get_senior_fundamental_tools(),
        "fundamentals_report",
    )
    validator = create_financial_health_validator_node()

    # Agent-specific tool nodes for parallel execution
    # CRITICAL: Uses create_agent_tool_node to filter messages by agent's tools
    # This prevents parallel agents from executing each other's tool_calls
    market_tools = create_agent_tool_node(toolkit.get_market_tools(), "market_analyst")
    sentiment_tools = create_agent_tool_node(
        toolkit.get_sentiment_tools(), "sentiment_analyst"
    )
    news_tools = create_agent_tool_node(toolkit.get_news_tools(), "news_analyst")
    junior_fund_tools = create_agent_tool_node(
        toolkit.get_junior_fundamental_tools(), "junior_fundamentals_analyst"
    )
    foreign_tools = create_agent_tool_node(
        toolkit.get_foreign_language_tools(), "foreign_language_analyst"
    )
    legal_tools = create_agent_tool_node(toolkit.get_legal_tools(), "legal_counsel")
    value_trap_tools = create_agent_tool_node(
        toolkit.get_value_trap_tools(), "value_trap_detector"
    )

    # Research & Decision nodes
    # Create separate nodes for each debate round (enables parallel execution)
    bull_r1 = create_researcher_node(
        bull_llm, bull_memory, "bull_researcher", round_num=1
    )
    bear_r1 = create_researcher_node(
        bear_llm, bear_memory, "bear_researcher", round_num=1
    )
    bull_r2 = create_researcher_node(
        bull_llm, bull_memory, "bull_researcher", round_num=2
    )
    bear_r2 = create_researcher_node(
        bear_llm, bear_memory, "bear_researcher", round_num=2
    )
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

    # Valuation Calculator node (extracts params for chart generation)
    valuation_calc = create_valuation_calculator_node(valuation_llm)

    # Chart Generator node (runs after PM, reflects verdict in visuals)
    chart_generator = create_chart_generator_node(
        chart_format=chart_format,
        transparent=transparent_charts,
        image_dir=image_dir,
        skip_charts=skip_charts or quick_mode,  # Skip charts in quick mode
    )

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

    # Fundamentals Sync Check node (waits for Junior + Foreign Language analysts)
    async def fundamentals_sync_node(state: AgentState, config: RunnableConfig):
        """Synchronization barrier for fundamentals data streams - routing logic in conditional edges."""
        return {}

    workflow.add_node("Fundamentals Sync Check", fundamentals_sync_node)

    # Debate Sync R1 node - assembles R1 outputs for R2 context
    async def debate_sync_r1_node(state: AgentState, config: RunnableConfig):
        """
        Synchronization point after Round 1 of Bull/Bear debate.
        Assembles R1 outputs so R2 agents can reference opponent arguments.
        """
        debate = state.get("investment_debate_state", {})
        bull_r1 = debate.get("bull_round1", "")
        bear_r1 = debate.get("bear_round1", "")

        # Build partial history (R1 only) for any downstream use
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
                "count": 2,  # 2 arguments completed
            }
        }

    workflow.add_node("Debate Sync R1", debate_sync_r1_node)

    # Debate Sync Final node - assembles full debate history for Research Manager
    async def debate_sync_final_node(state: AgentState, config: RunnableConfig):
        """
        Final synchronization after debate completion.
        Assembles full debate history (R1 + R2) for Research Manager.
        Works for both quick mode (R1 only) and standard mode (R1 + R2).
        """
        debate = state.get("investment_debate_state", {})
        bull_r1 = debate.get("bull_round1", "")
        bear_r1 = debate.get("bear_round1", "")
        bull_r2 = debate.get("bull_round2", "")
        bear_r2 = debate.get("bear_round2", "")

        # Build full history
        if bull_r2 or bear_r2:
            # Standard mode: 2 rounds
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
            # Quick mode: 1 round only
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

    workflow.add_node("Debate Sync Final", debate_sync_final_node)

    # Add analyst nodes
    workflow.add_node("Market Analyst", market)
    workflow.add_node("Sentiment Analyst", sentiment)
    workflow.add_node("News Analyst", news)
    workflow.add_node("Junior Fundamentals Analyst", junior_fund)
    workflow.add_node("Foreign Language Analyst", foreign_analyst)
    workflow.add_node("Legal Counsel", legal_counsel)
    workflow.add_node("Value Trap Detector", value_trap_detector)
    workflow.add_node("Fundamentals Analyst", senior_fund)
    workflow.add_node("Financial Validator", validator)
    # Separate tool nodes for parallel execution (avoids sender race condition)
    workflow.add_node("market_tools", market_tools)
    workflow.add_node("sentiment_tools", sentiment_tools)
    workflow.add_node("news_tools", news_tools)
    workflow.add_node("junior_fund_tools", junior_fund_tools)
    workflow.add_node("foreign_tools", foreign_tools)
    workflow.add_node("legal_tools", legal_tools)
    workflow.add_node("value_trap_tools", value_trap_tools)

    if auditor is not None:
        workflow.add_node("Auditor", auditor)
        workflow.add_node("auditor_tools", auditor_tools)

    # Add research nodes (separate nodes for each round enables parallel execution)
    workflow.add_node("Bull Researcher R1", bull_r1)
    workflow.add_node("Bear Researcher R1", bear_r1)
    workflow.add_node("Bull Researcher R2", bull_r2)
    workflow.add_node("Bear Researcher R2", bear_r2)
    workflow.add_node("Research Manager", res_mgr)
    workflow.add_node("Valuation Calculator", valuation_calc)

    if consultant is not None:
        workflow.add_node("Consultant", consultant)

    workflow.add_node("Trader", trader)
    workflow.add_node("Risky Analyst", risky)
    workflow.add_node("Safe Analyst", safe)
    workflow.add_node("Neutral Analyst", neutral)
    workflow.add_node("Portfolio Manager", pm)
    workflow.add_node("Chart Generator", chart_generator)

    # --- Edge Configuration ---

    # Entry point
    workflow.set_entry_point("Dispatcher")

    # Fan-out from Dispatcher to all parallel analyst streams
    # Note: fan_out_to_analysts dynamically includes "Auditor" if enabled
    # Path map must match what fan_out_to_analysts can return
    fan_out_destinations = [
        "Market Analyst",
        "Sentiment Analyst",
        "News Analyst",
        "Junior Fundamentals Analyst",
        "Foreign Language Analyst",
        "Legal Counsel",
        "Value Trap Detector",
    ]
    if auditor is not None:
        fan_out_destinations.append("Auditor")

    workflow.add_conditional_edges(
        "Dispatcher",
        fan_out_to_analysts,
        fan_out_destinations,
    )

    # Market, Sentiment, News: tools loop → Sync Check
    # All branches converge at Sync Check for proper synchronization
    # Each analyst has its own tool node to avoid parallel routing conflicts
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

    # Junior Fundamentals flow: tools loop → Fundamentals Sync Check
    workflow.add_conditional_edges(
        "Junior Fundamentals Analyst",
        should_continue_analyst,
        {"tools": "junior_fund_tools", "continue": "Fundamentals Sync Check"},
    )
    workflow.add_edge("junior_fund_tools", "Junior Fundamentals Analyst")

    # Foreign Language Analyst flow: tools loop → Fundamentals Sync Check
    workflow.add_conditional_edges(
        "Foreign Language Analyst",
        should_continue_analyst,
        {"tools": "foreign_tools", "continue": "Fundamentals Sync Check"},
    )
    workflow.add_edge("foreign_tools", "Foreign Language Analyst")

    # Legal Counsel flow: tools loop → Fundamentals Sync Check
    workflow.add_conditional_edges(
        "Legal Counsel",
        should_continue_analyst,
        {"tools": "legal_tools", "continue": "Fundamentals Sync Check"},
    )
    workflow.add_edge("legal_tools", "Legal Counsel")

    # Value Trap Detector flow: tools loop → Sync Check (independent governance stream)
    workflow.add_conditional_edges(
        "Value Trap Detector",
        should_continue_analyst,
        {"tools": "value_trap_tools", "continue": "Sync Check"},
    )
    workflow.add_edge("value_trap_tools", "Value Trap Detector")

    # Auditor flow: tools loop → Sync Check
    if auditor is not None:
        workflow.add_conditional_edges(
            "Auditor",
            should_continue_analyst,
            {"tools": "auditor_tools", "continue": "Sync Check"},
        )
        workflow.add_edge("auditor_tools", "Auditor")

    # Fundamentals Sync Check: wait for Junior + Foreign + Legal, then route to Senior
    workflow.add_conditional_edges(
        "Fundamentals Sync Check",
        fundamentals_sync_router,
        {"__end__": END, "Fundamentals Analyst": "Fundamentals Analyst"},
    )

    # Senior Fundamentals → Validator → Sync Check
    workflow.add_edge("Fundamentals Analyst", "Financial Validator")
    workflow.add_edge("Financial Validator", "Sync Check")

    # PM Fast-Fail node: separate from main Portfolio Manager to avoid edge conflict
    # This node runs when pre_screening == "REJECT" (fast-fail path)
    # Using the same PM implementation but as a distinct graph node
    pm_fast_fail = create_portfolio_manager_node(pm_llm, risk_manager_memory)
    workflow.add_node("PM Fast-Fail", pm_fast_fail)
    workflow.add_edge(
        "PM Fast-Fail", "Chart Generator"
    )  # Charts still generated (verdict suppresses football field)

    # Sync Check: wait for all streams, then route
    # Uses list-style destinations to support fan-out (router returns list for parallel R1)
    # CRITICAL: PM Fast-Fail is separate from Portfolio Manager to avoid edge conflicts
    workflow.add_conditional_edges(
        "Sync Check",
        sync_check_router,
        ["__end__", "PM Fast-Fail", "Bull Researcher R1", "Bear Researcher R1"],
    )

    # === PARALLEL BULL/BEAR DEBATE ===

    # Round 1: Bull and Bear run in parallel
    # Fan-out handled by sync_check_router returning ["Bull Researcher R1", "Bear Researcher R1"]
    # Both R1 nodes converge at Debate Sync R1
    workflow.add_edge("Bull Researcher R1", "Debate Sync R1")
    workflow.add_edge("Bear Researcher R1", "Debate Sync R1")

    # Router after R1: go to R2 (standard) or skip to Final (quick mode)
    def debate_r1_router(
        state: AgentState, config: RunnableConfig
    ) -> Literal["Debate Sync Final"] | list[str]:
        """Route after Round 1: to Round 2 or directly to final sync (quick mode)."""
        context = config.get("configurable", {}).get("context")
        max_rounds = getattr(context, "max_debate_rounds", 2) if context else 2

        if max_rounds <= 1:
            # Quick mode: skip R2, go directly to final sync
            logger.info("debate_r1_router", decision="skip_r2_quick_mode")
            return "Debate Sync Final"
        else:
            # Standard mode: fan-out to parallel R2
            logger.info("debate_r1_router", decision="proceed_to_r2")
            return ["Bull Researcher R2", "Bear Researcher R2"]

    # Uses list-style destinations to support fan-out (router returns list for parallel R2)
    workflow.add_conditional_edges(
        "Debate Sync R1",
        debate_r1_router,
        ["Debate Sync Final", "Bull Researcher R2", "Bear Researcher R2"],
    )

    # Round 2: Bull and Bear run in parallel (both have R1 context)
    # Fan-out handled by debate_r1_router returning ["Bull Researcher R2", "Bear Researcher R2"]
    # Both R2 nodes converge at Debate Sync Final
    workflow.add_edge("Bull Researcher R2", "Debate Sync Final")
    workflow.add_edge("Bear Researcher R2", "Debate Sync Final")

    # Final sync → Research Manager
    workflow.add_edge("Debate Sync Final", "Research Manager")

    # Research Manager → [Valuation Calculator || Consultant] → Trader
    # Valuation Calculator always runs; Consultant is optional
    # Both run in parallel after Research Manager, converge at Trader
    workflow.add_edge("Research Manager", "Valuation Calculator")
    workflow.add_edge("Valuation Calculator", "Trader")

    if consultant is not None:
        workflow.add_edge("Research Manager", "Consultant")
        workflow.add_edge("Consultant", "Trader")

    # Trader → Risk Team (parallel) → Portfolio Manager
    # Risk analysts run in parallel for ~40 sec savings
    workflow.add_edge("Trader", "Risky Analyst")
    workflow.add_edge("Trader", "Safe Analyst")
    workflow.add_edge("Trader", "Neutral Analyst")
    # All three converge - PM waits for all to complete
    workflow.add_edge("Risky Analyst", "Portfolio Manager")
    workflow.add_edge("Safe Analyst", "Portfolio Manager")
    workflow.add_edge("Neutral Analyst", "Portfolio Manager")
    # PM → Chart Generator → END (charts generated post-verdict)
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
            if consultant is not None
            else "Valuation Calculator"
        ),
        risk_team_parallel=["Risky Analyst", "Safe Analyst", "Neutral Analyst"],
        post_pm="Chart Generator (verdict-aligned visuals)",
        chart_generation=not (skip_charts or quick_mode),
        quick_mode=quick_mode,
    )

    return workflow.compile()
