"""
Multi-Agent Trading System - Agent Definitions
FIXED: All data passing issues - agents now receive complete reports.
ADDED: Debug logging to track data flow.
FIXED: Memory retrieval now contextualized per ticker to prevent cross-contamination.
UPDATED (Pass 3 Fixes): Added Negative Constraint to prompts and strict metadata filtering.
UPDATED: Added explicit 429/ResourceExhausted handling for Gemini free tier.
FIXED: Corrected memory query parameter name to 'metadata_filter'.
"""

import asyncio
import os
from typing import Annotated, List, Dict, Any, Optional, Set, Callable
from typing_extensions import TypedDict
from datetime import datetime

from langgraph.graph import MessagesState
from langgraph.types import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage

import structlog

logger = structlog.get_logger(__name__)

# --- Rate Limit Handling ---

async def invoke_with_rate_limit_handling(
    runnable,
    input_data: Dict[str, Any],
    max_attempts: int = 3,
    context: str = "LLM"
) -> Any:
    """
    Invoke LLM with explicit 429/ResourceExhausted handling for free tier.

    This wrapper adds extended backoff (60-180s) beyond LangChain's default retry logic,
    which maxes out at ~60s. Critical for Gemini free tier (15 RPM).

    Args:
        runnable: LangChain runnable (LLM, chain, etc.)
        input_data: Input dictionary for runnable
        max_attempts: Number of attempts at this wrapper level (default 3)
        context: Description for logging (e.g., "Market Analyst")

    Returns:
        Result from runnable.ainvoke()

    Raises:
        Exception: Re-raises if not a rate limit error or after max attempts
    """
    quiet_mode = os.environ.get("QUIET_MODE", "false").lower() == "true"

    for attempt in range(max_attempts):
        try:
            return await runnable.ainvoke(input_data)
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__

            # Detect rate limit errors (429, ResourceExhausted, quota exceeded)
            is_rate_limit = any([
                "429" in error_str,
                "rate limit" in error_str,
                "quota" in error_str,
                "resourceexhausted" in error_str,
                "resource exhausted" in error_str,
                "too many requests" in error_str
            ])

            if is_rate_limit and attempt < max_attempts - 1:
                # Extended exponential backoff: 60s, 120s, 180s
                wait_time = 60 * (attempt + 1)

                # Log unless in quiet mode
                if not quiet_mode:
                    logger.warning(
                        "rate_limit_detected",
                        context=context,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        wait_seconds=wait_time,
                        error_type=error_type,
                        error_message=str(e)[:200]  # Truncate long errors
                    )

                await asyncio.sleep(wait_time)
                continue  # Retry

            # Not a rate limit error, or final attempt - re-raise
            raise

# --- State Definitions ---
class InvestDebateState(TypedDict):
    """State tracking bull/bear investment debate progression."""
    bull_history: str
    bear_history: str
    history: str
    current_response: str
    judge_decision: str
    count: int

class RiskDebateState(TypedDict):
    """State tracking multi-perspective risk assessment debate."""
    risky_history: str
    safe_history: str
    neutral_history: str
    history: str
    latest_speaker: str
    current_risky_response: str
    current_safe_response: str
    current_neutral_response: str
    judge_decision: str
    count: int

def take_last(x, y):
    """Reducer function: takes the most recent value. Used with Annotated state fields."""
    return y

class AgentState(MessagesState):
    company_of_interest: str
    company_name: str  # ADDED: Verified company name to prevent LLM hallucination
    trade_date: str
    sender: str

    market_report: Annotated[str, take_last]
    sentiment_report: Annotated[str, take_last]
    news_report: Annotated[str, take_last]
    fundamentals_report: Annotated[str, take_last]
    investment_debate_state: Annotated[InvestDebateState, take_last]
    investment_plan: Annotated[str, take_last]
    consultant_review: Annotated[str, take_last]  # ADDED: External consultant cross-validation
    trader_investment_plan: Annotated[str, take_last]
    risk_debate_state: Annotated[RiskDebateState, take_last]
    final_trade_decision: Annotated[str, take_last]
    tools_called: Annotated[Dict[str, Set[str]], take_last]
    prompts_used: Annotated[Dict[str, Dict[str, str]], take_last]

    # Red-flag detection fields
    red_flags: Annotated[List[Dict[str, Any]], take_last]
    pre_screening_result: Annotated[str, take_last]  # "PASS" or "REJECT"

# --- Helper Functions ---

def get_context_from_config(config: RunnableConfig) -> Optional[Any]:
    """Extract TradingContext from RunnableConfig.configurable dict."""
    try:
        configurable = config.get("configurable", {})
        return configurable.get("context")
    except (AttributeError, TypeError):
        return None

def get_analysis_context(ticker: str) -> str:
    """Generate contextual analysis guidance based on asset type (ETF vs individual stock)."""
    etf_indicators = ['VTI', 'SPY', 'QQQ', 'IWM', 'VOO', 'VEA', 'VWO', 'BND', 'AGG', 'EFA', 'EEM', 'TLT', 'GLD', 'DIA']
    if any(ind in ticker.upper() for ind in etf_indicators) or 'ETF' in ticker.upper():
        return "This is an ETF (Exchange-Traded Fund). Focus on holdings, expense ratio, and liquidity."
    return "This is an individual stock. Focus on fundamentals, valuation, and competitive advantage."

def filter_messages_for_gemini(messages: List[BaseMessage]) -> List[BaseMessage]:
    if not messages:
        return []
    filtered = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if filtered and isinstance(msg, HumanMessage) and isinstance(filtered[-1], HumanMessage):
            last_msg = filtered.pop()
            new_content = f"{last_msg.content}\n\n{msg.content}"
            filtered.append(HumanMessage(content=new_content))
        else:
            filtered.append(msg)
    return filtered

# --- Agent Factory Functions ---

def create_analyst_node(llm, agent_key: str, tools: List[Any], output_field: str) -> Callable:
    """
    Factory function creating data analyst agent nodes.
    """
    async def analyst_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        from src.prompts import get_prompt
        agent_prompt = get_prompt(agent_key)
        if not agent_prompt:
            logger.error(f"Missing prompt for agent: {agent_key}")
            return {output_field: f"Error: Could not load prompt for {agent_key}."}
        messages_template = [MessagesPlaceholder(variable_name="messages")]
        prompt_template = ChatPromptTemplate.from_messages(messages_template)
        runnable = prompt_template | llm.bind_tools(tools) if tools else prompt_template | llm
        try:
            prompts_used = state.get("prompts_used", {})
            prompts_used[output_field] = {"agent_name": agent_prompt.agent_name, "version": agent_prompt.version}
            filtered_messages = filter_messages_for_gemini(state.get("messages", []))
            context = get_context_from_config(config)
            current_date = context.trade_date if context else datetime.now().strftime("%Y-%m-%d")
            ticker = context.ticker if context else state.get("company_of_interest", "UNKNOWN")
            company_name = state.get("company_name", ticker)  # Get verified company name from state

            # --- CRITICAL FIX: Inject News Report into Fundamentals Analyst Context ---
            extra_context = ""
            if agent_key == "fundamentals_analyst":
                news_report = state.get("news_report", "")
                if news_report:
                    extra_context = f"\n\n### NEWS CONTEXT (Use for Qualitative Growth Scoring)\n{news_report}\n"

            # CRITICAL FIX: Include verified company name to prevent hallucination
            full_system_instruction = f"{agent_prompt.system_message}\n\nDate: {current_date}\nTicker: {ticker}\nCompany: {company_name}\n{get_analysis_context(ticker)}{extra_context}"
            invocation_messages = [SystemMessage(content=full_system_instruction)] + filtered_messages

            # Use rate limit handling wrapper for free tier support
            response = await invoke_with_rate_limit_handling(
                runnable,
                {"messages": invocation_messages},
                context=agent_prompt.agent_name
            )
            new_state = {"sender": agent_key, "messages": [response], "prompts_used": prompts_used}
            
            # Check for tool calls
            has_tool_calls = False
            try:
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    has_tool_calls = isinstance(response.tool_calls, list) and len(response.tool_calls) > 0
            except (AttributeError, TypeError):
                pass

            if has_tool_calls:
                return new_state

            new_state[output_field] = response.content          

            if agent_key == "fundamentals_analyst":
                logger.info("fundamentals_output", has_datablock="DATA_BLOCK" in response.content, length=len(response.content))
            return new_state
        except Exception as e:
            logger.error(f"Analyst node error {output_field}: {str(e)}")
            return {"messages": [AIMessage(content=f"Error: {str(e)}")], output_field: f"Error: {str(e)}"}
    return analyst_node

def create_researcher_node(llm, memory: Optional[Any], agent_key: str) -> Callable:
    async def researcher_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        from src.prompts import get_prompt
        agent_prompt = get_prompt(agent_key)
        if not agent_prompt:
            logger.error(f"Missing prompt for researcher: {agent_key}")
            debate_state = state.get('investment_debate_state', {}).copy()
            debate_state['history'] += f"\n\n[SYSTEM]: Error - Missing prompt for {agent_key}."
            debate_state['count'] = debate_state.get('count', 0) + 1
            return {"investment_debate_state": debate_state}
        agent_name = agent_prompt.agent_name
        reports = f"MARKET: {state.get('market_report')}\nFUNDAMENTALS: {state.get('fundamentals_report')}"
        history = state.get('investment_debate_state', {}).get('history', '')
        
        # FIX: Contextualize memory retrieval to prevent cross-contamination
        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)  # Get verified company name

        # If we have memory, retrieve RELEVANT past insights for THIS ticker
        past_insights = ""
        if memory:
            try:
                # FIX: Strictly enforce metadata filtering
                # CORRECTED PARAMETER NAME: metadata_filter (was filter_metadata in some versions)
                relevant = await memory.query_similar_situations(
                    f"risks and upside for {ticker}",
                    n_results=3,
                    metadata_filter={"ticker": ticker}
                )
                if relevant:
                    past_insights = f"\n\nPAST MEMORY INSIGHTS (STRICTLY FOR {ticker}):\n" + "\n".join([r['document'] for r in relevant])
                else:
                    # If no strict match, do NOT fallback to semantic search to avoid contamination
                    logger.info("memory_no_exact_match", ticker=ticker)
                    past_insights = ""

            except Exception as e:
                logger.error("memory_retrieval_failed", ticker=ticker, error=str(e))
                past_insights = ""

        # FIX: Add Negative Constraint with explicit company name to prevent hallucination
        negative_constraint = f"""
CRITICAL INSTRUCTION:
You are analyzing **{ticker} ({company_name})**.
If the provided context or memory contains information about a DIFFERENT company (e.g., from a previous analysis run), you MUST IGNORE IT.
Only use data explicitly related to {ticker} ({company_name}).
"""

        prompt = f"""{agent_prompt.system_message}\n{negative_constraint}\n\nREPORTS:\n{reports}\n{past_insights}\n\nDEBATE HISTORY:\n{history}\n\nProvide your argument."""
        try:
            # Use rate limit handling wrapper for free tier support
            response = await invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_name
            )
            debate_state = state.get('investment_debate_state', {}).copy()
            argument = f"{agent_name}: {response.content}"
            debate_state['history'] = debate_state.get('history', '') + f"\n\n{argument}"
            debate_state['count'] = debate_state.get('count', 0) + 1
            if agent_name == 'Bull Analyst': 
                debate_state['bull_history'] = debate_state.get('bull_history', '') + f"\n{argument}"
            else: 
                debate_state['bear_history'] = debate_state.get('bear_history', '') + f"\n{argument}"
            return {"investment_debate_state": debate_state}
        except Exception as e:
            logger.error(f"Researcher error {agent_key}: {str(e)}")
            return {"investment_debate_state": state.get('investment_debate_state', {})}
    return researcher_node

def create_research_manager_node(llm, memory: Optional[Any]) -> Callable:
    async def research_manager_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        from src.prompts import get_prompt
        agent_prompt = get_prompt("research_manager")
        if not agent_prompt:
            return {"investment_plan": "Error: Missing prompt"}
        debate = state.get('investment_debate_state', {})
        all_reports = f"""MARKET ANALYST REPORT:\n{state.get('market_report', 'N/A')}\n\nSENTIMENT ANALYST REPORT:\n{state.get('sentiment_report', 'N/A')}\n\nNEWS ANALYST REPORT:\n{state.get('news_report', 'N/A')}\n\nFUNDAMENTALS ANALYST REPORT:\n{state.get('fundamentals_report', 'N/A')}\n\nBULL RESEARCHER:\n{debate.get('bull_history', 'N/A')}\n\nBEAR RESEARCHER:\n{debate.get('bear_history', 'N/A')}"""
        prompt = f"""{agent_prompt.system_message}\n\n{all_reports}\n\nProvide Investment Plan."""
        try:
            response = await invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_prompt.agent_name
            )
            return {"investment_plan": response.content}
        except Exception as e:
            return {"investment_plan": f"Error: {str(e)}"}
    return research_manager_node

def create_trader_node(llm, memory: Optional[Any]) -> Callable:
    async def trader_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        from src.prompts import get_prompt
        agent_prompt = get_prompt("trader")
        if not agent_prompt:
            return {"trader_investment_plan": "Error: Missing prompt"}

        # Include consultant review if available (external cross-validation)
        consultant = state.get('consultant_review', '')
        consultant_section = f"""\n\nEXTERNAL CONSULTANT REVIEW (Cross-Validation):\n{consultant if consultant else 'N/A (consultant disabled or unavailable)'}"""

        all_input = f"""MARKET ANALYST REPORT:\n{state.get('market_report', 'N/A')}\n\nFUNDAMENTALS ANALYST REPORT:\n{state.get('fundamentals_report', 'N/A')}\n\nRESEARCH MANAGER PLAN:\n{state.get('investment_plan', 'N/A')}{consultant_section}"""
        prompt = f"""{agent_prompt.system_message}\n\n{all_input}\n\nCreate Trading Plan."""
        try:
            response = await invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_prompt.agent_name
            )
            return {"trader_investment_plan": response.content}
        except Exception as e:
            return {"trader_investment_plan": f"Error: {str(e)}"}
    return trader_node

def create_risk_debater_node(llm, agent_key: str) -> Callable:
    async def risk_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        from src.prompts import get_prompt
        agent_prompt = get_prompt(agent_key)
        if not agent_prompt:
            risk_state = state.get('risk_debate_state', {}).copy()
            risk_state['history'] += f"\n[SYSTEM]: Error - Missing prompt for {agent_key}"
            risk_state['count'] += 1
            return {"risk_debate_state": risk_state}

        # Include consultant review if available (external cross-validation)
        consultant = state.get('consultant_review', '')
        consultant_section = f"""\n\nEXTERNAL CONSULTANT REVIEW (Cross-Validation):\n{consultant if consultant else 'N/A (consultant disabled or unavailable)'}"""

        prompt = f"""{agent_prompt.system_message}\n\nTRADER PLAN: {state.get('trader_investment_plan')}{consultant_section}\n\nProvide risk assessment."""
        try:
            response = await invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_prompt.agent_name
            )
            risk_state = state.get('risk_debate_state', {}).copy()
            risk_state['history'] += f"\n{agent_prompt.agent_name}: {response.content}\n"
            risk_state['count'] += 1
            return {"risk_debate_state": risk_state}
        except Exception as e:
            return {"risk_debate_state": state.get('risk_debate_state', {})}
    return risk_node

def create_portfolio_manager_node(llm, memory: Optional[Any]) -> Callable:
    async def pm_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        from src.prompts import get_prompt
        agent_prompt = get_prompt("portfolio_manager")
        if not agent_prompt:
            return {"final_trade_decision": "Error: Missing prompt"}
        market = state.get('market_report', '')
        sentiment = state.get('sentiment_report', '')
        news = state.get('news_report', '')
        fundamentals = state.get('fundamentals_report', '')
        inv_plan = state.get('investment_plan', '')
        consultant = state.get('consultant_review', '')
        trader = state.get('trader_investment_plan', '')
        risk = state.get('risk_debate_state', {}).get('history', '')

        # Red-flag pre-screening results
        pre_screening_result = state.get('pre_screening_result', 'N/A')
        red_flags_detected = state.get('red_flags_detected', [])
        quality_note = state.get('fundamentals_quality_note', '')

        logger.info("pm_inputs", has_market=bool(market), has_sentiment=bool(sentiment), has_news=bool(news), has_fundamentals=bool(fundamentals), has_consultant=bool(consultant), has_datablock="DATA_BLOCK" in fundamentals if fundamentals else False, fund_len=len(fundamentals) if fundamentals else 0)

        # Include consultant review in context (if available)
        consultant_section = f"""\n\nEXTERNAL CONSULTANT REVIEW (Cross-Validation):\n{consultant if consultant else 'N/A (consultant disabled or unavailable)'}"""

        # Include red-flag pre-screening results (critical safety gate)
        red_flag_section = f"""\n\nRED-FLAG PRE-SCREENING:\nPre-Screening Result: {pre_screening_result}"""
        if red_flags_detected:
            red_flag_list = '\n'.join([f"  - {flag}" for flag in red_flags_detected])
            red_flag_section += f"\nRed Flags Detected:\n{red_flag_list}"
        else:
            red_flag_section += f"\nRed Flags Detected: None"
        if quality_note:
            red_flag_section += f"\nNote: {quality_note}"

        all_context = f"""MARKET ANALYST REPORT:\n{market if market else 'N/A'}\n\nSENTIMENT ANALYST REPORT:\n{sentiment if sentiment else 'N/A'}\n\nNEWS ANALYST REPORT:\n{news if news else 'N/A'}\n\nFUNDAMENTALS ANALYST REPORT:\n{fundamentals if fundamentals else 'N/A'}{red_flag_section}\n\nRESEARCH MANAGER RECOMMENDATION:\n{inv_plan if inv_plan else 'N/A'}{consultant_section}\n\nTRADER PROPOSAL:\n{trader if trader else 'N/A'}\n\nRISK TEAM DEBATE:\n{risk if risk else 'N/A'}"""
        prompt = f"""{agent_prompt.system_message}\n\n{all_context}\n\nMake Final Decision."""
        try:
            response = await invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_prompt.agent_name
            )
            return {"final_trade_decision": response.content}
        except Exception as e:
            logger.error(f"PM error: {str(e)}")
            return {"final_trade_decision": f"Error: {str(e)}"}
    return pm_node

def create_consultant_node(llm, agent_key: str = "consultant") -> Callable:
    """
    Factory function creating external consultant node for cross-validation.

    Uses a different LLM (OpenAI) to review Gemini's analysis outputs and detect
    biases, groupthink, and factual errors that internal agents may miss.

    Args:
        llm: LLM instance (typically OpenAI ChatGPT for cross-validation)
        agent_key: Agent key for prompt lookup (default: "consultant")

    Returns:
        Async function compatible with LangGraph StateGraph.add_node()
    """
    async def consultant_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt(agent_key)
        if not agent_prompt:
            logger.error(f"Missing prompt for consultant: {agent_key}")
            return {"consultant_review": "Error: Missing consultant prompt configuration"}

        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)

        context = get_context_from_config(config)
        current_date = context.trade_date if context else datetime.now().strftime("%Y-%m-%d")

        # Assemble complete context (everything the Research Manager saw + the synthesis)
        # Safely extract debate history (handle None/missing debate state)
        debate_state = state.get('investment_debate_state')
        debate_history = 'N/A'
        if debate_state and isinstance(debate_state, dict):
            debate_history = debate_state.get('history', 'N/A')
        elif debate_state is None:
            # DIAGNOSTIC: Log when debate state is unexpectedly None
            # This shouldn't happen in normal execution (consultant runs after debate)
            # If this occurs, it indicates a potential LangGraph state propagation issue
            ticker = state.get('company_of_interest', 'UNKNOWN')
            logger.error(
                "consultant_received_none_debate_state",
                ticker=ticker,
                message="Consultant node received None debate state - this may indicate a graph execution bug or fast-fail path issue"
            )
            debate_history = '[SYSTEM DIAGNOSTIC: Debate state unexpectedly None. This may indicate the debate was skipped (fast-fail path) or a state propagation issue. Consultant cross-validation may be limited without debate context.]'

        all_context = f"""
=== ANALYST REPORTS (SOURCE DATA) ===

MARKET ANALYST REPORT:
{state.get('market_report', 'N/A')}

SENTIMENT ANALYST REPORT:
{state.get('sentiment_report', 'N/A')}

NEWS ANALYST REPORT:
{state.get('news_report', 'N/A')}

FUNDAMENTALS ANALYST REPORT:
{state.get('fundamentals_report', 'N/A')}

=== BULL/BEAR DEBATE HISTORY ===

{debate_history}

=== RESEARCH MANAGER SYNTHESIS ===

{state.get('investment_plan', 'N/A')}

=== RED FLAGS (Pre-Screening Results) ===

Red Flags Detected: {state.get('red_flags', [])}
Pre-Screening Result: {state.get('pre_screening_result', 'UNKNOWN')}
"""

        prompt = f"""{agent_prompt.system_message}

ANALYSIS DATE: {current_date}
TICKER: {ticker}
COMPANY: {company_name}

{all_context}

Provide your independent consultant review."""

        try:
            # Use rate limit handling wrapper for robustness
            response = await invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_prompt.agent_name
            )

            logger.info(
                "consultant_review_complete",
                ticker=ticker,
                review_length=len(response.content),
                has_errors="ERROR" in response.content.upper() or "FAIL" in response.content.upper()
            )

            return {"consultant_review": response.content}

        except Exception as e:
            logger.error(f"Consultant node error for {ticker}: {str(e)}")
            # Return error but don't block the graph
            return {"consultant_review": f"Consultant Review Error: {str(e)}\n\nNote: Analysis will proceed without external validation."}

    return consultant_node


def create_state_cleaner_node() -> Callable:
    async def clean_state(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        context = get_context_from_config(config)
        ticker = context.ticker if context else state.get("company_of_interest", "UNKNOWN")

        logger.debug(
            "State cleaner running",
            context_ticker=context.ticker if context else None,
            state_ticker=state.get("company_of_interest"),
            final_ticker=ticker
        )

        return {
            "messages": [HumanMessage(content=f"Analyze {ticker}")],
            "tools_called": state.get("tools_called", {}),
            "company_of_interest": ticker
        }

    return clean_state


# --- Red-Flag Detection System ---

def create_financial_health_validator_node() -> Callable:
    """
    Factory function creating a pre-screening validator node to catch extreme financial risks
    before proceeding to bull/bear debate.

    This validator implements a "red-flag detection" pattern to save token costs and enforce
    financial discipline. Uses deterministic threshold-based logic from RedFlagDetector.

    Why code-driven instead of LLM-driven:
    - Exact thresholds required (D/E > 500%, not "very high")
    - Fast-fail pattern (avoid LLM calls for doomed stocks)
    - Reliability (no hallucination risk on number parsing)
    - Cost savings (~60% token reduction for rejected stocks)

    Architecture integration:
    - Runs AFTER Fundamentals Analyst (has data)
    - Runs BEFORE Bull/Bear Debate (saves cost if doomed)
    - Sets state.pre_screening_result = "REJECT" | "PASS"
    - Graph routing: REJECT → Portfolio Manager (skip debate)
                     PASS → Bull Researcher (normal flow)

    Returns:
        Async function compatible with LangGraph StateGraph.add_node()
    """
    async def financial_health_validator_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Pre-screening layer to catch extreme financial risks before detailed scoring.

        Delegates to RedFlagDetector for deterministic validation logic.

        Args:
            state: Current agent state with fundamentals_report populated
            config: Runtime configuration (not currently used)

        Returns:
            Updated state dict with:
            - red_flags: List of detected red flags (severity, type, detail, action)
            - pre_screening_result: "REJECT" if any AUTO_REJECT flags, else "PASS"
        """
        from src.validators.red_flag_detector import RedFlagDetector

        fundamentals_report = state.get('fundamentals_report', '')
        
        # --- FIX: DEFENSIVE HANDLING FOR LIST STATE ACCUMULATION ---
        # LangGraph can sometimes pass the accumulated list of state updates
        # instead of the reduced string. If we get a list, we take the last element
        # which represents the most recent/final report.
        if isinstance(fundamentals_report, list):
            fundamentals_report = fundamentals_report[-1] if fundamentals_report else ""

        ticker = state.get('company_of_interest', 'UNKNOWN')
        company_name = state.get('company_name', ticker)

        quiet_mode = os.environ.get("QUIET_MODE", "false").lower() == "true"

        # Graceful handling of missing fundamentals
        if not fundamentals_report:
            logger.warning(
                "validator_no_fundamentals",
                ticker=ticker,
                message="No fundamentals report available - skipping pre-screening"
            )
            return {
                'red_flags': [],
                'pre_screening_result': 'PASS'
            }

        # Extract sector classification from fundamentals report
        sector = RedFlagDetector.detect_sector(fundamentals_report)

        # Extract metrics from DATA_BLOCK
        metrics = RedFlagDetector.extract_metrics(fundamentals_report)

        # Log extracted metrics (unless in quiet mode)
        if not quiet_mode:
            logger.info(
                "validator_extracted_metrics",
                ticker=ticker,
                sector=sector.value,
                debt_to_equity=metrics.get('debt_to_equity'),
                fcf=metrics.get('fcf'),
                net_income=metrics.get('net_income'),
                interest_coverage=metrics.get('interest_coverage'),
                adjusted_health_score=metrics.get('adjusted_health_score')
            )

        # Apply sector-aware red-flag detection logic
        red_flags, pre_screening_result = RedFlagDetector.detect_red_flags(metrics, ticker, sector)

        # Log results
        if pre_screening_result == 'REJECT':
            logger.warning(
                "pre_screening_rejected",
                ticker=ticker,
                company_name=company_name,
                red_flags_count=len(red_flags),
                flag_types=[f['type'] for f in red_flags],
                message=f"REJECTED: {ticker} ({company_name}) failed pre-screening due to {len(red_flags)} critical red flag(s)"
            )
        elif red_flags:
            logger.info(
                "pre_screening_warnings",
                ticker=ticker,
                warnings_count=len(red_flags),
                message=f"{ticker} has {len(red_flags)} warning(s) but passed pre-screening"
            )
        else:
            if not quiet_mode:
                logger.info(
                    "pre_screening_passed",
                    ticker=ticker,
                    message=f"{ticker} passed pre-screening validation"
                )

        return {
            'red_flags': red_flags,
            'pre_screening_result': pre_screening_result
        }

    return financial_health_validator_node