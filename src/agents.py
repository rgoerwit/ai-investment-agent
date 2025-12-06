"""
Multi-Agent Trading System - Agent Definitions
FIXED: All data passing issues - agents now receive complete reports.
ADDED: Debug logging to track data flow.
FIXED: Memory retrieval now contextualized per ticker to prevent cross-contamination.
UPDATED (Pass 3 Fixes): Added Negative Constraint to prompts and strict metadata filtering.
"""

import asyncio
from typing import Annotated, List, Dict, Any, Optional, Set, Callable
from typing_extensions import TypedDict
from datetime import datetime

from langgraph.graph import MessagesState
from langgraph.types import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage

import structlog

logger = structlog.get_logger(__name__)

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

    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str
    investment_debate_state: Annotated[InvestDebateState, take_last]
    investment_plan: str
    trader_investment_plan: str
    risk_debate_state: Annotated[RiskDebateState, take_last]
    final_trade_decision: str
    tools_called: Annotated[Dict[str, Set[str]], take_last]
    prompts_used: Annotated[Dict[str, Dict[str, str]], take_last]

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
            response = await runnable.ainvoke({"messages": invocation_messages})
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
                relevant = await memory.query_similar_situations(
                    f"risks and upside for {ticker}",
                    n_results=3,
                    filter_metadata={"ticker": ticker}
                )
                if relevant:
                    past_insights = f"\n\nPAST MEMORY INSIGHTS (STRICTLY FOR {ticker}):\n" + "\n".join([r['document'] for r in relevant])
                else:
                    # If no strict match, do NOT fallback to semantic search to avoid contamination
                    # (e.g. don't return Canon data for HSBC just because they are both 'value stocks')
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
            response = await llm.ainvoke([HumanMessage(content=prompt)])
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
            response = await llm.ainvoke([HumanMessage(content=prompt)])
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
        all_input = f"""MARKET ANALYST REPORT:\n{state.get('market_report', 'N/A')}\n\nFUNDAMENTALS ANALYST REPORT:\n{state.get('fundamentals_report', 'N/A')}\n\nRESEARCH MANAGER PLAN:\n{state.get('investment_plan', 'N/A')}"""
        prompt = f"""{agent_prompt.system_message}\n\n{all_input}\n\nCreate Trading Plan."""
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
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
        prompt = f"""{agent_prompt.system_message}\n\nTRADER PLAN: {state.get('trader_investment_plan')}\n\nProvide risk assessment."""
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
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
        trader = state.get('trader_investment_plan', '')
        risk = state.get('risk_debate_state', {}).get('history', '')
        logger.info("pm_inputs", has_market=bool(market), has_sentiment=bool(sentiment), has_news=bool(news), has_fundamentals=bool(fundamentals), has_datablock="DATA_BLOCK" in fundamentals if fundamentals else False, fund_len=len(fundamentals) if fundamentals else 0)
        all_context = f"""MARKET ANALYST REPORT:\n{market if market else 'N/A'}\n\nSENTIMENT ANALYST REPORT:\n{sentiment if sentiment else 'N/A'}\n\nNEWS ANALYST REPORT:\n{news if news else 'N/A'}\n\nFUNDAMENTALS ANALYST REPORT:\n{fundamentals if fundamentals else 'N/A'}\n\nRESEARCH MANAGER RECOMMENDATION:\n{inv_plan if inv_plan else 'N/A'}\n\nTRADER PROPOSAL:\n{trader if trader else 'N/A'}\n\nRISK TEAM DEBATE:\n{risk if risk else 'N/A'}"""
        prompt = f"""{agent_prompt.system_message}\n\n{all_context}\n\nMake Final Decision."""
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return {"final_trade_decision": response.content}
        except Exception as e:
            logger.error(f"PM error: {str(e)}")
            return {"final_trade_decision": f"Error: {str(e)}"}
    return pm_node

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