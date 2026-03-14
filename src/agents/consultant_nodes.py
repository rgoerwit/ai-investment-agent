from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage as TM
from langgraph.prebuilt import create_react_agent
from langgraph.types import RunnableConfig

from src.runtime_diagnostics import failure_artifact, success_artifact
from src.tooling.runtime import TOOL_SERVICE, ToolInvocation

from . import message_utils, support
from . import runtime as agent_runtime
from .state import AgentState

logger = structlog.get_logger(__name__)


def _build_legal_fallback_report(
    *,
    ticker: str,
    country: str,
    sector: str,
    reason: str,
) -> str:
    return json.dumps(
        {
            "pfic_status": "UNCERTAIN",
            "pfic_evidence": f"Legal counsel unavailable for {ticker}: {reason}",
            "vie_structure": "N/A",
            "vie_evidence": None,
            "cmic_status": "N/A",
            "cmic_evidence": None,
            "other_regulatory_risks": [],
            "country": country,
            "sector": sector,
        }
    )


def create_consultant_node(
    llm, agent_key: str = "consultant", tools: list | None = None
) -> Callable:
    """
    Create external consultant node for cross-validation.
    """
    max_tool_iterations = 3
    max_tool_calls_per_turn = 4

    tools_by_name = {tool.name: tool for tool in tools} if tools else {}
    llm_with_tools = llm.bind_tools(tools) if tools else None

    async def consultant_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, str]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt(agent_key)
        if not agent_prompt:
            logger.error("missing_prompt", agent=agent_key)
            return failure_artifact(
                "consultant_review",
                "Missing consultant prompt configuration",
                provider="unknown",
            )

        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)
        company_resolved = state.get("company_name_resolved", True)

        context = support.get_context_from_config(config)
        current_date = (
            context.trade_date if context else datetime.now().strftime("%Y-%m-%d")
        )

        debate_state = state.get("investment_debate_state")
        debate_history = "N/A"
        if debate_state and isinstance(debate_state, dict):
            debate_history = debate_state.get("history", "N/A")
        elif debate_state is None:
            logger.error(
                "consultant_received_none_debate_state",
                ticker=ticker,
                message="Consultant node received None debate state",
            )
            debate_history = (
                "[SYSTEM DIAGNOSTIC: Debate state unexpectedly None. This may indicate "
                "the debate was skipped or a state propagation issue.]"
            )

        field_sources = support.extract_field_sources_from_messages(
            state.get("messages", [])
        )
        attribution_table = support.format_attribution_table(field_sources)
        conflict_table = support.format_conflict_table(state.get("messages", []))

        market = state.get("market_report", "N/A")
        sentiment = state.get("sentiment_report", "N/A")
        news = state.get("news_report", "N/A")
        fundamentals = state.get("fundamentals_report", "N/A")
        investment_plan = state.get("investment_plan", "N/A")
        auditor = state.get("auditor_report", "N/A")

        all_context = f"""
=== ANALYST REPORTS (SOURCE DATA) ===

MARKET ANALYST REPORT:
{support.summarize_for_pm(market, "market", 2000) if market != "N/A" else "N/A"}

SENTIMENT ANALYST REPORT:
{support.summarize_for_pm(sentiment, "sentiment", 1500) if sentiment != "N/A" else "N/A"}

NEWS ANALYST REPORT:
{support.summarize_for_pm(news, "news", 2000) if news != "N/A" else "N/A"}

FUNDAMENTALS ANALYST REPORT:
{support.summarize_for_pm(fundamentals, "fundamentals", 5000) if fundamentals != "N/A" else "N/A"}
{attribution_table}{conflict_table}
=== BULL/BEAR DEBATE HISTORY ===

{support.summarize_for_pm(debate_history, "debate", 4000) if debate_history != "N/A" else "N/A"}

=== RESEARCH MANAGER SYNTHESIS ===

{support.summarize_for_pm(investment_plan, "research", 4000) if investment_plan != "N/A" else "N/A"}

=== RED FLAGS (Pre-Screening Results) ===

Red Flags Detected: {state.get("red_flags", [])}
Pre-Screening Result: {state.get("pre_screening_result", "UNKNOWN")}

=== INDEPENDENT FORENSIC AUDIT ===
{support.summarize_for_pm(auditor, "auditor", 3000) if auditor != "N/A" else "N/A"}
"""

        company_warning = (
            "" if company_resolved else f"\n{support._UNRESOLVED_NAME_WARNING}"
        )
        prompt = f"""{agent_prompt.system_message}

ANALYSIS DATE: {support._format_date_with_fy_hint(current_date)}
TICKER: {ticker}
COMPANY: {company_name}{company_warning}

{all_context}

Provide your independent consultant review."""

        try:
            messages = [HumanMessage(content=prompt)]
            active_llm = llm_with_tools or llm
            content_str = ""

            for iteration in range(max_tool_iterations + 1):
                response = await agent_runtime.invoke_with_rate_limit_handling(
                    active_llm,
                    messages,
                    context=agent_prompt.agent_name,
                    provider=support.infer_provider_name(active_llm),
                    model_name=support.get_model_name(active_llm),
                )
                tool_calls = getattr(response, "tool_calls", None)
                if (
                    not isinstance(tool_calls, list)
                    or not tool_calls
                    or iteration == max_tool_iterations
                ):
                    content_str = message_utils.extract_string_content(response.content)
                    break

                messages.append(response)
                capped = tool_calls[:max_tool_calls_per_turn]
                if len(tool_calls) > max_tool_calls_per_turn:
                    logger.warning(
                        "consultant_tool_calls_capped",
                        ticker=ticker,
                        requested=len(tool_calls),
                        cap=max_tool_calls_per_turn,
                    )

                for tool_call in capped:
                    tool_fn = tools_by_name.get(tool_call["name"])
                    tool_call_id = tool_call.get("id", tool_call["name"])
                    if tool_fn:
                        try:
                            tool_result = await TOOL_SERVICE.execute(
                                ToolInvocation(
                                    name=tool_call["name"],
                                    args=tool_call["args"],
                                    source="consultant",
                                    agent_key=agent_key,
                                ),
                                runner=lambda args, tool=tool_fn: tool.ainvoke(args),
                            )
                            result = tool_result.value
                        except Exception as tool_err:
                            result = f"TOOL_ERROR: {str(tool_err)}"
                    else:
                        result = f"Unknown tool: {tool_call['name']}"
                    messages.append(TM(content=str(result), tool_call_id=tool_call_id))

                for tool_call in tool_calls[max_tool_calls_per_turn:]:
                    skip_id = tool_call.get("id", f"skip_{tool_call['name']}")
                    messages.append(
                        TM(
                            content="SKIPPED: Too many tool calls in one turn.",
                            tool_call_id=skip_id,
                        )
                    )

                logger.info(
                    "consultant_tool_iteration",
                    ticker=ticker,
                    iteration=iteration + 1,
                    tools_called=[tool_call["name"] for tool_call in capped],
                )

            if not content_str:
                response = await agent_runtime.invoke_with_rate_limit_handling(
                    llm,
                    messages,
                    context=agent_prompt.agent_name,
                    provider=support.infer_provider_name(llm),
                    model_name=support.get_model_name(llm),
                )
                content_str = message_utils.extract_string_content(response.content)

            from src.utils import detect_truncation

            trunc_info = detect_truncation(content_str, agent="consultant")
            if trunc_info["truncated"]:
                logger.warning(
                    "agent_output_truncated",
                    agent="consultant",
                    ticker=ticker,
                    source=trunc_info["source"],
                    marker=trunc_info["marker"],
                    confidence=trunc_info["confidence"],
                    output_len=len(content_str),
                )

            logger.info(
                "consultant_review_complete",
                ticker=ticker,
                review_length=len(content_str),
                has_errors="ERROR" in content_str.upper()
                or "FAIL" in content_str.upper(),
                truncated=trunc_info["truncated"],
            )
            return success_artifact(
                "consultant_review",
                content_str,
                provider=support.infer_provider_name(llm),
            )
        except Exception as exc:
            logger.error("consultant_node_error", ticker=ticker, error=str(exc))
            return failure_artifact(
                "consultant_review",
                exc,
                provider=support.infer_provider_name(llm),
            )

    return consultant_node


def create_legal_counsel_node(llm, tools: list) -> Callable:
    """
    Create Legal Counsel node for PFIC or VIE detection.
    """

    async def legal_counsel_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, str]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("legal_counsel")
        if not agent_prompt:
            logger.error("missing_prompt", agent="legal_counsel")
            return failure_artifact(
                "legal_report",
                "Missing legal_counsel prompt",
                provider="unknown",
            )

        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)
        company_resolved = state.get("company_name_resolved", True)

        context = support.get_context_from_config(config)
        current_date = (
            context.trade_date if context else datetime.now().strftime("%Y-%m-%d")
        )

        raw_data = state.get("raw_fundamentals_data", "")
        sector, country = support._extract_sector_country(raw_data)

        company_warning = (
            "" if company_resolved else f"\n{support._UNRESOLVED_NAME_WARNING}"
        )
        human_msg = f"""Analyze legal/tax risks for:
Ticker: {ticker}
Company: {company_name}{company_warning}
Sector: {sector}
Country: {country}
Date: {support._format_date_with_fy_hint(current_date)}

Call the search_legal_tax_disclosures tool with these parameters, then provide your JSON assessment."""

        try:
            agent = create_react_agent(llm, tools)
            result = await agent.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=agent_prompt.system_message),
                        HumanMessage(content=human_msg),
                    ]
                }
            )

            response = result["messages"][-1].content
            response_str = message_utils.extract_string_content(response)

            try:
                parsed = json.loads(response_str)
                logger.info(
                    "legal_counsel_complete",
                    ticker=ticker,
                    pfic_status=parsed.get("pfic_status"),
                    vie_structure=parsed.get("vie_structure"),
                )
                result = success_artifact(
                    "legal_report",
                    response_str,
                    provider=support.infer_provider_name(llm),
                )
                result["sender"] = "legal_counsel"
                return result
            except json.JSONDecodeError:
                json_match = re.search(
                    r'\{[^{}]*"pfic_status"[^{}]*\}', response_str, re.DOTALL
                )
                if json_match:
                    extracted = json_match.group()
                    try:
                        json.loads(extracted)
                        logger.info("legal_counsel_extracted_json", ticker=ticker)
                        result = success_artifact(
                            "legal_report",
                            extracted,
                            provider=support.infer_provider_name(llm),
                        )
                        result["sender"] = "legal_counsel"
                        return result
                    except json.JSONDecodeError:
                        pass

                logger.warning(
                    "legal_counsel_invalid_json",
                    ticker=ticker,
                    response_preview=response_str[:200],
                )
                fallback_report = _build_legal_fallback_report(
                    ticker=ticker,
                    country=country,
                    sector=sector,
                    reason="Invalid JSON response from legal counsel",
                )
                result = failure_artifact(
                    "legal_report",
                    "Invalid JSON response from legal counsel",
                    provider=support.infer_provider_name(llm),
                    fallback_content=fallback_report,
                )
                result["sender"] = "legal_counsel"
                return result
        except Exception as exc:
            logger.error("legal_counsel_error", ticker=ticker, error=str(exc))
            fallback_report = _build_legal_fallback_report(
                ticker=ticker,
                country=country,
                sector=sector,
                reason=str(exc),
            )
            result = failure_artifact(
                "legal_report",
                exc,
                provider=support.infer_provider_name(llm),
                fallback_content=fallback_report,
            )
            result["sender"] = "legal_counsel"
            return result

    return legal_counsel_node


def create_auditor_node(llm, tools: list) -> Callable:
    """
    Create the Global Forensic Auditor node.
    """
    max_tool_output_chars = 63500

    def truncate_tool_outputs_hook(state: dict) -> dict:
        from langchain_core.messages import ToolMessage

        messages = state.get("messages", [])
        modified = []
        for message in messages:
            if isinstance(message, ToolMessage):
                content = (
                    message.content
                    if isinstance(message.content, str)
                    else str(message.content)
                )
                if len(content) > max_tool_output_chars:
                    head_size = 58000
                    tail_size = 5500
                    truncated_chars = len(content) - head_size - tail_size
                    truncated = (
                        content[:head_size]
                        + f"\n\n[...TRUNCATED {truncated_chars:,} chars...]\n"
                        + "[NOTE: Data truncated due to size limits. Partial analysis may still be useful. "
                        + "Key financial metrics may appear in head or tail sections above/below.]\n\n"
                        + content[-tail_size:]
                    )
                    modified.append(
                        ToolMessage(
                            content=truncated,
                            tool_call_id=message.tool_call_id,
                            name=getattr(message, "name", None),
                        )
                    )
                    logger.debug(
                        "auditor_tool_output_truncated",
                        original_len=len(content),
                        truncated_len=len(truncated),
                    )
                else:
                    modified.append(message)
            else:
                modified.append(message)
        return {"llm_input_messages": modified}

    async def auditor_node(state: AgentState, config: RunnableConfig) -> dict[str, str]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("global_forensic_auditor")
        if not agent_prompt:
            logger.error("missing_prompt", agent="global_forensic_auditor")
            return failure_artifact(
                "auditor_report",
                "Missing prompt",
                provider="unknown",
            )

        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)
        company_resolved = state.get("company_name_resolved", True)

        context = support.get_context_from_config(config)
        current_date = (
            context.trade_date if context else datetime.now().strftime("%Y-%m-%d")
        )

        company_warning = (
            "" if company_resolved else f"\n{support._UNRESOLVED_NAME_WARNING}"
        )
        human_msg = f"""Analyze financial statements for:
Ticker: {ticker}
Company: {company_name}{company_warning}
Date: {support._format_date_with_fy_hint(current_date)}

Perform a forensic audit using your tools."""

        logger.info("auditor_start", ticker=ticker)

        try:
            agent = create_react_agent(
                llm,
                tools,
                pre_model_hook=truncate_tool_outputs_hook,
            )
            result = await agent.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=agent_prompt.system_message),
                        HumanMessage(content=human_msg),
                    ]
                },
                config={"recursion_limit": 12},
            )

            response = result["messages"][-1].content
            response_str = message_utils.extract_string_content(response)

            logger.info("auditor_complete", ticker=ticker, length=len(response_str))
            result = success_artifact(
                "auditor_report",
                response_str,
                provider=support.infer_provider_name(llm),
            )
            result["sender"] = "global_forensic_auditor"
            return result
        except Exception as exc:
            error_str = str(exc)
            logger.error("auditor_error", ticker=ticker, error=error_str)

            is_context_error = (
                "context_length_exceeded" in error_str
                or "maximum context length" in error_str
            )
            is_param_error = (
                "does not support" in error_str
                or "Unsupported value" in error_str
                or "invalid_request_error" in error_str
            )

            if is_context_error:
                graceful_msg = f"""## FORENSIC AUDITOR REPORT

**STATUS**: CONTEXT_LIMIT_EXCEEDED

**Reason**: Tool results exceeded capacity even after truncation.

**Recommendation**:
Downstream agents should rely on Fundamentals Analyst DATA_BLOCK (structured APIs: yfinance, FMP, EODHD) as primary source. Independent forensic audit unavailable for {ticker}.

---
FORENSIC_DATA_BLOCK:
STATUS: UNAVAILABLE
META: CONTEXT_LIMIT_EXCEEDED
REASON: Data volume exceeded 128k token limit
VERDICT: Rely on DATA_BLOCK metrics for {ticker}.
"""
                result = failure_artifact(
                    "auditor_report",
                    "Auditor context limit exceeded",
                    provider=support.infer_provider_name(llm),
                    fallback_content=graceful_msg,
                    error_kind="application_error",
                )
                result["sender"] = "global_forensic_auditor"
                return result

            if is_param_error:
                logger.warning(
                    "auditor_param_error_retry",
                    ticker=ticker,
                    error=error_str,
                )
                try:
                    from langchain_openai import ChatOpenAI

                    fallback_llm = ChatOpenAI(
                        model=llm.model_name,
                        timeout=120,
                        max_retries=3,
                        streaming=False,
                        use_responses_api=True,
                        output_version="responses/v1",
                    )
                    agent = create_react_agent(
                        fallback_llm,
                        tools,
                        pre_model_hook=truncate_tool_outputs_hook,
                    )
                    result = await agent.ainvoke(
                        {
                            "messages": [
                                SystemMessage(content=agent_prompt.system_message),
                                HumanMessage(content=human_msg),
                            ]
                        },
                        config={"recursion_limit": 12},
                    )
                    response = result["messages"][-1].content
                    response_str = message_utils.extract_string_content(response)
                    logger.info(
                        "auditor_complete_after_retry",
                        ticker=ticker,
                        length=len(response_str),
                    )
                    result = success_artifact(
                        "auditor_report",
                        response_str,
                        provider=support.infer_provider_name(fallback_llm),
                    )
                    result["sender"] = "global_forensic_auditor"
                    return result
                except Exception as retry_exc:
                    logger.error(
                        "auditor_retry_failed",
                        ticker=ticker,
                        error=str(retry_exc),
                    )

            result = failure_artifact(
                "auditor_report",
                exc,
                provider=support.infer_provider_name(llm),
            )
            result["sender"] = "global_forensic_auditor"
            return result

    return auditor_node
