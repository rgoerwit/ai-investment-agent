from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog
from langchain_core.messages import HumanMessage
from langgraph.types import RunnableConfig

from src.runtime_diagnostics import failure_artifact, success_artifact

from . import message_utils, support
from . import runtime as agent_runtime
from .output_validation import (
    log_output_diagnostics,
    should_fail_closed,
    validate_required_output,
)
from .state import AgentState

logger = structlog.get_logger(__name__)

_STRICT_RM_ADDENDUM = """
---
## STRICT MODE — Research Manager Instruction

You are operating in STRICT mode. Apply this lens when synthesizing analyst inputs:

1. **Evidence over narrative**: Weight your synthesis toward concrete, documented facts
   (filed cash flows, signed contracts, observable margin trends) over projections or
   "could benefit from" language. If the bull case relies primarily on potential rather
   than demonstrated momentum, frame your synthesis toward DO_NOT_INITIATE.

2. **Near-term catalyst requirement**: A BUY recommendation requires at least one
   high-probability catalyst already in motion (≤12 months). "Eventual" or "possible"
   catalysts do not qualify in strict mode.

3. **Data vacuum discipline**: If the combined analyst reports show significant data gaps
   (missing OCF, unknown ownership structure, no analyst coverage data), flag them
   explicitly and weight toward caution — do not paper over gaps with qualitative reasoning.

4. **Bear argument weighting**: Give bear arguments proportionally more weight than in
   normal mode. The burden of proof is on the bull case in strict mode.
"""


def create_researcher_node(
    llm, memory: Any | None, agent_key: str, round_num: int = 1
) -> Callable:
    """
    Create a researcher node for Bull/Bear debate.
    """
    is_bull = agent_key == "bull_researcher"
    researcher_type = "bull" if is_bull else "bear"
    opponent_type = "bear" if is_bull else "bull"

    async def researcher_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt(agent_key)
        if not agent_prompt:
            logger.error("missing_prompt", agent=agent_key)
            field_name = f"{researcher_type}_round{round_num}"
            return {
                "investment_debate_state": {
                    field_name: f"[SYSTEM]: Error - Missing prompt for {agent_key}.",
                    "count": state.get("investment_debate_state", {}).get("count", 0)
                    + 1,
                }
            }

        market_report = state.get("market_report", "N/A")
        sentiment_report = state.get("sentiment_report", "N/A")
        news_report = state.get("news_report", "N/A")
        fundamentals_report = state.get("fundamentals_report", "N/A")
        reports = f"""MARKET ANALYST REPORT:
{support.summarize_for_pm(market_report, "market", 1800) if market_report != "N/A" else "N/A"}

SENTIMENT ANALYST REPORT:
{support.summarize_for_pm(sentiment_report, "sentiment", 1200) if sentiment_report != "N/A" else "N/A"}

NEWS ANALYST REPORT:
{support.summarize_for_pm(news_report, "news", 1800) if news_report != "N/A" else "N/A"}

FUNDAMENTALS ANALYST REPORT:
{support.summarize_for_pm(fundamentals_report, "fundamentals", 5500) if fundamentals_report != "N/A" else "N/A"}"""

        debate_state = state.get("investment_debate_state", {})
        if round_num == 1:
            debate_history = ""
        else:
            opponent_r1 = debate_state.get(f"{opponent_type}_round1", "")
            own_r1 = debate_state.get(f"{researcher_type}_round1", "")
            debate_history = f"""
=== ROUND 1 ARGUMENTS ===

YOUR ROUND 1 ARGUMENT:
{own_r1}

OPPONENT'S ROUND 1 ARGUMENT (REBUT THIS):
{opponent_r1}

=== END ROUND 1 ===

Now provide your Round 2 rebuttal, addressing the opponent's key points."""

        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)
        company_resolved = state.get("company_name_resolved", True)

        past_insights = ""
        if memory:
            try:
                relevant = await memory.query_similar_situations(
                    f"risks and upside for {ticker}",
                    n_results=3,
                    metadata_filter={"ticker": ticker},
                )
                if relevant:
                    past_insights = (
                        f"\n\nPAST MEMORY INSIGHTS (STRICTLY FOR {ticker}):\n"
                        + "\n".join([result["document"] for result in relevant])
                    )
                else:
                    logger.info("memory_no_exact_match", ticker=ticker)
            except Exception as exc:
                logger.error("memory_retrieval_failed", ticker=ticker, error=str(exc))

        lessons_text = ""
        try:
            from src.retrospective import (
                create_lessons_memory,
                format_lessons_for_injection,
            )

            lessons_memory = create_lessons_memory()
            sector = support._extract_sector_from_state(state)
            lessons_text = await format_lessons_for_injection(
                lessons_memory, ticker, sector
            )
            if lessons_text:
                logger.info(
                    "lessons_injected",
                    agent=agent_key,
                    ticker=ticker,
                    lessons_length=len(lessons_text),
                )
            else:
                logger.debug("no_lessons_available", agent=agent_key, ticker=ticker)
        except Exception as exc:
            logger.warning("lessons_injection_failed", agent=agent_key, error=str(exc))

        unresolved_warning = (
            "" if company_resolved else f"\n{support._UNRESOLVED_NAME_WARNING}"
        )
        negative_constraint = f"""
CRITICAL INSTRUCTION:
You are analyzing **{ticker} ({company_name})**.{unresolved_warning}
If the provided context or memory contains information about a different company, you MUST IGNORE IT.
Only use data explicitly related to {ticker} ({company_name}).
"""

        round_instruction = (
            "Provide your initial argument."
            if round_num == 1
            else "Provide your rebuttal to the opponent's Round 1 argument."
        )

        context_block = past_insights
        if lessons_text:
            context_block += f"\n\n{lessons_text}"

        prompt = (
            f"{agent_prompt.system_message}\n{negative_constraint}\n\nREPORTS:\n"
            f"{reports}\n{context_block}\n\nDEBATE CONTEXT:\n{debate_history}\n\n"
            f"{round_instruction}"
        )

        try:
            response = await agent_runtime.invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=f"{agent_prompt.agent_name} R{round_num}",
                provider=support.infer_provider_name(llm),
                model_name=support.get_model_name(llm),
            )
            content_str = message_utils.extract_string_content(response.content)
            from src.utils import detect_truncation

            trunc_info = detect_truncation(content_str, agent=agent_key)
            if trunc_info["truncated"]:
                logger.warning(
                    "agent_output_truncated",
                    agent=agent_key,
                    ticker=ticker,
                    source=trunc_info["source"],
                    marker=trunc_info["marker"],
                    confidence=trunc_info["confidence"],
                    output_len=len(content_str),
                )
            log_output_diagnostics(
                agent_key=agent_key,
                ticker=ticker,
                runnable=llm,
                response=response,
                content=content_str,
                truncated=trunc_info["truncated"],
                validation=None,
            )
            argument = f"{agent_prompt.agent_name} (Round {round_num}): {content_str}"
            field_name = f"{researcher_type}_round{round_num}"

            logger.info(
                "researcher_completed",
                agent=agent_key,
                round=round_num,
                field=field_name,
                content_length=len(content_str),
            )

            return {"investment_debate_state": {field_name: argument}}
        except Exception as exc:
            logger.error(
                "researcher_error",
                agent=agent_key,
                round=round_num,
                error=str(exc),
            )
            field_name = f"{researcher_type}_round{round_num}"
            return {
                "investment_debate_state": {
                    field_name: (
                        f"[SYSTEM ERROR]: {agent_key} R{round_num} failed - {str(exc)}"
                    ),
                }
            }

    return researcher_node


def create_research_manager_node(
    llm, memory: Any | None, strict_mode: bool = False
) -> Callable:
    async def research_manager_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("research_manager")
        if not agent_prompt:
            return {"investment_plan": "Error: Missing prompt"}

        debate = state.get("investment_debate_state", {})
        value_trap = state.get("value_trap_report", "N/A")
        field_sources = support.extract_field_sources_from_messages(
            state.get("messages", [])
        )
        attribution_note = ""
        if field_sources:
            sources_used = sorted(set(field_sources.values()))
            attribution_note = (
                "\n\n### DATA PROVENANCE NOTE\n"
                f"Fundamentals sourced from: {', '.join(sources_used)}. "
                "News may reflect more recent periods (e.g., Q3 headlines vs TTM API data). "
                "When Bull/Bear cite conflicting figures, check if they reference different time periods."
            )

        market_report = state.get("market_report", "N/A")
        sentiment_report = state.get("sentiment_report", "N/A")
        news_report = state.get("news_report", "N/A")
        fundamentals_report = state.get("fundamentals_report", "N/A")
        bull_history = debate.get("bull_history", "N/A")
        bear_history = debate.get("bear_history", "N/A")
        all_reports = f"""MARKET ANALYST REPORT:\n{support.summarize_for_pm(market_report, "market", 1800) if market_report != "N/A" else "N/A"}\n\nSENTIMENT ANALYST REPORT:\n{support.summarize_for_pm(sentiment_report, "sentiment", 1200) if sentiment_report != "N/A" else "N/A"}\n\nNEWS ANALYST REPORT:\n{support.summarize_for_pm(news_report, "news", 1800) if news_report != "N/A" else "N/A"}\n\nFUNDAMENTALS ANALYST REPORT:\n{support.summarize_for_pm(fundamentals_report, "fundamentals", 6000) if fundamentals_report != "N/A" else "N/A"}{attribution_note}\n\nVALUE TRAP ANALYSIS:\n{support.summarize_for_pm(value_trap, "value_trap", 2200) if value_trap != "N/A" else "N/A"}\n\nBULL RESEARCHER:\n{support.summarize_for_pm(bull_history, "research", 2500) if bull_history != "N/A" else "N/A"}\n\nBEAR RESEARCHER:\n{support.summarize_for_pm(bear_history, "research", 2500) if bear_history != "N/A" else "N/A"}"""
        system_msg = agent_prompt.system_message
        if strict_mode:
            system_msg += _STRICT_RM_ADDENDUM
        prompt = f"{system_msg}\n\n{all_reports}\n\nProvide Investment Plan."

        try:
            response = await agent_runtime.invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_prompt.agent_name,
                provider=support.infer_provider_name(llm),
                model_name=support.get_model_name(llm),
            )
            content_str = message_utils.extract_string_content(response.content)

            from src.utils import detect_truncation

            trunc_info = detect_truncation(content_str, agent="research_manager")
            if trunc_info["truncated"]:
                logger.warning(
                    "agent_output_truncated",
                    agent="research_manager",
                    ticker=state.get("company_of_interest", "UNKNOWN"),
                    source=trunc_info["source"],
                    marker=trunc_info["marker"],
                    confidence=trunc_info["confidence"],
                    output_len=len(content_str),
                )

            validation = validate_required_output("research_manager", content_str)
            log_output_diagnostics(
                agent_key="research_manager",
                ticker=state.get("company_of_interest", "UNKNOWN"),
                runnable=llm,
                response=response,
                content=content_str,
                truncated=trunc_info["truncated"],
                validation=validation,
            )
            if should_fail_closed(
                "research_manager",
                validation=validation,
                truncated=trunc_info["truncated"],
                content=content_str,
            ):
                logger.error(
                    "research_manager_invalid_structure",
                    ticker=state.get("company_of_interest", "UNKNOWN"),
                    missing_sections=validation["missing"],
                )
                return failure_artifact(
                    "investment_plan",
                    "Research Manager output missing required structure",
                    provider=support.infer_provider_name(llm),
                    fallback_content=content_str,
                )

            return success_artifact(
                "investment_plan",
                content_str,
                provider=support.infer_provider_name(llm),
            )
        except Exception as exc:
            return failure_artifact(
                "investment_plan",
                exc,
                provider=support.infer_provider_name(llm),
            )

    return research_manager_node
