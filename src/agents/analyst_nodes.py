from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import RunnableConfig

from src.data_block_utils import (
    detect_legacy_data_block_shape,
    extract_last_data_block,
    has_parseable_data_block,
    has_parseable_fenced_block,
    normalize_legacy_data_block_report,
    normalize_structured_block_boundaries,
)
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

_FUNDAMENTALS_RETRY_FORMAT_SUFFIX = """
CRITICAL FORMAT CORRECTION:
Emit the DATA_BLOCK first.
Use exactly these fenced markers:
### --- START DATA_BLOCK ---
...
### --- END DATA_BLOCK ---
Inside DATA_BLOCK, use plain KEY: VALUE lines only.
Do NOT use markdown tables inside DATA_BLOCK.
"""


def _normalize_structured_output(agent_key: str, content: str, ticker: str) -> str:
    """Apply narrow deterministic output repairs for known model-format drift."""
    if agent_key != "fundamentals_analyst":
        return content

    repair_kind = detect_legacy_data_block_shape(content)
    normalized = normalize_legacy_data_block_report(content)
    if normalized != content:
        event = (
            "fundamentals_markdown_table_datablock_repaired"
            if repair_kind == "table"
            else "fundamentals_legacy_datablock_repaired"
        )
        logger.warning(
            event,
            ticker=ticker,
            repair_kind=repair_kind,
            original_has_datablock=has_parseable_fenced_block(content, "DATA_BLOCK"),
            repaired_has_datablock=has_parseable_data_block(normalized),
        )
    boundary_normalized = normalize_structured_block_boundaries(normalized)
    if boundary_normalized != normalized:
        logger.warning(
            "fundamentals_datablock_boundary_repaired",
            ticker=ticker,
            original_has_datablock=has_parseable_fenced_block(content, "DATA_BLOCK"),
            repaired_has_datablock=has_parseable_data_block(boundary_normalized),
        )
    normalized = boundary_normalized
    return normalized


def _should_retry_output(content: str, agent_key: str) -> bool:
    """Return True when the initial output should get one deep-model retry."""
    if support._is_output_insufficient(content, agent_key):
        return True

    return agent_key == "fundamentals_analyst" and not has_parseable_data_block(content)


def _build_retry_invocation_messages(
    invocation_messages: list[Any], agent_key: str, content: str
) -> list[Any]:
    """Add a short corrective suffix for fundamentals format retries only."""
    if agent_key != "fundamentals_analyst" or has_parseable_data_block(content):
        return invocation_messages

    return [
        *invocation_messages,
        HumanMessage(content=_FUNDAMENTALS_RETRY_FORMAT_SUFFIX.strip()),
    ]


def create_analyst_node(
    llm,
    agent_key: str,
    tools: list[Any],
    output_field: str,
    retry_llm: Any | None = None,
    allow_retry: bool = False,
) -> Callable:
    """
    Factory function creating data analyst agent nodes.
    """

    async def analyst_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt(agent_key)
        if not agent_prompt:
            logger.error("missing_prompt", agent=agent_key)
            return failure_artifact(
                output_field,
                f"Could not load prompt for {agent_key}.",
                provider="unknown",
            )

        messages_template = [MessagesPlaceholder(variable_name="messages")]
        prompt_template = ChatPromptTemplate.from_messages(messages_template)
        runnable = (
            prompt_template | llm.bind_tools(tools) if tools else prompt_template | llm
        )

        try:
            prompts_used = state.get("prompts_used", {})
            prompts_used[output_field] = {
                "agent_name": agent_prompt.agent_name,
                "version": agent_prompt.version,
            }

            filtered_messages = message_utils.filter_messages_for_gemini(
                state.get("messages", []), agent_key=agent_key
            )
            msg_types = [type(message).__name__ for message in filtered_messages]
            msg_has_tool_calls = [
                bool(getattr(message, "tool_calls", None))
                for message in filtered_messages
                if hasattr(message, "tool_calls")
            ]
            logger.debug(
                "analyst_filtered_messages",
                agent_key=agent_key,
                total_state_messages=len(state.get("messages", [])),
                filtered_count=len(filtered_messages),
                message_types=msg_types,
                has_tool_calls_list=msg_has_tool_calls,
            )

            context = support.get_context_from_config(config)
            current_date = (
                context.trade_date if context else datetime.now().strftime("%Y-%m-%d")
            )
            ticker = (
                context.ticker
                if context
                else state.get("company_of_interest", "UNKNOWN")
            )
            company_name = state.get("company_name", ticker)
            company_resolved = state.get("company_name_resolved", True)

            extra_context = ""

            if agent_key == "junior_fundamentals_analyst":
                news_report = state.get("news_report", "")
                if news_report:
                    extra_context = (
                        "\n\n### NEWS CONTEXT (for ADR/analyst search queries)"
                        f"\n{news_report}\n"
                    )

            if agent_key == "fundamentals_analyst":
                raw_data = state.get("raw_fundamentals_data", "")
                foreign_data = state.get("foreign_language_report", "")
                news_report = state.get("news_report", "")

                if raw_data:
                    extra_context = (
                        "\n\n### RAW FINANCIAL DATA FROM JUNIOR ANALYST (Primary Source)"
                        f"\n{raw_data}\n"
                    )
                else:
                    logger.warning(
                        "senior_fundamentals_no_raw_data",
                        ticker=ticker,
                        message="Junior Analyst data not available - this should not happen",
                    )

                if foreign_data:
                    extra_context += (
                        "\n\n### FOREIGN/ALTERNATIVE SOURCE DATA (Cross-Reference)"
                        "\nNote: Use this data to fill gaps in Junior Analyst data. "
                        "Prioritize Junior's data when both sources have the same metric.\n"
                        f"{foreign_data}\n"
                    )
                    logger.info(
                        "senior_fundamentals_has_foreign_data",
                        ticker=ticker,
                        foreign_data_length=len(foreign_data),
                    )
                else:
                    logger.info(
                        "senior_fundamentals_no_foreign_data",
                        ticker=ticker,
                        message="Foreign Language Analyst data not available - proceeding with Junior data only",
                    )

                if news_report:
                    news_highlights = support.extract_news_highlights(news_report)
                    extra_context += (
                        "\n\n### NEWS HIGHLIGHTS (for Qualitative Growth Scoring)"
                        f"\n{news_highlights}\n"
                    )
                else:
                    logger.info(
                        "senior_fundamentals_no_news",
                        ticker=ticker,
                        message="News report not yet available (parallel execution) - proceeding without news context",
                    )

                conflict_report = support.compute_data_conflicts(raw_data, foreign_data)
                if conflict_report:
                    extra_context += conflict_report
                    logger.info(
                        "senior_fundamentals_conflicts_detected",
                        ticker=ticker,
                        conflict_count=conflict_report.count("\n- "),
                    )

                legal_report = state.get("legal_report", "")
                if legal_report:
                    extra_context += (
                        "\n\n### LEGAL/TAX RISK ASSESSMENT (From Legal Counsel)"
                        "\nUse this to inform your PFIC_RISK assessment in DATA_BLOCK. "
                        "If Legal Counsel found PFIC disclosure (pfic_status: PROBABLE), set PFIC_RISK: MEDIUM or HIGH. "
                        "If no disclosure found in high-risk sector (pfic_status: UNCERTAIN), set PFIC_RISK: MEDIUM.\n"
                        f"{legal_report}\n"
                    )
                    logger.info(
                        "senior_fundamentals_has_legal_data",
                        ticker=ticker,
                        legal_data_length=len(legal_report),
                    )
                else:
                    logger.info(
                        "senior_fundamentals_no_legal_data",
                        ticker=ticker,
                        message="Legal Counsel data not yet available - proceeding without legal context",
                    )

            if agent_key == "news_analyst":
                try:
                    from src.memory import create_macro_events_store
                    from src.retrospective import _get_ticker_suffix

                    macro_store = create_macro_events_store()
                    if macro_store.available:
                        region = _get_ticker_suffix(ticker)
                        events = macro_store.get_active_events(
                            region_filter=region or None
                        )
                        if events:
                            lines = ["### MACRO EVENT CONTEXT (portfolio-detected)"]
                            for event in events[:2]:
                                lines.append(
                                    f"- {event.event_date} | {event.impact} | "
                                    f"{event.scope}: {event.news_headline}"
                                )
                                if event.news_detail:
                                    lines.append(f"  {event.news_detail}")
                            lines.append(
                                "Instruction: Determine if this equity is an "
                                "'Innocent Bystander' (dropped due to the macro event, "
                                "fundamentals intact -> OPPORTUNITY) or "
                                "'Structurally Impaired' (business model affected -> EXIT). "
                                "Ignore if event is inapplicable to this region/sector."
                            )
                            extra_context += "\n\n" + "\n".join(lines) + "\n"
                            logger.info(
                                "macro_events_injected",
                                ticker=ticker,
                                count=len(events[:2]),
                            )
                except Exception as exc:
                    logger.debug("macro_events_injection_failed", error=str(exc))

            full_system_instruction = (
                f"{agent_prompt.system_message}\n\n"
                f"Date: {support._format_date_with_fy_hint(current_date)}\n"
                f"Ticker: {ticker}\n"
                f"{support._company_line(company_name, company_resolved)}\n"
                f"{support.get_analysis_context(ticker)}"
                f"{extra_context}"
            )
            invocation_messages = [
                SystemMessage(content=full_system_instruction),
                *filtered_messages,
            ]

            response = await agent_runtime.invoke_with_rate_limit_handling(
                runnable,
                {"messages": invocation_messages},
                context=agent_prompt.agent_name,
                provider=support.infer_provider_name(llm),
                model_name=support.get_model_name(llm),
            )
            response.name = agent_key

            new_state = {
                "sender": agent_key,
                "messages": [response],
                "prompts_used": prompts_used,
            }

            tool_calls = getattr(response, "tool_calls", None)
            has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
            logger.info(
                "analyst_response_details",
                agent_key=agent_key,
                content_type=type(response.content).__name__,
                content_len=len(response.content) if response.content else 0,
                tool_calls_count=len(tool_calls) if isinstance(tool_calls, list) else 0,
                has_tool_calls=has_tool_calls,
            )

            if has_tool_calls:
                return new_state

            content_str = message_utils.extract_string_content(response.content)
            content_str = _normalize_structured_output(agent_key, content_str, ticker)

            if (
                allow_retry
                and retry_llm is not None
                and _should_retry_output(content_str, agent_key)
            ):
                logger.warning(
                    "analyst_retry_with_deep_thinking",
                    agent_key=agent_key,
                    ticker=ticker,
                    original_length=len(content_str),
                    has_datablock=has_parseable_data_block(content_str),
                    message="Insufficient or unparseable output from quick LLM, retrying once with deep thinking",
                )
                retry_messages = _build_retry_invocation_messages(
                    invocation_messages, agent_key, content_str
                )
                retry_runnable = (
                    prompt_template | retry_llm.bind_tools(tools)
                    if tools
                    else prompt_template | retry_llm
                )

                try:
                    retry_response = (
                        await agent_runtime.invoke_with_rate_limit_handling(
                            retry_runnable,
                            {"messages": retry_messages},
                            context=f"{agent_prompt.agent_name} (RETRY-HIGH)",
                            provider=support.infer_provider_name(retry_llm),
                            model_name=support.get_model_name(retry_llm),
                        )
                    )
                    retry_response.name = agent_key
                    retry_content_str = message_utils.extract_string_content(
                        retry_response.content
                    )
                    retry_content_str = _normalize_structured_output(
                        agent_key, retry_content_str, ticker
                    )
                    retry_tool_calls = getattr(retry_response, "tool_calls", None)
                    retry_has_tool_calls = (
                        isinstance(retry_tool_calls, list) and len(retry_tool_calls) > 0
                    )

                    if retry_has_tool_calls:
                        new_state["messages"] = [retry_response]
                        logger.info(
                            "analyst_retry_produced_tool_calls",
                            agent_key=agent_key,
                            ticker=ticker,
                        )
                        return new_state

                    logger.info(
                        "analyst_retry_complete",
                        agent_key=agent_key,
                        ticker=ticker,
                        original_length=len(content_str),
                        retry_length=len(retry_content_str),
                        retry_has_datablock=has_parseable_data_block(retry_content_str),
                        retry_improved=len(retry_content_str) > len(content_str),
                    )
                    content_str = retry_content_str
                    response = retry_response
                except Exception as retry_error:
                    logger.error(
                        "analyst_retry_failed",
                        agent_key=agent_key,
                        ticker=ticker,
                        error=str(retry_error),
                    )

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

            validation = validate_required_output(agent_key, content_str)
            log_output_diagnostics(
                agent_key=agent_key,
                ticker=ticker,
                runnable=llm if response is not None else llm,
                response=response,
                content=content_str,
                truncated=trunc_info["truncated"],
                validation=validation if validation["checks"] else None,
            )
            if should_fail_closed(
                agent_key,
                validation=validation,
                truncated=trunc_info["truncated"],
                content=content_str,
            ):
                logger.error(
                    "analyst_invalid_structure",
                    agent=agent_key,
                    ticker=ticker,
                    missing_sections=validation["missing"],
                )
                result = failure_artifact(
                    output_field,
                    f"{agent_key} output missing required structure",
                    provider=support.infer_provider_name(llm),
                    fallback_content=content_str,
                )
                new_state.update(result)
                return new_state

            new_state.update(
                success_artifact(
                    output_field,
                    content_str,
                    provider=support.infer_provider_name(llm),
                )
            )

            if agent_key == "fundamentals_analyst":
                logger.info(
                    "fundamentals_output",
                    has_datablock=has_parseable_data_block(content_str),
                    length=len(content_str),
                )
            return new_state
        except Exception as exc:
            logger.error(
                "analyst_node_error", output_field=output_field, error=str(exc)
            )
            error_message = AIMessage(content=f"Error: {str(exc)}")
            error_message.name = agent_key
            result = failure_artifact(
                output_field,
                exc,
                provider=support.infer_provider_name(llm),
            )
            result["messages"] = [error_message]
            return result

    return analyst_node


def create_valuation_calculator_node(llm) -> Callable:
    """
    Factory function creating Valuation Calculator node for chart generation.
    """

    async def valuation_calculator_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("valuation_calculator")
        if not agent_prompt:
            logger.error("missing_prompt", agent="valuation_calculator")
            return failure_artifact(
                "valuation_params",
                "Missing valuation_calculator prompt",
                provider="unknown",
            )

        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)
        fundamentals_report = state.get("fundamentals_report", "")

        if not isinstance(fundamentals_report, str):
            fundamentals_report = message_utils.extract_string_content(
                fundamentals_report
            )

        data_block = extract_last_data_block(fundamentals_report, include_markers=True)
        if not data_block:
            logger.warning(
                "valuation_calculator_no_datablock",
                ticker=ticker,
                message="No DATA_BLOCK found - skipping valuation params extraction",
            )
            return failure_artifact(
                "valuation_params",
                "DATA_BLOCK missing",
                provider=support.infer_provider_name(llm),
            )

        prompt = f"""{agent_prompt.system_message}

TICKER: {ticker}
COMPANY: {company_name}

DATA_BLOCK:
{data_block}

Extract valuation parameters and output in the required format."""

        try:
            response = await agent_runtime.invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_prompt.agent_name,
                provider=support.infer_provider_name(llm),
                model_name=support.get_model_name(llm),
            )
            content_str = message_utils.extract_string_content(response.content)
            logger.info(
                "valuation_calculator_complete",
                ticker=ticker,
                has_params_block=has_parseable_fenced_block(
                    content_str, "VALUATION_PARAMS"
                ),
                content_length=len(content_str),
            )
            return success_artifact(
                "valuation_params",
                content_str,
                provider=support.infer_provider_name(llm),
            )
        except Exception as exc:
            logger.error("valuation_calculator_error", ticker=ticker, error=str(exc))
            return failure_artifact(
                "valuation_params",
                exc,
                provider=support.infer_provider_name(llm),
            )

    return valuation_calculator_node
