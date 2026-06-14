from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import RunnableConfig

from src.charts.extractors.valuation_signals import VALUATION_CONTEXT_TOKENS
from src.config import config as settings_config
from src.data_block_utils import (
    detect_legacy_data_block_shape,
    extract_block_text_value,
    extract_last_data_block,
    has_parseable_data_block,
    has_parseable_fenced_block,
    normalize_legacy_data_block_report,
    normalize_structured_block_boundaries,
    replace_or_append_block_line,
)
from src.error_safety import summarize_exception
from src.runtime_config import get_runtime_config
from src.runtime_diagnostics import failure_artifact, success_artifact
from src.tooling.text_boundary import format_untrusted_block

from . import message_utils, support
from . import runtime as agent_runtime
from .fundamentals_reconciler import (
    HORIZON_FIELD_RAW_KEYS,
    append_analyst_coverage_data_quality_note,
    extract_raw_metrics_payload,
    reconcile_high_risk_fields,
)
from .output_limits import cap_state_value
from .output_validation import (
    log_output_diagnostics,
    log_truncation_diagnostic,
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

_QUARANTINED_FORWARD_KEYS = ("PE_RATIO_FORWARD", "PEG_RATIO")
_VALID_ADR_EXCHANGES = {
    "NYSE",
    "NASDAQ",
    "AMEX",
    "OTC",
    "OTC-OTCQX",
    "OTC-OTCQB",
    "OTC-OTCPK",
    "PINK",
}
_AUTHORITATIVE_ADR_DOMAINS = (
    "sec.gov",
    "adr.db.com",
    "adrbny.com",
    "depositaryreceipts.citi.com",
    "citiadr.factsetdigitalsolutions.com",
    "adr.com",
    "jpmadr",
    "markitdigital.com/jpmadr",
    "otcmarkets.com",
)
_UNSPONSORED_ADR_MARKERS = (
    "unsponsored adr",
    "unsp/adr",
    "sponsorship level: unsponsored",
    "unsponsored adr program",
    "multi unsponsored",
)


def _extract_valuation_context(report: str | None) -> str:
    """Return compact valuation-relevant flags that sit outside DATA_BLOCK."""
    if not report:
        return "None"
    lines: list[str] = []
    for raw_line in report.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        upper = line.upper()
        if any(keyword in upper for keyword in VALUATION_CONTEXT_TOKENS):
            lines.append(line[:300])
        if len(lines) >= 8:
            break
    return "\n".join(lines) if lines else "None"


_SPONSORED_ADR_MARKERS = (
    "sponsorship level: sponsored",
    "sponsored level i adr",
    "sponsored level ii adr",
    "sponsored level iii adr",
    "sponsored level 1 adr",
    "sponsored level 2 adr",
    "sponsored level 3 adr",
    "sponsored adr program",
    "sponsored depositary receipt program",
)


def _home_suffix(ticker: str) -> str:
    return ticker[ticker.rfind(".") :].upper() if "." in ticker else ""


def _invalid_adr_routing(body: str, ticker: str) -> bool:
    if extract_block_text_value(body, "ADR_EXISTS").upper() != "YES":
        return False

    adr_ticker = extract_block_text_value(body, "ADR_TICKER").upper()
    adr_exchange = extract_block_text_value(body, "ADR_EXCHANGE").upper()
    suffix = _home_suffix(ticker)
    ticker_root = ticker.upper().split(".", 1)[0]
    ticker_bad = adr_ticker not in {"", "NONE", "N/A"} and (
        adr_ticker == ticker.upper()
        or adr_ticker == ticker_root
        or (suffix and adr_ticker.endswith(suffix))
    )
    exchange_bad = (
        adr_exchange not in {"", "NONE", "N/A"}
        and adr_exchange not in _VALID_ADR_EXCHANGES
    )
    return ticker_bad or exchange_bad


def _classify_otc_adr_evidence(raw_text: str) -> str | None:
    """Classify OTC ADR sponsorship only from explicit authoritative evidence."""
    text = raw_text.lower()
    if any(marker in text for marker in _UNSPONSORED_ADR_MARKERS):
        return "UNSPONSORED"
    has_authoritative_source = any(
        domain in text for domain in _AUTHORITATIVE_ADR_DOMAINS
    )
    if has_authoritative_source and any(
        marker in text for marker in _SPONSORED_ADR_MARKERS
    ):
        return "SPONSORED"
    return None


def _sanitize_fundamentals_output(
    content: str,
    raw_data: str,
    ticker: str,
    foreign_data: str = "",
) -> str:
    if not raw_data or not has_parseable_data_block(content):
        return content

    payload = extract_raw_metrics_payload(raw_data)
    has_structured_payload = bool(payload)

    block_body = extract_last_data_block(content, include_markers=False)
    block_with_markers = extract_last_data_block(content, include_markers=True)
    if block_body is None or block_with_markers is None:
        return content

    updated_body = block_body
    if has_structured_payload:
        updated_body = reconcile_high_risk_fields(updated_body, payload)

    updated_body = append_analyst_coverage_data_quality_note(
        updated_body,
        foreign_data,
    )

    if (
        has_structured_payload
        and payload.get("_split_sensitive_metrics_quarantined") is True
    ):
        for key in _QUARANTINED_FORWARD_KEYS:
            updated_body = replace_or_append_block_line(updated_body, key, "N/A")

    if has_structured_payload and payload.get("_pe_low_anomaly_quarantined") is True:
        updated_body = replace_or_append_block_line(updated_body, "PE_RATIO_TTM", "N/A")
        updated_body = replace_or_append_block_line(updated_body, "PEG_RATIO", "N/A")

    if has_structured_payload:
        for datablock_key, raw_key in HORIZON_FIELD_RAW_KEYS:
            if payload.get(raw_key) is None:
                updated_body = replace_or_append_block_line(
                    updated_body,
                    datablock_key,
                    "N/A",
                )

    latest_quarter_date = payload.get("latest_quarter_date")
    latest_quarter_date_reconciled = (
        has_structured_payload
        and payload.get("_latest_quarter_date_source")
        == "reconciled_most_recent_quarter"
    )
    if (
        latest_quarter_date_reconciled
        and isinstance(latest_quarter_date, str)
        and latest_quarter_date
    ):
        updated_body = replace_or_append_block_line(
            updated_body,
            "LATEST_QUARTER_DATE",
            latest_quarter_date,
        )

    if _invalid_adr_routing(updated_body, ticker):
        for key, value in {
            "ADR_TICKER": "None",
            "ADR_EXCHANGE": "None",
            "ADR_THESIS_IMPACT": "UNCERTAIN",
            "ADR_DATA_QUALITY_NOTE": (
                "Invalid ADR routing fields removed; ADR status unresolved."
            ),
        }.items():
            updated_body = replace_or_append_block_line(
                updated_body,
                key,
                value,
            )
        logger.warning("adr_routing_invalidated", ticker=ticker)

    adr_exchange = extract_block_text_value(updated_body, "ADR_EXCHANGE").upper()
    adr_type = extract_block_text_value(updated_body, "ADR_TYPE").upper()
    if adr_exchange.startswith("OTC") and adr_type == "SPONSORED":
        evidence = _classify_otc_adr_evidence(raw_data)
        replacements: dict[str, str] | None = None
        if evidence == "UNSPONSORED":
            replacements = {
                "ADR_TYPE": "UNSPONSORED",
                "ADR_THESIS_IMPACT": "EMERGING_INTEREST",
                "ADR_DATA_QUALITY_NOTE": (
                    "OTC ADR sponsorship corrected from explicit unsponsored "
                    "evidence."
                ),
            }
            logger.warning("adr_sponsorship_corrected_to_unsponsored", ticker=ticker)
        elif evidence is None:
            replacements = {
                "ADR_TYPE": "UNCERTAIN",
                "ADR_THESIS_IMPACT": "UNCERTAIN",
                "ADR_DATA_QUALITY_NOTE": (
                    "OTC sponsorship claim lacked authoritative evidence; loose "
                    "source language ignored."
                ),
            }
            logger.warning("adr_sponsorship_downgraded_to_uncertain", ticker=ticker)

        if replacements:
            for key, value in replacements.items():
                updated_body = replace_or_append_block_line(
                    updated_body,
                    key,
                    value,
                )

    if updated_body == block_body:
        return content

    logger.warning("fundamentals_datablock_sanitized", ticker=ticker)
    updated_block = (
        "### --- START DATA_BLOCK ---\n"
        f"{updated_body.rstrip()}\n"
        "### --- END DATA_BLOCK ---"
    )
    block_index = content.rfind(block_with_markers)
    if block_index < 0:
        return content
    return (
        content[:block_index]
        + updated_block
        + content[block_index + len(block_with_markers) :]
    )


def _normalize_structured_output(
    agent_key: str,
    content: str,
    ticker: str,
    *,
    raw_data: str = "",
    foreign_data: str = "",
) -> str:
    """Apply narrow deterministic output repairs for known model-format drift."""
    if agent_key != "fundamentals_analyst":
        return content

    repair_kind = detect_legacy_data_block_shape(content)
    normalized = normalize_legacy_data_block_report(content) or content
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
    normalized = boundary_normalized or normalized
    normalized = _sanitize_fundamentals_output(
        normalized,
        raw_data,
        ticker,
        foreign_data=foreign_data,
    )
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


def _build_portfolio_macro_event_context(ticker: str) -> str:
    """Return the existing portfolio-detected macro-event block for News Analyst."""
    try:
        from src.memory import create_macro_events_store
        from src.ticker_policy import get_ticker_suffix

        macro_store = create_macro_events_store()
        if not macro_store.available:
            return ""

        region = get_ticker_suffix(ticker)
        events = macro_store.get_active_events(region_filter=region or None)
        if not events:
            return ""

        lines = ["### PORTFOLIO MACRO EVENT"]
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
        logger.debug("macro_events_injected", ticker=ticker, count=len(events[:2]))
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("macro_events_injection_failed", ticker=ticker, error=str(exc))
        return ""


def _build_regional_macro_context_block(context: Any | None, ticker: str) -> str:
    """Return the cached regional macro brief block for News Analyst."""
    return support.format_macro_context_for_agent(context, audience="news")


def _build_news_macro_extra_context(ticker: str, context: Any | None) -> str:
    """Build deterministic macro context for News Analyst.

    Keep discrete portfolio shocks first and broader regional regime context
    second so the prompt can de-duplicate them rather than treating them as two
    unrelated signals.
    """
    blocks = []
    portfolio_block = _build_portfolio_macro_event_context(ticker)
    portfolio_macro_event_present = bool(portfolio_block)
    if portfolio_block:
        blocks.append(portfolio_block)

    regional_block = _build_regional_macro_context_block(context, ticker)
    regional_macro_context_present = bool(regional_block)
    if regional_block:
        macro_region = getattr(context, "macro_context_region", "GLOBAL") or "GLOBAL"
        macro_status = (
            getattr(context, "macro_context_status", "disabled") or "disabled"
        )
        macro_report = getattr(context, "macro_context_report", "") if context else ""
        logger.debug(
            "macro_context_injected",
            ticker=ticker,
            region=macro_region,
            status=macro_status,
            report_len=len(macro_report),
            agent="news_analyst",
            portfolio_macro_event_present=portfolio_macro_event_present,
            regional_macro_context_present=regional_macro_context_present,
        )
        blocks.append(regional_block)

    if not blocks:
        return ""
    return "\n\n" + "\n\n".join(blocks) + "\n"


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
            trusted_context_instructions = ""
            macro_context_injected_into_news = False

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
                    trusted_context_instructions += (
                        "\nCross-reference foreign or alternative-source data against "
                        "Junior Analyst data. Prioritize Junior data when both sources "
                        "report the same metric.\n"
                    )
                    extra_context += (
                        "\n\n### FOREIGN/ALTERNATIVE SOURCE DATA (Cross-Reference)"
                        f"{foreign_data}\n"
                    )
                    logger.debug(
                        "senior_fundamentals_has_foreign_data",
                        ticker=ticker,
                        foreign_data_length=len(foreign_data),
                    )
                else:
                    logger.debug(
                        "senior_fundamentals_no_foreign_data",
                        ticker=ticker,
                        message="Foreign Language Analyst data not available - proceeding with Junior data only",
                    )

                if news_report:
                    news_highlights = support.extract_news_highlights(
                        news_report,
                        max_chars=5000,
                    )
                    extra_context += (
                        "\n\n### NEWS HIGHLIGHTS (for Qualitative Growth Scoring)"
                        f"\n{news_highlights}\n"
                    )
                else:
                    logger.debug(
                        "senior_fundamentals_no_news",
                        ticker=ticker,
                        message="News report not yet available (parallel execution) - proceeding without news context",
                    )

                conflict_report = support.compute_data_conflicts(raw_data, foreign_data)
                if conflict_report:
                    trusted_context_instructions += (
                        "\nReview the computed data-conflict report below and resolve "
                        "discrepancies conservatively.\n"
                    )
                    extra_context += conflict_report
                    logger.debug(
                        "senior_fundamentals_conflicts_detected",
                        ticker=ticker,
                        conflict_count=conflict_report.count("\n- "),
                    )

                legal_report = state.get("legal_report", "")
                if legal_report:
                    trusted_context_instructions += (
                        "\nUse Legal Counsel output to inform PFIC_RISK in DATA_BLOCK. "
                        "If Legal Counsel found PFIC disclosure (pfic_status: PROBABLE), "
                        "set PFIC_RISK to MEDIUM or HIGH. If no disclosure was found in "
                        "a high-risk sector (pfic_status: UNCERTAIN), set PFIC_RISK to "
                        "at least MEDIUM.\n"
                    )
                    extra_context += (
                        "\n\n### LEGAL/TAX RISK ASSESSMENT (From Legal Counsel)"
                        f"{legal_report}\n"
                    )
                    logger.debug(
                        "senior_fundamentals_has_legal_data",
                        ticker=ticker,
                        legal_data_length=len(legal_report),
                    )
                else:
                    logger.debug(
                        "senior_fundamentals_no_legal_data",
                        ticker=ticker,
                        message="Legal Counsel data not yet available - proceeding without legal context",
                    )

            if agent_key == "news_analyst":
                news_macro_context = _build_news_macro_extra_context(ticker, context)
                extra_context += news_macro_context
                macro_context_injected_into_news = (
                    "### REGIONAL MACRO CONTEXT" in news_macro_context
                )

            core_system_instruction = (
                f"{agent_prompt.system_message}\n\n"
                f"Date: {support._format_date_with_fy_hint(current_date)}\n"
                f"Ticker: {ticker}\n"
                f"{support._company_line(company_name, company_resolved)}\n"
                f"{support.get_analysis_context(ticker)}"
                f"{trusted_context_instructions}"
            )
            invocation_messages: list[BaseMessage] = [
                SystemMessage(content=core_system_instruction),
            ]
            if extra_context:
                invocation_messages.append(
                    HumanMessage(
                        content=format_untrusted_block(
                            extra_context,
                            "SUPPLEMENTARY CONTEXT",
                            provenance="prior analysis stages and external data sources",
                        )
                    )
                )
            invocation_messages.extend(filtered_messages)

            response = await agent_runtime.invoke_with_rate_limit_handling(
                runnable,
                {"messages": invocation_messages},
                context=agent_prompt.agent_name,
                provider=support.infer_provider_name(llm),
                model_name=support.get_model_name(llm),
            )
            response.name = agent_key

            new_state: dict[str, Any] = {
                "sender": agent_key,
                "messages": [response],
                "prompts_used": prompts_used,
            }
            if agent_key == "news_analyst":
                new_state["macro_context_injected_into_news"] = (
                    macro_context_injected_into_news
                )

            tool_calls = getattr(response, "tool_calls", None)
            has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
            logger.debug(
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
            content_str = _normalize_structured_output(
                agent_key,
                content_str,
                ticker,
                raw_data=raw_data if agent_key == "fundamentals_analyst" else "",
                foreign_data=foreign_data
                if agent_key == "fundamentals_analyst"
                else "",
            )

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
                            overall_timeout_seconds=float(
                                get_runtime_config(
                                    settings_config
                                ).llm_call_hard_timeout_seconds
                            ),
                        )
                    )
                    retry_response.name = agent_key
                    retry_content_str = message_utils.extract_string_content(
                        retry_response.content
                    )
                    retry_content_str = _normalize_structured_output(
                        agent_key,
                        retry_content_str,
                        ticker,
                        raw_data=raw_data
                        if agent_key == "fundamentals_analyst"
                        else "",
                        foreign_data=foreign_data
                        if agent_key == "fundamentals_analyst"
                        else "",
                    )
                    retry_tool_calls = getattr(retry_response, "tool_calls", None)
                    retry_has_tool_calls = (
                        isinstance(retry_tool_calls, list) and len(retry_tool_calls) > 0
                    )

                    if retry_has_tool_calls:
                        new_state["messages"] = [retry_response]
                        logger.debug(
                            "analyst_retry_produced_tool_calls",
                            agent_key=agent_key,
                            ticker=ticker,
                        )
                        return new_state

                    logger.debug(
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
                        **summarize_exception(
                            retry_error, operation="analyst_retry_failed"
                        ),
                    )

            from src.utils import detect_truncation

            trunc_info = detect_truncation(content_str, agent=agent_key)
            log_truncation_diagnostic(
                agent_key=agent_key,
                ticker=ticker,
                runnable=llm if response is not None else llm,
                response=response,
                content=content_str,
                trunc_info=trunc_info,
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
                    cap_state_value(content_str, output_field),
                    provider=support.infer_provider_name(llm),
                )
            )

            if agent_key == "fundamentals_analyst":
                logger.debug(
                    "fundamentals_output",
                    has_datablock=has_parseable_data_block(content_str),
                    length=len(content_str),
                )
            return new_state
        except Exception as exc:
            logger.error(
                "analyst_node_error",
                output_field=output_field,
                **summarize_exception(exc, operation="analyst_node_error"),
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

        valuation_context = _extract_valuation_context(fundamentals_report)

        prompt = f"""{agent_prompt.system_message}

TICKER: {ticker}
COMPANY: {company_name}

VALUATION_CONTEXT:
{valuation_context}

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
            logger.debug(
                "valuation_calculator_complete",
                ticker=ticker,
                has_params_block=has_parseable_fenced_block(
                    content_str, "VALUATION_PARAMS"
                ),
                content_length=len(content_str),
            )
            return success_artifact(
                "valuation_params",
                cap_state_value(content_str, "valuation_params"),
                provider=support.infer_provider_name(llm),
            )
        except Exception as exc:
            logger.error(
                "valuation_calculator_error",
                ticker=ticker,
                **summarize_exception(exc, operation="valuation_calculator"),
            )
            return failure_artifact(
                "valuation_params",
                exc,
                provider=support.infer_provider_name(llm),
            )

    return valuation_calculator_node
