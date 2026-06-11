from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.types import RunnableConfig

from src.config import config as settings_config
from src.error_safety import redact_sensitive_text, summarize_exception
from src.runtime_diagnostics import ArtifactStatus, failure_artifact, success_artifact
from src.runtime_services import get_current_tool_service
from src.tooling.runtime import ToolInvocation

from . import message_utils, support
from . import runtime as agent_runtime
from .consultant_tool_loop import (
    ConsultantToolLoopPolicy,
    remaining_consultant_budget,
    run_bounded_consultant_loop,
)
from .forensic_repair import (
    canonicalize_forensic_auditor_output,
    repair_forensic_auditor_output,
)
from .governance_prompt import governance_block, governance_card
from .output_validation import (
    log_output_diagnostics,
    log_truncation_diagnostic,
    should_fail_closed,
    validate_required_output,
)
from .state import AgentState

logger = structlog.get_logger(__name__)

CONSULTANT_CALL_TIMEOUT_SECONDS = 90.0
CONSULTANT_TOTAL_TIMEOUT_SECONDS = 240.0
_CONSULTANT_QUICK_SCREENING_ADDENDUM = """
## QUICK SCREENING MODE

This is a bounded screening cross-check, not a full re-analysis.
- Do not request tools unless tool results are explicitly available in this turn.
- Preserve the required CONSULTANT REVIEW and FINAL CONSULTANT VERDICT headers.
- Focus only on decision-changing factual errors, biases, synthesis gaps, and mandate breaches.
- If the internal analysis is sound enough for screening, say so briefly.
"""
_CONSULTANT_CONTEXT_BUDGETS = {
    "full": {
        "market": 2000,
        "sentiment": 1500,
        "news": 2000,
        "fundamentals": 5000,
        "debate": 4000,
        "research": 4000,
        "auditor": 3000,
        "apac": 2500,
    },
    "quick_standard": {
        "market": 900,
        "sentiment": 600,
        "news": 900,
        "fundamentals": 2500,
        "debate": 1400,
        "research": 1400,
        "auditor": 1200,
        "apac": 1200,
    },
    "quick_expanded": {
        "market": 1200,
        "sentiment": 800,
        "news": 1200,
        "fundamentals": 3600,
        "debate": 2000,
        "research": 2200,
        "auditor": 1800,
        "apac": 1600,
    },
}


async def _invoke_consultant_with_deadline(
    runnable,
    messages,
    *,
    context: str,
    provider: str,
    model_name: str | None,
    ticker: str,
    deadline: float,
    total_timeout: float = CONSULTANT_TOTAL_TIMEOUT_SECONDS,
) -> object:
    remaining = remaining_consultant_budget(deadline)
    if remaining <= 0:
        raise TimeoutError(
            f"Consultant node exceeded total wall-clock timeout of {total_timeout:.0f}s for {ticker}"
        )

    timeout_s = min(CONSULTANT_CALL_TIMEOUT_SECONDS, remaining)
    try:
        return await agent_runtime.invoke_with_rate_limit_handling(
            runnable,
            messages,
            context=context,
            provider=provider,
            model_name=model_name,
            max_transient_attempts=1,
            overall_timeout_seconds=timeout_s,
        )
    except TimeoutError as exc:
        raise TimeoutError(
            f"Consultant call exceeded {timeout_s:.1f}s wall-clock timeout for {ticker}"
        ) from exc


async def _invoke_agent_loop_llm(
    runnable,
    messages,
    *,
    context: str,
) -> object:
    """Invoke an agent-loop LLM through the shared retry-aware runtime helper."""
    model_name = getattr(runnable, "model_name", None)
    return await agent_runtime.invoke_with_rate_limit_handling(
        runnable,
        messages,
        context=context,
        provider=support.infer_provider_name(runnable),
        model_name=model_name,
    )


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


def _select_quick_consultant_profile(state: AgentState) -> str:
    if state.get("red_flags"):
        return "quick_expanded"

    artifact_statuses = state.get("artifact_statuses", {}) or {}
    auditor_status = artifact_statuses.get("auditor_report") or {}
    if auditor_status and not auditor_status.get("ok", True):
        return "quick_expanded"

    investment_plan = str(state.get("investment_plan") or "").upper()
    if "HOLD" in investment_plan or "CONDITIONAL" in investment_plan:
        return "quick_expanded"

    return "quick_standard"


def _consultant_context_budget(section: str, *, profile: str) -> int:
    return _CONSULTANT_CONTEXT_BUDGETS[profile][section]


def _build_decision_critical_evidence_index(state: AgentState) -> str:
    lines: list[str] = []
    for flag in (state.get("red_flags") or [])[:5]:
        lines.append(str(flag))

    plan_upper = str(state.get("investment_plan") or "").upper()
    for marker in ("PFIC", "CMIC", "VALUE TRAP", "LIQUIDITY", "LEVERAGE"):
        if marker in plan_upper:
            lines.append(f"Research synthesis mentions {marker}")

    if not lines:
        return ""
    compact = "\n".join(f"- {line[:220]}" for line in lines[:8])
    return f"=== DECISION-CRITICAL EVIDENCE INDEX ===\n{compact}\n\n"


def _create_openai_responses_fallback_llm(llm):
    """Build an OpenAI Responses fallback only for OpenAI-compatible models."""
    if support.infer_provider_name(llm) != "openai":
        raise ValueError("OpenAI Responses fallback requires an OpenAI primary LLM")

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=support.get_model_name(llm),
        timeout=120,
        max_retries=3,
        streaming=False,
        use_responses_api=True,
        output_version="responses/v1",
    )


def create_consultant_node(
    llm,
    agent_key: str = "consultant",
    tools: list | None = None,
    *,
    quick_mode: bool = False,
) -> Callable:
    """
    Create external consultant node for cross-validation.
    """
    max_tool_iterations = 1 if quick_mode else 3
    max_tool_calls_per_turn = 2 if quick_mode else 4
    tools_enabled = bool(tools) and (
        not quick_mode or settings_config.consultant_tools_in_quick
    )
    if quick_mode and not settings_config.consultant_tools_in_quick:
        max_tool_iterations = 0

    active_tools = tools if tools_enabled else []
    tools_by_name = {tool.name: tool for tool in active_tools} if active_tools else {}
    llm_with_tools = llm.bind_tools(active_tools) if active_tools else None

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
        apac = state.get("apac_regional_report", "N/A")
        consultant_profile = (
            _select_quick_consultant_profile(state) if quick_mode else "full"
        )
        evidence_index = (
            _build_decision_critical_evidence_index(state) if quick_mode else ""
        )

        all_context = f"""
{evidence_index}\
=== ANALYST REPORTS (SOURCE DATA) ===

MARKET ANALYST REPORT:
{support.summarize_for_pm(market, "market", _consultant_context_budget("market", profile=consultant_profile)) if market != "N/A" else "N/A"}

SENTIMENT ANALYST REPORT:
{support.summarize_for_pm(sentiment, "sentiment", _consultant_context_budget("sentiment", profile=consultant_profile)) if sentiment != "N/A" else "N/A"}

NEWS ANALYST REPORT:
{support.summarize_for_pm(news, "news", _consultant_context_budget("news", profile=consultant_profile)) if news != "N/A" else "N/A"}

FUNDAMENTALS ANALYST REPORT:
{support.summarize_for_pm(fundamentals, "fundamentals", _consultant_context_budget("fundamentals", profile=consultant_profile)) if fundamentals != "N/A" else "N/A"}
{attribution_table}{conflict_table}
=== BULL/BEAR DEBATE HISTORY ===

{support.summarize_for_pm(debate_history, "debate", _consultant_context_budget("debate", profile=consultant_profile)) if debate_history != "N/A" else "N/A"}

=== RESEARCH MANAGER SYNTHESIS ===

{support.summarize_for_pm(investment_plan, "research", _consultant_context_budget("research", profile=consultant_profile)) if investment_plan != "N/A" else "N/A"}

=== RED FLAGS (Pre-Screening Results) ===

Red Flags Detected: {state.get("red_flags", [])}
Pre-Screening Result: {state.get("pre_screening_result", "UNKNOWN")}

=== INDEPENDENT FORENSIC AUDIT ===
{support.summarize_for_pm(auditor, "auditor", _consultant_context_budget("auditor", profile=consultant_profile)) if auditor != "N/A" else "N/A"}

=== APAC REGIONAL SPECIALIST ===
{support.summarize_for_pm(apac, "apac", _consultant_context_budget("apac", profile=consultant_profile)) if apac != "N/A" else "N/A"}
"""

        company_warning = (
            "" if company_resolved else f"\n{support._UNRESOLVED_NAME_WARNING}"
        )

        governance_directive = ""

        card_obj = governance_card(state)
        if card_obj:
            if card_obj.confidence == "conflict":
                governance_directive = (
                    "\n\nGOVERNANCE RECONCILIATION DIRECTIVE: The governance card "
                    "reports a conflict across sources on entity_role. Reconcile this "
                    "explicitly before making any quantitative claim that depends on "
                    "scope (consolidated vs separate), payout mechanics, or vehicle choice. "
                    "Cite the primary source you trust and identify the source you reject."
                )

        prompt = f"""{agent_prompt.system_message}

ANALYSIS DATE: {support._format_date_with_fy_hint(current_date)}
TICKER: {ticker}
COMPANY: {company_name}{company_warning}
{_CONSULTANT_QUICK_SCREENING_ADDENDUM if quick_mode else ""}{governance_block(state)}{governance_directive}

{all_context}

Provide your independent consultant review."""

        content_str = ""
        tool_failure_count = 0
        try:
            messages = [HumanMessage(content=prompt)]
            active_llm = llm_with_tools or llm
            total_timeout = (
                settings_config.consultant_quick_total_timeout_seconds
                if quick_mode
                else CONSULTANT_TOTAL_TIMEOUT_SECONDS
            )
            consultant_deadline = time.monotonic() + total_timeout

            async def _invoke_loop_llm(runnable, loop_messages: list) -> object:
                return await _invoke_consultant_with_deadline(
                    runnable,
                    loop_messages,
                    context=agent_prompt.agent_name,
                    provider=support.infer_provider_name(runnable),
                    model_name=support.get_model_name(runnable),
                    ticker=ticker,
                    deadline=consultant_deadline,
                    total_timeout=total_timeout,
                )

            loop_result = await run_bounded_consultant_loop(
                active_llm=active_llm,
                fallback_llm=llm,
                messages=messages,
                tools_by_name=tools_by_name,
                policy=ConsultantToolLoopPolicy(
                    max_tool_iterations=max_tool_iterations,
                    max_tool_calls_per_turn=max_tool_calls_per_turn,
                    deadline=consultant_deadline,
                    total_timeout=total_timeout,
                ),
                invoke_with_deadline=_invoke_loop_llm,
                tool_service_getter=get_current_tool_service,
                agent_name=agent_prompt.agent_name,
                agent_key=agent_key,
                ticker=ticker,
            )
            content_str = loop_result.content
            response = loop_result.response
            had_tool_errors = loop_result.had_tool_errors
            tool_failure_count = loop_result.tool_failure_count

            from src.utils import detect_truncation

            trunc_info = detect_truncation(content_str, agent="consultant")
            log_truncation_diagnostic(
                agent_key="consultant",
                ticker=ticker,
                runnable=llm,
                response=response,
                content=content_str,
                trunc_info=trunc_info,
            )

            validation = validate_required_output("consultant", content_str)
            log_output_diagnostics(
                agent_key="consultant",
                ticker=ticker,
                runnable=llm,
                response=response,
                content=content_str,
                truncated=trunc_info["truncated"],
                validation=validation,
            )
            if should_fail_closed(
                "consultant",
                validation=validation,
                truncated=trunc_info["truncated"],
                content=content_str,
            ):
                logger.error(
                    "consultant_invalid_structure",
                    ticker=ticker,
                    missing_sections=validation["missing"],
                )
                result = failure_artifact(
                    "consultant_review",
                    "Consultant output missing required structure",
                    provider=support.infer_provider_name(llm),
                    fallback_content=content_str,
                )
                if quick_mode:
                    result["consultant_quick_profile"] = consultant_profile
                return result

            logger.info(
                "consultant_review_complete",
                ticker=ticker,
                review_length=len(content_str),
                has_errors=had_tool_errors,
                tool_failure_count=tool_failure_count,
                truncated=trunc_info["truncated"],
            )
            if had_tool_errors:
                status = ArtifactStatus(
                    complete=True,
                    ok=False,
                    content=content_str,
                    error_kind="application_error",
                    provider=support.infer_provider_name(llm),
                    message="Consultant review completed with tool failures",
                    retryable=False,
                )
                tool_error_result: dict[str, Any] = {
                    "consultant_review": content_str,
                    "consultant_tool_failures": tool_failure_count,
                    **(
                        {"consultant_quick_profile": consultant_profile}
                        if quick_mode
                        else {}
                    ),
                    "artifact_statuses": {
                        "consultant_review": status.as_dict(),
                    },
                }
                return tool_error_result
            result = success_artifact(
                "consultant_review",
                content_str,
                provider=support.infer_provider_name(llm),
            ) | {"consultant_tool_failures": tool_failure_count}
            if quick_mode:
                result["consultant_quick_profile"] = consultant_profile
            return result
        except Exception as exc:
            if isinstance(exc, TimeoutError):
                logger.error(
                    "consultant_node_timeout",
                    ticker=ticker,
                    **summarize_exception(exc, operation="consultant_node_timeout"),
                )
            else:
                logger.error(
                    "consultant_node_error",
                    ticker=ticker,
                    **summarize_exception(exc, operation="consultant_node_error"),
                    exc_info=True,
                )
            result = failure_artifact(
                "consultant_review",
                exc,
                provider=support.infer_provider_name(llm),
            )
            result["consultant_tool_failures"] = tool_failure_count
            if quick_mode:
                result["consultant_quick_profile"] = (
                    locals().get("consultant_profile") or "quick_standard"
                )
            return result

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

        tools_by_name = {t.name: t for t in tools}
        max_tool_iterations = 4

        try:
            messages: list = [
                SystemMessage(content=agent_prompt.system_message),
                HumanMessage(content=human_msg),
            ]
            response_str = ""

            for iteration in range(max_tool_iterations + 1):
                response = await _invoke_agent_loop_llm(
                    llm,
                    messages,
                    context="legal_counsel",
                )
                tool_calls = getattr(response, "tool_calls", None)

                if (
                    not isinstance(tool_calls, list)
                    or not tool_calls
                    or iteration == max_tool_iterations
                ):
                    response_str = message_utils.extract_string_content(
                        getattr(response, "content", "")
                    )
                    break

                messages.append(response)
                for tool_call in tool_calls:
                    tool_fn = tools_by_name.get(tool_call["name"])
                    tool_call_id = tool_call.get("id", tool_call["name"])
                    if tool_fn:
                        try:

                            async def _run_legal_tool(
                                args: dict[str, Any], tool=tool_fn
                            ) -> Any:
                                return await tool.ainvoke(args)

                            tool_result = await get_current_tool_service().execute(
                                ToolInvocation(
                                    name=tool_call["name"],
                                    args=tool_call["args"],
                                    source="legal_counsel",
                                    agent_key="legal_counsel",
                                ),
                                runner=_run_legal_tool,
                            )
                            tool_output = str(tool_result.value)
                        except Exception as tool_err:
                            logger.warning(
                                "legal_counsel_tool_failed",
                                ticker=ticker,
                                tool=tool_call["name"],
                                **summarize_exception(
                                    tool_err, operation="legal_counsel_tool_failed"
                                ),
                            )
                            tool_output = f"TOOL_ERROR: {tool_err}"
                    else:
                        tool_output = f"Unknown tool: {tool_call['name']}"
                    messages.append(
                        ToolMessage(content=tool_output, tool_call_id=tool_call_id)
                    )

                logger.debug(
                    "legal_counsel_tool_iteration",
                    ticker=ticker,
                    iteration=iteration + 1,
                    tools_called=[tc["name"] for tc in tool_calls],
                )

            try:
                parsed = json.loads(response_str)
                logger.debug(
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
                        logger.debug("legal_counsel_extracted_json", ticker=ticker)
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
                    response_preview=redact_sensitive_text(response_str, max_chars=200),
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
            logger.error(
                "legal_counsel_error",
                ticker=ticker,
                **summarize_exception(exc, operation="legal_counsel_error"),
                exc_info=True,
            )
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

        tools_by_name = {t.name: t for t in tools}
        # recursion_limit=12 in the old create_react_agent maps to 6 tool-call rounds
        # (each round = 1 LLM call + 1 tool execution step in LangGraph).
        # We use 6 manual iterations here to preserve the same budget.
        max_tool_iterations = 6

        def _truncate_messages_for_llm(msgs: list) -> list:
            """Apply the auditor truncation hook to ToolMessages before LLM invocation."""
            from langchain_core.messages import ToolMessage as LCToolMessage

            result_msgs = []
            for msg in msgs:
                if isinstance(msg, LCToolMessage):
                    content = (
                        msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content)
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
                        result_msgs.append(
                            LCToolMessage(
                                content=truncated,
                                tool_call_id=msg.tool_call_id,
                                name=getattr(msg, "name", None),
                            )
                        )
                        logger.debug(
                            "auditor_tool_output_truncated",
                            original_len=len(content),
                            truncated_len=len(truncated),
                        )
                    else:
                        result_msgs.append(msg)
                else:
                    result_msgs.append(msg)
            return result_msgs

        async def _run_auditor_loop(active_llm, agent_prompt_sys: str) -> str:
            messages: list = [
                SystemMessage(content=agent_prompt_sys),
                HumanMessage(content=human_msg),
            ]
            for iteration in range(max_tool_iterations + 1):
                llm_input = _truncate_messages_for_llm(messages)
                response = await _invoke_agent_loop_llm(
                    active_llm,
                    llm_input,
                    context="global_forensic_auditor",
                )
                tool_calls = getattr(response, "tool_calls", None)

                if (
                    not isinstance(tool_calls, list)
                    or not tool_calls
                    or iteration == max_tool_iterations
                ):
                    return message_utils.extract_string_content(
                        getattr(response, "content", "")
                    )

                messages.append(response)
                for tool_call in tool_calls:
                    tool_fn = tools_by_name.get(tool_call["name"])
                    tool_call_id = tool_call.get("id", tool_call["name"])
                    if tool_fn:
                        try:

                            async def _run_auditor_tool(
                                args: dict[str, Any], tool=tool_fn
                            ) -> Any:
                                return await tool.ainvoke(args)

                            tool_result = await get_current_tool_service().execute(
                                ToolInvocation(
                                    name=tool_call["name"],
                                    args=tool_call["args"],
                                    source="auditor",
                                    agent_key="global_forensic_auditor",
                                ),
                                runner=_run_auditor_tool,
                            )
                            tool_output = str(tool_result.value)
                        except Exception as tool_err:
                            logger.warning(
                                "auditor_tool_failed",
                                ticker=ticker,
                                tool=tool_call["name"],
                                **summarize_exception(
                                    tool_err, operation="auditor_tool_failed"
                                ),
                            )
                            tool_output = f"TOOL_ERROR: {tool_err}"
                    else:
                        tool_output = f"Unknown tool: {tool_call['name']}"
                    messages.append(
                        ToolMessage(content=tool_output, tool_call_id=tool_call_id)
                    )

                logger.debug(
                    "auditor_tool_iteration",
                    ticker=ticker,
                    iteration=iteration + 1,
                    tools_called=[tc["name"] for tc in tool_calls],
                )
            return ""

        logger.debug("auditor_start", ticker=ticker)

        try:
            response_str = await _run_auditor_loop(llm, agent_prompt.system_message)
            response_str = canonicalize_forensic_auditor_output(response_str)
            validation = validate_required_output(
                "global_forensic_auditor", response_str
            )

            if not validation["ok"]:
                repaired = await repair_forensic_auditor_output(
                    llm,
                    invalid_output=response_str,
                )
                repaired = canonicalize_forensic_auditor_output(repaired)
                repaired_validation = validate_required_output(
                    "global_forensic_auditor", repaired
                )
                response_str = repaired
                validation = repaired_validation

            from src.utils import detect_truncation

            trunc_info = detect_truncation(
                response_str, agent="global_forensic_auditor"
            )
            log_truncation_diagnostic(
                agent_key="global_forensic_auditor",
                ticker=ticker,
                runnable=llm,
                response=None,
                content=response_str,
                trunc_info=trunc_info,
            )
            log_output_diagnostics(
                agent_key="global_forensic_auditor",
                ticker=ticker,
                runnable=llm,
                response=None,
                content=response_str,
                truncated=trunc_info["truncated"],
                validation=validation,
            )
            if should_fail_closed(
                "global_forensic_auditor",
                validation=validation,
                truncated=trunc_info["truncated"],
                content=response_str,
            ):
                logger.error(
                    "auditor_invalid_structure",
                    ticker=ticker,
                    missing_sections=validation["missing"],
                    output_preview=redact_sensitive_text(
                        response_str[:400].replace("\n", " "), max_chars=200
                    ),
                )
                result = failure_artifact(
                    "auditor_report",
                    "Auditor output missing required structure",
                    provider=support.infer_provider_name(llm),
                    fallback_content=response_str,
                )
                result["sender"] = "global_forensic_auditor"
                return result

            logger.debug("auditor_complete", ticker=ticker, length=len(response_str))
            result = success_artifact(
                "auditor_report",
                response_str,
                provider=support.infer_provider_name(llm),
            )
            result["sender"] = "global_forensic_auditor"
            return result
        except Exception as exc:
            error_str = str(exc)
            logger.error(
                "auditor_error",
                ticker=ticker,
                reason=error_str,
                exc_info=True,
            )

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
                    reason=error_str,
                )
                try:
                    fallback_llm = _create_openai_responses_fallback_llm(llm)
                    response_str = await _run_auditor_loop(
                        fallback_llm, agent_prompt.system_message
                    )
                    response_str = canonicalize_forensic_auditor_output(response_str)
                    logger.debug(
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
                        reason=str(retry_exc),
                        exc_info=True,
                    )

            result = failure_artifact(
                "auditor_report",
                exc,
                provider=support.infer_provider_name(llm),
            )
            result["sender"] = "global_forensic_auditor"
            return result

    return auditor_node
