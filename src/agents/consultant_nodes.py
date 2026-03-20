from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from datetime import datetime

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage as TM
from langgraph.types import RunnableConfig

from src.runtime_diagnostics import ArtifactStatus, failure_artifact, success_artifact
from src.tooling.runtime import TOOL_SERVICE, ToolInvocation

from . import message_utils, support
from . import runtime as agent_runtime
from .state import AgentState

logger = structlog.get_logger(__name__)

CONSULTANT_CALL_TIMEOUT_SECONDS = 90.0
CONSULTANT_TOTAL_TIMEOUT_SECONDS = 240.0


def _remaining_consultant_budget(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _build_consultant_fmp_skip_payload(
    *, ticker: str, metric: str, failure_kind: str
) -> str:
    reason = (
        "current FMP plan does not cover this request"
        if failure_kind == "auth_error"
        else "FMP cooldown is active after a quota or rate-limit response"
    )
    return json.dumps(
        {
            "error": f"SKIPPED: spot_check_metric_alt disabled after prior FMP {failure_kind}",
            "suggestion": "Use official filings or primary-source evidence for cross-checks in this run",
            "ticker": ticker,
            "metric": metric,
            "provider": "fmp",
            "failure_kind": failure_kind,
            "retryable": failure_kind == "rate_limit",
            "skipped": True,
            "reason": reason,
        }
    )


async def _invoke_consultant_with_deadline(
    runnable,
    messages,
    *,
    context: str,
    provider: str,
    model_name: str,
    ticker: str,
    deadline: float,
) -> object:
    remaining = _remaining_consultant_budget(deadline)
    if remaining <= 0:
        raise TimeoutError(
            f"Consultant node exceeded total wall-clock timeout of {CONSULTANT_TOTAL_TIMEOUT_SECONDS:.0f}s for {ticker}"
        )

    timeout_s = min(CONSULTANT_CALL_TIMEOUT_SECONDS, remaining)
    try:
        async with asyncio.timeout(timeout_s):
            return await agent_runtime.invoke_with_rate_limit_handling(
                runnable,
                messages,
                context=context,
                provider=provider,
                model_name=model_name,
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
            had_tool_errors = False
            tool_failure_count = 0
            consultant_deadline = time.monotonic() + CONSULTANT_TOTAL_TIMEOUT_SECONDS
            fmp_alt_disabled_kind: str | None = None
            active_llm_provider = support.infer_provider_name(active_llm)
            active_llm_model = support.get_model_name(active_llm)
            fallback_llm_provider = support.infer_provider_name(llm)
            fallback_llm_model = support.get_model_name(llm)

            for iteration in range(max_tool_iterations + 1):
                response = await _invoke_consultant_with_deadline(
                    active_llm,
                    messages,
                    context=agent_prompt.agent_name,
                    provider=active_llm_provider,
                    model_name=active_llm_model,
                    ticker=ticker,
                    deadline=consultant_deadline,
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
                    result_failed = False
                    count_failure = True
                    if tool_fn:
                        if (
                            tool_call["name"] == "spot_check_metric_alt"
                            and fmp_alt_disabled_kind is not None
                        ):
                            result = _build_consultant_fmp_skip_payload(
                                ticker=tool_call["args"].get("ticker", ticker),
                                metric=tool_call["args"].get("metric", "unknown"),
                                failure_kind=fmp_alt_disabled_kind,
                            )
                            count_failure = False
                            logger.info(
                                "consultant_tool_suppressed",
                                ticker=ticker,
                                tool=tool_call["name"],
                                failure_kind=fmp_alt_disabled_kind,
                            )
                        else:
                            if _remaining_consultant_budget(consultant_deadline) <= 0:
                                raise TimeoutError(
                                    f"Consultant node exceeded total wall-clock timeout of {CONSULTANT_TOTAL_TIMEOUT_SECONDS:.0f}s for {ticker}"
                                )
                            try:
                                tool_result = await TOOL_SERVICE.execute(
                                    ToolInvocation(
                                        name=tool_call["name"],
                                        args=tool_call["args"],
                                        source="consultant",
                                        agent_key=agent_key,
                                    ),
                                    runner=lambda args, tool=tool_fn: tool.ainvoke(
                                        args
                                    ),
                                )
                                result = tool_result.value
                            except Exception as tool_err:
                                result_failed = True
                                logger.warning(
                                    "consultant_tool_failed",
                                    ticker=ticker,
                                    tool=tool_call["name"],
                                    error=str(tool_err),
                                )
                                result = f"TOOL_ERROR: {str(tool_err)}"
                    else:
                        result_failed = True
                        result = f"Unknown tool: {tool_call['name']}"
                    if isinstance(result, str):
                        stripped = result.strip()
                        if stripped.startswith(("TOOL_ERROR:", "TOOL_BLOCKED:")):
                            result_failed = True
                        else:
                            try:
                                payload = json.loads(stripped)
                            except (TypeError, ValueError):
                                payload = None
                            if isinstance(payload, dict) and payload.get("error"):
                                result_failed = True
                                if (
                                    tool_call["name"] == "spot_check_metric_alt"
                                    and payload.get("provider") == "fmp"
                                    and payload.get("failure_kind")
                                    in {"auth_error", "rate_limit"}
                                    and not payload.get("skipped")
                                ):
                                    fmp_alt_disabled_kind = payload["failure_kind"]
                    if result_failed:
                        had_tool_errors = True
                        if count_failure:
                            tool_failure_count += 1
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
                response = await _invoke_consultant_with_deadline(
                    llm,
                    messages,
                    context=agent_prompt.agent_name,
                    provider=fallback_llm_provider,
                    model_name=fallback_llm_model,
                    ticker=ticker,
                    deadline=consultant_deadline,
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
                return {
                    "consultant_review": content_str,
                    "consultant_tool_failures": tool_failure_count,
                    "artifact_statuses": {
                        "consultant_review": status.as_dict(),
                    },
                }
            return success_artifact(
                "consultant_review",
                content_str,
                provider=support.infer_provider_name(llm),
            ) | {"consultant_tool_failures": tool_failure_count}
        except Exception as exc:
            logger.error(
                "consultant_node_error",
                ticker=ticker,
                error=str(exc),
                exc_info=True,
            )
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
                        response.content
                    )
                    break

                messages.append(response)
                for tool_call in tool_calls:
                    tool_fn = tools_by_name.get(tool_call["name"])
                    tool_call_id = tool_call.get("id", tool_call["name"])
                    if tool_fn:
                        try:
                            tool_result = await TOOL_SERVICE.execute(
                                ToolInvocation(
                                    name=tool_call["name"],
                                    args=tool_call["args"],
                                    source="legal_counsel",
                                    agent_key="legal_counsel",
                                ),
                                runner=lambda args, fn=tool_fn: fn.ainvoke(args),
                            )
                            tool_output = str(tool_result.value)
                        except Exception as tool_err:
                            logger.warning(
                                "legal_counsel_tool_failed",
                                ticker=ticker,
                                tool=tool_call["name"],
                                error=str(tool_err),
                            )
                            tool_output = f"TOOL_ERROR: {tool_err}"
                    else:
                        tool_output = f"Unknown tool: {tool_call['name']}"
                    messages.append(TM(content=tool_output, tool_call_id=tool_call_id))

                logger.info(
                    "legal_counsel_tool_iteration",
                    ticker=ticker,
                    iteration=iteration + 1,
                    tools_called=[tc["name"] for tc in tool_calls],
                )

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
            logger.error(
                "legal_counsel_error",
                ticker=ticker,
                error=str(exc),
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
                    return message_utils.extract_string_content(response.content)

                messages.append(response)
                for tool_call in tool_calls:
                    tool_fn = tools_by_name.get(tool_call["name"])
                    tool_call_id = tool_call.get("id", tool_call["name"])
                    if tool_fn:
                        try:
                            tool_result = await TOOL_SERVICE.execute(
                                ToolInvocation(
                                    name=tool_call["name"],
                                    args=tool_call["args"],
                                    source="auditor",
                                    agent_key="global_forensic_auditor",
                                ),
                                runner=lambda args, fn=tool_fn: fn.ainvoke(args),
                            )
                            tool_output = str(tool_result.value)
                        except Exception as tool_err:
                            logger.warning(
                                "auditor_tool_failed",
                                ticker=ticker,
                                tool=tool_call["name"],
                                error=str(tool_err),
                            )
                            tool_output = f"TOOL_ERROR: {tool_err}"
                    else:
                        tool_output = f"Unknown tool: {tool_call['name']}"
                    messages.append(TM(content=tool_output, tool_call_id=tool_call_id))

                logger.info(
                    "auditor_tool_iteration",
                    ticker=ticker,
                    iteration=iteration + 1,
                    tools_called=[tc["name"] for tc in tool_calls],
                )
            return ""

        logger.info("auditor_start", ticker=ticker)

        try:
            response_str = await _run_auditor_loop(llm, agent_prompt.system_message)

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
            logger.error(
                "auditor_error",
                ticker=ticker,
                error=error_str,
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
                    response_str = await _run_auditor_loop(
                        fallback_llm, agent_prompt.system_message
                    )
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
