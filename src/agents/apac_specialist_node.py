from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import RunnableConfig

from src.error_safety import summarize_exception
from src.runtime_diagnostics import failure_artifact, success_artifact
from src.tooling.text_boundary import format_untrusted_block
from src.validators.supplemental_extractors import extract_capital_efficiency_signals

from . import message_utils, support
from . import runtime as agent_runtime
from .governance_prompt import rendered_governance_card
from .output_limits import cap_state_value
from .state import AgentState

logger = structlog.get_logger(__name__)

APAC_NO_MATERIAL_SENTINEL = "NO_MATERIAL_APAC_CONNECTION"
APAC_UNAVAILABLE_SENTINEL = "APAC_SPECIALIST_UNAVAILABLE"
APAC_REPORT_FIELD = "apac_regional_report"


def _is_glm_1301_policy_block(exc: BaseException) -> bool:
    message = str(exc)
    return bool(
        re.search(r"(?<!\d)1301(?!\d)", message)
        and re.search(
            r"error|policy|content|sensitive|unsafe|moderation|不安全|敏感",
            message,
            re.IGNORECASE,
        )
    )


def _clip(value: object, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[...truncated]"


def _shareholder_return_signal(state: AgentState) -> str:
    """Surface the Value-Up plan-strength + execution markers verbatim.

    Senior Fundamentals promotes ``VALUE_UP_PLAN_STRENGTH`` and
    ``SHAREHOLDER_RETURN_EXECUTION`` into the DATA_BLOCK, but those lines can be
    clipped out of the larger ``fundamentals_report`` / ``foreign_language_report``
    payload slices. The APAC specialist needs them intact to adjudicate
    ``VALUE_UP_EXECUTION_CREDIT``, so pass them as a dedicated short key.
    """
    signals = extract_capital_efficiency_signals(state.get("fundamentals_report") or "")
    strength = signals.get("value_up_plan_strength") or "UNKNOWN"
    execution = signals.get("shareholder_return_execution") or "UNKNOWN"
    return (
        f"VALUE_UP_PLAN_STRENGTH: {strength}\nSHAREHOLDER_RETURN_EXECUTION: {execution}"
    )


def build_apac_specialist_payload(state: AgentState) -> dict[str, str]:
    """Build the allowlisted dossier sent to the APAC specialist."""
    return {
        "ticker": state.get("company_of_interest", ""),
        "company": state.get("company_name", ""),
        "trade_date": state.get("trade_date", ""),
        "entity_governance_card": _clip(
            rendered_governance_card(state),
            2200,
        ),
        "investment_plan": _clip(state.get("investment_plan"), 6000),
        "fundamentals_report": _clip(state.get("fundamentals_report"), 5000),
        "foreign_language_report": _clip(state.get("foreign_language_report"), 3000),
        "value_trap_report": _clip(state.get("value_trap_report"), 2500),
        "shareholder_return_signal": _shareholder_return_signal(state),
        "news_report": _clip(state.get("news_report"), 2000),
        "sentiment_report": _clip(state.get("sentiment_report"), 1200),
        "red_flags": _clip(state.get("red_flags"), 1800),
    }


def create_apac_specialist_node(llm, *, fallback_llm=None) -> Callable:
    """Create the no-tools APAC Regional Specialist node."""

    async def apac_specialist_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        from src.prompts import get_prompt

        ticker = state.get("company_of_interest", "")
        agent_prompt = get_prompt("apac_regional_specialist")
        if not agent_prompt:
            logger.error("missing_prompt", agent="apac_regional_specialist")
            return failure_artifact(
                APAC_REPORT_FIELD,
                "Missing APAC Regional Specialist prompt",
                provider="unknown",
            )

        prompts_used = dict(state.get("prompts_used", {}) or {})
        prompts_used[APAC_REPORT_FIELD] = {
            "agent_name": agent_prompt.agent_name,
            "version": agent_prompt.version,
        }

        payload = build_apac_specialist_payload(state)
        wrapped_payload = format_untrusted_block(
            json.dumps(payload, ensure_ascii=False, indent=2),
            "MINIMIZED_UPSTREAM_ANALYSIS",
            provenance="prior analyst artifacts",
        )
        messages = [
            SystemMessage(content=agent_prompt.system_message),
            HumanMessage(content=wrapped_payload),
        ]

        active_llm = llm
        try:
            try:
                response = await agent_runtime.invoke_with_rate_limit_handling(
                    active_llm,
                    messages,
                    context=agent_prompt.agent_name,
                    provider=support.infer_provider_name(active_llm),
                    model_name=support.get_model_name(active_llm),
                    overall_timeout_seconds=240,
                )
            except Exception as exc:
                if fallback_llm is None or not _is_glm_1301_policy_block(exc):
                    raise
                payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                logger.warning(
                    "apac_policy_block_direct_retry",
                    ticker=ticker,
                    provider_code=1301,
                    payload_sha256=hashlib.sha256(payload_json.encode()).hexdigest(),
                    payload_chars=len(payload_json),
                )
                active_llm = fallback_llm
                response = await agent_runtime.invoke_with_rate_limit_handling(
                    active_llm,
                    messages,
                    context=f"{agent_prompt.agent_name}_direct_retry",
                    provider=support.infer_provider_name(active_llm),
                    model_name=support.get_model_name(active_llm),
                    overall_timeout_seconds=240,
                    max_attempts=1,
                    max_transient_attempts=1,
                )
            text = message_utils.extract_string_content(response.content).strip()
            if text != APAC_NO_MATERIAL_SENTINEL:
                text = cap_state_value(text, APAC_REPORT_FIELD)

            result = success_artifact(
                APAC_REPORT_FIELD,
                text,
                provider=support.infer_provider_name(active_llm),
            )
            result["sender"] = "apac_regional_specialist"
            result["messages"] = [response]
            result["prompts_used"] = prompts_used
            return result
        except Exception as exc:
            logger.warning(
                "apac_specialist_failed",
                ticker=ticker,
                **summarize_exception(exc, operation="apac_specialist_failed"),
                exc_info=True,
            )
            result = failure_artifact(
                APAC_REPORT_FIELD,
                exc,
                provider=support.infer_provider_name(llm),
                fallback_content=APAC_UNAVAILABLE_SENTINEL,
            )
            result["sender"] = "apac_regional_specialist"
            result["prompts_used"] = prompts_used
            return result

    return apac_specialist_node
