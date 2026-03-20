from __future__ import annotations

from typing import Literal

import structlog
from langgraph.types import RunnableConfig

from src.agents import AgentState
from src.config import config
from src.llms import is_openai_consultant_available
from src.runtime_diagnostics import get_artifact_status, is_artifact_complete

logger = structlog.get_logger(__name__)

ANALYST_FAN_OUT_DESTINATIONS = (
    "Market Analyst",
    "Sentiment Analyst",
    "News Analyst",
    "Junior Fundamentals Analyst",
    "Foreign Language Analyst",
    "Legal Counsel",
    "Value Trap Detector",
)


def dispatch_destinations(*, include_auditor: bool) -> list[str]:
    """Return the dispatcher fan-out destinations from one shared source of truth."""
    destinations = list(ANALYST_FAN_OUT_DESTINATIONS)
    if include_auditor:
        destinations.append("Auditor")
    return destinations


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

    node_name = agent_map.get(sender)

    if node_name is None:
        logger.warning(
            "tool_routing_unknown_sender",
            sender=sender,
            fallback="Market Analyst",
            known_agents=list(agent_map.keys()),
            message="Unknown sender in route_tools - defaulting to Market Analyst. "
            "If a new agent was added, update agent_map in route_tools().",
        )
        node_name = "Market Analyst"

    logger.debug("tool_routing", sender=sender, routing_to=node_name)

    return node_name


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
    return is_openai_consultant_available()


def fan_out_to_analysts(state: AgentState, config: RunnableConfig) -> list[str]:
    """
    Fan-out router that triggers all parallel analyst streams.
    Returns a list of destinations for parallel execution.
    """
    destinations = dispatch_destinations(include_auditor=_is_auditor_enabled())
    return destinations


def fundamentals_sync_router(
    state: AgentState, config: RunnableConfig
) -> Literal["Fundamentals Analyst", "__end__"]:
    """
    Synchronization barrier for Junior Fundamentals, Foreign Language, and Legal Counsel.
    """
    junior_done = is_artifact_complete(state, "raw_fundamentals_data")
    foreign_done = is_artifact_complete(state, "foreign_language_report")
    legal_done = is_artifact_complete(state, "legal_report")

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

    return "__end__"


def sync_check_router(
    state: AgentState, config: RunnableConfig
) -> Literal["PM Fast-Fail", "__end__"] | list[str]:
    """
    Synchronization barrier for parallel analyst streams (fan-in pattern).
    """
    market_done = is_artifact_complete(state, "market_report")
    sentiment_done = is_artifact_complete(state, "sentiment_report")
    news_done = is_artifact_complete(state, "news_report")
    value_trap_done = is_artifact_complete(state, "value_trap_report")

    pre_screening = state.get("pre_screening_result")
    validator_done = pre_screening in ["PASS", "REJECT"]

    auditor_done = True
    if _is_auditor_enabled():
        auditor_done = is_artifact_complete(state, "auditor_report")

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
        market_error=get_artifact_status(state, "market_report").error_kind,
        sentiment_error=get_artifact_status(state, "sentiment_report").error_kind,
        news_error=get_artifact_status(state, "news_report").error_kind,
        fundamentals_error=get_artifact_status(state, "fundamentals_report").error_kind,
        pre_screening=pre_screening,
        all_done=all_done,
    )

    if not all_done:
        return "__end__"

    if pre_screening == "REJECT":
        logger.info(
            "sync_routing_to_pm_reject",
            message="Red flags detected - skipping debate, routing to PM Fast-Fail",
        )
        return "PM Fast-Fail"

    logger.info(
        "sync_routing_to_debate",
        message="All analysts complete - proceeding to Bull/Bear Debate Round 1",
    )
    return ["Bull Researcher R1", "Bear Researcher R1"]
