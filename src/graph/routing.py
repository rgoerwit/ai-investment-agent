from __future__ import annotations

from typing import Literal

import structlog
from langgraph.types import RunnableConfig

from src.agents import AgentState
from src.config import config
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


def is_openai_consultant_available() -> bool:
    from src.llms import (
        is_openai_consultant_available as _is_openai_consultant_available,
    )

    return _is_openai_consultant_available()


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

    result: Literal["tools", "continue"] = "tools" if has_tool_calls else "continue"

    logger.debug(
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
    ENABLE_CONSULTANT currently gates the shared OpenAI cross-check plane, so
    it applies to the auditor path too even though the setting name is narrower.
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

    logger.debug(
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

    logger.debug(
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


def post_research_sync_router(
    state: AgentState,
    config: RunnableConfig,
    *,
    apac_required: bool,
    consultant_required: bool,
) -> Literal["Trader", "__end__"]:
    """Wait for post-research artifacts before releasing Trader.

    Valuation and the optional APAC/Consultant branch run in parallel. Without
    this barrier, the faster valuation branch can trigger Trader and the entire
    risk/PM tail before Consultant finishes, then trigger it again afterward.
    """
    valuation_done = is_artifact_complete(state, "valuation_params")
    consultant_done = (
        is_artifact_complete(state, "consultant_review")
        if consultant_required
        else True
    )
    apac_done = (
        is_artifact_complete(state, "apac_regional_report")
        if apac_required and not consultant_required
        else True
    )
    all_done = valuation_done and consultant_done and apac_done

    logger.debug(
        "post_research_sync_status",
        valuation_done=valuation_done,
        consultant_done=consultant_done,
        apac_done=apac_done,
        consultant_required=consultant_required,
        apac_required=apac_required,
        valuation_error=get_artifact_status(state, "valuation_params").error_kind,
        consultant_error=get_artifact_status(state, "consultant_review").error_kind,
        apac_error=get_artifact_status(state, "apac_regional_report").error_kind,
        all_done=all_done,
    )

    if not all_done:
        return "__end__"

    logger.info(
        "post_research_sync_complete",
        message="Valuation and optional post-research reviews complete - proceeding to Trader",
    )
    return "Trader"


# --- Consultant bypass gate (P0-2) ------------------------------------------

CONSULTANT_SKIP_SENTINEL = (
    "SKIPPED_BY_GATE: External Consultant bypass active for quick-mode screening. "
    "Reason: {reason}. The internal analysis (analyst reports, debate, Research "
    "Manager synthesis, and forensic audit) was retained for the Portfolio "
    "Manager's decision."
)

# Warning-level flags that should keep the Consultant active even on a clean
# Auditor pass — these are exactly the situations where a second opinion is
# load-bearing for the final verdict.
_GATE_BLOCKING_RED_FLAGS = frozenset(
    {
        "VALUE_TRAP_HIGH_RISK",
        "VALUE_TRAP_MODERATE_RISK",
        "VALUE_TRAP_VERDICT",
        "PFIC_PROBABLE",
        "PFIC_UNCERTAIN",
        "VIE_STRUCTURE",
        "SEGMENT_DETERIORATION",
        "OCF_SOURCE_DISCREPANCY",
        "NO_CATALYST_DETECTED",
        # Must match what detect_legal_flags actually emits: the old "CMIC_LISTED"
        # entry was a dead string produced by nothing in src/, so a genuine
        # NS-CMIC hit never kept the Consultant active through this gate.
        "CMIC_FLAGGED",
        "CMIC_UNCERTAIN",
    }
)

# Unresolved-data-conflict flags: a negative RM verdict normally skips the
# Consultant (low value-add on a clear reject), but when the reject rests on a
# data discrepancy the Consultant's reconciliation is the only step that can
# resolve it (145020.KQ OCF period-mismatch, July 2026). Deliberately narrower
# than _GATE_BLOCKING_RED_FLAGS: well-founded value-trap/PFIC rejects still skip.
_GATE_DATA_DISCREPANCY_FLAGS = frozenset(
    {
        "OCF_SOURCE_DISCREPANCY",
        "OCF_FILING_VALUE_UNCORROBORATED",
        "CONSULTANT_DATA_DISCREPANCY",
    }
)

_AUDITOR_CLEAN_STATUSES = frozenset({"CLEAN", "INSUFFICIENT_DATA", "UNAVAILABLE"})


def parse_auditor_status(auditor_report: object) -> str | None:
    """Return the auditor report's declared STATUS token (upper-case), or None.

    Shared by the Consultant-gate cleanliness check and the run-summary success
    flag so both read the auditor's `STATUS:` line through one regex.
    """
    if not isinstance(auditor_report, str):
        return None
    text = auditor_report.strip()
    if not text or text.upper() == "N/A":
        return "N/A"
    import re

    match = re.search(r"(?im)^\s*STATUS\s*[:=]\s*([A-Z_]+)", text)
    return match.group(1).upper() if match else None


def _auditor_status_clean(auditor_report: object) -> bool:
    """True when the auditor produced no actionable problem."""
    if auditor_report is None:
        return True
    if not isinstance(auditor_report, str):
        return False
    status = parse_auditor_status(auditor_report)
    if status == "N/A":
        return True
    if status is not None:
        return status in _AUDITOR_CLEAN_STATUSES
    # No explicit status field — fall back to a conservative "keep Consultant"
    # decision so a misformatted auditor output never silently disables it.
    return False


def _has_marker_red_flag(red_flags: object, markers: frozenset[str]) -> bool:
    """True when any flag's stringified form contains one of ``markers``.

    Flags may be plain strings or dicts/objects (production emits dicts with a
    ``type`` key) — the str() form tolerates both.
    """
    if not red_flags:
        return False
    if isinstance(red_flags, list | tuple | set):
        items = list(red_flags)
    else:
        items = [red_flags]
    for raw in items:
        text = str(raw).upper()
        if any(marker in text for marker in markers):
            return True
    return False


def _has_blocking_red_flag(red_flags: object) -> bool:
    return _has_marker_red_flag(red_flags, _GATE_BLOCKING_RED_FLAGS)


def _investment_plan_has_conflict(investment_plan: object) -> bool:
    if not isinstance(investment_plan, str):
        return False
    upper = investment_plan.upper()
    return "CONFLICT" in upper or "DISAGREEMENT" in upper


_POSITIVE_VERDICT_RE = None
_NEGATIVE_VERDICT_RE = None


def _classify_rm_verdict(investment_plan: object) -> str:
    """Return one of "positive", "negative", "ambiguous"."""
    if not isinstance(investment_plan, str) or not investment_plan.strip():
        return "ambiguous"

    import re

    global _POSITIVE_VERDICT_RE, _NEGATIVE_VERDICT_RE
    if _POSITIVE_VERDICT_RE is None:
        # Tolerate markdown header prefixes and the INVESTMENT/FINAL qualifier,
        # e.g. "### FINAL RECOMMENDATION: BUY" / "### INVESTMENT RECOMMENDATION: REJECT".
        _POSITIVE_VERDICT_RE = re.compile(
            r"(?im)^\s*#*\s*(?:FINAL\s+|INVESTMENT\s+)?(?:RECOMMENDATION|VERDICT|DECISION)\s*[:=]\s*"
            r"(STRONG[\s_-]*BUY|BUY|ACCUMULATE|WATCH|INITIATE)"
        )
        _NEGATIVE_VERDICT_RE = re.compile(
            r"(?im)^\s*#*\s*(?:FINAL\s+|INVESTMENT\s+)?(?:RECOMMENDATION|VERDICT|DECISION)\s*[:=]\s*"
            r"(SELL|STRONG[\s_-]*SELL|DO[\s_-]*NOT[\s_-]*INITIATE|REJECT|AVOID|"
            r"STRONG[\s_-]*HOLD)"
        )

    if _NEGATIVE_VERDICT_RE.search(investment_plan):
        return "negative"
    if _POSITIVE_VERDICT_RE.search(investment_plan):
        return "positive"
    return "ambiguous"


def _quick_mode_active(config: RunnableConfig) -> bool:
    """Extract quick_mode from the TradingContext attached to the runnable config."""
    context = (config or {}).get("configurable", {}).get("context")
    return bool(getattr(context, "quick_mode", False))


def should_invoke_consultant(
    state: AgentState, config: RunnableConfig
) -> tuple[bool, str]:
    """Decide whether the Consultant LLM should run after Research Manager.

    Returns ``(True, reason)`` to invoke and ``(False, reason)`` to skip.
    Skips fire only in ``--quick`` mode; full runs always invoke Consultant
    so the deeper review remains the source of truth for production reports.
    A negative RM verdict skips only when it is *clear* — a reject resting on
    an unresolved data discrepancy keeps the Consultant so its reconciliation
    isn't lost from the final report.
    """
    if not _quick_mode_active(config):
        return True, "full_mode"

    investment_plan = state.get("investment_plan")
    verdict = _classify_rm_verdict(investment_plan)
    red_flags = state.get("red_flags") or []
    has_conflict = _investment_plan_has_conflict(investment_plan)

    if (
        verdict == "negative"
        and not has_conflict
        and not _has_marker_red_flag(red_flags, _GATE_DATA_DISCREPANCY_FLAGS)
    ):
        return False, "rm_clear_negative"

    if verdict == "positive":
        auditor_clean = _auditor_status_clean(state.get("auditor_report"))
        if auditor_clean and not _has_blocking_red_flag(red_flags) and not has_conflict:
            return False, "clean_consensus"

    return True, "default_invoke"


def consultant_gate_router(
    state: AgentState, config: RunnableConfig
) -> Literal["Consultant", "Consultant Skip"]:
    """Conditional edge after Research Manager — gates the Consultant LLM call."""
    invoke, reason = should_invoke_consultant(state, config)
    if invoke:
        logger.debug("consultant_gate", decision="invoke", reason=reason)
        return "Consultant"
    logger.info(
        "consultant_skipped_for_screening",
        reason=reason,
        message="Quick-mode Consultant bypass active",
    )
    return "Consultant Skip"
