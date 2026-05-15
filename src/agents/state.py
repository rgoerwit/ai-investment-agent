import json
from typing import Annotated, Any

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

PROVENANCE_MARKERS = ('"_field_sources"', '"_source_conflicts"')
MESSAGE_TAIL_LIMIT = 12


class InvestDebateState(TypedDict):
    """
    State tracking bull/bear investment debate progression (parallel-safe).

    Uses dedicated fields per round to allow parallel execution of Bull/Bear
    in each round without race conditions.
    """

    bull_round1: str
    bear_round1: str
    bull_round2: str
    bear_round2: str
    current_round: int
    bull_history: str
    bear_history: str
    history: str
    current_response: str
    judge_decision: str
    count: int


class RiskDebateState(TypedDict):
    """State tracking multi-perspective risk assessment debate (parallel-safe)."""

    latest_speaker: str
    current_risky_response: str
    current_safe_response: str
    current_neutral_response: str


def take_last(x, y):
    """Reducer: takes the most recent value. Used with Annotated fields."""
    return y


def _message_key(message: BaseMessage) -> tuple[str, str, str]:
    return (
        type(message).__name__,
        getattr(message, "id", "") or "",
        getattr(message, "tool_call_id", "") or "",
    )


def _is_provenance_tool_message(message: BaseMessage) -> bool:
    if not isinstance(message, ToolMessage):
        return False
    try:
        content = (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
    except Exception:
        return False
    return any(marker in content for marker in PROVENANCE_MARKERS)


def merge_and_cap_messages(
    x: list[BaseMessage] | None, y: list[BaseMessage] | BaseMessage | None
) -> list[BaseMessage]:
    """Merge messages using LangGraph semantics, then cap generic history."""
    merged = add_messages(x or [], y or [])
    if not merged:
        return []

    preserved_indices: set[int] = set()

    for idx, message in enumerate(merged):
        if isinstance(message, HumanMessage):
            preserved_indices.add(idx)
            break

    for idx, message in enumerate(merged):
        if _is_provenance_tool_message(message):
            preserved_indices.add(idx)

    tail_candidates = [
        idx for idx in range(len(merged)) if idx not in preserved_indices
    ]
    preserved_indices.update(tail_candidates[-MESSAGE_TAIL_LIMIT:])

    result: list[BaseMessage] = []
    seen_keys: set[tuple[str, str, str, int]] = set()
    for idx, message in enumerate(merged):
        if idx not in preserved_indices:
            continue
        identity_key = (*_message_key(message), id(message))
        if identity_key in seen_keys:
            continue
        seen_keys.add(identity_key)
        result.append(message)

    return result


def merge_dicts(x: dict | None, y: dict | None) -> dict:
    """Reducer: merges dictionaries. Used for parallel agent state updates."""
    if x is None:
        return y or {}
    if y is None:
        return x
    return {**x, **y}


def merge_flag_lists(
    x: list[dict[str, Any]] | None,
    y: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge cumulative red-flag findings without losing parallel updates."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in [*(x or []), *(y or [])]:
        key = json.dumps(item, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    return merged


def merge_risk_state(
    x: RiskDebateState | None, y: RiskDebateState | None
) -> RiskDebateState:
    """
    Reducer for RiskDebateState that merges parallel updates.

    Simple merge is safe because each parallel agent writes to a distinct key.
    """
    if x is None:
        return y or RiskDebateState(
            latest_speaker="",
            current_risky_response="",
            current_safe_response="",
            current_neutral_response="",
        )
    if y is None:
        return x
    return {**x, **y}


def merge_invest_debate_state(
    x: InvestDebateState | None, y: InvestDebateState | None
) -> InvestDebateState:
    """
    Reducer for InvestDebateState that merges parallel updates.

    Safe for parallel Bull/Bear execution because each writes to distinct fields.
    """
    default_state = InvestDebateState(
        bull_round1="",
        bear_round1="",
        bull_round2="",
        bear_round2="",
        current_round=1,
        bull_history="",
        bear_history="",
        history="",
        current_response="",
        judge_decision="",
        count=0,
    )
    if x is None:
        return y or default_state
    if y is None:
        return x

    result = {}
    all_keys = set(x.keys()) | set(y.keys())
    for key in all_keys:
        x_val = x.get(key, default_state.get(key))
        y_val = y.get(key, default_state.get(key))
        if isinstance(x_val, str) and isinstance(y_val, str):
            result[key] = y_val if y_val else x_val
        else:
            result[key] = y_val if y_val is not None else x_val

    return result


class AgentState(MessagesState):
    messages: Annotated[list[BaseMessage], merge_and_cap_messages]
    company_of_interest: str
    company_name: str
    company_name_resolved: bool
    trade_date: str
    sender: Annotated[str, take_last]

    market_report: Annotated[str, take_last]
    sentiment_report: Annotated[str, take_last]
    news_report: Annotated[str, take_last]
    raw_fundamentals_data: Annotated[str, take_last]
    foreign_language_report: Annotated[str, take_last]
    legal_report: Annotated[str, take_last]
    fundamentals_report: Annotated[str, take_last]
    auditor_report: Annotated[str, take_last]
    value_trap_report: Annotated[str, take_last]
    investment_debate_state: Annotated[InvestDebateState, merge_invest_debate_state]
    investment_plan: Annotated[str, take_last]
    valuation_params: Annotated[str, take_last]
    consultant_review: Annotated[str, take_last]
    trader_investment_plan: Annotated[str, take_last]
    risk_debate_state: Annotated[RiskDebateState, merge_risk_state]
    final_trade_decision: Annotated[str, take_last]
    tools_called: Annotated[dict[str, set[str]], merge_dicts]
    prompts_used: Annotated[dict[str, dict[str, str]], merge_dicts]
    artifact_statuses: Annotated[dict[str, dict[str, Any]], merge_dicts]
    consultant_tool_failures: Annotated[int, take_last]
    red_flags: Annotated[list[dict[str, Any]], merge_flag_lists]
    pre_screening_result: Annotated[str, take_last]
    chart_paths: Annotated[dict[str, str], take_last]
    macro_context_injected_into_news: Annotated[bool, take_last]
