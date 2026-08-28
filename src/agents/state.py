import json
from typing import Annotated, Any, cast

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.tooling.structured_ingress import merge_structured_inputs

from .message_utils import message_agent_key, tool_call_ids

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
    bull_round1_handoff: dict[str, str]
    bear_round1_handoff: dict[str, str]
    handoff_telemetry: dict[str, Any]
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


def _message_units(messages: list[BaseMessage]) -> list[tuple[int, ...]]:
    """Group assistant tool calls and their results into indivisible units."""

    owned_calls: dict[tuple[str | None, str], int] = {}
    calls_by_id: dict[str, list[int]] = {}
    unit_indices: dict[int, list[int]] = {}
    standalone: list[int] = []
    for index, message in enumerate(messages):
        call_ids = tool_call_ids(message)
        if call_ids:
            unit_indices[index] = [index]
            owner = message_agent_key(message)
            for call_id in call_ids:
                owned_calls[(owner, call_id)] = index
                calls_by_id.setdefault(call_id, []).append(index)
            continue
        if isinstance(message, ToolMessage):
            owner = message_agent_key(message)
            call_index = owned_calls.get((owner, message.tool_call_id))
            if call_index is None:
                candidates = calls_by_id.get(message.tool_call_id, [])
                call_index = candidates[0] if len(candidates) == 1 else None
            if call_index is not None:
                unit_indices[call_index].append(index)
                continue
        standalone.append(index)

    units = [tuple(indices) for indices in unit_indices.values()]
    units.extend((index,) for index in standalone)
    return sorted(units, key=lambda unit: unit[0])


def _unit_agent_key(messages: list[BaseMessage], unit: tuple[int, ...]) -> str | None:
    for index in unit:
        if owner := message_agent_key(messages[index]):
            return owner
    return None


def _select_recent_units(
    units: list[tuple[int, ...]], *, message_limit: int
) -> set[int]:
    """Select recent complete units without splitting a tool exchange."""

    selected: set[int] = set()
    used = 0
    for unit in reversed(units):
        size = len(unit)
        if selected and used + size > message_limit:
            break
        selected.update(unit)
        used += size
        if used >= message_limit:
            break
    return selected


def merge_and_cap_messages(
    x: list[BaseMessage] | None, y: list[BaseMessage] | BaseMessage | None
) -> list[BaseMessage]:
    """Merge messages while retaining bounded, complete per-agent transcripts.

    Parallel analysts share this state field but models consume it per agent.
    Capping one global tail allowed one provider's permissive tool parsing to
    mask orphaned results. Retention therefore follows the same ownership
    boundary as invocation and treats each tool-call/result exchange atomically.
    """
    merged = cast(
        list[BaseMessage], add_messages(cast(Any, x or []), cast(Any, y or []))
    )
    if not merged:
        return []

    units = _message_units(merged)
    preserved_indices: set[int] = set()

    for idx, message in enumerate(merged):
        if isinstance(message, HumanMessage):
            preserved_indices.add(idx)
            break

    unit_by_index = {index: unit for unit in units for index in unit}
    for idx, message in enumerate(merged):
        if _is_provenance_tool_message(message):
            preserved_indices.update(unit_by_index[idx])

    units_by_owner: dict[str | None, list[tuple[int, ...]]] = {}
    for unit in units:
        owner = _unit_agent_key(merged, unit)
        units_by_owner.setdefault(owner, []).append(unit)
    for owner_units in units_by_owner.values():
        preserved_indices.update(
            _select_recent_units(owner_units, message_limit=MESSAGE_TAIL_LIMIT)
        )

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
        bull_round1_handoff={},
        bear_round1_handoff={},
        handoff_telemetry={},
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

    x_values = cast(dict[str, Any], x)
    y_values = cast(dict[str, Any], y)
    result: dict[str, Any] = {}
    all_keys = set(x_values) | set(y_values)
    for key in all_keys:
        # Partial node updates omit fields they do not own. Distinguish absence
        # from an explicit empty value before consulting defaults; otherwise a
        # Bear update carrying no Bull handoff can erase the Bull's parallel dict.
        if key not in y_values:
            result[key] = x_values[key]
            continue
        if key not in x_values:
            result[key] = y_values[key]
            continue
        x_val = x_values.get(key, default_state.get(key))
        y_val = y_values.get(key, default_state.get(key))
        if isinstance(x_val, str) and isinstance(y_val, str):
            result[key] = y_val if y_val else x_val
        else:
            result[key] = y_val if y_val is not None else x_val

    return cast(InvestDebateState, result)


class AgentState(TypedDict, total=False):
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
    management_guidance_evidence: Annotated[str, take_last]
    foreign_language_report: Annotated[str, take_last]
    legal_report: Annotated[str, take_last]
    fundamentals_report: Annotated[str, take_last]
    auditor_report: Annotated[str, take_last]
    auditor_budget: Annotated[dict[str, Any], take_last]
    research_budgets: Annotated[dict[str, dict[str, Any]], merge_dicts]
    value_trap_report: Annotated[str, take_last]
    investment_debate_state: Annotated[InvestDebateState, merge_invest_debate_state]
    investment_plan: Annotated[str, take_last]
    valuation_params: Annotated[str, take_last]
    apac_regional_report: Annotated[str, take_last]
    consultant_review: Annotated[str, take_last]
    trader_investment_plan: Annotated[str, take_last]
    risk_debate_state: Annotated[RiskDebateState, merge_risk_state]
    final_trade_decision: Annotated[str, take_last]
    tools_called: Annotated[dict[str, set[str]], merge_dicts]
    structured_inputs: Annotated[dict[str, dict[str, Any]], merge_structured_inputs]
    prompts_used: Annotated[dict[str, dict[str, str]], merge_dicts]
    artifact_statuses: Annotated[dict[str, dict[str, Any]], merge_dicts]
    consultant_tool_failures: Annotated[int, take_last]
    red_flags: Annotated[list[dict[str, Any]], merge_flag_lists]
    pre_screening_result: Annotated[str, take_last]
    chart_paths: Annotated[dict[str, str], take_last]
    macro_context_injected_into_news: Annotated[bool, take_last]
    entity_governance_card: Annotated[dict[str, Any], take_last]
    analysis_snapshot: Annotated[dict[str, Any], take_last]
    decision_trace: Annotated[dict[str, Any], take_last]
