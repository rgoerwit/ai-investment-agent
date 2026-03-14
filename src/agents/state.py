from typing import Annotated, Any

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


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


def merge_dicts(x: dict | None, y: dict | None) -> dict:
    """Reducer: merges dictionaries. Used for parallel agent state updates."""
    if x is None:
        return y or {}
    if y is None:
        return x
    return {**x, **y}


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
    red_flags: Annotated[list[dict[str, Any]], take_last]
    pre_screening_result: Annotated[str, take_last]
    chart_paths: Annotated[dict[str, str], take_last]
