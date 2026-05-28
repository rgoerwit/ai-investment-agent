from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

GOVERNANCE_CARD_FIELD = "entity_governance_card"

DIRECT_PM_INPUT_FIELDS: tuple[str, ...] = (
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "value_trap_report",
    "investment_plan",
    "apac_regional_report",
    "consultant_review",
    "auditor_report",
    "legal_report",
    "valuation_params",
    "trader_investment_plan",
    GOVERNANCE_CARD_FIELD,
)

RISK_DEBATE_FIELD = "risk_debate_state"
RISK_DEBATE_RESPONSE_FIELDS: tuple[str, ...] = (
    "current_risky_response",
    "current_safe_response",
    "current_neutral_response",
)

DIRECT_PM_INPUTS: frozenset[str] = frozenset(
    (*DIRECT_PM_INPUT_FIELDS, RISK_DEBATE_FIELD)
)


def risk_debate_content(state: Mapping[str, Any]) -> str:
    risk_state = state.get(RISK_DEBATE_FIELD, {}) or {}
    if not isinstance(risk_state, Mapping):
        return ""
    parts = [
        str(risk_state.get(field) or "")
        for field in RISK_DEBATE_RESPONSE_FIELDS
        if risk_state.get(field)
    ]
    return "\n".join(parts)


def risk_debate_present(state: Mapping[str, Any]) -> bool:
    return bool(risk_debate_content(state))


def governance_card_present(state: Mapping[str, Any]) -> bool:
    card = state.get(GOVERNANCE_CARD_FIELD)
    return isinstance(card, Mapping) and bool(card.get("ticker"))


PM_INPUT_PRESENCE: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    GOVERNANCE_CARD_FIELD: governance_card_present,
}


def pm_input_present(state: Mapping[str, Any], field: str) -> bool:
    """Return whether a direct PM input is usable."""

    predicate = PM_INPUT_PRESENCE.get(field)
    if predicate:
        return predicate(state)

    from src.runtime_diagnostics import get_valid_artifact_content

    return bool(get_valid_artifact_content(state, field))
