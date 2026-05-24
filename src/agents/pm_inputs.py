from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
