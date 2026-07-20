"""Deterministic accounting calculations exposed to the forensic Auditor."""

from __future__ import annotations

import json
from datetime import date
from typing import Annotated

from langchain_core.tools import tool


@tool
def validate_forensic_evidence(
    analysis_date: str,
    report_date: str,
    balance_sheet_found: bool,
    income_statement_found: bool,
    cash_flow_statement_found: bool,
    statement_periods: list[str],
    statement_scopes: list[str],
    auditor_opinion_found: bool,
) -> str:
    """Apply the forensic freshness/completeness gates deterministically."""
    try:
        analysis = date.fromisoformat(analysis_date)
        report = date.fromisoformat(report_date)
    except ValueError:
        return json.dumps(
            {"status": "INSUFFICIENT_DATA", "reasons": ["REPORT_DATE_UNVERIFIED"]}
        )
    if report > analysis:
        return json.dumps(
            {"status": "INSUFFICIENT_DATA", "reasons": ["REPORT_DATE_IN_FUTURE"]}
        )

    age_months = (analysis.year - report.year) * 12 + analysis.month - report.month
    if analysis.day < report.day:
        age_months -= 1
    reasons: list[str] = []
    if age_months > 18:
        reasons.append("STALE_DATA")
    if not all(
        (balance_sheet_found, income_statement_found, cash_flow_statement_found)
    ):
        reasons.append("STATEMENT_TRIAD_INCOMPLETE")

    periods = {value.strip().casefold() for value in statement_periods if value.strip()}
    scopes = {value.strip().casefold() for value in statement_scopes if value.strip()}
    if len(periods) != 1:
        reasons.append("PERIOD_MISMATCH")
    if len(scopes) != 1:
        reasons.append("SCOPE_MISMATCH")
    if reasons:
        status = "INSUFFICIENT_DATA"
    elif not auditor_opinion_found:
        status = "PARTIAL_DATA"
        reasons.append("AUDITOR_OPINION_UNVERIFIED")
    else:
        status = "COMPLETE"

    return json.dumps(
        {"status": status, "reasons": reasons, "age_months": age_months},
        sort_keys=True,
    )


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


@tool
def calculate_forensic_ratios(
    period: Annotated[str, "Reporting period shared by every supplied value"],
    scope: Annotated[str, "Statement scope, such as consolidated or standalone"],
    revenue: float | None = None,
    cogs: float | None = None,
    accounts_receivable: float | None = None,
    inventory: float | None = None,
    accounts_payable: float | None = None,
    net_income: float | None = None,
    operating_cash_flow: float | None = None,
    free_cash_flow: float | None = None,
    total_assets: float | None = None,
    ebit: float | None = None,
    interest_expense: float | None = None,
    goodwill: float | None = None,
    intangibles: float | None = None,
    deferred_tax_assets: float | None = None,
    interest_income: float | None = None,
    cash_and_equivalents: float | None = None,
    restricted_cash: float | None = None,
    other_receivables: float | None = None,
    days_in_period: int = 365,
) -> str:
    """Compute forensic ratios without model arithmetic or denominator guessing."""
    if not period.strip() or not scope.strip():
        return json.dumps(
            {"status": "INSUFFICIENT_DATA", "reason": "PERIOD_OR_SCOPE_UNVERIFIED"}
        )
    if days_in_period <= 0 or days_in_period > 366:
        return json.dumps(
            {"status": "INSUFFICIENT_DATA", "reason": "INVALID_PERIOD_LENGTH"}
        )

    metrics = {
        "ocf_to_ni": _ratio(operating_cash_flow, net_income),
        "paper_profit": (
            _ratio((net_income or 0) - (operating_cash_flow or 0), total_assets)
            if net_income is not None and operating_cash_flow is not None
            else None
        ),
        "fcf_to_ni": _ratio(free_cash_flow, net_income),
        "dso": (
            value * days_in_period
            if (value := _ratio(accounts_receivable, revenue)) is not None
            else None
        ),
        "dio": (
            value * days_in_period
            if (value := _ratio(inventory, cogs)) is not None
            else None
        ),
        "dpo": (
            value * days_in_period
            if (value := _ratio(accounts_payable, cogs)) is not None
            else None
        ),
        "zombie_ratio": _ratio(ebit, interest_expense),
        "goodwill_to_assets": _ratio(goodwill, total_assets),
        "intangibles_to_assets": _ratio(intangibles, total_assets),
        "dta_to_assets": _ratio(deferred_tax_assets, total_assets),
        "ghost_yield": _ratio(interest_income, cash_and_equivalents),
        "restricted_cash_ratio": _ratio(restricted_cash, cash_and_equivalents),
        "trash_bin_ratio": _ratio(other_receivables, total_assets),
    }
    if all(value is None for value in metrics.values()):
        status = "INSUFFICIENT_DATA"
        reason = "REQUIRED_LINE_ITEMS_MISSING"
    else:
        status = "CALCULATED"
        reason = None
    return json.dumps(
        {
            "status": status,
            "reason": reason,
            "period": period,
            "scope": scope,
            "days_in_period": days_in_period,
            "metrics": {
                key: round(value, 6) if value is not None else None
                for key, value in metrics.items()
            },
            "rule": "N/A means missing or non-positive denominator; no values inferred",
        },
        sort_keys=True,
    )
