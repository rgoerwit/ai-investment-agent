"""Supplemental parser families for legal, moat, capital-efficiency, and consultant checks."""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from src.data_block_utils import extract_last_data_block
from src.validators.metric_extractor import parse_ratio_or_percent

logger = structlog.get_logger(__name__)

_CONSULTANT_GROWTH_QUALITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\borganic\s+vs\.?\s+acquired\b", re.IGNORECASE),
    re.compile(
        r"\bgrowth quality\b.*\b(?:inferred|not proven|unknown|unproven)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bacquisition-led growth\b", re.IGNORECASE),
    re.compile(r"\bm&a illusion\b", re.IGNORECASE),
    re.compile(
        r"\b(?:incremental roic|incremental return(?:s)?|synerg(?:y|ies))\b.*\b(?:unknown|unproven|not proven|not demonstrated|missing)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:accretive|value-creating)\b.*\b(?:not proven|unproven|unknown)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:recurring revenue|service mix|maintenance-as-a-service)\b.*\b(?:not evidenced|unsupported|unverified|unverifiable|not proven)\b",
        re.IGNORECASE,
    ),
)

_CONSULTANT_TRANSIENT_STRENGTH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:one-time|non-recurring|nonoperating|non-operating)\b", re.IGNORECASE
    ),
    re.compile(r"\b(?:asset sale|division sale|gain on sale)\b", re.IGNORECASE),
    re.compile(r"\b(?:legal settlement|settlement gain)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:regulatory windfall|government subsidy|subsidy windfall)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:restructuring gain|restructuring charge)\b", re.IGNORECASE),
)


def extract_legal_risks(legal_report: str) -> dict[str, Any]:
    """Extract legal/tax risk data from Legal Counsel output."""
    risks: dict[str, Any] = {
        "pfic_status": None,
        "pfic_evidence": None,
        "vie_structure": None,
        "vie_evidence": None,
        "cmic_status": None,
        "cmic_evidence": None,
        "other_regulatory_risks": [],
        "country": None,
        "sector": None,
    }

    if not legal_report:
        return risks

    try:
        json_str = legal_report.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_lines: list[str] = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.startswith("```") and in_block:
                    break
                if in_block:
                    json_lines.append(line)
            json_str = "\n".join(json_lines)

        data = json.loads(json_str)
        risks["pfic_status"] = data.get("pfic_status")
        risks["pfic_evidence"] = data.get("pfic_evidence")
        risks["vie_structure"] = data.get("vie_structure")
        risks["vie_evidence"] = data.get("vie_evidence")
        risks["cmic_status"] = data.get("cmic_status")
        risks["cmic_evidence"] = data.get("cmic_evidence")
        risks["other_regulatory_risks"] = data.get("other_regulatory_risks") or []
        risks["country"] = data.get("country")
        risks["sector"] = data.get("sector")
        logger.debug(
            "legal_risks_parsed_json",
            pfic_status=risks["pfic_status"],
            vie_structure=risks["vie_structure"],
            cmic_status=risks["cmic_status"],
        )
        return risks
    except json.JSONDecodeError:
        logger.debug("legal_report_not_json_falling_back_to_regex")

    pfic_match = re.search(
        r'"?pfic_status"?\s*:\s*"?(CLEAN|UNCERTAIN|PROBABLE|N/A)"?',
        legal_report,
        re.IGNORECASE,
    )
    if pfic_match:
        risks["pfic_status"] = pfic_match.group(1).upper()

    vie_match = re.search(
        r'"?vie_structure"?\s*:\s*"?(YES|NO|N/A)"?', legal_report, re.IGNORECASE
    )
    if vie_match:
        risks["vie_structure"] = vie_match.group(1).upper()

    cmic_match = re.search(
        r'"?cmic_status"?\s*:\s*"?(FLAGGED|UNCERTAIN|CLEAR|N/A)"?',
        legal_report,
        re.IGNORECASE,
    )
    if cmic_match:
        risks["cmic_status"] = cmic_match.group(1).upper()

    return risks


def extract_value_trap_score(value_trap_report: str) -> dict[str, Any]:
    """Extract key metrics from the value-trap detector output."""
    metrics: dict[str, Any] = {
        "score": None,
        "verdict": None,
        "trap_risk": None,
        "activist_present": None,
        "insider_trend": None,
        "has_catalyst": False,
        "capital_allocation_rating": None,
        "buyback_context": None,
        "payout_trend": None,
        "cash_position": None,
        "mid_term_plan": None,
    }

    if not value_trap_report:
        return metrics
    if not isinstance(value_trap_report, str):
        try:
            value_trap_report = str(value_trap_report)
        except Exception:
            return metrics

    score_match = re.search(
        r"SCORE:\s*(\d+)(?:/100|%)?", value_trap_report, re.IGNORECASE
    )
    if score_match:
        metrics["score"] = max(0, min(100, int(score_match.group(1))))

    verdict_match = re.search(
        r"VERDICT:\s*(TRAP|CAUTIOUS|WATCHABLE|ALIGNED)",
        value_trap_report,
        re.IGNORECASE,
    )
    if verdict_match:
        metrics["verdict"] = verdict_match.group(1).upper()

    risk_match = re.search(
        r"TRAP_RISK:\s*(HIGH|MEDIUM|LOW)", value_trap_report, re.IGNORECASE
    )
    if risk_match:
        metrics["trap_risk"] = risk_match.group(1).upper()

    activist_match = re.search(
        r"ACTIVIST_PRESENT:\s*(YES|NO|RUMORED)", value_trap_report, re.IGNORECASE
    )
    if activist_match:
        metrics["activist_present"] = activist_match.group(1).upper()

    insider_match = re.search(
        r"INSIDER_TREND:\s*(NET_BUYER|NET_SELLER|NEUTRAL|UNKNOWN)",
        value_trap_report,
        re.IGNORECASE,
    )
    if insider_match:
        metrics["insider_trend"] = insider_match.group(1).upper()

    capital_allocation_match = re.search(
        r"RATING:\s*(POOR|MIXED|GOOD|UNKNOWN)", value_trap_report, re.IGNORECASE
    )
    if capital_allocation_match:
        metrics["capital_allocation_rating"] = capital_allocation_match.group(1).upper()

    for field, key in (
        ("BUYBACK_CONTEXT", "buyback_context"),
        ("PAYOUT_TREND", "payout_trend"),
        ("CASH_POSITION", "cash_position"),
        ("MID_TERM_PLAN", "mid_term_plan"),
    ):
        match = re.search(
            rf"{field}:\s*(.+?)(?:\n|$)", value_trap_report, re.IGNORECASE
        )
        if match:
            value = match.group(1).strip()
            if value.upper() not in ("NONE", "N/A"):
                metrics[key] = value

    catalysts_section = re.search(
        r"CATALYSTS:(.+?)(?:KEY_RISKS:|$)", value_trap_report, re.DOTALL
    )
    if catalysts_section:
        catalyst_text = catalysts_section.group(1)
        if re.search(
            r"(?:INDEX_CANDIDATE|ACTIVIST_RUMOR|RESTRUCTURING|MID_TERM_PLAN):\s*(?!NONE)[A-Za-z]",
            catalyst_text,
        ):
            metrics["has_catalyst"] = True

    logger.debug(
        "value_trap_metrics_extracted",
        score=metrics["score"],
        verdict=metrics["verdict"],
        trap_risk=metrics["trap_risk"],
    )
    return metrics


def extract_moat_signals(fundamentals_report: str) -> dict[str, Any]:
    """Extract moat signal metrics from the fundamentals DATA_BLOCK."""
    metrics: dict[str, Any] = {
        "margin_stability": None,
        "margin_cv": None,
        "margin_avg": None,
        "cash_conversion": None,
        "cfo_ni_avg": None,
    }

    if not fundamentals_report:
        return metrics
    if not isinstance(fundamentals_report, str):
        try:
            fundamentals_report = str(fundamentals_report)
        except Exception:
            return metrics

    data_block = extract_last_data_block(fundamentals_report)
    if not data_block:
        return metrics

    stability_match = re.search(
        r"MOAT_MARGIN_STABILITY:\s*(HIGH|MEDIUM|LOW)", data_block, re.IGNORECASE
    )
    if stability_match:
        metrics["margin_stability"] = stability_match.group(1).upper()

    cash_match = re.search(
        r"MOAT_CASH_CONVERSION:\s*(STRONG|ADEQUATE|WEAK)", data_block, re.IGNORECASE
    )
    if cash_match:
        metrics["cash_conversion"] = cash_match.group(1).upper()

    cv_match = re.search(r"MOAT_MARGIN_CV:\s*([0-9]+\.?[0-9]*)", data_block)
    if cv_match:
        try:
            metrics["margin_cv"] = float(cv_match.group(1))
        except ValueError:
            pass

    avg_match = re.search(r"MOAT_GROSS_MARGIN_AVG:\s*([0-9]+\.?[0-9]*)%?", data_block)
    if avg_match:
        try:
            value = float(avg_match.group(1))
            metrics["margin_avg"] = value / 100 if value > 1 else value
        except ValueError:
            pass

    cfo_match = re.search(r"MOAT_CFO_NI_AVG:\s*([0-9]+\.?[0-9]*)", data_block)
    if cfo_match:
        try:
            metrics["cfo_ni_avg"] = float(cfo_match.group(1))
        except ValueError:
            pass

    logger.debug(
        "moat_signals_extracted",
        margin_stability=metrics["margin_stability"],
        cash_conversion=metrics["cash_conversion"],
    )
    return metrics


def extract_capital_efficiency_signals(fundamentals_report: str) -> dict[str, Any]:
    """Extract capital-efficiency signals from fundamentals DATA_BLOCK."""
    if not fundamentals_report or not isinstance(fundamentals_report, str):
        return {}

    signals: dict[str, Any] = {}
    data_block = extract_last_data_block(fundamentals_report)
    if not data_block:
        return {}

    roic_quality_match = re.search(
        r"ROIC_QUALITY:\s*(STRONG|ADEQUATE|WEAK|DESTRUCTIVE|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if roic_quality_match:
        value = roic_quality_match.group(1).upper()
        if value != "N/A":
            signals["roic_quality"] = value

    leverage_quality_match = re.search(
        r"LEVERAGE_QUALITY:\s*(GENUINE|CONSERVATIVE|SUSPECT|ENGINEERED|VALUE_DESTRUCTION|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if leverage_quality_match:
        value = leverage_quality_match.group(1).upper()
        if value != "N/A":
            signals["leverage_quality"] = value

    roic_match = re.search(
        r"ROIC_PERCENT:\s*(-?[\d.]+)([%]?)", data_block, re.IGNORECASE
    )
    if roic_match:
        try:
            value = float(roic_match.group(1))
            if roic_match.group(2):
                value = value / 100
            elif abs(value) >= 2.0:
                value = value / 100
            signals["roic"] = value
        except ValueError:
            pass

    ratio_match = re.search(r"ROE_ROIC_RATIO:\s*([\d.]+)", data_block, re.IGNORECASE)
    if ratio_match:
        try:
            signals["roe_roic_ratio"] = float(ratio_match.group(1))
        except ValueError:
            pass

    for field, key in (
        ("NET_CASH_TO_MARKET_CAP", "net_cash_to_market_cap"),
        ("CASH_TO_ASSETS", "cash_to_assets"),
    ):
        match = re.search(rf"{field}:\s*([^\n]+)", data_block, re.IGNORECASE)
        if match:
            value = parse_ratio_or_percent(match.group(1))
            if value is not None:
                signals[key] = value

    capex_to_da_match = re.search(r"CAPEX_TO_DA:\s*([^\n]+)", data_block, re.IGNORECASE)
    if capex_to_da_match:
        raw_value = capex_to_da_match.group(1).strip()
        if raw_value.upper() != "N/A":
            try:
                signals["capex_to_da"] = float(raw_value)
            except ValueError:
                pass

    capex_status_match = re.search(
        r"CAPEX_TO_DA_STATUS:\s*(UNDERINVESTING|MAINTENANCE|GROWTH_INVESTING|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if capex_status_match:
        value = capex_status_match.group(1).upper()
        if value != "N/A":
            signals["capex_to_da_status"] = value

    backlog_coverage_match = re.search(
        r"REVENUE_BACKLOG_COVERAGE:\s*([0-9]+(?:\.\d+)?)", data_block, re.IGNORECASE
    )
    if backlog_coverage_match:
        signals["revenue_backlog_coverage"] = float(backlog_coverage_match.group(1))

    capital_plan_match = re.search(
        r"CAPITAL_PLAN_STATUS:\s*(EXPLICIT|NONE|UNKNOWN|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if capital_plan_match:
        value = capital_plan_match.group(1).upper()
        if value != "N/A":
            signals["capital_plan_status"] = value

    return signals


def parse_consultant_conditions(consultant_review: str) -> dict[str, Any]:
    """Parse consultant output for verdict and material concerns."""
    result: dict[str, Any] = {
        "verdict": "UNKNOWN",
        "has_mandate_breach": False,
        "has_hard_stop": False,
        "concern_count": 0,
        "spot_check_discrepancies": [],
        "growth_quality_unproven": False,
        "transient_strength_unproven": False,
    }

    if not consultant_review:
        return result
    if not isinstance(consultant_review, str):
        try:
            consultant_review = str(consultant_review)
        except Exception:
            return result

    upper_review = consultant_review.upper()
    if "MAJOR CONCERNS" in upper_review or "MAJOR_CONCERNS" in upper_review:
        result["verdict"] = "MAJOR_CONCERNS"
    elif (
        "CONDITIONAL APPROVAL" in upper_review or "CONDITIONAL_APPROVAL" in upper_review
    ):
        result["verdict"] = "CONDITIONAL_APPROVAL"
    elif "APPROVED" in upper_review:
        result["verdict"] = "APPROVED"

    if "MANDATE BREACH" in upper_review or "MANDATE_BREACH" in upper_review:
        result["has_mandate_breach"] = True
    if "HARD STOP" in upper_review or "HARD_STOP" in upper_review:
        result["has_hard_stop"] = True

    discrepancy_matches = re.findall(
        r"SPOT_CHECK.*?→\s*DISCREPANCY.*",
        consultant_review,
        re.IGNORECASE,
    )
    result["spot_check_discrepancies"] = discrepancy_matches
    result["growth_quality_unproven"] = any(
        pattern.search(consultant_review)
        for pattern in _CONSULTANT_GROWTH_QUALITY_PATTERNS
    )
    result["transient_strength_unproven"] = any(
        pattern.search(consultant_review)
        for pattern in _CONSULTANT_TRANSIENT_STRENGTH_PATTERNS
    )

    concern_patterns = re.findall(
        r"(?:^|\n)\s*(?:\d+\.|[-•])\s+(?:Material|Critical|Significant|Concern|Error|Discrepancy)",
        consultant_review,
        re.IGNORECASE,
    )
    result["concern_count"] = len(concern_patterns)

    logger.debug(
        "consultant_conditions_parsed",
        verdict=result["verdict"],
        has_mandate_breach=result["has_mandate_breach"],
        has_hard_stop=result["has_hard_stop"],
        discrepancy_count=len(result["spot_check_discrepancies"]),
        growth_quality_unproven=result["growth_quality_unproven"],
        transient_strength_unproven=result["transient_strength_unproven"],
    )
    return result
