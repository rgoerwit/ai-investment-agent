"""Supplemental warning and bonus flag generators for validator workflows."""

from __future__ import annotations

from typing import Any

import structlog

from src.validators.metric_extractor import extract_metrics
from src.validators.sector_classifier import FINANCIALS_SECTORS, Sector, detect_sector
from src.validators.supplemental_extractors import (
    extract_capital_efficiency_signals,
    extract_moat_signals,
    extract_value_trap_score,
)

logger = structlog.get_logger(__name__)


def detect_legal_flags(
    legal_risks: dict[str, Any], ticker: str = "UNKNOWN"
) -> list[dict]:
    """Detect legal/tax warning flags from Legal Counsel output."""
    warnings: list[dict[str, Any]] = []

    pfic_status = legal_risks.get("pfic_status")
    vie_structure = legal_risks.get("vie_structure")
    pfic_evidence = legal_risks.get("pfic_evidence") or "No evidence provided"

    if pfic_status == "PROBABLE":
        warnings.append(
            {
                "type": "PFIC_PROBABLE",
                "severity": "WARNING",
                "detail": f"Company likely classified as PFIC. Evidence: {pfic_evidence[:100]}...",
                "action": "RISK_PENALTY",
                "risk_penalty": 1.0,
                "rationale": "PFIC classification requires onerous US tax reporting (Form 8621). Mark-to-market or QEF election required. Not a viability issue, but increases compliance burden for US investors.",
            }
        )
        logger.info(
            "legal_flag_pfic_probable", ticker=ticker, evidence=pfic_evidence[:50]
        )
    elif pfic_status == "UNCERTAIN":
        warnings.append(
            {
                "type": "PFIC_UNCERTAIN",
                "severity": "WARNING",
                "detail": f"PFIC status unclear. Evidence: {pfic_evidence[:100]}...",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": "PFIC status cannot be confirmed. Company may use hedge language or is in a high-risk sector without clear disclosure. Recommend consulting tax advisor before investing.",
            }
        )
        logger.info(
            "legal_flag_pfic_uncertain", ticker=ticker, evidence=pfic_evidence[:50]
        )

    if vie_structure == "YES":
        vie_evidence = legal_risks.get("vie_evidence") or "VIE structure detected"
        warnings.append(
            {
                "type": "VIE_STRUCTURE",
                "severity": "WARNING",
                "detail": f"Company uses VIE contractual structure for China operations. {vie_evidence[:80]}",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": "VIE structure means investors own contracts, not equity. China regulatory risk if VIE agreements are invalidated. Common for China tech/education stocks but adds legal uncertainty.",
            }
        )
        logger.info(
            "legal_flag_vie_structure", ticker=ticker, evidence=vie_evidence[:50]
        )

    cmic_status = legal_risks.get("cmic_status")
    if cmic_status == "FLAGGED":
        cmic_evidence = legal_risks.get("cmic_evidence") or "NS-CMIC list match"
        warnings.append(
            {
                "type": "CMIC_FLAGGED",
                "severity": "HIGH",
                "detail": f"Company appears on NS-CMIC list. {cmic_evidence[:80]}",
                "action": "RISK_PENALTY",
                "risk_penalty": 2.0,
                "rationale": "US Executive Orders prohibit US persons from investing in NS-CMIC listed companies. Verify current OFAC status before investing. Restrictions may be modified by future executive orders.",
            }
        )
        logger.info(
            "legal_flag_cmic_flagged", ticker=ticker, evidence=cmic_evidence[:50]
        )
    elif cmic_status == "UNCERTAIN":
        cmic_evidence = legal_risks.get("cmic_evidence") or "Possible CMIC connection"
        warnings.append(
            {
                "type": "CMIC_UNCERTAIN",
                "severity": "WARNING",
                "detail": f"Possible CMIC connection. {cmic_evidence[:80]}",
                "action": "RISK_PENALTY",
                "risk_penalty": 1.0,
                "rationale": "Company may have ties to Chinese military-industrial complex. Recommend verifying against current OFAC NS-CMIC list before investing.",
            }
        )
        logger.info(
            "legal_flag_cmic_uncertain", ticker=ticker, evidence=cmic_evidence[:50]
        )

    other_risks = legal_risks.get("other_regulatory_risks") or []
    severity_penalties = {"HIGH": 1.5, "MEDIUM": 1.0, "LOW": 0.5}
    for risk in other_risks:
        if not isinstance(risk, dict):
            continue
        risk_type = risk.get("risk_type", "OTHER")
        description = risk.get("description", "Regulatory risk detected")
        severity = risk.get("severity", "MEDIUM").upper()
        penalty = severity_penalties.get(severity, 1.0)
        warnings.append(
            {
                "type": f"REGULATORY_{risk_type}",
                "severity": "WARNING" if severity != "HIGH" else "HIGH",
                "detail": f"{risk_type}: {description[:100]}",
                "action": "RISK_PENALTY",
                "risk_penalty": penalty,
                "rationale": f"Regulatory risk identified by Legal Counsel. Type: {risk_type}, Severity: {severity}. Review before investing.",
            }
        )
        logger.info(
            "legal_flag_other_regulatory",
            ticker=ticker,
            risk_type=risk_type,
            severity=severity,
        )

    return warnings


def detect_value_trap_flags(
    value_trap_report: str, ticker: str = "UNKNOWN"
) -> list[dict]:
    """Parse VALUE_TRAP_BLOCK for deterministic warning flags."""
    flags: list[dict[str, Any]] = []

    metrics = extract_value_trap_score(value_trap_report)
    score = metrics.get("score")
    verdict = metrics.get("verdict")
    has_catalyst = metrics.get("has_catalyst", False)
    activist_present = metrics.get("activist_present")

    if score is not None and score < 40:
        flags.append(
            {
                "type": "VALUE_TRAP_HIGH_RISK",
                "severity": "WARNING",
                "detail": f"Value Trap Score {score}/100 (< 40 threshold indicates probable trap)",
                "action": "RISK_PENALTY",
                "risk_penalty": 1.0,
                "rationale": "Low governance score suggests entrenched ownership, poor capital allocation, or no catalyst for re-rating.",
            }
        )
        logger.info(
            "value_trap_flag_high_risk", ticker=ticker, score=score, verdict=verdict
        )
    elif score is not None and score < 60:
        flags.append(
            {
                "type": "VALUE_TRAP_MODERATE_RISK",
                "severity": "WARNING",
                "detail": f"Value Trap Score {score}/100 (40-60 range indicates mixed signals)",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": "Moderate governance concerns. Some trap characteristics present but not conclusive. Monitor for catalyst development.",
            }
        )
        logger.info(
            "value_trap_flag_moderate_risk", ticker=ticker, score=score, verdict=verdict
        )

    if verdict == "TRAP" and not any(
        flag["type"] == "VALUE_TRAP_HIGH_RISK" for flag in flags
    ):
        flags.append(
            {
                "type": "VALUE_TRAP_VERDICT",
                "severity": "WARNING",
                "detail": "Value Trap Detector verdict: TRAP",
                "action": "RISK_PENALTY",
                "risk_penalty": 1.0,
                "rationale": "Agent assessment indicates high probability of value trap. Stock may remain cheap indefinitely without catalyst.",
            }
        )
        logger.info("value_trap_flag_verdict", ticker=ticker, verdict=verdict)

    if not has_catalyst and activist_present == "NO":
        flags.append(
            {
                "type": "NO_CATALYST_DETECTED",
                "severity": "WARNING",
                "detail": "No activist presence, no index candidacy, no restructuring signals",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": "Without a catalyst, cheap stocks can remain cheap. Value realization depends on external pressure or internal change.",
            }
        )
        logger.info(
            "value_trap_flag_no_catalyst",
            ticker=ticker,
            activist_present=activist_present,
        )

    return flags


def detect_moat_flags(fundamentals_report: str, ticker: str = "UNKNOWN") -> list[dict]:
    """Detect economic moat indicators and create bonus flags."""
    flags: list[dict[str, Any]] = []
    metrics = extract_moat_signals(fundamentals_report)
    margin_stability = metrics.get("margin_stability")
    cash_conversion = metrics.get("cash_conversion")
    margin_cv = metrics.get("margin_cv")
    cfo_ni_avg = metrics.get("cfo_ni_avg")

    if margin_stability == "HIGH" and cash_conversion == "STRONG":
        detail_parts: list[str] = []
        if margin_cv is not None:
            detail_parts.append(f"Margin CV: {margin_cv:.3f}")
        if cfo_ni_avg is not None:
            detail_parts.append(f"CFO/NI: {cfo_ni_avg:.2f}")
        detail = "; ".join(detail_parts) if detail_parts else "Multiple moat signals"
        flags.append(
            {
                "type": "MOAT_DURABLE_ADVANTAGE",
                "severity": "POSITIVE",
                "detail": f"Pricing power + earnings quality confirmed. {detail}",
                "action": "RISK_BONUS",
                "risk_penalty": -1.0,
                "rationale": "Company exhibits both stable gross margins (CV < 8%) and high cash conversion (CFO/NI > 90%) over multiple years. This combination suggests a durable competitive advantage with pricing power.",
            }
        )
        logger.info(
            "moat_flag_durable_advantage",
            ticker=ticker,
            margin_stability=margin_stability,
            cash_conversion=cash_conversion,
        )
        return flags

    if margin_stability == "HIGH":
        detail = (
            f"Gross margin CV: {margin_cv:.3f}" if margin_cv is not None else "CV < 8%"
        )
        flags.append(
            {
                "type": "MOAT_PRICING_POWER",
                "severity": "POSITIVE",
                "detail": f"Stable gross margins over 5 years. {detail}",
                "action": "RISK_BONUS",
                "risk_penalty": -0.5,
                "rationale": "Low gross margin volatility (CV < 8%) over 5 years suggests pricing power. Company can maintain margins without aggressive discounting, indicating competitive advantage.",
            }
        )
        logger.info("moat_flag_pricing_power", ticker=ticker, margin_cv=margin_cv)

    if cash_conversion == "STRONG":
        detail = (
            f"3Y avg CFO/NI: {cfo_ni_avg:.2f}" if cfo_ni_avg is not None else "> 0.90"
        )
        flags.append(
            {
                "type": "MOAT_EARNINGS_QUALITY",
                "severity": "POSITIVE",
                "detail": f"High cash conversion ratio. {detail}",
                "action": "RISK_BONUS",
                "risk_penalty": -0.5,
                "rationale": "CFO/Net Income ratio averaging > 90% over 3 years indicates reported earnings are converting to actual cash flow. Not relying on accounting accruals or channel stuffing.",
            }
        )
        logger.info("moat_flag_earnings_quality", ticker=ticker, cfo_ni_avg=cfo_ni_avg)

    return flags


def detect_capital_efficiency_flags(
    fundamentals_report: str,
    ticker: str = "UNKNOWN",
    value_trap_report: str | None = None,
    sector: Sector | None = None,
) -> list[dict]:
    """Detect capital-efficiency risk and bonus flags."""
    flags: list[dict[str, Any]] = []

    from src.config import config

    metrics = extract_capital_efficiency_signals(fundamentals_report)
    base_metrics = extract_metrics(fundamentals_report)
    value_trap_metrics = extract_value_trap_score(value_trap_report)
    if not metrics:
        return flags

    roic_quality = metrics.get("roic_quality")
    leverage_quality = metrics.get("leverage_quality")
    roic = metrics.get("roic")
    roe_roic_ratio = metrics.get("roe_roic_ratio")
    net_cash_to_mc = metrics.get("net_cash_to_market_cap")
    cash_to_assets = metrics.get("cash_to_assets")
    capex_to_da_status = metrics.get("capex_to_da_status")
    revenue_backlog_coverage = metrics.get("revenue_backlog_coverage")
    payout_ratio = base_metrics.get("payout_ratio")
    capital_plan_status = metrics.get("capital_plan_status")
    if capital_plan_status is None and value_trap_metrics.get("mid_term_plan"):
        capital_plan_status = "EXPLICIT"
    if sector is None:
        sector = detect_sector(fundamentals_report)

    if leverage_quality == "VALUE_DESTRUCTION":
        detail = f"ROIC: {roic:.1%}" if roic is not None else "Negative ROIC"
        flags.append(
            {
                "type": "CAPITAL_VALUE_DESTRUCTION",
                "severity": "CRITICAL",
                "detail": f"Negative operating returns masked by leverage. {detail}",
                "action": "REJECT_REVIEW",
                "risk_penalty": 1.5,
                "rationale": "Company has negative ROIC but positive ROE. This means the core business is destroying value while financial leverage creates the illusion of shareholder returns. Classic value trap pattern.",
            }
        )
        logger.info(
            "capital_flag_value_destruction",
            ticker=ticker,
            roic=roic,
            leverage_quality=leverage_quality,
        )
        return flags

    if leverage_quality == "ENGINEERED":
        ratio_str = f"ROE/ROIC: {roe_roic_ratio:.1f}x" if roe_roic_ratio else ""
        flags.append(
            {
                "type": "CAPITAL_ENGINEERED_RETURNS",
                "severity": "HIGH",
                "detail": f"Returns primarily from financial engineering. {ratio_str}",
                "action": "RISK_ADJUST",
                "risk_penalty": 1.0,
                "rationale": "ROE significantly exceeds ROIC (ratio > 3x), indicating shareholder returns come from leverage, buybacks, or capital structure rather than underlying business quality.",
            }
        )
        logger.info(
            "capital_flag_engineered_returns",
            ticker=ticker,
            roe_roic_ratio=roe_roic_ratio,
        )
    elif leverage_quality == "SUSPECT":
        ratio_str = f"ROE/ROIC: {roe_roic_ratio:.1f}x" if roe_roic_ratio else ""
        flags.append(
            {
                "type": "CAPITAL_SUSPECT_RETURNS",
                "severity": "MEDIUM",
                "detail": f"Moderate leverage amplification detected. {ratio_str}",
                "action": "RISK_ADJUST",
                "risk_penalty": 0.5,
                "rationale": "ROE moderately exceeds ROIC (ratio 2-3x). Returns partially driven by leverage rather than operational excellence.",
            }
        )
        logger.info(
            "capital_flag_suspect_returns", ticker=ticker, roe_roic_ratio=roe_roic_ratio
        )

    if roic_quality == "WEAK":
        roic_str = f"ROIC: {roic:.1%}" if roic is not None else ""
        flags.append(
            {
                "type": "CAPITAL_BELOW_HURDLE",
                "severity": "MEDIUM",
                "detail": f"Returns below cost of capital proxy. {roic_str}",
                "action": "RISK_ADJUST",
                "risk_penalty": 0.5,
                "rationale": "ROIC below 8% hurdle rate suggests the company may be destroying value on a risk-adjusted basis. Acceptable only with clear turnaround thesis and improving trajectory.",
            }
        )
        logger.info("capital_flag_below_hurdle", ticker=ticker, roic=roic)

    if roic_quality == "STRONG" and leverage_quality in ("GENUINE", "CONSERVATIVE"):
        roic_str = f"ROIC: {roic:.1%}" if roic is not None else ""
        flags.append(
            {
                "type": "CAPITAL_EFFICIENT",
                "severity": "POSITIVE",
                "detail": f"Strong genuine capital efficiency. {roic_str}",
                "action": "RISK_BONUS",
                "risk_penalty": -0.5,
                "rationale": "High ROIC (>15%) with ROE/ROIC ratio below 2x indicates returns driven by operational excellence rather than financial leverage. Suggests sustainable competitive advantage.",
            }
        )
        logger.info(
            "capital_flag_efficient",
            ticker=ticker,
            roic=roic,
            leverage_quality=leverage_quality,
        )

    excess_cash = (
        net_cash_to_mc is not None
        and net_cash_to_mc >= config.idle_cash_net_cash_to_mc_threshold
    ) or (
        cash_to_assets is not None
        and cash_to_assets >= config.idle_cash_cash_to_assets_threshold
    )
    weak_deployment = roic_quality in {"WEAK", "DESTRUCTIVE"} or (
        roic_quality == "ADEQUATE" and capex_to_da_status != "GROWTH_INVESTING"
    )
    weak_shareholder_return = (
        payout_ratio is None or payout_ratio < config.idle_cash_min_payout_ratio
    )
    mitigated = (
        capital_plan_status == "EXPLICIT"
        or capex_to_da_status == "GROWTH_INVESTING"
        or (revenue_backlog_coverage is not None and revenue_backlog_coverage >= 1.0)
    )
    severe_idle_cash = (
        net_cash_to_mc is not None
        and net_cash_to_mc >= config.idle_cash_severe_net_cash_to_mc_threshold
        and roic_quality in {"WEAK", "DESTRUCTIVE"}
        and capital_plan_status == "NONE"
        and (payout_ratio is None or payout_ratio < 10.0)
        and not mitigated
    )

    if sector in FINANCIALS_SECTORS:
        return flags

    if severe_idle_cash:
        flags.append(
            {
                "type": "CAPITAL_IDLE_CASH_SEVERE",
                "severity": "HIGH",
                "detail": "Extreme excess cash with weak deployment and no credible capital allocation plan.",
                "action": "RISK_ADJUST",
                "risk_penalty": 1.0,
                "rationale": "Large excess cash relative to market value combined with weak returns, weak shareholder distributions, and no explicit use plan suggests capital is being warehoused rather than deployed.",
            }
        )
        logger.info(
            "capital_flag_idle_cash_severe",
            ticker=ticker,
            net_cash_to_market_cap=net_cash_to_mc,
            cash_to_assets=cash_to_assets,
        )
    elif (
        excess_cash
        and weak_deployment
        and weak_shareholder_return
        and capital_plan_status == "NONE"
        and not mitigated
    ):
        flags.append(
            {
                "type": "CAPITAL_IDLE_CASH_RISK",
                "severity": "MEDIUM",
                "detail": "Excess cash with weak deployment, weak payout, and no credible capital allocation plan.",
                "action": "RISK_ADJUST",
                "risk_penalty": 0.5,
                "rationale": "Cash-rich balance sheets are not automatically a problem, but retained capital with weak ROIC, low payout, and no disclosed deployment plan can become a value trap.",
            }
        )
        logger.info(
            "capital_flag_idle_cash_risk",
            ticker=ticker,
            net_cash_to_market_cap=net_cash_to_mc,
            cash_to_assets=cash_to_assets,
        )

    return flags


def detect_consultant_flags(
    conditions: dict[str, Any], ticker: str = "UNKNOWN"
) -> list[dict]:
    """Generate risk flags from parsed consultant conditions."""
    flags: list[dict[str, Any]] = []
    verdict = conditions.get("verdict", "UNKNOWN")
    discrepancies = conditions.get("spot_check_discrepancies", [])

    if conditions.get("has_hard_stop"):
        flags.append(
            {
                "type": "CONSULTANT_HARD_STOP",
                "severity": "CRITICAL",
                "detail": "Consultant issued HARD STOP — restricted security",
                "action": "AUTO_REJECT",
                "risk_penalty": 3.0,
                "rationale": "External consultant flagged a hard stop condition (e.g., CMIC restricted list). Position must not be initiated.",
            }
        )
        logger.info("consultant_flag_hard_stop", ticker=ticker)
        return flags

    if conditions.get("has_mandate_breach"):
        flags.append(
            {
                "type": "CONSULTANT_MANDATE_BREACH",
                "severity": "HIGH",
                "detail": "Consultant flagged MANDATE BREACH",
                "action": "RISK_PENALTY",
                "risk_penalty": 2.0,
                "rationale": "External consultant identified a mandate compliance issue (e.g., PFIC threshold, jurisdiction risk). PM must explicitly address this before proceeding.",
            }
        )
        logger.info("consultant_flag_mandate_breach", ticker=ticker)

    if verdict == "MAJOR_CONCERNS":
        flags.append(
            {
                "type": "CONSULTANT_MAJOR_CONCERNS",
                "severity": "HIGH",
                "detail": "Consultant raised MAJOR CONCERNS — PM must address each",
                "action": "RISK_PENALTY",
                "risk_penalty": 1.5,
                "rationale": "External consultant found material issues with the analysis. These could be factual errors, severe biases, or fundamentally flawed synthesis. PM decision should reflect these concerns.",
            }
        )
        logger.info("consultant_flag_major_concerns", ticker=ticker)
    elif verdict == "CONDITIONAL_APPROVAL":
        flags.append(
            {
                "type": "CONSULTANT_CONDITIONAL",
                "severity": "WARNING",
                "detail": "Consultant gave CONDITIONAL APPROVAL — conditions must be met",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": "External consultant approved with conditions. PM should verify conditions are addressed in the final decision rationale.",
            }
        )
        logger.info("consultant_flag_conditional", ticker=ticker)

    if conditions.get("growth_quality_unproven"):
        flags.append(
            {
                "type": "CONSULTANT_GROWTH_QUALITY_UNPROVEN",
                "severity": "WARNING",
                "detail": "Consultant says growth durability is unproven (organic vs acquired / synergy evidence unresolved)",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": "External consultant could not verify that recent growth is organic, accretive, or supported by recurring-revenue evidence. Treat current strength as provisional.",
            }
        )
        logger.info("consultant_flag_growth_quality_unproven", ticker=ticker)

    if conditions.get("transient_strength_unproven"):
        flags.append(
            {
                "type": "CONSULTANT_TRANSIENT_STRENGTH",
                "severity": "WARNING",
                "detail": "Consultant flagged possible one-time or non-operating strength distortion",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": "External consultant identified a named non-recurring driver that may be inflating current strength. Do not treat this as durable baseline performance without further proof.",
            }
        )
        logger.info("consultant_flag_transient_strength", ticker=ticker)

    if discrepancies:
        disc_penalty = min(len(discrepancies) * 0.5, 1.5)
        disc_details = "; ".join(d.strip() for d in discrepancies[:3])
        flags.append(
            {
                "type": "CONSULTANT_DATA_DISCREPANCY",
                "severity": "WARNING",
                "detail": f"{len(discrepancies)} spot-check discrepancies: {disc_details}",
                "action": "RISK_PENALTY",
                "risk_penalty": disc_penalty,
                "rationale": "Consultant's independent spot-checks found discrepancies between DATA_BLOCK values and direct API queries. This suggests potential data quality issues that should be investigated.",
            }
        )
        logger.info(
            "consultant_flag_discrepancies",
            ticker=ticker,
            count=len(discrepancies),
            penalty=disc_penalty,
        )

    return flags
