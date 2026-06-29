"""Supplemental warning and bonus flag generators for validator workflows."""

from __future__ import annotations

from typing import Any

import structlog

from src.validators.financial_rules import contains_transient_strength_marker
from src.validators.metric_extractor import extract_metrics
from src.validators.sector_classifier import FINANCIALS_SECTORS, Sector, detect_sector
from src.validators.supplemental_extractors import (
    extract_capital_efficiency_signals,
    extract_material_unverified_operating_signal,
    extract_moat_signals,
    extract_value_trap_score,
)

logger = structlog.get_logger(__name__)


def _truncate_at_boundary(text: str, limit: int = 100) -> str:
    """Truncate to ``limit`` chars at the nearest sentence/word boundary.

    Flag ``detail`` strings are surfaced verbatim in the investment memo and the
    Red-Flag section; a raw ``text[:limit]`` slice cuts mid-word ("...for US tax
    purposes, and "). This ends cleanly on a separator and appends a single
    ellipsis only when content was actually dropped.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", "; ", ", ", " "):
        idx = cut.rfind(sep)
        if idx > limit // 2:
            return cut[:idx].rstrip(" ,;") + "…"
    return cut.rstrip() + "…"


def _peak_or_transient_blocker(
    fundamentals_report: str,
    *,
    base_metrics: dict[str, Any] | None = None,
) -> str | None:
    """Return a reason string if durable-quality bonuses should be suppressed.

    Moat and capital-efficiency bonuses are computed from margin stability,
    cash conversion, and current ROIC — the same metrics a cyclical peak or a
    one-time event inflates. When the earnings base is flagged suspect we
    suppress the bonus (set it to 0.0) rather than letting it net against a
    concurrent peak/transient warning. The conditions mirror the existing
    ``CYCLICAL_PEAK_WARNING`` / ``TRANSIENT_STRENGTH_DISTORTION`` semantics in
    ``financial_rules`` so suppression fires exactly when the earnings base is
    already questioned.
    """
    metrics = (
        base_metrics
        if base_metrics is not None
        else extract_metrics(fundamentals_report)
    )
    if metrics.get("cycle_position") == "PEAK":
        return "DATA_BLOCK CYCLE_POSITION: PEAK"
    roa_current = metrics.get("roa_current")
    roa_5y_avg = metrics.get("roa_5y_avg")
    trend = metrics.get("profitability_trend")
    if (
        roa_current is not None
        and roa_5y_avg
        and roa_5y_avg > 0
        and roa_current / roa_5y_avg > 1.5
        and trend in {"UNSTABLE", "DECLINING"}
    ):
        return (
            f"peak-cycle distortion (ROA {roa_current:.1f}% vs 5Y avg "
            f"{roa_5y_avg:.1f}% with {trend} trend)"
        )
    if contains_transient_strength_marker(fundamentals_report):
        return "transient-strength / one-time event marker present"
    return None


def detect_material_operating_signal_flags(
    report: str, ticker: str = "UNKNOWN"
) -> list[dict]:
    """Flag a large, unverified operating decline as BUY-blocking until verified.

    A narrative claim such as "operating profit down 53% YoY" is price-relevant
    even before it is confirmed in structured data. It must not become a generic
    low-weight UNVERIFIABLE item, nor an auto-SELL: it blocks BUY and forces
    primary-source verification (0.0 tally weight — unverified is not confirmed
    risk).
    """
    flags: list[dict[str, Any]] = []
    signal = extract_material_unverified_operating_signal(report)
    if not signal:
        return flags
    flags.append(
        {
            "type": "MATERIAL_UNVERIFIED_OPERATING_SIGNAL",
            "severity": "WARNING",
            "detail": (
                f"Narrative reports {signal['metric']} down {signal['decline_pct']:.1f}% "
                "— not yet verified against a primary source."
            ),
            "action": "REVIEW",
            "risk_penalty": 0.0,
            "blocks_buy": True,
            "rationale": "A large operating-profit/earnings decline is material even when unverified. Do not score it as confirmed risk (0.0 tally — unverified is not confirmed), but BLOCK BUY and require primary-source verification before initiating. Never auto-SELL/REJECT on this signal alone.",
        }
    )
    logger.debug(
        "material_unverified_operating_signal",
        ticker=ticker,
        metric=signal["metric"],
        decline_pct=signal["decline_pct"],
    )
    return flags


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
                "detail": f"Company likely classified as PFIC. Evidence: {_truncate_at_boundary(pfic_evidence)}",
                "action": "RISK_PENALTY",
                "risk_penalty": 1.0,
                "rationale": "PFIC classification requires onerous US tax reporting (Form 8621). Mark-to-market or QEF election required. Not a viability issue, but increases compliance burden for US investors.",
            }
        )
        logger.debug(
            "legal_flag_pfic_probable", ticker=ticker, evidence=pfic_evidence[:50]
        )
    elif pfic_status == "UNCERTAIN":
        warnings.append(
            {
                "type": "PFIC_UNCERTAIN",
                "severity": "WARNING",
                "detail": f"PFIC status unclear. Evidence: {_truncate_at_boundary(pfic_evidence)}",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": "PFIC status cannot be confirmed. Company may use hedge language or is in a high-risk sector without clear disclosure. Recommend consulting tax advisor before investing.",
            }
        )
        logger.debug(
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
        logger.debug(
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
        logger.debug(
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
        logger.debug(
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
        logger.debug(
            "legal_flag_other_regulatory",
            ticker=ticker,
            risk_type=risk_type,
            severity=severity,
        )

    return warnings


def _reinvestment_context_contradicts_trap(
    capital_context: dict[str, Any] | None,
) -> bool:
    """True when the DATA_BLOCK shows genuine growth investment, not value destruction.

    The Value-Trap Detector runs in parallel with Fundamentals and is blind to ROIC /
    capex status, so it can mislabel heavy reinvestment as "POOR" capital allocation and
    sink the score below 40 (APR.WA). When the Senior Fundamentals DATA_BLOCK independently
    confirms GROWTH_INVESTING capex AND ADEQUATE/STRONG ROIC AND an EXPLICIT capital plan,
    the "trap" rationale is contradicted and the penalty is downgraded (never removed —
    a cash-hoarder is not GROWTH_INVESTING, so this cannot whitewash a true trap).
    """
    if not capital_context:
        return False
    return (
        (capital_context.get("capex_to_da_status") or "").upper() == "GROWTH_INVESTING"
        and (capital_context.get("roic_quality") or "").upper()
        in {"ADEQUATE", "STRONG"}
        and (capital_context.get("capital_plan_status") or "").upper() == "EXPLICIT"
    )


def detect_value_trap_flags(
    value_trap_report: str,
    ticker: str = "UNKNOWN",
    *,
    m_and_a_status: str | None = None,
    capital_context: dict[str, Any] | None = None,
) -> list[dict]:
    """Parse VALUE_TRAP_BLOCK for deterministic warning flags.

    ``m_and_a_status`` (from the Senior Fundamentals DATA_BLOCK) is an
    independent catalyst signal: an active tender/offer is, by definition, a
    re-rating catalyst, so it suppresses ``NO_CATALYST_DETECTED`` even when the
    Value-Trap Detector's own governance signals show no activist/index/
    restructuring catalyst. Without this, an active-takeover name (e.g. GAMA.L)
    self-contradicts — flagged "no catalyst" while the DATA_BLOCK reports
    ``M_AND_A_STATUS: ACTIVE_TENDER``.

    ``capital_context`` (parsed DATA_BLOCK capex/ROIC/plan fields) lets a HIGH/TRAP
    penalty be **downgraded** to MODERATE when the trap was driven by reinvestment that
    the DATA_BLOCK independently shows is genuine growth investment (see
    ``_reinvestment_context_contradicts_trap``). ``NO_CATALYST_DETECTED`` is **not**
    dropped here — a capex plan explains reinvestment but is not itself a near-term catalyst.
    """
    flags: list[dict[str, Any]] = []

    metrics = extract_value_trap_score(value_trap_report)
    score = metrics.get("score")
    verdict = metrics.get("verdict")
    has_catalyst = metrics.get("has_catalyst", False)
    activist_present = metrics.get("activist_present")
    active_ma = (m_and_a_status or "").strip().upper() in {
        "ACTIVE_TENDER",
        "ACTIVE_OFFER",
    }
    # Only downgrade when the trap was actually driven by a POOR capital-allocation
    # rating AND the DATA_BLOCK contradicts it (genuine growth investment). A trap
    # scored low for governance/concentration/no-catalyst (RATING != POOR) must NOT
    # be whitewashed just because the company also happens to be investing.
    reinvestment_downgrade = metrics.get(
        "capital_allocation_rating"
    ) == "POOR" and _reinvestment_context_contradicts_trap(capital_context)

    if score is not None and score < 40:
        if reinvestment_downgrade:
            flags.append(
                {
                    "type": "VALUE_TRAP_MODERATE_RISK",
                    "severity": "WARNING",
                    "detail": (
                        f"Value Trap Score {score}/100 downgraded HIGH->MODERATE: "
                        "DATA_BLOCK shows GROWTH_INVESTING capex + ADEQUATE/STRONG ROIC "
                        "+ EXPLICIT capital plan (reinvestment, not value destruction)"
                    ),
                    "action": "RISK_PENALTY",
                    "risk_penalty": 0.5,
                    "rationale": "Value-Trap Detector runs blind to ROIC/capex and can mislabel reinvestment as poor allocation. The Senior Fundamentals DATA_BLOCK independently confirms genuine growth investment with adequate returns, so the HIGH-risk trap penalty is downgraded (not removed) to MODERATE.",
                }
            )
            logger.info(
                "value_trap_high_downgraded_reinvestment_context",
                ticker=ticker,
                score=score,
                verdict=verdict,
            )
        else:
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
            logger.debug(
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
        logger.debug(
            "value_trap_flag_moderate_risk", ticker=ticker, score=score, verdict=verdict
        )

    if verdict == "TRAP" and not any(
        flag["type"] == "VALUE_TRAP_HIGH_RISK" for flag in flags
    ):
        already_moderate = any(
            flag["type"] == "VALUE_TRAP_MODERATE_RISK" for flag in flags
        )
        if reinvestment_downgrade:
            # Reinvestment context contradicts the TRAP verdict: do not add a full
            # +1.0 verdict penalty. If a MODERATE downgrade flag already represents the
            # residual risk, add nothing; otherwise record a single MODERATE penalty.
            if not already_moderate:
                flags.append(
                    {
                        "type": "VALUE_TRAP_MODERATE_RISK",
                        "severity": "WARNING",
                        "detail": "Value Trap verdict TRAP downgraded to MODERATE: DATA_BLOCK confirms GROWTH_INVESTING + ADEQUATE/STRONG ROIC + EXPLICIT capital plan",
                        "action": "RISK_PENALTY",
                        "risk_penalty": 0.5,
                        "rationale": "TRAP verdict rests on a capital-allocation read the DATA_BLOCK contradicts (genuine growth investment with adequate returns). Penalty downgraded (not removed) to MODERATE.",
                    }
                )
            logger.info(
                "value_trap_verdict_downgraded_reinvestment_context",
                ticker=ticker,
                verdict=verdict,
            )
        else:
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
            logger.debug("value_trap_flag_verdict", ticker=ticker, verdict=verdict)

    if not has_catalyst and activist_present == "NO" and not active_ma:
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
        logger.debug(
            "value_trap_flag_no_catalyst",
            ticker=ticker,
            activist_present=activist_present,
        )
    elif not has_catalyst and activist_present == "NO" and active_ma:
        logger.debug(
            "value_trap_no_catalyst_suppressed_by_ma",
            ticker=ticker,
            m_and_a_status=m_and_a_status,
        )

    return flags


def detect_shareholder_return_execution_flags(
    fundamentals_report: str,
    value_trap_report: str | None = None,
    ticker: str = "UNKNOWN",
) -> list[dict]:
    """Credit a strong, company-specific Value-Up plan that is PROVEN executed.

    Fires a single bounded RISK_BONUS only when BOTH gates hold:
      * the Foreign Language Analyst-sourced DATA_BLOCK marks the plan STRONG and
        its execution PROVEN (realized payouts/buybacks/cancellations met or
        exceeded targets), AND
      * the Value-Trap signal is not a hard fail (verdict != TRAP and
        score >= 40), so the credit can never partially offset a hard-fail tally.

    Announced-only / weak plans, and genuine traps, receive nothing. This is the
    symmetric counterpart to the value-trap governance penalties (Korea-discount
    right-sizing) and is intentionally capped at the moat-bonus scale. It lives in
    its own detector — not inside ``detect_capital_efficiency_flags`` — so the
    Financials-sector early-return there cannot suppress it for credit-bureau or
    holding-company names.
    """
    flags: list[dict[str, Any]] = []

    signals = extract_capital_efficiency_signals(fundamentals_report)
    plan_strength = signals.get("value_up_plan_strength")
    execution = signals.get("shareholder_return_execution")

    if plan_strength != "STRONG" or execution != "PROVEN":
        return flags

    vt_metrics = extract_value_trap_score(value_trap_report or "")
    verdict = vt_metrics.get("verdict")
    score = vt_metrics.get("score")
    if verdict == "TRAP" or (score is not None and score < 40):
        logger.debug(
            "value_up_executed_bonus_withheld_hard_fail",
            ticker=ticker,
            verdict=verdict,
            score=score,
        )
        return flags

    flags.append(
        {
            "type": "KOREA_VALUE_UP_EXECUTED",
            "severity": "POSITIVE",
            "detail": "Strong company-specific Value-Up plan with PROVEN execution (realized shareholder returns met or exceeded targets).",
            "action": "RISK_BONUS",
            "risk_penalty": -0.5,
            "rationale": "Controlling shareholder has a concrete shareholder-return plan AND has demonstrably executed it (dividends raised, buybacks completed, or treasury shares cancelled per filings). This right-sizes the governance/Korea discount. Bounded credit: does not apply to announced-only or weak plans and cannot rescue a value-trap hard fail.",
        }
    )
    logger.debug(
        "value_up_executed_bonus_detected",
        ticker=ticker,
        plan_strength=plan_strength,
        execution=execution,
        verdict=verdict,
        score=score,
    )
    return flags


def detect_moat_flags(
    fundamentals_report: str,
    ticker: str = "UNKNOWN",
    *,
    base_metrics: dict[str, Any] | None = None,
) -> list[dict]:
    """Detect economic moat indicators and create bonus flags."""
    flags: list[dict[str, Any]] = []

    metrics = extract_moat_signals(fundamentals_report)
    margin_stability = metrics.get("margin_stability")
    cash_conversion = metrics.get("cash_conversion")
    margin_cv = metrics.get("margin_cv")
    cfo_ni_avg = metrics.get("cfo_ni_avg")

    # A moat bonus only exists when margins are stable OR cash conversion is
    # strong. Suppress (rather than net) it under a peak/transient earnings base
    # — but only when a bonus would actually have fired, so we never emit a
    # spurious "suppressed" flag for a name that had no moat bonus to begin with.
    would_emit_bonus = margin_stability == "HIGH" or cash_conversion == "STRONG"
    if would_emit_bonus:
        suppression_reason = _peak_or_transient_blocker(
            fundamentals_report,
            base_metrics=base_metrics,
        )
        if suppression_reason:
            flags.append(
                {
                    "type": "MOAT_BONUS_SUPPRESSED_PEAK_TRANSIENT",
                    "severity": "INFO",
                    "detail": f"Moat bonus suppressed: {suppression_reason}",
                    "action": "RISK_BONUS",
                    "risk_penalty": 0.0,
                    "rationale": "Moat bonuses derive from margin stability and cash conversion, which a cyclical peak or one-time event inflates. The bonus is suppressed (not netted against the peak/transient warning) while the earnings base is flagged suspect.",
                }
            )
            logger.debug(
                "moat_bonus_suppressed_peak_transient",
                ticker=ticker,
                reason=suppression_reason,
            )
            return flags

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
        logger.debug(
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
        logger.debug("moat_flag_pricing_power", ticker=ticker, margin_cv=margin_cv)

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
        logger.debug("moat_flag_earnings_quality", ticker=ticker, cfo_ni_avg=cfo_ni_avg)

    return flags


def detect_return_quality_fragility_flags(
    fundamentals_report: str,
    ticker: str = "UNKNOWN",
    *,
    base_metrics: dict[str, Any] | None = None,
) -> list[dict]:
    """Flag fragile return quality: unstable profitability or unproven turnaround.

    This is the deterministic relocation of the former PM-prompt free-form rubric
    item ("Return Quality Fragility +0.5"). A pure threshold check belongs in code,
    not in the LLM's hand-summed tally where it drifted (the APR.WA case applied it
    on ``DECLINING`` / ``ROA_5Y_AVG 11.1%`` — neither branch of the stated rule).

    Fires when ``PROFITABILITY_TREND == UNSTABLE`` OR (``ROA >= 7%`` AND
    ``ROA_5Y_AVG < 5%`` — a current-strong/history-weak unproven turnaround).
    Suppressed under ``_peak_or_transient_blocker`` so it never double-counts the
    deterministic ``CYCLICAL_PEAK_WARNING`` / ``TRANSIENT_STRENGTH_DISTORTION``.
    """
    flags: list[dict[str, Any]] = []

    metrics = (
        base_metrics
        if base_metrics is not None
        else extract_metrics(fundamentals_report)
    )
    trend = metrics.get("profitability_trend")
    roa_current = metrics.get("roa_current")
    roa_5y_avg = metrics.get("roa_5y_avg")

    unproven_turnaround = (
        roa_current is not None
        and roa_current >= 7.0
        and roa_5y_avg is not None
        and roa_5y_avg < 5.0
    )
    if trend != "UNSTABLE" and not unproven_turnaround:
        return flags

    # Do not double-count: CYCLICAL_PEAK_WARNING / TRANSIENT_STRENGTH_DISTORTION
    # already charge the same suspect-earnings base.
    if _peak_or_transient_blocker(fundamentals_report, base_metrics=metrics):
        return flags

    if trend == "UNSTABLE":
        detail = "PROFITABILITY_TREND: UNSTABLE — return base is volatile."
    else:
        detail = (
            f"Unproven turnaround: ROA {roa_current:.1f}% above threshold but "
            f"5Y avg {roa_5y_avg:.1f}% below 5% — new return level not yet durable."
        )
    flags.append(
        {
            "type": "RETURN_QUALITY_FRAGILITY",
            "severity": "WARNING",
            "detail": detail,
            "action": "RISK_PENALTY",
            "risk_penalty": 0.5,
            "rationale": "Returns are either volatile (UNSTABLE) or sit well above a weak 5-year base (unproven turnaround). Either way the current return level is not yet demonstrated to be durable, warranting a modest risk premium. Computed deterministically from the DATA_BLOCK to keep the threshold off the PM's free-form tally.",
        }
    )
    logger.debug(
        "return_quality_fragility_flag",
        ticker=ticker,
        profitability_trend=trend,
        roa_current=roa_current,
        roa_5y_avg=roa_5y_avg,
    )
    return flags


def detect_capital_efficiency_flags(
    fundamentals_report: str,
    ticker: str = "UNKNOWN",
    value_trap_report: str | None = None,
    sector: Sector | None = None,
    *,
    base_metrics: dict[str, Any] | None = None,
) -> list[dict]:
    """Detect capital-efficiency risk and bonus flags."""
    flags: list[dict[str, Any]] = []

    from src.config import config

    metrics = extract_capital_efficiency_signals(fundamentals_report)
    value_trap_metrics = extract_value_trap_score(value_trap_report or "")
    if not metrics:
        return flags
    if base_metrics is None:
        base_metrics = extract_metrics(fundamentals_report)

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
        logger.debug(
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
        logger.debug(
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
        logger.debug(
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
        logger.debug("capital_flag_below_hurdle", ticker=ticker, roic=roic)

    if roic_quality == "STRONG" and leverage_quality in ("GENUINE", "CONSERVATIVE"):
        suppression_reason = _peak_or_transient_blocker(
            fundamentals_report,
            base_metrics=base_metrics,
        )
        if suppression_reason:
            flags.append(
                {
                    "type": "CAPITAL_EFFICIENCY_BONUS_SUPPRESSED",
                    "severity": "INFO",
                    "detail": f"Capital-efficiency bonus suppressed: {suppression_reason}",
                    "action": "RISK_BONUS",
                    "risk_penalty": 0.0,
                    "rationale": "Current ROIC strength may reflect a cyclical peak or one-time event. The capital-efficiency bonus is suppressed (not netted against the peak/transient warning) until the earnings base is verified.",
                }
            )
            logger.debug(
                "capital_efficiency_bonus_suppressed_peak_transient",
                ticker=ticker,
                reason=suppression_reason,
            )
        else:
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
            logger.debug(
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
        logger.debug(
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
        logger.debug(
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
        logger.debug("consultant_flag_hard_stop", ticker=ticker)
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
        logger.debug("consultant_flag_mandate_breach", ticker=ticker)

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
        logger.debug("consultant_flag_major_concerns", ticker=ticker)
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
        logger.debug("consultant_flag_conditional", ticker=ticker)

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
        logger.debug("consultant_flag_growth_quality_unproven", ticker=ticker)

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
        logger.debug("consultant_flag_transient_strength", ticker=ticker)

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
        logger.debug(
            "consultant_flag_discrepancies",
            ticker=ticker,
            count=len(discrepancies),
            penalty=disc_penalty,
        )

    return flags
