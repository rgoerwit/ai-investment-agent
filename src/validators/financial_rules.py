"""Core financial red-flag rules and PASS/REJECT aggregation."""

from __future__ import annotations

import re
from typing import Any

import structlog

from src.validators.sector_classifier import (
    CAPITAL_INTENSIVE_SECTORS,
    FINANCIALS_SECTORS,
    Sector,
)

logger = structlog.get_logger(__name__)

_TRANSIENT_STRENGTH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "acquisition-led consolidation",
        re.compile(
            r"\b(?:acquisition[- ]driven|acquisition-led|m&a(?:[- ]driven)?|merger[- ]driven|inorganic growth|organic vs acquired|m&a illusion)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "asset or division sale",
        re.compile(
            r"\b(?:asset sale|division sale|sale of (?:a )?division|gain on sale)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "legal settlement",
        re.compile(r"\b(?:legal settlement|settlement gain)\b", re.IGNORECASE),
    ),
    (
        "restructuring gain",
        re.compile(r"\b(?:restructuring gain|one-time gain)\b", re.IGNORECASE),
    ),
    (
        "regulatory windfall or subsidy",
        re.compile(
            r"\b(?:regulatory windfall|government subsidy|subsidy windfall)\b",
            re.IGNORECASE,
        ),
    ),
)


def detect_red_flags(
    metrics: dict[str, float | None],
    ticker: str = "UNKNOWN",
    sector: Sector = Sector.INDUSTRIALS,
    strict_mode: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """Apply sector-aware threshold-based red-flag detection logic."""
    red_flags: list[dict[str, Any]] = []

    if sector in FINANCIALS_SECTORS:
        leverage_threshold = None
        coverage_threshold = None
        coverage_de_threshold = None
    elif sector in CAPITAL_INTENSIVE_SECTORS:
        if strict_mode:
            leverage_threshold = 500
            coverage_threshold = 1.8
            coverage_de_threshold = 300
        else:
            leverage_threshold = 800
            coverage_threshold = 1.5
            coverage_de_threshold = 200
    else:
        if strict_mode:
            leverage_threshold = 300
            coverage_threshold = 2.5
            coverage_de_threshold = 150
        else:
            leverage_threshold = 500
            coverage_threshold = 2.0
            coverage_de_threshold = 100

    debt_to_equity = metrics.get("debt_to_equity")
    if (
        leverage_threshold is not None
        and debt_to_equity is not None
        and debt_to_equity > leverage_threshold
    ):
        red_flags.append(
            {
                "type": "EXTREME_LEVERAGE",
                "severity": "CRITICAL",
                "detail": f"D/E ratio {debt_to_equity:.1f}% is extreme (>{leverage_threshold}% threshold for {sector.value})",
                "action": "AUTO_REJECT",
                "rationale": f"Leverage exceeds sector-appropriate threshold - bankruptcy risk (sector: {sector.value})",
            }
        )
        logger.info(
            "red_flag_extreme_leverage",
            ticker=ticker,
            debt_to_equity=debt_to_equity,
            threshold=leverage_threshold,
            sector=sector.value,
        )

    net_income = metrics.get("net_income")
    fcf = metrics.get("fcf")
    fcf_data_uncertain = "FCF DATA QUALITY UNCERTAIN" in (
        metrics.get("_raw_report", "") or ""
    )

    if (
        net_income is not None
        and net_income > 0
        and fcf is not None
        and fcf < 0
        and abs(fcf) > (2 * net_income)
    ):
        disconnect_ratio = abs(fcf / net_income) if net_income != 0 else 0
        if fcf_data_uncertain or disconnect_ratio > 4.0:
            red_flags.append(
                {
                    "type": "EARNINGS_QUALITY_UNCERTAIN",
                    "severity": "WARNING",
                    "detail": f"NI ${net_income:,.0f} but FCF ${fcf:,.0f} ({disconnect_ratio:.1f}x) - data quality uncertain",
                    "action": "RISK_PENALTY",
                    "risk_penalty": 1.0,
                    "rationale": "FCF/NI disconnect may reflect TTM data misalignment, not fraud",
                }
            )
        else:
            red_flags.append(
                {
                    "type": "EARNINGS_QUALITY",
                    "severity": "CRITICAL",
                    "detail": f"Positive net income (${net_income:,.0f}) but negative FCF (${fcf:,.0f}) >2x income",
                    "action": "AUTO_REJECT",
                    "rationale": "Earnings likely fabricated through accounting tricks - FCF disconnect",
                }
            )
            logger.info(
                "red_flag_earnings_quality",
                ticker=ticker,
                net_income=net_income,
                fcf=fcf,
                disconnect_multiple=disconnect_ratio,
            )

    interest_coverage = metrics.get("interest_coverage")
    if (
        coverage_threshold is not None
        and coverage_de_threshold is not None
        and interest_coverage is not None
        and interest_coverage < coverage_threshold
        and debt_to_equity is not None
        and debt_to_equity > coverage_de_threshold
    ):
        red_flags.append(
            {
                "type": "REFINANCING_RISK",
                "severity": "CRITICAL",
                "detail": f"Interest coverage {interest_coverage:.2f}x with {debt_to_equity:.1f}% D/E ratio (thresholds: <{coverage_threshold}x coverage + >{coverage_de_threshold}% D/E for {sector.value})",
                "action": "AUTO_REJECT",
                "rationale": f"Cannot comfortably service debt - refinancing/default risk (sector: {sector.value})",
            }
        )
        logger.info(
            "red_flag_refinancing_risk",
            ticker=ticker,
            interest_coverage=interest_coverage,
            debt_to_equity=debt_to_equity,
            coverage_threshold=coverage_threshold,
            de_threshold=coverage_de_threshold,
            sector=sector.value,
        )

    payout_ratio = metrics.get("payout_ratio")
    dividend_coverage = metrics.get("dividend_coverage")
    roic_quality = metrics.get("roic_quality")
    profitability_trend = metrics.get("profitability_trend")
    if (
        payout_ratio is not None
        and payout_ratio > 100
        and dividend_coverage == "UNCOVERED"
    ):
        is_value_destroying = roic_quality in ("WEAK", "DESTRUCTIVE")
        is_recovering = profitability_trend == "IMPROVING"
        if is_value_destroying and not is_recovering:
            red_flags.append(
                {
                    "type": "UNSUSTAINABLE_DISTRIBUTION",
                    "severity": "CRITICAL",
                    "detail": f"Payout {payout_ratio:.0f}% + uncovered dividend + ROIC {roic_quality} + trend {profitability_trend}",
                    "action": "AUTO_REJECT",
                    "rationale": "Dividend exceeds earnings, FCF doesn't cover it, ROIC below hurdle, and no improving trend. Mathematically unsustainable value destruction.",
                }
            )
            logger.info(
                "red_flag_unsustainable_distribution_critical",
                ticker=ticker,
                payout_ratio=payout_ratio,
                dividend_coverage=dividend_coverage,
                roic_quality=roic_quality,
                profitability_trend=profitability_trend,
            )
        else:
            red_flags.append(
                {
                    "type": "UNSUSTAINABLE_DISTRIBUTION",
                    "severity": "WARNING",
                    "detail": f"Payout {payout_ratio:.0f}% with {dividend_coverage} dividend coverage",
                    "action": "RISK_PENALTY",
                    "risk_penalty": 1.5,
                    "rationale": "Dividend funded by debt/reserves. Watch for dividend cut or verify cyclical recovery thesis if ROIC improving.",
                }
            )
            logger.info(
                "red_flag_unsustainable_distribution_warning",
                ticker=ticker,
                payout_ratio=payout_ratio,
                dividend_coverage=dividend_coverage,
                roic_quality=roic_quality,
            )

    net_margin = metrics.get("net_margin")
    pb_ratio = metrics.get("pb_ratio")
    debt_to_equity = metrics.get("debt_to_equity")
    if (
        net_margin is not None
        and net_margin < 5.0
        and pb_ratio is not None
        and pb_ratio > 4.0
        and debt_to_equity is not None
        and debt_to_equity > 80
    ):
        red_flags.append(
            {
                "type": "FRAGILE_VALUATION",
                "severity": "CRITICAL",
                "detail": f"P/B {pb_ratio:.1f}x with {net_margin:.1f}% margins and {debt_to_equity:.0f}% leverage",
                "action": "CRITICAL_WARNING",
                "rationale": "Valuation mismatch: Paying high-growth multiples for a low-margin, capital-intensive business. No margin of safety against execution risk.",
            }
        )
        logger.info(
            "red_flag_fragile_valuation",
            ticker=ticker,
            net_margin=net_margin,
            pb_ratio=pb_ratio,
            debt_to_equity=debt_to_equity,
        )

    roa_current = metrics.get("roa_current")
    roa_5y_avg = metrics.get("roa_5y_avg")
    peg_ratio = metrics.get("peg_ratio")
    peak_signals: list[str] = []
    if (
        roa_current is not None
        and roa_5y_avg is not None
        and roa_5y_avg > 0
        and roa_current / roa_5y_avg > 1.5
    ):
        peak_signals.append(
            f"ROA {roa_current:.1f}% vs 5Y avg {roa_5y_avg:.1f}% ({roa_current/roa_5y_avg:.1f}x)"
        )
    if peg_ratio is not None and peg_ratio < 0.2 and profitability_trend == "UNSTABLE":
        peak_signals.append(
            f"PEG {peg_ratio:.2f} with UNSTABLE profitability (cyclical earnings peak)"
        )
    if peak_signals and profitability_trend in ("UNSTABLE", "DECLINING"):
        red_flags.append(
            {
                "type": "CYCLICAL_PEAK_WARNING",
                "severity": "WARNING",
                "detail": "; ".join(peak_signals),
                "action": "RISK_PENALTY",
                "risk_penalty": 1.0,
                "rationale": "Current metrics significantly exceed historical averages with unstable profitability. P/E and PEG are calculated on peak earnings and may revert. Normalize valuations using 5-year averages before deciding.",
            }
        )
        logger.info(
            "red_flag_cyclical_peak_warning",
            ticker=ticker,
            signals=peak_signals,
            profitability_trend=profitability_trend,
        )

    revenue_growth_ttm = metrics.get("revenue_growth_ttm")
    growth_quality_signals: list[str] = []
    if revenue_growth_ttm is not None and revenue_growth_ttm >= 25.0:
        if profitability_trend == "DECLINING":
            growth_quality_signals.append(
                "profitability trend declining despite strong revenue growth"
            )
        if (
            roa_current is not None
            and roa_5y_avg is not None
            and roa_5y_avg > 0
            and roa_current < 0.85 * roa_5y_avg
        ):
            growth_quality_signals.append(
                f"ROA {roa_current:.1f}% vs 5Y avg {roa_5y_avg:.1f}%"
            )
    if growth_quality_signals:
        roic_note = (
            f"; ROIC quality {roic_quality.lower()}"
            if isinstance(roic_quality, str) and roic_quality in {"WEAK", "ADEQUATE"}
            else ""
        )
        red_flags.append(
            {
                "type": "GROWTH_QUALITY_UNPROVEN",
                "severity": "WARNING",
                "detail": f"Revenue growth {revenue_growth_ttm:.1f}% with {'; '.join(growth_quality_signals)}{roic_note}",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.75,
                "rationale": "Strong reported growth is not yet supported by improving capital efficiency. Treat the new baseline as unproven until returns stabilize or improve.",
            }
        )
        logger.info(
            "red_flag_growth_quality_unproven",
            ticker=ticker,
            revenue_growth_ttm=revenue_growth_ttm,
            profitability_trend=profitability_trend,
            roa_current=roa_current,
            roa_5y_avg=roa_5y_avg,
            roic_quality=roic_quality,
        )

    raw_report = metrics.get("_raw_report", "") or ""
    transient_strength_labels = [
        label
        for label, pattern in _TRANSIENT_STRENGTH_PATTERNS
        if isinstance(raw_report, str) and pattern.search(raw_report)
    ]
    ocf_current = metrics.get("ocf")
    has_current_strength = (
        revenue_growth_ttm is not None and revenue_growth_ttm >= 15.0
    ) or (
        net_income is not None
        and net_income > 0
        and ocf_current is not None
        and ocf_current > 0
        and metrics.get("adjusted_health_score") is not None
        and metrics.get("adjusted_health_score", 0) >= 60.0
    )
    if transient_strength_labels and has_current_strength:
        detail_parts: list[str] = []
        if revenue_growth_ttm is not None and revenue_growth_ttm >= 15.0:
            detail_parts.append(f"revenue growth {revenue_growth_ttm:.1f}%")
        if (
            net_income is not None
            and net_income > 0
            and ocf_current is not None
            and ocf_current > 0
        ):
            detail_parts.append("positive net income and OCF")
        red_flags.append(
            {
                "type": "TRANSIENT_STRENGTH_DISTORTION",
                "severity": "WARNING",
                "detail": f"Named transient driver detected ({', '.join(transient_strength_labels[:2])}) alongside {'; '.join(detail_parts)}",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.75,
                "rationale": "Current-period strength may reflect a non-recurring driver rather than durable operating improvement. Do not treat this as proven baseline earning power.",
            }
        )
        logger.info(
            "red_flag_transient_strength_distortion",
            ticker=ticker,
            drivers=transient_strength_labels,
            revenue_growth_ttm=revenue_growth_ttm,
        )

    ocf = metrics.get("ocf")
    ni_for_ocf = metrics.get("net_income")
    if (
        sector not in FINANCIALS_SECTORS
        and ocf is not None
        and ni_for_ocf is not None
        and ocf > 0
        and ni_for_ocf > 0
    ):
        ocf_ni_ratio = ocf / ni_for_ocf
        if ocf_ni_ratio > 3.0:
            penalty, label = (
                (1.5, "likely data error or period mismatch")
                if ocf_ni_ratio > 5.0
                else (1.0, "unusual, verify data source")
            )
            red_flags.append(
                {
                    "type": "SUSPICIOUS_OCF_NI_RATIO",
                    "severity": "WARNING",
                    "detail": f"OCF {ocf_ni_ratio:.1f}x net income — {label}",
                    "action": "RISK_PENALTY",
                    "risk_penalty": penalty,
                    "rationale": f"Operating cash flow exceeding net income by >{ocf_ni_ratio:.0f}x is unusual and may indicate a data source error, wrong currency, or period mismatch. Cross-validate with an independent source.",
                }
            )
            logger.info(
                "red_flag_suspicious_ocf_ni_ratio",
                ticker=ticker,
                ocf=ocf,
                net_income=ni_for_ocf,
                ratio=ocf_ni_ratio,
            )

    peg_for_floor = metrics.get("peg_ratio")
    if peg_for_floor is not None and 0 <= peg_for_floor < 0.05:
        rev_growth = metrics.get("revenue_growth_ttm")
        peg_explained_by_growth = (
            peg_for_floor > 0 and rev_growth is not None and rev_growth >= 50.0
        )
        if peg_explained_by_growth:
            logger.info(
                "unreliable_peg_skipped_high_growth",
                ticker=ticker,
                peg=peg_for_floor,
                revenue_growth_ttm=rev_growth,
            )
        else:
            detail = (
                "PEG 0.00 — mathematically undefined (growth denominator is zero, negative, or infinite). Valuation metrics are unreliable."
                if peg_for_floor == 0
                else f"PEG {peg_for_floor:.3f} — growth rate input is missing or stale. Treat PEG-derived conclusions as unreliable."
            )
            red_flags.append(
                {
                    "type": "UNRELIABLE_PEG",
                    "severity": "WARNING",
                    "detail": detail,
                    "action": "RISK_PENALTY",
                    "risk_penalty": 1.0,
                    "rationale": "A PEG ratio below 0.05 without confirmed high revenue growth means the growth rate input is likely missing or stale. All PEG-derived conclusions should be discounted.",
                }
            )
            logger.info("red_flag_unreliable_peg", ticker=ticker, peg=peg_for_floor)

    segment_flag = metrics.get("segment_flag")
    if segment_flag == "DETERIORATING":
        red_flags.append(
            {
                "type": "SEGMENT_DETERIORATION",
                "severity": "WARNING",
                "detail": "Dominant segment showing profit decline (flagged by Senior Fundamentals)",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": "A major business segment contributing >20% of revenue has operating profit declining >20% YoY. Consolidated metrics may mask deterioration in a key business unit.",
            }
        )
        logger.info("red_flag_segment_deterioration", ticker=ticker)

    ocf_source = metrics.get("ocf_source")
    ocf_reason = (metrics.get("ocf_filing_reason") or "DISCREPANCY").upper()
    if ocf_source == "FILING" and ocf_reason == "API_UNAVAILABLE":
        red_flags.append(
            {
                "type": "OCF_SINGLE_SOURCE",
                "severity": "INFO",
                "detail": "OCF value sourced from filing only — API unavailable, no discrepancy detected",
                "action": "NOTE",
                "risk_penalty": 0.0,
                "rationale": "The filing provided the only usable OCF value because the aggregator/API source was unavailable. This is a process limitation, not evidence of a company data inconsistency.",
            }
        )
        logger.info("red_flag_ocf_single_source", ticker=ticker)
    elif ocf_source == "FILING":
        red_flags.append(
            {
                "type": "OCF_SOURCE_DISCREPANCY",
                "severity": "WARNING",
                "detail": "OCF value sourced from filing differs from API data — verify",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": "The Senior Fundamentals Analyst preferred the filing-sourced OCF over the API-sourced value due to a >30% discrepancy. This may indicate a yfinance data error, currency mismatch, or period mismatch. The filing value is likely more accurate but warrants cross-validation.",
            }
        )
        logger.info(
            "red_flag_ocf_source_discrepancy",
            ticker=ticker,
            ocf_filing_reason=ocf_reason,
        )

    if revenue_growth_ttm is not None and revenue_growth_ttm < -15.0:
        red_flags.append(
            {
                "type": "GROWTH_CLIFF",
                "severity": "WARNING",
                "detail": f"TTM revenue growth {revenue_growth_ttm:.1f}% — sharp deterioration not reflected in annual data",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": "Trailing twelve-month revenue shows sharp decline. This may indicate loss of key contracts, competitive disruption, or demand collapse. Annual data may still look acceptable, masking the deterioration.",
            }
        )
        logger.info(
            "red_flag_growth_cliff",
            ticker=ticker,
            revenue_growth_ttm=revenue_growth_ttm,
        )

    total_est = metrics.get("analyst_coverage_total_est")
    if isinstance(total_est, int) and total_est < 3:
        red_flags.append(
            {
                "type": "THIN_CONSENSUS",
                "severity": "WARNING",
                "detail": f"Total estimated analyst coverage is {total_est} — consensus targets, PEG, and forward P/E based on <3 analysts are statistically unreliable",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": "Price targets, PEG ratio, and forward P/E are all derived from consensus analyst estimates. With fewer than 3 analysts, these figures reflect individual opinions, not statistical consensus. Prefer trailing P/E, P/B, and intrinsic valuation (DCF, asset-based) over consensus-derived metrics for this stock.",
            }
        )
        logger.info("red_flag_thin_consensus", ticker=ticker, total_est=total_est)
    if total_est == "HIGH" or (isinstance(total_est, int) and total_est > 20):
        red_flags.append(
            {
                "type": "LOCAL_COVERAGE_HIGH",
                "severity": "WARNING",
                "detail": "Home-market analyst coverage is high — information edge is weaker than a typical undiscovered thesis candidate",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.25,
                "rationale": "English-language coverage may still be low, but high local coverage means the home market has likely already absorbed segment-level, governance, and catalyst information. The undiscovered edge is therefore weaker.",
            }
        )
        logger.info("red_flag_local_coverage_high", ticker=ticker, total_est=total_est)

    if strict_mode:
        sector_str = (metrics.get("sector") or "").lower()
        industry_str = (metrics.get("industry") or "").lower()
        is_reit = (
            "reit" in industry_str
            or "real estate investment trust" in industry_str
            or (
                sector == Sector.REAL_ESTATE
                and "developer" not in industry_str
                and "builder" not in industry_str
                and industry_str
            )
        )
        if is_reit:
            red_flags.append(
                {
                    "type": "STRICT_REIT_ETF",
                    "severity": "CRITICAL",
                    "detail": f"REIT/ETF excluded in strict mode (sector: {sector_str or sector.value}, industry: {industry_str or 'N/A'})",
                    "action": "AUTO_REJECT",
                    "rationale": "REITs are pass-through vehicles; not compatible with GARP growth-transition strategy",
                }
            )
            logger.info(
                "strict_reit_etf_rejected",
                ticker=ticker,
                industry=industry_str,
                sector=sector_str,
            )

    if strict_mode:
        ocf = metrics.get("ocf")
        ni = metrics.get("net_income")
        if ocf is not None and ni is not None and ni > 0:
            ratio = ocf / ni
            if ratio < 0.8:
                red_flags.append(
                    {
                        "type": "STRICT_EARNINGS_QUALITY",
                        "severity": "CRITICAL",
                        "detail": f"OCF/NI ratio {ratio:.2f} < 0.8 (accrual-heavy accounting; OCF={ocf:,.0f}, NI={ni:,.0f})",
                        "action": "AUTO_REJECT",
                        "rationale": "Operating cash flow well below net income — earnings likely overstated via accruals",
                    }
                )
                logger.info(
                    "strict_earnings_quality_rejected",
                    ticker=ticker,
                    ocf_ni_ratio=ratio,
                    ocf=ocf,
                    net_income=ni,
                )

    has_auto_reject = any(flag["action"] == "AUTO_REJECT" for flag in red_flags)
    return red_flags, "REJECT" if has_auto_reject else "PASS"
