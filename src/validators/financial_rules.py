"""Core financial red-flag rules and PASS/REJECT aggregation."""

from __future__ import annotations

import re
from typing import Any

import structlog

from src.data_block_utils import extract_data_block_field
from src.thesis_constants import PE_MAX, PE_VS_SECTOR_RICH
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
    (
        "deconsolidation, disposal, or spin-off",
        re.compile(
            r"\b(?:deconsolidat\w*|desincorporat\w*|spin[- ]off|divestit\w*|"
            r"disposal of|bargain purchase|acquisition accounting)\b",
            re.IGNORECASE,
        ),
    ),
)

# Language indicating the analysis already reconciled normalized (ex-one-time)
# economics — the "bridge" that must exist before a one-time item may be framed
# as a catalyst rather than a distortion.
_NORMALIZED_EARNINGS_BRIDGE_RE = re.compile(
    r"\b(?:normaliz\w+\s+(?:eps|earnings|revenue|margin|ocf|net income|profit)|"
    r"ex[- ]one[- ]time|excluding the (?:gain|disposal|sale|settlement|deconsolidation)|"
    r"adjusted for the (?:gain|disposal|sale|deconsolidation|divestiture)|"
    r"underlying (?:earnings|eps|profit)|organic (?:revenue|growth)|"
    r"like[- ]for[- ]like|recurring (?:earnings|revenue|profit))\b",
    re.IGNORECASE,
)


def requires_normalized_earnings_bridge(report: str | None) -> bool:
    """True if a one-time event is present but no normalized-earnings bridge is.

    Distortion-before-catalyst discipline: a disposal, deconsolidation,
    acquisition-accounting benefit, settlement, or subsidy must be reconciled
    to normalized (ex-one-time) revenue/margin/EPS/OCF before it may be treated
    as a catalyst. Reuses the canonical one-time-event pattern set.
    """
    if not report:
        return False
    return contains_transient_strength_marker(report) and not bool(
        _NORMALIZED_EARNINGS_BRIDGE_RE.search(report)
    )


def contains_transient_strength_marker(report: str | None) -> bool:
    """Return True if the report names a non-recurring strength driver.

    Shared, public entry point over ``_TRANSIENT_STRENGTH_PATTERNS`` (asset
    sale, M&A-led consolidation, settlement, restructuring/one-time gain,
    subsidy windfall) so other validators can reuse the canonical pattern set
    without re-authoring it.
    """
    if not report:
        return False
    return any(
        pattern.search(report) for _label, pattern in _TRANSIENT_STRENGTH_PATTERNS
    )


_OCF_MAGNITUDE: dict[str, float] = {
    "T": 1e12,
    "TRILLION": 1e12,
    "B": 1e9,
    "BN": 1e9,
    "BILLION": 1e9,
    "M": 1e6,
    "MN": 1e6,
    "MM": 1e6,
    "MILLION": 1e6,
    "K": 1e3,
}
_OCF_NUM_MAG_RE = re.compile(
    r"(\d[\d,]*\.?\d*)\s*(trillion|billion|million|bn|mm|mn|[tbmk])\b",
    re.IGNORECASE,
)
_OCF_LINE_RE = re.compile(
    r"(?im)^[^\n]*\b(?:operating cash flow|cash flow from operations|"
    r"net cash from operating activities|cfo)\b[^\n]*$"
)
_OCF_CONSULTANT_CONTEXT_RE = re.compile(
    r"\b(?:OCF|operating\s+cash\s+flow|operatingCashflow|cash[- ]flow)\b",
    re.IGNORECASE,
)
_OCF_PERIOD_MISMATCH_RE = re.compile(
    r"\b(?:period\s+mismatch|period\s+normali[sz]ation|"
    r"TTM(?:/Q[1-4])?\s*(?:vs\.?|versus|/)\s*FY|"
    r"FY\s*\d{4}\s*(?:vs\.?|versus|/)\s*TTM|not\s+comparable)\b",
    re.IGNORECASE,
)
_OCF_RESOLVED_RE = re.compile(
    r"\b(?:not\s+(?:a\s+)?data\s+conflict|not\s+(?:a\s+)?data\s+error|"
    r"not\s+comparable|resolved|reconciled|corroborated|matches)\b",
    re.IGNORECASE,
)
_OCF_UNRESOLVED_RE = re.compile(
    r"\b(?:unresolved|wrong\s+(?:statement\s+)?line|currency\s+mismatch|"
    r"data\s+conflict\s+remains|not\s+reconciled)\b",
    re.IGNORECASE,
)


def parse_ocf_amount(text: str | None) -> float | None:
    """Parse one money amount from free text (handles ``~``, currency, magnitude).

    Accepts forms like ``1.148B PLN``, ``~PLN 971m``, ``920,000,000``. Returns the
    value in base units, or ``None`` if no amount is found.
    """
    if not text:
        return None
    match = _OCF_NUM_MAG_RE.search(text)
    if match:
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            return None
        return value * _OCF_MAGNITUDE.get(match.group(2).upper(), 1.0)
    bare = re.search(r"(\d[\d,]{6,})", text)  # bare large number, e.g. 920000000
    if bare:
        try:
            return float(bare.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def extract_auditor_ocf(report: str | None) -> float | None:
    """Return the forensic auditor's independently-computed OCF, if stated.

    The auditor reports a line such as ``Operating cash flow: ~PLN 971m``. We
    anchor on a cash-flow-labelled line so we do not pick up an unrelated figure.
    """
    if not report:
        return None
    for match in _OCF_LINE_RE.finditer(report):
        amount = parse_ocf_amount(match.group(0))
        if amount:
            return amount
    return None


def detect_ocf_corroboration_flag(
    datablock_ocf: float | None,
    auditor_ocf: float | None,
    ticker: str = "UNKNOWN",
    threshold: float = 0.15,
) -> dict[str, Any] | None:
    """Flag a headline OCF that the forensic auditor's independent figure contradicts.

    The forensic auditor computes OCF from primary documents independently of the
    Foreign-Language "filing" value the Senior Fundamentals Analyst may have
    promoted under the FILING AUTHORITY PRINCIPLE. A material divergence means the
    headline cash-generation narrative is *uncorroborated* — it must not be
    asserted as fact. Risk-neutral by design (the existing OCF_SOURCE_DISCREPANCY
    already carries the penalty); this flag exists to block the overclaim and
    surface the conflict, not to escalate the tally.
    """
    if not datablock_ocf or not auditor_ocf or datablock_ocf <= 0 or auditor_ocf <= 0:
        return None
    divergence = abs(datablock_ocf - auditor_ocf) / min(datablock_ocf, auditor_ocf)
    if divergence <= threshold:
        return None
    headline_high = datablock_ocf > auditor_ocf
    detail = (
        f"Headline OCF {datablock_ocf:,.0f} diverges {divergence * 100:.0f}% from the "
        f"forensic auditor's independent OCF {auditor_ocf:,.0f}"
    )
    if headline_high:
        detail += (
            "; do not treat the higher headline as verified 'elite cash generation' "
            "until reconciled to the actual cash-flow statement line."
        )
    else:
        detail += "."
    logger.debug(
        "red_flag_ocf_filing_value_uncorroborated",
        ticker=ticker,
        datablock_ocf=datablock_ocf,
        auditor_ocf=auditor_ocf,
        divergence_pct=round(divergence * 100, 1),
    )
    return {
        "type": "OCF_FILING_VALUE_UNCORROBORATED",
        "severity": "WARNING",
        "detail": detail,
        "action": "DOWNWEIGHT_CASH_NARRATIVE",
        "risk_penalty": 0.0,
        "rationale": (
            "The forensic auditor independently computed operating cash flow from "
            "primary documents. A material divergence from the headline DATA_BLOCK "
            "OCF means cash-conversion, dividend-coverage, and 'elite cash "
            "generation' claims are not corroborated and must not be asserted as "
            "fact until the figure is traced to the cash-flow statement line."
        ),
    }


def _consultant_resolves_ocf_period_mismatch(
    consultant_review: str | None,
    consultant_conditions: dict[str, Any] | None = None,
) -> bool:
    """Conservatively detect consultant text resolving OCF as period mismatch.

    Numeric auditor corroboration is the load-bearing gate elsewhere. This text gate
    only confirms the subtype: period mismatch, not currency/wrong-line conflict.
    """
    if not consultant_review:
        return False
    conditions = consultant_conditions or {}
    if conditions.get("has_hard_stop") or conditions.get("has_mandate_breach"):
        return False
    if conditions.get("verdict") == "MAJOR_CONCERNS":
        return False

    for match in _OCF_CONSULTANT_CONTEXT_RE.finditer(consultant_review):
        start = max(0, match.start() - 280)
        end = min(len(consultant_review), match.end() + 280)
        window = consultant_review[start:end]
        if _OCF_UNRESOLVED_RE.search(window):
            return False
        if _OCF_PERIOD_MISMATCH_RE.search(window) and _OCF_RESOLVED_RE.search(window):
            return True
    return False


def is_ocf_period_mismatch_resolved(
    fundamentals_report: str | None,
    consultant_review: str | None,
    auditor_report: str | None,
    ticker: str = "UNKNOWN",
    *,
    consultant_conditions: dict[str, Any] | None = None,
    threshold: float = 0.15,
) -> bool:
    """True when a filing/API OCF discrepancy is resolved as period mismatch.

    This is intentionally post-research: the independent auditor report is only
    available after the Senior DATA_BLOCK has already emitted OCF_SOURCE_DISCREPANCY.
    """
    ocf_source = (
        (
            extract_data_block_field(fundamentals_report, "OPERATING_CASH_FLOW_SOURCE")
            or ""
        )
        .strip()
        .upper()
    )
    ocf_reason = (
        (extract_data_block_field(fundamentals_report, "OCF_FILING_REASON") or "")
        .strip()
        .upper()
    )
    if ocf_source != "FILING" or ocf_reason != "DISCREPANCY":
        return False

    datablock_ocf = parse_ocf_amount(
        extract_data_block_field(fundamentals_report, "OPERATING_CASH_FLOW")
    )
    auditor_ocf = extract_auditor_ocf(auditor_report)
    if not datablock_ocf or not auditor_ocf:
        return False

    if detect_ocf_corroboration_flag(
        datablock_ocf, auditor_ocf, ticker=ticker, threshold=threshold
    ):
        return False

    return _consultant_resolves_ocf_period_mismatch(
        consultant_review, consultant_conditions=consultant_conditions
    )


def reconcile_ocf_period_mismatch_flags(
    red_flags: list[dict[str, Any]] | None,
    fundamentals_report: str | None,
    consultant_review: str | None,
    auditor_report: str | None,
    ticker: str = "UNKNOWN",
    *,
    consultant_conditions: dict[str, Any] | None = None,
    threshold: float = 0.15,
) -> list[dict[str, Any]]:
    """Replace resolved OCF_SOURCE_DISCREPANCY with a risk-neutral note."""
    flags = list(red_flags or [])
    if not any(flag.get("type") == "OCF_SOURCE_DISCREPANCY" for flag in flags):
        return flags
    if not is_ocf_period_mismatch_resolved(
        fundamentals_report,
        consultant_review,
        auditor_report,
        ticker=ticker,
        consultant_conditions=consultant_conditions,
        threshold=threshold,
    ):
        return flags

    resolved_flag = {
        "type": "OCF_PERIOD_MISMATCH_RESOLVED",
        "severity": "INFO",
        "detail": (
            "Filing/API OCF difference appears to be a FY-vs-TTM period mismatch; "
            "filing OCF is corroborated by the forensic auditor."
        ),
        "action": "NOTE",
        "risk_penalty": 0.0,
        "rationale": (
            "The original OCF discrepancy warning is neutralized only because the "
            "consultant identified a period mismatch and the forensic auditor's "
            "independent OCF is within the existing corroboration band."
        ),
    }
    reconciled = [
        resolved_flag if flag.get("type") == "OCF_SOURCE_DISCREPANCY" else flag
        for flag in flags
    ]
    logger.info("ocf_period_mismatch_resolved", ticker=ticker)
    return reconciled


def detect_red_flags(
    metrics: dict[str, Any],
    ticker: str = "UNKNOWN",
    sector: Sector = Sector.INDUSTRIALS,
    strict_mode: bool = False,
    entity_role: str | None = None,
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

    pe_ratio = metrics.get("pe_ratio")
    pe_vs_sector = metrics.get("pe_vs_sector")
    sector_median_pe = metrics.get("sector_median_pe")
    if (
        isinstance(pe_ratio, int | float)
        and isinstance(pe_vs_sector, int | float)
        and pe_ratio <= PE_MAX
        and pe_vs_sector >= PE_VS_SECTOR_RICH
    ):
        median_detail = (
            f" sector median P/E {sector_median_pe:.1f},"
            if isinstance(sector_median_pe, int | float)
            else ""
        )
        red_flags.append(
            {
                "type": "SECTOR_RELATIVE_VALUATION_RICH",
                "severity": "WARNING",
                "detail": (
                    f"P/E {pe_ratio:.1f} passes the absolute {PE_MAX:.0f}x gate but is "
                    f"{pe_vs_sector:.2f}x{median_detail} making valuation dear vs peers"
                ),
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": (
                    "Sector-relative valuation catches names that look cheap on the "
                    "flat thesis threshold but are expensive against GICS peers."
                ),
            }
        )

    for kind in ("health", "growth"):
        if metrics.get(f"{kind}_score_consistency") != "SUSPECT":
            continue
        red_flags.append(
            {
                "type": f"{kind.upper()}_SCORE_UNRELIABLE",
                "severity": "WARNING",
                "detail": (
                    f"{kind.capitalize()} score arithmetic/denominator is inconsistent "
                    f"with the scoring rubric; the reported ADJUSTED_{kind.upper()}"
                    "_SCORE cannot be trusted."
                ),
                "action": "REVIEW",
                # Data-quality flag, not stock risk (mirrors VALUATION_INPUT
                # quarantine): the model's arithmetic error must not penalize
                # the name, only block mechanical use of the score.
                "risk_penalty": 0.0,
                # Enforced deterministically post-PM by
                # verdict_policy.maybe_demote_buy_on_unreliable_score.
                "blocks_buy": True,
                "rationale": (
                    f"The score is indeterminate in BOTH directions: do NOT apply "
                    f"the hard quality gate (Adjusted {kind.capitalize()} < 50% -> "
                    "SELL) mechanically on this value, and do NOT count it as a "
                    "pass supporting BUY — cap at HOLD until the rubric reconciles."
                ),
            }
        )
        logger.warning("score_consistency_flag", ticker=ticker, kind=kind)

    debt_to_equity = metrics.get("debt_to_equity")
    role = str(entity_role or metrics.get("listing_role") or "").upper()
    holdco_leverage_explained = role in {"PURE_HOLDCO", "INTERMEDIATE_HOLDCO"} and (
        (
            isinstance(metrics.get("net_debt_ebitda"), int | float)
            and metrics["net_debt_ebitda"] <= 4.0
        )
        or (
            isinstance(metrics.get("net_cash_to_market_cap"), int | float)
            and metrics["net_cash_to_market_cap"] >= 0.25
        )
    )
    if (
        leverage_threshold is not None
        and debt_to_equity is not None
        and debt_to_equity > leverage_threshold
    ):
        if holdco_leverage_explained:
            logger.debug(
                "red_flag_extreme_leverage_suppressed_holdco",
                ticker=ticker,
                debt_to_equity=debt_to_equity,
                net_debt_ebitda=metrics.get("net_debt_ebitda"),
                net_cash_to_market_cap=metrics.get("net_cash_to_market_cap"),
                entity_role=role,
            )
        else:
            red_flags.append(
                {
                    "type": "EXTREME_LEVERAGE",
                    "severity": "CRITICAL",
                    "detail": f"D/E ratio {debt_to_equity:.1f}% is extreme (>{leverage_threshold}% threshold for {sector.value})",
                    "action": "AUTO_REJECT",
                    "rationale": f"Leverage exceeds sector-appropriate threshold - bankruptcy risk (sector: {sector.value})",
                }
            )
            logger.debug(
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
            logger.debug(
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
        logger.debug(
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
            logger.debug(
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
            logger.debug(
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
        logger.debug(
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
        logger.debug(
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
        logger.debug(
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
        logger.debug(
            "red_flag_transient_strength_distortion",
            ticker=ticker,
            drivers=transient_strength_labels,
            revenue_growth_ttm=revenue_growth_ttm,
        )

    # Fire only when a one-time item coincides with reported strength — that is
    # the catalyst-framing risk. Distressed companies merely discussing e.g.
    # "asset sale options" have no strength to normalize, so they are excluded.
    if has_current_strength and requires_normalized_earnings_bridge(raw_report):
        red_flags.append(
            {
                "type": "NORMALIZED_EARNINGS_REQUIRED",
                "severity": "WARNING",
                "detail": "One-time event (disposal/deconsolidation/M&A/settlement/subsidy) credited alongside current strength without a normalized-earnings bridge.",
                "action": "REVIEW",
                "risk_penalty": 0.0,
                "rationale": "Distortion-before-catalyst discipline: classify the one-time item as an earnings/cash-flow distortion first. It may be credited as a catalyst only after normalized (ex-one-time) revenue, margin, EPS, and OCF are reconciled. Cap/withhold BUY until then; carries 0.0 tally weight to avoid double-counting TRANSIENT_STRENGTH_DISTORTION.",
            }
        )
        logger.debug("red_flag_normalized_earnings_required", ticker=ticker)

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
            logger.debug(
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
            logger.debug(
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
            logger.debug("red_flag_unreliable_peg", ticker=ticker, peg=peg_for_floor)

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
        logger.debug("red_flag_segment_deterioration", ticker=ticker)

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
        logger.debug("red_flag_ocf_single_source", ticker=ticker)
    elif ocf_source == "FILING":
        red_flags.append(
            {
                "type": "OCF_SOURCE_DISCREPANCY",
                "severity": "WARNING",
                "detail": "OCF value sourced from filing differs from API data — verify",
                "action": "RISK_PENALTY",
                "risk_penalty": 0.5,
                "rationale": "The Senior Fundamentals Analyst preferred the filing-sourced OCF over the API-sourced value due to a >30% discrepancy. This may indicate a yfinance data error, currency mismatch, or period mismatch. Neither source is presumptively correct: a search-derived 'filing' figure can be the wrong statement line. Reconcile to the actual cash-flow statement and corroborate against the forensic auditor's independent OCF before building any cash-quality narrative on the higher value.",
            }
        )
        logger.debug(
            "red_flag_ocf_source_discrepancy",
            ticker=ticker,
            ocf_filing_reason=ocf_reason,
        )

    ocf_period = (metrics.get("ocf_period") or "").upper()
    if ocf_source == "FILING" and re.match(r"\s*(Q[1-4]|H[12])", ocf_period):
        red_flags.append(
            {
                "type": "OCF_PERIOD_NORMALIZATION",
                "severity": "INFO",
                "detail": (
                    f"Headline OCF is sub-annual ({ocf_period}); compare cash-flow "
                    "claims only on a same-period or TTM/annualized basis"
                ),
                "action": "NOTE",
                "risk_penalty": 0.0,
                "rationale": (
                    "A single-quarter or half-year filing OCF is not directly "
                    "comparable to TTM net income, TTM free cash flow, or annual "
                    "payout ratios. Any cash-conversion or dividend-coverage claim "
                    "must use a same-period figure or an explicitly labeled "
                    "annualized estimate."
                ),
            }
        )
        logger.debug(
            "red_flag_ocf_period_normalization", ticker=ticker, ocf_period=ocf_period
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
        logger.debug(
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
        logger.debug("red_flag_thin_consensus", ticker=ticker, total_est=total_est)
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
        logger.debug("red_flag_local_coverage_high", ticker=ticker, total_est=total_est)

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
            logger.debug(
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
                logger.debug(
                    "strict_earnings_quality_rejected",
                    ticker=ticker,
                    ocf_ni_ratio=ratio,
                    ocf=ocf,
                    net_income=ni,
                )

    has_auto_reject = any(flag["action"] == "AUTO_REJECT" for flag in red_flags)
    return red_flags, "REJECT" if has_auto_reject else "PASS"
