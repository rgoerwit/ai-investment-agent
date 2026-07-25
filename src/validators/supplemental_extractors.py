"""Supplemental parser families for legal, moat, capital-efficiency, and consultant checks."""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from src.data_block_utils import extract_last_data_block
from src.validators.metric_extractor import parse_ratio_or_percent

logger = structlog.get_logger(__name__)

# Canonical legal-status enum vocabularies. These are the single source of truth
# for the alternations below AND for the L0 parity test, which asserts each set
# equals the tokens the Legal Counsel prompt advertises (and, for CMIC, the chart
# extractor's own copy). Keep in sync with prompts/legal_counsel.json.
PFIC_STATUS_TOKENS = ("CLEAN", "UNCERTAIN", "PROBABLE", "N/A")
VIE_STRUCTURE_TOKENS = ("YES", "NO", "N/A")
CMIC_STATUS_TOKENS = ("FLAGGED", "UNCERTAIN", "CLEAR", "N/A")


def _alternation(tokens: tuple[str, ...]) -> str:
    """Build a capturing regex alternation from an enum token tuple."""
    return "(" + "|".join(tokens) + ")"


# A large operating-metric decline stated in narrative text. Deliberately
# limited to operating metrics + decline language so share-price drawdowns and
# small moves do not match. Threshold applied in the extractor below.
#
# Pattern A: English prose with an explicit decline verb
#   ("operating profit was down 53%", "OP collapsed 53%").
_MATERIAL_OPERATING_MOVE_RE = re.compile(
    r"\b(operating profit|operating income|operating earnings|OP|ordinary profit|"
    r"recurring profit|ebit|ebitda|net (?:income|profit)|pre-?tax profit|earnings)\b"
    r"[^\n]{0,80}?\b(?:down|declin\w*|decreas\w*|fell|fall|drop\w*|collaps\w*|"
    r"plunge\w*|slump\w*|sank|tumbl\w*)\b"
    r"[^\n]{0,15}?(\d{2,3}(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
# Pattern B: terse earnings-table / abbreviation / Japanese forms with an
# explicit negative marker (-, unicode minus, JP ▲/△) or the JP suffix 減
#   ("operating profit -53.2% YoY", "OP -53.2%", "営業利益 ▲53.2%", "営業利益 53.2%減").
_MATERIAL_OPERATING_TERSE_RE = re.compile(
    r"(?P<metric>\bOP\b|op\.?\s*profit|operating profit|operating income|"
    r"net (?:income|profit)|recurring profit|ordinary profit|"
    r"営業利益|経常利益|純利益|当期純利益)"
    r"[^\n]{0,20}?"
    r"(?:[-−▲△]\s*(?P<pct_sign>\d{2,3}(?:\.\d+)?)\s*%"
    r"|(?P<pct_jp>\d{2,3}(?:\.\d+)?)\s*%\s*減)",
    re.IGNORECASE,
)
_MATERIAL_OPERATING_MIN_DECLINE_PCT = 30.0


def extract_material_unverified_operating_signal(
    text: str | None,
) -> dict[str, Any] | None:
    """Detect a large operating-metric decline mentioned in narrative text.

    Returns ``{"metric", "decline_pct"}`` for a >=30% decline of an operating
    metric (operating profit/income, EBIT(DA), net income, earnings), else None.
    Matches both English prose (pattern A) and terse table / abbreviation /
    Japanese forms with an explicit negative marker or 減 suffix (pattern B).
    Restricted to operating metrics + decline markers so share-price drawdowns,
    positive moves, and sub-30% moves do not match.
    """
    if not text:
        return None

    match = _MATERIAL_OPERATING_MOVE_RE.search(text)
    if match:
        decline_pct = float(match.group(2))
        if decline_pct >= _MATERIAL_OPERATING_MIN_DECLINE_PCT:
            return {"metric": match.group(1).lower(), "decline_pct": decline_pct}

    terse = _MATERIAL_OPERATING_TERSE_RE.search(text)
    if terse:
        raw_pct = terse.group("pct_sign") or terse.group("pct_jp")
        if raw_pct:
            decline_pct = float(raw_pct)
            if decline_pct >= _MATERIAL_OPERATING_MIN_DECLINE_PCT:
                return {
                    "metric": terse.group("metric").strip().lower(),
                    "decline_pct": decline_pct,
                }

    return None


MATERIAL_EVENTS_TOKENS = ("FOUND", "NONE_FOUND")

# Leading [\s*-]* tolerates bullets and markdown-bold token wrappers.
_MATERIAL_EVENTS_TOKEN_RE = re.compile(
    rf"(?im)^[\s*-]*MATERIAL_EVENTS_90D\*{{0,2}}:\s*\*{{0,2}}"
    rf"({_alternation(MATERIAL_EVENTS_TOKENS)})\b"
)
# Legacy fallback for pre-v5.4 news reports that state the absence in prose,
# e.g. "No material operational events ... have been reported in the last 90 days."
_NO_MATERIAL_EVENTS_PROSE_RE = re.compile(
    r"(?i)\bno (?:material|significant|notable)\b"
    r"[^.\n]{0,80}\b(?:operational\s+)?(?:events?|news|developments?)\b"
)
_DRAWDOWN_EXPLANATION_RE = re.compile(
    r"(?im)^[\s*-]*DRAWDOWN_EXPLANATION\*{0,2}:\s*([^\n]+)"
)


def extract_material_events_status(news_report: str | None) -> str | None:
    """Return FOUND / NONE_FOUND from the news report, or None when unstated.

    Prefers the structured ``MATERIAL_EVENTS_90D`` token (news prompt >= v5.4);
    falls back to the legacy "no material ... events" prose so older reports
    still classify. An empty/absent report returns None — artifact absence is
    penalized elsewhere and must not read as "no events".
    """
    if not news_report:
        return None
    token = _MATERIAL_EVENTS_TOKEN_RE.search(news_report)
    if token:
        return token.group(1).upper()
    if _NO_MATERIAL_EVENTS_PROSE_RE.search(news_report):
        return "NONE_FOUND"
    return None


_DRAWDOWN_NOT_FOUND_VALUES = {"NOT_FOUND", "N/A", "NONE"}


def extract_drawdown_explanation(news_report: str | None) -> str | None:
    """Return the DRAWDOWN_EXPLANATION line value, or None when absent/NOT_FOUND."""
    if not news_report:
        return None
    match = _DRAWDOWN_EXPLANATION_RE.search(news_report)
    if not match:
        return None
    value = match.group(1).strip().strip("*").strip()
    if not value or value.upper().rstrip(".") in _DRAWDOWN_NOT_FOUND_VALUES:
        return None
    return value


def drawdown_explanation_not_found(news_report: str | None) -> bool:
    """True when the news report *explicitly* reports DRAWDOWN_EXPLANATION: NOT_FOUND.

    Distinct from the line being absent: an explicit NOT_FOUND means the drawdown
    protocol ran its targeted searches and failed — the strongest available
    evidence the decline is uninvestigated.
    """
    if not news_report:
        return False
    match = _DRAWDOWN_EXPLANATION_RE.search(news_report)
    if not match:
        return False
    value = match.group(1).strip().strip("*").strip()
    return value.upper().rstrip(".") in _DRAWDOWN_NOT_FOUND_VALUES


_CONSULTANT_GROWTH_QUALITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\borganic\s+vs\.?\s+acquired\b", re.IGNORECASE),
    re.compile(
        r"\bgrowth quality\b.*\b(?:inferred|not proven|unknown|unproven)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bacquisition-led growth\b", re.IGNORECASE),
    re.compile(r"\bm&a illusion\b", re.IGNORECASE),
    re.compile(
        r"\b(?:incremental roic|incremental return(?:s)?|return on invested capital|synerg(?:y|ies))\b.*\b(?:unknown|unproven|not proven|not demonstrated|missing|weak|poor)\b",
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
    re.compile(
        r"\b(?:buybacks?|repurchases?|shareholder returns?|payouts?|dilution)\b.*\b(?:unsupported|unverified|unverifiable|not proven|(?<!not )weak|(?<!not )poor|(?<!not )excessive)\b",
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

# Consultant verdict markers, most severe first (first match wins). Tolerant of
# space/underscore/hyphen separators and singular "CONCERN".
_CONSULTANT_VERDICT_PATTERNS = (
    (re.compile(r"MAJOR[\s_-]CONCERNS?", re.IGNORECASE), "MAJOR_CONCERNS"),
    (re.compile(r"CONDITIONAL[\s_-]APPROVAL", re.IGNORECASE), "CONDITIONAL_APPROVAL"),
    (re.compile(r"\bAPPROVED\b", re.IGNORECASE), "APPROVED"),
)
_CONSULTANT_MANDATE_BREACH_PATTERN = re.compile(r"MANDATE[\s_-]BREACH", re.IGNORECASE)
_CONSULTANT_HARD_STOP_PATTERN = re.compile(r"HARD[\s_-]STOP", re.IGNORECASE)
# A negation shortly before a breach/hard-stop mention on the same
# line/sentence ("No mandate breach triggered", "not yet a mandate breach")
# means the consultant is clearing the condition, not raising it (3393.T
# 2026-07-04: a cleared mandate charged a false +2.0 CONSULTANT_MANDATE_BREACH).
_CONSULTANT_NEGATION_BEFORE_PATTERN = re.compile(
    r"\b(?:no|not|none|without|never|absent)\b[^.\n]{0,50}$", re.IGNORECASE
)
# When the review has a FINAL CONSULTANT VERDICT section, breach/hard-stop
# markers are read from that section only — body prose discusses these
# conditions hypothetically ("...though not yet a mandate breach").
_CONSULTANT_FINAL_VERDICT_SECTION_PATTERN = re.compile(
    r"FINAL\s+CONSULTANT\s+VERDICT.*\Z", re.IGNORECASE | re.DOTALL
)
# Structured verdict-block tokens (consultant prompt v2.12+): explicit
# `MANDATE_BREACH: NONE | <description>` lines. Preferred over prose scanning
# when present; tolerate bullet/bold decoration around the key.
_CONSULTANT_MANDATE_BREACH_TOKEN_PATTERN = re.compile(
    r"^[ \t]*(?:[-•*][ \t]+)?\**MANDATE[ _-]BREACH\**[ \t]*:[ \t]*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_CONSULTANT_HARD_STOP_TOKEN_PATTERN = re.compile(
    r"^[ \t]*(?:[-•*][ \t]+)?\**HARD[ _-]STOP\**[ \t]*:[ \t]*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
# Token values that clear the condition: "NONE", "N/A", a bare "No", or a
# negated restatement ("No breach detected", "Not triggered"). Anything else
# is treated as a breach description.
_CONSULTANT_TOKEN_CLEAR_PATTERN = re.compile(
    r"(?:NONE|N/?A)\b"
    r"|NO[.!]*$"
    r"|(?:NO|NOT)\b[^.\n]{0,40}"
    r"\b(?:BREACH|STOP|TRIGGERED|DETECTED|IDENTIFIED|FOUND|APPLICABLE)\b",
    re.IGNORECASE,
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
        "capital_structure": None,
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
        capital_structure = data.get("capital_structure")
        risks["capital_structure"] = (
            capital_structure if isinstance(capital_structure, dict) else None
        )
        risks["country"] = data.get("country")
        risks["sector"] = data.get("sector")
        logger.debug(
            "legal_risks_parsed_json",
            pfic_status=risks["pfic_status"],
            vie_structure=risks["vie_structure"],
            cmic_status=risks["cmic_status"],
        )
        return risks
    except json.JSONDecodeError as exc:
        from src.error_safety import redact_sensitive_text

        # Operator-visible: the Legal Counsel prompt promises JSON output, so a
        # parse failure means format drift — the regex fallback may miss fields.
        logger.warning(
            "legal_report_json_parse_failed_using_regex_fallback",
            reason=str(exc.msg)[:120],
            report_prefix=redact_sensitive_text(json_str, max_chars=80),
        )

    pfic_match = re.search(
        rf'"?pfic_status"?\s*:\s*"?{_alternation(PFIC_STATUS_TOKENS)}"?',
        legal_report,
        re.IGNORECASE,
    )
    if pfic_match:
        risks["pfic_status"] = pfic_match.group(1).upper()

    vie_match = re.search(
        rf'"?vie_structure"?\s*:\s*"?{_alternation(VIE_STRUCTURE_TOKENS)}"?',
        legal_report,
        re.IGNORECASE,
    )
    if vie_match:
        risks["vie_structure"] = vie_match.group(1).upper()

    cmic_match = re.search(
        rf'"?cmic_status"?\s*:\s*"?{_alternation(CMIC_STATUS_TOKENS)}"?',
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
        "m_and_a_context_evidence": None,
        "m_and_a_context_source_url": None,
        "m_and_a_context": None,
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
        ("M&A_CONTEXT_EVIDENCE", "m_and_a_context_evidence"),
        ("M&A_CONTEXT_SOURCE_URL", "m_and_a_context_source_url"),
        ("M&A_CONTEXT", "m_and_a_context"),
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
                metrics[key] = (
                    value.upper() if field == "M&A_CONTEXT_EVIDENCE" else value
                )

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

    plan_strength_match = re.search(
        r"VALUE_UP_PLAN_STRENGTH:\s*(STRONG|MODERATE|WEAK|NONE|UNKNOWN|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if plan_strength_match:
        value = plan_strength_match.group(1).upper()
        if value != "N/A":
            signals["value_up_plan_strength"] = value

    execution_match = re.search(
        r"SHAREHOLDER_RETURN_EXECUTION:\s*(PROVEN|PARTIAL|ANNOUNCED_ONLY|NONE|UNKNOWN|N/A)",
        data_block,
        re.IGNORECASE,
    )
    if execution_match:
        value = execution_match.group(1).upper()
        if value != "N/A":
            signals["shareholder_return_execution"] = value

    return signals


def _non_negated_search(pattern: re.Pattern[str], text: str) -> bool:
    """True when *pattern* matches without a negation shortly before it."""
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 60) : match.start()]
        if not _CONSULTANT_NEGATION_BEFORE_PATTERN.search(prefix):
            return True
    return False


def _breach_token_value(token_pattern: re.Pattern[str], text: str) -> bool | None:
    """Read structured ``KEY: value`` verdict tokens; None when absent/empty.

    A token carrying a breach description wins over a clearing token —
    conservative when a review contains both forms.
    """
    saw_clear = False
    for match in token_pattern.finditer(text):
        value = match.group(1).strip("* \t")
        if not value:
            continue
        if _CONSULTANT_TOKEN_CLEAR_PATTERN.match(value):
            saw_clear = True
        else:
            return True
    return False if saw_clear else None


def _breach_marker_present(
    token_pattern: re.Pattern[str],
    prose_pattern: re.Pattern[str],
    text: str,
) -> bool:
    """Structured token wins when present; otherwise negation-aware prose scan."""
    token = _breach_token_value(token_pattern, text)
    if token is not None:
        return token
    return _non_negated_search(prose_pattern, text)


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

    # The verdict is authoritative only in the FINAL CONSULTANT VERDICT section
    # ("Overall Assessment: <verdict>"). Body prose can mention "major concern"
    # discursively and must not override a section-level CONDITIONAL APPROVAL
    # (3771.T 2026-07-12: a stray body "major concern" beat the section verdict).
    # Scope the verdict scan to the same section as the breach markers; fall back
    # to the whole review when no section exists.
    section_match = _CONSULTANT_FINAL_VERDICT_SECTION_PATTERN.search(consultant_review)
    scan_text = section_match.group(0) if section_match else consultant_review

    for pattern, verdict in _CONSULTANT_VERDICT_PATTERNS:
        if pattern.search(scan_text):
            result["verdict"] = verdict
            break

    result["has_mandate_breach"] = _breach_marker_present(
        _CONSULTANT_MANDATE_BREACH_TOKEN_PATTERN,
        _CONSULTANT_MANDATE_BREACH_PATTERN,
        scan_text,
    )
    result["has_hard_stop"] = _breach_marker_present(
        _CONSULTANT_HARD_STOP_TOKEN_PATTERN,
        _CONSULTANT_HARD_STOP_PATTERN,
        scan_text,
    )

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
