from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import structlog
from langchain_core.messages import HumanMessage
from langgraph.types import RunnableConfig

from src.agents.context_flags import format_pm_context_flags
from src.agents.pm_inputs import (
    DIRECT_PM_INPUT_FIELDS,
    RISK_DEBATE_FIELD,
    pm_input_present,
    risk_debate_present,
)
from src.agents.pm_verdict_metadata import (
    PMVerdictMetadata,
    PMVerdictRecovery,
    pm_verdict_metadata_from_text,
)
from src.data_block_utils import (
    extract_data_block_field,
    fenced_block_pattern,
    has_parseable_data_block,
    unfenced_label,
)
from src.error_safety import summarize_exception

# Verdict canonicalization lives in the neutral, dependency-free parser (it used
# to live on src.agents.pm_verdict_metadata, which created a charts circular
# import: pm_block -> pm_verdict_metadata -> agents/__init__ -> decision_nodes).
from src.pm_decision_parser import canonicalize_pm_verdict, parse_final_decision_scores
from src.runtime_diagnostics import (
    failure_artifact,
    get_artifact_status,
    get_valid_artifact_content,
    success_artifact,
)

from . import message_utils, support
from . import runtime as agent_runtime
from .governance_prompt import governance_block, governance_card
from .output_limits import cap_state_value
from .output_validation import (
    log_output_diagnostics,
    log_truncation_diagnostic,
    should_fail_closed,
    validate_required_output,
)
from .state import AgentState

logger = structlog.get_logger(__name__)


async def _recover_pm_verdict_metadata(
    pm_output: str,
    llm: Any,
) -> PMVerdictMetadata | None:
    """Recover final PM verdict from prose when PM_BLOCK/text regex parsing fails."""
    if not hasattr(llm, "with_structured_output"):
        return None
    try:
        structured_llm = llm.with_structured_output(PMVerdictRecovery, strict=True)
        response = await agent_runtime.invoke_with_rate_limit_handling(
            structured_llm,
            [
                HumanMessage(
                    content=(
                        "Extract only the final Portfolio Manager verdict from this "
                        "analysis. Return the structured verdict. Valid values are "
                        "BUY, HOLD, SELL, DO_NOT_INITIATE.\n\n"
                        f"{pm_output[:8000]}"
                    )
                )
            ],
            context="pm_verdict_recovery",
            provider=support.infer_provider_name(llm),
            model_name=support.get_model_name(llm),
            max_attempts=1,
            max_transient_attempts=1,
        )
    except Exception as exc:
        logger.warning(
            "pm_verdict_recovery_failed",
            **summarize_exception(exc, operation="pm_verdict_recovery"),
        )
        return None

    verdict = getattr(response, "verdict", None)
    canonical = canonicalize_pm_verdict(verdict)
    if canonical == "UNPARSEABLE":
        return None
    return PMVerdictMetadata(verdict=canonical)


def _present_pm_inputs(state: AgentState) -> tuple[list[str], list[str]]:
    """Return (present, missing) lists of direct PM inputs by validity.

    Uses get_valid_artifact_content so failure-artifact stubs and "N/A"
    fallbacks count as missing, not present.
    """
    present: list[str] = []
    missing: list[str] = []
    for field in DIRECT_PM_INPUT_FIELDS:
        valid = pm_input_present(state, field)
        bucket = present if valid else missing
        bucket.append(field)
    if risk_debate_present(state):
        present.append(RISK_DEBATE_FIELD)
    else:
        missing.append(RISK_DEBATE_FIELD)
    return present, missing


def _parse_price_value(raw: str | None) -> float | None:
    if not raw:
        return None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(raw))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


_STRICT_PM_ADDENDUM = """
---
## STRICT MODE ACTIVE — Additional Screening Rules

Apply these as a final quality gate AFTER your normal Step 1A-1C analysis.
They OVERRIDE normal zone decisions where they conflict.

### AUTOMATIC DO_NOT_INITIATE (regardless of risk zone):

1. **Any PFIC flag** (PFIC_PROBABLE or PFIC_UNCERTAIN): → DO_NOT_INITIATE
2. **VIE structure detected**: → DO_NOT_INITIATE
3. **Value Trap HIGH** (score < 40 or TRAP verdict): → DO_NOT_INITIATE
4. **Risk tally ≥ 1.5** (vs normal 2.0 cutoff): → DO_NOT_INITIATE
5. **Data vacuum present** (missing sector, OCF, or analyst coverage): → DO_NOT_INITIATE
6. **Normal HOLD verdict**: → Upgrade to DO_NOT_INITIATE (no watchlist positions)

### TIGHTER BUY REQUIREMENTS (all must hold):
- Financial Health ≥ 60% (vs 50% normal)
- Growth Score ≥ 55% (vs 50% normal)
- Analyst Coverage ≤ 10 (vs 15 normal)
- P/E ≤ 15 (vs 18/25 contextual normal)
- Liquidity ≥ $750k daily USD (vs $250k full-pass normal)
- Graham Earnings Test: PASS
- Risk tally < 1.5

### POSITION SIZING:
- Maximum: 5% (vs 10% normal)
- Authoritarian regimes: MAX 1.5% (vs 2%)
- Any remaining uncertainty: MAX 2%
"""


def _extract_consultant_resolution_concerns(consultant_review: str) -> list[str]:
    """Extract a small set of material consultant concerns for PM fallback output."""
    if not consultant_review:
        return []

    concern_keywords = (
        "error",
        "concern",
        "discrep",
        "conflict",
        "bias",
        "unanswered",
        "unsupported",
        "unsubstantiated",
        "mismatch",
        "risk",
    )
    concerns: list[str] = []
    seen: set[str] = set()

    for raw_line in consultant_review.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not re.match(r"^(?:[-*•]|\d+\.)\s+", line):
            continue
        normalized = re.sub(r"^(?:[-*•]|\d+\.)\s+", "", line).strip()
        if len(normalized) < 12:
            continue
        if not any(keyword in normalized.lower() for keyword in concern_keywords):
            continue
        normalized = normalized.rstrip(".")
        if normalized not in seen:
            seen.add(normalized)
            concerns.append(normalized)
        if len(concerns) >= 3:
            break

    return concerns


def _ensure_consultant_resolution_block(
    pm_output: str, consultant_review: str | None
) -> str:
    """Ensure PM output always includes CONSULTANT_RESOLUTION when consultant ran."""
    label = unfenced_label("CONSULTANT_RESOLUTION")
    if not consultant_review or label in pm_output:
        return pm_output

    from src.validators.red_flag_detector import RedFlagDetector

    conditions = RedFlagDetector.parse_consultant_conditions(consultant_review)
    concerns = _extract_consultant_resolution_concerns(consultant_review)

    if conditions["verdict"] == "APPROVED" and not concerns:
        resolution_lines = [
            label,
            "- CONCERN: NONE",
            "- DATA_CHECK: N/A",
            "- VERDICT: N/A",
        ]
    else:
        if not concerns:
            concerns = [
                "Consultant raised unresolved issues that Portfolio Manager did not explicitly address"
            ]
        resolution_lines = []
        for concern in concerns:
            resolution_lines.extend(
                [
                    label,
                    f"- CONCERN: {concern}",
                    "- DATA_CHECK: NOT_PROVIDED",
                    "- VERDICT: UNVERIFIABLE",
                ]
            )

    resolution_block = "\n".join(resolution_lines).rstrip()
    return _insert_block_before_pm_block(pm_output, resolution_block)


_APAC_SILENCE_SENTINELS = {
    "NO_MATERIAL_APAC_CONNECTION",
    "APAC_SPECIALIST_UNAVAILABLE",
}


def _requires_apac_resolution(apac_report: str | None) -> bool:
    """True iff APAC produced non-silent, non-error output that PM should reconcile."""
    if not apac_report:
        return False
    stripped = apac_report.strip()
    if not stripped:
        return False
    return stripped not in _APAC_SILENCE_SENTINELS


def _extract_apac_verdict_line(apac_report: str) -> str:
    """Pull a one-line summary of the APAC specialist's verdict, if available."""
    if not apac_report:
        return ""
    match = re.search(
        r"\*\*VERDICT FOR CONSULTANT AND PM\*\*\s*:\s*(.+)",
        apac_report,
    )
    if match:
        # Take only the first line of the verdict sentence.
        line = match.group(1).splitlines()[0].strip()
        # Cap absurdly long values defensively.
        return line[:400]
    for tag in ("OVERRIDE", "CAUTION", "SUPPORT"):
        if tag in apac_report:
            return f"APAC specialist verdict {tag} (verdict line not extractable)."
    return ""


def _insert_block_before_pm_block(pm_output: str, block_text: str) -> str:
    """Insert a resolution block immediately above PM_BLOCK (or at tail if absent)."""
    match = fenced_block_pattern("PM_BLOCK").search(pm_output)
    if match:
        return (
            f"{pm_output[: match.start()]}{block_text.rstrip()}\n\n"
            f"{pm_output[match.start() :]}"
        )
    return f"{pm_output.rstrip()}\n\n{block_text.rstrip()}\n"


def _normalize_pm_block_contract(pm_output: str) -> str:
    """Rewrite final PM_BLOCK text when its sizing violates verdict semantics.

    PM_BLOCK extraction also clamps no-initiation position size for all readers,
    including older artifacts. This boundary rewrite keeps newly persisted PM
    output internally coherent with that same contract.
    """
    blocks = list(fenced_block_pattern("PM_BLOCK").finditer(pm_output))
    if not blocks:
        return pm_output

    last = blocks[-1]
    body = last.group(1)
    verdict_match = re.search(r"(?im)^VERDICT:\s*([^\n]+)", body)
    size_match = re.search(r"(?im)^(POSITION_SIZE:\s*)([\d.]+)", body)
    if not verdict_match or not size_match:
        return pm_output

    verdict = canonicalize_pm_verdict(verdict_match.group(1))
    try:
        emitted_size = float(size_match.group(2))
    except ValueError:
        return pm_output

    if verdict not in {"HOLD", "DO_NOT_INITIATE", "SELL"} or emitted_size == 0.0:
        return pm_output

    logger.warning(
        "pm_block_position_size_rewritten",
        verdict=verdict,
        emitted_position_size=emitted_size,
    )
    rewritten_body = re.sub(
        r"(?im)^(POSITION_SIZE:\s*)[\d.]+",
        r"\g<1>0.0",
        body,
        count=1,
    )
    return pm_output[: last.start(1)] + rewritten_body + pm_output[last.end(1) :]


def _ensure_apac_resolution_block(pm_output: str, apac_report: str | None) -> str:
    """Ensure PM output includes APAC_RESOLUTION when APAC produced non-silent output.

    Mirrors the consultant pattern: pure programmatic fallback insertion, no
    LLM retry. If PM already emitted an APAC_RESOLUTION block we leave it
    alone; otherwise we splice in a deterministic placeholder so downstream
    tooling can rely on the block being present.
    """
    if not _requires_apac_resolution(apac_report):
        return pm_output
    label = unfenced_label("APAC_RESOLUTION")
    if label in pm_output:
        return pm_output
    summary = _extract_apac_verdict_line(apac_report or "") or (
        "APAC specialist output not reconciled in PM rationale."
    )
    fallback = (
        f"{label}\n"
        f"- FINDING: {summary}\n"
        "- DATA_CHECK: NOT_PROVIDED\n"
        "- VERDICT: UNVERIFIABLE"
    )
    return _insert_block_before_pm_block(pm_output, fallback)


# Auditor fallback gating (Tranche 5, Step 7).
#
# The original implementation inserted a fallback "Forensic Auditor flagged
# anomalies not explicitly addressed" block for any non-empty, non-
# INSUFFICIENT_DATA auditor output — which was misleading on clean audits.
#
# The corrected gating runs negative phrases FIRST so substring-style
# positive tokens (e.g. "RED FLAG") don't trip on negations like
# "no red flags." Positive detection then requires evidence of a NAMED
# forensic check from the auditor's framework (Paper Profit, Zombie Ratio,
# Trash Bin, …) rather than generic words.

_AUDITOR_NEGATIVE_PHRASES: tuple[str, ...] = (
    "NO ANOMAL",
    "NO MATERIAL ANOMAL",
    "NO MATERIAL CONCERN",
    "NO RED FLAG",
    "NO CONCERNS DETECTED",
    "NO ISSUES DETECTED",
    "ANOMALY_COUNT: 0",
    "ANOMALIES: 0",
    "ANOMALIES: NONE",
    "STATUS=INSUFFICIENT_DATA",
    "STATUS: INSUFFICIENT_DATA",
    "INSUFFICIENT DATA",
    "AUDITOR_VERDICT: CLEAN",
    "AUDIT VERDICT: CLEAN",
)

_AUDITOR_POSITIVE_TOKENS: tuple[str, ...] = (
    "PAPER PROFIT",
    "BALLOONING DSO",
    "ZOMBIE RATIO",
    "INVENTORY HOARDING",
    "ACQUISITION HANGOVER",
    "STRETCHING PAYABLES",
    "TRASH BIN",
    "GHOST CASH",
    "ACCRUAL RATIO",
    "FORENSIC FLAG",
)


def _auditor_has_material_concern(auditor_report: str | None) -> bool:
    """True only when the auditor named at least one specific forensic anomaly.

    Conservative by design: a non-empty report that says "no red flags" /
    "clean" / "INSUFFICIENT_DATA" returns False. Only fires when we can name
    the specific forensic check from the auditor framework that flagged.
    """
    if not auditor_report:
        return False
    stripped = auditor_report.strip()
    if not stripped:
        return False
    upper = stripped.upper()
    if any(phrase in upper for phrase in _AUDITOR_NEGATIVE_PHRASES):
        return False
    return any(token in upper for token in _AUDITOR_POSITIVE_TOKENS)


def _ensure_auditor_resolution_block(pm_output: str, auditor_report: str | None) -> str:
    """Ensure PM output includes AUDITOR_RESOLUTION when the auditor named anomalies.

    Refined in Tranche 5, Step 7: fallback fires only when the auditor named
    at least one specific forensic check (Paper Profit, Zombie Ratio, etc.),
    never on clean output or sentinel-coded INSUFFICIENT_DATA. PM-emitted
    blocks are left alone.
    """
    if not _auditor_has_material_concern(auditor_report):
        return pm_output
    label = unfenced_label("AUDITOR_RESOLUTION")
    if label in pm_output:
        return pm_output
    fallback = (
        f"{label}\n"
        "- FINDING: Forensic Auditor flagged anomalies not explicitly addressed by PM rationale.\n"
        "- DATA_CHECK: NOT_PROVIDED\n"
        "- VERDICT: UNVERIFIABLE"
    )
    return _insert_block_before_pm_block(pm_output, fallback)


def resolve_pfic_display_status(
    legal_pfic_status: str | None,
    data_block_pfic_risk: str | None,
) -> tuple[str, str | None]:
    """
    Return (canonical_status, note).

    Legal Counsel primary-source research overrides the quantitative heuristic
    in DATA_BLOCK when they disagree.

    Args:
        legal_pfic_status: pfic_status from Legal Counsel (CLEAN/UNCERTAIN/PROBABLE/N/A)
        data_block_pfic_risk: PFIC_RISK from Senior Fundamentals DATA_BLOCK (LOW/MEDIUM/HIGH/N/A)

    Returns:
        (canonical_status, note): note is non-None only when the override fires.
    """
    if not legal_pfic_status or legal_pfic_status.upper() in ("N/A", "CLEAN"):
        return data_block_pfic_risk or "N/A", None

    mapping = {"PROBABLE": "HIGH", "UNCERTAIN": "UNCERTAIN"}
    legal_resolved = mapping.get(legal_pfic_status.upper())
    if not legal_resolved:
        return data_block_pfic_risk or "N/A", None

    if data_block_pfic_risk and data_block_pfic_risk.upper() == "LOW":
        return legal_resolved, (
            f"PFIC_NOTE: Legal Counsel assessment ({legal_pfic_status}) overrides "
            f"quantitative heuristic (LOW). Legal research is primary."
        )
    return legal_resolved, None


def create_trader_node(llm, memory: Any | None) -> Callable:
    async def trader_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("trader")
        if not agent_prompt:
            return failure_artifact(
                "trader_investment_plan",
                "Missing trader prompt",
                provider="unknown",
            )

        consultant = get_valid_artifact_content(state, "consultant_review")
        consultant_section = (
            "\n\nEXTERNAL CONSULTANT REVIEW (Cross-Validation):\n"
            f"{support.summarize_for_pm(consultant, 'consultant', 2500) if consultant else 'N/A (consultant disabled or unavailable)'}"
        )
        apac = get_valid_artifact_content(state, "apac_regional_report")
        apac_section = (
            "\n\nAPAC REGIONAL SPECIALIST:\n"
            f"{support.summarize_for_pm(apac, 'apac', 1800) if apac else 'N/A'}"
        )
        valuation = get_valid_artifact_content(state, "valuation_params")
        valuation_section = (
            f"\n\nVALUATION PARAMETERS:\n{valuation}" if valuation else ""
        )
        governance_section = governance_block(state, with_label=True)
        macro_section = support.macro_section_for(config)

        market_report = get_valid_artifact_content(state, "market_report") or "N/A"
        sentiment_report = (
            get_valid_artifact_content(state, "sentiment_report") or "N/A"
        )
        news_report = get_valid_artifact_content(state, "news_report") or "N/A"
        fundamentals_report = (
            get_valid_artifact_content(state, "fundamentals_report") or "N/A"
        )
        investment_plan = get_valid_artifact_content(state, "investment_plan") or "N/A"
        all_input = f"""MARKET ANALYST REPORT:
{support.summarize_for_pm(market_report, "market", 1800) if market_report != "N/A" else "N/A"}

SENTIMENT ANALYST REPORT:
{support.summarize_for_pm(sentiment_report, "sentiment", 1200) if sentiment_report != "N/A" else "N/A"}

NEWS ANALYST REPORT:
{support.summarize_for_pm(news_report, "news", 1800) if news_report != "N/A" else "N/A"}

FUNDAMENTALS ANALYST REPORT:
{support.summarize_for_pm(fundamentals_report, "fundamentals", 6000) if fundamentals_report != "N/A" else "N/A"}

RESEARCH MANAGER PLAN:
{support.summarize_for_pm(investment_plan, "research", 3500) if investment_plan != "N/A" else "N/A"}{macro_section}{apac_section}{consultant_section}{valuation_section}{governance_section}"""
        prompt = f"{agent_prompt.system_message}\n\n{all_input}\n\nCreate Trading Plan."

        try:
            response = await agent_runtime.invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_prompt.agent_name,
                provider=support.infer_provider_name(llm),
                model_name=support.get_model_name(llm),
            )
            content_str = message_utils.extract_string_content(response.content)
            from src.utils import detect_truncation

            trunc_info = detect_truncation(content_str, agent="trader")
            log_truncation_diagnostic(
                agent_key="trader",
                ticker=state.get("company_of_interest", "UNKNOWN"),
                runnable=llm,
                response=response,
                content=content_str,
                trunc_info=trunc_info,
            )
            log_output_diagnostics(
                agent_key="trader",
                ticker=state.get("company_of_interest", "UNKNOWN"),
                runnable=llm,
                response=response,
                content=content_str,
                truncated=trunc_info["truncated"],
                validation=None,
            )
            return success_artifact(
                "trader_investment_plan",
                cap_state_value(content_str, "trader_investment_plan"),
                provider=support.infer_provider_name(llm),
            )
        except Exception as exc:
            return failure_artifact(
                "trader_investment_plan",
                exc,
                provider=support.infer_provider_name(llm),
            )

    return trader_node


def create_risk_debater_node(llm, agent_key: str) -> Callable:
    agent_field_map = {
        "risky_analyst": "current_risky_response",
        "safe_analyst": "current_safe_response",
        "neutral_analyst": "current_neutral_response",
    }

    async def risk_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt(agent_key)
        field_name = agent_field_map.get(agent_key, "history")

        if not agent_prompt:
            return {
                "risk_debate_state": {
                    field_name: f"[SYSTEM]: Error - Missing prompt for {agent_key}",
                    "latest_speaker": agent_key,
                }
            }

        consultant = get_valid_artifact_content(state, "consultant_review")
        consultant_section = (
            "\n\nEXTERNAL CONSULTANT REVIEW (Cross-Validation):\n"
            f"{consultant if consultant else 'N/A (consultant disabled or unavailable)'}"
        )
        governance_section = governance_block(state, with_label=True)
        macro_section = support.macro_section_for(config)

        trader_plan = (
            get_valid_artifact_content(state, "trader_investment_plan") or "N/A"
        )
        prompt = (
            f"{agent_prompt.system_message}\n\nTRADER PLAN: "
            f"{trader_plan}{consultant_section}{governance_section}{macro_section}\n\n"
            "Provide risk assessment."
        )
        try:
            response = await agent_runtime.invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_prompt.agent_name,
            )
            content_str = message_utils.extract_string_content(response.content)
            return {
                "risk_debate_state": {
                    field_name: f"{agent_prompt.agent_name}: {content_str}",
                    "latest_speaker": agent_prompt.agent_name,
                }
            }
        except Exception as exc:
            summary = summarize_exception(exc, operation=f"risk_node:{agent_key}")
            logger.error("risk_node_failed", agent_key=agent_key, **summary)
            return {
                "risk_debate_state": {
                    field_name: (
                        f"[ERROR]: {agent_key} failed - {summary['error_type']}"
                    ),
                    "latest_speaker": agent_key,
                }
            }

    return risk_node


def _log_risk_tally_reconciliation(
    content_str: str, code_subtotal: float, ticker: str
) -> float | None:
    """Warn when the PM's narrated TOTAL RISK COUNT falls below the deterministic
    code-computed floor — a dropped, weighted pre-screen penalty.

    Best-effort, no verdict override (surface + log only). Returns the shortfall (the
    amount dropped) when the floor is breached, else ``None``. A missing/unparseable
    narrated tally yields ``None`` (cannot reconcile, do not warn).
    """
    narrated = parse_final_decision_scores(content_str).get("risk_tally")
    if narrated is None or narrated >= code_subtotal - 0.01:
        return None
    dropped = round(code_subtotal - narrated, 2)
    logger.warning(
        "pm_risk_tally_below_code_floor",
        ticker=ticker,
        narrated=narrated,
        code_subtotal=round(code_subtotal, 2),
        dropped=dropped,
    )
    return dropped


# Pre-screen flags whose unresolved presence forbids a Zone-2 BUY override (prompt rule).
_OVERRIDE_BLOCKING_FLAGS = {"GROWTH_QUALITY_UNPROVEN", "TRANSIENT_STRENGTH_DISTORTION"}


def _log_pm_discipline_checks(
    content_str: str,
    red_flags: list[dict[str, Any]],
    valuation_reliability: str,
    ticker: str,
) -> None:
    """Log-only reconciliation of the PM override path against the prompt's stated
    thresholds (no verdict/zone mutation — surface + log only).

    Fires only on unambiguous, deterministically-parseable misses. The blocking-flag check
    is load-bearing: it catches the dropped-penalty + override combo where a weaker model
    drops a +0.75 penalty (so ``risk_tally`` looks clean) yet the flag is still present.
    Best-effort; missing scores/verdict/zone yield no warning. Non-raising by construction.
    """
    scores = parse_final_decision_scores(content_str)
    verdict = canonicalize_pm_verdict(scores.get("verdict"))
    zone = str(scores.get("zone") or "").upper()
    health = scores.get("health_adj")
    growth = scores.get("growth_adj")
    risk = scores.get("risk_tally")
    flag_types = {str(flag.get("type", "")).upper() for flag in red_flags}

    if verdict == "BUY" and zone == "MODERATE":
        reasons = []
        if health is not None and health < 50:
            reasons.append("health_below_50")
        if risk is not None and risk > 1.5:
            reasons.append("risk_above_1_5")
        if flag_types & _OVERRIDE_BLOCKING_FLAGS:
            reasons.append("blocking_growth_quality_flag")
        if reasons:
            logger.warning(
                "pm_override_threshold_unmet", ticker=ticker, reasons=reasons
            )

    if verdict == "HOLD" and zone == "HIGH":
        reasons = []
        if health is not None and health < 80:
            reasons.append("health_below_80")
        if growth is not None and growth < 80:
            reasons.append("growth_below_80")
        if reasons:
            logger.warning(
                "pm_hold_override_threshold_unmet", ticker=ticker, reasons=reasons
            )

    if verdict == "BUY" and valuation_reliability == "QUARANTINED":
        logger.warning("pm_buy_on_quarantined_valuation_inputs", ticker=ticker)


def create_portfolio_manager_node(
    llm, memory: Any | None, strict_mode: bool = False
) -> Callable:
    async def pm_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from src.prompts import get_prompt
        from src.validators.red_flag_detector import RedFlagDetector

        agent_prompt = get_prompt("portfolio_manager")
        if not agent_prompt:
            return failure_artifact(
                "final_trade_decision",
                "Missing portfolio_manager prompt",
                provider="unknown",
            )

        market = get_valid_artifact_content(state, "market_report")
        sentiment = get_valid_artifact_content(state, "sentiment_report")
        news = get_valid_artifact_content(state, "news_report")
        fundamentals = get_valid_artifact_content(state, "fundamentals_report")
        foreign_language = get_valid_artifact_content(state, "foreign_language_report")
        value_trap = get_valid_artifact_content(state, "value_trap_report")
        inv_plan = get_valid_artifact_content(state, "investment_plan")
        consultant = get_valid_artifact_content(state, "consultant_review")
        trader = get_valid_artifact_content(state, "trader_investment_plan")

        risk_state = state.get("risk_debate_state", {})
        risky_view = risk_state.get("current_risky_response", "")
        safe_view = risk_state.get("current_safe_response", "")
        neutral_view = risk_state.get("current_neutral_response", "")
        risk = f"""RISKY ANALYST (Aggressive):
{risky_view if risky_view else "N/A"}

SAFE ANALYST (Conservative):
{safe_view if safe_view else "N/A"}

NEUTRAL ANALYST (Balanced):
{neutral_view if neutral_view else "N/A"}"""

        pre_screening_result = state.get("pre_screening_result", "N/A")
        red_flags = list(state.get("red_flags", []))
        ticker = state.get("company_of_interest", "UNKNOWN")
        macro_section = support.macro_section_for(config)

        if value_trap:
            value_trap_warnings = RedFlagDetector.detect_value_trap_flags(
                value_trap,
                ticker,
                m_and_a_status=extract_data_block_field(fundamentals, "M_AND_A_STATUS"),
            )
            if value_trap_warnings:
                red_flags.extend(value_trap_warnings)
                logger.info(
                    "value_trap_warnings_detected",
                    ticker=ticker,
                    warning_types=[warning["type"] for warning in value_trap_warnings],
                    total_risk_penalty=sum(
                        warning.get("risk_penalty", 0)
                        for warning in value_trap_warnings
                    ),
                )

        if fundamentals:
            sector = RedFlagDetector.detect_sector(fundamentals)

            moat_bonuses = RedFlagDetector.detect_moat_flags(fundamentals, ticker)
            if moat_bonuses:
                red_flags.extend(moat_bonuses)
                logger.info(
                    "moat_bonuses_detected",
                    ticker=ticker,
                    bonus_types=[bonus["type"] for bonus in moat_bonuses],
                    total_risk_bonus=sum(
                        bonus.get("risk_penalty", 0) for bonus in moat_bonuses
                    ),
                )

            capital_flags = RedFlagDetector.detect_capital_efficiency_flags(
                fundamentals,
                ticker,
                value_trap_report=value_trap,
                sector=sector,
            )
            if capital_flags:
                red_flags.extend(capital_flags)
                logger.info(
                    "capital_efficiency_flags_detected",
                    ticker=ticker,
                    flag_types=[flag["type"] for flag in capital_flags],
                    total_risk_adjustment=sum(
                        flag.get("risk_penalty", 0) for flag in capital_flags
                    ),
                )

            value_up_bonuses = (
                RedFlagDetector.detect_shareholder_return_execution_flags(
                    fundamentals,
                    value_trap_report=value_trap,
                    ticker=ticker,
                )
            )
            if value_up_bonuses:
                red_flags.extend(value_up_bonuses)
                logger.info(
                    "value_up_executed_bonus_detected",
                    ticker=ticker,
                    bonus_types=[bonus["type"] for bonus in value_up_bonuses],
                    total_risk_bonus=sum(
                        bonus.get("risk_penalty", 0) for bonus in value_up_bonuses
                    ),
                )

        # Scan narrative artifacts (not the structured DATA_BLOCK) for a large,
        # unverified operating decline that must block BUY pending verification.
        narrative_text = "\n\n".join(
            str(part)
            for part in (news, value_trap, foreign_language)
            if isinstance(part, str) and part
        )
        if narrative_text:
            material_signal_flags = (
                RedFlagDetector.detect_material_operating_signal_flags(
                    narrative_text, ticker
                )
            )
            if material_signal_flags:
                red_flags.extend(material_signal_flags)
                logger.info(
                    "material_operating_signal_flags_detected",
                    ticker=ticker,
                    flag_types=[flag["type"] for flag in material_signal_flags],
                )

        consultant_review = get_valid_artifact_content(state, "consultant_review")
        if consultant_review:
            if not isinstance(consultant_review, str):
                consultant_review = message_utils.extract_string_content(
                    consultant_review
                )
            consultant_conditions = RedFlagDetector.parse_consultant_conditions(
                consultant_review
            )
            consultant_flags = RedFlagDetector.detect_consultant_flags(
                consultant_conditions,
                ticker,
            )
            if consultant_flags:
                red_flags.extend(consultant_flags)
                logger.info(
                    "consultant_flags_detected",
                    ticker=ticker,
                    flag_types=[flag["type"] for flag in consultant_flags],
                    total_risk_penalty=sum(
                        flag.get("risk_penalty", 0) for flag in consultant_flags
                    ),
                )

        # OCF corroboration: the forensic auditor computes operating cash flow
        # independently of the Foreign-Language "filing" value the Senior may have
        # promoted under FILING AUTHORITY. The auditor only completes post-research,
        # so this cross-check belongs here (not in the pre-screening validator). A
        # material divergence blocks the "elite cash generation" overclaim without
        # escalating the risk tally (OCF_SOURCE_DISCREPANCY already carries any
        # penalty). See KTY.WA 2026-06-27: filing 1.148B vs auditor ~971M. The
        # auditor content is read through the validity-gated accessor so a failed
        # auditor artifact is never parsed as a corroborating figure.
        ocf_corroboration_flag = RedFlagDetector.detect_ocf_corroboration_flag(
            RedFlagDetector.parse_ocf_amount(
                extract_data_block_field(fundamentals, "OPERATING_CASH_FLOW")
            ),
            RedFlagDetector.extract_auditor_ocf(
                get_valid_artifact_content(state, "auditor_report") or None
            ),
            ticker,
        )
        if ocf_corroboration_flag:
            red_flags.append(ocf_corroboration_flag)
            logger.info(
                "ocf_corroboration_flag_detected",
                ticker=ticker,
                detail=ocf_corroboration_flag["detail"],
            )

        logger.info(
            "pm_inputs",
            market_present=bool(state.get("market_report")),
            market_valid=get_artifact_status(state, "market_report").ok,
            sentiment_present=bool(state.get("sentiment_report")),
            sentiment_valid=get_artifact_status(state, "sentiment_report").ok,
            news_present=bool(state.get("news_report")),
            news_valid=get_artifact_status(state, "news_report").ok,
            fundamentals_present=bool(state.get("fundamentals_report")),
            fundamentals_valid=get_artifact_status(state, "fundamentals_report").ok,
            value_trap_present=bool(state.get("value_trap_report")),
            value_trap_valid=get_artifact_status(state, "value_trap_report").ok,
            consultant_valid=get_artifact_status(state, "consultant_review").ok,
            apac_specialist_valid=get_artifact_status(state, "apac_regional_report").ok,
            has_datablock=has_parseable_data_block(fundamentals),
            fund_len=len(fundamentals) if fundamentals else 0,
            value_trap_len=len(value_trap) if value_trap else 0,
            red_flags_count=len(red_flags),
        )

        if not fundamentals or not has_parseable_data_block(fundamentals):
            logger.error(
                "pm_skipped_invalid_fundamentals",
                ticker=ticker,
                fundamentals_valid=get_artifact_status(state, "fundamentals_report").ok,
                has_datablock=has_parseable_data_block(fundamentals),
            )
            return failure_artifact(
                "final_trade_decision",
                "Portfolio Manager skipped due to invalid fundamentals input",
                provider=support.infer_provider_name(llm),
            )

        field_sources = support.extract_field_sources_from_messages(
            state.get("messages", [])
        )
        attribution_table = support.format_attribution_table(field_sources)
        conflict_table = support.format_conflict_table(state.get("messages", []))

        consultant_section = (
            "\n\nEXTERNAL CONSULTANT REVIEW (Cross-Validation):\n"
            f"{consultant if consultant else 'N/A (consultant disabled or unavailable)'}"
        )
        apac = get_valid_artifact_content(state, "apac_regional_report")
        apac_section = (
            "\n\nAPAC REGIONAL SPECIALIST:\n"
            f"{support.summarize_for_pm(apac, 'apac', 1800) if apac else 'N/A'}"
        )

        kill_criteria = support.extract_kill_criteria(support.get_bear_history(state))
        if kill_criteria:
            kill_lines = "\n".join(f"- {trigger}" for trigger in kill_criteria)
            kill_criteria_section = (
                "\n\nBEAR_KILL_CRITERIA (measurable triggers for immediate SELL; "
                "surface these in the investment memo, not PM_BLOCK):\n"
                f"{kill_lines}"
            )
        else:
            kill_criteria_section = ""

        # Scenario valuation section — only emitted when the Valuation
        # Calculator produced a parseable VALUATION_SCENARIOS block AND the
        # fundamentals provide enough data to derive EPS_TTM. The v9.7 PM
        # prompt directs PM to anchor stop-loss to BEAR_IV and reference
        # WEIGHTED_IV; that hint is only useful if PM actually sees the
        # values, which is exactly what this section provides.
        from src.charts.extractors.valuation import (
            extract_valuation_scenarios_for_fundamentals,
            format_iv,
        )

        valuation_params = get_valid_artifact_content(state, "valuation_params")
        scenarios = None
        if valuation_params and fundamentals:
            try:
                scenarios = extract_valuation_scenarios_for_fundamentals(
                    valuation_params, fundamentals
                )
            except Exception as exc:  # pragma: no cover — defense-in-depth
                logger.warning(
                    "pm_scenario_extraction_failed",
                    ticker=ticker,
                    **summarize_exception(exc, operation="pm_scenario_extraction"),
                )
                scenarios = None
        if scenarios is not None:
            current_price = _parse_price_value(
                extract_data_block_field(fundamentals, "CURRENT_PRICE")
            )
            weighted_upside_text = ""
            downside_probability_text = ""
            if current_price and current_price > 0 and scenarios.weighted_iv:
                weighted_upside = (scenarios.weighted_iv / current_price) - 1.0
                downside_probability = sum(
                    scenario.probability
                    for scenario, intrinsic_value in (
                        (scenarios.bear, scenarios.bear_iv),
                        (scenarios.base, scenarios.base_iv),
                        (scenarios.bull, scenarios.bull_iv),
                    )
                    if intrinsic_value < current_price
                )
                weighted_upside_text = (
                    f", implied upside {weighted_upside * 100:.1f}% vs current price "
                    f"{format_iv(current_price)}"
                )
                downside_probability_text = (
                    f", downside probability {downside_probability:.0f}%"
                )
            valuation_section = (
                "\n\nVALUATION SCENARIOS (Python-computed IVs from "
                f"{scenarios.methodology}; sufficiency {scenarios.data_sufficiency}; "
                f"earnings basis {scenarios.earnings_basis}; "
                "anchor stop-loss to BEAR_IV, reference WEIGHTED_IV in rationale):\n"
                f"- BEAR_IV: {format_iv(scenarios.bear_iv)} "
                f"({scenarios.bear.probability:.0f}%) — {scenarios.bear.drivers}\n"
                f"- BASE_IV: {format_iv(scenarios.base_iv)} "
                f"({scenarios.base.probability:.0f}%) — {scenarios.base.drivers}\n"
                f"- BULL_IV: {format_iv(scenarios.bull_iv)} "
                f"({scenarios.bull.probability:.0f}%) — {scenarios.bull.drivers}\n"
                f"- WEIGHTED_IV: {format_iv(scenarios.weighted_iv)}{weighted_upside_text}"
                f"{downside_probability_text}"
            )
            if scenarios.normalization_required and not scenarios.normalized_earnings:
                valuation_section += (
                    "\n- NORMALIZATION WARNING: Earnings normalization was flagged, "
                    "but no lower forward EPS baseline was available; treat "
                    "WEIGHTED_IV as conditional, not normalized fair value."
                )
        else:
            valuation_section = ""

        supplemental_flags_section = format_pm_context_flags(
            fundamentals,
            market,
            news,
            foreign_language,
            inv_plan,
            apac,
            consultant,
        )

        # Surface distrusted valuation inputs to the PM as a zero-penalty (data-quality)
        # warning. Assigned unconditionally so it is in scope for the discipline log below.
        valuation_reliability = (
            extract_data_block_field(fundamentals, "VALUATION_INPUT_RELIABILITY") or ""
        ).upper()
        if valuation_reliability == "QUARANTINED":
            red_flags.append(
                {
                    "type": "VALUATION_INPUT_QUARANTINED",
                    "severity": "WARNING",
                    "detail": (
                        "Forward/trailing valuation multiples were quarantined by "
                        "structured data checks; they cannot independently support a BUY."
                    ),
                    "action": "REVIEW",
                    "risk_penalty": 0.0,
                    "rationale": (
                        "Distrusted valuation inputs — verify before using as BUY support."
                    ),
                }
            )

        red_flag_section, code_risk_subtotal = support.format_red_flag_section(
            pre_screening_result, red_flags
        )

        all_context = f"""MARKET ANALYST REPORT:
{support.summarize_for_pm(market, "market", 2500) if market else "N/A"}

SENTIMENT ANALYST REPORT:
{support.summarize_for_pm(sentiment, "sentiment", 1500) if sentiment else "N/A"}

NEWS ANALYST REPORT:
{support.summarize_for_pm(news, "news", 2000) if news else "N/A"}

FUNDAMENTALS ANALYST REPORT:
{support.summarize_for_pm(fundamentals, "fundamentals", 4000) if fundamentals else "N/A"}{attribution_table}{conflict_table}

FOREIGN LANGUAGE / NATIVE-SOURCE ANALYST REPORT:
{support.summarize_for_pm(foreign_language, "foreign_language", 2500) if foreign_language else "N/A"}

VALUE TRAP ANALYSIS:
{support.extract_value_trap_verdict(value_trap)}{support.summarize_for_pm(value_trap, "value_trap", 2500) if value_trap else "N/A"}{red_flag_section}{macro_section}

RESEARCH MANAGER RECOMMENDATION:
{support.summarize_for_pm(inv_plan, "research", 3000) if inv_plan else "N/A"}{apac_section}{consultant_section}{kill_criteria_section}{valuation_section}{supplemental_flags_section}

TRADER PROPOSAL:
{support.summarize_for_pm(trader, "trader", 2000) if trader else "N/A"}

RISK TEAM DEBATE:
{risk if risk else "N/A"}"""
        pm_system_msg = agent_prompt.system_message
        if strict_mode:
            pm_system_msg += _STRICT_PM_ADDENDUM

        vehicle_directive = ""

        card_obj = governance_card(state)
        if card_obj:
            # PM-specific rule: when entity is non-standard AND a related
            # listed ticker exists, the verdict must address vehicle choice.
            if (
                card_obj.entity_role
                in {"PURE_HOLDCO", "INTERMEDIATE_HOLDCO", "LISTED_SUBSIDIARY"}
                and card_obj.related_listed
            ):
                vehicle_directive = (
                    "\n\nVEHICLE-CHOICE DIRECTIVE: This ticker is a non-standard "
                    "vehicle and a related listed ticker is known. Your verdict "
                    "must explicitly state whether this vehicle (versus its related "
                    "listed counterpart) is the correct one for the investment "
                    "thesis, and quote the basis for that choice."
                )

        prompt = (
            f"{pm_system_msg}{governance_block(state)}{vehicle_directive}\n\n"
            f"{all_context}\n\nMake Portfolio Manager Verdict."
        )

        try:
            response = await agent_runtime.invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_prompt.agent_name,
                provider=support.infer_provider_name(llm),
                model_name=support.get_model_name(llm),
            )
            content_str = message_utils.extract_string_content(response.content)
            content_str = _ensure_consultant_resolution_block(
                content_str,
                consultant if consultant else None,
            )
            content_str = _ensure_apac_resolution_block(
                content_str,
                apac if apac else None,
            )
            content_str = _ensure_auditor_resolution_block(
                content_str,
                state.get("auditor_report") or None,
            )

            from src.utils import detect_truncation

            trunc_info = detect_truncation(content_str, agent="portfolio_manager")
            log_truncation_diagnostic(
                agent_key="portfolio_manager",
                ticker=ticker,
                runnable=llm,
                response=response,
                content=content_str,
                trunc_info=trunc_info,
            )

            validation = validate_required_output("portfolio_manager", content_str)
            log_output_diagnostics(
                agent_key="portfolio_manager",
                ticker=ticker,
                runnable=llm,
                response=response,
                content=content_str,
                truncated=trunc_info["truncated"],
                validation=validation,
            )
            if should_fail_closed(
                "portfolio_manager",
                validation=validation,
                truncated=trunc_info["truncated"],
                content=content_str,
            ):
                logger.error(
                    "portfolio_manager_invalid_structure",
                    ticker=ticker,
                    missing_sections=validation["missing"],
                )
                present_inputs, missing_inputs = _present_pm_inputs(state)
                logger.info(
                    "final_verdict_formed",
                    ticker=ticker,
                    verdict="PARSE_FAILURE",
                    pre_screening_result=state.get("pre_screening_result"),
                    direct_pm_inputs_present=present_inputs,
                    direct_pm_inputs_missing=missing_inputs,
                    missing_sections=validation["missing"],
                    strict_mode=strict_mode,
                )
                return failure_artifact(
                    "final_trade_decision",
                    "Portfolio Manager output missing required structure",
                    provider=support.infer_provider_name(llm),
                    fallback_content=content_str,
                )

            content_str = _normalize_pm_block_contract(content_str)
            _log_risk_tally_reconciliation(content_str, code_risk_subtotal, ticker)
            _log_pm_discipline_checks(
                content_str, red_flags, valuation_reliability, ticker
            )
            present_inputs, missing_inputs = _present_pm_inputs(state)
            pm_metadata = pm_verdict_metadata_from_text(content_str)
            pm_verdict_recovered = False
            if pm_metadata.verdict == "UNPARSEABLE":
                recovered_metadata = await _recover_pm_verdict_metadata(
                    content_str, llm
                )
                if recovered_metadata is not None:
                    pm_metadata = recovered_metadata
                    pm_verdict_recovered = True
            logger.info(
                "final_verdict_formed",
                ticker=ticker,
                verdict=pm_metadata.verdict,
                pm_verdict_recovered=pm_verdict_recovered,
                pm_verdict_metadata=pm_metadata.model_dump(exclude_none=True),
                pre_screening_result=state.get("pre_screening_result"),
                direct_pm_inputs_present=present_inputs,
                direct_pm_inputs_missing=missing_inputs,
                debate_rounds=(state.get("investment_debate_state") or {}).get(
                    "count", 0
                ),
                strict_mode=strict_mode,
            )
            return success_artifact(
                "final_trade_decision",
                cap_state_value(content_str, "final_trade_decision"),
                provider=support.infer_provider_name(llm),
            )
        except Exception as exc:
            logger.error(
                "pm_error",
                ticker=ticker,
                **summarize_exception(exc, operation="portfolio_manager"),
            )
            return failure_artifact(
                "final_trade_decision",
                exc,
                provider=support.infer_provider_name(llm),
            )

    return pm_node


def create_state_cleaner_node() -> Callable:
    async def clean_state(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        context = support.get_context_from_config(config)
        ticker = (
            context.ticker if context else state.get("company_of_interest", "UNKNOWN")
        )

        logger.debug(
            "state_cleaner_running",
            context_ticker=context.ticker if context else None,
            state_ticker=state.get("company_of_interest"),
            final_ticker=ticker,
        )

        return {
            "messages": [HumanMessage(content=f"Analyze {ticker}")],
            "tools_called": state.get("tools_called", {}),
            "company_of_interest": ticker,
        }

    return clean_state


def create_financial_health_validator_node(strict_mode: bool = False) -> Callable:
    """
    Create a pre-screening validator node to catch extreme financial risks.
    """

    async def financial_health_validator_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        from src.config import config as settings_config
        from src.validators.entity_governance_card import (
            build_card,
            extract_merged_subset_from_raw,
        )
        from src.validators.red_flag_detector import RedFlagDetector

        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)

        try:
            fundamentals_report = state.get("fundamentals_report", "")
            if not isinstance(fundamentals_report, str):
                fundamentals_report = message_utils.extract_string_content(
                    fundamentals_report
                )

            quiet_mode = settings_config.quiet_mode

            if not fundamentals_report:
                logger.warning(
                    "validator_no_fundamentals",
                    ticker=ticker,
                    message="No fundamentals report available - skipping pre-screening",
                )
                return {"red_flags": [], "pre_screening_result": "PASS"}

            sector = RedFlagDetector.detect_sector(fundamentals_report)
            metrics = RedFlagDetector.extract_metrics(fundamentals_report)

            has_data_block = has_parseable_data_block(fundamentals_report)
            core_metrics = [
                metrics.get("debt_to_equity"),
                metrics.get("net_income"),
                metrics.get("fcf"),
                metrics.get("adjusted_health_score"),
            ]
            if not has_data_block or all(metric is None for metric in core_metrics):
                logger.warning(
                    "validator_no_usable_metrics",
                    ticker=ticker,
                    has_data_block=has_data_block,
                    message="DATA_BLOCK missing or unparseable - cannot validate financial health",
                )
                return {
                    "red_flags": [
                        {
                            "type": "DATA_QUALITY_WARNING",
                            "severity": "WARNING",
                            "detail": "DATA_BLOCK missing or unparseable in fundamentals report; financial health checks could not be performed",
                            "action": "RISK_PENALTY",
                            "risk_penalty": 1.0,
                            "rationale": "Pre-screening was unable to verify financial health due to missing structured data. Proceeding with caution.",
                        }
                    ],
                    "pre_screening_result": "PASS",
                }

            if not quiet_mode:
                logger.info(
                    "validator_extracted_metrics",
                    ticker=ticker,
                    sector=sector.value,
                    debt_to_equity=metrics.get("debt_to_equity"),
                    fcf=metrics.get("fcf"),
                    net_income=metrics.get("net_income"),
                    interest_coverage=metrics.get("interest_coverage"),
                    adjusted_health_score=metrics.get("adjusted_health_score"),
                )

            card_payload = None
            entity_role = metrics.get("listing_role")
            try:
                merged_subset = extract_merged_subset_from_raw(
                    state.get("raw_fundamentals_data", "")
                )
                card = build_card(
                    ticker=ticker,
                    company_name=company_name,
                    merged_data=merged_subset,
                    senior_metrics=metrics,
                    fla_report=state.get("foreign_language_report", "") or "",
                    value_trap_report=state.get("value_trap_report", "") or "",
                )
                card_payload = card.to_dict()
                entity_role = card.entity_role
            except Exception as card_exc:
                logger.warning(
                    "governance_card_build_failed",
                    ticker=ticker,
                    **summarize_exception(card_exc, operation="entity_governance_card"),
                )

            red_flags, pre_screening_result = RedFlagDetector.detect_red_flags(
                metrics,
                ticker,
                sector,
                strict_mode=strict_mode,
                entity_role=str(entity_role) if entity_role else None,
            )

            legal_report = state.get("legal_report", "")
            if legal_report:
                if not isinstance(legal_report, str):
                    legal_report = message_utils.extract_string_content(legal_report)

                legal_risks = RedFlagDetector.extract_legal_risks(legal_report)
                legal_warnings = RedFlagDetector.detect_legal_flags(legal_risks, ticker)

                if legal_warnings:
                    red_flags.extend(legal_warnings)
                    if not quiet_mode:
                        logger.info(
                            "legal_warnings_detected",
                            ticker=ticker,
                            warning_types=[
                                warning["type"] for warning in legal_warnings
                            ],
                            total_risk_penalty=sum(
                                warning.get("risk_penalty", 0)
                                for warning in legal_warnings
                            ),
                        )

            if strict_mode:
                value_trap_report = state.get("value_trap_report", "")
                if value_trap_report:
                    if not isinstance(value_trap_report, str):
                        value_trap_report = message_utils.extract_string_content(
                            value_trap_report
                        )
                    vt_warnings = RedFlagDetector.detect_value_trap_flags(
                        value_trap_report,
                        ticker,
                        m_and_a_status=extract_data_block_field(
                            fundamentals_report, "M_AND_A_STATUS"
                        ),
                    )
                    if vt_warnings:
                        red_flags.extend(vt_warnings)

            if strict_mode and pre_screening_result == "PASS":
                flag_types = {flag["type"] for flag in red_flags}
                if "PFIC_PROBABLE" in flag_types or "PFIC_UNCERTAIN" in flag_types:
                    pre_screening_result = "REJECT"
                    red_flags.append(
                        {
                            "type": "STRICT_PFIC_ESCALATED",
                            "severity": "CRITICAL",
                            "detail": "PFIC risk escalated to reject in strict mode",
                            "action": "AUTO_REJECT",
                            "rationale": "PFIC tax reporting burden is disqualifying in strict mode",
                        }
                    )
                    logger.info("strict_pfic_escalated_to_reject", ticker=ticker)
                elif "VIE_STRUCTURE" in flag_types:
                    pre_screening_result = "REJECT"
                    red_flags.append(
                        {
                            "type": "STRICT_VIE_ESCALATED",
                            "severity": "CRITICAL",
                            "detail": "VIE structure escalated to reject in strict mode",
                            "action": "AUTO_REJECT",
                            "rationale": "Contractual VIE ownership (not equity) is disqualifying in strict mode",
                        }
                    )
                    logger.info("strict_vie_escalated_to_reject", ticker=ticker)
                elif (
                    "VALUE_TRAP_HIGH_RISK" in flag_types
                    or "VALUE_TRAP_VERDICT" in flag_types
                ):
                    pre_screening_result = "REJECT"
                    red_flags.append(
                        {
                            "type": "STRICT_VALUE_TRAP_ESCALATED",
                            "severity": "CRITICAL",
                            "detail": "High-risk value trap escalated to reject in strict mode",
                            "action": "AUTO_REJECT",
                            "rationale": "Value trap high-risk score is disqualifying in strict mode",
                        }
                    )
                    logger.info("strict_value_trap_escalated_to_reject", ticker=ticker)

            if pre_screening_result == "REJECT":
                logger.info(
                    "pre_screening_rejected",
                    ticker=ticker,
                    company_name=company_name,
                    red_flags_count=len(red_flags),
                    flag_types=[flag["type"] for flag in red_flags],
                )
            elif red_flags:
                logger.info(
                    "pre_screening_warnings",
                    ticker=ticker,
                    warnings_count=len(red_flags),
                )
            elif not quiet_mode:
                logger.info("pre_screening_passed", ticker=ticker)

            result = {
                "red_flags": red_flags,
                "pre_screening_result": pre_screening_result,
            }
            if card_payload is not None:
                result["entity_governance_card"] = card_payload
            return result
        except Exception as exc:
            logger.error(
                "validator_crashed",
                ticker=ticker,
                **summarize_exception(exc, operation="financial_health_validator"),
            )
            return {"red_flags": [], "pre_screening_result": "PASS"}

    return financial_health_validator_node
