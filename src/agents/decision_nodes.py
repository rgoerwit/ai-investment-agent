from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog
from langchain_core.messages import HumanMessage
from langgraph.types import RunnableConfig

from . import message_utils, support
from . import runtime as agent_runtime
from .state import AgentState

logger = structlog.get_logger(__name__)

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
- Liquidity ≥ $750k daily USD (vs $500k normal)
- Graham Earnings Test: PASS
- Risk tally < 1.5

### POSITION SIZING:
- Maximum: 5% (vs 10% normal)
- Authoritarian regimes: MAX 1.5% (vs 2%)
- Any remaining uncertainty: MAX 2%
"""


def create_trader_node(llm, memory: Any | None) -> Callable:
    async def trader_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("trader")
        if not agent_prompt:
            return {"trader_investment_plan": "Error: Missing prompt"}

        consultant = state.get("consultant_review", "")
        consultant_section = (
            "\n\nEXTERNAL CONSULTANT REVIEW (Cross-Validation):\n"
            f"{consultant if consultant else 'N/A (consultant disabled or unavailable)'}"
        )
        valuation = state.get("valuation_params", "")
        valuation_section = (
            f"\n\nVALUATION PARAMETERS:\n{valuation}" if valuation else ""
        )

        all_input = f"""MARKET ANALYST REPORT:
{state.get("market_report", "N/A")}

SENTIMENT ANALYST REPORT:
{state.get("sentiment_report", "N/A")}

NEWS ANALYST REPORT:
{state.get("news_report", "N/A")}

FUNDAMENTALS ANALYST REPORT:
{state.get("fundamentals_report", "N/A")}

RESEARCH MANAGER PLAN:
{state.get("investment_plan", "N/A")}{consultant_section}{valuation_section}"""
        prompt = f"{agent_prompt.system_message}\n\n{all_input}\n\nCreate Trading Plan."

        try:
            response = await agent_runtime.invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_prompt.agent_name,
            )
            return {
                "trader_investment_plan": message_utils.extract_string_content(
                    response.content
                )
            }
        except Exception as exc:
            return {"trader_investment_plan": f"Error: {str(exc)}"}

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

        consultant = state.get("consultant_review", "")
        consultant_section = (
            "\n\nEXTERNAL CONSULTANT REVIEW (Cross-Validation):\n"
            f"{consultant if consultant else 'N/A (consultant disabled or unavailable)'}"
        )

        prompt = (
            f"{agent_prompt.system_message}\n\nTRADER PLAN: "
            f"{state.get('trader_investment_plan')}{consultant_section}\n\n"
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
            return {
                "risk_debate_state": {
                    field_name: f"[ERROR]: {agent_key} failed - {str(exc)}",
                    "latest_speaker": agent_key,
                }
            }

    return risk_node


def create_portfolio_manager_node(
    llm, memory: Any | None, strict_mode: bool = False
) -> Callable:
    async def pm_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from src.prompts import get_prompt
        from src.validators.red_flag_detector import RedFlagDetector

        agent_prompt = get_prompt("portfolio_manager")
        if not agent_prompt:
            return {"final_trade_decision": "Error: Missing prompt"}

        market = state.get("market_report", "")
        sentiment = state.get("sentiment_report", "")
        news = state.get("news_report", "")
        fundamentals = state.get("fundamentals_report", "")
        value_trap = state.get("value_trap_report", "")
        inv_plan = state.get("investment_plan", "")
        consultant = state.get("consultant_review", "")
        trader = state.get("trader_investment_plan", "")

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

        if value_trap:
            value_trap_warnings = RedFlagDetector.detect_value_trap_flags(
                value_trap, ticker
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
                fundamentals, ticker
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

        consultant_review = state.get("consultant_review", "")
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

        logger.info(
            "pm_inputs",
            has_market=bool(market),
            has_sentiment=bool(sentiment),
            has_news=bool(news),
            has_fundamentals=bool(fundamentals),
            has_value_trap=bool(value_trap),
            has_consultant=bool(consultant),
            has_datablock="DATA_BLOCK" in fundamentals if fundamentals else False,
            fund_len=len(fundamentals) if fundamentals else 0,
            value_trap_len=len(value_trap) if value_trap else 0,
            red_flags_count=len(red_flags),
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

        red_flag_section = (
            "\n\nRED-FLAG PRE-SCREENING:\n"
            f"Pre-Screening Result: {pre_screening_result}"
        )
        if red_flags:
            red_flag_list = "\n".join(
                [
                    f"  - {flag.get('type', 'Unknown')}: {flag.get('detail', 'No detail')}"
                    for flag in red_flags
                ]
            )
            red_flag_section += f"\nRed Flags/Warnings Detected:\n{red_flag_list}"
        else:
            red_flag_section += "\nRed Flags Detected: None"

        all_context = f"""MARKET ANALYST REPORT:
{support.summarize_for_pm(market, "market", 2500) if market else "N/A"}

SENTIMENT ANALYST REPORT:
{support.summarize_for_pm(sentiment, "sentiment", 1500) if sentiment else "N/A"}

NEWS ANALYST REPORT:
{support.summarize_for_pm(news, "news", 2000) if news else "N/A"}

FUNDAMENTALS ANALYST REPORT:
{support.summarize_for_pm(fundamentals, "fundamentals", 6000) if fundamentals else "N/A"}{attribution_table}{conflict_table}

VALUE TRAP ANALYSIS:
{support.extract_value_trap_verdict(value_trap)}{support.summarize_for_pm(value_trap, "value_trap", 2500) if value_trap else "N/A"}{red_flag_section}

RESEARCH MANAGER RECOMMENDATION:
{support.summarize_for_pm(inv_plan, "research", 3000) if inv_plan else "N/A"}{consultant_section}

TRADER PROPOSAL:
{support.summarize_for_pm(trader, "trader", 2000) if trader else "N/A"}

RISK TEAM DEBATE:
{risk if risk else "N/A"}"""
        pm_system_msg = agent_prompt.system_message
        if strict_mode:
            pm_system_msg += _STRICT_PM_ADDENDUM
        prompt = f"{pm_system_msg}\n\n{all_context}\n\nMake Portfolio Manager Verdict."

        try:
            response = await agent_runtime.invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=agent_prompt.agent_name,
            )
            content_str = message_utils.extract_string_content(response.content)

            from src.utils import detect_truncation

            trunc_info = detect_truncation(content_str, agent="portfolio_manager")
            if trunc_info["truncated"]:
                logger.warning(
                    "agent_output_truncated",
                    agent="portfolio_manager",
                    ticker=ticker,
                    source=trunc_info["source"],
                    marker=trunc_info["marker"],
                    confidence=trunc_info["confidence"],
                    output_len=len(content_str),
                )

            return {"final_trade_decision": content_str}
        except Exception as exc:
            logger.error("pm_error", error=str(exc))
            return {"final_trade_decision": f"Error: {str(exc)}"}

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

            has_data_block = "### --- START DATA_BLOCK" in fundamentals_report
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

            red_flags, pre_screening_result = RedFlagDetector.detect_red_flags(
                metrics, ticker, sector, strict_mode=strict_mode
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
                        value_trap_report, ticker
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

            return {
                "red_flags": red_flags,
                "pre_screening_result": pre_screening_result,
            }
        except Exception as exc:
            logger.error(
                "validator_crashed",
                ticker=ticker,
                error=str(exc),
                message="Validator failed - defaulting to PASS to avoid blocking graph",
            )
            return {"red_flags": [], "pre_screening_result": "PASS"}

    return financial_health_validator_node
