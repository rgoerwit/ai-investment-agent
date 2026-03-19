from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from src.agents import (
    create_analyst_node,
    create_auditor_node,
    create_consultant_node,
    create_financial_health_validator_node,
    create_legal_counsel_node,
    create_portfolio_manager_node,
    create_research_manager_node,
    create_researcher_node,
    create_risk_debater_node,
    create_trader_node,
    create_valuation_calculator_node,
)
from src.charts.chart_node import create_chart_generator_node
from src.config import config
from src.llms import (
    create_auditor_llm,
    create_deep_thinking_llm,
    create_quick_thinking_llm,
    get_consultant_llm,
    is_gemini_v3_or_greater,
)
from src.memory import (
    FinancialSituationMemory,
    cleanup_all_memories,
    create_memory_instances,
    sanitize_ticker_for_collection,
)
from src.token_tracker import TokenTrackingCallback, get_tracker
from src.toolkit import toolkit

from .routing import _is_auditor_enabled
from .tool_nodes import create_agent_tool_node

logger = structlog.get_logger(__name__)


@dataclass
class TradingContext:
    """Context object passed to graph nodes via configuration."""

    ticker: str
    trade_date: str
    quick_mode: bool = False
    enable_memory: bool = True
    max_debate_rounds: int = 2
    max_risk_rounds: int = 1
    ticker_memories: dict[str, Any] | None = None
    cleanup_previous_memories: bool = True


@dataclass
class GraphComponents:
    """Constructed graph nodes, tool nodes, and flags used by the builder."""

    nodes: dict[str, Any]
    tool_nodes: dict[str, Any]
    consultant_enabled: bool
    auditor_enabled: bool


def _create_legacy_memories() -> (
    tuple[
        FinancialSituationMemory,
        FinancialSituationMemory,
        FinancialSituationMemory,
        FinancialSituationMemory,
        FinancialSituationMemory,
    ]
):
    return (
        FinancialSituationMemory("legacy_bull_memory"),
        FinancialSituationMemory("legacy_bear_memory"),
        FinancialSituationMemory("legacy_invest_judge_memory"),
        FinancialSituationMemory("legacy_trader_memory"),
        FinancialSituationMemory("legacy_risk_manager_memory"),
    )


def build_graph_components(
    *,
    max_debate_rounds: int,
    enable_memory: bool,
    ticker: str | None,
    cleanup_previous: bool,
    quick_mode: bool,
    strict_mode: bool,
    chart_format: str,
    transparent_charts: bool,
    image_dir: Path | None,
    skip_charts: bool,
) -> GraphComponents:
    """Build graph memories, LLMs, nodes, and agent-specific tool nodes."""
    if ticker and enable_memory:
        if cleanup_previous:
            logger.info("cleaning_previous_memories", ticker=ticker)
            cleanup_all_memories(days=0, ticker=ticker)

        logger.info("creating_ticker_memories", ticker=ticker)
        memories = create_memory_instances(ticker)

        safe_ticker = sanitize_ticker_for_collection(ticker)
        bull_memory = memories.get(f"{safe_ticker}_bull_memory")
        bear_memory = memories.get(f"{safe_ticker}_bear_memory")
        invest_judge_memory = memories.get(f"{safe_ticker}_invest_judge_memory")
        trader_memory = memories.get(f"{safe_ticker}_trader_memory")
        risk_manager_memory = memories.get(f"{safe_ticker}_risk_manager_memory")

        all_memories = [
            bull_memory,
            bear_memory,
            invest_judge_memory,
            trader_memory,
            risk_manager_memory,
        ]
        if not all(all_memories):
            missing = []
            if not bull_memory:
                missing.append("bull_memory")
            if not bear_memory:
                missing.append("bear_memory")
            if not invest_judge_memory:
                missing.append("invest_judge_memory")
            if not trader_memory:
                missing.append("trader_memory")
            if not risk_manager_memory:
                missing.append("risk_manager_memory")
            raise ValueError(
                f"Failed to create memory instances for {ticker}. Missing: {', '.join(missing)}"
            )

        logger.info(
            "ticker_memories_ready",
            ticker=ticker,
            bull_available=bull_memory.available,
            bear_available=bear_memory.available,
        )
    else:
        if enable_memory:
            logger.warning("using_legacy_memories_no_ticker")
        else:
            logger.info("memory_disabled_using_legacy_memories", ticker=ticker)

        (
            bull_memory,
            bear_memory,
            invest_judge_memory,
            trader_memory,
            risk_manager_memory,
        ) = _create_legacy_memories()

    logger.info(
        "creating_trading_graph",
        ticker=ticker,
        max_debate_rounds=max_debate_rounds,
        enable_memory=enable_memory,
        architecture="parallel",
    )

    tracker = get_tracker()

    market_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Market Analyst", tracker)]
    )
    social_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Sentiment Analyst", tracker)]
    )
    news_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("News Analyst", tracker)]
    )
    junior_fund_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Junior Fundamentals Analyst", tracker)]
    )
    # Senior Fundamentals stays on the quick model even in normal mode.
    # This node does structured synthesis over large upstream inputs; using the
    # deep/thinking-heavy model here has historically increased timeout risk
    # without improving downstream scoring quality enough to justify it.
    senior_fund_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Fundamentals Analyst", tracker)]
    )

    retry_llm = None
    allow_retry = False
    if not quick_mode and is_gemini_v3_or_greater(config.quick_think_llm):
        retry_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Retry Agent (Deep)", tracker)]
        )
        allow_retry = True
        logger.info("retry_llm_enabled", ticker=ticker)
    elif quick_mode:
        logger.info("retry_llm_disabled_quick_mode", ticker=ticker)

    if quick_mode:
        logger.info("synthesis_llm_mode", quick_mode=quick_mode, thinking_level="low")
        bull_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Bull Researcher", tracker)]
        )
        bear_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Bear Researcher", tracker)]
        )
        res_mgr_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Research Manager", tracker)]
        )
        pm_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Portfolio Manager", tracker)]
        )
        risky_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Risky Analyst", tracker)]
        )
        safe_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Safe Analyst", tracker)]
        )
        neutral_llm = create_quick_thinking_llm(
            callbacks=[TokenTrackingCallback("Neutral Analyst", tracker)]
        )
    else:
        logger.info("synthesis_llm_mode", quick_mode=quick_mode, thinking_level="high")
        bull_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Bull Researcher", tracker)]
        )
        bear_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Bear Researcher", tracker)]
        )
        res_mgr_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Research Manager", tracker)]
        )
        pm_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Portfolio Manager", tracker)]
        )
        risky_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Risky Analyst", tracker)]
        )
        safe_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Safe Analyst", tracker)]
        )
        neutral_llm = create_deep_thinking_llm(
            callbacks=[TokenTrackingCallback("Neutral Analyst", tracker)]
        )

    trader_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Trader", tracker)]
    )
    valuation_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Valuation Calculator", tracker)]
    )

    consultant_llm = get_consultant_llm(
        callbacks=[TokenTrackingCallback("Consultant", tracker)],
        quick_mode=quick_mode,
    )

    auditor_requested = _is_auditor_enabled()
    auditor_llm = (
        create_auditor_llm(
            callbacks=[TokenTrackingCallback("Global Forensic Auditor", tracker)]
        )
        if auditor_requested
        else None
    )
    if auditor_requested and auditor_llm is None:
        raise RuntimeError(
            "Auditor routing was enabled, but auditor LLM creation returned None."
        )

    consultant_enabled = consultant_llm is not None
    auditor_enabled = auditor_llm is not None

    logger.info(
        "graph_llm_plan",
        quick_mode=quick_mode,
        quick_model_name=config.quick_think_llm,
        deep_model_name=config.deep_think_llm,
        retry_llm_enabled=allow_retry,
        consultant_enabled=consultant_enabled,
        auditor_enabled=auditor_enabled,
    )

    market = create_analyst_node(
        market_llm,
        "market_analyst",
        toolkit.get_technical_tools(),
        "market_report",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )
    sentiment = create_analyst_node(
        social_llm,
        "sentiment_analyst",
        toolkit.get_sentiment_tools(),
        "sentiment_report",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )
    news = create_analyst_node(
        news_llm,
        "news_analyst",
        toolkit.get_news_tools(),
        "news_report",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )

    foreign_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Foreign Language Analyst", tracker)]
    )
    foreign_analyst = create_analyst_node(
        foreign_llm,
        "foreign_language_analyst",
        toolkit.get_foreign_language_tools(),
        "foreign_language_report",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )

    legal_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Legal Counsel", tracker)]
    )
    legal_counsel = create_legal_counsel_node(legal_llm, toolkit.get_legal_tools())

    value_trap_llm = create_quick_thinking_llm(
        callbacks=[TokenTrackingCallback("Value Trap Detector", tracker)]
    )
    value_trap_detector = create_analyst_node(
        value_trap_llm,
        "value_trap_detector",
        toolkit.get_value_trap_tools(),
        "value_trap_report",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )

    auditor = None
    auditor_tools = None
    if auditor_enabled:
        auditor_tool_list = (
            toolkit.get_foreign_language_tools()
            + toolkit.get_junior_fundamental_tools()
            + toolkit.get_news_tools()
        )
        auditor = create_auditor_node(auditor_llm, auditor_tool_list)
        auditor_tools = create_agent_tool_node(
            auditor_tool_list, "global_forensic_auditor"
        )
        logger.info("auditor_node_enabled", ticker=ticker)

    junior_fund = create_analyst_node(
        junior_fund_llm,
        "junior_fundamentals_analyst",
        toolkit.get_junior_fundamental_tools(),
        "raw_fundamentals_data",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )
    senior_fund = create_analyst_node(
        senior_fund_llm,
        "fundamentals_analyst",
        toolkit.get_senior_fundamental_tools(),
        "fundamentals_report",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )
    validator = create_financial_health_validator_node(strict_mode=strict_mode)

    market_tools = create_agent_tool_node(toolkit.get_market_tools(), "market_analyst")
    sentiment_tools = create_agent_tool_node(
        toolkit.get_sentiment_tools(), "sentiment_analyst"
    )
    news_tools = create_agent_tool_node(toolkit.get_news_tools(), "news_analyst")
    junior_fund_tools = create_agent_tool_node(
        toolkit.get_junior_fundamental_tools(), "junior_fundamentals_analyst"
    )
    foreign_tools = create_agent_tool_node(
        toolkit.get_foreign_language_tools(), "foreign_language_analyst"
    )
    legal_tools = create_agent_tool_node(toolkit.get_legal_tools(), "legal_counsel")
    value_trap_tools = create_agent_tool_node(
        toolkit.get_value_trap_tools(), "value_trap_detector"
    )

    bull_r1 = create_researcher_node(
        bull_llm, bull_memory, "bull_researcher", round_num=1
    )
    bear_r1 = create_researcher_node(
        bear_llm, bear_memory, "bear_researcher", round_num=1
    )
    bull_r2 = create_researcher_node(
        bull_llm, bull_memory, "bull_researcher", round_num=2
    )
    bear_r2 = create_researcher_node(
        bear_llm, bear_memory, "bear_researcher", round_num=2
    )
    res_mgr = create_research_manager_node(
        res_mgr_llm, invest_judge_memory, strict_mode=strict_mode
    )
    trader = create_trader_node(trader_llm, trader_memory)
    risky = create_risk_debater_node(risky_llm, "risky_analyst")
    safe = create_risk_debater_node(safe_llm, "safe_analyst")
    neutral = create_risk_debater_node(neutral_llm, "neutral_analyst")
    pm = create_portfolio_manager_node(
        pm_llm, risk_manager_memory, strict_mode=strict_mode
    )
    pm_fast_fail = create_portfolio_manager_node(
        pm_llm, risk_manager_memory, strict_mode=strict_mode
    )

    consultant = None
    if consultant_enabled:
        from src.consultant_tools import get_consultant_tools

        consultant_tools = get_consultant_tools()
        consultant = create_consultant_node(
            consultant_llm, "consultant", tools=consultant_tools
        )
        logger.info("consultant_node_enabled", ticker=ticker)
    else:
        logger.info("consultant_node_disabled", ticker=ticker)

    valuation_calc = create_valuation_calculator_node(valuation_llm)
    chart_generator = create_chart_generator_node(
        chart_format=chart_format,
        transparent=transparent_charts,
        image_dir=image_dir,
        skip_charts=skip_charts or quick_mode,
    )

    nodes: dict[str, Any] = {
        "Market Analyst": market,
        "Sentiment Analyst": sentiment,
        "News Analyst": news,
        "Junior Fundamentals Analyst": junior_fund,
        "Foreign Language Analyst": foreign_analyst,
        "Legal Counsel": legal_counsel,
        "Value Trap Detector": value_trap_detector,
        "Fundamentals Analyst": senior_fund,
        "Financial Validator": validator,
        "Bull Researcher R1": bull_r1,
        "Bear Researcher R1": bear_r1,
        "Bull Researcher R2": bull_r2,
        "Bear Researcher R2": bear_r2,
        "Research Manager": res_mgr,
        "Valuation Calculator": valuation_calc,
        "Trader": trader,
        "Risky Analyst": risky,
        "Safe Analyst": safe,
        "Neutral Analyst": neutral,
        "Portfolio Manager": pm,
        "PM Fast-Fail": pm_fast_fail,
        "Chart Generator": chart_generator,
    }
    tool_nodes = {
        "market_tools": market_tools,
        "sentiment_tools": sentiment_tools,
        "news_tools": news_tools,
        "junior_fund_tools": junior_fund_tools,
        "foreign_tools": foreign_tools,
        "legal_tools": legal_tools,
        "value_trap_tools": value_trap_tools,
    }

    if auditor_enabled and auditor is not None and auditor_tools is not None:
        nodes["Auditor"] = auditor
        tool_nodes["auditor_tools"] = auditor_tools

    if consultant_enabled and consultant is not None:
        nodes["Consultant"] = consultant

    return GraphComponents(
        nodes=nodes,
        tool_nodes=tool_nodes,
        consultant_enabled=consultant_enabled,
        auditor_enabled=auditor_enabled,
    )
