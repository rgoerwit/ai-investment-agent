from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from langchain_core.callbacks import BaseCallbackHandler

from src.agents import (
    create_analyst_node,
    create_apac_specialist_node,
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
from src.llm_budgets import get_agent_output_budget
from src.llm_runtime.bindings import BindingPlan, resolve_binding_plan
from src.llm_runtime.capabilities import Capability
from src.llm_runtime.construction import (
    LegacyGraphFactories,
    LegacySeatRequest,
    build_legacy_model,
    build_model_for_seat,
)
from src.llm_runtime.factory import SeatModelFactory
from src.llm_runtime.seats import SEATS, SeatId
from src.runtime_config import get_runtime_config
from src.token_tracker import TokenTrackingCallback, get_tracker
from src.tools.registry import toolkit

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
    ticker_memories: dict[str, Any] | None = None
    cleanup_previous_memories: bool = True
    macro_context_report: str = ""
    macro_context_region: str = ""
    macro_context_status: str = "disabled"
    macro_regime: dict[str, str | bool] = field(default_factory=dict)
    price_snapshot: dict[str, float] | None = None


@dataclass
class GraphComponents:
    """Constructed graph nodes, tool nodes, and flags used by the builder."""

    nodes: dict[str, Any]
    tool_nodes: dict[str, Any]
    consultant_enabled: bool
    auditor_enabled: bool
    apac_specialist_enabled: bool


def create_auditor_llm(*args: Any, **kwargs: Any) -> Any:
    from src.llms import create_auditor_llm as _create_auditor_llm

    return _create_auditor_llm(*args, **kwargs)


def create_apac_specialist_llm(*args: Any, **kwargs: Any) -> Any:
    from src.llms import create_apac_specialist_llm as _create_apac_specialist_llm

    return _create_apac_specialist_llm(*args, **kwargs)


def create_deep_thinking_llm(*args: Any, **kwargs: Any) -> Any:
    from src.llms import create_deep_thinking_llm as _create_deep_thinking_llm

    return _create_deep_thinking_llm(*args, **kwargs)


def create_quick_thinking_llm(*args: Any, **kwargs: Any) -> Any:
    from src.llms import create_quick_thinking_llm as _create_quick_thinking_llm

    return _create_quick_thinking_llm(*args, **kwargs)


def create_apex_llm(*args: Any, **kwargs: Any) -> Any:
    from src.llms import create_apex_llm as _create_apex_llm

    return _create_apex_llm(*args, **kwargs)


def get_consultant_llm(*args: Any, **kwargs: Any) -> Any:
    from src.llms import get_consultant_llm as _get_consultant_llm

    return _get_consultant_llm(*args, **kwargs)


def _build_legacy_seat_model(
    request: LegacySeatRequest,
) -> Any:
    """Inject patchable legacy facades into the canonical dispatcher."""

    return build_legacy_model(
        request,
        graph_factories=LegacyGraphFactories(
            quick=create_quick_thinking_llm,
            deep=create_deep_thinking_llm,
            apex=create_apex_llm,
            consultant=get_consultant_llm,
            auditor=create_auditor_llm,
            apac=create_apac_specialist_llm,
        ),
    )


def build_seat_model(
    seat_id: SeatId,
    *,
    plan: BindingPlan,
    model_factory: SeatModelFactory,
    quick_mode: bool,
    callbacks: Sequence[BaseCallbackHandler],
    output_tokens: int | None,
) -> Any:
    """Build one fresh model from a canonical seat and resolved binding."""

    return build_model_for_seat(
        seat_id,
        plan=plan,
        factory=model_factory,
        quick_mode=quick_mode,
        callbacks=list(callbacks),
        output_tokens=output_tokens,
        # The quick-mode APEX standard-tier pin is seat data
        # (``standard_tier_in_quick_mode``), so every caller inherits it — not
        # only the graph.
        legacy_builder=_build_legacy_seat_model,
    )


def _create_legacy_memories() -> tuple[Any, Any, Any, Any, Any]:
    from src.memory import FinancialSituationMemory

    return (
        FinancialSituationMemory("legacy_bull_memory"),
        FinancialSituationMemory("legacy_bear_memory"),
        FinancialSituationMemory("legacy_invest_judge_memory"),
        FinancialSituationMemory("legacy_trader_memory"),
        FinancialSituationMemory("legacy_risk_manager_memory"),
    )


def cleanup_all_memories(*args: Any, **kwargs: Any) -> Any:
    from src.memory import cleanup_all_memories as _cleanup_all_memories

    return _cleanup_all_memories(*args, **kwargs)


def create_memory_instances(*args: Any, **kwargs: Any) -> Any:
    from src.memory import create_memory_instances as _create_memory_instances

    return _create_memory_instances(*args, **kwargs)


def sanitize_ticker_for_collection(*args: Any, **kwargs: Any) -> Any:
    from src.memory import (
        sanitize_ticker_for_collection as _sanitize_ticker_for_collection,
    )

    return _sanitize_ticker_for_collection(*args, **kwargs)


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
    binding_plan: BindingPlan | None = None,
    model_factory: SeatModelFactory | None = None,
) -> GraphComponents:
    runtime_config = get_runtime_config(config)
    """Build graph memories, LLMs, nodes, and agent-specific tool nodes."""
    if ticker and enable_memory:
        if cleanup_previous:
            logger.debug("cleaning_previous_memories", ticker=ticker)
            cleanup_all_memories(days=0, ticker=ticker)

        logger.debug("creating_ticker_memories", ticker=ticker)
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

        logger.debug(
            "ticker_memories_ready",
            ticker=ticker,
            bull_available=bull_memory.available,
            bear_available=bear_memory.available,
        )
    else:
        if enable_memory:
            logger.warning("using_legacy_memories_no_ticker")
        else:
            logger.debug("memory_disabled_using_legacy_memories", ticker=ticker)

        (
            bull_memory,
            bear_memory,
            invest_judge_memory,
            trader_memory,
            risk_manager_memory,
        ) = _create_legacy_memories()

    logger.debug(
        "creating_trading_graph",
        ticker=ticker,
        max_debate_rounds=max_debate_rounds,
        enable_memory=enable_memory,
        architecture="parallel",
    )

    tracker = get_tracker()
    base_output_tokens = config.llm_base_output_tokens
    plan = binding_plan or resolve_binding_plan(config)
    factory = model_factory or SeatModelFactory()

    def output_budget(agent_name: str) -> int:
        return get_agent_output_budget(agent_name, base_output_tokens)

    def tracked_callbacks(agent_name: str) -> list[TokenTrackingCallback]:
        return [
            TokenTrackingCallback(
                agent_name,
                tracker,
                output_token_cap=output_budget(agent_name),
            )
        ]

    def seat_model(seat_id: SeatId, *, tracked: bool = True) -> Any:
        spec = SEATS[seat_id]
        budget = output_budget(spec.budget_key) if spec.budget_key else None
        callbacks = tracked_callbacks(spec.callback_name) if tracked else []
        return build_seat_model(
            seat_id,
            plan=plan,
            model_factory=factory,
            quick_mode=quick_mode,
            callbacks=callbacks,
            output_tokens=budget,
        )

    market_llm = seat_model(SeatId.MARKET)
    social_llm = seat_model(SeatId.SENTIMENT)
    news_llm = seat_model(SeatId.NEWS)
    junior_fund_llm = seat_model(SeatId.JUNIOR_FUNDAMENTALS)
    # Senior Fundamentals and the PM are the two gate-critical (APEX) seats:
    # the largest, most rule-dense prompts, whose outputs feed the hard <50%
    # gates and the verdict contract. Both route through create_apex_llm —
    # APEX_MODEL pins them in full mode; in --quick they drop to
    # APEX_QUICK_MODEL (or the plain quick floor) so screening stays cheap.
    senior_fund_llm = seat_model(SeatId.SENIOR_FUNDAMENTALS)
    pm_llm = seat_model(SeatId.PORTFOLIO_MANAGER)

    retry_llm = None
    allow_retry = False
    if not quick_mode:
        retry_binding = plan.bindings[SeatId.ANALYST_RETRY]
        if plan.schema == "legacy":
            # The compatibility window promises byte-for-byte legacy behavior:
            # older Gemini quick floors did not opt into RETRY-HIGH.
            from src.llms import is_gemini_v3_or_greater

            allow_retry = is_gemini_v3_or_greater(runtime_config.quick_think_llm)
        else:
            allow_retry = Capability.TOOL_CALLING in retry_binding.profile.capabilities
        # No bound token-tracking callback: this deep-model instance is shared
        # across every analyst's retry path, so its cost is attributed to the
        # ORIGINATING agent at the retry call site (analyst_nodes) via a per-call
        # callback — not pooled into a synthetic "Retry Agent (Deep)" bucket.
        if allow_retry:
            retry_llm = seat_model(SeatId.ANALYST_RETRY, tracked=False)
            logger.debug("retry_llm_enabled", ticker=ticker)
        else:
            logger.warning(
                "retry_llm_disabled_binding_capability",
                ticker=ticker,
                provider=retry_binding.provider,
                model=retry_binding.model,
                reason="retry binding lacks the reviewed tool-calling policy",
            )
    elif quick_mode:
        logger.debug("retry_llm_disabled_quick_mode", ticker=ticker)

    logger.debug(
        "synthesis_llm_mode",
        quick_mode=quick_mode,
        reasoning_intent="fast" if quick_mode else "reasoning",
    )
    bull_llm = seat_model(SeatId.BULL)
    bear_llm = seat_model(SeatId.BEAR)
    res_mgr_llm = seat_model(SeatId.RESEARCH_MANAGER)
    risky_llm = seat_model(SeatId.RISKY)
    safe_llm = seat_model(SeatId.SAFE)
    neutral_llm = seat_model(SeatId.NEUTRAL)
    trader_llm = seat_model(SeatId.TRADER)
    valuation_llm = seat_model(SeatId.VALUATION)

    consultant_output_budget = output_budget("Consultant")
    if quick_mode:
        consultant_output_budget = min(
            consultant_output_budget,
            int(config.consultant_quick_max_completion_tokens),
        )
    consultant_requested = (
        plan.statuses[SeatId.CONSULTANT].enabled
        if plan.schema == "new"
        else config.enable_consultant
    )
    consultant_llm = (
        build_seat_model(
            SeatId.CONSULTANT,
            plan=plan,
            model_factory=factory,
            quick_mode=quick_mode,
            callbacks=tracked_callbacks("Consultant"),
            output_tokens=consultant_output_budget,
        )
        if consultant_requested
        else None
    )

    auditor_requested = (
        plan.statuses[SeatId.AUDITOR].enabled
        if plan.schema == "new"
        else _is_auditor_enabled()
    )
    auditor_llm = (
        build_seat_model(
            SeatId.AUDITOR,
            plan=plan,
            model_factory=factory,
            quick_mode=quick_mode,
            callbacks=tracked_callbacks("Global Forensic Auditor"),
            output_tokens=output_budget("Global Forensic Auditor"),
        )
        if auditor_requested
        else None
    )
    if auditor_requested and auditor_llm is None:
        raise RuntimeError(
            "Auditor routing was enabled, but auditor LLM creation returned None."
        )

    auditor_escalation_llm = None
    escalation_differs = (
        plan.bindings[SeatId.AUDITOR_ESCALATION].model
        != plan.bindings[SeatId.AUDITOR].model
        if plan.schema == "new"
        else bool(
            config.auditor_escalation_model
            and config.auditor_escalation_model != config.auditor_model
        )
    )
    if auditor_llm is not None and not quick_mode and escalation_differs:
        auditor_escalation_llm = build_seat_model(
            SeatId.AUDITOR_ESCALATION,
            plan=plan,
            model_factory=factory,
            quick_mode=False,
            callbacks=tracked_callbacks("Global Forensic Auditor Escalation"),
            output_tokens=output_budget("Global Forensic Auditor"),
        )

    consultant_enabled = consultant_llm is not None
    auditor_enabled = auditor_llm is not None
    apac_requested = (
        plan.status_for(SeatId.APAC, quick_mode=quick_mode).enabled
        if plan.schema == "new"
        else not quick_mode
    )
    apac_specialist_llm = (
        build_seat_model(
            SeatId.APAC,
            plan=plan,
            model_factory=factory,
            quick_mode=quick_mode,
            callbacks=tracked_callbacks("APAC Regional Specialist"),
            output_tokens=output_budget("APAC Regional Specialist"),
        )
        if apac_requested
        else None
    )
    apac_specialist_enabled = apac_specialist_llm is not None
    apac_specialist_fallback_llm = (
        build_seat_model(
            SeatId.APAC_DIRECT_RETRY,
            plan=plan,
            model_factory=factory,
            quick_mode=quick_mode,
            callbacks=tracked_callbacks("APAC Regional Specialist Direct Retry"),
            output_tokens=output_budget("APAC Regional Specialist"),
        )
        if apac_specialist_enabled
        else None
    )

    logger.debug(
        "graph_llm_plan",
        quick_mode=quick_mode,
        quick_model_name=runtime_config.quick_think_llm,
        deep_model_name=runtime_config.deep_think_llm,
        retry_llm_enabled=allow_retry,
        consultant_enabled=consultant_enabled,
        auditor_enabled=auditor_enabled,
        apac_specialist_enabled=apac_specialist_enabled,
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

    foreign_llm = seat_model(SeatId.FOREIGN_LANGUAGE)
    foreign_analyst = create_analyst_node(
        foreign_llm,
        "foreign_language_analyst",
        toolkit.get_foreign_language_tools(),
        "foreign_language_report",
        retry_llm=retry_llm,
        allow_retry=allow_retry,
    )

    legal_llm = seat_model(SeatId.LEGAL_COUNSEL)
    legal_counsel = create_legal_counsel_node(legal_llm, toolkit.get_legal_tools())

    value_trap_llm = seat_model(SeatId.VALUE_TRAP)
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
        auditor_tool_list = toolkit.get_auditor_tools()
        auditor = create_auditor_node(
            auditor_llm,
            auditor_tool_list,
            escalation_llm=auditor_escalation_llm,
        )
        auditor_tools = create_agent_tool_node(
            auditor_tool_list, "global_forensic_auditor"
        )
        logger.debug("auditor_node_enabled", ticker=ticker)

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
            consultant_llm,
            "consultant",
            tools=consultant_tools,
            quick_mode=quick_mode,
        )
        logger.debug("consultant_node_enabled", ticker=ticker)
    else:
        logger.debug("consultant_node_disabled", ticker=ticker)

    apac_specialist = None
    if apac_specialist_enabled:
        apac_specialist = create_apac_specialist_node(
            apac_specialist_llm,
            fallback_llm=apac_specialist_fallback_llm,
        )
        logger.debug("apac_specialist_node_enabled", ticker=ticker)
    else:
        logger.debug("apac_specialist_node_disabled", ticker=ticker)

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

    if apac_specialist_enabled and apac_specialist is not None:
        nodes["APAC Regional Specialist"] = apac_specialist

    return GraphComponents(
        nodes=nodes,
        tool_nodes=tool_nodes,
        consultant_enabled=consultant_enabled,
        auditor_enabled=auditor_enabled,
        apac_specialist_enabled=apac_specialist_enabled,
    )
