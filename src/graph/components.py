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
    create_screen_rejection_node,
    create_trader_node,
    create_valuation_calculator_node,
)
from src.agents.debate_handoffs import DebateReasoningPolicy
from src.charts.chart_node import create_chart_generator_node
from src.config import config
from src.forensic_budget import graph_research_budget_policies
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

from .tool_nodes import create_agent_tool_node

logger = structlog.get_logger(__name__)


def _is_auditor_enabled(plan: BindingPlan, *, quick_mode: bool) -> bool:
    """Return the binding plan's effective Auditor availability for this graph."""

    return plan.status_for(SeatId.AUDITOR, quick_mode=quick_mode).enabled


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
    debate_reasoning_policy: DebateReasoningPolicy


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
    include_reasoning_output: bool = False,
) -> Any:
    """Build one fresh model from a canonical seat and resolved binding."""

    return build_model_for_seat(
        seat_id,
        plan=plan,
        factory=model_factory,
        quick_mode=quick_mode,
        callbacks=list(callbacks),
        output_tokens=output_tokens,
        include_reasoning_output=include_reasoning_output,
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
    debate_reasoning_policy = DebateReasoningPolicy(
        enabled=bool(getattr(runtime_config, "debate_reasoning_handoffs", False)),
        max_rounds=max_debate_rounds,
    )
    plan = binding_plan or resolve_binding_plan(config)
    factory = model_factory or SeatModelFactory()
    research_policies = graph_research_budget_policies(quick_mode=quick_mode)

    def output_budget(agent_name: str) -> int:
        return get_agent_output_budget(
            agent_name, base_output_tokens
        ) + debate_reasoning_policy.output_bonus(agent_name)

    def tracked_callbacks(
        agent_name: str,
        *,
        output_token_cap: int | None = None,
        originating_seat_id: SeatId | None = None,
    ) -> list[TokenTrackingCallback]:
        return [
            TokenTrackingCallback(
                agent_name,
                tracker,
                output_token_cap=(
                    output_budget(agent_name)
                    if output_token_cap is None
                    else output_token_cap
                ),
                originating_seat_id=(
                    originating_seat_id.value if originating_seat_id else None
                ),
            )
        ]

    def seat_model(
        seat_id: SeatId,
        *,
        tracked: bool = True,
        include_reasoning_output: bool = False,
        use_quick_binding: bool = False,
        output_tokens_override: int | None = None,
        tracking_agent_name: str | None = None,
        originating_seat_id: SeatId | None = None,
    ) -> Any:
        spec = SEATS[seat_id]
        budget = (
            output_tokens_override
            if output_tokens_override is not None
            else output_budget(spec.budget_key)
            if spec.budget_key
            else None
        )
        callbacks = (
            tracked_callbacks(
                tracking_agent_name or spec.callback_name,
                output_token_cap=budget,
                originating_seat_id=originating_seat_id,
            )
            if tracked
            else []
        )
        return build_seat_model(
            seat_id,
            plan=plan,
            model_factory=factory,
            quick_mode=quick_mode or use_quick_binding,
            callbacks=callbacks,
            output_tokens=budget,
            include_reasoning_output=include_reasoning_output,
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

    retry_llms: dict[SeatId, Any] = {}
    allow_retry = False
    retry_binding = (plan.quick_bindings if quick_mode else plan.bindings)[
        SeatId.ANALYST_RETRY
    ]
    recovery_origins: tuple[SeatId, ...]
    if quick_mode:
        # Quick mode pays for structural recovery only at the two required,
        # gate-critical seats. The recovery binding uses a reasoning intent but
        # inherits the originating seat's visible-output budget.
        allow_retry = plan.status_for(SeatId.ANALYST_RETRY, quick_mode=True).enabled
        recovery_origins = (
            SeatId.SENIOR_FUNDAMENTALS,
            SeatId.PORTFOLIO_MANAGER,
        )
    else:
        if plan.schema == "legacy":
            # The compatibility window promises byte-for-byte legacy behavior:
            # older Gemini quick floors did not opt into RETRY-HIGH.
            from src.llms import is_gemini_v3_or_greater

            allow_retry = is_gemini_v3_or_greater(runtime_config.quick_think_llm)
        else:
            allow_retry = (
                Capability.TEXT_GENERATION in retry_binding.profile.capabilities
            )
        recovery_origins = (
            SeatId.MARKET,
            SeatId.SENTIMENT,
            SeatId.NEWS,
            SeatId.JUNIOR_FUNDAMENTALS,
            SeatId.SENIOR_FUNDAMENTALS,
            SeatId.FOREIGN_LANGUAGE,
            SeatId.VALUE_TRAP,
            SeatId.PORTFOLIO_MANAGER,
        )
    if allow_retry:
        for origin in recovery_origins:
            budget_key = SEATS[origin].budget_key
            if budget_key is None:
                continue
            retry_model = seat_model(
                SeatId.ANALYST_RETRY,
                output_tokens_override=output_budget(budget_key),
                tracking_agent_name=SEATS[origin].callback_name,
                originating_seat_id=origin,
            )
            if retry_model is not None:
                retry_llms[origin] = retry_model
        logger.debug(
            "structural_recovery_models_enabled",
            ticker=ticker,
            quick_mode=quick_mode,
            origins=[seat.value for seat in retry_llms],
        )
    else:
        logger.warning(
            "structural_recovery_binding_unavailable",
            ticker=ticker,
            provider=retry_binding.provider,
            model=retry_binding.model,
            reason="recovery binding lacks the required text-generation policy",
        )

    logger.debug(
        "synthesis_llm_mode",
        quick_mode=quick_mode,
        reasoning_intent="fast" if quick_mode else "reasoning",
    )
    bull_llm = seat_model(SeatId.BULL)
    bear_llm = seat_model(SeatId.BEAR)
    bull_r1_llm = (
        seat_model(SeatId.BULL, include_reasoning_output=True)
        if debate_reasoning_policy.active
        else bull_llm
    )
    bear_r1_llm = (
        seat_model(SeatId.BEAR, include_reasoning_output=True)
        if debate_reasoning_policy.active
        else bear_llm
    )
    # Formatting a completed canonical argument is not a second investment
    # opinion. Use each role's reviewed fast binding, no native-reasoning output,
    # and a small explicit cap instead of exposing the full researcher allowance.
    bull_repair_llm = (
        seat_model(
            SeatId.BULL,
            use_quick_binding=True,
            output_tokens_override=(
                debate_reasoning_policy.structured_repair_output_tokens
            ),
        )
        if debate_reasoning_policy.active
        else None
    )
    bear_repair_llm = (
        seat_model(
            SeatId.BEAR,
            use_quick_binding=True,
            output_tokens_override=(
                debate_reasoning_policy.structured_repair_output_tokens
            ),
        )
        if debate_reasoning_policy.active
        else None
    )
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

    auditor_requested = _is_auditor_enabled(plan, quick_mode=quick_mode)
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
        retry_llm_enabled=bool(retry_llms),
        consultant_enabled=consultant_enabled,
        auditor_enabled=auditor_enabled,
        apac_specialist_enabled=apac_specialist_enabled,
    )

    market = create_analyst_node(
        market_llm,
        "market_analyst",
        toolkit.get_technical_tools(),
        "market_report",
        retry_llm=retry_llms.get(SeatId.MARKET),
        allow_retry=SeatId.MARKET in retry_llms,
        research_budget_policy=research_policies["market_analyst"],
    )
    sentiment = create_analyst_node(
        social_llm,
        "sentiment_analyst",
        toolkit.get_sentiment_tools(),
        "sentiment_report",
        retry_llm=retry_llms.get(SeatId.SENTIMENT),
        allow_retry=SeatId.SENTIMENT in retry_llms,
        research_budget_policy=research_policies["sentiment_analyst"],
    )
    news = create_analyst_node(
        news_llm,
        "news_analyst",
        toolkit.get_news_tools(),
        "news_report",
        retry_llm=retry_llms.get(SeatId.NEWS),
        allow_retry=SeatId.NEWS in retry_llms,
        research_budget_policy=research_policies["news_analyst"],
    )

    foreign_llm = seat_model(SeatId.FOREIGN_LANGUAGE)
    foreign_analyst = create_analyst_node(
        foreign_llm,
        "foreign_language_analyst",
        toolkit.get_foreign_language_tools(),
        "foreign_language_report",
        retry_llm=retry_llms.get(SeatId.FOREIGN_LANGUAGE),
        allow_retry=SeatId.FOREIGN_LANGUAGE in retry_llms,
        research_budget_policy=research_policies["foreign_language_analyst"],
    )

    legal_llm = seat_model(SeatId.LEGAL_COUNSEL)
    legal_counsel = create_legal_counsel_node(legal_llm, toolkit.get_legal_tools())

    value_trap_llm = seat_model(SeatId.VALUE_TRAP)
    value_trap_detector = create_analyst_node(
        value_trap_llm,
        "value_trap_detector",
        toolkit.get_value_trap_tools(),
        "value_trap_report",
        retry_llm=retry_llms.get(SeatId.VALUE_TRAP),
        allow_retry=SeatId.VALUE_TRAP in retry_llms,
        research_budget_policy=research_policies["value_trap_detector"],
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
        retry_llm=retry_llms.get(SeatId.JUNIOR_FUNDAMENTALS),
        allow_retry=SeatId.JUNIOR_FUNDAMENTALS in retry_llms,
        research_budget_policy=research_policies["junior_fundamentals_analyst"],
    )
    senior_fund = create_analyst_node(
        senior_fund_llm,
        "fundamentals_analyst",
        toolkit.get_senior_fundamental_tools(),
        "fundamentals_report",
        retry_llm=retry_llms.get(SeatId.SENIOR_FUNDAMENTALS),
        allow_retry=SeatId.SENIOR_FUNDAMENTALS in retry_llms,
    )
    validator = create_financial_health_validator_node(strict_mode=strict_mode)

    market_tools = create_agent_tool_node(
        toolkit.get_market_tools(),
        "market_analyst",
        budget_policy=research_policies["market_analyst"],
    )
    sentiment_tools = create_agent_tool_node(
        toolkit.get_sentiment_tools(),
        "sentiment_analyst",
        budget_policy=research_policies["sentiment_analyst"],
    )
    news_tools = create_agent_tool_node(
        toolkit.get_news_tools(),
        "news_analyst",
        budget_policy=research_policies["news_analyst"],
    )
    junior_fund_tools = create_agent_tool_node(
        toolkit.get_junior_fundamental_tools(),
        "junior_fundamentals_analyst",
        budget_policy=research_policies["junior_fundamentals_analyst"],
    )
    foreign_tools = create_agent_tool_node(
        toolkit.get_foreign_language_tools(),
        "foreign_language_analyst",
        budget_policy=research_policies["foreign_language_analyst"],
    )
    value_trap_tools = create_agent_tool_node(
        toolkit.get_value_trap_tools(),
        "value_trap_detector",
        budget_policy=research_policies["value_trap_detector"],
    )

    researcher_handoff_kwargs: dict[str, Any] = (
        {"handoff_policy": debate_reasoning_policy}
        if debate_reasoning_policy.active
        else {}
    )

    def researcher(
        seat_llm: Any,
        memory: Any,
        agent_key: str,
        *,
        round_num: int,
        r1_llm: Any | None = None,
        repair_llm: Any | None = None,
    ) -> Any:
        # Only the R1 seats emit a handoff. The fallback exists to recover a
        # response the capsule contract degraded, so it is meaningless when the
        # policy is inactive — that omission is what keeps the feature inert by
        # default. The recovery kwargs are omitted rather than passed as None so
        # a caller (and the wiring tests) can see which seats are handoff-bearing
        # from the call itself.
        recovery: dict[str, Any] = {}
        if debate_reasoning_policy.active and r1_llm is not None:
            recovery["fallback_llm"] = seat_llm
            recovery["structured_repair_llm"] = repair_llm
        return create_researcher_node(
            r1_llm or seat_llm,
            memory,
            agent_key,
            round_num=round_num,
            **recovery,
            **researcher_handoff_kwargs,
        )

    bull_r1 = researcher(
        bull_llm,
        bull_memory,
        "bull_researcher",
        round_num=1,
        r1_llm=bull_r1_llm,
        repair_llm=bull_repair_llm,
    )
    bear_r1 = researcher(
        bear_llm,
        bear_memory,
        "bear_researcher",
        round_num=1,
        r1_llm=bear_r1_llm,
        repair_llm=bear_repair_llm,
    )
    bull_r2 = researcher(bull_llm, bull_memory, "bull_researcher", round_num=2)
    bear_r2 = researcher(bear_llm, bear_memory, "bear_researcher", round_num=2)
    res_mgr = create_research_manager_node(
        res_mgr_llm,
        invest_judge_memory,
        strict_mode=strict_mode,
        **researcher_handoff_kwargs,
    )
    trader = create_trader_node(trader_llm, trader_memory)
    risky = create_risk_debater_node(risky_llm, "risky_analyst")
    safe = create_risk_debater_node(safe_llm, "safe_analyst")
    neutral = create_risk_debater_node(neutral_llm, "neutral_analyst")
    pm = create_portfolio_manager_node(
        pm_llm,
        risk_manager_memory,
        strict_mode=strict_mode,
        recovery_llm=retry_llms.get(SeatId.PORTFOLIO_MANAGER),
    )
    pm_fast_fail = create_screen_rejection_node()

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
        debate_reasoning_policy=debate_reasoning_policy,
    )
