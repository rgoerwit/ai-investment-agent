"""Canonical registry for every application-owned LLM consumer."""

from dataclasses import dataclass
from enum import StrEnum

from src.llm_runtime.capabilities import Capability


class SeatId(StrEnum):
    MARKET = "market_analyst"
    SENTIMENT = "sentiment_analyst"
    NEWS = "news_analyst"
    JUNIOR_FUNDAMENTALS = "junior_fundamentals_analyst"
    SENIOR_FUNDAMENTALS = "fundamentals_analyst"
    FOREIGN_LANGUAGE = "foreign_language_analyst"
    LEGAL_COUNSEL = "legal_counsel"
    VALUE_TRAP = "value_trap_detector"
    BULL = "bull_researcher"
    BEAR = "bear_researcher"
    RESEARCH_MANAGER = "research_manager"
    VALUATION = "valuation_calculator"
    TRADER = "trader"
    RISKY = "risky_analyst"
    SAFE = "safe_analyst"
    NEUTRAL = "neutral_analyst"
    PORTFOLIO_MANAGER = "portfolio_manager"
    ANALYST_RETRY = "analyst_retry"
    CONSULTANT = "consultant"
    AUDITOR = "forensic_auditor"
    AUDITOR_ESCALATION = "forensic_auditor_escalation"
    APAC = "apac_regional_specialist"
    APAC_DIRECT_RETRY = "apac_regional_specialist_direct_retry"
    ARTICLE_WRITER = "article_writer"
    ARTICLE_WRITER_REVIEW_FALLBACK = "article_writer_review_fallback"
    ARTICLE_WRITER_BASE_FALLBACK = "article_writer_base_fallback"
    EDITOR = "article_editor"
    CONTENT_INSPECTOR = "content_inspector"
    MACRO_CONTEXT = "macro_context_analyst"
    RETROSPECTIVE = "retrospective_lessons"
    HEALTH_CHECK = "health_check"
    PORTFOLIO_MACRO_CLASSIFIER = "portfolio_macro_classifier"
    SIGNAL_PROCESSOR = "signal_processor"
    SEMANTIC_JUDGE = "semantic_eval_judge"


class BindingGroup(StrEnum):
    BASE = "base"
    REVIEW = "review"
    REGIONAL_REVIEW = "regional_review"
    WRITER = "writer"
    OPERATIONAL = "operational"
    JUDGE = "judge"


class AuthorityStage(StrEnum):
    DECISION = "decision"
    DECISION_REVIEW = "decision_review"
    EDITORIAL = "editorial"
    OPERATIONAL = "operational"
    EVALUATION = "evaluation"


class ModelIntent(StrEnum):
    FAST = "fast"
    REASONING = "reasoning"
    CRITICAL = "critical"
    ESCALATION = "escalation"
    PROSE = "prose"
    CLASSIFIER = "classifier"


class ReasoningAdjustment(StrEnum):
    NONE = "none"
    ONE_STEP = "one_step"


@dataclass(frozen=True)
class SeatExecutionPolicy:
    """Stable call semantics that belong to a seat rather than a provider."""

    sampling_temperature: float | None = None
    client_timeout_seconds: float | None = None
    sdk_max_retries: int | None = None
    standard_tier_only: bool = False
    # Gate-critical seats must not queue on a best-effort tier under ``--quick``:
    # the in-process cap would never fire and the pipeline watchdog would SIGTERM
    # the run with no verdict (the 2026-07-06 "every quick ticker ANALYSIS FAILED"
    # incident). Full mode keeps the configured tier plus its widened watchdog.
    standard_tier_in_quick_mode: bool = False
    reasoning_control_enabled: bool = True
    output_token_override_enabled: bool = True

    def __post_init__(self) -> None:
        if self.sampling_temperature is not None and not (
            0.0 <= self.sampling_temperature <= 2.0
        ):
            raise ValueError("sampling_temperature must be between 0.0 and 2.0")
        if self.client_timeout_seconds is not None and self.client_timeout_seconds <= 0:
            raise ValueError("client_timeout_seconds must be positive")
        if self.sdk_max_retries is not None and self.sdk_max_retries < 0:
            raise ValueError("sdk_max_retries cannot be negative")


@dataclass(frozen=True)
class SeatSpec:
    seat_id: SeatId
    callback_name: str
    prompt_key: str | None
    budget_key: str | None
    state_field: str | None
    binding_group: BindingGroup
    authority_stage: AuthorityStage
    normal_intent: ModelIntent
    quick_intent: ModelIntent
    requires: frozenset[Capability]
    retry_intent: ModelIntent | None = None
    normal_reasoning_adjustment: ReasoningAdjustment = ReasoningAdjustment.NONE
    optional_mode_field: str | None = None
    execution_policy: SeatExecutionPolicy = SeatExecutionPolicy()
    disabled_in_quick_mode: bool = False


_TEXT = frozenset({Capability.TEXT_GENERATION})
_TOOLS = frozenset({Capability.TEXT_GENERATION, Capability.TOOL_CALLING})


def _seat(
    seat_id: SeatId,
    callback_name: str,
    *,
    prompt_key: str | None = None,
    budget_key: str | None = None,
    state_field: str | None = None,
    group: BindingGroup = BindingGroup.BASE,
    stage: AuthorityStage = AuthorityStage.DECISION,
    normal: ModelIntent = ModelIntent.FAST,
    quick: ModelIntent = ModelIntent.FAST,
    requires: frozenset[Capability] = _TEXT,
    retry: ModelIntent | None = None,
    adjustment: ReasoningAdjustment = ReasoningAdjustment.NONE,
    optional_mode_field: str | None = None,
    execution_policy: SeatExecutionPolicy = SeatExecutionPolicy(),
    disabled_in_quick_mode: bool = False,
) -> SeatSpec:
    return SeatSpec(
        seat_id=seat_id,
        callback_name=callback_name,
        prompt_key=prompt_key,
        budget_key=budget_key,
        state_field=state_field,
        binding_group=group,
        authority_stage=stage,
        normal_intent=normal,
        quick_intent=quick,
        requires=requires,
        retry_intent=retry,
        normal_reasoning_adjustment=adjustment,
        optional_mode_field=optional_mode_field,
        execution_policy=execution_policy,
        disabled_in_quick_mode=disabled_in_quick_mode,
    )


SEATS: dict[SeatId, SeatSpec] = {
    SeatId.MARKET: _seat(
        SeatId.MARKET,
        "Market Analyst",
        prompt_key="market_analyst",
        budget_key="Market Analyst",
        state_field="market_report",
        requires=_TOOLS,
        retry=ModelIntent.REASONING,
    ),
    SeatId.SENTIMENT: _seat(
        SeatId.SENTIMENT,
        "Sentiment Analyst",
        prompt_key="sentiment_analyst",
        budget_key="Sentiment Analyst",
        state_field="sentiment_report",
        requires=_TOOLS,
        retry=ModelIntent.REASONING,
    ),
    SeatId.NEWS: _seat(
        SeatId.NEWS,
        "News Analyst",
        prompt_key="news_analyst",
        budget_key="News Analyst",
        state_field="news_report",
        requires=_TOOLS,
        retry=ModelIntent.REASONING,
    ),
    SeatId.JUNIOR_FUNDAMENTALS: _seat(
        SeatId.JUNIOR_FUNDAMENTALS,
        "Junior Fundamentals Analyst",
        prompt_key="junior_fundamentals_analyst",
        budget_key="Junior Fundamentals Analyst",
        state_field="raw_fundamentals_data",
        requires=_TOOLS,
        retry=ModelIntent.REASONING,
    ),
    SeatId.SENIOR_FUNDAMENTALS: _seat(
        SeatId.SENIOR_FUNDAMENTALS,
        "Fundamentals Analyst",
        prompt_key="fundamentals_analyst",
        budget_key="Fundamentals Analyst",
        state_field="fundamentals_report",
        normal=ModelIntent.CRITICAL,
        quick=ModelIntent.CRITICAL,
        requires=_TOOLS,
        retry=ModelIntent.REASONING,
        execution_policy=SeatExecutionPolicy(standard_tier_in_quick_mode=True),
    ),
    SeatId.FOREIGN_LANGUAGE: _seat(
        SeatId.FOREIGN_LANGUAGE,
        "Foreign Language Analyst",
        prompt_key="foreign_language_analyst",
        budget_key="Foreign Language Analyst",
        state_field="foreign_language_report",
        requires=_TOOLS,
        retry=ModelIntent.REASONING,
    ),
    SeatId.LEGAL_COUNSEL: _seat(
        SeatId.LEGAL_COUNSEL,
        "Legal Counsel",
        prompt_key="legal_counsel",
        budget_key="Legal Counsel",
        state_field="legal_report",
        requires=_TOOLS,
    ),
    SeatId.VALUE_TRAP: _seat(
        SeatId.VALUE_TRAP,
        "Value Trap Detector",
        prompt_key="value_trap_detector",
        budget_key="Value Trap Detector",
        state_field="value_trap_report",
        requires=_TOOLS,
        retry=ModelIntent.REASONING,
        adjustment=ReasoningAdjustment.ONE_STEP,
    ),
    SeatId.BULL: _seat(
        SeatId.BULL,
        "Bull Researcher",
        prompt_key="bull_researcher",
        budget_key="Bull Researcher",
        normal=ModelIntent.REASONING,
    ),
    SeatId.BEAR: _seat(
        SeatId.BEAR,
        "Bear Researcher",
        prompt_key="bear_researcher",
        budget_key="Bear Researcher",
        normal=ModelIntent.REASONING,
    ),
    SeatId.RESEARCH_MANAGER: _seat(
        SeatId.RESEARCH_MANAGER,
        "Research Manager",
        prompt_key="research_manager",
        budget_key="Research Manager",
        state_field="investment_plan",
        normal=ModelIntent.REASONING,
    ),
    SeatId.VALUATION: _seat(
        SeatId.VALUATION,
        "Valuation Calculator",
        prompt_key="valuation_calculator",
        budget_key="Valuation Calculator",
        state_field="valuation_report",
    ),
    SeatId.TRADER: _seat(
        SeatId.TRADER,
        "Trader",
        prompt_key="trader",
        budget_key="Trader",
        state_field="trader_investment_plan",
    ),
    SeatId.RISKY: _seat(
        SeatId.RISKY,
        "Risky Analyst",
        prompt_key="risky_analyst",
        budget_key="Risky Analyst",
        normal=ModelIntent.REASONING,
    ),
    SeatId.SAFE: _seat(
        SeatId.SAFE,
        "Safe Analyst",
        prompt_key="safe_analyst",
        budget_key="Safe Analyst",
        normal=ModelIntent.REASONING,
    ),
    SeatId.NEUTRAL: _seat(
        SeatId.NEUTRAL,
        "Neutral Analyst",
        prompt_key="neutral_analyst",
        budget_key="Neutral Analyst",
        normal=ModelIntent.REASONING,
    ),
    SeatId.PORTFOLIO_MANAGER: _seat(
        SeatId.PORTFOLIO_MANAGER,
        "Portfolio Manager",
        prompt_key="portfolio_manager",
        budget_key="Portfolio Manager",
        state_field="final_trade_decision",
        normal=ModelIntent.CRITICAL,
        quick=ModelIntent.CRITICAL,
        execution_policy=SeatExecutionPolicy(standard_tier_in_quick_mode=True),
    ),
    SeatId.ANALYST_RETRY: _seat(
        SeatId.ANALYST_RETRY,
        "Analyst Retry",
        normal=ModelIntent.REASONING,
        quick=ModelIntent.REASONING,
        requires=_TOOLS,
        disabled_in_quick_mode=True,
    ),
    SeatId.CONSULTANT: _seat(
        SeatId.CONSULTANT,
        "Consultant",
        prompt_key="consultant",
        budget_key="Consultant",
        state_field="consultant_review",
        group=BindingGroup.REVIEW,
        stage=AuthorityStage.DECISION_REVIEW,
        normal=ModelIntent.REASONING,
        requires=_TOOLS,
        optional_mode_field="llm_consultant_mode",
        execution_policy=SeatExecutionPolicy(sdk_max_retries=0),
    ),
    SeatId.AUDITOR: _seat(
        SeatId.AUDITOR,
        "Global Forensic Auditor",
        prompt_key="auditor",
        budget_key="Global Forensic Auditor",
        state_field="auditor_report",
        group=BindingGroup.REVIEW,
        stage=AuthorityStage.DECISION_REVIEW,
        normal=ModelIntent.REASONING,
        requires=_TOOLS,
        optional_mode_field="llm_auditor_mode",
    ),
    SeatId.AUDITOR_ESCALATION: _seat(
        SeatId.AUDITOR_ESCALATION,
        "Global Forensic Auditor Escalation",
        prompt_key="auditor",
        budget_key="Global Forensic Auditor",
        group=BindingGroup.REVIEW,
        stage=AuthorityStage.DECISION_REVIEW,
        normal=ModelIntent.ESCALATION,
        quick=ModelIntent.ESCALATION,
        requires=_TOOLS,
        optional_mode_field="llm_auditor_mode",
    ),
    SeatId.APAC: _seat(
        SeatId.APAC,
        "APAC Regional Specialist",
        prompt_key="apac_regional_specialist",
        budget_key="APAC Regional Specialist",
        state_field="apac_specialist_report",
        group=BindingGroup.REGIONAL_REVIEW,
        stage=AuthorityStage.DECISION_REVIEW,
        normal=ModelIntent.REASONING,
        quick=ModelIntent.REASONING,
        requires=_TEXT,
        optional_mode_field="llm_apac_mode",
        disabled_in_quick_mode=True,
    ),
    SeatId.APAC_DIRECT_RETRY: _seat(
        SeatId.APAC_DIRECT_RETRY,
        "APAC Regional Specialist Direct Retry",
        prompt_key="apac_regional_specialist",
        budget_key="APAC Regional Specialist",
        group=BindingGroup.REGIONAL_REVIEW,
        stage=AuthorityStage.DECISION_REVIEW,
        normal=ModelIntent.REASONING,
        quick=ModelIntent.REASONING,
        requires=_TEXT,
        optional_mode_field="llm_apac_mode",
        disabled_in_quick_mode=True,
    ),
    SeatId.ARTICLE_WRITER: _seat(
        SeatId.ARTICLE_WRITER,
        "Article Writer",
        prompt_key="writer",
        group=BindingGroup.WRITER,
        stage=AuthorityStage.EDITORIAL,
        normal=ModelIntent.PROSE,
        quick=ModelIntent.PROSE,
    ),
    SeatId.ARTICLE_WRITER_REVIEW_FALLBACK: _seat(
        SeatId.ARTICLE_WRITER_REVIEW_FALLBACK,
        "Article Writer Fallback",
        prompt_key="writer",
        group=BindingGroup.REVIEW,
        stage=AuthorityStage.EDITORIAL,
        normal=ModelIntent.PROSE,
        quick=ModelIntent.PROSE,
    ),
    SeatId.ARTICLE_WRITER_BASE_FALLBACK: _seat(
        SeatId.ARTICLE_WRITER_BASE_FALLBACK,
        "Article Writer Fallback",
        prompt_key="writer",
        group=BindingGroup.BASE,
        stage=AuthorityStage.EDITORIAL,
        normal=ModelIntent.PROSE,
        quick=ModelIntent.PROSE,
    ),
    SeatId.EDITOR: _seat(
        SeatId.EDITOR,
        "Article Editor",
        prompt_key="editor",
        budget_key="Article Editor",
        group=BindingGroup.REVIEW,
        stage=AuthorityStage.EDITORIAL,
        normal=ModelIntent.REASONING,
        quick=ModelIntent.REASONING,
        requires=frozenset(
            {
                Capability.TEXT_GENERATION,
                Capability.TOOL_CALLING,
                Capability.STRUCTURED_OUTPUT,
            }
        ),
        optional_mode_field="llm_editor_mode",
    ),
    SeatId.CONTENT_INSPECTOR: _seat(
        SeatId.CONTENT_INSPECTOR,
        "Content Inspector",
        group=BindingGroup.OPERATIONAL,
        stage=AuthorityStage.OPERATIONAL,
        normal=ModelIntent.CLASSIFIER,
        quick=ModelIntent.CLASSIFIER,
        execution_policy=SeatExecutionPolicy(
            sampling_temperature=0.0,
            standard_tier_only=True,
        ),
    ),
    SeatId.MACRO_CONTEXT: _seat(
        SeatId.MACRO_CONTEXT,
        "Macro Context Analyst",
        prompt_key="macro_context_analyst",
        group=BindingGroup.OPERATIONAL,
        stage=AuthorityStage.OPERATIONAL,
        execution_policy=SeatExecutionPolicy(sampling_temperature=0.1),
    ),
    SeatId.RETROSPECTIVE: _seat(
        SeatId.RETROSPECTIVE,
        "Retrospective Lessons",
        group=BindingGroup.OPERATIONAL,
        stage=AuthorityStage.OPERATIONAL,
    ),
    SeatId.HEALTH_CHECK: _seat(
        SeatId.HEALTH_CHECK,
        "Health Check",
        group=BindingGroup.OPERATIONAL,
        stage=AuthorityStage.OPERATIONAL,
        execution_policy=SeatExecutionPolicy(
            sampling_temperature=0.0,
            client_timeout_seconds=10.0,
            sdk_max_retries=1,
            standard_tier_only=True,
            reasoning_control_enabled=False,
            output_token_override_enabled=False,
        ),
    ),
    SeatId.PORTFOLIO_MACRO_CLASSIFIER: _seat(
        SeatId.PORTFOLIO_MACRO_CLASSIFIER,
        "Portfolio Macro Classifier",
        group=BindingGroup.OPERATIONAL,
        stage=AuthorityStage.OPERATIONAL,
        normal=ModelIntent.REASONING,
        quick=ModelIntent.CLASSIFIER,
    ),
    SeatId.SIGNAL_PROCESSOR: _seat(
        SeatId.SIGNAL_PROCESSOR,
        "Signal Processor",
        group=BindingGroup.OPERATIONAL,
        stage=AuthorityStage.OPERATIONAL,
        normal=ModelIntent.CLASSIFIER,
        quick=ModelIntent.CLASSIFIER,
    ),
    SeatId.SEMANTIC_JUDGE: _seat(
        SeatId.SEMANTIC_JUDGE,
        "Semantic Eval Judge",
        group=BindingGroup.JUDGE,
        stage=AuthorityStage.EVALUATION,
        normal=ModelIntent.REASONING,
        quick=ModelIntent.FAST,
    ),
}


def get_seat(seat_id: SeatId) -> SeatSpec:
    return SEATS[seat_id]
