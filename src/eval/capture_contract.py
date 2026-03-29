from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeCaptureSpec:
    node_name: str
    capture_role: str
    baseline_eligible: bool
    prompt_key: str | None = None
    expects_llm_calls: bool = False
    artifact_fields: tuple[str, ...] = ()
    # Applicability is agent-scoped, not inferred from global run-level fields.
    # Future evaluators should only judge requirements declared here. If a field
    # is absent from evaluator_scope, it is not applicable for that node.
    evaluator_scope: tuple[str, ...] = ()
    rubric_family: str | None = None


_DEFAULT_SPEC = NodeCaptureSpec(
    node_name="__unknown__",
    capture_role="deterministic_helper",
    baseline_eligible=False,
)


NODE_CAPTURE_SPECS: dict[str, NodeCaptureSpec] = {
    "Dispatcher": NodeCaptureSpec("Dispatcher", "barrier", False),
    "Sync Check": NodeCaptureSpec("Sync Check", "barrier", False),
    "Fundamentals Sync Check": NodeCaptureSpec(
        "Fundamentals Sync Check", "barrier", False
    ),
    "Debate Sync R1": NodeCaptureSpec("Debate Sync R1", "barrier", False),
    "Debate Sync Final": NodeCaptureSpec("Debate Sync Final", "barrier", False),
    "State Cleaner": NodeCaptureSpec("State Cleaner", "deterministic_helper", False),
    "Financial Validator": NodeCaptureSpec(
        "Financial Validator", "deterministic_helper", False
    ),
    "Chart Generator": NodeCaptureSpec(
        "Chart Generator", "deterministic_helper", False
    ),
    "Market Analyst": NodeCaptureSpec(
        "Market Analyst",
        "agent",
        True,
        prompt_key="market_analyst",
        expects_llm_calls=True,
        artifact_fields=("market_report",),
        evaluator_scope=("artifact_complete",),
    ),
    "Sentiment Analyst": NodeCaptureSpec(
        "Sentiment Analyst",
        "agent",
        True,
        prompt_key="sentiment_analyst",
        expects_llm_calls=True,
        artifact_fields=("sentiment_report",),
        evaluator_scope=("artifact_complete",),
    ),
    "News Analyst": NodeCaptureSpec(
        "News Analyst",
        "agent",
        True,
        prompt_key="news_analyst",
        expects_llm_calls=True,
        artifact_fields=("news_report",),
        evaluator_scope=("artifact_complete",),
    ),
    "Junior Fundamentals Analyst": NodeCaptureSpec(
        "Junior Fundamentals Analyst",
        "agent",
        True,
        prompt_key="junior_fundamentals_analyst",
        expects_llm_calls=True,
        artifact_fields=("raw_fundamentals_data",),
        evaluator_scope=("artifact_complete", "raw_data_wrapper_complete"),
    ),
    "Foreign Language Analyst": NodeCaptureSpec(
        "Foreign Language Analyst",
        "agent",
        True,
        prompt_key="foreign_language_analyst",
        expects_llm_calls=True,
        artifact_fields=("foreign_language_report",),
        evaluator_scope=("artifact_complete",),
    ),
    "Legal Counsel": NodeCaptureSpec(
        "Legal Counsel",
        "agent",
        True,
        prompt_key="legal_counsel",
        expects_llm_calls=True,
        artifact_fields=("legal_report",),
        evaluator_scope=("artifact_complete", "legal_json_valid"),
        rubric_family="legal_structured",
    ),
    "Value Trap Detector": NodeCaptureSpec(
        "Value Trap Detector",
        "agent",
        True,
        prompt_key="value_trap_detector",
        expects_llm_calls=True,
        artifact_fields=("value_trap_report",),
        evaluator_scope=(
            "artifact_complete",
            "value_trap_block_present",
            "value_trap_score_parseable",
        ),
        rubric_family="risk_structured",
    ),
    "Auditor": NodeCaptureSpec(
        "Auditor",
        "agent",
        True,
        prompt_key="global_forensic_auditor",
        expects_llm_calls=True,
        artifact_fields=("auditor_report",),
        evaluator_scope=("artifact_complete",),
    ),
    "Fundamentals Analyst": NodeCaptureSpec(
        "Fundamentals Analyst",
        "agent",
        True,
        prompt_key="fundamentals_analyst",
        expects_llm_calls=True,
        artifact_fields=("fundamentals_report",),
        evaluator_scope=("artifact_complete", "data_block_present"),
        rubric_family="analysis_report",
    ),
    "Bull Researcher R1": NodeCaptureSpec(
        "Bull Researcher R1",
        "agent",
        True,
        prompt_key="bull_researcher",
        expects_llm_calls=True,
        artifact_fields=("investment_debate_state",),
        evaluator_scope=("artifact_complete",),
    ),
    "Bull Researcher R2": NodeCaptureSpec(
        "Bull Researcher R2",
        "agent",
        True,
        prompt_key="bull_researcher",
        expects_llm_calls=True,
        artifact_fields=("investment_debate_state",),
        evaluator_scope=("artifact_complete",),
    ),
    "Bear Researcher R1": NodeCaptureSpec(
        "Bear Researcher R1",
        "agent",
        True,
        prompt_key="bear_researcher",
        expects_llm_calls=True,
        artifact_fields=("investment_debate_state",),
        evaluator_scope=("artifact_complete",),
    ),
    "Bear Researcher R2": NodeCaptureSpec(
        "Bear Researcher R2",
        "agent",
        True,
        prompt_key="bear_researcher",
        expects_llm_calls=True,
        artifact_fields=("investment_debate_state",),
        evaluator_scope=("artifact_complete",),
    ),
    "Research Manager": NodeCaptureSpec(
        "Research Manager",
        "agent",
        True,
        prompt_key="research_manager",
        expects_llm_calls=True,
        artifact_fields=("investment_plan",),
        evaluator_scope=("artifact_complete",),
    ),
    "Valuation Calculator": NodeCaptureSpec(
        "Valuation Calculator",
        "agent",
        True,
        prompt_key="valuation_calculator",
        expects_llm_calls=True,
        artifact_fields=("valuation_params",),
        evaluator_scope=("artifact_complete", "valuation_params_present"),
        rubric_family="valuation_structured",
    ),
    "Consultant": NodeCaptureSpec(
        "Consultant",
        "agent",
        True,
        prompt_key="consultant",
        expects_llm_calls=True,
        artifact_fields=("consultant_review",),
        evaluator_scope=("artifact_complete", "consultant_verdict_present"),
    ),
    "Trader": NodeCaptureSpec(
        "Trader",
        "agent",
        True,
        prompt_key="trader",
        expects_llm_calls=True,
        artifact_fields=("trader_investment_plan",),
        evaluator_scope=("artifact_complete", "trade_block_present"),
        rubric_family="trade_decision",
    ),
    "Risky Analyst": NodeCaptureSpec(
        "Risky Analyst",
        "agent",
        True,
        prompt_key="risky_analyst",
        expects_llm_calls=True,
        artifact_fields=("risk_debate_state",),
        evaluator_scope=("artifact_complete",),
    ),
    "Safe Analyst": NodeCaptureSpec(
        "Safe Analyst",
        "agent",
        True,
        prompt_key="safe_analyst",
        expects_llm_calls=True,
        artifact_fields=("risk_debate_state",),
        evaluator_scope=("artifact_complete",),
    ),
    "Neutral Analyst": NodeCaptureSpec(
        "Neutral Analyst",
        "agent",
        True,
        prompt_key="neutral_analyst",
        expects_llm_calls=True,
        artifact_fields=("risk_debate_state",),
        evaluator_scope=("artifact_complete",),
    ),
    "Portfolio Manager": NodeCaptureSpec(
        "Portfolio Manager",
        "agent",
        True,
        prompt_key="portfolio_manager",
        expects_llm_calls=True,
        artifact_fields=("final_trade_decision",),
        evaluator_scope=(
            "artifact_complete",
            "pm_block_present",
            "pm_verdict_present",
        ),
        rubric_family="portfolio_decision",
    ),
    "PM Fast-Fail": NodeCaptureSpec(
        "PM Fast-Fail",
        "agent",
        True,
        prompt_key="portfolio_manager",
        expects_llm_calls=True,
        artifact_fields=("final_trade_decision",),
        evaluator_scope=(
            "artifact_complete",
            "pm_block_present",
            "pm_verdict_present",
        ),
        rubric_family="portfolio_decision",
    ),
}


def get_node_capture_spec(node_name: str) -> NodeCaptureSpec:
    return NODE_CAPTURE_SPECS.get(
        node_name,
        NodeCaptureSpec(
            node_name=node_name,
            capture_role=_DEFAULT_SPEC.capture_role,
            baseline_eligible=_DEFAULT_SPEC.baseline_eligible,
            prompt_key=_DEFAULT_SPEC.prompt_key,
            expects_llm_calls=_DEFAULT_SPEC.expects_llm_calls,
            artifact_fields=_DEFAULT_SPEC.artifact_fields,
            evaluator_scope=_DEFAULT_SPEC.evaluator_scope,
            rubric_family=_DEFAULT_SPEC.rubric_family,
        ),
    )


def iter_baseline_eligible_specs() -> list[NodeCaptureSpec]:
    return [spec for spec in NODE_CAPTURE_SPECS.values() if spec.baseline_eligible]


def iter_stage3_judge_specs() -> list[NodeCaptureSpec]:
    specs: list[NodeCaptureSpec] = []
    seen_prompt_keys: set[str] = set()
    for spec in NODE_CAPTURE_SPECS.values():
        prompt_key = spec.prompt_key
        if not prompt_key:
            continue
        if not spec.baseline_eligible or spec.capture_role != "agent":
            continue
        if spec.rubric_family is None or prompt_key in seen_prompt_keys:
            continue
        seen_prompt_keys.add(prompt_key)
        specs.append(spec)
    return specs
