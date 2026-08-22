from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import structlog
from langchain_core.messages import HumanMessage
from langgraph.types import RunnableConfig

from src.config import config as settings_config
from src.error_safety import redact_sensitive_text, summarize_exception
from src.runtime_config import get_runtime_config
from src.runtime_diagnostics import (
    classify_failure,
    failure_artifact,
    get_valid_artifact_content,
    success_artifact,
)
from src.tooling.text_boundary import format_untrusted_block

from . import message_utils, support
from . import runtime as agent_runtime
from .debate_handoffs import (
    DebateReasoningPolicy,
    extract_native_reasoning,
    parse_structured_rationale_candidate,
    render_balanced_handoffs,
    render_opponent_handoff,
    split_structured_rationale,
    structured_rationale_addendum,
    structured_rationale_repair_prompt,
)
from .evidence_constraints import downstream_evidence_constraints
from .governance_prompt import governance_block
from .output_limits import cap_state_value
from .output_validation import (
    log_output_diagnostics,
    log_truncation_diagnostic,
    should_fail_closed,
    validate_required_output,
)
from .state import AgentState

logger = structlog.get_logger(__name__)

_ROUND1_REPORT_BUDGETS = {
    "market": 1800,
    "sentiment": 1200,
    "news": 1800,
    "fundamentals": 4000,
}

_ROUND2_ANCHOR_BUDGETS = {
    "market": 400,
    "sentiment": 250,
    "news": 500,
    "fundamentals": 1200,
    # foreign_language intentionally omitted — anchors only. Senior's
    # DATA_BLOCK already carries the structured FLA fields by round 2.
}

_STRICT_RM_ADDENDUM = """
---
## STRICT MODE — Research Manager Instruction

You are operating in STRICT mode. Apply this lens when synthesizing analyst inputs:

1. **Evidence over narrative**: Weight your synthesis toward concrete, documented facts
   (filed cash flows, signed contracts, observable margin trends) over projections or
   "could benefit from" language. If the bull case relies primarily on potential rather
   than demonstrated operating traction, frame your synthesis toward DO_NOT_INITIATE.

2. **Value-realization evidence**: A BUY recommendation requires at least one verifiable
   path for undervaluation to close or fundamentals to compound. That path may take
   multiple years; a near-term catalyst is not required. Unsupported "eventual" or
   "possible" catalysts do not qualify in strict mode.

3. **Data vacuum discipline**: If the combined analyst reports show significant data gaps
   (missing OCF, unknown ownership structure, no analyst coverage data), flag them
   explicitly and weight toward caution — do not paper over gaps with qualitative reasoning.

4. **Bear argument weighting**: Give bear arguments proportionally more weight than in
   normal mode. The burden of proof is on the bull case in strict mode.
"""


def _summarize_report(report: str, kind: str, budget: int) -> str:
    if report == "N/A":
        return "N/A"
    return support.summarize_for_pm(report, kind, budget)


def _build_research_report_bundle(state: AgentState, budgets: dict[str, int]) -> str:
    market_report = get_valid_artifact_content(state, "market_report") or "N/A"
    sentiment_report = get_valid_artifact_content(state, "sentiment_report") or "N/A"
    news_report = get_valid_artifact_content(state, "news_report") or "N/A"
    fundamentals_report = (
        get_valid_artifact_content(state, "fundamentals_report") or "N/A"
    )
    parts = [
        f"MARKET ANALYST REPORT:\n{_summarize_report(market_report, 'market', budgets['market'])}",
        f"SENTIMENT ANALYST REPORT:\n{_summarize_report(sentiment_report, 'sentiment', budgets['sentiment'])}",
        f"NEWS ANALYST REPORT:\n{_summarize_report(news_report, 'news', budgets['news'])}",
        f"FUNDAMENTALS ANALYST REPORT:\n{_summarize_report(fundamentals_report, 'fundamentals', budgets['fundamentals'])}",
    ]
    return "\n\n".join(parts)


@dataclass(frozen=True)
class _ResearcherSeat:
    """Per-node invariants that response resolution needs.

    Grouped so the resolver below takes one argument instead of seven; every
    field is fixed at node-construction time.
    """

    agent_key: str
    researcher_type: str
    round_num: int
    policy: DebateReasoningPolicy
    llm: Any
    fallback_llm: Any | None
    structured_repair_llm: Any | None


@dataclass(frozen=True)
class _ResolvedResearcherOutput:
    """What the researcher actually produced, after any recovery path ran."""

    response: Any
    # The model that produced `response` — not necessarily `seat.llm`, so
    # truncation and output diagnostics must be attributed to this one.
    runnable: Any
    canonical: str
    structured_rationale: str
    native_reasoning: str


async def _resolve_researcher_output(
    seat: _ResearcherSeat,
    *,
    agent_name: str,
    prompt: str,
    canonical_prompt: str,
    runtime_config: Any,
) -> _ResolvedResearcherOutput:
    """Run the researcher call and any recovery it needs, in one place.

    Four paths converge here — the primary call, a transport fallback when a
    provider rejects the reasoning-output request, a re-ask on a degraded or
    empty response, and a narrow structuring repair. Extracted from
    ``researcher_node`` so the node itself reads linearly; this is a move, not a
    behaviour change. Exceptions propagate to the node's artifact-failure
    handler exactly as before, except where a path documents its own recovery.
    ``runtime_config`` is threaded as an object rather than a pre-read flag so
    that ``developer_debug_active`` is still read only on the repair path — an
    unconditional read here would be a behaviour change, not a move.
    """

    policy = seat.policy
    round_num = seat.round_num

    async def invoke(
        target_llm: Any, target_prompt: str, *, context_suffix: str = ""
    ) -> Any:
        return await agent_runtime.invoke_with_rate_limit_handling(
            target_llm,
            [HumanMessage(content=target_prompt)],
            context=f"{agent_name} R{round_num}{context_suffix}",
            canonical_agent=agent_name,
            provider=support.infer_provider_name(target_llm),
            model_name=support.get_model_name(target_llm),
        )

    runnable = seat.llm
    try:
        response = await invoke(runnable, prompt)
    except Exception as exc:
        details = classify_failure(
            exc,
            provider=support.infer_provider_name(runnable),
            model_name=support.get_model_name(runnable),
        )
        if seat.fallback_llm is None or details.kind != "bad_request":
            raise
        logger.warning(
            "debate_reasoning_output_unsupported",
            agent=seat.agent_key,
            round=round_num,
            provider=details.provider,
            model=support.get_model_name(runnable),
        )
        runnable = seat.fallback_llm
        response = await invoke(runnable, prompt)

    canonical, structured_rationale = split_structured_rationale(
        message_utils.extract_string_content(response.content),
        policy=policy,
        round_num=round_num,
    )
    partial_reason = agent_runtime.response_partial_reason(response)
    if seat.fallback_llm is not None and (not canonical.strip() or partial_reason):
        logger.warning(
            "debate_reasoning_output_degraded",
            agent=seat.agent_key,
            round=round_num,
            reason="empty_canonical_output"
            if not canonical.strip()
            else partial_reason,
        )
        runnable = seat.fallback_llm
        response = await invoke(runnable, canonical_prompt)
        canonical, structured_rationale = split_structured_rationale(
            message_utils.extract_string_content(response.content),
            policy=policy,
            round_num=round_num,
        )

    native_reasoning = (
        extract_native_reasoning(response, char_cap=policy.native_char_cap)
        if policy.emits_in(round_num)
        else ""
    )
    if policy.emits_in(round_num) and not structured_rationale:
        structured_rationale = await _repair_structured_rationale(
            seat,
            invoke=invoke,
            fallback_runnable=runnable,
            canonical=canonical,
            runtime_config=runtime_config,
        )

    return _ResolvedResearcherOutput(
        response=response,
        runnable=runnable,
        canonical=canonical,
        structured_rationale=structured_rationale,
        native_reasoning=native_reasoning,
    )


async def _repair_structured_rationale(
    seat: _ResearcherSeat,
    *,
    invoke: Any,
    fallback_runnable: Any,
    canonical: str,
    runtime_config: Any,
) -> str:
    """Structure an already-written argument when the inline capsule is missing.

    Live models may ignore an extra output block in a long debate prompt. This
    follow-up sees only the canonical argument, so it restructures an existing
    claim set rather than forming a second opinion, and it cannot introduce
    source material the researcher did not cite. Failure is optional: a balanced
    native pair can still carry the handoff, so every path returns "" rather
    than raising.
    """

    repair_target = seat.structured_repair_llm or seat.fallback_llm or fallback_runnable
    repair_prompt = (
        structured_rationale_repair_prompt(seat.policy)
        + "\n\n"
        + format_untrusted_block(
            canonical,
            "CANONICAL ROUND-1 ARGUMENT TO STRUCTURE",
            provenance=f"{seat.researcher_type} researcher canonical output",
        )
    )
    repair_content = ""
    structured_rationale = ""
    try:
        repair_response = await invoke(
            repair_target, repair_prompt, context_suffix=" structured handoff"
        )
        repair_partial_reason = agent_runtime.response_partial_reason(repair_response)
        if not repair_partial_reason:
            repair_content = message_utils.extract_string_content(
                repair_response.content
            )
            structured_rationale = parse_structured_rationale_candidate(
                repair_content, policy=seat.policy
            )
        if structured_rationale:
            logger.info(
                "debate_structured_rationale_repaired",
                agent=seat.agent_key,
                round=seat.round_num,
                content_length=len(structured_rationale),
            )
        else:
            logger.warning(
                "debate_structured_rationale_unavailable",
                agent=seat.agent_key,
                round=seat.round_num,
                reason=repair_partial_reason or "invalid_repair_shape",
            )
            if runtime_config.developer_debug_active:
                logger.debug(
                    "debate_structured_rationale_invalid_content",
                    agent=seat.agent_key,
                    round=seat.round_num,
                    content_preview=redact_sensitive_text(
                        repair_content, max_chars=2_000
                    ),
                )
    except Exception as exc:
        logger.warning(
            "debate_structured_rationale_unavailable",
            agent=seat.agent_key,
            round=seat.round_num,
            **summarize_exception(exc, operation="debate_structured_rationale_repair"),
        )
    return structured_rationale


def create_researcher_node(
    llm,
    memory: Any | None,
    agent_key: str,
    round_num: int = 1,
    *,
    handoff_policy: DebateReasoningPolicy | None = None,
    fallback_llm: Any | None = None,
    structured_repair_llm: Any | None = None,
) -> Callable:
    """
    Create a researcher node for Bull/Bear debate.
    """
    is_bull = agent_key == "bull_researcher"
    researcher_type = "bull" if is_bull else "bear"
    opponent_type = "bear" if is_bull else "bull"
    policy = handoff_policy or DebateReasoningPolicy(enabled=False, max_rounds=2)
    seat = _ResearcherSeat(
        agent_key=agent_key,
        researcher_type=researcher_type,
        round_num=round_num,
        policy=policy,
        llm=llm,
        fallback_llm=fallback_llm,
        structured_repair_llm=structured_repair_llm,
    )

    async def researcher_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt(agent_key)
        if not agent_prompt:
            logger.error("missing_prompt", agent=agent_key)
            field_name = f"{researcher_type}_round{round_num}"
            return {
                "investment_debate_state": {
                    field_name: f"[SYSTEM]: Error - Missing prompt for {agent_key}.",
                    "count": state.get("investment_debate_state", {}).get("count", 0)
                    + 1,
                }
            }

        debate_state = state.get("investment_debate_state", {})
        if round_num == 1:
            context_section_title = "REPORTS"
            reports = _build_research_report_bundle(state, _ROUND1_REPORT_BUDGETS)
            debate_history = ""
            round_instruction = "Provide your initial argument."
        else:
            context_section_title = "FACTUAL ANCHORS"
            reports = _build_research_report_bundle(state, _ROUND2_ANCHOR_BUDGETS)
            opponent_r1 = debate_state.get(f"{opponent_type}_round1", "")
            own_r1 = debate_state.get(f"{researcher_type}_round1", "")
            debate_history = f"""
=== ROUND 1 ARGUMENTS ===

YOUR ROUND 1 ARGUMENT:
{own_r1}

OPPONENT'S ROUND 1 ARGUMENT (REBUT THIS):
{opponent_r1}

=== END ROUND 1 ===

Now provide your Round 2 rebuttal, addressing the opponent's key points."""
            if policy.consumes_in(round_num):
                opponent_handoff = render_opponent_handoff(
                    opponent_role=opponent_type,
                    opponent=cast(
                        dict[str, str],
                        debate_state.get(f"{opponent_type}_round1_handoff", {}),
                    ),
                    telemetry=cast(
                        dict[str, Any], debate_state.get("handoff_telemetry", {})
                    ),
                )
                if opponent_handoff:
                    debate_history += "\n\n" + format_untrusted_block(
                        opponent_handoff,
                        "OPPONENT ROUND-1 REASONING ADJUNCT",
                        provenance=f"{opponent_type} researcher model output",
                    )
            round_instruction = (
                "Provide your rebuttal to the opponent's Round 1 argument. "
                "Use the factual anchors and round-1 arguments as your basis, "
                "and do not introduce unsupported new facts."
            )

        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)
        company_resolved = state.get("company_name_resolved", True)
        runtime_config = get_runtime_config(settings_config)

        past_insights = ""
        if memory and runtime_config.enable_memory:
            try:
                relevant = await memory.query_similar_situations(
                    f"risks and upside for {ticker}",
                    n_results=3,
                    metadata_filter={"ticker": ticker},
                )
                if relevant:
                    past_insights = (
                        f"\n\nPAST MEMORY INSIGHTS (STRICTLY FOR {ticker}):\n"
                        + "\n".join([result["document"] for result in relevant])
                    )
                else:
                    logger.info("memory_no_exact_match", ticker=ticker)
            except Exception as exc:
                logger.error(
                    "memory_retrieval_failed",
                    ticker=ticker,
                    **summarize_exception(exc, operation="memory_retrieval"),
                )

        lessons_text = ""
        if runtime_config.enable_memory:
            try:
                from src.retrospective import (
                    create_lessons_memory,
                    format_lessons_for_injection,
                )

                lessons_memory = create_lessons_memory()
                sector = support._extract_sector_from_state(state)
                context = support.get_context_from_config(config)
                current_regime = (
                    getattr(context, "macro_regime", None) if context else None
                )
                lessons_text = await format_lessons_for_injection(
                    lessons_memory,
                    ticker,
                    sector,
                    current_regime=current_regime,
                )
                if lessons_text:
                    logger.info(
                        "lessons_injected",
                        agent=agent_key,
                        ticker=ticker,
                        lessons_length=len(lessons_text),
                    )
                else:
                    logger.debug("no_lessons_available", agent=agent_key, ticker=ticker)
            except Exception as exc:
                logger.warning(
                    "lessons_injection_failed",
                    agent=agent_key,
                    **summarize_exception(exc, operation="lessons_injection_failed"),
                )
        else:
            # --no-memory: don't even touch the global lessons_learned
            # collection. Mirrors the per-ticker retrospective gate in
            # `_maybe_run_ticker_retrospective` (src/main.py).
            logger.debug(
                "lessons_injection_skipped_no_memory",
                agent=agent_key,
                ticker=ticker,
            )

        unresolved_warning = (
            "" if company_resolved else f"\n{support._UNRESOLVED_NAME_WARNING}"
        )

        negative_constraint = f"""
CRITICAL INSTRUCTION:
You are analyzing **{ticker} ({company_name})**.{unresolved_warning}
If the provided context or memory contains information about a different company, you MUST IGNORE IT.
Only use data explicitly related to {ticker} ({company_name}).{governance_block(state)}
{downstream_evidence_constraints(state)}
"""

        context_block = ""
        if past_insights:
            context_block = format_untrusted_block(
                past_insights,
                "MEMORY RETRIEVAL",
                provenance=f"ChromaDB collection for {ticker}",
            )
        if lessons_text:
            wrapped_lessons = format_untrusted_block(
                lessons_text,
                "RETROSPECTIVE LESSONS",
                provenance="global lessons_learned collection",
            )
            context_block += (
                f"\n\n{wrapped_lessons}" if context_block else wrapped_lessons
            )
        if agent_key == "bear_researcher":
            macro_context = support.macro_section_for(
                config,
                prefix="",
            )
            if macro_context:
                context_block += (
                    f"\n\n{macro_context}" if context_block else macro_context
                )

        canonical_prompt = (
            f"{agent_prompt.system_message}\n{negative_constraint}\n\n{context_section_title}:\n"
            f"{reports}\n{context_block}\n\nDEBATE CONTEXT:\n{debate_history}\n\n"
            f"{round_instruction}"
        )
        prompt = canonical_prompt + structured_rationale_addendum(policy, round_num)

        try:
            resolved = await _resolve_researcher_output(
                seat,
                agent_name=agent_prompt.agent_name,
                prompt=prompt,
                canonical_prompt=canonical_prompt,
                runtime_config=runtime_config,
            )
            response = resolved.response
            content_str = resolved.canonical
            structured_rationale = resolved.structured_rationale
            native_reasoning = resolved.native_reasoning

            from src.utils import detect_truncation

            trunc_info = detect_truncation(content_str, agent=agent_key)
            log_truncation_diagnostic(
                agent_key=agent_key,
                ticker=ticker,
                runnable=resolved.runnable,
                response=response,
                content=content_str,
                trunc_info=trunc_info,
            )
            log_output_diagnostics(
                agent_key=agent_key,
                ticker=ticker,
                runnable=resolved.runnable,
                response=response,
                content=content_str,
                truncated=trunc_info["truncated"],
                validation=None,
            )
            argument = f"{agent_prompt.agent_name} (Round {round_num}): {content_str}"
            field_name = f"{researcher_type}_round{round_num}"

            logger.info(
                "researcher_completed",
                agent=agent_key,
                round=round_num,
                field=field_name,
                content_length=len(content_str),
            )

            debate_update: dict[str, Any] = {field_name: argument}
            if policy.emits_in(round_num):
                debate_update[f"{researcher_type}_round1_handoff"] = {
                    "structured": structured_rationale,
                    "native": native_reasoning,
                }
            return {"investment_debate_state": debate_update}
        except Exception as exc:
            logger.error(
                "researcher_error",
                agent=agent_key,
                round=round_num,
                **summarize_exception(exc, operation="researcher_error"),
            )
            field_name = f"{researcher_type}_round{round_num}"
            return {
                "investment_debate_state": {
                    field_name: (
                        f"[SYSTEM ERROR]: {agent_key} R{round_num} failed - {str(exc)}"
                    ),
                }
            }

    return researcher_node


def create_research_manager_node(
    llm,
    memory: Any | None,
    strict_mode: bool = False,
    *,
    handoff_policy: DebateReasoningPolicy | None = None,
) -> Callable:
    policy = handoff_policy or DebateReasoningPolicy(enabled=False, max_rounds=2)

    async def research_manager_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("research_manager")
        if not agent_prompt:
            return {"investment_plan": "Error: Missing prompt"}

        debate = state.get("investment_debate_state", {})
        value_trap = get_valid_artifact_content(state, "value_trap_report") or "N/A"
        field_sources = support.extract_field_sources_from_messages(
            state.get("messages", [])
        )
        attribution_note = ""
        if field_sources:
            sources_used = sorted(set(field_sources.values()))
            attribution_note = (
                "\n\n### DATA PROVENANCE NOTE\n"
                f"Fundamentals sourced from: {', '.join(sources_used)}. "
                "News may reflect more recent periods (e.g., Q3 headlines vs TTM API data). "
                "When Bull/Bear cite conflicting figures, check if they reference different time periods."
            )

        market_report = get_valid_artifact_content(state, "market_report") or "N/A"
        sentiment_report = (
            get_valid_artifact_content(state, "sentiment_report") or "N/A"
        )
        news_report = get_valid_artifact_content(state, "news_report") or "N/A"
        fundamentals_report = (
            get_valid_artifact_content(state, "fundamentals_report") or "N/A"
        )
        bull_history = debate.get("bull_history", "N/A")
        bear_history = debate.get("bear_history", "N/A")
        all_reports = f"""MARKET ANALYST REPORT:\n{support.summarize_for_pm(market_report, "market", 1800) if market_report != "N/A" else "N/A"}\n\nSENTIMENT ANALYST REPORT:\n{support.summarize_for_pm(sentiment_report, "sentiment", 1200) if sentiment_report != "N/A" else "N/A"}\n\nNEWS ANALYST REPORT:\n{support.summarize_for_pm(news_report, "news", 1800) if news_report != "N/A" else "N/A"}\n\nFUNDAMENTALS ANALYST REPORT:\n{support.summarize_for_pm(fundamentals_report, "fundamentals", 4000) if fundamentals_report != "N/A" else "N/A"}{attribution_note}\n\nVALUE TRAP ANALYSIS:\n{support.summarize_for_pm(value_trap, "value_trap", 2200) if value_trap != "N/A" else "N/A"}\n\nBULL RESEARCHER:\n{support.summarize_for_pm(bull_history, "research", 2500) if bull_history != "N/A" else "N/A"}\n\nBEAR RESEARCHER:\n{support.summarize_for_pm(bear_history, "research", 2500) if bear_history != "N/A" else "N/A"}"""
        if policy.active:
            balanced_handoffs = render_balanced_handoffs(
                bull=debate.get("bull_round1_handoff", {}),
                bear=debate.get("bear_round1_handoff", {}),
                telemetry=debate.get("handoff_telemetry", {}),
            )
            if balanced_handoffs:
                all_reports += "\n\n" + format_untrusted_block(
                    balanced_handoffs,
                    "BALANCED ROUND-1 REASONING ADJUNCTS",
                    provenance="Bull and Bear researcher model outputs",
                )
        system_msg = agent_prompt.system_message
        if strict_mode:
            system_msg += _STRICT_RM_ADDENDUM

        # Route retrospective lessons into the synthesis agent too: the recurring
        # defects (period-mixing, undiscovered overclaim) originate at RM synthesis,
        # which previously never saw lessons (only Bull/Bear did). RM receives the
        # researchers' *outputs* — not their lesson prompts — so this is not a
        # double-injection. Reuse the existing format_lessons_for_injection path.
        ticker = state.get("company_of_interest", "UNKNOWN")
        runtime_config = get_runtime_config(settings_config)
        lessons_block = ""
        if runtime_config.enable_memory:
            try:
                from src.retrospective import (
                    create_lessons_memory,
                    format_lessons_for_injection,
                )

                lessons_memory = create_lessons_memory()
                sector = support._extract_sector_from_state(state)
                rm_context = support.get_context_from_config(config)
                current_regime = (
                    getattr(rm_context, "macro_regime", None) if rm_context else None
                )
                lessons_text = await format_lessons_for_injection(
                    lessons_memory,
                    ticker,
                    sector,
                    current_regime=current_regime,
                )
                if lessons_text:
                    lessons_block = "\n\n" + format_untrusted_block(
                        lessons_text,
                        "RETROSPECTIVE LESSONS",
                        provenance="global lessons_learned collection",
                    )
                    logger.info(
                        "lessons_injected",
                        agent="research_manager",
                        ticker=ticker,
                        lessons_length=len(lessons_text),
                    )
            except Exception as exc:
                logger.warning(
                    "lessons_injection_failed",
                    agent="research_manager",
                    **summarize_exception(exc, operation="lessons_injection_failed"),
                )

        evidence_constraints = downstream_evidence_constraints(state)
        prompt = f"{system_msg}{governance_block(state)}{evidence_constraints}{lessons_block}\n\n{all_reports}\n\nProvide Investment Plan."

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

            trunc_info = detect_truncation(content_str, agent="research_manager")
            log_truncation_diagnostic(
                agent_key="research_manager",
                ticker=state.get("company_of_interest", "UNKNOWN"),
                runnable=llm,
                response=response,
                content=content_str,
                trunc_info=trunc_info,
            )

            validation = validate_required_output("research_manager", content_str)
            log_output_diagnostics(
                agent_key="research_manager",
                ticker=state.get("company_of_interest", "UNKNOWN"),
                runnable=llm,
                response=response,
                content=content_str,
                truncated=trunc_info["truncated"],
                validation=validation,
            )
            if should_fail_closed(
                "research_manager",
                validation=validation,
                truncated=trunc_info["truncated"],
                content=content_str,
            ):
                logger.error(
                    "research_manager_invalid_structure",
                    ticker=state.get("company_of_interest", "UNKNOWN"),
                    missing_sections=validation["missing"],
                )
                return failure_artifact(
                    "investment_plan",
                    "Research Manager output missing required structure",
                    provider=support.infer_provider_name(llm),
                    fallback_content=content_str,
                )

            return success_artifact(
                "investment_plan",
                cap_state_value(content_str, "investment_plan"),
                provider=support.infer_provider_name(llm),
            )
        except Exception as exc:
            return failure_artifact(
                "investment_plan",
                exc,
                provider=support.infer_provider_name(llm),
            )

    return research_manager_node
