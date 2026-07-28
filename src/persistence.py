"""Persistence helpers for analysis artifacts and retrospective records."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.agents.output_limits import cap_state_value
from src.agents.pm_inputs import (
    DIRECT_PM_INPUTS,
    GOVERNANCE_CARD_FIELD,
    RISK_DEBATE_FIELD,
    governance_card_present,
    risk_debate_content,
)
from src.agents.verdict_policy import DNI_REVIEW_CANDIDATE_MARKER
from src.config import config
from src.runtime_config import get_runtime_config
from src.sector_normalization import normalize_sector_label

logger = structlog.get_logger(__name__)

_SOURCE_ARTIFACT_MAX_CHARS = 50_000


# Maps each saved-JSON artifact field to its originating graph agent and the
# TokenTrackingCallback display name(s) used in src/graph/components.py.
# A single artifact can have multiple contributing token-agents (e.g.,
# investment_plan is synthesized from research_manager but pulls work done by
# bull/bear researchers; risk_debate_state aggregates three risk analysts).
# The Stage 1 AST drift test (tests/test_agent_attribution.py) verifies
# every name listed here appears at a tracked_callbacks(...) call site.
_ARTIFACT_AGENT_MAP: list[tuple[str, str, tuple[str, ...]]] = [
    ("market_report", "market_analyst", ("Market Analyst",)),
    ("sentiment_report", "sentiment_analyst", ("Sentiment Analyst",)),
    ("news_report", "news_analyst", ("News Analyst",)),
    ("raw_fundamentals_data", "junior_fundamentals", ("Junior Fundamentals Analyst",)),
    (
        "foreign_language_report",
        "foreign_language_analyst",
        ("Foreign Language Analyst",),
    ),
    ("legal_report", "legal_counsel", ("Legal Counsel",)),
    ("fundamentals_report", "senior_fundamentals", ("Fundamentals Analyst",)),
    ("value_trap_report", "value_trap_detector", ("Value Trap Detector",)),
    (GOVERNANCE_CARD_FIELD, "financial_health_validator", ()),
    (
        "auditor_report",
        "global_forensic_auditor",
        ("Global Forensic Auditor", "Global Forensic Auditor Escalation"),
    ),
    (
        "apac_regional_report",
        "apac_regional_specialist",
        ("APAC Regional Specialist", "APAC Regional Specialist Direct Retry"),
    ),
    (
        "investment_plan",
        "research_manager",
        ("Research Manager", "Bull Researcher", "Bear Researcher"),
    ),
    ("valuation_params", "valuation_calculator", ("Valuation Calculator",)),
    ("consultant_review", "consultant", ("Consultant",)),
    ("trader_investment_plan", "trader", ("Trader",)),
    (
        "risk_debate_state",
        "risk_analysts",
        ("Risky Analyst", "Safe Analyst", "Neutral Analyst"),
    ),
    ("final_trade_decision", "portfolio_manager", ("Portfolio Manager",)),
]


def _aggregate_token_usage(token_agents: dict, names: tuple[str, ...]) -> dict | None:
    contributors = [name for name in names if name in token_agents]
    rows = [token_agents[name] for name in contributors]
    if not rows:
        return None
    return {
        "calls": sum(int(r.get("calls", 0) or 0) for r in rows),
        "prompt_tokens": sum(int(r.get("prompt_tokens", 0) or 0) for r in rows),
        "completion_tokens": sum(int(r.get("completion_tokens", 0) or 0) for r in rows),
        "total_tokens": sum(int(r.get("total_tokens", 0) or 0) for r in rows),
        "cost_usd": round(sum(float(r.get("cost_usd", 0.0) or 0.0) for r in rows), 6),
        "contributors": contributors,
    }


def _persisted_source_artifact(value: Any, field: str) -> str:
    text = value if isinstance(value, str) else ""
    return cap_state_value(
        text,
        f"persistence:{field}",
        max_chars=_SOURCE_ARTIFACT_MAX_CHARS,
    )


def _build_agent_attribution(result: dict, token_agents: dict) -> dict:
    """Build per-artifact attribution: agent, validity, char count, token usage.

    Validity is taken from the project's artifact_statuses pipeline via
    get_valid_artifact_content() — failure-artifact stubs and "N/A" content
    do not count as a contributing input. The synthetic risk_debate_state
    field has no artifact_status entry, so it uses a non-empty heuristic.
    """
    from src.runtime_diagnostics import get_valid_artifact_content

    artifacts: dict[str, dict[str, Any]] = {}
    for field, agent_slug, token_names in _ARTIFACT_AGENT_MAP:
        if field == RISK_DEBATE_FIELD:
            content = risk_debate_content(result)
            valid = bool(content)
        elif field == GOVERNANCE_CARD_FIELD:
            valid = governance_card_present(result)
            content = "<entity_governance_card>" if valid else ""
        else:
            content = get_valid_artifact_content(result, field) or ""
            valid = bool(content)

        artifacts[field] = {
            "agent": agent_slug,
            "token_agents": [name for name in token_names if name in token_agents],
            "artifact_field": field,
            "present": valid,
            "char_count": len(content) if isinstance(content, str) else 0,
            "token_usage": _aggregate_token_usage(token_agents, token_names),
            "direct_pm_input": field in DIRECT_PM_INPUTS,
        }
    return artifacts


def _build_quick_consultant_summary(
    result: dict[str, Any], tracker_stats: dict[str, Any]
) -> dict[str, object]:
    artifact_statuses = result.get("artifact_statuses", {}) or {}
    consultant_status = artifact_statuses.get("consultant_review") or {}
    attempts = [
        attempt
        for attempt in tracker_stats.get("call_attempts", []) or []
        if "consultant" in str(attempt.get("agent_name", "")).lower()
    ]
    agent_rows = tracker_stats.get("agents", {}) or {}
    token_rows = [
        row for name, row in agent_rows.items() if "consultant" in str(name).lower()
    ]
    tokens = sum(int(row.get("total_tokens") or 0) for row in token_rows)
    elapsed = sum(float(attempt.get("elapsed_seconds") or 0.0) for attempt in attempts)
    timeout = any(attempt.get("failure_kind") == "timeout" for attempt in attempts)

    if consultant_status.get("complete"):
        status = "ok" if consultant_status.get("ok") else "failed"
    elif attempts:
        status = "attempted"
    else:
        status = "not_run"

    return {
        "status": status,
        "elapsed_seconds": round(elapsed, 4),
        "tokens": tokens,
        "attempts": len(attempts),
        "timeout": timeout,
        "tool_failures": int(result.get("consultant_tool_failures") or 0),
        "profile": result.get("consultant_quick_profile") or "unknown",
    }


def _derive_consultant_verdict(consultant_status: dict[str, Any]) -> str:
    """Distinguish 'consultant ran' from 'consultant approved'.

    Mirrors the auditor's ran-vs-clean distinction. Reuses the canonical
    ``parse_consultant_conditions`` (the same parser the PM uses to raise
    ``CONSULTANT_*`` flags) on the stored review content rather than inventing a
    second parse, so the verdict stays consistent with red-flag detection.

    Returns one of: ``REJECTED`` (hard stop), ``MAJOR_CONCERNS`` (mandate breach
    or major concerns), ``CONDITIONAL`` (conditional approval), ``CLEAN``
    (approved), ``SKIPPED`` (intentionally bypassed by the quick-mode gate),
    ``UNPARSED`` (ran ok but verdict unclassifiable), ``ERROR`` (ran but failed),
    ``NOT_RUN`` (absent).
    """
    if not consultant_status:
        return "NOT_RUN"
    if not consultant_status.get("ok"):
        return "ERROR" if consultant_status.get("complete") else "NOT_RUN"

    # The quick-mode gate bypass writes a completed, ok artifact whose content is
    # the SKIPPED_BY_GATE sentinel (provider="bypass"; see routing.py). That is an
    # intentional cost-saving skip, not a garbled review — surface it as SKIPPED so
    # run_summary and the memo/source-confidence renderers don't misread the bypass
    # as an unparseable verdict (which wrongly downgrades cross-check confidence).
    content = consultant_status.get("content") or ""
    if consultant_status.get("provider") == "bypass" or (
        isinstance(content, str) and content.startswith("SKIPPED_BY_GATE")
    ):
        return "SKIPPED"

    from src.validators.supplemental_extractors import parse_consultant_conditions

    conditions = parse_consultant_conditions(consultant_status.get("content") or "")
    if conditions.get("has_hard_stop"):
        return "REJECTED"
    if conditions.get("has_mandate_breach"):
        return "MAJOR_CONCERNS"
    verdict = conditions.get("verdict")
    if verdict == "MAJOR_CONCERNS":
        return "MAJOR_CONCERNS"
    if verdict == "CONDITIONAL_APPROVAL":
        return "CONDITIONAL"
    if verdict == "APPROVED":
        return "CLEAN"
    return "UNPARSED"


def build_run_summary(
    result: dict,
    *,
    quick_mode: bool,
    article_requested: bool,
    provider_preflight: dict[str, dict[str, str]] | None = None,
) -> dict[str, object]:
    """Build a compact summary for saved artifacts and end-of-run logs."""
    from langchain_core.messages import ToolMessage

    from src.token_tracker import get_tracker

    def _tool_message_failed(content: object) -> bool:
        if not isinstance(content, str):
            return False
        text = content.strip()
        if not text:
            return False
        if text.startswith(
            (
                "TOOL_ERROR:",
                "TOOL_BLOCKED:",
                "FETCH_FAILED:",
                "SEARCH_FAILED:",
                "INVALID_URL:",
            )
        ):
            return True
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return False
        return isinstance(payload, dict) and bool(payload.get("error"))

    def _collect_used_providers() -> list[str]:
        providers: set[str] = set()
        configured = str(config.llm_provider or "").strip()
        if configured:
            providers.add(configured)
        artifact_statuses = result.get("artifact_statuses", {}) or {}
        for status in artifact_statuses.values():
            provider = str((status or {}).get("provider") or "").strip()
            if provider:
                providers.add(provider)
        return sorted(providers)

    manual_tool_failures = sum(
        value
        for key, value in result.items()
        if key.endswith("_tool_failures") and isinstance(value, int) and value > 0
    )

    tracker_stats = get_tracker().get_total_stats()
    messages = result.get("messages", []) or []
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    tool_failures = manual_tool_failures + sum(
        1
        for msg in tool_messages
        if getattr(msg, "status", None) == "error" or _tool_message_failed(msg.content)
    )
    artifact_statuses = result.get("artifact_statuses", {}) or {}
    consultant_status = artifact_statuses.get("consultant_review") or {}
    auditor_status = artifact_statuses.get("auditor_report") or {}
    apac_status = artifact_statuses.get("apac_regional_report") or {}
    consultant_finished = bool(consultant_status.get("complete"))
    auditor_finished = bool(auditor_status.get("complete"))
    apac_finished = bool(apac_status.get("complete"))
    providers_used = _collect_used_providers()
    runtime_config = get_runtime_config(config)

    # "Successful" must mean the auditor completed a verified audit, not merely that it
    # emitted well-formed prose. Caveated statuses are no data (INSUFFICIENT_DATA/
    # UNAVAILABLE/N/A) or an incomplete/unverified audit (PARTIAL_DATA). A missing or
    # unparseable STATUS line (parse → None) is *also* not a clean pass — we must not
    # let malformed output read as HIGH confidence. So success requires an explicitly
    # parsed, non-caveated status; everything else reads as "ran with caveats" (MEDIUM).
    from src.graph.routing import parse_auditor_status

    _AUDITOR_CAVEATED_STATUSES = {
        "INSUFFICIENT_DATA",
        "UNAVAILABLE",
        "N/A",
        "PARTIAL_DATA",
    }
    auditor_report_status = parse_auditor_status(auditor_status.get("content"))
    auditor_successful = (
        bool(auditor_status.get("ok"))
        and auditor_report_status is not None
        and auditor_report_status not in _AUDITOR_CAVEATED_STATUSES
    )

    # The consultant gets the same "ran vs approved" distinction the auditor has:
    # `consultant_successful` (= bare `ok`) only means it returned a parseable
    # review, NOT that it approved. Derive the approval verdict once, here, by
    # reusing the canonical consultant-condition parser on the stored review
    # content — the same parser the PM uses to raise CONSULTANT_* flags — so the
    # memo/source-confidence renderers can branch on a single ready value.
    consultant_verdict = _derive_consultant_verdict(consultant_status)

    summary = {
        "quick_mode": quick_mode,
        "quick_model": runtime_config.quick_think_llm,
        "deep_model": runtime_config.deep_think_llm,
        "provider_preflight": provider_preflight or {},
        "pre_screening_result": result.get("pre_screening_result", ""),
        # `count` tallies debate *turns* (one Bull + one Bear per round → even), so
        # actual rounds = count // 2 (quick=1, full=2). `debate_turns` keeps the raw value.
        "debate_rounds": result.get("investment_debate_state", {}).get("count", 0) // 2,
        "debate_turns": result.get("investment_debate_state", {}).get("count", 0),
        # Honest flag: reflects whether the quick-mode qualification note was actually
        # appended to the PM text (marker presence), never a recomputed `quick and BUY`
        # that would lie if the hook no-ops on PM-block parse drift.
        "verdict_qualified_by_quick_mode": "QUICK-MODE QUALIFICATION"
        in (result.get("final_trade_decision") or ""),
        # Honest flag: marker presence of the weak-asymmetry BUY caveat, same
        # rationale as verdict_qualified_by_quick_mode above.
        "verdict_weak_valuation_asymmetry": "WEAK VALUATION ASYMMETRY"
        in (result.get("final_trade_decision") or ""),
        # Honest flag: marker presence of the gate-passing-DNI review-candidate
        # note (shared constant, so the key cannot drift from the note wording).
        "verdict_dni_review_candidate": DNI_REVIEW_CANDIDATE_MARKER
        in (result.get("final_trade_decision") or ""),
        "consultant_completed": consultant_finished,
        "auditor_completed": auditor_finished,
        "consultant_finished": consultant_finished,
        "auditor_finished": auditor_finished,
        "consultant_successful": bool(consultant_status.get("ok")),
        "consultant_verdict": consultant_verdict,
        "auditor_successful": auditor_successful,
        "auditor_status": auditor_report_status,
        "apac_specialist_completed": apac_finished,
        "apac_specialist_successful": bool(apac_status.get("ok")),
        "apac_specialist_status": (
            "not_run"
            if not apac_status
            else "ok"
            if apac_status.get("ok")
            else "failed"
        ),
        "article_requested": article_requested,
        "llm_attempts": tracker_stats["total_calls"] + tracker_stats["failed_attempts"],
        "llm_failures": tracker_stats["failed_attempts"],
        "tool_calls": len(tool_messages),
        "tool_failures": tool_failures,
        "llm_providers_used": providers_used,
        "llm_provider": providers_used[0]
        if len(providers_used) == 1
        else "multi-provider",
        "macro_context_status": result.get("macro_context_status", "failed"),
        "macro_context_region": result.get("macro_context_region", "GLOBAL"),
        "macro_context_report_present": bool(result.get("macro_context_report")),
        "macro_context_injected_into_news": bool(
            result.get("macro_context_injected_into_news", False)
        ),
        "publishable": result.get("analysis_validity", {}).get("publishable", False),
        "required_failures": sorted(
            (result.get("analysis_validity", {}) or {})
            .get("required_failures", {})
            .keys()
        ),
        "optional_failures": sorted(
            (result.get("analysis_validity", {}) or {})
            .get("optional_failures", {})
            .keys()
        ),
    }
    if quick_mode:
        summary["quick_consultant"] = _build_quick_consultant_summary(
            result, tracker_stats
        )
    return summary


def attach_run_summary(
    result: dict[str, Any],
    *,
    quick_mode: bool,
    article_requested: bool = False,
    provider_preflight: dict[str, dict[str, str]] | None = None,
) -> None:
    """Attach ``analysis_validity`` + the compact ``run_summary`` onto ``result``.

    Single source of truth for run-summary enrichment, shared by the main analyzer
    (``src.main._attach_run_summary``) and the portfolio_manager refresh save path.
    Without this, refresh-saved analyses carried a bare ``run_summary`` missing the
    macro-context provenance fields, which made the macro self-check in
    ``save_results_to_file`` fire a spurious ``analysis_artifact_macro_mismatch``
    on every refreshed ticker.

    ``build_analysis_validity`` is imported lazily to avoid a module-level import
    cycle (``runtime_diagnostics`` imports from this module at call time).
    """
    from src.runtime_diagnostics import build_analysis_validity

    result["analysis_validity"] = build_analysis_validity(result)
    result["run_summary"] = build_run_summary(
        result,
        quick_mode=quick_mode,
        article_requested=article_requested,
        provider_preflight=provider_preflight or {},
    )


def _normalize_macro_context_metadata(
    result: dict[str, Any],
    *,
    cache_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Return one normalized macro-context metadata view for persistence."""
    run_summary = result.get("run_summary", {}) or {}
    status = result.get("macro_context_status")
    region = result.get("macro_context_region")
    report_present = result.get("macro_context_report")

    if status is None:
        status = run_summary.get("macro_context_status", "failed")
    if region is None:
        region = run_summary.get("macro_context_region", "GLOBAL")
    if report_present is None:
        report_present = run_summary.get("macro_context_report_present", False)

    return {
        "status": status,
        "region": region,
        "report_present": bool(report_present),
        "injected_into_news": bool(
            result.get("macro_context_injected_into_news", False)
        ),
        "llm_invoked": bool(result.get("macro_context_llm_invoked", False)),
        "generated_at": result.get("macro_context_generated_at"),
        "cache_dir": str(
            Path(cache_dir)
            if cache_dir is not None
            else Path(config.results_dir) / ".macro_context_cache"
        ),
    }


def _normalize_prediction_snapshot(
    snapshot: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Canonicalize snapshot fields that feed long-lived downstream workflows."""
    if snapshot is None:
        return None
    normalized = dict(snapshot)
    if "sector" in normalized:
        normalized["sector"] = normalize_sector_label(normalized.get("sector"))
    return normalized


def save_results_to_file(
    result: dict,
    ticker: str,
    quick_mode: bool = False,
    *,
    results_dir: Path | str | None = None,
    trace_id: str | None = None,
    strict_mode: bool = False,
    logger_obj=logger,
) -> Path:
    """Save analysis results to a JSON file in the results directory.

    `strict_mode` is recorded in the prediction_snapshot so retrospectives
    can weight strict-mode rejections differently from normal-mode ones
    (strict mode rejects valid REIT/PFIC/VIE candidates at the gate).
    """
    from src.error_safety import summarize_exception
    from src.memory import get_ticker_memory_stats
    from src.prompts import get_all_prompts

    results_dir = (
        Path(results_dir) if results_dir is not None else Path(config.results_dir)
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    previous_dir_mtime_ns = (
        results_dir.stat().st_mtime_ns if results_dir.exists() else None
    )
    analysis_file_count_before_save = sum(
        1 for candidate in results_dir.glob("*_analysis.json") if candidate.is_file()
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_{timestamp}_analysis.json"
    filepath = results_dir / filename

    prompts_used = result.get("prompts_used", {})
    all_prompts = get_all_prompts()
    available_prompts = {
        key: {
            "agent_name": prompt.agent_name,
            "version": prompt.version,
            "category": prompt.category,
            "requires_tools": prompt.requires_tools,
        }
        for key, prompt in all_prompts.items()
    }

    prompts_dir = Path("./prompts")
    custom_prompts_loaded = []
    if prompts_dir.exists():
        for json_file in prompts_dir.glob("*.json"):
            custom_prompts_loaded.append(json_file.stem)

    memory_stats = {}
    runtime_config = get_runtime_config(config)
    if runtime_config.enable_memory:
        try:
            memory_stats = get_ticker_memory_stats(ticker)
        except Exception as exc:
            logger_obj.warning(
                "memory_stats_unavailable",
                **summarize_exception(exc, operation="save memory stats"),
            )

    from src.token_tracker import get_tracker

    tracker = get_tracker()
    token_stats = tracker.get_total_stats()

    save_data = {
        "metadata": {
            "ticker": ticker,
            "company_name": result.get("company_name"),
            "company_name_resolved": bool(result.get("company_name_resolved", False)),
            "timestamp": timestamp,
            "analysis_date": datetime.now().isoformat(),
            "environment": config.environment,
            "quick_model": runtime_config.quick_think_llm,
            "deep_model": runtime_config.deep_think_llm,
            "memory_enabled": runtime_config.enable_memory,
            "online_tools_enabled": config.online_tools,
            "llm_provider": (
                (result.get("run_summary", {}) or {}).get("llm_provider")
                or config.llm_provider
            ),
            "llm_providers_used": (
                (result.get("run_summary", {}) or {}).get("llm_providers_used")
                or [config.llm_provider]
            ),
        },
        "token_usage": token_stats,
        "macro_context": _normalize_macro_context_metadata(
            result,
            cache_dir=results_dir / ".macro_context_cache",
        ),
        "macro_regime_block": result.get("macro_regime_block") or {},
        "macro_regime_raw": result.get("macro_regime_raw", ""),
        "prompts_metadata": {
            "prompts_used": prompts_used,
            "available_prompts": available_prompts,
            "custom_prompts_loaded": custom_prompts_loaded,
            "prompts_directory": str(prompts_dir),
            "total_agents": len(prompts_used),
            "note": (
                "system_message field contains the actual prompt text used by each "
                "graph agent or pre-graph helper"
            ),
        },
        "memory_statistics": memory_stats,
        "entity_governance_card": result.get("entity_governance_card") or None,
        "auditor_budget": result.get("auditor_budget") or None,
        "source_artifacts": {
            "management_guidance_evidence": _persisted_source_artifact(
                result.get("management_guidance_evidence"),
                "management_guidance_evidence",
            ),
            "raw_fundamentals_data": _persisted_source_artifact(
                result.get("raw_fundamentals_data"),
                "raw_fundamentals_data",
            ),
            "foreign_language_report": _persisted_source_artifact(
                result.get("foreign_language_report"),
                "foreign_language_report",
            ),
            "legal_report": _persisted_source_artifact(
                result.get("legal_report"),
                "legal_report",
            ),
            "value_trap_report": _persisted_source_artifact(
                result.get("value_trap_report"),
                "value_trap_report",
            ),
        },
        "reports": {
            "market_report": result.get("market_report", ""),
            "sentiment_report": result.get("sentiment_report", ""),
            "news_report": result.get("news_report", ""),
            "fundamentals_report": result.get("fundamentals_report", ""),
            "apac_regional_report": result.get("apac_regional_report", ""),
            # Optional cross-validation artifacts — empty string when the
            # respective agent did not run or produced nothing.
            "auditor_report": result.get("auditor_report", ""),
            "consultant_review": result.get("consultant_review", ""),
            "valuation_params": result.get("valuation_params", ""),
        },
        "investment_analysis": {
            "investment_debate": {
                "bull_history": result.get("investment_debate_state", {}).get(
                    "bull_history", ""
                ),
                "bear_history": result.get("investment_debate_state", {}).get(
                    "bear_history", ""
                ),
                "debate_rounds": result.get("investment_debate_state", {}).get(
                    "count", 0
                ),
            },
            "investment_plan": result.get("investment_plan", ""),
            "trader_plan": result.get("trader_investment_plan", ""),
        },
        "risk_analysis": {
            "risk_debate": {
                "risky_perspective": result.get("risk_debate_state", {}).get(
                    "current_risky_response", ""
                ),
                "safe_perspective": result.get("risk_debate_state", {}).get(
                    "current_safe_response", ""
                ),
                "neutral_perspective": result.get("risk_debate_state", {}).get(
                    "current_neutral_response", ""
                ),
                "debate_rounds": 1,
            }
        },
        "final_decision": {
            "decision": result.get("final_trade_decision", ""),
            "processed_signal": None,
        },
        "red_flags": result.get("red_flags", []),
        "pre_screening_result": result.get("pre_screening_result", ""),
        "run_summary": result.get("run_summary", {}),
        "analysis_validity": result.get("analysis_validity", {}),
        "artifact_statuses": result.get("artifact_statuses", {}),
        "evidence_records": result.get("evidence_records", []),
        "structured_inputs": result.get("structured_inputs", {}),
        "analysis_snapshot": result.get("analysis_snapshot", {}),
        "decision_trace": result.get("decision_trace", {}),
        "agent_attribution": _build_agent_attribution(
            result, (token_stats or {}).get("agents", {}) or {}
        ),
    }

    run_summary = save_data.get("run_summary", {}) or {}
    macro_context_payload = save_data.get("macro_context", {}) or {}
    prompt_records = (save_data.get("prompts_metadata", {}) or {}).get(
        "prompts_used", {}
    ) or {}
    token_agents = (save_data.get("token_usage", {}) or {}).get("agents", {}) or {}
    has_run_summary_macro_fields = all(
        field in run_summary
        for field in (
            "macro_context_status",
            "macro_context_region",
            "macro_context_report_present",
            "macro_context_injected_into_news",
        )
    )
    has_macro_context_block = all(
        field in macro_context_payload
        for field in (
            "status",
            "region",
            "report_present",
            "injected_into_news",
            "llm_invoked",
            "generated_at",
            "cache_dir",
        )
    )
    has_macro_prompt_metadata = "macro_context_analyst" in prompt_records
    has_macro_token_row = "Macro Context Analyst" in token_agents

    logger_obj.info(
        "analysis_artifact_macro_snapshot",
        ticker=ticker,
        has_macro_context_block=has_macro_context_block,
        has_run_summary_macro_fields=has_run_summary_macro_fields,
        has_macro_prompt_metadata=has_macro_prompt_metadata,
        has_macro_token_row=has_macro_token_row,
    )

    macro_expected = bool(
        result.get("macro_context_llm_invoked", False)
        or result.get("macro_context_injected_into_news", False)
        or result.get("macro_context_report")
    )
    macro_mismatch = macro_expected and (
        not has_macro_context_block
        or not has_run_summary_macro_fields
        or (
            result.get("macro_context_llm_invoked", False)
            and not has_macro_prompt_metadata
        )
        or (result.get("macro_context_llm_invoked", False) and not has_macro_token_row)
    )
    if macro_mismatch:
        logger_obj.warning(
            "analysis_artifact_macro_mismatch",
            ticker=ticker,
            macro_expected=macro_expected,
            macro_llm_invoked=bool(result.get("macro_context_llm_invoked", False)),
            macro_context_injected_into_news=bool(
                result.get("macro_context_injected_into_news", False)
            ),
            has_macro_context_block=has_macro_context_block,
            has_run_summary_macro_fields=has_run_summary_macro_fields,
            has_macro_prompt_metadata=has_macro_prompt_metadata,
            has_macro_token_row=has_macro_token_row,
        )

    try:
        from src.retrospective import extract_snapshot

        save_data["prediction_snapshot"] = _normalize_prediction_snapshot(
            extract_snapshot(
                result,
                ticker,
                quick_mode,
                trace_id=trace_id,
                is_strict_mode=strict_mode,
            )
        )
    except Exception as exc:
        logger_obj.warning(
            "snapshot_extraction_failed",
            **summarize_exception(exc, operation="prediction snapshot extraction"),
        )

    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=2)

    try:
        from src.ibkr.analysis_index import (
            _build_analysis_record_from_data,
            load_latest_analyses,
            update_latest_analyses_index,
        )

        record = _build_analysis_record_from_data(filepath, save_data)
        if record is not None:
            updated_index = update_latest_analyses_index(
                results_dir,
                record,
                previous_dir_mtime_ns=previous_dir_mtime_ns,
                analysis_file_count_before_save=analysis_file_count_before_save,
            )
            if not updated_index:
                refreshed = load_latest_analyses(results_dir)
                logger_obj.info(
                    "analysis_index_refreshed_after_save",
                    ticker=ticker,
                    path=str(results_dir),
                    refreshed_count=len(refreshed),
                )
    except Exception as exc:
        logger_obj.debug(
            "analysis_index_update_skipped",
            **summarize_exception(exc, operation="analysis index update"),
        )

    logger_obj.info(
        "results_saved",
        filepath=str(filepath),
        prompts_tracked=len(prompts_used),
        custom_prompts=len(custom_prompts_loaded),
    )
    if token_stats["total_calls"] > 0:
        logger_obj.info(
            "token_usage_tracked",
            llm_calls=token_stats["total_calls"],
            total_tokens=token_stats["total_tokens"],
            projected_cost_usd=round(token_stats["total_cost_usd"], 4),
            filepath=str(filepath),
        )
    return filepath


def patch_saved_sections(
    path: str | Path,
    sections: dict[str, dict[str, Any]],
    *,
    logger_obj=logger,
) -> None:
    """Merge top-level metadata sections into an already-saved analysis JSON.

    Used for post-persistence facts without re-running the full save path.
    Fail-open: a patch failure must never break the run.
    """
    from src.error_safety import summarize_exception

    try:
        filepath = Path(path)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        for section, fields in sections.items():
            data.setdefault(section, {}).update(fields)
        filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        logger_obj.warning(
            "saved_sections_patch_failed",
            **summarize_exception(exc, operation="patching saved analysis sections"),
        )


def patch_saved_run_summary(
    path: str | Path,
    fields: dict[str, Any],
    *,
    logger_obj=logger,
) -> None:
    """Compatibility wrapper for post-persistence ``run_summary`` updates."""
    patch_saved_sections(path, {"run_summary": fields}, logger_obj=logger_obj)


def _persist_analysis_outputs(
    result: dict,
    args: Any,
    *,
    trace_id: str | None = None,
    logger_obj=logger,
    console_obj=None,
    cost_suffix_fn=None,
    error_message_formatter=None,
) -> None:
    """Persist JSON artifacts and rejection records."""
    from src.error_safety import summarize_exception

    if cost_suffix_fn is None:

        def cost_suffix_fn():
            return ""

    if error_message_formatter is None:

        def error_message_formatter(operation, exc):
            return f"Error {type(exc).__name__}"

    try:
        filepath = save_results_to_file(
            result,
            args.ticker,
            quick_mode=args.quick,
            results_dir=Path(config.results_dir),
            trace_id=trace_id,
            strict_mode=getattr(args, "strict", False),
            logger_obj=logger_obj,
        )
        # Internal-only marker (save_data is a curated dict, so this never
        # self-persists): lets post-persistence steps such as the article
        # writer-model stamp patch the saved JSON in place.
        result["_saved_analysis_path"] = str(filepath)
        if not args.quiet and not args.brief and console_obj is not None:
            console_obj.print(
                f"[green]Results saved to:[/green] [cyan]{filepath}[/cyan]{cost_suffix_fn()}"
            )
    except Exception as exc:
        logger_obj.error(
            "results_save_failed",
            **summarize_exception(
                exc,
                operation="saving analysis results",
                provider="unknown",
            ),
            exc_info=True,
        )
        if not args.quiet and not args.brief and console_obj is not None:
            console_obj.print(
                f"\n[yellow]Warning: {error_message_formatter('saving analysis results', exc)}[/yellow]\n"
            )


async def _maybe_save_rejection_record(
    result: dict,
    args: Any,
    *,
    trace_id: str | None = None,
    logger_obj=logger,
) -> None:
    """Persist non-BUY verdicts as retrospective rejection records.

    Honors ``--no-memory`` (runtime config ``enable_memory == False``) — the
    rejection record lives in the same global ``lessons_learned`` ChromaDB
    collection as full retrospective lessons, and skipping memory should
    skip *all* writes to it. The retrospective comparison itself is gated
    in ``src/main.py``; this matches that contract.
    """
    from src.error_safety import summarize_exception

    if not get_runtime_config(config).enable_memory:
        logger_obj.debug(
            "rejection_record_save_skipped_no_memory",
            ticker=getattr(args, "ticker", None),
        )
        return

    try:
        from src.retrospective import (
            create_lessons_memory,
            extract_snapshot,
            save_rejection_record,
        )

        snapshot = _normalize_prediction_snapshot(
            extract_snapshot(
                result,
                args.ticker,
                is_quick_mode=args.quick,
                trace_id=trace_id,
                is_strict_mode=getattr(args, "strict", False),
            )
        )
        verdict = (snapshot or {}).get("verdict", "")
        if snapshot is not None and verdict and verdict != "BUY":
            rejection_memory = create_lessons_memory()
            await save_rejection_record(snapshot, rejection_memory)
    except Exception as exc:
        logger_obj.debug(
            "rejection_record_save_skipped",
            **summarize_exception(
                exc,
                operation="saving rejection record",
                provider="unknown",
            ),
        )
