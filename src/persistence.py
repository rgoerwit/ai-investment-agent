"""Persistence helpers for analysis artifacts and retrospective records."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.config import config

logger = structlog.get_logger(__name__)


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
    consultant_finished = bool(consultant_status.get("complete"))
    auditor_finished = bool(auditor_status.get("complete"))
    providers_used = _collect_used_providers()

    return {
        "quick_mode": quick_mode,
        "quick_model": config.quick_think_llm,
        "deep_model": config.deep_think_llm,
        "provider_preflight": provider_preflight or {},
        "pre_screening_result": result.get("pre_screening_result", ""),
        "debate_rounds": result.get("investment_debate_state", {}).get("count", 0),
        "consultant_completed": consultant_finished,
        "auditor_completed": auditor_finished,
        "consultant_finished": consultant_finished,
        "auditor_finished": auditor_finished,
        "consultant_successful": bool(consultant_status.get("ok")),
        "auditor_successful": bool(auditor_status.get("ok")),
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


def save_results_to_file(
    result: dict,
    ticker: str,
    quick_mode: bool = False,
    *,
    results_dir: Path | str | None = None,
    trace_id: str | None = None,
    logger_obj=logger,
) -> Path:
    """Save analysis results to a JSON file in the results directory."""
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
    if config.enable_memory:
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
            "timestamp": timestamp,
            "analysis_date": datetime.now().isoformat(),
            "environment": config.environment,
            "quick_model": config.quick_think_llm,
            "deep_model": config.deep_think_llm,
            "memory_enabled": config.enable_memory,
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
        "reports": {
            "market_report": result.get("market_report", ""),
            "sentiment_report": result.get("sentiment_report", ""),
            "news_report": result.get("news_report", ""),
            "fundamentals_report": result.get("fundamentals_report", ""),
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
        "pre_screening_result": result.get("pre_screening_result", ""),
        "run_summary": result.get("run_summary", {}),
        "analysis_validity": result.get("analysis_validity", {}),
        "artifact_statuses": result.get("artifact_statuses", {}),
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

        save_data["prediction_snapshot"] = extract_snapshot(
            result,
            ticker,
            quick_mode,
            trace_id=trace_id,
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
        f"Results saved to {filepath} ({len(prompts_used)} prompts tracked, {len(custom_prompts_loaded)} custom)"
    )
    if token_stats["total_calls"] > 0:
        logger_obj.info(
            f"Token usage tracked: {token_stats['total_calls']} LLM calls, "
            f"{token_stats['total_tokens']:,} total tokens, "
            f"${token_stats['total_cost_usd']:.4f} projected cost (paid tier) - "
            f"saved to {filepath}"
        )
    return filepath


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
            logger_obj=logger_obj,
        )
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
    """Persist non-BUY verdicts as retrospective rejection records."""
    from src.error_safety import summarize_exception

    try:
        from src.retrospective import (
            create_lessons_memory,
            extract_snapshot,
            save_rejection_record,
        )

        snapshot = extract_snapshot(
            result,
            args.ticker,
            is_quick_mode=args.quick,
            trace_id=trace_id,
        )
        verdict = snapshot.get("verdict", "")
        if verdict and verdict != "BUY":
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
