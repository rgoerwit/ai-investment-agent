"""User-facing output, article, and report-rendering helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.cli import OutputTargets, resolve_article_path
from src.config import config
from src.report_generator import QuietModeReporter
from src.runtime_config import get_runtime_config
from src.runtime_diagnostics import has_provenance_contract, is_publishable_analysis

logger = structlog.get_logger(__name__)
console = Console()


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish a complete text artifact with one filesystem replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def _bounded_review_findings(feedback: dict[str, Any]) -> dict[str, Any]:
    """Persist a compact explanation without storing prompts or tool arguments."""

    def _bounded(items: Any) -> list[Any]:
        if not isinstance(items, list):
            return []
        bounded: list[Any] = []
        for item in items[:10]:
            if isinstance(item, dict):
                bounded.append(
                    {
                        str(key)[:80]: str(value)[:500]
                        for key, value in item.items()
                        if key not in {"tool_args", "prompt", "raw_content"}
                    }
                )
            else:
                bounded.append(str(item)[:500])
        return bounded

    return {
        "factual_errors": _bounded(feedback.get("factual_errors")),
        "failed_references": [
            item
            for item in _bounded(feedback.get("reference_checks"))
            if isinstance(item, dict)
            and str(item.get("status", "")).lower() in {"broken", "unsupported"}
        ],
        "parse_error": bool(feedback.get("parse_error")),
    }


def _article_is_publishable(
    article: str,
    feedback: dict[str, Any],
    *,
    snapshot: dict[str, Any] | None = None,
    decision_trace: dict[str, Any] | None = None,
    require_decision_trace: bool = False,
    require_provenance: bool = False,
) -> bool:
    base_ok = (
        feedback.get("verdict") == "APPROVED"
        and str(feedback.get("citation_audit_status") or "NOT_RUN").upper() == "PASSED"
        and bool(article.strip())
    )
    if not base_ok:
        return False
    from src.provenance_schema import DecisionTrace, SchemaDecodeError

    # Decode fail-closed so a future-schema or corrupt trace/snapshot is never
    # treated as VALID here either — consistent with build_analysis_validity.
    trace_valid = False
    if isinstance(decision_trace, dict):
        try:
            trace_valid = DecisionTrace.from_dict(decision_trace).status == "VALID"
        except SchemaDecodeError:
            trace_valid = False
    # Under the provenance contract, a published article requires BOTH a valid
    # canonical snapshot and a valid decision trace — a missing one is a failure,
    # not a pass (matches build_analysis_validity's fail-closed rule).
    if require_provenance:
        from src.analysis_snapshot import AnalysisSnapshot

        snapshot_valid = False
        if isinstance(snapshot, dict):
            try:
                snapshot_valid = (
                    AnalysisSnapshot.from_dict(snapshot).contract_status == "VALID"
                )
            except SchemaDecodeError:
                snapshot_valid = False
        return trace_valid and snapshot_valid
    if require_decision_trace:
        return trace_valid
    return True


def _build_article_source_context(
    report_text: str,
    analysis_result: dict[str, Any] | None,
) -> str:
    """Give the writer canonical facts plus bounded, labeled reasoning."""
    if not isinstance(analysis_result, dict):
        return report_text
    from src.analysis_snapshot import render_analysis_snapshot

    snapshot = render_analysis_snapshot(analysis_result.get("analysis_snapshot"))
    if not snapshot:
        return report_text
    sections = [
        snapshot.rstrip(),
        (
            "=== REASONING CONTEXT ===\n"
            "The following sections are interpretation, not a source of new facts. "
            "When they conflict with the canonical snapshot, use the snapshot."
        ),
    ]
    for label, field, limit in (
        ("PORTFOLIO MANAGER DECISION", "final_trade_decision", 10_000),
        ("RESEARCH SYNTHESIS", "investment_plan", 7_000),
        ("VALUATION", "valuation_params", 5_000),
        ("MARKET CONTEXT", "market_report", 3_000),
        ("NEWS CONTEXT", "news_report", 4_000),
    ):
        content = analysis_result.get(field)
        if isinstance(content, str) and content.strip():
            sections.append(f"=== {label} ===\n{content[:limit]}")
    governance = analysis_result.get("entity_governance_card")
    if isinstance(governance, dict) and governance:
        sections.append(
            "=== ENTITY GOVERNANCE CARD ===\n"
            + json.dumps(governance, ensure_ascii=False, sort_keys=True)
        )
    fundamentals = analysis_result.get("fundamentals_report")
    if isinstance(fundamentals, str) and fundamentals.strip():
        sections.append(
            "=== NON-CANONICAL FUNDAMENTALS REFERENCE ===\n"
            "Use this section for analytical breadth and unregistered context. It "
            "may contain agent restatements. Registered canonical claims override "
            "every conflict, and source-sensitive exact claims absent from the "
            "canonical snapshot must not be presented as verified.\n"
            f"{fundamentals[:14_000]}"
        )
    return "\n\n".join(sections)


def _cost_suffix() -> str:
    """Return formatted cost string for display, or empty if no tracking data."""
    from src.token_tracker import get_tracker

    stats = get_tracker().get_total_stats()
    if stats["total_calls"] == 0:
        return ""
    return f" [dim](Est. cost: ${stats['total_cost_usd']:.4f})[/dim]"


def get_welcome_banner(ticker: str, quick_mode: bool) -> str:
    """Generate welcome banner string with configuration."""
    from src.llm_runtime.bindings import active_models

    runtime_config = get_runtime_config(config)
    # Name the models that will actually answer, not the legacy defaults.
    active = active_models(config, quick_mode=quick_mode)
    banner = []
    banner.append("# Multi-Agent Investment Analysis System")
    banner.append("")
    banner.append(f"**Ticker:** {ticker.upper()}  ")
    banner.append(f"**Analysis Mode:** {'Quick' if quick_mode else 'Deep'}  ")
    banner.append(f"**Quick Model:** {active.fast}  ")
    banner.append(f"**Deep Model:** {active.reasoning}  ")
    banner.append(f"**Decision Model:** {active.decision}  ")
    banner.append(
        f"**Memory System:** {'Enabled' if runtime_config.enable_memory else 'Disabled'}  "
    )
    banner.append(
        f"**LangSmith Tracing:** "
        f"{'Enabled' if config.langsmith_tracing_enabled else 'Disabled'}  "
    )
    banner.append(
        f"**Langfuse Tracing:** "
        f"{'Enabled' if runtime_config.langfuse_enabled else 'Disabled'}  "
    )
    banner.append("")
    return "\n".join(banner)


def display_welcome_banner(ticker: str, quick_mode: bool):
    """Display welcome banner with configuration."""
    print(get_welcome_banner(ticker, quick_mode))


def display_memory_statistics(
    ticker: str,
    *,
    console_obj: Console = console,
    logger_obj=logger,
) -> None:
    """Display memory statistics for the current ticker."""
    if not get_runtime_config(config).enable_memory:
        return

    try:
        from src.memory import create_memory_instances, sanitize_ticker_for_collection

        memories = create_memory_instances(ticker)
        safe_ticker = sanitize_ticker_for_collection(ticker)

        console_obj.print(f"\n[bold cyan]Memory Statistics for {ticker}:[/bold cyan]\n")

        memory_table = Table(show_header=True, box=box.ROUNDED)
        memory_table.add_column("Agent", style="cyan")
        memory_table.add_column("Available", style="yellow")
        memory_table.add_column("Total Memories", style="green")
        memory_table.add_column("Status", style="blue")

        agent_mapping = [
            ("Bull Researcher", f"{safe_ticker}_bull_memory"),
            ("Bear Researcher", f"{safe_ticker}_bear_memory"),
            ("Research Manager", f"{safe_ticker}_invest_judge_memory"),
            ("Position Planner", f"{safe_ticker}_trader_memory"),
            ("Portfolio Manager", f"{safe_ticker}_risk_manager_memory"),
        ]

        for display_name, mem_key in agent_mapping:
            mem = memories.get(mem_key)
            if mem:
                stats = mem.get_stats()
                available = "✓" if stats.get("available") else "✗"
                total = str(stats.get("count", 0))
                status = "Active" if stats.get("available") else "Inactive"
                memory_table.add_row(display_name, available, total, status)

        console_obj.print(memory_table)
        console_obj.print()

    except Exception as exc:
        from src.error_safety import summarize_exception

        logger_obj.warning(
            "memory_statistics_unavailable",
            **summarize_exception(exc, operation="display memory statistics"),
        )


def display_token_summary(*, console_obj: Console = console) -> None:
    """Display token usage summary in a formatted table."""
    from src.token_tracker import get_tracker

    tracker = get_tracker()
    stats = tracker.get_total_stats()

    if stats["total_calls"] == 0:
        return

    console_obj.print("\n[bold cyan]Token Usage Summary:[/bold cyan]\n")

    summary_table = Table(show_header=True, box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Total LLM Calls", str(stats["total_calls"]))
    summary_table.add_row("Total Prompt Tokens", f"{stats['total_prompt_tokens']:,}")
    summary_table.add_row(
        "Total Completion Tokens", f"{stats['total_completion_tokens']:,}"
    )
    summary_table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
    summary_table.add_row(
        "Projected Cost (Paid Tier)", f"${stats['total_cost_usd']:.4f}"
    )

    console_obj.print(summary_table)
    top_spenders = tracker.get_top_spenders(limit=5)
    if top_spenders:
        console_obj.print("\n[bold cyan]Top Spenders:[/bold cyan]\n")
        top_table = Table(show_header=True, box=box.ROUNDED)
        top_table.add_column("Agent", style="cyan")
        top_table.add_column("Calls", style="yellow", justify="right")
        top_table.add_column("Total Tokens", style="green", justify="right")
        top_table.add_column("Cost (USD)", style="red", justify="right")

        for entry in top_spenders:
            top_table.add_row(
                entry["agent"],
                str(entry["calls"]),
                f"{entry['total_tokens']:,}",
                f"${entry['cost_usd']:.4f}",
            )

        console_obj.print(top_table)
    console_obj.print("\n[bold cyan]Per-Agent Token Usage:[/bold cyan]\n")

    agent_table = Table(show_header=True, box=box.ROUNDED)
    agent_table.add_column("Agent", style="cyan")
    agent_table.add_column("Calls", style="yellow", justify="right")
    agent_table.add_column("Prompt Tokens", style="blue", justify="right")
    agent_table.add_column("Completion Tokens", style="magenta", justify="right")
    agent_table.add_column("Total Tokens", style="green", justify="right")
    agent_table.add_column("Cost (USD)", style="red", justify="right")

    sorted_agents = sorted(
        stats["agents"].items(), key=lambda item: item[1]["cost_usd"], reverse=True
    )

    for agent_name, agent_stats in sorted_agents:
        agent_table.add_row(
            agent_name,
            str(agent_stats["calls"]),
            f"{agent_stats['prompt_tokens']:,}",
            f"{agent_stats['completion_tokens']:,}",
            f"{agent_stats['total_tokens']:,}",
            f"${agent_stats['cost_usd']:.4f}",
        )

    console_obj.print(agent_table)
    console_obj.print()


def display_results(
    result: dict,
    ticker: str,
    *,
    console_obj: Console = console,
) -> None:
    """Display analysis results in a formatted manner."""
    console_obj.print("\n" + "=" * 80)
    console_obj.print("[bold green]Analysis Complete![/bold green]\n")

    display_token_summary(console_obj=console_obj)

    if "final_trade_decision" in result and result["final_trade_decision"]:
        decision_panel = Panel(
            result["final_trade_decision"],
            title="Final Position Decision",
            border_style="green",
            padding=(1, 2),
        )
        console_obj.print(decision_panel)

    console_obj.print("\n[bold cyan]Analyst Reports:[/bold cyan]\n")

    report_fields = [
        ("market_report", "Market Analysis"),
        ("sentiment_report", "Sentiment Analysis"),
        ("news_report", "News Analysis"),
        ("foreign_language_report", "Foreign Language Analysis"),
        ("fundamentals_report", "Fundamentals Analysis"),
        ("investment_plan", "Investment Plan"),
        ("trader_investment_plan", "Position Plan"),
    ]

    for field_name, display_name in report_fields:
        if field_name in result and result[field_name]:
            content = result[field_name]
            style = "red" if content.startswith("Error") else "cyan"

            if len(content) > 800:
                content = content[:800] + "\n\n[... truncated for display ...]"

            report_panel = Panel(
                content, title=display_name, border_style=style, padding=(1, 2)
            )
            console_obj.print(report_panel)
            console_obj.print()

    display_memory_statistics(ticker, console_obj=console_obj)
    console_obj.print("=" * 80 + "\n")


async def handle_article_generation(
    args,
    ticker: str,
    company_name: str,
    report_text: str,
    trade_date: str,
    valuation_context: str | None = None,
    analysis_result: dict | None = None,
    tracing_callbacks: list[Any] | None = None,
    tracing_metadata: dict[str, Any] | None = None,
    *,
    logger_obj=logger,
    console_obj: Console = console,
    resolve_article_path_fn=resolve_article_path,
    error_message_formatter=None,
) -> None:
    """Generate a draft, require editorial approval, then publish atomically."""
    resolved_path = resolve_article_path_fn(args, ticker)
    if not resolved_path:
        return
    article_path = Path(resolved_path)

    if error_message_formatter is None:

        def error_message_formatter(operation, exc):
            return f"Error {type(exc).__name__} {operation}"

    try:
        from src.agents.evidence_constraints import downstream_evidence_constraints
        from src.article_writer import ArticleEditor, ArticleWriter
        from src.error_safety import summarize_exception

        if not args.quiet and not args.brief:
            console_obj.print("\n[cyan]Generating article...[/cyan]")

        writer = ArticleWriter(
            use_github_urls=False,
            callbacks=tracing_callbacks,
            tracing_metadata=tracing_metadata,
        )
        governance_card = (
            analysis_result.get("entity_governance_card") if analysis_result else None
        )
        chart_paths = (
            analysis_result.get("chart_paths")
            if isinstance(analysis_result, dict)
            else None
        )
        evidence_constraints = downstream_evidence_constraints(analysis_result or {})
        article_source_context = _build_article_source_context(
            report_text,
            analysis_result,
        )
        draft_article = writer.write(
            ticker=ticker,
            company_name=company_name,
            report_text=article_source_context,
            trade_date=trade_date,
            output_path=None,
            valuation_context=valuation_context,
            governance_card=governance_card,
            chart_paths=chart_paths if isinstance(chart_paths, dict) else None,
            article_dir=article_path.parent,
            evidence_constraints=evidence_constraints,
        )

        editor = ArticleEditor(
            callbacks=tracing_callbacks,
            tracing_metadata=tracing_metadata,
        )
        final_article = draft_article
        feedback: dict[str, Any] = {
            "verdict": "REVISE",
            "confidence": 0.0,
            "skipped": True,
            "citation_audit_status": "NOT_RUN",
        }

        if editor.is_available():
            if not args.quiet and not args.brief:
                console_obj.print("[cyan]Running Editor-in-Chief review...[/cyan]")

            data_block = ""
            pm_block = ""
            valuation_params = ""
            consultant_review = ""
            if analysis_result:
                data_block = analysis_result.get("fundamentals_report", "")
                pm_block = analysis_result.get("final_trade_decision", "")
                valuation_params = analysis_result.get("valuation_params", "")
                consultant_review = analysis_result.get("consultant_review", "")

            try:
                final_article, feedback = await editor.edit(
                    writer=writer,
                    article_draft=draft_article,
                    ticker=ticker,
                    company_name=company_name,
                    data_block=data_block,
                    pm_block=pm_block,
                    valuation_params=valuation_params,
                    consultant_review=consultant_review,
                    voice_samples=writer.load_voice_samples(max_chars=5000),
                    valuation_context=valuation_context,
                    governance_card=governance_card
                    if isinstance(governance_card, dict)
                    else None,
                    evidence_constraints=evidence_constraints,
                )
            except Exception as exc:
                final_article = draft_article
                feedback = {
                    "verdict": "REVISE",
                    "confidence": 0.0,
                    "review_error": True,
                    "citation_audit_status": "NOT_RUN",
                }
                logger_obj.warning(
                    "article_editor_failed",
                    **summarize_exception(exc, operation="article editorial review"),
                )
                if not args.quiet and not args.brief:
                    console_obj.print(
                        "[yellow]Editor review failed; draft retained for review.[/yellow]"
                    )

        citation_audit_status = str(
            feedback.get("citation_audit_status") or "NOT_RUN"
        ).upper()
        from src.article_audit import (
            audit_article_claim_support,
            audit_article_claim_usage,
            strip_claim_usage,
        )

        canonical_snapshot = (
            analysis_result.get("analysis_snapshot")
            if isinstance(analysis_result, dict)
            else None
        )
        claim_audit_errors = [
            *audit_article_claim_support(final_article, canonical_snapshot),
            *audit_article_claim_usage(final_article, canonical_snapshot),
        ]
        if claim_audit_errors:
            feedback.setdefault("factual_errors", []).extend(claim_audit_errors)
            feedback["verdict"] = "REVISE"
            feedback["citation_audit_status"] = "FAILED"
            feedback["canonical_claim_audit"] = True
            citation_audit_status = "FAILED"
        final_article = strip_claim_usage(final_article)
        snapshot = canonical_snapshot
        strict_snapshot = (
            isinstance(snapshot, dict) and snapshot.get("contract_status") == "VALID"
        )
        decision_trace = (
            analysis_result.get("decision_trace")
            if isinstance(analysis_result, dict)
            else None
        )
        require_provenance = isinstance(
            analysis_result, dict
        ) and has_provenance_contract(analysis_result)
        approved = isinstance(final_article, str) and _article_is_publishable(
            final_article,
            feedback,
            snapshot=snapshot,
            decision_trace=decision_trace,
            require_decision_trace=strict_snapshot,
            require_provenance=require_provenance,
        )
        saved_article_path = (
            article_path
            if approved
            else article_path.with_name(
                f"{article_path.stem}.draft{article_path.suffix or '.md'}"
            )
        )
        _atomic_write_text(saved_article_path, final_article)
        article_hash = hashlib.sha256(final_article.encode("utf-8")).hexdigest()

        writer_model = getattr(writer, "current_model_name", "")
        writer_fell_back = bool(getattr(writer, "writer_fell_back", False))
        if isinstance(analysis_result, dict):
            run_summary = analysis_result.setdefault("run_summary", {})
            run_summary["article_writer_model"] = writer_model
            run_summary["article_writer_fell_back"] = writer_fell_back
            article_generation = {
                "status": "APPROVED" if approved else "REVIEW_REQUIRED",
                "writer_model": writer_model,
                "writer_fell_back": writer_fell_back,
                "editor_verdict": feedback.get("verdict", "REVISE"),
                "editor_confidence": feedback.get("confidence", 0.0),
                "editor_revisions": feedback.get("revisions", 0),
                "editor_skipped": bool(feedback.get("skipped")),
                "editor_review_error": bool(feedback.get("review_error")),
                "citation_audit_status": citation_audit_status,
                "canonical_claim_audit_status": (
                    "FAILED" if claim_audit_errors else "PASSED"
                ),
                "review_findings": _bounded_review_findings(feedback),
                "saved_path": str(saved_article_path),
                "intended_final_path": str(article_path),
                "sha256": article_hash,
            }
            analysis_result["article_generation"] = article_generation
            saved_path = analysis_result.get("_saved_analysis_path")
            if saved_path:
                from src.persistence import patch_saved_sections

                patch_saved_sections(
                    saved_path,
                    {
                        "run_summary": {
                            "article_writer_model": writer_model,
                            "article_writer_fell_back": writer_fell_back,
                        },
                        "article_generation": article_generation,
                    },
                    logger_obj=logger_obj,
                )

        if writer_fell_back and not args.quiet and not args.brief:
            console_obj.print(
                f"[yellow]Claude writer unavailable — article written by "
                f"fallback model {writer_model}.[/yellow]"
            )

        if not args.quiet and not args.brief:
            label = "Article saved to" if approved else "Draft saved for review to"
            color = "green" if approved else "yellow"
            console_obj.print(
                f"[{color}]{label}:[/{color}] "
                f"[cyan]{saved_article_path}[/cyan]{_cost_suffix()}"
            )
            console_obj.print(
                f"[dim]Word count: {len(final_article.split())} words[/dim]"
            )

    except Exception as exc:
        from src.error_safety import summarize_exception

        logger_obj.error(
            "article_generation_failed",
            **summarize_exception(
                exc,
                operation="generating article",
                provider="unknown",
            ),
            exc_info=True,
        )
        if not args.quiet and not args.brief:
            console_obj.print(
                f"[yellow]Warning: {error_message_formatter('generating article', exc)}[/yellow]"
            )


def _load_company_name_for_output(
    ticker: str,
    *,
    thread_pool_executor_cls=ThreadPoolExecutor,
) -> str | None:
    """Best-effort company-name lookup for markdown output contexts."""
    executor = None
    try:
        import yfinance as yf

        from src.ticker_utils import (
            _company_name_lookup_candidates,
            _is_valid_company_name,
        )

        executor = thread_pool_executor_cls(max_workers=1)
        for lookup_ticker, _lookup_strategy in _company_name_lookup_candidates(ticker):
            future = executor.submit(
                lambda symbol=lookup_ticker: yf.Ticker(symbol).info
            )
            info = future.result(timeout=5)
            if not info:
                continue
            raw_name = info.get("longName") or info.get("shortName")
            if isinstance(raw_name, str) and _is_valid_company_name(
                raw_name, lookup_ticker
            ):
                # Return canonical (un-normalized) — markdown report headers and
                # the writer downstream both want the full legal name.
                return raw_name.strip()
        return None
    except FuturesTimeoutError:
        return None
    except Exception:
        return None
    finally:
        shutdown = getattr(executor, "shutdown", None)
        if shutdown is not None:
            try:
                shutdown(wait=False, cancel_futures=True)
            except TypeError:
                shutdown(wait=False)


async def _load_company_name_for_output_async(ticker: str) -> str | None:
    """Async wrapper for output paths running inside the main event loop."""
    from src.blocking_io import OUTPUT_COMPANY_NAME_POLICY, run_blocking_call

    try:
        return await run_blocking_call(
            OUTPUT_COMPANY_NAME_POLICY.with_label(f"output_company_name:{ticker}"),
            lambda: _load_company_name_for_output(ticker),
        )
    except TimeoutError:
        return None


def _emit_start_banner(
    args,
    output_targets: OutputTargets,
    *,
    logger_obj=logger,
    print_fn=print,
    welcome_banner_fn=get_welcome_banner,
) -> str:
    """Render or log the startup banner and return it for file output."""
    welcome_banner = welcome_banner_fn(args.ticker, args.quick)
    if not output_targets.output_file and not args.quiet and not args.brief:
        print_fn(welcome_banner)
    if output_targets.output_file and not args.quiet and not args.brief:
        logger_obj.info(
            "analysis_output_starting",
            ticker=args.ticker,
            output_path=str(output_targets.output_file),
        )
    return str(welcome_banner)


def _resolved_output_company_name(
    result: dict,
    ticker: str,
    company_name_loader,
) -> str | None:
    """Return the canonical display name for reports and articles.

    Runtime state is authoritative for identity. Fresh lookups are only a
    fallback for legacy/test paths where the result lacks identity fields.
    """

    runtime_name = result.get("company_name")
    if isinstance(runtime_name, str) and runtime_name.strip():
        return runtime_name.strip()

    governance_card = result.get("entity_governance_card")
    if isinstance(governance_card, dict):
        card_name = governance_card.get("canonical_name")
        if isinstance(card_name, str) and card_name.strip():
            return card_name.strip()

    loaded_name = company_name_loader(ticker)
    if isinstance(loaded_name, str) and loaded_name.strip():
        return loaded_name.strip()
    return None


def _render_primary_output(
    result: dict,
    args,
    output_targets: OutputTargets,
    welcome_banner: str,
    *,
    console_obj: Console = console,
    logger_obj=logger,
    company_name_loader=None,
    display_results_fn=display_results,
    reporter_cls=QuietModeReporter,
    cost_suffix_fn=_cost_suffix,
) -> tuple[str | None, str | None, QuietModeReporter | None]:
    """Render the main user-facing report to stdout or file."""
    del welcome_banner
    use_markdown = (
        args.brief
        or args.quiet
        or not __import__("sys").stdout.isatty()
        or output_targets.output_file
    )
    company_name = None
    report = None
    reporter = None

    if company_name_loader is None:
        company_name_loader = _load_company_name_for_output

    if not use_markdown:
        display_results_fn(result, args.ticker, console_obj=console_obj)
        return company_name, report, reporter

    company_name = _resolved_output_company_name(
        result, args.ticker, company_name_loader
    )
    if "analysis_validity" not in result:
        from src.runtime_diagnostics import build_analysis_validity

        result["analysis_validity"] = build_analysis_validity(result)
    reporter = reporter_cls(
        args.ticker,
        company_name,
        quick_mode=args.quick,
        chart_format="svg" if args.svg else "png",
        transparent_charts=args.transparent,
        skip_charts=output_targets.skip_charts,
        image_dir=output_targets.image_dir,
        report_dir=output_targets.output_dir,
        report_stem=output_targets.output_file.stem
        if output_targets.output_file
        else None,
    )
    report = reporter.generate_report(result, brief_mode=args.brief)

    if output_targets.output_file:
        try:
            if output_targets.output_file.parent != Path("."):
                output_targets.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_targets.output_file, "w", encoding="utf-8") as f:
                f.write(report)
            if not args.quiet and not args.brief:
                console_obj.print(
                    f"[green]Report saved to:[/green] [cyan]{output_targets.output_file}[/cyan]{cost_suffix_fn()}"
                )
        except Exception as exc:
            from src.error_safety import summarize_exception

            logger_obj.error(
                "report_write_failed",
                path=str(output_targets.output_file),
                **summarize_exception(
                    exc,
                    operation="writing markdown report",
                    provider="unknown",
                ),
                exc_info=True,
            )
            raise SystemExit(1) from exc
    else:
        print(report)

    return company_name, report, reporter


async def _maybe_generate_article(
    result: dict,
    args,
    output_targets: OutputTargets,
    company_name: str | None,
    report: str | None,
    reporter: QuietModeReporter | None,
    tracing_callbacks: list[Any] | None = None,
    tracing_metadata: dict[str, Any] | None = None,
    *,
    logger_obj=logger,
    console_obj: Console = console,
    company_name_loader=None,
    handle_article_generation_fn=handle_article_generation,
    reporter_cls=QuietModeReporter,
    publishable_analysis_fn=is_publishable_analysis,
) -> bool:
    """Generate an article from a publishable analysis when requested."""
    if not args.article:
        return False

    if not publishable_analysis_fn(result):
        logger_obj.warning(
            "article_generation_skipped_invalid_analysis",
            ticker=args.ticker,
            analysis_validity=result.get("analysis_validity", {}),
        )
        if not args.quiet and not args.brief:
            console_obj.print(
                "[yellow]Skipping article generation because the analysis is incomplete or invalid.[/yellow]"
            )
        return False

    if (
        output_targets.skip_charts
        and not output_targets.output_file
        and not args.imagedir
    ):
        print(
            "Warning: Article generated without images (stdout mode).",
            file=__import__("sys").stderr,
        )

    trade_date = result.get("trade_date") or datetime.now().strftime("%Y-%m-%d")

    if company_name_loader is None:
        company_name_loader = _load_company_name_for_output

    if report is None or reporter is None:
        if company_name is None:
            company_name = (
                _resolved_output_company_name(result, args.ticker, company_name_loader)
                or args.ticker
            )
        reporter = reporter_cls(
            args.ticker,
            company_name,
            quick_mode=args.quick,
            chart_format="svg" if args.svg else "png",
            transparent_charts=args.transparent,
            skip_charts=output_targets.skip_charts,
            image_dir=output_targets.image_dir,
            report_dir=output_targets.output_dir,
            report_stem=output_targets.output_file.stem
            if output_targets.output_file
            else None,
        )
        report = reporter.generate_report(result, brief_mode=False)

    await handle_article_generation_fn(
        args=args,
        ticker=args.ticker,
        company_name=company_name or args.ticker,
        report_text=report,
        trade_date=trade_date,
        valuation_context=reporter.get_valuation_context(),
        analysis_result=result,
        tracing_callbacks=tracing_callbacks,
        tracing_metadata=tracing_metadata,
    )
    return True


def _report_analysis_failure(args, *, console_obj: Console = console) -> None:
    """Print the standard top-level analysis failure message."""
    if args.quiet or args.brief:
        print(
            "# Analysis Failed\n\nAn error occurred during analysis. Check logs for details."
        )
    else:
        console_obj.print(
            "\n[bold red]Analysis failed. Check logs for details.[/bold red]\n"
        )
