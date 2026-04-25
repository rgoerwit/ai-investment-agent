"""User-facing output, article, and report-rendering helpers."""

from __future__ import annotations

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
from src.runtime_diagnostics import is_publishable_analysis

logger = structlog.get_logger(__name__)
console = Console()


def _cost_suffix() -> str:
    """Return formatted cost string for display, or empty if no tracking data."""
    from src.token_tracker import get_tracker

    stats = get_tracker().get_total_stats()
    if stats["total_calls"] == 0:
        return ""
    return f" [dim](Est. cost: ${stats['total_cost_usd']:.4f})[/dim]"


def get_welcome_banner(ticker: str, quick_mode: bool) -> str:
    """Generate welcome banner string with configuration."""
    banner = []
    banner.append("# Multi-Agent Investment Analysis System")
    banner.append("")
    banner.append(f"**Ticker:** {ticker.upper()}  ")
    banner.append(f"**Analysis Mode:** {'Quick' if quick_mode else 'Deep'}  ")
    banner.append(f"**Quick Model:** {config.quick_think_llm}  ")
    banner.append(f"**Deep Model:** {config.deep_think_llm}  ")
    banner.append(
        f"**Memory System:** {'Enabled' if config.enable_memory else 'Disabled'}  "
    )
    banner.append(
        f"**LangSmith Tracing:** "
        f"{'Enabled' if config.langsmith_tracing_enabled else 'Disabled'}  "
    )
    banner.append(
        f"**Langfuse Tracing:** "
        f"{'Enabled' if config.langfuse_enabled else 'Disabled'}  "
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
    if not config.enable_memory:
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
            ("Trader", f"{safe_ticker}_trader_memory"),
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
            title="Final Trading Decision",
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
        ("trader_investment_plan", "Trading Proposal"),
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
    """Generate article if requested, then run Editor-in-Chief review."""
    article_path = resolve_article_path_fn(args, ticker)
    if not article_path:
        return

    if error_message_formatter is None:

        def error_message_formatter(operation, exc):
            return f"Error {type(exc).__name__} {operation}"

    try:
        from src.article_writer import ArticleEditor, ArticleWriter

        if not args.quiet and not args.brief:
            console_obj.print("\n[cyan]Generating article...[/cyan]")

        writer = ArticleWriter(
            use_github_urls=False,
            callbacks=tracing_callbacks,
            tracing_metadata=tracing_metadata,
        )
        draft_article = writer.write(
            ticker=ticker,
            company_name=company_name,
            report_text=report_text,
            trade_date=trade_date,
            output_path=article_path,
            valuation_context=valuation_context,
        )

        editor = ArticleEditor(
            callbacks=tracing_callbacks,
            tracing_metadata=tracing_metadata,
        )
        final_article = draft_article

        if editor.is_available():
            if not args.quiet and not args.brief:
                console_obj.print("[cyan]Running Editor-in-Chief review...[/cyan]")

            data_block = ""
            pm_block = ""
            valuation_params = ""
            if analysis_result:
                data_block = analysis_result.get("fundamentals_report", "")
                pm_block = analysis_result.get("final_trade_decision", "")
                valuation_params = analysis_result.get("valuation_params", "")

            try:
                final_article, feedback = await editor.edit(
                    writer=writer,
                    article_draft=draft_article,
                    ticker=ticker,
                    company_name=company_name,
                    data_block=data_block,
                    pm_block=pm_block,
                    valuation_params=valuation_params,
                )

                if feedback.get("skipped"):
                    logger_obj.info("Editor skipped (not available)")
                elif feedback.get("verdict") == "APPROVED":
                    logger_obj.info(
                        "Article approved by editor",
                        confidence=feedback.get("confidence"),
                    )
                else:
                    logger_obj.info(
                        "Article revised by editor",
                        revisions=feedback.get("revisions", 0),
                    )

                if final_article != draft_article:
                    with open(article_path, "w") as f:
                        f.write(final_article)
                    if not args.quiet and not args.brief:
                        console_obj.print("[green]Article revised and saved.[/green]")

            except Exception:
                final_article = draft_article
                if not args.quiet and not args.brief:
                    console_obj.print(
                        "[yellow]Editor revision failed, using original draft.[/yellow]"
                    )

        if not args.quiet and not args.brief:
            console_obj.print(
                f"[green]Article saved to:[/green] [cyan]{article_path}[/cyan]{_cost_suffix()}"
            )
            word_count = (
                len(final_article.split()) if isinstance(final_article, str) else 0
            )
            console_obj.print(f"[dim]Word count: {word_count} words[/dim]")

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
    try:
        import yfinance as yf

        from src.ticker_utils import (
            _company_name_lookup_candidates,
            _is_valid_company_name,
            normalize_company_name,
        )

        with thread_pool_executor_cls(max_workers=1) as executor:
            for lookup_ticker, _lookup_strategy in _company_name_lookup_candidates(
                ticker
            ):
                future = executor.submit(
                    lambda symbol=lookup_ticker: yf.Ticker(symbol).info
                )
                info = future.result(timeout=5)
                if not info:
                    continue
                raw_name = info.get("longName") or info.get("shortName")
                if _is_valid_company_name(raw_name, lookup_ticker):
                    return normalize_company_name(raw_name)
        return None
    except FuturesTimeoutError:
        return None
    except Exception:
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
    return welcome_banner


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

    company_name = company_name_loader(args.ticker)
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
            with open(output_targets.output_file, "w") as f:
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
            company_name = company_name_loader(args.ticker) or args.ticker
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
