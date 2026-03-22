#!/usr/bin/env python3
"""
Main entry point for the Multi-Agent Investment Analysis System.
Updated for Gemini 3 (Nov 2025).
"""

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import structlog
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import config FIRST to set telemetry/system env vars before any library imports
from src.config import config, validate_environment_variables
from src.eval import (
    CURRENT_CAPTURE_SCHEMA_VERSION,
    BaselineCaptureConfig,
    BaselineCaptureManager,
    reset_active_capture_manager,
    set_active_capture_manager,
)
from src.report_generator import QuietModeReporter
from src.runtime_diagnostics import build_analysis_validity, is_publishable_analysis

# IMPORTANT: Don't import get_tracker here - it instantiates the singleton immediately
# Import it lazily in functions that need it, after quiet mode is set

logger = structlog.get_logger(__name__)
console = Console()

CLI_APP_DEBUG_LOGGERS = ("__main__", "src")
CLI_NOISY_DEPENDENCY_LOGGERS: dict[str, int] = {
    "anthropic": logging.WARNING,
    "google": logging.INFO,
    "google_genai": logging.WARNING,
    "hpack": logging.WARNING,
    "httpcore": logging.WARNING,
    "httpx": logging.WARNING,
    "langchain": logging.INFO,
    "langgraph": logging.INFO,
    "openai": logging.WARNING,
    "urllib3": logging.WARNING,
}
HTTP_TRACE_LOGGERS = ("openai", "httpx", "httpcore", "hpack")


@dataclass(frozen=True)
class OutputTargets:
    """Resolved output and chart destinations for a CLI run."""

    output_file: Path | None
    image_dir: Path
    skip_charts: bool

    @property
    def output_dir(self) -> Path:
        return self.output_file.parent if self.output_file else Path.cwd()


def _cost_suffix() -> str:
    """Return formatted cost string for display, or empty if no tracking data."""
    from src.token_tracker import get_tracker

    stats = get_tracker().get_total_stats()
    if stats["total_calls"] == 0:
        return ""
    return f" [dim](Est. cost: ${stats['total_cost_usd']:.4f})[/dim]"


def suppress_all_logging():
    """Suppress all logging output for quiet mode."""
    logging.getLogger().setLevel(logging.CRITICAL)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).propagate = False
    for logger_name in [
        "httpx",
        "openai",
        "httpcore",
        "langchain",
        "langgraph",
        "google",
    ]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    # Suppress structlog (used by token_tracker)
    import structlog

    structlog.configure(
        processors=[],
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    import warnings

    warnings.filterwarnings("ignore")


def _cli_logging_mode(
    args,
) -> Literal["quiet", "brief", "normal", "verbose", "debug"]:
    if getattr(args, "quiet", False):
        return "quiet"
    if getattr(args, "brief", False):
        return "brief"
    if getattr(args, "debug", False):
        return "debug"
    if getattr(args, "verbose", False):
        return "verbose"
    return "normal"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser (separated for testability)."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Investment Analysis System (Gemini 3 Edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python -m src.main --ticker AAPL

  # Quick analysis mode (Gemini Flash)
  python -m src.main --ticker NVDA --quick

  # Strict quality gate (tighter thresholds, fewer BUYs, token savings on rejects)
  python -m src.main --ticker 0005.HK --strict

  # Composable: strict quality bar + quick/cheap models
  python -m src.main --ticker 0005.HK --strict --quick

  # Quiet mode (markdown report only)
  python -m src.main --ticker AAPL --quiet

  # Brief mode (header, summary, decision only)
  python -m src.main --ticker AAPL --brief

  # Custom models
  python -m src.main --ticker TSLA --quick-model gemini-2.5-flash --deep-model gemini-3-pro-preview

  # Enable Langfuse tracing for this run
  python -m src.main --ticker 0005.HK --trace-langfuse

  # Batch retrospective: process all past tickers
  python -m src.main --retrospective-only

  # With Poetry
  poetry run python -m src.main --ticker MSFT --quick
        """,
    )

    parser.add_argument(
        "--ticker",
        type=str,
        required=False,
        default=None,
        help="Stock ticker symbol to analyze (e.g., AAPL, NVDA, TSLA)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick analysis mode (faster, less detailed)",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help=(
            "Apply stricter financial health criteria: tighter D/E and coverage thresholds, "
            "auto-reject REITs/ETFs, PFIC, and VIE structures, escalate value-trap warnings "
            "to rejects, and require higher conviction for BUY. Reduces BUY count and saves "
            "tokens by rejecting candidates before Bull/Bear debate. Composable with --quick."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all logging and output only final markdown report",
    )

    parser.add_argument(
        "--brief",
        action="store_true",
        help="Output only header, executive summary, and decision rationale",
    )

    parser.add_argument(
        "--quick-model",
        type=str,
        default=None,
        help=f"Model to use for quick analysis (default: {config.quick_think_llm})",
    )

    parser.add_argument(
        "--deep-model",
        type=str,
        default=None,
        help=f"Model to use for deep analysis (default: {config.deep_think_llm})",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable high-signal application diagnostics",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable developer debug logging; use INVESTMENT_AGENT_TRACE_HTTP=1 "
            "for raw transport traces"
        ),
    )

    parser.add_argument(
        "--no-memory", action="store_true", help="Disable persistent memory (ChromaDB)"
    )

    parser.add_argument(
        "--svg",
        action="store_true",
        help="Generate charts in SVG format (default: PNG)",
    )

    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Use transparent background for charts (default: white grid)",
    )

    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation entirely",
    )

    parser.add_argument(
        "--trace-langfuse",
        action="store_true",
        help=(
            "Enable Langfuse tracing for this run (overrides LANGFUSE_ENABLED in .env). "
            "Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout). If set, images will default to {output_dir}/images",
    )

    parser.add_argument(
        "--imagedir",
        type=str,
        default=None,
        help="Directory for chart images. If --output is set, defaults to {output_dir}/images. If not set, defaults to 'images' in current dir.",
    )

    parser.add_argument(
        "--article",
        nargs="?",
        const=True,
        default=False,
        help=(
            "Generate a Medium-style article from the analysis. "
            "Can specify output path (e.g., --article article.md) or use default "
            "(e.g., --article generates {ticker}_article.md in results dir)."
        ),
    )

    parser.add_argument(
        "--retrospective-only",
        action="store_true",
        help="Run retrospective evaluation on all past analyses without running "
        "a new analysis. Processes all tickers found in results directory.",
    )

    parser.add_argument(
        "--capture-baseline",
        action="store_true",
        help=(
            "Capture a versioned baseline bundle for this run under evals/captures/. "
            "Does not perform evaluation or baseline promotion."
        ),
    )

    parser.add_argument(
        "--capture-baseline-cleanup",
        action="store_true",
        help=(
            "Clean up stale inflight baseline captures under evals/captures/ and exit, "
            "or run cleanup before capture when combined with --capture-baseline."
        ),
    )

    return parser


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate: --ticker is required unless --retrospective-only or cleanup-only
    if (
        not args.retrospective_only
        and not args.capture_baseline_cleanup
        and not args.ticker
    ):
        parser.error(
            "--ticker is required unless --retrospective-only or "
            "--capture-baseline-cleanup is specified"
        )

    if args.debug:
        args.verbose = True

    return args


def resolve_output_paths(args) -> tuple[Path | None, Path]:
    """
    Determine output file and image directory based on arguments.

    Args:
        args: Parsed arguments namespace

    Returns:
        Tuple of (output_file_path, image_dir_path)
    """
    # Determine output location
    output_file = Path(args.output) if args.output else None
    output_dir = output_file.parent if output_file else Path.cwd()

    # Determine image directory
    if args.imagedir:
        # User specified image directory
        image_dir = Path(args.imagedir)
    elif output_file:
        # Default to {output_dir}/images if writing to file
        image_dir = output_dir / "images"
    else:
        # Default to current directory's images if stdout
        image_dir = Path("images")

    return output_file, image_dir


def validate_imagedir(imagedir: str) -> Path:
    """Validate image directory path.

    Allow any path (relative or absolute).
    """
    return Path(imagedir)


def resolve_article_path(args, ticker: str) -> Path | None:
    """
    Determine article output path based on arguments.

    Args:
        args: Parsed arguments namespace
        ticker: Stock ticker symbol

    Returns:
        Path for article output, or None if --article not specified

    Path resolution logic:
        1. --article /abs/path.md  -> Use absolute path as-is
        2. --article rel.md --output /dir/report.md  -> /dir/rel.md (relative to output dir)
        3. --article rel.md (no --output)  -> rel.md (relative to cwd)
        4. --article --output /dir/report.md  -> /dir/report-ARTICLE.md
        5. --article (no --output)  -> results/{ticker}_article.md
    """
    if not args.article:
        return None

    if isinstance(args.article, str):
        # --article with explicit path
        article_path = Path(args.article)
        if not article_path.suffix:
            article_path = article_path.with_suffix(".md")

        # If absolute path, use as-is
        if article_path.is_absolute():
            return article_path

        # If relative path and --output specified, resolve relative to output directory
        if args.output:
            output_dir = Path(args.output).parent
            return output_dir / article_path

        # Otherwise, relative to current working directory
        return article_path

    elif args.article is True:
        # --article with no value
        if args.output:
            # Derive from --output path: add "_article" suffix (consistent with image naming)
            output_path = Path(args.output)
            stem = output_path.stem  # e.g., "0005_HK_2026-01-01"
            suffix = output_path.suffix or ".md"  # e.g., ".md"
            article_name = f"{stem}_article{suffix}"
            return output_path.parent / article_name
        else:
            # No --output: use default path in results dir
            safe_ticker = ticker.replace(".", "_").replace("/", "_")
            return config.results_dir / f"{safe_ticker}_article.md"
    else:
        return None


def run_provider_preflight() -> dict[str, dict[str, str]]:
    """Log and return a concise provider/network preflight summary."""
    results: dict[str, dict[str, str]] = {}
    provider_hosts = [
        ("google_gemini", "generativelanguage.googleapis.com"),
        ("anthropic", "api.anthropic.com"),
        ("openai", "api.openai.com"),
        ("yahoo", "guce.yahoo.com"),
    ]
    for provider, host in provider_hosts:
        try:
            socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            results[provider] = {"host": host, "dns": "ok"}
            logger.info("provider_preflight", provider=provider, host=host, dns="ok")
        except Exception as exc:
            results[provider] = {
                "host": host,
                "dns": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            logger.warning(
                "provider_preflight",
                provider=provider,
                host=host,
                dns="failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
    return results


def configure_tool_audit_logging(enabled: bool) -> None:
    """Install or remove verbose tool audit hooks for this CLI run."""
    from src.tooling.audit import LoggingToolAuditHook
    from src.tooling.runtime import TOOL_SERVICE

    hooks = [h for h in TOOL_SERVICE.hooks if not isinstance(h, LoggingToolAuditHook)]
    if enabled:
        hooks.insert(0, LoggingToolAuditHook())
        TOOL_SERVICE.set_hooks(hooks)
        logger.info("tool_audit_logging_enabled")
    else:
        TOOL_SERVICE.set_hooks(hooks)


def configure_content_inspection_from_config() -> None:
    """Wire up content inspection from config settings.

    Called independently of logging configuration — security inspection must
    not be gated on CLI verbosity flags.
    """
    from src.tooling.inspection_hook import ContentInspectionHook
    from src.tooling.inspection_service import configure_content_inspection
    from src.tooling.inspector import NullInspector
    from src.tooling.runtime import TOOL_SERVICE

    hooks = [h for h in TOOL_SERVICE.hooks if not isinstance(h, ContentInspectionHook)]

    if not config.untrusted_content_inspection_enabled:
        TOOL_SERVICE.set_hooks(hooks)
        configure_content_inspection(
            NullInspector(), mode="warn", fail_policy="fail_open"
        )
        return

    mode = config.untrusted_content_inspection_mode
    fail_policy = config.untrusted_content_fail_policy
    backend_name = config.untrusted_content_backend

    if backend_name == "null" or not backend_name:
        inspector = NullInspector()
    else:
        raise ValueError(
            "UNTRUSTED_CONTENT_BACKEND is set to "
            f"{backend_name!r}, but only 'null' is implemented in this branch."
        )

    configure_content_inspection(inspector, mode=mode, fail_policy=fail_policy)
    hooks.append(ContentInspectionHook())
    TOOL_SERVICE.set_hooks(hooks)
    logger.info(
        "content_inspection_enabled",
        mode=mode,
        fail_policy=fail_policy,
        backend=backend_name,
    )


def configure_cli_logging(args) -> dict[str, dict[str, str]]:
    """Configure CLI logging without globally enabling dependency debug output."""
    mode = _cli_logging_mode(args)
    if mode in {"quiet", "brief"}:
        configure_tool_audit_logging(False)
        suppress_all_logging()
        return {}

    logging.getLogger().setLevel(logging.INFO)

    app_level = logging.DEBUG if mode in {"verbose", "debug"} else logging.INFO
    for name in CLI_APP_DEBUG_LOGGERS:
        logging.getLogger(name).setLevel(app_level)

    for name, level in CLI_NOISY_DEPENDENCY_LOGGERS.items():
        logging.getLogger(name).setLevel(level)

    if mode == "debug" and os.getenv("INVESTMENT_AGENT_TRACE_HTTP") == "1":
        for name in HTTP_TRACE_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)

    enable_diagnostics = mode in {"verbose", "debug"}
    configure_tool_audit_logging(enable_diagnostics)
    return run_provider_preflight() if enable_diagnostics else {}


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
        # Backward-compatible aliases: "completed" here means "finished", not "succeeded".
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


async def handle_article_generation(
    args,
    ticker: str,
    company_name: str,
    report_text: str,
    trade_date: str,
    valuation_context: str | None = None,
    analysis_result: dict | None = None,
) -> None:
    """
    Generate article if --article flag is set, then run Editor-in-Chief review.

    Args:
        args: Parsed arguments namespace
        ticker: Stock ticker symbol
        company_name: Full company name
        report_text: The full analysis report
        trade_date: Date of the analysis
        valuation_context: Optional context about chart valuation vs decision
        analysis_result: Raw result dictionary containing DATA_BLOCK/PM_BLOCK
    """
    article_path = resolve_article_path(args, ticker)
    if not article_path:
        return

    try:
        from src.article_writer import ArticleEditor, ArticleWriter

        if not args.quiet and not args.brief:
            console.print("\n[cyan]Generating article...[/cyan]")

        # Default to local paths so markdown renders immediately in editors
        # Users who want GitHub URLs can set GITHUB_RAW_BASE env var
        writer = ArticleWriter(use_github_urls=False)
        draft_article = writer.write(
            ticker=ticker,
            company_name=company_name,
            report_text=report_text,
            trade_date=trade_date,
            output_path=article_path,
            valuation_context=valuation_context,
        )

        # Run Editor-in-Chief review loop
        editor = ArticleEditor()
        final_article = draft_article  # Default to draft if editor unavailable or fails

        if editor.is_available():
            if not args.quiet and not args.brief:
                console.print("[cyan]Running Editor-in-Chief review...[/cyan]")

            # Extract ground truth from analysis result
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

                # Log editor outcome
                if feedback.get("skipped"):
                    logger.info("Editor skipped (not available)")
                elif feedback.get("verdict") == "APPROVED":
                    logger.info(
                        "Article approved by editor",
                        confidence=feedback.get("confidence"),
                    )
                else:
                    logger.info(
                        "Article revised by editor",
                        revisions=feedback.get("revisions", 0),
                    )

                # Save the final (possibly edited) article
                if final_article != draft_article:
                    with open(article_path, "w") as f:
                        f.write(final_article)
                    if not args.quiet and not args.brief:
                        console.print("[green]Article revised and saved.[/green]")

            except Exception as e:
                # Safety net: if editor fails, preserve the draft
                logger.warning(
                    f"Editor revision failed, preserving original draft: {e}"
                )
                final_article = draft_article
                if not args.quiet and not args.brief:
                    console.print(
                        "[yellow]Editor revision failed, using original draft.[/yellow]"
                    )

        if not args.quiet and not args.brief:
            console.print(
                f"[green]Article saved to:[/green] [cyan]{article_path}[/cyan]{_cost_suffix()}"
            )
            # Defensive: ensure article is a string before counting words
            word_count = (
                len(final_article.split()) if isinstance(final_article, str) else 0
            )
            console.print(f"[dim]Word count: {word_count} words[/dim]")

    except Exception as e:
        logger.error("article_generation_failed", error=str(e), exc_info=True)
        if not args.quiet and not args.brief:
            console.print(f"[yellow]Warning: Article generation failed: {e}[/yellow]")


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
    """Display welcome banner with configuration.

    Deprecated: Use get_welcome_banner() instead.
    """
    print(get_welcome_banner(ticker, quick_mode))


def display_memory_statistics(ticker: str):
    """Display memory statistics for the current ticker."""
    if not config.enable_memory:
        return

    try:
        from src.memory import create_memory_instances, sanitize_ticker_for_collection

        # Get memories specific to THIS ticker
        memories = create_memory_instances(ticker)
        safe_ticker = sanitize_ticker_for_collection(ticker)

        console.print(f"\n[bold cyan]Memory Statistics for {ticker}:[/bold cyan]\n")

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

        console.print(memory_table)
        console.print()

    except Exception as e:
        logger.warning("memory_statistics_unavailable", error=str(e))


def display_token_summary():
    """Display token usage summary in a formatted table."""
    from src.token_tracker import get_tracker

    tracker = get_tracker()
    stats = tracker.get_total_stats()

    if stats["total_calls"] == 0:
        return

    console.print("\n[bold cyan]Token Usage Summary:[/bold cyan]\n")

    # Overall stats table
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

    console.print(summary_table)

    # Per-agent breakdown
    console.print("\n[bold cyan]Per-Agent Token Usage:[/bold cyan]\n")

    agent_table = Table(show_header=True, box=box.ROUNDED)
    agent_table.add_column("Agent", style="cyan")
    agent_table.add_column("Calls", style="yellow", justify="right")
    agent_table.add_column("Prompt Tokens", style="blue", justify="right")
    agent_table.add_column("Completion Tokens", style="magenta", justify="right")
    agent_table.add_column("Total Tokens", style="green", justify="right")
    agent_table.add_column("Cost (USD)", style="red", justify="right")

    # Sort by cost descending
    sorted_agents = sorted(
        stats["agents"].items(), key=lambda x: x[1]["cost_usd"], reverse=True
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

    console.print(agent_table)
    console.print()


def display_results(result: dict, ticker: str):
    """Display analysis results in a formatted manner."""
    console.print("\n" + "=" * 80)
    console.print("[bold green]Analysis Complete![/bold green]\n")

    # Display token usage first
    display_token_summary()

    # Display final trading decision
    if "final_trade_decision" in result and result["final_trade_decision"]:
        decision_panel = Panel(
            result["final_trade_decision"],
            title="Final Trading Decision",
            border_style="green",
            padding=(1, 2),
        )
        console.print(decision_panel)

    # Display individual analyst reports
    console.print("\n[bold cyan]Analyst Reports:[/bold cyan]\n")

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

            if content.startswith("Error"):
                style = "red"
            else:
                style = "cyan"

            if len(content) > 800:
                content = content[:800] + "\n\n[... truncated for display ...]"

            report_panel = Panel(
                content, title=f"{display_name}", border_style=style, padding=(1, 2)
            )
            console.print(report_panel)
            console.print()

    display_memory_statistics(ticker)
    console.print("=" * 80 + "\n")


def save_results_to_file(result: dict, ticker: str, quick_mode: bool = False) -> Path:
    """Save analysis results to a JSON file in the results directory."""
    from src.memory import get_ticker_memory_stats
    from src.prompts import get_all_prompts

    results_dir = Path(config.results_dir)
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
        except Exception as e:
            logger.warning("memory_stats_unavailable", error=str(e))

    # Get token usage stats
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
        "prompts_metadata": {
            "prompts_used": prompts_used,
            "available_prompts": available_prompts,
            "custom_prompts_loaded": custom_prompts_loaded,
            "prompts_directory": str(prompts_dir),
            "total_agents": len(prompts_used),
            "note": "system_message field contains the actual prompt text used by each agent",
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
                "debate_rounds": 1,  # Risk analysts run in parallel (1 round each)
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

    # Extract prediction snapshot for future retrospective evaluation (zero LLM cost)
    try:
        from src.retrospective import extract_snapshot

        save_data["prediction_snapshot"] = extract_snapshot(result, ticker, quick_mode)
    except Exception as e:
        logger.warning("snapshot_extraction_failed", error=str(e))

    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=2)

    try:
        from src.ibkr.reconciler import (
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
                logger.info(
                    "analysis_index_refreshed_after_save",
                    ticker=ticker,
                    path=str(results_dir),
                    refreshed_count=len(refreshed),
                )
    except Exception as exc:
        logger.debug("analysis_index_update_skipped", error=str(exc))

    logger.info(
        f"Results saved to {filepath} ({len(prompts_used)} prompts tracked, {len(custom_prompts_loaded)} custom)"
    )

    # Log token tracking info
    if token_stats["total_calls"] > 0:
        logger.info(
            f"Token usage tracked: {token_stats['total_calls']} LLM calls, "
            f"{token_stats['total_tokens']:,} total tokens, "
            f"${token_stats['total_cost_usd']:.4f} projected cost (paid tier) - "
            f"saved to {filepath}"
        )

    return filepath


_BENCH_NAMES: dict[str, str] = {
    "^N225": "Nikkei-225",
    "^HSI": "Hang Seng",
    "^TWII": "Taiwan Weighted",
    "^KS11": "KOSPI",
    "^AEX": "AEX",
    "^GDAXI": "DAX",
    "^FTSE": "FTSE 100",
    "^FCHI": "CAC 40",
    "^GSPTSE": "TSX",
    "^AXJO": "ASX 200",
    "^STI": "STI",
    "^FTSEMIB": "FTSE MIB",
    "^OMX": "OMX Stockholm",
    "^GSPC": "S&P 500",
}


async def _fetch_market_context(ticker: str, trade_date: str) -> str:
    """
    Return a one-line benchmark performance note for the ticker's home market.

    Example: "MARKET NOTE: Nikkei-225 down 4.2% on 2026-03-05."

    Returns empty string on any error so callers never need to handle exceptions.
    """
    try:
        import yfinance as yf

        from src.retrospective import (
            EXCHANGE_BENCHMARK,
            FALLBACK_BENCHMARK,
            _get_ticker_suffix,
        )

        suffix = _get_ticker_suffix(ticker)
        benchmark = EXCHANGE_BENCHMARK.get(suffix, FALLBACK_BENCHMARK)
        hist = await asyncio.to_thread(
            lambda: yf.Ticker(benchmark).history(period="2d")
        )
        if len(hist) >= 2:
            pct = (
                (hist["Close"].iloc[-1] - hist["Close"].iloc[0])
                / hist["Close"].iloc[0]
                * 100
            )
            direction = "up" if pct >= 0 else "down"
            name = _BENCH_NAMES.get(benchmark, benchmark.lstrip("^"))
            actual_date = (
                hist.index[-1].strftime("%Y-%m-%d")
                if hasattr(hist.index[-1], "strftime")
                else trade_date
            )
            return f"MARKET NOTE: {name} {direction} {abs(pct):.1f}% on {actual_date}."
    except Exception as e:
        logger.debug("market_context_fetch_failed", error=str(e))
    return ""


async def run_analysis(
    ticker: str,
    quick_mode: bool,
    strict_mode: bool = False,
    chart_format: str = "png",
    transparent_charts: bool = False,
    image_dir: Path | None = None,
    skip_charts: bool = False,
    baseline_capture: BaselineCaptureManager | None = None,
    capture_args: argparse.Namespace | None = None,
) -> dict | None:
    """Run the multi-agent analysis workflow.

    Args:
        ticker: Stock ticker symbol
        quick_mode: If True, use faster/cheaper models and skip some steps
        strict_mode: If True, apply tighter quality gates and reject REITs/PFIC/VIE
        chart_format: Chart output format ('png' or 'svg')
        transparent_charts: Whether to use transparent chart backgrounds
        image_dir: Directory for chart output (None = use config default)
        skip_charts: If True, skip chart generation entirely
    """
    try:
        from langchain_core.messages import HumanMessage

        from src.agents import AgentState, InvestDebateState, RiskDebateState
        from src.graph import TradingContext, create_trading_graph
        from src.token_tracker import get_tracker

        # Reset token tracker for fresh analysis
        tracker = get_tracker()
        tracker.reset()

        logger.info("analysis_starting", ticker=ticker, quick_mode=quick_mode)

        # CRITICAL FIX: Enforce real-world date to prevent "Time Travel" hallucinations
        # This overrides potentially stale system prompts or environment defaults
        real_date = datetime.now().strftime("%Y-%m-%d")

        # CRITICAL FIX: Fetch and verify company name BEFORE graph execution
        # Multi-source resolution prevents identity hallucination when yfinance fails
        # (e.g., delisted tickers like 2154.HK where agents guess different companies)
        from src.ticker_utils import resolve_company_name

        name_result = await resolve_company_name(ticker)
        company_name = name_result.name

        if not name_result.is_resolved:
            logger.warning(
                "company_name_unresolved_at_startup",
                ticker=ticker,
                message="No source could resolve company name — LLM hallucination risk",
            )

        # Fetch benchmark context once (non-blocking) before graph starts.
        # Prepended to the HumanMessage so every agent receives it as session context.
        market_context = await _fetch_market_context(ticker, real_date)

        session_id = f"{ticker}-{real_date}-{uuid.uuid4().hex[:8]}"
        if baseline_capture:
            baseline_capture.start_run(
                ticker=ticker,
                trade_date=real_date,
                args=capture_args
                or argparse.Namespace(
                    ticker=ticker,
                    quick=quick_mode,
                    strict=strict_mode,
                    no_memory=not config.enable_memory,
                    capture_baseline=True,
                ),
                session_id=session_id,
            )

        graph = create_trading_graph(
            ticker=ticker,  # BUG FIX #1: Pass ticker for isolation
            cleanup_previous=True,  # BUG FIX #1: Cleanup to prevent contamination
            max_debate_rounds=1 if quick_mode else 2,
            max_risk_discuss_rounds=1,
            enable_memory=config.enable_memory,
            recursion_limit=100,
            quick_mode=quick_mode,  # Pass quick_mode for consultant LLM selection
            strict_mode=strict_mode,  # Pass strict_mode for quality gates
            # Chart generation (post-PM)
            chart_format=chart_format,
            transparent_charts=transparent_charts,
            image_dir=image_dir,
            skip_charts=skip_charts,
            baseline_capture=baseline_capture,
        )

        _base_msg = f"Analyze {ticker} ({company_name}) for investment decision. Current Date: {real_date}."
        if market_context:
            _base_msg += f" {market_context}"
        initial_state = AgentState(
            messages=[HumanMessage(content=_base_msg)],
            company_of_interest=ticker,
            company_name=company_name,  # ADDED: Anchor verified company name in state
            company_name_resolved=name_result.is_resolved,
            trade_date=real_date,
            sender="user",
            market_report="",
            sentiment_report="",
            news_report="",
            raw_fundamentals_data="",
            foreign_language_report="",
            fundamentals_report="",
            investment_debate_state=InvestDebateState(
                bull_round1="",
                bear_round1="",
                bull_round2="",
                bear_round2="",
                current_round=1,
                bull_history="",
                bear_history="",
                history="",
                current_response="",
                judge_decision="",
                count=0,
            ),
            investment_plan="",
            trader_investment_plan="",
            risk_debate_state=RiskDebateState(
                latest_speaker="",
                current_risky_response="",
                current_safe_response="",
                current_neutral_response="",
            ),
            final_trade_decision="",
            tools_called={},
            prompts_used={},
            artifact_statuses={},
            red_flags=[],
            pre_screening_result="",
            legal_report="",
            auditor_report="",
            value_trap_report="",
        )

        context = TradingContext(
            ticker=ticker,
            trade_date=real_date,
            quick_mode=quick_mode,
            enable_memory=config.enable_memory,
            max_debate_rounds=1 if quick_mode else 2,
            max_risk_rounds=1,
        )

        logger.info(
            "multi_agent_analysis_starting", ticker=ticker, trade_date=real_date
        )

        # Get observability callbacks (Langfuse if enabled)
        from src.observability import flush_traces, get_tracing_callbacks

        tags = [
            "quick" if quick_mode else "normal",
            f"quick-model:{config.quick_think_llm}",
            f"deep-model:{config.deep_think_llm}",
            f"memory:{'on' if config.enable_memory else 'off'}",
        ]
        tracing_callbacks, tracing_metadata = get_tracing_callbacks(
            ticker=ticker,
            session_id=session_id,
            tags=tags,
        )

        capture_token = None
        if baseline_capture:
            capture_token = set_active_capture_manager(baseline_capture)

        try:
            result = await graph.ainvoke(
                initial_state,
                config={
                    "recursion_limit": 100,
                    "configurable": {"context": context},
                    "callbacks": tracing_callbacks,
                    "metadata": tracing_metadata,
                },
            )
        finally:
            if capture_token is not None:
                reset_active_capture_manager(capture_token)

        # Flush traces before returning
        flush_traces()

        logger.info("analysis_completed", ticker=ticker)

        # Log token usage summary
        from src.token_tracker import get_tracker

        tracker = get_tracker()
        tracker.print_summary()

        if isinstance(result, dict):
            result["analysis_validity"] = build_analysis_validity(result)

        return result

    except Exception as e:
        if baseline_capture:
            baseline_capture.reject_run(
                [f"analysis_exception:{type(e).__name__}"],
                stage="run_analysis",
            )
        logger.error("analysis_failed", ticker=ticker, error=str(e), exc_info=True)
        console.print(f"\n[bold red]Error during analysis:[/bold red] {str(e)}\n")
        return None


def _enable_quiet_runtime_if_needed(args: argparse.Namespace) -> None:
    """Suppress tracker/log noise before any later imports can initialize it."""
    if not (args.quiet or args.brief):
        return

    from src.token_tracker import TokenTracker

    TokenTracker.set_quiet_mode(True)
    suppress_all_logging()

    tracker = TokenTracker()
    tracker._quiet_mode = True
    config.quiet_mode = True


def _apply_runtime_overrides(args: argparse.Namespace) -> None:
    """Apply per-run CLI overrides to the config singleton."""
    if args.quick_model:
        config.quick_think_llm = args.quick_model
    if args.deep_model:
        config.deep_think_llm = args.deep_model
    if args.no_memory:
        config.enable_memory = False
    if args.trace_langfuse:
        config.langfuse_enabled = True


def _validate_cli_args(args: argparse.Namespace) -> None:
    """Validate incompatible flag combinations."""
    if not args.quick:
        return

    chart_flags = [
        f
        for f, value in [("--transparent", args.transparent), ("--svg", args.svg)]
        if value
    ]
    if not chart_flags:
        return

    flags_str = " and ".join(chart_flags)
    verb = "has" if len(chart_flags) == 1 else "have"
    noun = "that flag" if len(chart_flags) == 1 else "those flags"
    print(
        f"error: {flags_str} {verb} no effect with --quick "
        f"(chart generation is skipped in quick mode). "
        f"Remove {noun} or drop --quick.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _resolve_output_targets(args: argparse.Namespace) -> OutputTargets:
    """Resolve output/image paths and derived chart behavior for this run."""
    output_file, image_dir = resolve_output_paths(args)
    skip_charts = bool(args.no_charts or (output_file is None and not args.imagedir))
    return OutputTargets(
        output_file=output_file,
        image_dir=image_dir,
        skip_charts=skip_charts,
    )


def _setup_runtime(
    args: argparse.Namespace, output_targets: OutputTargets
) -> dict[str, dict[str, str]]:
    """Configure logging, runtime paths, and environment validation."""
    _enable_quiet_runtime_if_needed(args)
    config.images_dir = output_targets.image_dir

    provider_preflight = configure_cli_logging(args)

    if (
        output_targets.skip_charts
        and not args.no_charts
        and not args.imagedir
        and not args.quiet
        and not args.brief
    ):
        logger.warning(
            "Writing to stdout: Charts disabled (no way to link them). "
            "Use --output to enable charts, or --imagedir to save images separately."
        )

    if not output_targets.skip_charts and output_targets.output_file:
        try:
            output_targets.image_dir.resolve().relative_to(
                output_targets.output_dir.resolve()
            )
        except ValueError:
            logger.warning(
                f"Image directory ({output_targets.image_dir}) is not a subdirectory of output directory ({output_targets.output_dir}). "
                "Report will contain absolute paths to images, which may not render correctly on other systems."
            )

    try:
        validate_environment_variables()
    except ValueError as exc:
        if args.quiet or args.brief:
            print(f"# Configuration Error\n\n{str(exc)}")
        else:
            console.print(f"\n[bold red]Configuration Error:[/bold red] {str(exc)}\n")
            console.print(
                "Please check your .env file and ensure all required API keys are set.\n"
            )
        raise SystemExit(1) from exc

    # Content inspection is configured independently of logging verbosity.
    try:
        configure_content_inspection_from_config()
    except ValueError as exc:
        if args.quiet or args.brief:
            print(f"# Configuration Error\n\n{str(exc)}")
        else:
            console.print(f"\n[bold red]Configuration Error:[/bold red] {str(exc)}\n")
        raise SystemExit(1) from exc

    return provider_preflight


async def _run_retrospective_only(args: argparse.Namespace) -> int:
    """Run retrospective-only mode and return the process exit code."""
    try:
        from src.retrospective import SnapshotLoadProgress, run_retrospective

        def report(update: SnapshotLoadProgress) -> None:
            if update.phase == "discovered":
                if update.total_files:
                    print(
                        f"Scanning {update.total_files} saved analysis file"
                        f"{'' if update.total_files == 1 else 's'} for retrospective snapshots...",
                        file=sys.stderr,
                        flush=True,
                    )
                return
            if update.phase == "parsing":
                print(
                    f"  Snapshot load progress: {update.processed_files}/{update.total_files} "
                    f"files scanned; {update.loaded_snapshots} snapshots across {update.loaded_tickers} tickers",
                    file=sys.stderr,
                    flush=True,
                )
                return
            if update.phase == "complete":
                print(
                    f"Loaded {update.loaded_snapshots} retrospective snapshots from {update.total_files} files.",
                    file=sys.stderr,
                    flush=True,
                )

        results_dir = Path(config.results_dir)
        if not args.quiet and not args.brief:
            console.print(
                "[cyan]Running retrospective evaluation on all past analyses...[/cyan]"
            )

        lessons = await run_retrospective(
            ticker=None,
            results_dir=results_dir,
            progress=report if not (args.quiet or args.brief) else None,
        )

        if lessons:
            if not args.quiet and not args.brief:
                console.print(f"\n[green]Generated {len(lessons)} lesson(s):[/green]")
                for lesson in lessons:
                    stored = "[stored]" if lesson.get("stored") else "[skipped]"
                    console.print(
                        f"  {stored} [{lesson['ticker']}] {lesson['lesson']} "
                        f"({lesson['failure_mode']} | conf: {lesson['confidence']:.2f})"
                    )
            else:
                print(
                    f"# Retrospective Complete\n\nGenerated {len(lessons)} lesson(s)."
                )
        else:
            msg = "No significant prediction deltas found."
            if not args.quiet and not args.brief:
                console.print(f"[yellow]{msg}[/yellow]")
            else:
                print(f"# Retrospective Complete\n\n{msg}")
    except Exception as exc:
        logger.error("retrospective_failed", error=str(exc), exc_info=True)
        if not args.quiet and not args.brief:
            console.print(
                f"[yellow]Warning: Retrospective evaluation failed: {exc}[/yellow]"
            )
        return 1
    return 0


async def _maybe_run_ticker_retrospective(args: argparse.Namespace) -> None:
    """Optionally generate retrospective lessons for the current ticker."""
    if not args.ticker:
        return
    if args.no_memory:
        logger.info("retrospective_skipped_no_memory", ticker=args.ticker)
        return

    try:
        from src.retrospective import SnapshotLoadProgress, run_retrospective

        def report(update: SnapshotLoadProgress) -> None:
            if update.phase != "parsing" or args.quiet or args.brief:
                return
            print(
                f"  Retrospective scan: {update.processed_files}/{update.total_files} "
                f"files; {update.loaded_snapshots} candidate snapshots",
                file=sys.stderr,
                flush=True,
            )

        results_dir = Path(config.results_dir)
        lessons = await run_retrospective(
            ticker=args.ticker,
            results_dir=results_dir,
            progress=report,
        )
        if lessons and not args.quiet and not args.brief:
            console.print(
                f"\n[green]Generated {len(lessons)} new lesson(s) from past analyses[/green]"
            )
    except Exception as exc:
        logger.warning("retrospective_failed", ticker=args.ticker, error=str(exc))


def _emit_start_banner(args: argparse.Namespace, output_targets: OutputTargets) -> str:
    """Render or log the startup banner and return it for file output."""
    welcome_banner = get_welcome_banner(args.ticker, args.quick)
    if not output_targets.output_file and not args.quiet and not args.brief:
        print(welcome_banner)
    if output_targets.output_file and not args.quiet and not args.brief:
        logger.info(
            "analysis_output_starting",
            ticker=args.ticker,
            output_path=str(output_targets.output_file),
        )
    return welcome_banner


async def _execute_analysis(
    args: argparse.Namespace,
    output_targets: OutputTargets,
    baseline_capture: BaselineCaptureManager | None = None,
) -> dict | None:
    """Run the analysis graph for the current CLI request."""
    capture_hook = None
    original_hooks = None
    if baseline_capture:
        from src.tooling.runtime import TOOL_SERVICE

        original_hooks = TOOL_SERVICE.hooks
        capture_hook = baseline_capture.make_tool_hook()
        TOOL_SERVICE.set_hooks([*original_hooks, capture_hook])

    try:
        return await run_analysis(
            args.ticker,
            args.quick,
            strict_mode=args.strict,
            chart_format="svg" if args.svg else "png",
            transparent_charts=args.transparent,
            image_dir=output_targets.image_dir,
            skip_charts=output_targets.skip_charts,
            baseline_capture=baseline_capture,
            capture_args=args,
        )
    finally:
        if baseline_capture and original_hooks is not None:
            from src.tooling.runtime import TOOL_SERVICE

            TOOL_SERVICE.set_hooks(original_hooks)


def _create_baseline_capture_manager(
    args: argparse.Namespace,
) -> BaselineCaptureManager | None:
    if (
        not getattr(args, "capture_baseline", False)
        and not getattr(args, "capture_baseline_cleanup", False)
    ) or getattr(args, "retrospective_only", False):
        return None
    return BaselineCaptureManager(
        BaselineCaptureConfig(
            enabled=True,
            schema_version=CURRENT_CAPTURE_SCHEMA_VERSION,
            output_root=Path("evals") / "captures",
        )
    )


def _run_baseline_capture_preflight(
    args: argparse.Namespace,
    baseline_capture: BaselineCaptureManager | None,
) -> tuple[bool, list[str]]:
    if baseline_capture is None:
        return True, []

    cleanup_summary = baseline_capture.cleanup_stale_inflight_runs()
    messages: list[str] = []
    if cleanup_summary.moved_to_rejected or cleanup_summary.removed_empty:
        messages.append(
            "Cleaned "
            f"{cleanup_summary.moved_to_rejected} stale inflight capture(s)"
            f" and removed {cleanup_summary.removed_empty} empty inflight directory(ies)."
        )

    if not getattr(args, "capture_baseline", False):
        return True, messages

    ok, errors = baseline_capture.preflight_git_clean()
    if not ok:
        messages.extend(errors)
    return ok, messages


def _print_capture_preflight_messages(
    messages: list[str],
    *,
    blocked: bool,
    quiet: bool,
    brief: bool,
) -> None:
    if not messages:
        return
    if quiet or brief:
        print("\n".join(messages))
        return
    if blocked:
        console.print(
            "[bold yellow]Baseline capture blocked before analysis[/bold yellow]"
        )
        for message in messages:
            console.print(f"- {message}")
        console.print()
    else:
        for message in messages:
            console.print(f"[yellow]{message}[/yellow]")


def _print_capture_result(
    args: argparse.Namespace,
    baseline_capture: BaselineCaptureManager | None,
    capture_path: Path | None,
) -> None:
    if baseline_capture is None or capture_path is None or args.quiet or args.brief:
        return
    status = baseline_capture.final_status or "accepted"
    if status == "accepted":
        console.print(
            f"[green]Baseline capture accepted:[/green] [cyan]{capture_path}[/cyan]"
        )
        return
    first_reason = getattr(baseline_capture, "_first_rejection_reason", None)
    console.print(
        f"[yellow]Baseline capture rejected:[/yellow] [cyan]{capture_path}[/cyan]"
    )
    if first_reason:
        console.print(f"[yellow]Reason:[/yellow] {first_reason}")


def _finalize_baseline_capture(
    baseline_capture: BaselineCaptureManager | None,
    result: dict | None,
) -> Path | None:
    if not baseline_capture or result is None:
        return None
    try:
        return baseline_capture.finalize_run(result)
    except Exception as exc:
        logger.error("baseline_capture_finalize_failed", error=str(exc), exc_info=True)
        return None


def _attach_run_summary(
    result: dict,
    args: argparse.Namespace,
    provider_preflight: dict[str, dict[str, str]],
) -> None:
    """Attach the compact run summary before any persistence/output steps."""
    result.setdefault("run_summary", {})
    result["run_summary"]["quick_mode"] = bool(args.quick)
    result["run_summary"] = build_run_summary(
        result,
        quick_mode=args.quick,
        article_requested=bool(args.article),
        provider_preflight=provider_preflight,
    )
    result["analysis_validity"] = build_analysis_validity(result)
    result["run_summary"] = build_run_summary(
        result,
        quick_mode=args.quick,
        article_requested=bool(args.article),
        provider_preflight=provider_preflight,
    )


def _load_company_name_for_output(ticker: str) -> str | None:
    """Best-effort company-name lookup for markdown output contexts."""
    try:
        import yfinance as yf

        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        return info.get("longName") or info.get("shortName")
    except Exception:
        return None


def _render_primary_output(
    result: dict,
    args: argparse.Namespace,
    output_targets: OutputTargets,
    welcome_banner: str,
) -> tuple[str | None, str | None, QuietModeReporter | None]:
    """Render the main user-facing report to stdout or file."""
    use_markdown = (
        args.brief
        or args.quiet
        or not sys.stdout.isatty()
        or output_targets.output_file
    )
    company_name = None
    report = None
    reporter = None

    if not use_markdown:
        display_results(result, args.ticker)
        return company_name, report, reporter

    company_name = _load_company_name_for_output(args.ticker)
    reporter = QuietModeReporter(
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
        full_content = welcome_banner + "\n" + report
        try:
            if output_targets.output_file.parent != Path("."):
                output_targets.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_targets.output_file, "w") as f:
                f.write(full_content)
            if not args.quiet and not args.brief:
                console.print(
                    f"[green]Report saved to:[/green] [cyan]{output_targets.output_file}[/cyan]{_cost_suffix()}"
                )
        except Exception as exc:
            logger.error(
                "report_write_failed",
                path=str(output_targets.output_file),
                error=str(exc),
                exc_info=True,
            )
            raise SystemExit(1) from exc
    else:
        print(report)

    return company_name, report, reporter


def _persist_analysis_outputs(result: dict, args: argparse.Namespace) -> None:
    """Persist JSON artifacts and rejection records."""
    try:
        filepath = save_results_to_file(result, args.ticker, quick_mode=args.quick)
        if not args.quiet and not args.brief:
            console.print(
                f"[green]Results saved to:[/green] [cyan]{filepath}[/cyan]{_cost_suffix()}"
            )
    except Exception as exc:
        logger.error("results_save_failed", error=str(exc), exc_info=True)
        if not args.quiet and not args.brief:
            console.print(
                f"\n[yellow]Warning: Could not save results to file: {exc}[/yellow]\n"
            )


async def _maybe_save_rejection_record(result: dict, args: argparse.Namespace) -> None:
    """Persist non-BUY verdicts as retrospective rejection records."""
    try:
        from src.retrospective import (
            create_lessons_memory,
            extract_snapshot,
            save_rejection_record,
        )

        snapshot = extract_snapshot(result, args.ticker, is_quick_mode=args.quick)
        verdict = snapshot.get("verdict", "")
        if verdict and verdict != "BUY":
            rejection_memory = create_lessons_memory()
            await save_rejection_record(snapshot, rejection_memory)
    except Exception as exc:
        logger.debug("rejection_record_save_skipped", error=str(exc))


async def _maybe_generate_article(
    result: dict,
    args: argparse.Namespace,
    output_targets: OutputTargets,
    company_name: str | None,
    report: str | None,
    reporter: QuietModeReporter | None,
) -> bool:
    """Generate an article from a publishable analysis when requested."""
    if not args.article:
        return False

    if not is_publishable_analysis(result):
        logger.warning(
            "article_generation_skipped_invalid_analysis",
            ticker=args.ticker,
            analysis_validity=result.get("analysis_validity", {}),
        )
        if not args.quiet and not args.brief:
            console.print(
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
            file=sys.stderr,
        )

    trade_date = result.get("trade_date") or datetime.now().strftime("%Y-%m-%d")

    if report is None or reporter is None:
        if company_name is None:
            company_name = _load_company_name_for_output(args.ticker) or args.ticker
        reporter = QuietModeReporter(
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

    await handle_article_generation(
        args=args,
        ticker=args.ticker,
        company_name=company_name or args.ticker,
        report_text=report,
        trade_date=trade_date,
        valuation_context=reporter.get_valuation_context(),
        analysis_result=result,
    )
    return True


def _log_final_summary(
    result: dict, args: argparse.Namespace, article_generated: bool
) -> None:
    """Emit the structured end-of-run summary log."""
    logger.info(
        "analysis_run_summary",
        ticker=args.ticker,
        **{**result.get("run_summary", {}), "article_generated": article_generated},
    )


def _report_analysis_failure(args: argparse.Namespace) -> None:
    """Print the standard top-level analysis failure message."""
    if args.quiet or args.brief:
        print(
            "# Analysis Failed\n\nAn error occurred during analysis. Check logs for details."
        )
    else:
        console.print(
            "\n[bold red]Analysis failed. Check logs for details.[/bold red]\n"
        )


async def main() -> int:
    """Main entry point for the application."""
    args = None
    try:
        args = parse_arguments()
        _apply_runtime_overrides(args)
        _validate_cli_args(args)
        output_targets = _resolve_output_targets(args)
        provider_preflight = _setup_runtime(args, output_targets)
        baseline_capture = _create_baseline_capture_manager(args)

        preflight_ok, preflight_messages = _run_baseline_capture_preflight(
            args, baseline_capture
        )
        if preflight_messages:
            _print_capture_preflight_messages(
                preflight_messages,
                blocked=not preflight_ok,
                quiet=args.quiet,
                brief=args.brief,
            )

        if getattr(args, "capture_baseline_cleanup", False) and not getattr(
            args, "capture_baseline", False
        ):
            return 0 if preflight_ok else 1

        if not preflight_ok:
            return 1

        if args.retrospective_only:
            return await _run_retrospective_only(args)

        await _maybe_run_ticker_retrospective(args)
        welcome_banner = _emit_start_banner(args, output_targets)
        if baseline_capture is None:
            result = await _execute_analysis(args, output_targets)
        else:
            result = await _execute_analysis(args, output_targets, baseline_capture)

        if not result:
            if baseline_capture:
                baseline_capture.reject_run(
                    ["analysis_returned_no_result"], stage="main"
                )
                _finalize_baseline_capture(
                    baseline_capture,
                    {
                        "analysis_validity": {"publishable": False},
                        "artifact_statuses": {},
                    },
                )
            _report_analysis_failure(args)
            return 1

        _attach_run_summary(result, args, provider_preflight)
        capture_path = _finalize_baseline_capture(baseline_capture, result)
        _print_capture_result(args, baseline_capture, capture_path)
        company_name, report, reporter = _render_primary_output(
            result, args, output_targets, welcome_banner
        )
        _persist_analysis_outputs(result, args)
        await _maybe_save_rejection_record(result, args)
        article_generated = await _maybe_generate_article(
            result, args, output_targets, company_name, report, reporter
        )
        _log_final_summary(result, args, article_generated)
        return 0

    except KeyboardInterrupt:
        if not (
            args and (getattr(args, "quiet", False) or getattr(args, "brief", False))
        ):
            console.print("\n[yellow]Analysis interrupted by user.[/yellow]\n")
        return 1
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        logger.error("unexpected_error", error=str(exc), exc_info=True)
        if args and (getattr(args, "quiet", False) or getattr(args, "brief", False)):
            print(f"# Unexpected Error\n\n{str(exc)}")
        else:
            console.print(f"\n[bold red]Unexpected error:[/bold red] {str(exc)}\n")
        return 1
    finally:
        try:
            from src.cleanup import cleanup_async_resources

            await cleanup_async_resources()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
