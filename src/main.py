#!/usr/bin/env python3
"""
Main entry point for the Multi-Agent Investment Analysis System.
Updated for Gemini 3 (Nov 2025).
"""

import argparse
import asyncio
import logging
import os
import socket
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console

import src.cli as cli

# Import config FIRST to set telemetry/system env vars before any library imports
import src.output as output
import src.persistence as persistence
from src.config import config, validate_environment_variables
from src.error_safety import format_error_message, summarize_exception
from src.eval import (
    CURRENT_CAPTURE_SCHEMA_VERSION,
    BaselineCaptureConfig,
    BaselineCaptureManager,
    BaselinePreflightResult,
    reset_active_capture_manager,
    set_active_capture_manager,
)
from src.runtime_diagnostics import build_analysis_validity

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


def _cost_suffix() -> str:
    """Return formatted cost string for display, or empty if no tracking data."""
    from src.token_tracker import get_tracker

    stats = get_tracker().get_total_stats()
    if stats["total_calls"] == 0:
        return ""
    return f" [dim](Est. cost: ${stats['total_cost_usd']:.4f})[/dim]"


def _safe_cli_error_message(operation: str, exc: BaseException) -> str:
    summary = summarize_exception(exc, operation=operation, provider="unknown")
    return format_error_message(
        operation=operation,
        error_type=summary["error_type"],
        message_preview=summary["message_preview"],
    )


def suppress_all_logging():
    """Suppress INFO and below for quiet mode; WARNING/ERROR/CRITICAL still surface."""
    logging.getLogger().setLevel(logging.WARNING)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.WARNING)
    for logger_name in [
        "httpx",
        "openai",
        "httpcore",
        "langchain",
        "langgraph",
        "google",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Suppress structlog INFO chatter (used by token_tracker and all src/ modules)
    # but keep WARNING and above so LLM failures and data-source errors are visible.
    # processors MUST NOT be empty: an empty chain passes the raw event dict as
    # **kwargs to PrintLogger.msg(), which only accepts positional args → TypeError.
    import structlog

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.KeyValueRenderer(
                key_order=["timestamp", "level", "event"]
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,  # False: force-apply to already-imported loggers
    )

    import warnings

    warnings.filterwarnings("ignore")


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
                "error": _safe_cli_error_message(
                    f"provider preflight for {provider}",
                    exc,
                ),
            }
            logger.warning(
                "provider_preflight",
                provider=provider,
                dns="failed",
                **summarize_exception(
                    exc,
                    operation=f"provider preflight for {provider}",
                    provider="unknown",
                ),
            )
    return results


def build_runtime_services_from_config(
    *,
    enable_tool_audit: bool,
    provider_runtime=None,
):
    """Build runtime-scoped services for a CLI/worker/app process."""
    from src.runtime_services import build_runtime_services_from_config as _build

    return _build(
        config,
        enable_tool_audit=enable_tool_audit,
        provider_runtime=provider_runtime,
        logger=logger,
    )


def configure_content_inspection_from_config(*, provider_runtime=None):
    """Build runtime services with content inspection wired from config."""
    return build_runtime_services_from_config(
        enable_tool_audit=False,
        provider_runtime=provider_runtime,
    )


def _resolve_langfuse_session_id(default_session_id: str) -> str:
    """Resolve the session ID, honoring batch/session overrides."""
    return os.getenv("LANGFUSE_SESSION_ID") or default_session_id


def _build_analysis_trace_tags(quick_mode: bool) -> list[str]:
    """Return stable tags for an analysis trace."""
    return [
        "analysis",
        "quick" if quick_mode else "full",
        f"quick-model:{config.quick_think_llm}",
        f"deep-model:{config.deep_think_llm}",
        f"memory:{'on' if config.enable_memory else 'off'}",
        f"consultant:{'on' if config.enable_consultant else 'off'}",
        f"auditor:{'on' if bool(config.auditor_model) else 'off'}",
    ]


def _build_analysis_trace_metadata(
    *,
    ticker: str,
    session_id: str,
    quick_mode: bool,
) -> dict[str, Any]:
    """Return the stable metadata attached to an analysis trace."""
    return {
        "ticker": ticker,
        "session_id": session_id,
        "environment": config.environment,
        "run_mode": "quick" if quick_mode else "full",
        "quick_mode": quick_mode,
        "deep_model": config.deep_think_llm,
        "quick_model": config.quick_think_llm,
        "prompt_source": (
            "langfuse" if config.langfuse_prompt_fetch_enabled else "local"
        ),
        "release": config.app_release,
    }


def configure_cli_logging(args) -> dict[str, dict[str, str]]:
    """Configure CLI logging without globally enabling dependency debug output."""
    mode = cli._cli_logging_mode(args)
    if mode in {"quiet", "brief"}:
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
    return run_provider_preflight() if enable_diagnostics else {}


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
        )
        from src.ticker_policy import get_ticker_suffix

        suffix = get_ticker_suffix(ticker)
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
        logger.debug(
            "market_context_fetch_failed",
            **summarize_exception(
                e,
                operation="fetching market context",
                provider="unknown",
            ),
        )
    return ""


async def _prefetch_macro_context(
    ticker: str,
    trade_date: str,
    *,
    callbacks: list[Any] | None = None,
) -> dict[str, Any]:
    """Load macro context with a deterministic failed fallback."""
    default_result = {
        "report": "",
        "region": "GLOBAL",
        "status": "failed",
        "generated_at": None,
        "llm_invoked": False,
        "prompt_used": None,
    }

    try:
        from src.macro_context import get_macro_context

        macro_context = await get_macro_context(
            ticker,
            trade_date,
            callbacks=callbacks,
        )
        result = {
            "report": macro_context.report,
            "region": macro_context.region,
            "status": macro_context.status,
            "generated_at": macro_context.generated_at,
            "llm_invoked": macro_context.llm_invoked,
            "prompt_used": macro_context.prompt_used,
        }
        logger.info(
            "macro_context_prefetch_complete",
            ticker=ticker,
            trade_date=trade_date,
            region=result["region"],
            status=result["status"],
            llm_invoked=result["llm_invoked"],
            generated_at=result["generated_at"],
            prompt_recorded=bool(result["prompt_used"]),
        )
        return result
    except Exception as exc:
        logger.warning(
            "macro_context_prefetch_failed",
            ticker=ticker,
            **summarize_exception(
                exc,
                operation="prefetching macro context",
                provider="unknown",
            ),
        )
        return default_result


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
    node_observer: Any | None = None,
    session_id: str | None = None,
    tracing_callbacks: list[Any] | None = None,
    tracing_metadata: dict[str, Any] | None = None,
    runtime_services: Any | None = None,
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
        from src.runtime_services import use_runtime_services
        from src.token_tracker import get_tracker

        with (
            use_runtime_services(runtime_services)
            if runtime_services
            else nullcontext()
        ):
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
            from src.ticker_utils import (
                _company_name_lookup_candidates,
                get_ticker_info,
                resolve_company_name,
            )

            name_result = await resolve_company_name(ticker)
            company_name = name_result.name

            if not name_result.is_resolved:
                logger.warning(
                    "company_name_unresolved_at_startup",
                    ticker=ticker,
                    requested_ticker=ticker,
                    lookup_candidates=[
                        symbol
                        for symbol, _strategy in _company_name_lookup_candidates(ticker)
                    ],
                    message="No source could resolve company name — LLM hallucination risk",
                )

            # Fetch benchmark context once (non-blocking) before graph starts.
            # Prepended to the HumanMessage so every agent receives it as session context.
            market_context = await _fetch_market_context(ticker, real_date)
            # The macro brief remains advisory News Analyst context, but its LLM call
            # should still flow through the same callback-based cost/tracing surface.
            macro_context = await _prefetch_macro_context(
                ticker,
                real_date,
                callbacks=tracing_callbacks,
            )
            macro_context_report = macro_context["report"]
            macro_context_region = macro_context["region"]
            macro_context_status = macro_context["status"]
            macro_context_generated_at = macro_context["generated_at"]
            macro_context_llm_invoked = macro_context["llm_invoked"]
            macro_context_prompt_used = macro_context["prompt_used"]

            session_id = _resolve_langfuse_session_id(
                session_id or f"{ticker}-{real_date}-{uuid.uuid4().hex[:8]}"
            )
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
                node_observer=node_observer,
            )

            _tinfo = get_ticker_info(ticker)
            _exch = _tinfo.get("exchange_name", "")
            _exch_note = (
                f" [{_exch}]"
                if _exch and _exch not in ("Unknown", "US Exchange (assumed)")
                else ""
            )
            _base_msg = f"Analyze {ticker}{_exch_note} ({company_name}) for investment decision. Current Date: {real_date}."
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
                macro_context_injected_into_news=False,
            )

            context = TradingContext(
                ticker=ticker,
                trade_date=real_date,
                quick_mode=quick_mode,
                enable_memory=config.enable_memory,
                max_debate_rounds=1 if quick_mode else 2,
                max_risk_rounds=1,
                macro_context_report=macro_context_report,
                macro_context_region=macro_context_region,
                macro_context_status=macro_context_status,
            )

            logger.info(
                "multi_agent_analysis_starting", ticker=ticker, trade_date=real_date
            )

            tags = _build_analysis_trace_tags(quick_mode)
            graph_metadata = (
                dict(tracing_metadata)
                if tracing_metadata is not None
                else _build_analysis_trace_metadata(
                    ticker=ticker,
                    session_id=session_id,
                    quick_mode=quick_mode,
                )
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
                        "callbacks": tracing_callbacks or [],
                        "tags": tags,
                        "metadata": graph_metadata,
                    },
                )
            finally:
                if capture_token is not None:
                    reset_active_capture_manager(capture_token)

            logger.info("analysis_completed", ticker=ticker)

            # Log token usage summary
            from src.token_tracker import get_tracker

            tracker = get_tracker()
            tracker.print_summary()

            if isinstance(result, dict):
                result["macro_context_report"] = macro_context_report
                result["macro_context_region"] = macro_context_region
                result["macro_context_status"] = macro_context_status
                result["macro_context_generated_at"] = macro_context_generated_at
                result["macro_context_llm_invoked"] = macro_context_llm_invoked
                result["macro_context_injected_into_news"] = bool(
                    result.get("macro_context_injected_into_news", False)
                )
                if macro_context_llm_invoked and macro_context_prompt_used:
                    prompts_used = dict(result.get("prompts_used", {}) or {})
                    prompts_used["macro_context_analyst"] = macro_context_prompt_used
                    result["prompts_used"] = prompts_used
                result["analysis_validity"] = build_analysis_validity(result)

            return result

    except Exception as e:
        if baseline_capture:
            baseline_capture.reject_run(
                [f"analysis_exception:{type(e).__name__}"],
                stage="run_analysis",
            )
        logger.error(
            "analysis_failed",
            ticker=ticker,
            **summarize_exception(
                e,
                operation="running analysis",
                provider="unknown",
            ),
            exc_info=True,
        )
        console.print(
            f"\n[bold red]{_safe_cli_error_message('running analysis', e)}[/bold red]\n"
        )
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
    if getattr(args, "enable_langfuse", False) or getattr(
        args, "trace_langfuse", False
    ):
        config.langfuse_enabled = True


def _setup_runtime(
    args: argparse.Namespace, output_targets: cli.OutputTargets
) -> tuple[dict[str, dict[str, str]], Any]:
    """Configure logging, runtime paths, and environment validation."""
    _enable_quiet_runtime_if_needed(args)
    config.images_dir = output_targets.image_dir

    provider_preflight = configure_cli_logging(args)
    enable_tool_audit = cli._cli_logging_mode(args) in {"verbose", "debug"}

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
        message = _safe_cli_error_message("validating environment configuration", exc)
        if args.quiet or args.brief:
            print(f"# Configuration Error\n\n{message}")
        else:
            console.print(f"\n[bold red]Configuration Error:[/bold red] {message}\n")
            console.print(
                "Please check your .env file and ensure all required API keys are set.\n"
            )
        raise SystemExit(1) from exc

    # Content inspection is configured independently of logging verbosity.
    try:
        runtime_services = build_runtime_services_from_config(
            enable_tool_audit=enable_tool_audit,
        )
    except ValueError as exc:
        message = _safe_cli_error_message("building runtime services", exc)
        if args.quiet or args.brief:
            print(f"# Configuration Error\n\n{message}")
        else:
            console.print(f"\n[bold red]Configuration Error:[/bold red] {message}\n")
        raise SystemExit(1) from exc

    return provider_preflight, runtime_services


async def _run_retrospective_only(args: argparse.Namespace) -> int:
    """Run retrospective-only mode and return the process exit code."""
    try:
        from src.observability import flush_traces, get_observability_runtime
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

        runtime = get_observability_runtime(config)
        trace_context = runtime.start_retrospective_trace(
            ticker="all",
            session_id=_resolve_langfuse_session_id(
                f"retrospective-all-{datetime.now().strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:8]}"
            ),
            tags=["retrospective", "batch"],
            metadata={
                "ticker": "all",
                "environment": config.environment,
                "run_mode": "retrospective_only",
                "release": config.app_release,
            },
            input_payload={"workflow": "retrospective_batch"},
        )
        try:
            lessons = await run_retrospective(
                ticker=None,
                results_dir=results_dir,
                progress=report if not (args.quiet or args.brief) else None,
            )
        finally:
            trace_context.close()
            flush_traces()

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
        logger.error(
            "retrospective_failed",
            **summarize_exception(
                exc,
                operation="running retrospective batch",
                provider="unknown",
            ),
            exc_info=True,
        )
        if not args.quiet and not args.brief:
            console.print(
                f"[yellow]Warning: {_safe_cli_error_message('running retrospective batch', exc)}[/yellow]"
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
        from src.observability import flush_traces, get_observability_runtime
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
        runtime = get_observability_runtime(config)
        trace_context = runtime.start_retrospective_trace(
            ticker=args.ticker,
            session_id=_resolve_langfuse_session_id(
                f"retrospective-{args.ticker}-{datetime.now().strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:8]}"
            ),
            tags=["retrospective", "single_ticker"],
            metadata={
                "ticker": args.ticker,
                "environment": config.environment,
                "run_mode": "retrospective_single_ticker",
                "release": config.app_release,
            },
            input_payload={"ticker": args.ticker, "workflow": "retrospective_single"},
        )
        try:
            lessons = await run_retrospective(
                ticker=args.ticker,
                results_dir=results_dir,
                progress=report,
            )
        finally:
            trace_context.close()
            flush_traces()
        if lessons and not args.quiet and not args.brief:
            console.print(
                f"\n[green]Generated {len(lessons)} new lesson(s) from past analyses[/green]"
            )
    except Exception as exc:
        logger.warning(
            "retrospective_failed",
            ticker=args.ticker,
            **summarize_exception(
                exc,
                operation="running single-ticker retrospective",
                provider="unknown",
            ),
        )


async def _execute_analysis(
    args: argparse.Namespace,
    output_targets: cli.OutputTargets,
    baseline_capture: BaselineCaptureManager | None = None,
    *,
    runtime_services: Any,
    session_id: str | None = None,
    tracing_callbacks: list[Any] | None = None,
    tracing_metadata: dict[str, Any] | None = None,
) -> dict | None:
    """Run the analysis graph for the current CLI request."""
    scoped_runtime_services = runtime_services
    if baseline_capture:
        capture_hook = baseline_capture.make_tool_hook()
        scoped_runtime_services = runtime_services.with_extra_tool_hooks([capture_hook])

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
        session_id=session_id,
        tracing_callbacks=tracing_callbacks,
        tracing_metadata=tracing_metadata,
        runtime_services=scoped_runtime_services,
    )


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
        logger.error(
            "baseline_capture_finalize_failed",
            **summarize_exception(
                exc,
                operation="finalizing baseline capture",
                provider="unknown",
            ),
            exc_info=True,
        )
        return None


def _attach_run_summary(
    result: dict,
    args: argparse.Namespace,
    provider_preflight: dict[str, dict[str, str]],
) -> None:
    """Attach the compact run summary before any persistence/output steps."""
    result.setdefault("run_summary", {})
    result["run_summary"]["quick_mode"] = bool(args.quick)
    result["analysis_validity"] = build_analysis_validity(result)
    result["run_summary"] = persistence.build_run_summary(
        result,
        quick_mode=args.quick,
        article_requested=bool(args.article),
        provider_preflight=provider_preflight,
    )


def _score_analysis_trace(result: dict, trace_context: Any) -> None:
    """Attach high-signal trace scores after analysis completion."""
    if not getattr(trace_context, "enabled", False):
        return

    from src.charts.extractors.pm_block import (
        extract_pm_block,
        extract_verdict_from_text,
    )

    pm_output = result.get("final_trade_decision", "") or ""
    pm_data = extract_pm_block(pm_output)
    verdict = pm_data.verdict or extract_verdict_from_text(pm_output) or "UNKNOWN"
    pre_screening = result.get("pre_screening_result", "")
    validity = (result.get("analysis_validity") or {}).get("publishable", False)

    trace_context.score_trace(
        name="pm_verdict",
        value=verdict,
        data_type="CATEGORICAL",
        comment=f"ticker={result.get('company_of_interest') or 'unknown'}",
    )
    trace_context.score_trace(
        name="pre_screening_pass",
        value=1.0 if pre_screening == "PASS" else 0.0,
        data_type="BOOLEAN",
    )
    trace_context.score_trace(
        name="analysis_validity",
        value=1.0 if validity else 0.0,
        data_type="BOOLEAN",
    )


def _log_final_summary(
    result: dict, args: argparse.Namespace, article_generated: bool
) -> None:
    """Emit the structured end-of-run summary log."""
    logger.info(
        "analysis_run_summary",
        ticker=args.ticker,
        **{**result.get("run_summary", {}), "article_generated": article_generated},
    )


async def main() -> int:
    """Main entry point for the application."""
    return await run_with_args(cli.parse_arguments())


async def run_with_args(
    args: argparse.Namespace,
    *,
    perform_capture_preflight: bool = True,
    capture_preflight_override: BaselinePreflightResult | None = None,
) -> int:
    """Run the analysis CLI flow for already-parsed arguments."""
    from src.async_utils import install_pending_task_dump_handler

    # `kill -USR1 <pid>` will dump every pending asyncio task with stack so
    # operators can diagnose hangs without restarting the process.
    _uninstall_dump_handler = install_pending_task_dump_handler()
    try:
        _apply_runtime_overrides(args)
        cli._validate_cli_args(args)
        output_targets = cli._resolve_output_targets(args)
        provider_preflight, runtime_services = _setup_runtime(args, output_targets)
        baseline_capture = _create_baseline_capture_manager(args)

        if baseline_capture is not None and capture_preflight_override is not None:
            baseline_capture.apply_preflight_result(
                git_clean=capture_preflight_override.git_clean,
                cleanup_summary=capture_preflight_override.cleanup_summary,
            )

        preflight_ok = True
        if perform_capture_preflight:
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

        from src.runtime_services import use_runtime_services

        if args.retrospective_only:
            with use_runtime_services(runtime_services):
                return await _run_retrospective_only(args)

        with use_runtime_services(runtime_services):
            await _maybe_run_ticker_retrospective(args)
        welcome_banner = output._emit_start_banner(
            args,
            output_targets,
            logger_obj=logger,
            print_fn=print,
            welcome_banner_fn=output.get_welcome_banner,
        )
        from src.observability import flush_traces, get_observability_runtime

        default_session_id = f"{args.ticker}-{datetime.now().strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:8]}"
        session_id = _resolve_langfuse_session_id(default_session_id)
        trace_tags = _build_analysis_trace_tags(args.quick)
        trace_metadata = _build_analysis_trace_metadata(
            ticker=args.ticker,
            session_id=session_id,
            quick_mode=args.quick,
        )
        trace_runtime = get_observability_runtime(config)
        trace_context = trace_runtime.start_analysis_trace(
            ticker=args.ticker,
            session_id=session_id,
            tags=trace_tags,
            metadata=trace_metadata,
            input_payload={
                "ticker": args.ticker,
                "quick_mode": args.quick,
                "workflow": "analysis",
            },
        )

        try:
            with use_runtime_services(runtime_services):
                if baseline_capture is None:
                    result = await _execute_analysis(
                        args,
                        output_targets,
                        runtime_services=runtime_services,
                        session_id=session_id,
                        tracing_callbacks=trace_context.callbacks,
                        tracing_metadata=trace_context.graph_metadata,
                    )
                else:
                    result = await _execute_analysis(
                        args,
                        output_targets,
                        baseline_capture=baseline_capture,
                        runtime_services=runtime_services,
                        session_id=session_id,
                        tracing_callbacks=trace_context.callbacks,
                        tracing_metadata=trace_context.graph_metadata,
                    )

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
                    output._report_analysis_failure(args, console_obj=console)
                    return 1

                _attach_run_summary(result, args, provider_preflight)
                _score_analysis_trace(result, trace_context)
                capture_path = _finalize_baseline_capture(baseline_capture, result)
                _print_capture_result(args, baseline_capture, capture_path)
                company_name_loader = partial(
                    output._load_company_name_for_output,
                    thread_pool_executor_cls=ThreadPoolExecutor,
                )
                company_name, report, reporter = output._render_primary_output(
                    result,
                    args,
                    output_targets,
                    welcome_banner,
                    console_obj=console,
                    logger_obj=logger,
                    company_name_loader=company_name_loader,
                    display_results_fn=output.display_results,
                    cost_suffix_fn=_cost_suffix,
                )
                persistence._persist_analysis_outputs(
                    result,
                    args,
                    trace_id=trace_context.trace_id,
                    logger_obj=logger,
                    console_obj=console,
                    cost_suffix_fn=_cost_suffix,
                    error_message_formatter=_safe_cli_error_message,
                )
                await persistence._maybe_save_rejection_record(
                    result,
                    args,
                    trace_id=trace_context.trace_id,
                    logger_obj=logger,
                )
                article_generated = await output._maybe_generate_article(
                    result,
                    args,
                    output_targets,
                    company_name,
                    report,
                    reporter,
                    tracing_callbacks=trace_context.callbacks,
                    tracing_metadata={
                        **trace_context.graph_metadata,
                        "workflow": "article",
                        "source_trace_id": trace_context.trace_id,
                    },
                    logger_obj=logger,
                    console_obj=console,
                    company_name_loader=company_name_loader,
                    handle_article_generation_fn=partial(
                        output.handle_article_generation,
                        logger_obj=logger,
                        console_obj=console,
                        error_message_formatter=_safe_cli_error_message,
                    ),
                )
        finally:
            trace_context.close()
            if trace_context.trace_url:
                logger.info(
                    "langfuse_trace_ready",
                    trace_id=trace_context.trace_id,
                    trace_url=trace_context.trace_url,
                )
            flush_traces()

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
        logger.error(
            "unexpected_error",
            **summarize_exception(
                exc,
                operation="running CLI entrypoint",
                provider="unknown",
            ),
            exc_info=True,
        )
        message = _safe_cli_error_message("running CLI entrypoint", exc)
        if args and (getattr(args, "quiet", False) or getattr(args, "brief", False)):
            print(f"# Unexpected Error\n\n{message}")
        else:
            console.print(f"\n[bold red]Unexpected error:[/bold red] {message}\n")
        return 1
    finally:
        try:
            from src.cleanup import cleanup_async_resources

            await cleanup_async_resources()
        except Exception:
            pass
        try:
            _uninstall_dump_handler()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
