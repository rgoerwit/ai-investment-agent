#!/usr/bin/env python3
"""
Main entry point for the Multi-Agent Investment Analysis System.
Updated for Gemini 3 (Nov 2025).
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import structlog
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import config FIRST to set telemetry/system env vars before any library imports
from src.config import config, validate_environment_variables
from src.report_generator import QuietModeReporter

# IMPORTANT: Don't import get_tracker here - it instantiates the singleton immediately
# Import it lazily in functions that need it, after quiet mode is set

logger = structlog.get_logger(__name__)
console = Console()


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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Investment Analysis System (Gemini 3 Edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python -m src.main --ticker AAPL

  # Quick analysis mode (Gemini Flash)
  python -m src.main --ticker NVDA --quick

  # Quiet mode (markdown report only)
  python -m src.main --ticker AAPL --quiet

  # Brief mode (header, summary, decision only)
  python -m src.main --ticker AAPL --brief

  # Custom models
  python -m src.main --ticker TSLA --quick-model gemini-2.5-flash --deep-model gemini-3-pro-preview

  # With Poetry
  poetry run python -m src.main --ticker MSFT --quick
        """,
    )

    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol to analyze (e.g., AAPL, NVDA, TSLA)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick analysis mode (faster, less detailed)",
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

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

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

    return parser.parse_args()


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


async def handle_article_generation(
    args,
    ticker: str,
    company_name: str,
    report_text: str,
    trade_date: str,
) -> None:
    """
    Generate article if --article flag is set.

    Args:
        args: Parsed arguments namespace
        ticker: Stock ticker symbol
        company_name: Full company name
        report_text: The full analysis report
        trade_date: Date of the analysis
    """
    article_path = resolve_article_path(args, ticker)
    if not article_path:
        return

    try:
        from src.article_writer import ArticleWriter

        if not args.quiet and not args.brief:
            console.print("\n[cyan]Generating article...[/cyan]")

        # Default to local paths so markdown renders immediately in editors
        # Users who want GitHub URLs can set GITHUB_RAW_BASE env var
        writer = ArticleWriter(use_github_urls=False)
        article = writer.write(
            ticker=ticker,
            company_name=company_name,
            report_text=report_text,
            trade_date=trade_date,
            output_path=article_path,
        )

        if not args.quiet and not args.brief:
            console.print(
                f"[green]Article saved to:[/green] [cyan]{article_path}[/cyan]"
            )
            # Defensive: ensure article is a string before counting words
            word_count = len(article.split()) if isinstance(article, str) else 0
            console.print(f"[dim]Word count: {word_count} words[/dim]")

    except Exception as e:
        logger.error(f"Article generation failed: {e}", exc_info=True)
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
        logger.warning(f"Could not display memory statistics: {e}")


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


def save_results_to_file(result: dict, ticker: str) -> Path:
    """Save analysis results to a JSON file in the results directory."""
    from src.memory import create_memory_instances, sanitize_ticker_for_collection
    from src.prompts import get_all_prompts

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

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
            # Get actual memories for THIS ticker
            memories = create_memory_instances(ticker)
            safe_ticker = sanitize_ticker_for_collection(ticker)

            memory_stats = {
                "bull_researcher": memories.get(
                    f"{safe_ticker}_bull_memory"
                ).get_stats(),
                "bear_researcher": memories.get(
                    f"{safe_ticker}_bear_memory"
                ).get_stats(),
                "research_manager": memories.get(
                    f"{safe_ticker}_invest_judge_memory"
                ).get_stats(),
                "trader": memories.get(f"{safe_ticker}_trader_memory").get_stats(),
                "portfolio_manager": memories.get(
                    f"{safe_ticker}_risk_manager_memory"
                ).get_stats(),
            }
        except Exception as e:
            logger.warning(f"Could not get memory stats: {e}")

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
            "llm_provider": config.llm_provider,
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
    }

    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=2)

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


async def run_analysis(ticker: str, quick_mode: bool) -> dict | None:
    """Run the multi-agent analysis workflow."""
    try:
        from langchain_core.messages import HumanMessage

        from src.agents import AgentState, InvestDebateState, RiskDebateState
        from src.graph import TradingContext, create_trading_graph
        from src.token_tracker import get_tracker

        # Reset token tracker for fresh analysis
        tracker = get_tracker()
        tracker.reset()

        logger.info(f"Starting analysis for {ticker} (quick_mode={quick_mode})")

        # CRITICAL FIX: Enforce real-world date to prevent "Time Travel" hallucinations
        # This overrides potentially stale system prompts or environment defaults
        real_date = datetime.now().strftime("%Y-%m-%d")

        # CRITICAL FIX: Fetch and verify company name BEFORE graph execution
        # This prevents LLM hallucination when tickers are similar (e.g., 0291.HK vs 0293.HK)
        company_name = ticker  # Default fallback
        try:
            import yfinance as yf

            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            company_name = info.get("longName") or info.get("shortName") or ticker
            logger.info(
                "company_name_verified",
                ticker=ticker,
                company_name=company_name,
                source="yfinance",
            )
        except Exception as e:
            logger.warning(
                "company_name_fetch_failed",
                ticker=ticker,
                error=str(e),
                fallback=ticker,
            )

        graph = create_trading_graph(
            ticker=ticker,  # BUG FIX #1: Pass ticker for isolation
            cleanup_previous=True,  # BUG FIX #1: Cleanup to prevent contamination
            max_debate_rounds=1 if quick_mode else 2,
            max_risk_discuss_rounds=1,
            enable_memory=config.enable_memory,
            recursion_limit=100,
            quick_mode=quick_mode,  # Pass quick_mode for consultant LLM selection
        )

        initial_state = AgentState(
            messages=[
                HumanMessage(
                    content=f"Analyze {ticker} ({company_name}) for investment decision. Current Date: {real_date}"
                )
            ],
            company_of_interest=ticker,
            company_name=company_name,  # ADDED: Anchor verified company name in state
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
            red_flags=[],
            pre_screening_result="",
        )

        context = TradingContext(
            ticker=ticker,
            trade_date=real_date,
            quick_mode=quick_mode,
            enable_memory=config.enable_memory,
            max_debate_rounds=1 if quick_mode else 2,
            max_risk_rounds=1,
        )

        logger.info(f"Starting multi-agent analysis for {ticker} on {real_date}")

        result = await graph.ainvoke(
            initial_state,
            config={"recursion_limit": 100, "configurable": {"context": context}},
        )

        logger.info(f"Analysis completed for {ticker}")

        # Log token usage summary
        from src.token_tracker import get_tracker

        tracker = get_tracker()
        tracker.print_summary()

        return result

    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {str(e)}", exc_info=True)
        console.print(f"\n[bold red]Error during analysis:[/bold red] {str(e)}\n")
        return None


async def main():
    """Main entry point for the application."""
    args = None
    try:
        args = parse_arguments()

        # Import cleanup module (will be used in finally block)
        from src.cleanup import cleanup_async_resources

        if args.quiet or args.brief:
            # Suppress token tracker logging BEFORE any imports that might initialize it
            # CRITICAL: Must set quiet mode before importing get_tracker() or any module that uses it
            from src.token_tracker import TokenTracker

            TokenTracker.set_quiet_mode(True)
            suppress_all_logging()

            # Force re-initialization with quiet mode active
            # (in case global_tracker was already imported elsewhere)
            tracker = TokenTracker()
            tracker._quiet_mode = True

            # Set config flag for rate limit handler to check
            config.quiet_mode = True

        if args.quick_model:
            config.quick_think_llm = args.quick_model
        if args.deep_model:
            config.deep_think_llm = args.deep_model

        if args.no_memory:
            config.enable_memory = False

        # --- Output and Image Directory Logic ---
        output_file, image_dir = resolve_output_paths(args)
        output_dir = output_file.parent if output_file else Path.cwd()

        config.images_dir = image_dir

        # Handle stdout case: suppress charts unless user explicitly requested imagedir
        if not output_file and not args.no_charts and not args.imagedir:
            # Disable charts when writing to stdout (no way to link them)
            # Exception: if user specified --imagedir, they want images saved separately
            if not args.quiet and not args.brief:
                logger.warning(
                    "Writing to stdout: Charts disabled (no way to link them). "
                    "Use --output to enable charts, or --imagedir to save images separately."
                )
            args.no_charts = True

        # Validate image directory relative to output directory (for linking)
        # Only relevant if we are generating charts
        if not args.no_charts and output_file:
            try:
                # Check if image_dir is inside output_dir
                # We use absolute paths for the check to be robust
                abs_image_dir = image_dir.resolve()
                abs_output_dir = output_dir.resolve()

                # attempt to find relative path
                abs_image_dir.relative_to(abs_output_dir)
            except ValueError:
                # Not a subdirectory
                logger.warning(
                    f"Image directory ({image_dir}) is not a subdirectory of output directory ({output_dir}). "
                    "Report will contain absolute paths to images, which may not render correctly on other systems."
                )

        if args.verbose and not args.quiet and not args.brief:
            logging.getLogger().setLevel(logging.DEBUG)
            for name in logging.root.manager.loggerDict:
                logging.getLogger(name).setLevel(logging.DEBUG)

        try:
            validate_environment_variables()
        except ValueError as e:
            if args.quiet or args.brief:
                print(f"# Configuration Error\n\n{str(e)}")
            else:
                console.print(f"\n[bold red]Configuration Error:[/bold red] {str(e)}\n")
                console.print(
                    "Please check your .env file and ensure all required API keys are set.\n"
                )
            sys.exit(1)

        # Generate welcome banner
        welcome_banner = get_welcome_banner(args.ticker, args.quick)

        # If writing to stdout (no output file), print banner immediately unless quiet/brief
        if not output_file and not args.quiet and not args.brief:
            print(welcome_banner)

        # If writing to file, we will prepend the banner to the file content later
        # We might still want to log that analysis is starting, but use logger for that
        if output_file and not args.quiet and not args.brief:
            logger.info(
                f"Starting analysis for {args.ticker} (output to {output_file})"
            )

        result = await run_analysis(args.ticker, args.quick)

        if result:
            # Auto-detect non-TTY stdout (e.g., output redirected to file)
            # Use markdown output instead of Rich formatting to avoid box-drawing characters
            use_markdown = (
                args.brief or args.quiet or not sys.stdout.isatty() or args.output
            )

            # Initialize variables that may be needed for article generation
            company_name = None
            report = None

            if use_markdown:
                try:
                    import yfinance as yf

                    ticker_obj = yf.Ticker(args.ticker)
                    info = ticker_obj.info
                    company_name = info.get("longName") or info.get("shortName")
                except Exception:
                    pass

                reporter = QuietModeReporter(
                    args.ticker,
                    company_name,
                    quick_mode=args.quick,
                    chart_format="svg" if args.svg else "png",
                    transparent_charts=args.transparent,
                    skip_charts=args.no_charts,
                    image_dir=image_dir,
                    report_dir=output_dir,  # Pass output dir for relative link calculation
                    report_stem=Path(args.output).stem if args.output else None,
                )
                report = reporter.generate_report(result, brief_mode=args.brief)

                # Prepend welcome banner if we generated it and are writing to file (or stdout in full markdown mode)
                # But careful: if we already printed it (stdout case above), don't duplicate.
                # Logic:
                # 1. If output_file: Prepend banner to report content.
                # 2. If stdout: We already printed banner above. BUT generate_report creates a full markdown string.
                #    If we print that string, it's fine.

                if args.output:
                    # Prepend banner to file content
                    full_content = welcome_banner + "\n" + report

                    try:
                        # Ensure parent directory exists
                        if output_file.parent != Path("."):
                            output_file.parent.mkdir(parents=True, exist_ok=True)

                        with open(output_file, "w") as f:
                            f.write(full_content)

                        if not args.quiet and not args.brief:
                            console.print(
                                f"[green]Report saved to:[/green] [cyan]{output_file}[/cyan]"
                            )
                    except Exception as e:
                        logger.error(f"Failed to write report to {output_file}: {e}")
                        sys.exit(1)
                else:
                    # Writing to stdout
                    print(report)
            else:
                display_results(result, args.ticker)

            try:
                filepath = save_results_to_file(result, args.ticker)
                if not args.quiet and not args.brief:
                    console.print(
                        f"[green]Results saved to:[/green] [cyan]{filepath}[/cyan]"
                    )
            except Exception as e:
                logger.error(f"Failed to save results: {e}")
                if not args.quiet and not args.brief:
                    console.print(
                        f"\n[yellow]Warning: Could not save results to file: {e}[/yellow]\n"
                    )

            # Generate article if --article flag is set
            if args.article:
                # Warn if article will lack images (stdout mode disables charts)
                # Use stderr so warning is visible even in quiet mode
                if args.no_charts and not output_file and not args.imagedir:
                    print(
                        "Warning: Article generated without images (stdout mode).",
                        file=sys.stderr,
                    )
                # Get trade_date from result or current date
                trade_date = result.get("trade_date") or datetime.now().strftime(
                    "%Y-%m-%d"
                )

                # Generate report text for article if not already done
                if not use_markdown:
                    # Need to generate markdown report for article
                    if company_name is None:
                        try:
                            import yfinance as yf

                            ticker_obj = yf.Ticker(args.ticker)
                            info = ticker_obj.info
                            company_name = (
                                info.get("longName")
                                or info.get("shortName")
                                or args.ticker
                            )
                        except Exception:
                            company_name = args.ticker

                    reporter = QuietModeReporter(
                        args.ticker,
                        company_name,
                        quick_mode=args.quick,
                        chart_format="svg" if args.svg else "png",
                        transparent_charts=args.transparent,
                        skip_charts=args.no_charts,
                        image_dir=image_dir,
                        report_dir=output_dir,
                        report_stem=Path(args.output).stem if args.output else None,
                    )
                    report = reporter.generate_report(result, brief_mode=False)

                await handle_article_generation(
                    args=args,
                    ticker=args.ticker,
                    company_name=company_name or args.ticker,
                    report_text=report,
                    trade_date=trade_date,
                )

            sys.exit(0)
        else:
            if args.quiet or args.brief:
                print(
                    "# Analysis Failed\n\nAn error occurred during analysis. Check logs for details."
                )
            else:
                console.print(
                    "\n[bold red]Analysis failed. Check logs for details.[/bold red]\n"
                )
            sys.exit(1)

    except KeyboardInterrupt:
        if args and (args.quiet or args.brief):
            pass
        else:
            console.print("\n[yellow]Analysis interrupted by user.[/yellow]\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        if args and (args.quiet or args.brief):
            print(f"# Unexpected Error\n\n{str(e)}")
        else:
            console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}\n")
        sys.exit(1)
    finally:
        # Clean up async resources (aiohttp sessions, etc.)
        # This prevents "coroutine was never awaited" warnings
        try:
            from src.cleanup import cleanup_async_resources

            await cleanup_async_resources()
        except Exception:
            pass  # Cleanup errors shouldn't prevent exit


if __name__ == "__main__":
    asyncio.run(main())
