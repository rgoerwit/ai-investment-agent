#!/usr/bin/env python3
"""
Main entry point for the Multi-Agent Investment Analysis System.
Updated for Gemini 3 (Nov 2025).
"""

import argparse

import os

# Chroma telemetry is just a pain; need to catch this early
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

# Suppress gRPC fork warnings; if gRPC skips fork handlers, so be it
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1" # handle forks gracefully
os.environ["GRPC_POLL_STRATEGY"] = "poll"

import sys
import asyncio
import logging
import structlog
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from datetime import datetime
from pathlib import Path
import json

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
    for logger_name in ['httpx', 'openai', 'httpcore', 'langchain', 'langgraph', 'google']:
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
    warnings.filterwarnings('ignore')


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
        """
    )
    
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol to analyze (e.g., AAPL, NVDA, TSLA)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick analysis mode (faster, less detailed)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all logging and output only final markdown report"
    )
    
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Output only header, executive summary, and decision rationale"
    )
    
    parser.add_argument(
        "--quick-model",
        type=str,
        default=None,
        help=f"Model to use for quick analysis (default: {config.quick_think_llm})"
    )
    
    parser.add_argument(
        "--deep-model",
        type=str,
        default=None,
        help=f"Model to use for deep analysis (default: {config.deep_think_llm})"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable persistent memory (ChromaDB)"
    )
    
    return parser.parse_args()


def display_welcome_banner(ticker: str, quick_mode: bool):
    """Display welcome banner with configuration."""
    title = "Multi-Agent Investment Analysis System (Gemini Powered)"
    
    config_table = Table(show_header=False, box=box.SIMPLE)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Ticker", ticker.upper())
    config_table.add_row("Analysis Mode", "Quick" if quick_mode else "Deep")
    config_table.add_row("Quick Model", config.quick_think_llm)
    config_table.add_row("Deep Model", config.deep_think_llm)
    config_table.add_row("Memory System", "Enabled" if config.enable_memory else "Disabled")
    config_table.add_row("LangSmith Tracing", "Enabled" if config.langsmith_tracing_enabled else "Disabled")
    
    console.print(Panel(config_table, title=title, border_style="blue"))


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
            ("Portfolio Manager", f"{safe_ticker}_risk_manager_memory")
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

    if stats['total_calls'] == 0:
        return

    console.print("\n[bold cyan]Token Usage Summary:[/bold cyan]\n")

    # Overall stats table
    summary_table = Table(show_header=True, box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Total LLM Calls", str(stats['total_calls']))
    summary_table.add_row("Total Prompt Tokens", f"{stats['total_prompt_tokens']:,}")
    summary_table.add_row("Total Completion Tokens", f"{stats['total_completion_tokens']:,}")
    summary_table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
    summary_table.add_row("Projected Cost (Paid Tier)", f"${stats['total_cost_usd']:.4f}")

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
        stats['agents'].items(),
        key=lambda x: x[1]['cost_usd'],
        reverse=True
    )

    for agent_name, agent_stats in sorted_agents:
        agent_table.add_row(
            agent_name,
            str(agent_stats['calls']),
            f"{agent_stats['prompt_tokens']:,}",
            f"{agent_stats['completion_tokens']:,}",
            f"{agent_stats['total_tokens']:,}",
            f"${agent_stats['cost_usd']:.4f}"
        )

    console.print(agent_table)
    console.print()


def display_results(result: dict, ticker: str):
    """Display analysis results in a formatted manner."""
    console.print("\n" + "="*80)
    console.print("[bold green]Analysis Complete![/bold green]\n")

    # Display token usage first
    display_token_summary()
    
    # Display final trading decision
    if "final_trade_decision" in result and result["final_trade_decision"]:
        decision_panel = Panel(
            result["final_trade_decision"],
            title="Final Trading Decision",
            border_style="green",
            padding=(1, 2)
        )
        console.print(decision_panel)
    
    # Display individual analyst reports
    console.print("\n[bold cyan]Analyst Reports:[/bold cyan]\n")
    
    report_fields = [
        ("market_report", "Market Analysis"),
        ("sentiment_report", "Sentiment Analysis"),
        ("news_report", "News Analysis"),
        ("fundamentals_report", "Fundamentals Analysis"),
        ("investment_plan", "Investment Plan"),
        ("trader_investment_plan", "Trading Proposal")
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
                content,
                title=f"{display_name}",
                border_style=style,
                padding=(1, 2)
            )
            console.print(report_panel)
            console.print()
    
    display_memory_statistics(ticker)
    console.print("="*80 + "\n")


def save_results_to_file(result: dict, ticker: str) -> Path:
    """Save analysis results to a JSON file in the results directory."""
    from src.prompts import get_all_prompts
    from src.memory import create_memory_instances, sanitize_ticker_for_collection
    
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
            "requires_tools": prompt.requires_tools
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
                "bull_researcher": memories.get(f"{safe_ticker}_bull_memory").get_stats(),
                "bear_researcher": memories.get(f"{safe_ticker}_bear_memory").get_stats(),
                "research_manager": memories.get(f"{safe_ticker}_invest_judge_memory").get_stats(),
                "trader": memories.get(f"{safe_ticker}_trader_memory").get_stats(),
                "portfolio_manager": memories.get(f"{safe_ticker}_risk_manager_memory").get_stats()
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
            "llm_provider": config.llm_provider
        },
        "token_usage": token_stats,
        "prompts_metadata": {
            "prompts_used": prompts_used,
            "available_prompts": available_prompts,
            "custom_prompts_loaded": custom_prompts_loaded,
            "prompts_directory": str(prompts_dir),
            "total_agents": len(prompts_used),
            "note": "system_message field contains the actual prompt text used by each agent"
        },
        "memory_statistics": memory_stats,
        "reports": {
            "market_report": result.get("market_report", ""),
            "sentiment_report": result.get("sentiment_report", ""),
            "news_report": result.get("news_report", ""),
            "fundamentals_report": result.get("fundamentals_report", "")
        },
        "investment_analysis": {
            "investment_debate": {
                "bull_history": result.get("investment_debate_state", {}).get("bull_history", ""),
                "bear_history": result.get("investment_debate_state", {}).get("bear_history", ""),
                "debate_rounds": result.get("investment_debate_state", {}).get("count", 0)
            },
            "investment_plan": result.get("investment_plan", ""),
            "trader_plan": result.get("trader_investment_plan", "")
        },
        "risk_analysis": {
            "risk_debate": {
                "risky_perspective": result.get("risk_debate_state", {}).get("current_risky_response", ""),
                "safe_perspective": result.get("risk_debate_state", {}).get("current_safe_response", ""),
                "neutral_perspective": result.get("risk_debate_state", {}).get("current_neutral_response", ""),
                "debate_rounds": result.get("risk_debate_state", {}).get("count", 0)
            }
        },
        "final_decision": {
            "decision": result.get("final_trade_decision", ""),
            "processed_signal": None
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)

    logger.info(f"Results saved to {filepath} ({len(prompts_used)} prompts tracked, {len(custom_prompts_loaded)} custom)")

    # Log token tracking info
    if token_stats['total_calls'] > 0:
        logger.info(
            f"Token usage tracked: {token_stats['total_calls']} LLM calls, "
            f"{token_stats['total_tokens']:,} total tokens, "
            f"${token_stats['total_cost_usd']:.4f} projected cost (paid tier) - "
            f"saved to {filepath}"
        )

    return filepath


async def run_analysis(ticker: str, quick_mode: bool) -> Optional[dict]:
    """Run the multi-agent analysis workflow."""
    try:
        from src.graph import create_trading_graph, TradingContext
        from src.agents import AgentState, InvestDebateState, RiskDebateState
        from langchain_core.messages import HumanMessage
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
            company_name = info.get('longName') or info.get('shortName') or ticker
            logger.info(
                "company_name_verified",
                ticker=ticker,
                company_name=company_name,
                source="yfinance"
            )
        except Exception as e:
            logger.warning(
                "company_name_fetch_failed",
                ticker=ticker,
                error=str(e),
                fallback=ticker
            )

        graph = create_trading_graph(
            ticker=ticker,  # BUG FIX #1: Pass ticker for isolation
            cleanup_previous=True,  # BUG FIX #1: Cleanup to prevent contamination
            max_debate_rounds=1 if quick_mode else 2,
            max_risk_discuss_rounds=1,
            enable_memory=config.enable_memory,
            recursion_limit=100,
            quick_mode=quick_mode  # Pass quick_mode for consultant LLM selection
        )
        
        initial_state = AgentState(
            messages=[HumanMessage(content=f"Analyze {ticker} ({company_name}) for investment decision. Current Date: {real_date}")],
            company_of_interest=ticker,
            company_name=company_name,  # ADDED: Anchor verified company name in state
            trade_date=real_date,
            sender="user",
            market_report="",
            sentiment_report="",
            news_report="",
            fundamentals_report="",
            investment_debate_state=InvestDebateState(
                bull_history="",
                bear_history="",
                history="",
                current_response="",
                judge_decision="",
                count=0
            ),
            investment_plan="",
            trader_investment_plan="",
            risk_debate_state=RiskDebateState(
                risky_history="",
                safe_history="",
                neutral_history="",
                history="",
                latest_speaker="",
                current_risky_response="",
                current_safe_response="",
                current_neutral_response="",
                judge_decision="",
                count=0
            ),
            final_trade_decision="",
            tools_called={},
            prompts_used={},
            red_flags=[],
            pre_screening_result=""
        )
        
        context = TradingContext(
            ticker=ticker,
            trade_date=real_date,
            quick_mode=quick_mode,
            enable_memory=config.enable_memory,
            max_debate_rounds=1 if quick_mode else 2,
            max_risk_rounds=1
        )
        
        logger.info(f"Starting multi-agent analysis for {ticker} on {real_date}")
        
        result = await graph.ainvoke(
            initial_state,
            config={
                "recursion_limit": 100,
                "configurable": {
                    "context": context
                }
            }
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

            # Set environment variable for rate limit handler to check
            os.environ["QUIET_MODE"] = "true"

        if args.quick_model:
            config.quick_think_llm = args.quick_model
        if args.deep_model:
            config.deep_think_llm = args.deep_model
        
        if args.no_memory:
            config.enable_memory = False
        
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
                console.print("Please check your .env file and ensure all required API keys are set.\n")
            sys.exit(1)
        
        if not args.quiet and not args.brief:
            display_welcome_banner(args.ticker, args.quick)
        
        result = await run_analysis(args.ticker, args.quick)
        
        if result:
            if args.brief or args.quiet:
                company_name = None
                try:
                    import yfinance as yf
                    ticker_obj = yf.Ticker(args.ticker)
                    info = ticker_obj.info
                    company_name = info.get('longName') or info.get('shortName')
                except:
                    pass
                
                reporter = QuietModeReporter(args.ticker, company_name, quick_mode=args.quick)
                report = reporter.generate_report(result, brief_mode=args.brief)
                print(report)
            else:
                display_results(result, args.ticker)
            
            try:
                filepath = save_results_to_file(result, args.ticker)
                if not args.quiet and not args.brief:
                    console.print(f"\n[green]Results saved to:[/green] [cyan]{filepath}[/cyan]\n")
            except Exception as e:
                logger.error(f"Failed to save results: {e}")
                if not args.quiet and not args.brief:
                    console.print(f"\n[yellow]Warning: Could not save results to file: {e}[/yellow]\n")
            
            sys.exit(0)
        else:
            if args.quiet or args.brief:
                print("# Analysis Failed\n\nAn error occurred during analysis. Check logs for details.")
            else:
                console.print("\n[bold red]Analysis failed. Check logs for details.[/bold red]\n")
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


if __name__ == "__main__":
    asyncio.run(main())