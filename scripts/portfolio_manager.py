#!/usr/bin/env python3
"""
IBKR Portfolio Reconciliation Tool

Compares live IBKR positions against the equity evaluator's latest
analysis recommendations. Produces position-aware BUY/SELL/HOLD/TRIM/REVIEW
actions that account for existing holdings.

Usage:
    python scripts/portfolio_manager.py                      # Report only
    python scripts/portfolio_manager.py --recommend          # + order suggestions
    python scripts/portfolio_manager.py --execute            # + place orders (with confirmation)
    python scripts/portfolio_manager.py --read-only          # No IBKR connection (offline)

Requires: poetry install -E ibkr
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ibkr.exceptions import IBKRAuthError, IBKRError
from src.ibkr.models import (
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
)
from src.ibkr.order_builder import build_order_dict
from src.ibkr.reconciler import load_latest_analyses, reconcile
from src.ibkr.ticker_mapper import resolve_conid


def _prompt_for_missing_secret(config) -> None:
    """Prompt for OAuth token secret if absent. Held in memory only — never written to disk."""
    if not config.get_oauth_access_token_secret():
        from pydantic import SecretStr

        secret = getpass.getpass("IBKR OAuth Access Token Secret: ")
        if not secret:
            print("No secret provided. Cannot connect to IBKR.", file=sys.stderr)
            sys.exit(1)
        config.ibkr_oauth_access_token_secret = SecretStr(secret)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile IBKR portfolio against evaluator recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--report-only",
        action="store_true",
        help="Show reconciliation report (default)",
    )
    mode_group.add_argument(
        "--recommend",
        action="store_true",
        help="Report + concrete order suggestions with sizes/prices",
    )
    mode_group.add_argument(
        "--execute",
        action="store_true",
        help="Report + recommendations + place orders (with per-order confirmation)",
    )

    # Options
    parser.add_argument(
        "--max-age", type=int, default=14, help="Max analysis age in days (default: 14)"
    )
    parser.add_argument(
        "--drift-pct",
        type=float,
        default=15.0,
        help="Price drift threshold %% (default: 15)",
    )
    parser.add_argument(
        "--cash-buffer",
        type=float,
        default=0.05,
        help="Cash reserve fraction (default: 0.05)",
    )
    parser.add_argument(
        "--refresh-stale", action="store_true", help="Re-run evaluator on stale tickers"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Use quick mode for re-analysis"
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Never create IBKR connection (offline mode)",
    )
    parser.add_argument(
        "--account-id", type=str, default="", help="Override IBKR account ID"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/", help="Override results directory"
    )
    parser.add_argument(
        "--output", type=str, default="", help="Write report to file (default: stdout)"
    )
    parser.add_argument("--json", action="store_true", help="Structured JSON output")
    parser.add_argument("--debug", action="store_true", help="Debug output")

    args = parser.parse_args()

    # --recommend and --execute override --report-only
    if args.recommend or args.execute:
        args.report_only = False

    return args


# ══════════════════════════════════════════════════════════════════════════════
# Report Formatting
# ══════════════════════════════════════════════════════════════════════════════

ACTION_SYMBOLS = {
    "BUY": "+",
    "SELL": "!",
    "TRIM": "~",
    "HOLD": " ",
    "REVIEW": "?",
}

URGENCY_COLORS = {
    "HIGH": "!!",
    "MEDIUM": " !",
    "LOW": "  ",
}


def format_report(
    items: list[ReconciliationItem],
    portfolio: PortfolioSummary,
    show_recommendations: bool = False,
) -> str:
    """Format reconciliation results as human-readable text."""
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"=== IBKR Portfolio Reconciliation ({now}) ===")
    lines.append("")
    lines.append(
        f"Account: {portfolio.account_id} | "
        f"Portfolio: ${portfolio.portfolio_value_usd:,.0f} | "
        f"Cash: ${portfolio.cash_balance_usd:,.0f} ({portfolio.cash_pct:.1f}%) | "
        f"Available: ${portfolio.available_cash_usd:,.0f}"
    )
    lines.append("")

    # Held positions
    held_items = [i for i in items if i.ibkr_position is not None]
    if held_items:
        lines.append("--- POSITIONS vs EVALUATOR ---")
        lines.append("")
        for item in held_items:
            pos = item.ibkr_position
            sym = ACTION_SYMBOLS.get(item.action, " ")
            urg = URGENCY_COLORS.get(item.urgency, "  ")

            price_info = ""
            if pos and pos.current_price_local > 0 and pos.avg_cost_local > 0:
                gain = (
                    (pos.current_price_local - pos.avg_cost_local) / pos.avg_cost_local
                ) * 100
                price_info = f" {int(pos.quantity)} @ {pos.avg_cost_local:.2f} → {pos.current_price_local:.2f} ({gain:+.1f}%)"

            verdict_info = ""
            if item.analysis:
                a = item.analysis
                verdict_info = f"  {a.verdict} ({a.analysis_date})"
                if a.health_adj or a.growth_adj:
                    verdict_info += f"  H:{a.health_adj or '?'} G:{a.growth_adj or '?'}"

            lines.append(f"{urg} [{sym}] {item.ticker}{price_info}{verdict_info}")
            lines.append(f"       {item.action} — {item.reason}")

            if show_recommendations and item.suggested_quantity:
                order_type = item.suggested_order_type or "LMT"
                price_str = (
                    f" @ {item.suggested_price:.2f}" if item.suggested_price else ""
                )
                lines.append(
                    f"       → {item.action} {item.suggested_quantity} shares ({order_type}){price_str}"
                )

            lines.append("")

    # New buy recommendations (not held)
    buy_items = [i for i in items if i.ibkr_position is None and i.action == "BUY"]
    if buy_items:
        lines.append("--- NEW BUY RECOMMENDATIONS (not held) ---")
        lines.append("")
        for item in buy_items:
            a = item.analysis
            if a:
                size_pct = a.trade_block.size_pct or a.position_size or 0
                lines.append(
                    f"  {item.ticker}  BUY ({a.analysis_date})  Size:{size_pct:.1f}%  Entry:{a.entry_price or '?'}"
                )
                if show_recommendations and item.suggested_price:
                    lines.append(f"       → BUY @ {item.suggested_price:.2f} (LMT)")
            else:
                lines.append(f"  {item.ticker}  BUY — {item.reason}")
            lines.append("")

    if not items:
        lines.append(
            "No reconciliation items. Portfolio is empty and no BUY recommendations found."
        )
        lines.append("")

    # Summary
    action_counts = {}
    for item in items:
        action_counts[item.action] = action_counts.get(item.action, 0) + 1
    summary_parts = [
        f"{action}: {count}" for action, count in sorted(action_counts.items())
    ]
    lines.append(f"--- Summary: {', '.join(summary_parts) or 'empty'} ---")

    return "\n".join(lines)


def format_json(
    items: list[ReconciliationItem],
    portfolio: PortfolioSummary,
) -> str:
    """Format reconciliation results as JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "portfolio": portfolio.model_dump(),
        "items": [item.model_dump() for item in items],
    }
    return json.dumps(data, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


async def refresh_stale_analysis(ticker: str, quick: bool = True) -> bool:
    """Re-run evaluator on a stale ticker."""
    try:
        from src.main import run_analysis, save_results_to_file

        print(f"  Re-analyzing {ticker} (quick={quick})...", file=sys.stderr)
        result = await run_analysis(ticker=ticker, quick_mode=quick, skip_charts=True)
        if result:
            save_results_to_file(result, ticker)
            print(f"  {ticker} re-analysis complete.", file=sys.stderr)
            return True
        print(f"  {ticker} re-analysis returned no result.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  {ticker} re-analysis failed: {e}", file=sys.stderr)
        return False


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    # Load analyses from disk (always works, no IBKR needed)
    analyses = load_latest_analyses(results_dir)
    if not analyses:
        print(f"No analysis JSONs found in {results_dir}/", file=sys.stderr)
        print(
            "Run some analyses first: poetry run python -m src.main --ticker 7203.T --output results/7203.T.md",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded {len(analyses)} analyses from {results_dir}/", file=sys.stderr)

    # Read IBKR portfolio (or use empty if --read-only)
    positions: list[NormalizedPosition] = []
    portfolio = PortfolioSummary()

    if args.read_only:
        print("Read-only mode: no IBKR connection", file=sys.stderr)
        # In read-only mode, just show analyses status
        portfolio = PortfolioSummary(
            portfolio_value_usd=0,
            cash_balance_usd=0,
            available_cash_usd=0,
        )
    else:
        try:
            from src.ibkr.client import IbkrClient
            from src.ibkr.portfolio import read_portfolio
            from src.ibkr_config import ibkr_config

            _prompt_for_missing_secret(ibkr_config)
            client = IbkrClient(ibkr_config)
            client.connect(brokerage_session=args.execute)

            account_id = args.account_id or ibkr_config.ibkr_account_id
            positions, portfolio = read_portfolio(
                client,
                account_id,
                args.cash_buffer,
            )
            client.close()

        except ImportError:
            print(
                "ibind not installed. Run: poetry install -E ibkr\n"
                "Or use --read-only for offline mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        except IBKRAuthError as e:
            print(f"IBKR auth error: {e}", file=sys.stderr)
            print("Check IBKR credentials in .env or use --read-only", file=sys.stderr)
            sys.exit(1)
        except IBKRError as e:
            print(f"IBKR error: {e}", file=sys.stderr)
            sys.exit(1)

    # Reconcile
    items = reconcile(
        positions=positions,
        analyses=analyses,
        portfolio=portfolio,
        max_age_days=args.max_age,
        drift_threshold_pct=args.drift_pct,
    )

    # Handle stale refreshes
    if args.refresh_stale:
        stale_tickers = [
            item.ticker
            for item in items
            if item.action == "REVIEW" and "stale" in (item.reason or "").lower()
        ]
        if stale_tickers:
            print(
                f"\nRefreshing {len(stale_tickers)} stale analyses...", file=sys.stderr
            )
            for ticker in stale_tickers:
                asyncio.run(refresh_stale_analysis(ticker, quick=args.quick))

            # Reload and re-reconcile
            analyses = load_latest_analyses(results_dir)
            items = reconcile(
                positions=positions,
                analyses=analyses,
                portfolio=portfolio,
                max_age_days=args.max_age,
                drift_threshold_pct=args.drift_pct,
            )

    # Output
    show_recs = args.recommend or args.execute

    if args.json:
        output = format_json(items, portfolio)
    else:
        output = format_report(items, portfolio, show_recommendations=show_recs)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Order execution (Phase 5 — with per-order confirmation)
    if args.execute:
        actionable = [
            i
            for i in items
            if i.action in ("BUY", "SELL", "TRIM") and i.suggested_quantity
        ]
        if not actionable:
            print("\nNo actionable orders to execute.", file=sys.stderr)
            return

        print(f"\n{len(actionable)} orders ready for execution:", file=sys.stderr)
        for item in actionable:
            print(
                f"  {item.action} {item.ticker}: {item.suggested_quantity} shares",
                file=sys.stderr,
            )

        from src.ibkr.client import IbkrClient
        from src.ibkr_config import ibkr_config

        _prompt_for_missing_secret(ibkr_config)
        exec_client = IbkrClient(ibkr_config)
        exec_client.connect(brokerage_session=True)

        try:
            for item in actionable:
                prompt = (
                    f"\nExecute {item.action} {item.suggested_quantity} {item.ticker}"
                    f" @ {item.suggested_price or 'MKT'}? [y/N]: "
                )
                confirm = input(prompt).strip().lower()
                if confirm != "y":
                    print(f"  Skipped {item.ticker}", file=sys.stderr)
                    continue

                try:
                    conid = resolve_conid(item.ticker, exec_client)
                    if not conid:
                        print(
                            f"  Failed to resolve conid for {item.ticker}",
                            file=sys.stderr,
                        )
                        continue

                    order = build_order_dict(
                        conid=conid,
                        action=item.action if item.action != "TRIM" else "SELL",
                        quantity=item.suggested_quantity,
                        price=item.suggested_price,
                        order_type=item.suggested_order_type,
                        account_id=portfolio.account_id,
                    )
                    result = exec_client.place_order(portfolio.account_id, order)
                    print(
                        f"  Order placed for {item.ticker}: {result}", file=sys.stderr
                    )

                except Exception as e:
                    print(f"  Order failed for {item.ticker}: {e}", file=sys.stderr)
        finally:
            exec_client.close()


if __name__ == "__main__":
    main()
