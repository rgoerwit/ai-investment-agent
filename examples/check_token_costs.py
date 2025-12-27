0  #!/usr/bin/env python3
"""
Example script demonstrating programmatic access to token tracking.

Usage:
    poetry run python examples/check_token_costs.py results/0005_HK_*.json
"""

import json
import sys
from pathlib import Path


def analyze_token_costs(json_files: list[Path]):
    """Analyze token costs from saved analysis JSON files."""

    total_cost = 0.0
    total_tokens = 0
    total_analyses = 0

    agent_totals: dict[str, dict[str, float]] = {}

    print("=" * 80)
    print("TOKEN USAGE & COST ANALYSIS (Paid Tier Rates)")
    print("=" * 80)
    print("NOTE: Costs assume GCP project with billing enabled.")
    print("      If using free tier (no billing), actual cost = $0")
    print("=" * 80)
    print()

    for filepath in json_files:
        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            continue

        with open(filepath) as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        token_usage = data.get("token_usage", {})

        if not token_usage:
            print(f"⚠️  No token usage data in {filepath.name}")
            continue

        ticker = metadata.get("ticker", "UNKNOWN")
        analysis_cost = token_usage.get("total_cost_usd", 0.0)
        analysis_tokens = token_usage.get("total_tokens", 0)

        total_cost += analysis_cost
        total_tokens += analysis_tokens
        total_analyses += 1

        print(f"📊 {ticker} ({filepath.name})")
        print(f"   Tokens: {analysis_tokens:,}")
        print(f"   Cost: ${analysis_cost:.4f}")
        print()

        # Aggregate per-agent costs
        agents = token_usage.get("agents", {})
        for agent_name, agent_stats in agents.items():
            if agent_name not in agent_totals:
                agent_totals[agent_name] = {"tokens": 0, "cost": 0.0, "calls": 0}

            agent_totals[agent_name]["tokens"] += agent_stats.get("total_tokens", 0)
            agent_totals[agent_name]["cost"] += agent_stats.get("cost_usd", 0.0)
            agent_totals[agent_name]["calls"] += agent_stats.get("calls", 0)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Analyses: {total_analyses}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Projected Cost (Paid Tier): ${total_cost:.4f}")

    if total_analyses > 0:
        print(f"Average Tokens per Analysis: {total_tokens / total_analyses:,.0f}")
        print(f"Average Cost per Analysis: ${total_cost / total_analyses:.4f}")

    print()
    print("=" * 80)
    print("PER-AGENT BREAKDOWN (All Analyses)")
    print("=" * 80)

    # Sort by cost descending
    sorted_agents = sorted(
        agent_totals.items(), key=lambda x: x[1]["cost"], reverse=True
    )

    print(f"{'Agent':<25} {'Calls':>8} {'Tokens':>15} {'Cost':>12}")
    print("-" * 80)

    for agent_name, stats in sorted_agents:
        print(
            f"{agent_name:<25} "
            f"{stats['calls']:>8} "
            f"{stats['tokens']:>15,} "
            f"${stats['cost']:>11.4f}"
        )

    print("=" * 80)
    print()

    # Cost optimization recommendations
    print("💡 COST OPTIMIZATION TIPS")
    print("=" * 80)

    # Find most expensive agent
    if sorted_agents:
        most_expensive = sorted_agents[0]
        print(
            f"• Most expensive agent: {most_expensive[0]} (${most_expensive[1]['cost']:.4f})"
        )

        if (
            "Portfolio Manager" in most_expensive[0]
            or "Research Manager" in most_expensive[0]
        ):
            print("  → Consider switching DEEP_MODEL to gemini-2.0-flash-exp (free)")

    if total_analyses > 0:
        avg_cost = total_cost / total_analyses
        print(f"• Average cost per analysis (paid tier): ${avg_cost:.4f}")
        if avg_cost > 0.10:
            print("  → Try using --quick flag to reduce debate rounds")
            print("  → Try using --brief flag to reduce output tokens")
            print("  → Consider gemini-2.5-flash-lite for 70% cost reduction")
        else:
            print("  ✅ Cost per analysis is well optimized for paid tier")

    print()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python examples/check_token_costs.py <json_file1> [json_file2] ..."
        )
        print()
        print("Examples:")
        print("  # Single file")
        print(
            "  python examples/check_token_costs.py results/0005_HK_20251205_analysis.json"
        )
        print()
        print("  # All analyses for a ticker")
        print("  python examples/check_token_costs.py results/0005_HK_*.json")
        print()
        print("  # All analyses in results directory")
        print("  python examples/check_token_costs.py results/*.json")
        sys.exit(1)

    # Expand globs and convert to Path objects
    json_files = []
    for pattern in sys.argv[1:]:
        path = Path(pattern)
        if "*" in pattern:
            # Handle glob patterns
            parent = path.parent if path.parent.exists() else Path(".")
            glob_pattern = path.name
            json_files.extend(parent.glob(glob_pattern))
        else:
            json_files.append(path)

    if not json_files:
        print("❌ No JSON files found matching the pattern")
        sys.exit(1)

    analyze_token_costs(json_files)


if __name__ == "__main__":
    main()
