#!/usr/bin/env python
"""Cost report over saved analysis artifacts — make every cheap-model lever measurable.

Reads ``results/*_analysis.json`` (read-only) and ranks LLM spend by agent,
provider, model, or service tier, and can **diff two run sets** to quantify a
lever's dollar impact (e.g. flex on vs off, model A vs B).

Consumes the ``token_usage`` rollups added July 2026 (``by_provider`` /
``by_model`` / ``by_tier`` / ``unpriced_models``). For older artifacts that
predate those fields it falls back to the per-agent ``cost_usd`` rows and a
keyword agent→provider map (labeled approximate).

Examples:
    poetry run python scripts/cost_report.py --since 2026-07-20 --by model
    poetry run python scripts/cost_report.py --ticker 6782.TW --by tier
    poetry run python scripts/cost_report.py \\
        --baseline results/flex_off --candidate results/flex_on
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- keyword agent→provider fallback (only for pre-rollup artifacts) ----------
_OPENAI_AGENT_MARKERS = ("consultant", "auditor", "accountant", "apac")


def _provider_from_agent_name(name: str) -> str:
    low = name.lower()
    if any(marker in low for marker in _OPENAI_AGENT_MARKERS):
        return "openai_compatible"
    if "writer" in low or "editor" in low:
        return "mixed"  # writer=anthropic, editor=openai — can't split by name
    return "google"


@dataclass
class RunCost:
    """One analysis run's cost, flattened to {dimension: {key: cost_usd}}."""

    path: str
    ticker: str
    date: str
    quick_mode: bool | None
    total_cost: float
    by_agent: dict[str, float]
    by_provider: dict[str, float]
    by_model: dict[str, float] | None  # None = pre-rollup artifact
    by_tier: dict[str, float] | None
    unpriced_models: list[str] = field(default_factory=list)
    approximate_provider: bool = False


def _costs(bucket: dict[str, Any]) -> dict[str, float]:
    """Extract {key: cost_usd} from a rollup bucket {key: {cost_usd, ...}}."""
    return {k: float(v.get("cost_usd", 0.0)) for k, v in (bucket or {}).items()}


def load_run(path: str | Path) -> RunCost | None:
    """Parse one analysis JSON into a RunCost, or None if it has no token usage."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    tu = data.get("token_usage") or {}
    agents = tu.get("agents") or {}
    if not agents:
        return None
    meta = data.get("metadata") or {}
    run_summary = data.get("run_summary") or {}

    by_agent = {name: float(row.get("cost_usd", 0.0)) for name, row in agents.items()}

    if tu.get("by_provider"):
        by_provider = _costs(tu["by_provider"])
        approximate = False
    else:
        by_provider = defaultdict(float)
        for name, cost in by_agent.items():
            by_provider[_provider_from_agent_name(name)] += cost
        by_provider = dict(by_provider)
        approximate = True

    return RunCost(
        path=str(path),
        ticker=str(meta.get("ticker", "?")),
        date=str(meta.get("analysis_date", "")),
        quick_mode=run_summary.get("quick_mode"),
        total_cost=float(tu.get("total_cost_usd", sum(by_agent.values()))),
        by_agent=by_agent,
        by_provider=by_provider,
        by_model=_costs(tu["by_model"]) if tu.get("by_model") else None,
        by_tier=_costs(tu["by_tier"]) if tu.get("by_tier") else None,
        unpriced_models=list(tu.get("unpriced_models") or []),
        approximate_provider=approximate,
    )


def discover_runs(
    results_dir: str | Path,
    *,
    since: str | None = None,
    tickers: set[str] | None = None,
) -> list[RunCost]:
    """Load every ``*_analysis.json`` under results_dir matching the filters."""
    runs: list[RunCost] = []
    for path in sorted(glob.glob(os.path.join(str(results_dir), "*_analysis.json"))):
        run = load_run(path)
        if run is None:
            continue
        if since and run.date[:10] < since:
            continue
        if tickers and run.ticker not in tickers:
            continue
        runs.append(run)
    return runs


def _dimension(run: RunCost, by: str) -> dict[str, float] | None:
    return {
        "agent": run.by_agent,
        "provider": run.by_provider,
        "model": run.by_model,
        "tier": run.by_tier,
    }[by]


def aggregate(runs: list[RunCost], by: str) -> tuple[dict[str, float], int]:
    """Sum cost per key over runs; returns (totals, n_runs_with_that_dimension)."""
    totals: dict[str, float] = defaultdict(float)
    counted = 0
    for run in runs:
        dim = _dimension(run, by)
        if dim is None:  # pre-rollup artifact lacks this dimension
            continue
        counted += 1
        for key, cost in dim.items():
            totals[key] += cost
    return dict(totals), counted


def _fmt_table(totals: dict[str, float], n_runs: int, grand: float) -> list[str]:
    lines = []
    for key, cost in sorted(totals.items(), key=lambda kv: -kv[1]):
        pct = (100 * cost / grand) if grand else 0.0
        per_run = cost / n_runs if n_runs else 0.0
        lines.append(f"  {key:32s} ${per_run:8.4f}/run  {pct:5.1f}%")
    return lines


def format_report(runs: list[RunCost], by: str) -> str:
    if not runs:
        return "No analysis runs matched."
    total_cost = sum(r.total_cost for r in runs)
    n = len(runs)
    totals, counted = aggregate(runs, by)
    grand = sum(totals.values())
    out = [
        f"{n} run(s); mean ${total_cost / n:.4f}/run; total ${total_cost:.4f}",
        "",
        f"By {by}"
        + (f" ({counted}/{n} runs carry this dimension)" if counted < n else ""),
        "-" * 56,
    ]
    if by in ("model", "tier") and counted == 0:
        out.append("  (no artifacts carry this rollup — re-run after the A4 change)")
    else:
        out += _fmt_table(totals, n, grand)
    if any(r.approximate_provider for r in runs) and by == "provider":
        out.append("  * provider split approximate for pre-rollup artifacts")
    unpriced = sorted({m for r in runs for m in r.unpriced_models})
    if unpriced:
        out += ["", f"⚠ unpriced models (cost fabricated at default rate): {unpriced}"]
    return "\n".join(out)


def diff_report(baseline: list[RunCost], candidate: list[RunCost], by: str) -> str:
    """A/B: per-key cost/run delta between two run sets (candidate − baseline)."""

    def per_run(runs: list[RunCost]) -> tuple[float, dict[str, float]]:
        n = max(len(runs), 1)
        totals, _ = aggregate(runs, by)
        return (
            sum(r.total_cost for r in runs) / n,
            {k: v / n for k, v in totals.items()},
        )

    base_total, base = per_run(baseline)
    cand_total, cand = per_run(candidate)
    keys = sorted(set(base) | set(cand))
    out = [
        f"baseline: {len(baseline)} run(s), mean ${base_total:.4f}/run",
        f"candidate: {len(candidate)} run(s), mean ${cand_total:.4f}/run",
        f"Δ total: ${cand_total - base_total:+.4f}/run "
        f"({100 * (cand_total - base_total) / base_total:+.1f}%)"
        if base_total
        else f"Δ total: ${cand_total - base_total:+.4f}/run",
        "",
        f"Δ by {by} (candidate − baseline, $/run)",
        "-" * 56,
    ]
    for key in sorted(
        keys, key=lambda k: abs(cand.get(k, 0) - base.get(k, 0)), reverse=True
    ):
        delta = cand.get(key, 0.0) - base.get(key, 0.0)
        out.append(f"  {key:32s} ${delta:+8.4f}/run")
    return "\n".join(out)


def _default_results_dir() -> str:
    try:
        from src.config import config

        return str(config.results_dir)
    except Exception:
        return "results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--since", default=None, help="YYYY-MM-DD (by analysis_date)")
    parser.add_argument(
        "--ticker", action="append", default=None, help="repeatable ticker filter"
    )
    parser.add_argument(
        "--by",
        choices=["agent", "provider", "model", "tier"],
        default="agent",
    )
    parser.add_argument("--baseline", default=None, help="A/B baseline results dir")
    parser.add_argument("--candidate", default=None, help="A/B candidate results dir")
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    args = parser.parse_args(argv)

    tickers = set(args.ticker) if args.ticker else None

    if bool(args.baseline) != bool(args.candidate):
        parser.error("--baseline and --candidate must be given together")

    if args.baseline:
        base = discover_runs(args.baseline, since=args.since, tickers=tickers)
        cand = discover_runs(args.candidate, since=args.since, tickers=tickers)
        if args.json:
            print(
                json.dumps(
                    {
                        "baseline": aggregate(base, args.by)[0],
                        "candidate": aggregate(cand, args.by)[0],
                    },
                    indent=2,
                )
            )
        else:
            print(diff_report(base, cand, args.by))
        return 0

    results_dir = args.results_dir or _default_results_dir()
    runs = discover_runs(results_dir, since=args.since, tickers=tickers)
    if args.json:
        totals, _ = aggregate(runs, args.by)
        print(json.dumps({"runs": len(runs), f"by_{args.by}": totals}, indent=2))
    else:
        print(format_report(runs, args.by))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
