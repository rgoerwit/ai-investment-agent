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
    recovery_cost: float | None = None
    recovery_calls: int | None = None
    cap_exhausted_attempts: int | None = None
    cap_exhausted_tokens: int | None = None
    pm_policy_corrections: int | None = None
    pm_trace_corrections: int | None = None
    research_budgets: dict[str, dict[str, Any]] | None = None
    run_fingerprint: dict[str, Any] | None = None


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
    attempts = tu.get("call_attempts")
    recovery = tu.get("recovery_usage")
    research_budgets = data.get("research_budgets")

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
        recovery_cost=(
            sum(float(row.get("cost_usd") or 0.0) for row in recovery)
            if isinstance(recovery, list)
            else None
        ),
        recovery_calls=(
            sum(int(row.get("calls") or 0) for row in recovery)
            if isinstance(recovery, list)
            else None
        ),
        cap_exhausted_attempts=(
            sum(
                1
                for attempt in attempts
                if attempt.get("failure_kind") == "output_cap_exhausted"
            )
            if isinstance(attempts, list)
            else None
        ),
        cap_exhausted_tokens=(
            sum(
                int(attempt.get("total_tokens") or 0)
                for attempt in attempts
                if attempt.get("failure_kind") == "output_cap_exhausted"
            )
            if isinstance(attempts, list)
            else None
        ),
        pm_policy_corrections=(
            sum(
                1
                for attempt in attempts
                if "policy correction" in str(attempt.get("agent_name", "")).casefold()
            )
            if isinstance(attempts, list)
            else None
        ),
        pm_trace_corrections=(
            sum(
                1
                for attempt in attempts
                if "trace correction" in str(attempt.get("agent_name", "")).casefold()
            )
            if isinstance(attempts, list)
            else None
        ),
        research_budgets=(
            research_budgets if isinstance(research_budgets, dict) else None
        ),
        run_fingerprint=(
            data["run_fingerprint"]
            if isinstance(data.get("run_fingerprint"), dict)
            else None
        ),
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


def _sum_available(runs: list[RunCost], field_name: str) -> tuple[float, int]:
    values = [getattr(run, field_name) for run in runs]
    available = [float(value) for value in values if value is not None]
    return sum(available), len(available)


def format_efficiency_report(runs: list[RunCost]) -> str:
    """Summarize recovery, cap, correction, and research-budget activity."""

    if not runs:
        return "No analysis runs matched."
    total_cost = sum(run.total_cost for run in runs)
    recovery_cost, recovery_available = _sum_available(runs, "recovery_cost")
    recovery_calls, _ = _sum_available(runs, "recovery_calls")
    cap_attempts, cap_available = _sum_available(runs, "cap_exhausted_attempts")
    cap_tokens, _ = _sum_available(runs, "cap_exhausted_tokens")
    policy_corrections, correction_available = _sum_available(
        runs, "pm_policy_corrections"
    )
    trace_corrections, _ = _sum_available(runs, "pm_trace_corrections")
    lines = ["Efficiency", "-" * 56]
    recovery_share = (
        f"{100 * recovery_cost / total_cost:.1f}%" if total_cost > 0 else "n/a"
    )
    lines.append(
        f"  recovery: {int(recovery_calls)} call(s), ${recovery_cost:.4f}, "
        f"{recovery_share} of spend"
        if recovery_available
        else "  recovery: unavailable in these artifacts"
    )
    lines.append(
        f"  output-cap attempts: {int(cap_attempts)}, {int(cap_tokens):,} tokens"
        if cap_available
        else "  output-cap attempts: unavailable in these artifacts"
    )
    lines.append(
        "  PM model corrections: "
        f"policy={int(policy_corrections)}, trace={int(trace_corrections)}"
        if correction_available
        else "  PM model corrections: unavailable in these artifacts"
    )

    research_runs = [run for run in runs if run.research_budgets is not None]
    if not research_runs:
        lines.append("  research budgets: unavailable in these artifacts")
        return "\n".join(lines)
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"llm_calls": 0, "tool_rounds": 0, "forced": 0, "violations": 0}
    )
    for run in research_runs:
        for agent, telemetry in (run.research_budgets or {}).items():
            row = totals[agent]
            row["llm_calls"] += int(telemetry.get("llm_calls") or 0)
            row["tool_rounds"] += int(telemetry.get("tool_rounds_used") or 0)
            row["forced"] += int(bool(telemetry.get("forced_synthesis_used")))
            outcomes = telemetry.get("outcomes") or []
            row["violations"] += sum(
                1
                for outcome in outcomes
                if "EXHAUSTED" in str(outcome) or "LIMIT" in str(outcome)
            )
    for agent, row in sorted(totals.items()):
        lines.append(
            f"  {agent}: llm={row['llm_calls']}, rounds={row['tool_rounds']}, "
            f"forced={row['forced']}, limits={row['violations']}"
        )
    return "\n".join(lines)


def _basket_overlap_note(
    baseline: list[RunCost], candidate: list[RunCost]
) -> list[str]:
    """Warn when the two run sets are not measuring the same names.

    Cost per run is only a like-for-like number when both sides analysed the
    same tickers: a Japanese small cap with sparse filings and a Mexican
    large cap do not cost the same to analyse, so a diff across disjoint
    baskets mostly reports basket composition and reads as if it reported a
    code change. Advisory only -- comparing a full basket against a quick one
    is a legitimate tier-vs-tier use, so this never changes the exit code.
    """
    base_tickers = {r.ticker for r in baseline}
    cand_tickers = {r.ticker for r in candidate}
    notes: list[str] = []
    if not base_tickers or not cand_tickers:
        return notes
    overlap = base_tickers & cand_tickers
    if not overlap:
        notes.extend(
            [
                f"NOTE: baseline and candidate share no tickers "
                f"({len(base_tickers)} vs {len(cand_tickers)}). Per-run cost reflects "
                "basket composition as much as any code change -- read the totals as "
                "tier cost, not as a per-name delta.",
                "",
            ]
        )
    elif len(overlap) < min(len(base_tickers), len(cand_tickers)):
        notes.extend(
            [
                f"NOTE: baskets overlap on {len(overlap)} of "
                f"{len(base_tickers)}/{len(cand_tickers)} tickers -- partial comparison.",
                "",
            ]
        )

    base_fp = [run.run_fingerprint for run in baseline if run.run_fingerprint]
    cand_fp = [run.run_fingerprint for run in candidate if run.run_fingerprint]
    if not base_fp or not cand_fp:
        notes.extend(["NOTE: run-fingerprint control check unavailable.", ""])
        return notes
    for key, label in (
        ("code_commit", "code commit"),
        ("prompt_set_digest", "prompt set"),
        ("thesis_digest", "thesis configuration"),
    ):
        base_values = {str(fp.get(key)) for fp in base_fp}
        cand_values = {str(fp.get(key)) for fp in cand_fp}
        if base_values != cand_values or len(base_values) != 1 or len(cand_values) != 1:
            notes.extend(
                [f"NOTE: not controlled — {label} differs within/across sets.", ""]
            )
    if any(bool(fp.get("code_dirty")) for fp in [*base_fp, *cand_fp]):
        notes.extend(["NOTE: at least one compared run used a dirty worktree.", ""])
    if {run.quick_mode for run in baseline} != {run.quick_mode for run in candidate}:
        notes.extend(["NOTE: not controlled — quick/full mode differs.", ""])
    return notes


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
        *_basket_overlap_note(baseline, candidate),
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
    parser.add_argument(
        "--efficiency",
        action="store_true",
        help="include recovery, output-cap, correction, and research-budget activity",
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
            report = diff_report(base, cand, args.by)
            if args.efficiency:
                report += "\n\nBaseline " + format_efficiency_report(base)
                report += "\n\nCandidate " + format_efficiency_report(cand)
            print(report)
        return 0

    results_dir = args.results_dir or _default_results_dir()
    runs = discover_runs(results_dir, since=args.since, tickers=tickers)
    if args.json:
        totals, _ = aggregate(runs, args.by)
        print(json.dumps({"runs": len(runs), f"by_{args.by}": totals}, indent=2))
    else:
        report = format_report(runs, args.by)
        if args.efficiency:
            report += "\n\n" + format_efficiency_report(runs)
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
