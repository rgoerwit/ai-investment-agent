#!/usr/bin/env python3
"""Longitudinal comparison report for repeatedly-analyzed tickers.

Read-only. Two independent things it can do, run either or both:

1. **Timeline diff** — for each ticker, gathers ``*_analysis.json`` artifacts
   from ``--results-dir`` (and optionally ``--archive-dir``) within a lookback
   window, extracts the deterministic fields that matter for a regression
   read (PM verdict, HEALTH_ADJ/GROWTH_ADJ, code-computed risk total,
   provenance contract status, consultant/auditor status, red-flag types),
   and prints a table per ticker with newly-appeared flags marked ``[NEW]``
   relative to the previous row.

2. **Run-log scan** (``--run-log``) — greps a batch run's log for the specific
   patterns worth separating from real regressions: the sandbox TLS/keychain
   warning, per-host DNS resolution failures (broken out by host/operation,
   not just a bare count), and consultant/auditor structural failures
   (``consultant_invalid_structure``, ``agent_output_truncated``,
   ``consultant_review_partial``).

This intentionally stops at the deterministic/structural layer. It will not
tell you whether a verdict flip was *justified* -- that still needs a human
(or an LLM) reading the actual PM rationale and consultant/auditor prose in
the flagged artifacts. What it removes is the manual glob-and-jq work to find
which artifacts and which flags are worth reading in the first place.

Usage:
    poetry run python scripts/eval_longitudinal_compare.py \\
        --tickers 1681.HK PINFRA.MX AGS.BR 7740.T 8002.T 1088.HK \\
        --lookback-weeks 6 --archive-dir ~/Developer/results_archive

    poetry run python scripts/eval_longitudinal_compare.py \\
        --tickers AGS.BR --run-log scratch/eval_rerun_*/run.log

    # Timeline only, default ticker set, markdown to a file:
    poetry run python scripts/eval_longitudinal_compare.py --output report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# The 6 tickers with the deepest run history reaching back to the corpus's
# earliest retained analyses (2025-12-01) -- see scripts/eval_rerun_longitudinal.sh.
DEFAULT_TICKERS = ["1681.HK", "PINFRA.MX", "AGS.BR", "7740.T", "8002.T", "1088.HK"]

_FILENAME_RE = re.compile(r"_(\d{8})_(\d{6})_analysis\.json$")


def _default_results_dir() -> str:
    try:
        from src.config import config

        return str(config.results_dir)
    except Exception:
        return "results"


# --------------------------------------------------------------------------
# Part 1: per-artifact field extraction (mirrors what the PM/consultant
# prompts actually emit -- see prompts/portfolio_manager.json's PM_BLOCK).
# --------------------------------------------------------------------------


def _get(d: dict, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


@dataclass
class RunRow:
    ticker: str
    timestamp: str
    path: str
    verdict: str | None = None
    health_adj: float | None = None
    growth_adj: float | None = None
    risk_total: float | None = None
    contract_status: str | None = None
    consultant_verdict: str | None = None
    auditor_status: str | None = None
    red_flag_types: list[str] = field(default_factory=list)

    @property
    def dt(self) -> datetime | None:
        try:
            return datetime.strptime(self.timestamp, "%Y%m%d_%H%M%S")
        except ValueError:
            return None


def extract_row(path: Path) -> RunRow:
    d = json.loads(path.read_text())
    m = _FILENAME_RE.search(path.name)
    ts = f"{m.group(1)}_{m.group(2)}" if m else "?"

    run_summary = _get(d, "run_summary", default={}) or {}
    final_decision = _get(d, "final_decision", default={}) or {}
    text = final_decision.get("decision") or ""
    if not isinstance(text, str):
        text = str(text)

    verdict_m = re.search(
        r"PORTFOLIO MANAGER VERDICT:\s*\[?([A-Z_/ ]+?)\]?\s*(?:\n|$)", text
    )
    verdict = verdict_m.group(1).strip() if verdict_m else None
    if verdict is None:
        action_m = re.search(r"\*\*Action\*\*:\s*([A-Z_/ ]+)", text)
        verdict = action_m.group(1).strip() if action_m else None

    health_m = re.search(r"Financial Health\*\*:\s*([\d.]+)\s*%", text)
    growth_m = re.search(r"Growth Transition\*\*:\s*([\d.]+)\s*%", text)
    risk_m = re.search(r"TOTAL RISK COUNT\*\*:\s*([\d.]+)", text)

    red_flags = d.get("red_flags")
    flag_types: list[str] = []
    if isinstance(red_flags, list):
        for f in red_flags:
            if isinstance(f, dict) and f.get("type"):
                flag_types.append(f["type"])
            elif isinstance(f, str):
                flag_types.append(f[:40])

    snapshot = d.get("analysis_snapshot")
    contract_status = None
    if isinstance(snapshot, dict):
        contract_status = snapshot.get("contract_status") or snapshot.get("status")

    return RunRow(
        ticker=d.get("metadata", {}).get("ticker") or path.stem.split("_")[0],
        timestamp=ts,
        path=str(path),
        verdict=verdict,
        health_adj=float(health_m.group(1)) if health_m else None,
        growth_adj=float(growth_m.group(1)) if growth_m else None,
        risk_total=float(risk_m.group(1)) if risk_m else None,
        contract_status=contract_status,
        consultant_verdict=run_summary.get("consultant_verdict"),
        auditor_status=run_summary.get("auditor_status"),
        red_flag_types=sorted(set(flag_types)),
    )


def discover_rows(
    ticker: str,
    results_dir: str,
    archive_dir: str | None,
    lookback_weeks: float,
    min_runs: int,
    max_lookback_weeks: float,
) -> list[RunRow]:
    """Widening-window discovery: start at lookback_weeks, widen up to
    max_lookback_weeks if fewer than min_runs artifacts are found. Sparse
    tickers (a single run in 6 weeks) still get a usable comparison instead
    of a table with one row."""
    dirs = [results_dir] + ([archive_dir] if archive_dir else [])
    all_paths: list[Path] = []
    for d in dirs:
        if not d:
            continue
        all_paths.extend(
            Path(p) for p in glob.glob(os.path.join(d, f"{ticker}_*_analysis.json"))
        )

    rows = []
    for p in all_paths:
        try:
            rows.append(extract_row(p))
        except (json.JSONDecodeError, OSError):
            continue
    rows = [r for r in rows if r.dt is not None]
    rows.sort(key=lambda r: r.dt)
    if not rows:
        return []

    latest = rows[-1].dt
    window = lookback_weeks
    while window <= max_lookback_weeks:
        cutoff = latest - timedelta(weeks=window)
        windowed = [r for r in rows if r.dt >= cutoff]
        if len(windowed) >= min_runs or window >= max_lookback_weeks:
            return windowed
        window *= 2
    return rows[-min_runs:]


def render_timeline_markdown(ticker: str, rows: list[RunRow]) -> str:
    if not rows:
        return f"### {ticker}\n\nNo artifacts found.\n"

    lines = [f"### {ticker}", ""]
    lines.append(
        "| Date | Verdict | Health | Growth | Risk | Contract | Consultant | Auditor | Flags |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    prev_flags: set[str] = set()
    for i, r in enumerate(rows):
        date_disp = f"{r.timestamp[4:6]}/{r.timestamp[6:8]}"
        if i == len(rows) - 1:
            date_disp += " **(latest)**"

        def fmt(v, suffix=""):
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:g}{suffix}"
            return f"{v}{suffix}"

        new_flags = set(r.red_flag_types) - prev_flags
        flag_cells = []
        for ft in r.red_flag_types:
            flag_cells.append(f"**{ft}[NEW]**" if ft in new_flags else ft)
        flags_disp = ", ".join(flag_cells) or "—"
        prev_flags = set(r.red_flag_types)

        lines.append(
            f"| {date_disp} | {fmt(r.verdict)} | {fmt(r.health_adj, '%')} | "
            f"{fmt(r.growth_adj, '%')} | {fmt(r.risk_total)} | {fmt(r.contract_status)} | "
            f"{fmt(r.consultant_verdict)} | {fmt(r.auditor_status)} | {flags_disp} |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Part 2: run-log scan (sandbox artifacts vs real consultant/auditor failures)
# --------------------------------------------------------------------------

_RE_ROOT_CERT = re.compile(r"failed to load native root certificate")
_RE_DNS_LINE = re.compile(
    r"event='(?P<event>\w+)'.*?failure_kind='dns_resolution'.*?"
    r"message_preview='Cannot connect to host (?P<host>[^.']*)"
)
_RE_TICKER_ATTR = re.compile(r"ticker='([^']+)'")
_RE_OPERATION_ATTR = re.compile(r"operation='([^']+)'")
_RE_CONSULTANT_INVALID = re.compile(
    r"event='consultant_invalid_structure' ticker='([^']+)'"
)
_RE_AGENT_TRUNCATED = re.compile(
    r"event='agent_output_truncated' agent='(?P<agent>[^']+)' ticker='(?P<ticker>[^']+)'"
    r".*?api_utilization_ratio='?(?P<util>[\d.]+)"
)
_RE_CONSULTANT_PARTIAL = re.compile(
    r"event='consultant_review_partial' ticker='([^']+)'.*?"
    r"tool_failure_count=(\d+) tool_call_count=(\d+)"
)


def scan_run_log(log_path: Path) -> str:
    text = log_path.read_text(errors="replace")

    root_cert_count = len(_RE_ROOT_CERT.findall(text))

    dns_by_host_op: dict[tuple[str, str], int] = {}
    for line in text.splitlines():
        m = _RE_DNS_LINE.search(line)
        if not m:
            continue
        op_m = _RE_OPERATION_ATTR.search(line)
        op = op_m.group(1) if op_m else "unknown_operation"
        host_hint = m.group("host") or "?"
        key = (op, host_hint)
        dns_by_host_op[key] = dns_by_host_op.get(key, 0) + 1

    consultant_invalid = _RE_CONSULTANT_INVALID.findall(text)
    truncated = [
        (m.group("agent"), m.group("ticker"), m.group("util"))
        for m in _RE_AGENT_TRUNCATED.finditer(text)
    ]
    partial = _RE_CONSULTANT_PARTIAL.findall(text)

    out = ["## Run-log scan", ""]
    out.append(
        f"- `failed to load native root certificate` warnings: **{root_cert_count}**"
        " (sandbox TLS/keychain artifact if present with 0 correlated LLM/tool failures"
        " -- see CLAUDE.md > Known Issues > Claude Code sandbox networking)"
    )

    if dns_by_host_op:
        out.append("- DNS resolution failures, by operation (host prefix in parens):")
        for (op, host), count in sorted(dns_by_host_op.items(), key=lambda kv: -kv[1]):
            out.append(f"  - `{op}` (`{host}...`): {count}")
    else:
        out.append("- DNS resolution failures: none")

    if consultant_invalid:
        out.append(
            f"- `consultant_invalid_structure` (empty/unparseable consultant output): "
            f"{len(consultant_invalid)} — tickers: {', '.join(consultant_invalid)}"
        )
    if truncated:
        out.append("- `agent_output_truncated` (thinking-budget exhaustion):")
        for agent, ticker, util in truncated:
            out.append(f"  - {agent} / {ticker}: api_utilization_ratio={util}")
    if partial:
        out.append("- `consultant_review_partial` (some tool calls failed mid-review):")
        for ticker, failed, total in partial:
            out.append(f"  - {ticker}: {failed}/{total} tool calls failed")
    if not consultant_invalid and not truncated and not partial:
        out.append("- No consultant/auditor structural failures detected")

    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--archive-dir", default=os.path.expanduser("~/Developer/results_archive")
    )
    parser.add_argument("--lookback-weeks", type=float, default=6.0)
    parser.add_argument(
        "--min-runs",
        type=int,
        default=2,
        help="widen the window until at least this many runs are found",
    )
    parser.add_argument("--max-lookback-weeks", type=float, default=16.0)
    parser.add_argument(
        "--run-log",
        default=None,
        help="path to a run.log from eval_rerun_longitudinal.sh (glob-expanded if it contains '*')",
    )
    parser.add_argument(
        "--output", default=None, help="write markdown here instead of stdout"
    )
    args = parser.parse_args(argv)

    results_dir = args.results_dir or _default_results_dir()
    archive_dir = args.archive_dir if os.path.isdir(args.archive_dir) else None

    sections = [f"# Longitudinal comparison — {datetime.now():%Y-%m-%d %H:%M}", ""]

    if args.run_log:
        matches = glob.glob(args.run_log)
        for m in sorted(matches) or [args.run_log]:
            p = Path(m)
            if p.is_file():
                sections.append(scan_run_log(p))

    sections.append("## Per-ticker timelines")
    sections.append("")
    for t in args.tickers:
        rows = discover_rows(
            t,
            results_dir,
            archive_dir,
            args.lookback_weeks,
            args.min_runs,
            args.max_lookback_weeks,
        )
        sections.append(render_timeline_markdown(t, rows))

    report = "\n".join(sections)
    if args.output:
        Path(args.output).write_text(report)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
