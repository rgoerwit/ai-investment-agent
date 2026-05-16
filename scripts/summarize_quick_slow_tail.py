#!/usr/bin/env python3
"""Summarize quick-mode slow-tail diagnostics from saved analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _iter_analysis_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*_analysis.json"))


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _ticker_from_artifact(data: dict[str, Any], path: Path) -> str:
    return str(
        data.get("ticker")
        or data.get("company_of_interest")
        or path.name.split("_", 1)[0]
        or "unknown"
    )


def collect_slow_tail_rows(
    results_dir: Path, *, min_timeout_seconds: float = 30.0
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _iter_analysis_files(results_dir):
        data = _load_json(path)
        if data is None:
            continue
        token_usage = data.get("token_usage") or {}
        diagnostics = token_usage.get("call_diagnostics") or {}
        timeout_loss = float(diagnostics.get("timeout_seconds_lost") or 0.0)
        slowest = diagnostics.get("slowest_call") or {}
        if timeout_loss < min_timeout_seconds and not diagnostics.get(
            "consultant_timeout"
        ):
            continue
        slowest_agents = diagnostics.get("slowest_agents") or []
        rows.append(
            {
                "ticker": _ticker_from_artifact(data, path),
                "timeout_seconds_lost": round(timeout_loss, 1),
                "slowest_agent": slowest.get("agent_name") or "unknown",
                "slowest_provider": slowest.get("provider") or "unknown",
                "slowest_origin": slowest.get("failure_origin") or "unknown",
                "slowest_elapsed_seconds": round(
                    float(slowest.get("elapsed_seconds") or 0.0), 1
                ),
                "consultant_timeout": bool(diagnostics.get("consultant_timeout")),
                "slowest_agents_top3": [
                    {
                        "agent_name": entry.get("agent_name") or "unknown",
                        "wall_clock_seconds": round(
                            float(entry.get("wall_clock_seconds") or 0.0), 1
                        ),
                        "calls": int(entry.get("calls") or 0),
                    }
                    for entry in slowest_agents[:3]
                ],
                "path": str(path),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["timeout_seconds_lost"]),
            str(row["ticker"]),
        )
    )
    return rows


def format_summary(rows: list[dict[str, Any]], *, limit: int) -> str:
    total_loss = sum(float(row["timeout_seconds_lost"]) for row in rows)
    lines = [
        f"quick_slow_tail_summary count={len(rows)} timeout_seconds_lost={total_loss:.1f}"
    ]
    for row in rows[:limit]:
        top3 = (
            ",".join(
                f"{entry['agent_name']}={entry['wall_clock_seconds']}s"
                for entry in row.get("slowest_agents_top3", [])
            )
            or "n/a"
        )
        lines.append(
            " ".join(
                [
                    f"ticker={row['ticker']}",
                    f"timeout_loss={row['timeout_seconds_lost']}s",
                    f"slowest_agent={row['slowest_agent']!r}",
                    f"provider={row['slowest_provider']}",
                    f"origin={row['slowest_origin']}",
                    f"elapsed={row['slowest_elapsed_seconds']}s",
                    f"consultant_timeout={row['consultant_timeout']}",
                    f"top3=[{top3}]",
                ]
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument("--min-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    rows = collect_slow_tail_rows(
        Path(args.results_dir), min_timeout_seconds=args.min_timeout_seconds
    )
    print(format_summary(rows, limit=max(0, args.limit)))
    return 1 if rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
