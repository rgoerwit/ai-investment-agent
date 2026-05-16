#!/usr/bin/env python3
"""End-of-batch warning summary for run_pipeline.sh.

Design intent (per operator request, May 2026): the pipeline must NOT get
chatty. This script emits a **single** [WARN] line, and only when the
batch had non-trivial trouble. A clean run prints nothing.

Trouble-worthy signals (any one is sufficient):
- failed_runs > 0
- pipeline_child_timeout in any log
- network_breaker_opened in any log
- llm_circuit_opened in any log
- pending_tasks_dump in any log
- ≥3 runs in a row each with dns_failures ≥ 5 (the May 2026 outage shape)

The summary names the dominant failure_kind and the longest consecutive-
failure stretch so a 5am operator scanning their terminal can tell at a
glance whether the trouble was "one bad ticker" vs "host network blackout".
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow direct script invocation.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline_log_parsing import LogStats, parse_log  # noqa: E402

_DNS_RUN_THRESHOLD = 5  # per-run DNS failures considered "elevated"
_CONSECUTIVE_DNS_RUNS = 3  # how many in a row to flag a cluster


def _consecutive_elevated_dns(stats_seq: list[LogStats]) -> int:
    """Return the longest streak of consecutive runs with dns_failures >=
    `_DNS_RUN_THRESHOLD`. Used to detect a sustained network outage that
    looked like 3+ individually-noisy runs."""
    best = current = 0
    for s in stats_seq:
        if s.dns_failures >= _DNS_RUN_THRESHOLD:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def build_summary(stats_seq: list[LogStats]) -> str | None:
    """Return a single-line summary string, or None when nothing is worth
    surfacing. Keep the line short — it ships to the operator's terminal."""
    if not stats_seq:
        return None

    total = len(stats_seq)
    failed = sum(
        1
        for s in stats_seq
        if (s.exit_status is not None and s.exit_status != 0)
        or s.pipeline_child_timeout
    )
    child_timeouts = sum(1 for s in stats_seq if s.pipeline_child_timeout)
    breaker_trips = sum(s.network_breaker_opened for s in stats_seq)
    circuit_opens = sum(s.llm_circuit_opened for s in stats_seq)
    pending_dumps = sum(s.pending_task_dumps for s in stats_seq)
    consecutive_dns = _consecutive_elevated_dns(stats_seq)

    notable = (
        failed > 0
        or child_timeouts > 0
        or breaker_trips > 0
        or circuit_opens > 0
        or pending_dumps > 0
        or consecutive_dns >= _CONSECUTIVE_DNS_RUNS
    )
    if not notable:
        return None

    # Dominant failure_kind by aggregate event count.
    kind_counts: Counter[str] = Counter()
    kind_counts["dns_resolution"] = sum(s.dns_failures for s in stats_seq)
    kind_counts["connect_error"] = sum(s.connect_errors for s in stats_seq)
    kind_counts["timeout"] = sum(s.timeouts for s in stats_seq)
    dominant_kind = None
    if kind_counts:
        dominant = kind_counts.most_common(1)[0]
        if dominant[1] > 0:
            dominant_kind = dominant[0]

    parts = [f"runs={total}", f"failed={failed}"]
    if child_timeouts:
        parts.append(f"child_timeouts={child_timeouts}")
    if breaker_trips:
        parts.append(f"breaker_trips={breaker_trips}")
    if circuit_opens:
        parts.append(f"circuit_opens={circuit_opens}")
    if pending_dumps:
        parts.append(f"pending_dumps={pending_dumps}")
    if consecutive_dns >= _CONSECUTIVE_DNS_RUNS:
        parts.append(f"consecutive_dns_runs={consecutive_dns}")
    if dominant_kind is not None:
        parts.append(f"dominant_failure_kind={dominant_kind}")

    return "pipeline_batch_summary " + " ".join(parts)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a one-line warning summary for a batch of pipeline logs."
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default="scratch",
        help="Directory containing the per-ticker logs (default: scratch).",
    )
    parser.add_argument(
        "--pattern",
        default="*-LOG-*.txt",
        help="Glob pattern for log files (default: *-LOG-*.txt).",
    )
    args = parser.parse_args(argv[1:])

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        # Silently do nothing — pipeline must never break on summary.
        return 0

    paths = sorted(log_dir.glob(args.pattern), key=lambda p: p.stat().st_mtime)
    stats_seq = [parse_log(p) for p in paths]

    summary = build_summary(stats_seq)
    if summary is not None:
        # Single warning line, emitted to stdout so bash callers can prefix
        # it with [WARN] using their existing helper. NO log_dir noise on
        # clean runs.
        sys.stdout.write(summary + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
