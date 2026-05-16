#!/usr/bin/env python3
"""Emit a compact network-health badge for one pipeline ticker log.

Usage: extract_run_health.py <log_file>

- Returns exit 0 always (this script must NEVER block the pipeline).
- Stdout is the badge string (or empty if nothing notable).
- Bash callers append the badge to their `[OK] ticker done [Verdict=...]`
  line only when stdout is non-empty.

The badge contains aggregated counts above per-key thresholds — see
`_pipeline_log_parsing.format_badge`. A single transient DNS failure
during a 5-minute run is *not* news; ten in one run is.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        # Silently no-op on misuse — the pipeline must never break because
        # the badge tool was called wrong.
        return 0

    log_path = Path(argv[1])
    try:
        # Local import keeps startup cheap and avoids importing src/ which
        # would pull in heavy LangChain modules.
        from _pipeline_log_parsing import format_badge, parse_log
    except ImportError:
        # When invoked from elsewhere, ensure the script's own directory
        # is on sys.path so it can find the shared parser sibling module.
        sys.path.insert(0, str(Path(__file__).parent.resolve()))
        from _pipeline_log_parsing import format_badge, parse_log

    stats = parse_log(log_path)
    badge = format_badge(stats)
    if badge:
        sys.stdout.write(badge)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
