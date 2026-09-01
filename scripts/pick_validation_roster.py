#!/usr/bin/env python3
"""Pick a validation-batch roster from the operator's local analysis history.

Read-only. Reads `results/*_analysis.json` filenames only — never their contents —
so it stays fast on an iCloud-synced tree where opening cold files blocks.

The slot design and the reasoning behind it live in the `validation-batch` skill.
In short:

  variance probe   one ticker run three times, to expose run-to-run instability
  carry-over       recently analyzed, the only slot that detects regression
  long-stale       randomized draw from names untouched for months, so slow
                   drift is visible instead of just the names already watched
  full runs        one overlapping a quick above, one exercising full-only paths

`results/` is operator-local and gitignored. An empty or missing directory is not an
error: the script reports what it found and picks nothing, so a fresh clone degrades
to "choose manually" rather than failing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# results/<TICKER>_<YYYYMMDD>_<HHMMSS>_analysis.json
ARTIFACT_RE = re.compile(
    r"^(?P<ticker>.+)_(?P<date>\d{8})_(?P<time>\d{6})_analysis\.json$"
)


def load_history(results_dir: Path) -> dict[str, list[dt.datetime]]:
    """Map ticker -> sorted run timestamps, from filenames alone."""
    history: dict[str, list[dt.datetime]] = defaultdict(list)
    if not results_dir.is_dir():
        return {}
    for entry in os.scandir(results_dir):
        match = ARTIFACT_RE.match(entry.name)
        if match is None:
            continue
        try:
            stamp = dt.datetime.strptime(
                match.group("date") + match.group("time"), "%Y%m%d%H%M%S"
            )
        except ValueError:
            continue
        history[match.group("ticker")].append(stamp)
    return {ticker: sorted(stamps) for ticker, stamps in history.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "results"),
        help="directory of *_analysis.json artifacts (default: repo results/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for the long-stale draw; record it so a roster is reproducible",
    )
    parser.add_argument(
        "--recent-days",
        type=float,
        default=7.0,
        help="a carry-over must have been analyzed within this window",
    )
    parser.add_argument(
        "--stale-weeks",
        type=float,
        default=8.0,
        help="a long-stale candidate must be untouched for at least this long",
    )
    parser.add_argument("--stale-picks", type=int, default=2)
    parser.add_argument("--carry-overs", type=int, default=2)
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    history = load_history(results_dir)
    if not history:
        print(f"No analysis artifacts found in {results_dir}.")
        print("Pick the roster manually; see the validation-batch skill for the slots.")
        return 0

    now = dt.datetime.now()
    recent_cut = now - dt.timedelta(days=args.recent_days)
    stale_cut = now - dt.timedelta(weeks=args.stale_weeks)

    latest = {ticker: stamps[-1] for ticker, stamps in history.items()}
    counts = {ticker: len(stamps) for ticker, stamps in history.items()}

    recent = sorted(
        (t for t, when in latest.items() if when >= recent_cut),
        key=lambda t: latest[t],
        reverse=True,
    )
    stale = sorted(
        (t for t, when in latest.items() if when <= stale_cut), key=lambda t: latest[t]
    )

    rng = random.Random(args.seed)
    stale_picks = rng.sample(stale, min(args.stale_picks, len(stale))) if stale else []

    # The best variance probe is a name with enough history to have a known-quiet
    # background. Most-analyzed first; the operator overrides if it is not cheap.
    probe_candidates = sorted(recent or latest, key=lambda t: counts[t], reverse=True)

    def show(ticker: str) -> str:
        age = (now - latest[ticker]).days
        return f"{ticker:<12} last={latest[ticker]:%Y-%m-%d} ({age:>3}d ago)  runs={counts[ticker]}"

    print(
        f"History: {len(history)} tickers, {sum(counts.values())} runs, in {results_dir}"
    )
    print(
        f"Seed: {args.seed!r}  (record this — it makes the stale draw reproducible)\n"
    )

    print("VARIANCE PROBE — pick one, run it 3x quick in this batch")
    print("  prefer a cheap ticker whose verdict comes from a code-owned gate")
    for ticker in probe_candidates[:5]:
        print("   ", show(ticker))

    print(
        f"\nCARRY-OVER — analyzed within {args.recent_days:g}d, the regression detector"
    )
    if not recent:
        print("    (none — this batch cannot detect regression; run one first)")
    for ticker in recent[: args.carry_overs]:
        print("   ", show(ticker))

    print(
        f"\nLONG-STALE — random draw from {len(stale)} untouched ≥{args.stale_weeks:g}w"
    )
    if not stale_picks:
        print("    (none available)")
    for ticker in stale_picks:
        print("   ", show(ticker))

    print("\nFULL RUNS — pick two:")
    print("    1. one ticker that also has a quick run above (quick→full conversion)")
    print("    2. one exercising full-only paths (auditor, regional, round-2 debate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
