#!/usr/bin/env python3
"""Read-only audit of what evidence the retrospective would actually see.

Exists because ad-hoc shell sampling produced three wrong numbers in a row. The
failure was always the same: matching a lesson to a snapshot **by ticker**, when
a ticker routinely has several snapshots across the live and archive trees. The
audit compared prose against the wrong analysis and reported inventions that
were grounded, and grounded lessons that were inventions.

So identity here is `snapshot_identity()` plus the source file — never the
ticker — and this is a committed script rather than a one-off so the next
measurement is reproducible.

Costs nothing: no pricing, no LLM call, no write. Run it before spending money.

    poetry run python scripts/retrospective_evidence_audit.py
    poetry run python scripts/retrospective_evidence_audit.py --ticker 2530.TW
    poetry run python scripts/retrospective_evidence_audit.py --sample 5
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import config  # noqa: E402
from src.retrospective import (  # noqa: E402
    BEAR_EVIDENCE_MISSING,
    MINIMUM_DAYS_ELAPSED,
    _resolve_bear_evidence,
    _snapshot_days_elapsed,
    has_grounding_context,
    snapshot_identity,
)
from src.retrospective_sources import resolve_retrospective_sources  # noqa: E402


def _rows(
    dirs: tuple[Path, ...], ticker: str | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One row per *snapshot*, keyed by identity. Live tree wins over archive.

    Also returns a scan report. Without it the denominator is unexplainable: two
    runs of this audit reported 7,952 and 4,732 identities and both were right —
    one had RETROSPECTIVE_ARCHIVE_DIRS set and the other did not. A count with no
    stated scope invites exactly that confusion.
    """
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "scanned": [],
        "missing_dirs": [],
        "files": 0,
        "malformed": 0,
        "no_snapshot": 0,
        "duplicate_identity": 0,
    }
    for directory in dirs:
        if not directory.exists():
            report["missing_dirs"].append(str(directory))
            continue
        report["scanned"].append(str(directory))
        for path in sorted(directory.glob("*_analysis.json")):
            if ticker and not path.name.startswith(f"{ticker}_"):
                continue
            report["files"] += 1
            try:
                artifact = json.loads(path.read_text())
            except Exception:
                report["malformed"] += 1
                continue
            snapshot = artifact.get("prediction_snapshot")
            if not snapshot:
                report["no_snapshot"] += 1
                continue
            snapshot["_source_file"] = path.name
            identity = snapshot_identity(snapshot)
            if identity in seen:
                report["duplicate_identity"] += 1
                continue
            seen.add(identity)

            stored = snapshot.get("bear_risks_excerpt") or ""
            _resolve_bear_evidence(snapshot, artifact)
            days = _snapshot_days_elapsed(snapshot)
            rows.append(
                {
                    "identity": identity,
                    "source_file": path.name,
                    "ticker": snapshot.get("ticker", "?"),
                    "days_elapsed": days,
                    "stored_chars": len(stored),
                    "resolved_chars": len(snapshot.get("bear_risks_excerpt") or ""),
                    "provenance": snapshot.get("bear_evidence_provenance", "?"),
                    "grounded": has_grounding_context(snapshot),
                    # Old enough and grounded. Deliberately NOT "would reach the model":
                    # dedup, the memo, the per-run budget and the trigger threshold all sit
                    # downstream, and each removes more. This is an upper bound.
                    "eligible_pre_memo": bool(
                        days is not None
                        and days >= MINIMUM_DAYS_ELAPSED
                        and has_grounding_context(snapshot)
                    ),
                }
            )
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", help="restrict to one ticker")
    parser.add_argument(
        "--sample", type=int, default=0, help="print N reconstructed examples"
    )
    args = parser.parse_args()

    rows, report = _rows(resolve_retrospective_sources(config), args.ticker)
    if not rows:
        print("no snapshots found")
        return 1

    prov = collections.Counter(row["provenance"] for row in rows)
    total = len(rows)
    print("sources scanned (in precedence order):")
    for name in report["scanned"]:
        print(f"    {name}")
    for name in report["missing_dirs"]:
        print(f"    {name}  [CONFIGURED BUT MISSING — corpus is smaller than intended]")
    print(
        f"\nfiles read {report['files']}"
        f" | malformed {report['malformed']}"
        f" | no snapshot {report['no_snapshot']}"
        f" | duplicate identity {report['duplicate_identity']}"
    )
    print(f"\nsnapshots (by identity, live tree preferred): {total}\n")
    print("bear evidence provenance:")
    for name, count in prov.most_common():
        print(f"  {count:6d} ({100 * count / total:4.1f}%)  {name}")

    grounded = sum(1 for row in rows if row["grounded"])
    eligible = sum(1 for row in rows if row["eligible_pre_memo"])
    print(f"\n  {grounded:6d} ({100 * grounded / total:4.1f}%)  grounded")
    print(
        f"  {eligible:6d} ({100 * eligible / total:4.1f}%)  eligible for pricing "
        f"(upper bound: before dedup, memo, budget and trigger threshold)"
    )
    print(
        f"  {total - eligible:6d} ({100 * (total - eligible) / total:4.1f}%)  "
        f"excluded here (too recent, or no grounding)"
    )

    repaired = [
        r for r in rows if r["provenance"] not in (BEAR_EVIDENCE_MISSING, "snapshot")
    ]
    if repaired:
        gained = sum(r["resolved_chars"] - r["stored_chars"] for r in repaired)
        print(
            f"\nreconstruction recovered {gained:,} characters of bear evidence "
            f"across {len(repaired):,} snapshots "
            f"(median stored {sorted(r['stored_chars'] for r in repaired)[len(repaired) // 2]} "
            f"-> resolved {sorted(r['resolved_chars'] for r in repaired)[len(repaired) // 2]})"
        )

    for row in repaired[: args.sample]:
        print(f"\n  {row['source_file']}  ({row['identity']})")
        print(f"    stored   : {row['stored_chars']:4d} chars")
        print(f"    resolved : {row['resolved_chars']:4d} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
