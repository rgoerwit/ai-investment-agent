#!/usr/bin/env python3
"""Batch health digest — scan a run-date's analyses for anomalies.

Read-only. Consumes only already-persisted fields (``run_summary``,
``analysis_validity``, ``prediction_snapshot``) from ``results/*_analysis.json`` and
prints a compact digest of anomalies worth a human glance:

- not publishable / required or fatal validity failures
- ``llm_failures > 0``
- consultant ``ERROR`` / ``UNPARSED`` (a ran-but-broken cross-check)
- optional-artifact failures (consultant/auditor provider errors)
- **verdict flip vs the prior run of the same ticker** (the highest-value signal)

By-design low-signal states are deliberately NOT flagged: auditor ``PARTIAL_DATA`` /
``INSUFFICIENT_DATA`` (honest completeness caveats) and consultant ``SKIPPED`` (the
quick-mode gate bypass).

Invoked automatically at each stage end by ``scripts/run_pipeline.sh`` so it is a
digest, not a manual step. Always exits 0 (informational) unless ``--strict`` is given.

Usage:
    poetry run python scripts/scan_batch_health.py [--run-date YYYY-MM-DD]
        [--results-dir DIR] [--json] [--strict]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_FILENAME_RE = re.compile(
    r"^(?P<ticker>.+)_(?P<date>\d{8})_(?P<time>\d{6})_analysis\.json$"
)

# Consultant verdicts that mean "ran but the cross-check is broken/unusable".
# SKIPPED (quick-mode gate bypass) and CONDITIONAL/CLEAN/MAJOR_CONCERNS/etc. are fine.
_CONSULTANT_BAD_VERDICTS = {"ERROR", "UNPARSED"}


def _normalize_verdict(verdict: str | None) -> str:
    """Collapse spacing/underscore variants so DO NOT INITIATE == DO_NOT_INITIATE."""
    return (verdict or "").strip().upper().replace("_", " ")


@dataclass
class Record:
    ticker: str
    date: str  # YYYYMMDD
    time: str  # HHMMSS
    path: Path
    verdict: str
    is_quick: bool | None  # None = mode not recorded (pre-tracking artifact)
    run_summary: dict
    validity: dict

    @property
    def sort_key(self) -> str:
        return f"{self.date}{self.time}"


def load_records(results_dir: Path) -> list[Record]:
    """Load every parseable ``*_analysis.json`` into a Record (skips malformed)."""
    records: list[Record] = []
    for path in results_dir.glob("*_analysis.json"):
        m = _FILENAME_RE.match(path.name)
        if not m:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        snapshot = data.get("prediction_snapshot") or {}
        run_summary = data.get("run_summary") or {}
        raw_quick = snapshot.get("is_quick_mode")
        records.append(
            Record(
                ticker=m.group("ticker"),
                date=m.group("date"),
                time=m.group("time"),
                path=path,
                verdict=_normalize_verdict(snapshot.get("verdict")),
                is_quick=bool(raw_quick) if raw_quick is not None else None,
                run_summary=run_summary,
                validity=data.get("analysis_validity") or {},
            )
        )
    return records


def prior_verdict(record: Record, all_records: list[Record]) -> str | None:
    """The verdict of the most-recent *same-mode* analysis of this ticker before ``record``.

    Same-mode (full-vs-full, quick-vs-quick) is deliberate: a quick-mode Stage-1
    screen casts a wide BUY net that Stage-2 full analysis routinely refines to
    HOLD/DNI, so a quick→full "flip" is the expected pipeline behavior, not an
    anomaly. Comparing within a mode surfaces only a genuine change of view under
    equal rigor.
    """
    if record.is_quick is None:
        return None  # current mode unknown → cannot make a same-mode comparison
    earlier = [
        r
        for r in all_records
        if r.ticker == record.ticker
        and r.is_quick is not None
        and r.is_quick == record.is_quick
        and r.sort_key < record.sort_key
        and r.verdict
    ]
    if not earlier:
        return None
    return max(earlier, key=lambda r: r.sort_key).verdict


def detect_anomalies(record: Record, prior: str | None) -> list[str]:
    """Pure predicate: return human-readable anomaly strings for one record."""
    rs = record.run_summary
    anomalies: list[str] = []

    if rs.get("publishable") is False:
        anomalies.append("not publishable")

    required = record.validity.get("required_failures") or rs.get("required_failures")
    if required:
        anomalies.append(f"required failures: {', '.join(map(str, required))}")
    fatal = record.validity.get("fatal_failures")
    if fatal:
        anomalies.append(f"fatal failures: {', '.join(map(str, fatal))}")

    if (rs.get("llm_failures") or 0) > 0:
        anomalies.append(f"llm_failures={rs.get('llm_failures')}")

    if rs.get("consultant_verdict") in _CONSULTANT_BAD_VERDICTS:
        anomalies.append(f"consultant {rs.get('consultant_verdict')}")

    optional = rs.get("optional_failures") or []
    if optional:
        names = sorted({str(o) for o in optional}) if isinstance(optional, list) else []
        anomalies.append(
            f"optional failures: {', '.join(names)}" if names else "optional failures"
        )

    if prior and record.verdict and prior != record.verdict:
        anomalies.append(f"verdict flip: {prior} → {record.verdict}")

    return anomalies


@dataclass
class ScanResult:
    run_date: str
    flagged: list[tuple[Record, list[str]]] = field(default_factory=list)
    total: int = 0


def scan(results_dir: Path, run_date_compact: str) -> ScanResult:
    """Scan the batch for ``run_date_compact`` (YYYYMMDD); compare against all history."""
    all_records = load_records(results_dir)
    batch = [r for r in all_records if r.date == run_date_compact]
    result = ScanResult(run_date=run_date_compact, total=len(batch))
    for rec in sorted(batch, key=lambda r: r.sort_key):
        anomalies = detect_anomalies(rec, prior_verdict(rec, all_records))
        if anomalies:
            result.flagged.append((rec, anomalies))
    return result


def _render(result: ScanResult) -> str:
    lines = [
        f"━━━ Batch health digest — {result.run_date} "
        f"({result.total} analyses, {len(result.flagged)} flagged) ━━━"
    ]
    if not result.flagged:
        lines.append("✓ No anomalies.")
        return "\n".join(lines)
    for rec, anomalies in result.flagged:
        tag = " [quick]" if rec.is_quick else ""
        lines.append(
            f"  ⚠ {rec.ticker}{tag} ({rec.verdict or '?'}): {'; '.join(anomalies)}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from datetime import datetime

    from src.config import config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Run date to scan (YYYY-MM-DD; default: today)",
    )
    parser.add_argument(
        "--results-dir",
        default=str(config.results_dir),
        help="Results directory (default: RESULTS_DIR / config.results_dir)",
    )
    parser.add_argument("--json", action="store_true", help="Structured JSON output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when anomalies are found (default: always exit 0)",
    )
    args = parser.parse_args(argv)

    run_date_compact = args.run_date.replace("-", "")
    result = scan(Path(args.results_dir), run_date_compact)

    if args.json:
        print(
            json.dumps(
                {
                    "run_date": args.run_date,
                    "total": result.total,
                    "flagged": [
                        {
                            "ticker": rec.ticker,
                            "verdict": rec.verdict,
                            "is_quick": rec.is_quick,
                            "anomalies": anomalies,
                        }
                        for rec, anomalies in result.flagged
                    ],
                },
                indent=2,
            )
        )
    else:
        print(_render(result))

    return 1 if (args.strict and result.flagged) else 0


if __name__ == "__main__":
    sys.exit(main())
