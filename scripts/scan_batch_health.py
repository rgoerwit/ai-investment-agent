#!/usr/bin/env python3
"""Batch health digest — scan a batch of analyses for anomalies.

Read-only. Consumes only already-persisted fields (``run_summary``,
``analysis_validity``, ``prediction_snapshot``) from ``results/*_analysis.json`` and
prints a compact digest of anomalies worth a human glance:

- not publishable / required or fatal validity failures
- ``llm_failures > 0``
- consultant ``ERROR`` / ``UNPARSED`` (a ran-but-broken cross-check)
- optional-artifact failures (consultant/auditor provider errors)
- **verdict flip vs the prior same-mode run of the same ticker** (highest-value signal)

By-design low-signal states are deliberately NOT flagged: auditor ``PARTIAL_DATA`` /
``INSUFFICIENT_DATA`` (honest completeness caveats) and consultant ``SKIPPED`` (the
quick-mode gate bypass).

Batch selection (mutually exclusive):

- ``--modified-since EPOCH`` — analyses whose file mtime is at/after a Unix epoch. This
  is what the pipeline uses at each stage end: it scopes the digest to the files *that
  stage actually wrote*, which is robust to a cross-day resume (the analysis filename
  carries the wall-clock date, which need not equal the pipeline's logical run-date) and
  to a stage that spans midnight.
- ``--run-date YYYY-MM-DD`` — analyses whose *filename* date matches (manual, retrospective).
- neither — defaults to ``--run-date`` = today.

The verdict-flip comparison always loads full history regardless of which selector picks
the batch. Always exits 0 (informational) unless ``--strict`` is given.

Usage:
    poetry run python scripts/scan_batch_health.py [--run-date YYYY-MM-DD |
        --modified-since EPOCH] [--results-dir DIR] [--json] [--strict]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
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


def _compact_to_display_date(compact: str | None) -> str:
    """YYYYMMDD → YYYY-MM-DD; pass through anything unexpected."""
    if compact and len(compact) == 8 and compact.isdigit():
        return f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"
    return compact or "unspecified"


@dataclass
class Record:
    ticker: str
    date: str  # YYYYMMDD (from filename)
    time: str  # HHMMSS (from filename)
    path: Path
    verdict: str
    is_quick: bool | None  # None = mode not recorded (pre-tracking artifact)
    run_summary: dict
    validity: dict
    mtime: float = 0.0  # file modification time (epoch); used by --modified-since

    @property
    def sort_key(self) -> str:
        return f"{self.date}{self.time}"


def load_records(results_dir: Path) -> list[Record]:
    """Load every parseable ``*_analysis.json`` into a Record (skips malformed/unreadable)."""
    records: list[Record] = []
    for path in results_dir.glob("*_analysis.json"):
        m = _FILENAME_RE.match(path.name)
        if not m:
            continue
        try:
            mtime = path.stat().st_mtime
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
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
                run_summary=run_summary if isinstance(run_summary, dict) else {},
                validity=(
                    data.get("analysis_validity")
                    if isinstance(data.get("analysis_validity"), dict)
                    else {}
                )
                or {},
                mtime=mtime,
            )
        )
    return records


def prior_verdict(record: Record, all_records: list[Record]) -> str | None:
    """The verdict of the most-recent *same-mode* analysis of this ticker before ``record``.

    Same-mode (full-vs-full, quick-vs-quick) is deliberate: a quick-mode Stage-1
    screen casts a wide BUY net that Stage-2 full analysis routinely refines to
    HOLD/DNI, so a quick→full "flip" is the expected pipeline behavior, not an
    anomaly. Comparing within a mode surfaces only a genuine change of view under
    equal rigor. "Before" is by filename timestamp (sort_key), so a re-run never
    compares against itself.
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
    if isinstance(optional, list) and optional:
        names = sorted({str(o) for o in optional})
        anomalies.append(f"optional failures: {', '.join(names)}")

    if prior and record.verdict and prior != record.verdict:
        anomalies.append(f"verdict flip: {prior} → {record.verdict}")

    return anomalies


@dataclass
class ScanResult:
    label: str
    flagged: list[tuple[Record, list[str]]] = field(default_factory=list)
    total: int = 0
    run_date: str | None = None  # YYYY-MM-DD (date mode)
    modified_since: float | None = None  # epoch (mtime mode)


@dataclass(frozen=True)
class FreshOutputCheck:
    """Validity result for one ticker expected to emit one fresh artifact."""

    status: str
    path: Path | None = None
    detail: str = ""

    @property
    def publishable(self) -> bool:
        return self.status == "PUBLISHABLE"


def check_fresh_ticker_output(
    results_dir: Path,
    ticker: str,
    modified_since: float,
) -> FreshOutputCheck:
    """Require exactly one fresh, publishable artifact for a completed invocation."""
    matches = [
        record
        for record in load_records(results_dir)
        if record.ticker == ticker and record.mtime >= modified_since
    ]
    if not matches:
        return FreshOutputCheck("MISSING", detail="no fresh analysis artifact")
    if len(matches) > 1:
        paths = ", ".join(str(record.path) for record in matches)
        return FreshOutputCheck(
            "AMBIGUOUS",
            detail=f"multiple fresh artifacts: {paths}",
        )

    record = matches[0]
    anomalies = detect_anomalies(record, prior=None)
    validity_failures = [
        anomaly
        for anomaly in anomalies
        if anomaly == "not publishable"
        or anomaly.startswith("required failures:")
        or anomaly.startswith("fatal failures:")
    ]
    if validity_failures:
        return FreshOutputCheck(
            "INCOMPLETE",
            path=record.path,
            detail="; ".join(validity_failures),
        )
    return FreshOutputCheck("PUBLISHABLE", path=record.path)


def scan(
    results_dir: Path,
    run_date_compact: str | None = None,
    *,
    modified_since: float | None = None,
) -> ScanResult:
    """Scan a batch and flag anomalies; the verdict-flip check spans all history.

    Selection precedence: ``modified_since`` (file mtime) wins when set; otherwise the
    filename-date match. Exactly one is normally supplied.
    """
    all_records = load_records(results_dir)
    if modified_since is not None:
        batch = [r for r in all_records if r.mtime >= modified_since]
        label = f"since {datetime.fromtimestamp(modified_since):%Y-%m-%d %H:%M}"
        run_date_display = None
    else:
        batch = [r for r in all_records if r.date == run_date_compact]
        run_date_display = _compact_to_display_date(run_date_compact)
        label = run_date_display

    result = ScanResult(
        label=label,
        total=len(batch),
        run_date=run_date_display,
        modified_since=modified_since,
    )
    for rec in sorted(batch, key=lambda r: r.sort_key):
        anomalies = detect_anomalies(rec, prior_verdict(rec, all_records))
        if anomalies:
            result.flagged.append((rec, anomalies))
    return result


def _render(result: ScanResult) -> str:
    lines = [
        f"━━━ Batch health digest — {result.label} "
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
    from src.config import config

    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--run-date",
        default=None,
        help="Scan analyses whose filename date matches (YYYY-MM-DD; default: today)",
    )
    selector.add_argument(
        "--modified-since",
        type=float,
        default=None,
        metavar="EPOCH",
        help=(
            "Scan analyses whose file mtime is at/after this Unix epoch "
            "(pipeline uses this to scope a stage's own output; robust to cross-day resume)"
        ),
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
    parser.add_argument(
        "--require-publishable-ticker",
        default=None,
        metavar="TICKER",
        help=(
            "Require exactly one publishable artifact for TICKER in the "
            "--modified-since window; intended for batch runners"
        ),
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if args.require_publishable_ticker:
        if args.modified_since is None:
            parser.error("--require-publishable-ticker requires --modified-since")
        check = check_fresh_ticker_output(
            results_dir,
            args.require_publishable_ticker,
            args.modified_since,
        )
        print(
            json.dumps(
                {
                    "ticker": args.require_publishable_ticker,
                    "status": check.status,
                    "publishable": check.publishable,
                    "path": str(check.path) if check.path else None,
                    "detail": check.detail,
                }
            )
        )
        return 0 if check.publishable else 1
    if args.modified_since is not None:
        result = scan(results_dir, modified_since=args.modified_since)
    else:
        run_date = args.run_date or datetime.now().strftime("%Y-%m-%d")
        result = scan(results_dir, run_date.replace("-", ""))

    if args.json:
        print(
            json.dumps(
                {
                    "label": result.label,
                    "run_date": result.run_date,
                    "modified_since": result.modified_since,
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
