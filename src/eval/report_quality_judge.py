"""Deterministic quality scorer for rendered analysis reports.

Walks a rendered markdown report and a saved analysis JSON, counts which of
the six "hedge-fund-grade" features are present, and assigns a letter grade.
Cheap regression guard: if a tranche change accidentally drops the memo or
strips the scenario block, the aggregate score across `results/*.json` will
visibly degrade.

CLI:
    poetry run python -m src.eval.report_quality_judge results/
    poetry run python -m src.eval.report_quality_judge results/3306.HK_20260518_*.json

The judge is intentionally lexical, not semantic. It answers "does the
artifact contain the feature?" — not "is the feature any good?" Semantic
quality is the job of the consultant + APAC + auditor agents already in the
graph.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


# --- Feature thresholds ---------------------------------------------------

BLOAT_CHAR_THRESHOLD = 60_000  # Reports above this in chars get the bloat flag.


@dataclass
class ReportQualityScore:
    """Boolean-feature scorecard for a single report + saved JSON pair."""

    has_memo: bool = False
    has_variant_view: bool = False
    has_kill_criteria: bool = False
    has_scenario_valuation: bool = False
    has_specialist_resolution: bool = False
    has_source_confidence: bool = False
    bloat_chars: int = 0
    bloat_flag: bool = False
    features_present: int = 0
    overall: str = "FAIL"
    notes: list[str] = field(default_factory=list)


def _has_memo(markdown: str) -> bool:
    return "## Investment Memo" in markdown[:4000]


def _has_variant_view(markdown: str, saved: dict | None) -> bool:
    if "**Variant view.**" in markdown[:4000]:
        return True
    # Saved-JSON fallback: VARIANT_VIEW emitted by Research Manager.
    if saved:
        plan = (
            saved.get("investment_analysis", {}).get("investment_plan")
            or saved.get("investment_plan")
            or ""
        )
        if plan and ("VARIANT_VIEW:" in plan or "CONSENSUS_VIEW:" in plan):
            return True
    return False


def _has_kill_criteria(markdown: str, saved: dict | None) -> bool:
    if "Kill criteria" in markdown[:6000]:
        return True
    if saved:
        from src.agents.support import extract_kill_criteria, get_bear_history

        bear = get_bear_history(saved)
        if extract_kill_criteria(bear):
            return True
    return False


_SCENARIO_LINE = re.compile(
    r"Bear\s+[^\s].*?Base\s+[^\s].*?Bull\s+[^\s]",
    re.IGNORECASE | re.DOTALL,
)


def _has_scenario_valuation(markdown: str, saved: dict | None) -> bool:
    if _SCENARIO_LINE.search(markdown):
        return True
    if saved:
        params = (saved.get("reports") or {}).get("valuation_params", "")
        if params and "VALUATION_SCENARIOS" in params:
            return True
    return False


_RESOLUTION_TOKENS = ("CONSULTANT_RESOLUTION", "APAC_RESOLUTION", "AUDITOR_RESOLUTION")


def _has_specialist_resolution(markdown: str) -> bool:
    return any(token in markdown for token in _RESOLUTION_TOKENS)


def _has_source_confidence(markdown: str) -> bool:
    return "**Source confidence.**" in markdown[:6000]


def _grade(features_present: int) -> str:
    if features_present >= 5:
        return "A"
    if features_present >= 4:
        return "B"
    if features_present >= 2:
        return "C"
    return "FAIL"


def score_report(markdown: str, saved: dict | None = None) -> ReportQualityScore:
    """Score a rendered markdown report against the six quality features.

    `saved` (the parsed analysis JSON) is optional but recommended; it lets
    the judge detect features that didn't render in the markdown but are
    available in the artifact (e.g. variant view emitted by RM but not yet
    surfaced in the memo).
    """
    if not isinstance(markdown, str):
        markdown = ""

    score = ReportQualityScore(
        has_memo=_has_memo(markdown),
        has_variant_view=_has_variant_view(markdown, saved),
        has_kill_criteria=_has_kill_criteria(markdown, saved),
        has_scenario_valuation=_has_scenario_valuation(markdown, saved),
        has_specialist_resolution=_has_specialist_resolution(markdown),
        has_source_confidence=_has_source_confidence(markdown),
        bloat_chars=len(markdown),
    )
    score.bloat_flag = score.bloat_chars > BLOAT_CHAR_THRESHOLD
    score.features_present = sum(
        [
            score.has_memo,
            score.has_variant_view,
            score.has_kill_criteria,
            score.has_scenario_valuation,
            score.has_specialist_resolution,
            score.has_source_confidence,
        ]
    )
    score.overall = _grade(score.features_present)
    if score.bloat_flag:
        score.notes.append(
            f"Bloat: {score.bloat_chars} chars exceeds {BLOAT_CHAR_THRESHOLD} threshold"
        )
    return score


def score_saved_analysis(json_path: Path) -> ReportQualityScore | None:
    """Load a saved analysis JSON and score whatever markdown it surfaces.

    If the JSON does not embed a rendered markdown report (current shape),
    the judge falls back to scoring the appended report under the matching
    ``results/<TICKER>_<TIMESTAMP>.md`` if one exists; otherwise it scores
    against an empty string and uses saved-JSON fallbacks for each feature.
    """
    try:
        saved = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "quality_judge_json_read_failed", path=str(json_path), error=str(exc)
        )
        return None

    markdown = ""
    # Look for a sibling markdown report with the same stem.
    md_candidate = json_path.with_name(json_path.stem.replace("_analysis", "") + ".md")
    if md_candidate.exists():
        try:
            markdown = md_candidate.read_text(encoding="utf-8")
        except Exception:  # pragma: no cover — best-effort
            markdown = ""
    return score_report(markdown, saved)


def aggregate(paths: Iterable[Path]) -> dict:
    """Aggregate scores across a set of saved analysis JSONs."""
    grades = {"A": 0, "B": 0, "C": 0, "FAIL": 0}
    feature_totals = {
        "has_memo": 0,
        "has_variant_view": 0,
        "has_kill_criteria": 0,
        "has_scenario_valuation": 0,
        "has_specialist_resolution": 0,
        "has_source_confidence": 0,
    }
    bloated = 0
    scored = 0
    rows: list[dict] = []
    for path in paths:
        score = score_saved_analysis(path)
        if score is None:
            continue
        scored += 1
        grades[score.overall] = grades.get(score.overall, 0) + 1
        for k in feature_totals:
            if getattr(score, k):
                feature_totals[k] += 1
        if score.bloat_flag:
            bloated += 1
        rows.append({"path": str(path), **asdict(score)})
    return {
        "count": scored,
        "grades": grades,
        "feature_totals": feature_totals,
        "bloated": bloated,
        "rows": rows,
    }


def _iter_targets(args: argparse.Namespace) -> list[Path]:
    targets: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            targets.extend(sorted(path.glob("*_analysis.json")))
        elif path.is_file():
            targets.append(path)
        else:  # Glob handled by the shell already; only warn for outliers.
            logger.warning("quality_judge_path_missing", path=str(path))
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.report_quality_judge",
        description="Score rendered analysis reports against the six quality features.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Directories or *_analysis.json files to score.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit aggregate as JSON instead of the human-readable summary.",
    )
    args = parser.parse_args(argv)

    targets = _iter_targets(args)
    summary = aggregate(targets)

    if args.json:
        json.dump(summary, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    count = summary["count"]
    if count == 0:
        print("No saved analyses found.")
        return 1
    print(f"Scored {count} report(s).")
    print(
        "  Grades:",
        ", ".join(f"{g}={summary['grades'][g]}" for g in ("A", "B", "C", "FAIL")),
    )
    print("  Feature presence (count / pct):")
    for feature, total in summary["feature_totals"].items():
        pct = (total / count) * 100
        print(f"    {feature:<28}{total:>4} ({pct:5.1f}%)")
    print(f"  Bloated ({BLOAT_CHAR_THRESHOLD}+ chars): {summary['bloated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
