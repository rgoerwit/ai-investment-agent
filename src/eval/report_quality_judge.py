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


_VARIANT_PLACEHOLDER_FRAGMENT = "Not explicitly stated"


def _has_variant_view(markdown: str, saved: dict | None) -> bool:
    """Counts a variant view only when it carries real content.

    Pre-Step-8 the memo always rendered ``**Variant view.** Not explicitly
    stated.`` — meaning the marker alone was effectively a free quality point.
    The corrected gate requires either substantive memo content (marker
    present AND placeholder fragment absent in the memo head) or an actual
    ``CONSENSUS_VIEW:`` / ``VARIANT_VIEW:`` / ``NO VARIANT`` declaration in
    the saved investment plan.
    """
    head = markdown[:4000]
    if "**Variant view.**" in head and _VARIANT_PLACEHOLDER_FRAGMENT not in head:
        return True
    if saved:
        from src.reporting.state_access import get_investment_plan

        plan = get_investment_plan(saved)
        if plan and (
            "VARIANT_VIEW:" in plan
            or "CONSENSUS_VIEW:" in plan
            or "NO VARIANT" in plan.upper()
        ):
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


def _find_markdown_for_json(json_path: Path, markdown_dir: Path | None) -> str:
    """Locate a rendered markdown report for the given analysis JSON.

    Resolution order:

    1. Sibling-by-stem in the JSON's directory (``X_analysis.json`` →
       ``X.md``) — covers the case where someone re-renders into ``results/``.
    2. Operator-supplied ``markdown_dir`` (typically ``scratch/``) — matches
       the pipeline's actual rendered-report convention
       ``README-<TICKER_DASHED>-<DATE>.md``.

    Returns ``""`` when no markdown sibling can be located.
    """
    sibling = json_path.with_name(json_path.stem.replace("_analysis", "") + ".md")
    if sibling.exists():
        try:
            return sibling.read_text(encoding="utf-8")
        except Exception:  # pragma: no cover — best-effort
            return ""

    if markdown_dir is None or not markdown_dir.exists():
        return ""

    # Pipeline renders to scratch/README-<ticker-with-dots-as-dashes>-<DATE>.md.
    # Stem looks like "3306.HK_20260518_204009" → ticker is "3306.HK".
    stem_parts = json_path.stem.split("_")
    if not stem_parts:
        return ""
    ticker = stem_parts[0]
    ticker_dashed = ticker.replace(".", "-").replace("_", "-")
    for candidate in markdown_dir.glob(f"README-{ticker_dashed}-*.md"):
        try:
            return candidate.read_text(encoding="utf-8")
        except Exception:  # pragma: no cover — best-effort
            continue
    return ""


def score_saved_analysis(
    json_path: Path,
    markdown_dir: Path | None = None,
) -> ReportQualityScore | None:
    """Load a saved analysis JSON and score whatever markdown it surfaces.

    Behavior:

    - If a sibling ``.md`` exists, score that markdown against the JSON.
    - Else if ``markdown_dir`` is supplied (typically ``scratch/``), look for
      the pipeline's ``README-<TICKER>-<DATE>.md`` and score that.
    - Else score with empty markdown and use saved-JSON feature fallbacks.

    A score with zero features detected AND no rendered markdown found is
    bucketed as ``LEGACY`` rather than ``FAIL`` — those are pre-Tranche-1
    artifacts that predate the features, not regressed reports.
    """
    try:
        saved = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        from src.error_safety import summarize_exception

        logger.warning(
            "quality_judge_json_read_failed",
            path=str(json_path),
            **summarize_exception(exc, operation="quality_judge_load_json"),
        )
        return None

    markdown = _find_markdown_for_json(json_path, markdown_dir)
    score = score_report(markdown, saved)
    # Legacy-artifact heuristic: no markdown AND no features → not "FAIL,"
    # just predates the features. Keeps the corpus-wide aggregate honest.
    if not markdown and score.features_present == 0:
        score.overall = "LEGACY"
        score.notes.append(
            "Legacy artifact: no rendered markdown found and no feature fields in JSON"
        )
    return score


def aggregate(paths: Iterable[Path], markdown_dir: Path | None = None) -> dict:
    """Aggregate scores across a set of saved analysis JSONs.

    ``markdown_dir`` is forwarded to :func:`score_saved_analysis` so the
    pipeline's ``scratch/`` rendered reports can be paired with the JSONs in
    ``results/``.
    """
    grades = {"A": 0, "B": 0, "C": 0, "FAIL": 0, "LEGACY": 0}
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
        score = score_saved_analysis(path, markdown_dir=markdown_dir)
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
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing rendered markdown reports "
            "(pipeline default: scratch/README-<TICKER>-<DATE>.md). "
            "Without this flag, only sibling .md files next to the JSON are scored."
        ),
    )
    args = parser.parse_args(argv)

    targets = _iter_targets(args)
    summary = aggregate(targets, markdown_dir=args.markdown_dir)

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
        ", ".join(
            f"{g}={summary['grades'][g]}" for g in ("A", "B", "C", "FAIL", "LEGACY")
        ),
    )
    print("  Feature presence (count / pct):")
    for feature, total in summary["feature_totals"].items():
        pct = (total / count) * 100
        print(f"    {feature:<28}{total:>4} ({pct:5.1f}%)")
    print(f"  Bloated ({BLOAT_CHAR_THRESHOLD}+ chars): {summary['bloated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
