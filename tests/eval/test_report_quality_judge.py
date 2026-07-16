"""Tests for the report quality judge (Tranche 4, Step 8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.report_quality_judge import (
    BLOAT_CHAR_THRESHOLD,
    ReportQualityScore,
    aggregate,
    main,
    score_report,
    score_saved_analysis,
)

# A "perfect" report — every feature present in the rendered markdown.
_FULL_REPORT = """# 3306.HK: BUY

## Investment Memo — BUY

**Thesis.** JNBY trades at 9.6x P/E with 39% ROIC.

**Variant view.** Market prices for melting ice cube; we see resilient fan economy.

**Key numbers.**
- P/E: 9.6
- ROIC: 39%

**Valuation.** Bear HKD 8.20 (30%) / Base HKD 11.50 (50%) / Bull HKD 15.00 (20%); weighted HKD 11.31.

**Top risks.**
- China consumer slowdown

**Kill criteria.**
- D/E exceeds 1.0
- Two consecutive quarters revenue decline

**Confidence.** Anchored on consultant cross-check.

**Source confidence.**

| Claim | Source | Confidence |
| --- | --- | --- |
| Core financials | FILING | HIGH |

---

## Full Analysis

CONSULTANT_RESOLUTION:
- CONCERN: NONE
- DATA_CHECK: N/A
- VERDICT: N/A

APAC_RESOLUTION:
- FINDING: SUPPORT
- DATA_CHECK: N/A
- VERDICT: N/A

AUDITOR_RESOLUTION: NONE
"""


_BARE_REPORT = (
    "# Some old report without any of the new sections.\n\nLegacy body text only."
)

_SAVED_FUNDAMENTALS = (
    "### --- START DATA_BLOCK ---\n"
    "PE_RATIO_TTM: 10\n"
    "CURRENT_PRICE: 100\n"
    "### --- END DATA_BLOCK ---\n"
)

_VALID_SCENARIOS = (
    "### --- START VALUATION_SCENARIOS ---\n"
    "METHODOLOGY: P/E\n"
    "DATA_SUFFICIENCY: HIGH\n"
    "BEAR_MULTIPLE: 8\n"
    "BEAR_GROWTH_PCT: -5\n"
    "BEAR_MARGIN_DELTA_BPS: -100\n"
    "BEAR_DRIVERS: Downcycle.\n"
    "BEAR_PROBABILITY: 30\n"
    "BASE_MULTIPLE: 12\n"
    "BASE_GROWTH_PCT: 5\n"
    "BASE_MARGIN_DELTA_BPS: 0\n"
    "BASE_DRIVERS: Mid-cycle.\n"
    "BASE_PROBABILITY: 50\n"
    "BULL_MULTIPLE: 16\n"
    "BULL_GROWTH_PCT: 10\n"
    "BULL_MARGIN_DELTA_BPS: 100\n"
    "BULL_DRIVERS: Re-rating.\n"
    "BULL_PROBABILITY: 20\n"
    "### --- END VALUATION_SCENARIOS ---\n"
)


# ---------- score_report happy / edge / error ----------


def test_score_report_perfect_grades_a() -> None:
    score = score_report(_FULL_REPORT)
    assert isinstance(score, ReportQualityScore)
    assert score.has_memo is True
    assert score.has_variant_view is True
    assert score.has_kill_criteria is True
    assert score.has_scenario_valuation is True
    assert score.has_specialist_resolution is True
    assert score.has_source_confidence is True
    assert score.features_present == 6
    assert score.overall == "A"
    assert score.bloat_flag is False


def test_score_report_bare_grades_fail() -> None:
    score = score_report(_BARE_REPORT)
    assert score.features_present == 0
    assert score.overall == "FAIL"
    assert score.bloat_flag is False


def test_score_report_empty_markdown_does_not_raise() -> None:
    score = score_report("")
    assert score.features_present == 0
    assert score.overall == "FAIL"


def test_score_report_non_string_markdown_handled() -> None:
    """Defensive: anything but a string is treated as empty markdown."""
    # mypy would catch this but operator tooling sometimes passes None.
    score = score_report(None)  # type: ignore[arg-type]
    assert score.features_present == 0
    assert score.overall == "FAIL"


def test_score_report_partial_features() -> None:
    md = "## Investment Memo — BUY\n\n**Variant view.** Market is wrong.\n\nCONSULTANT_RESOLUTION:\n- VERDICT: N/A\n"
    score = score_report(md)
    assert score.has_memo and score.has_variant_view and score.has_specialist_resolution
    assert not score.has_kill_criteria
    assert not score.has_scenario_valuation
    assert not score.has_source_confidence
    assert score.features_present == 3
    assert score.overall == "C"


def test_specialist_resolution_accepts_rendered_auditor_note() -> None:
    """The render layer rewrites the unresolved-auditor stub into a prose
    caveat (no raw AUDITOR_RESOLUTION token); the judge must still count it."""
    bolded = "> **Auditor note:** The forensic auditor flagged anomalies …\n"
    assert score_report(bolded).has_specialist_resolution is True
    # Unbolded variant also counts — decoupled from Markdown styling.
    unbolded = "> Auditor note: anomalies were not reconciled.\n"
    assert score_report(unbolded).has_specialist_resolution is True
    # Raw tokens still count (regression).
    assert score_report("APAC_RESOLUTION: NONE\n").has_specialist_resolution is True


def test_score_report_bloat_flag_fires_above_threshold() -> None:
    md = "## Investment Memo — BUY\n\n**Variant view.** v\n" + (
        "x" * (BLOAT_CHAR_THRESHOLD + 1)
    )
    score = score_report(md)
    assert score.bloat_flag is True
    assert any("Bloat" in note for note in score.notes)


# ---------- saved-JSON fallbacks ----------


def test_saved_json_fallback_picks_up_variant_view_from_investment_plan() -> None:
    md = "Bare report."
    saved = {
        "investment_plan": (
            "CONSENSUS_VIEW: Market sees decline.\n"
            "VARIANT_VIEW: We see recovery.\n"
            "BASIS: Order book +30% YoY.\n"
        )
    }
    score = score_report(md, saved)
    assert score.has_variant_view is True


def test_saved_json_fallback_picks_up_kill_criteria_from_bear_history() -> None:
    md = "Bare report."
    saved = {
        "investment_analysis": {
            "investment_debate": {
                "bear_history": (
                    "### --- START KILL_CRITERIA ---\n"
                    "TRIGGER_1: D/E > 1.0\n"
                    "### --- END KILL_CRITERIA ---\n"
                )
            }
        }
    }
    score = score_report(md, saved)
    assert score.has_kill_criteria is True


def test_saved_json_fallback_picks_up_parseable_scenarios() -> None:
    md = "Bare report."
    saved = {
        "reports": {
            "fundamentals_report": _SAVED_FUNDAMENTALS,
            "valuation_params": _VALID_SCENARIOS,
        }
    }
    score = score_report(md, saved)
    assert score.has_scenario_valuation is True


def test_saved_json_fallback_rejects_malformed_scenarios() -> None:
    md = "Bare report."
    saved = {
        "reports": {
            "fundamentals_report": _SAVED_FUNDAMENTALS,
            "valuation_params": (
                "### --- START VALUATION_SCENARIOS ---\n"
                "BEAR_MULTIPLE: 8\n"
                "### --- END VALUATION_SCENARIOS ---\n"
            ),
        }
    }
    score = score_report(md, saved)
    assert score.has_scenario_valuation is False


def test_saved_json_fallback_rejects_low_sufficiency_scenarios() -> None:
    md = "Bare report."
    saved = {
        "reports": {
            "fundamentals_report": _SAVED_FUNDAMENTALS,
            "valuation_params": _VALID_SCENARIOS.replace(
                "DATA_SUFFICIENCY: HIGH", "DATA_SUFFICIENCY: LOW"
            ),
        }
    }
    score = score_report(md, saved)
    assert score.has_scenario_valuation is False


# ---------- score_saved_analysis ----------


def test_score_saved_analysis_falls_back_when_no_sibling_md(tmp_path: Path) -> None:
    """Without a sibling .md file, the judge scores against JSON fallbacks only."""
    json_path = tmp_path / "TICKER_20260520_analysis.json"
    json_path.write_text(
        json.dumps(
            {
                "investment_plan": "CONSENSUS_VIEW: X. VARIANT_VIEW: Y. BASIS: Z.",
                "investment_analysis": {
                    "investment_debate": {
                        "bear_history": "### --- START KILL_CRITERIA ---\nTRIGGER_1: x\n### --- END KILL_CRITERIA ---",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    score = score_saved_analysis(json_path)
    assert score is not None
    assert score.has_variant_view is True
    assert score.has_kill_criteria is True
    # Markdown-only features absent → still scores via JSON only.
    assert score.has_memo is False
    assert score.has_source_confidence is False


def test_score_saved_analysis_returns_none_on_unreadable_file(tmp_path: Path) -> None:
    bad = tmp_path / "broken_analysis.json"
    bad.write_text("{not json", encoding="utf-8")
    assert score_saved_analysis(bad) is None


def test_score_saved_analysis_picks_up_sibling_markdown(tmp_path: Path) -> None:
    json_path = tmp_path / "TICKER_20260520_analysis.json"
    json_path.write_text(json.dumps({"x": 1}), encoding="utf-8")
    md_path = tmp_path / "TICKER_20260520.md"
    md_path.write_text(_FULL_REPORT, encoding="utf-8")
    score = score_saved_analysis(json_path)
    assert score is not None
    assert score.overall == "A"


# ---------- aggregate + CLI ----------


def test_aggregate_counts_grades_and_features(tmp_path: Path) -> None:
    # Two A-grade pairs (perfect markdown) and one FAIL-grade pair (bare).
    for stem, md in [
        ("ONE_20260520", _FULL_REPORT),
        ("TWO_20260520", _FULL_REPORT),
        ("THREE_20260520", _BARE_REPORT),
    ]:
        (tmp_path / f"{stem}_analysis.json").write_text(
            json.dumps({"x": 1}), encoding="utf-8"
        )
        (tmp_path / f"{stem}.md").write_text(md, encoding="utf-8")
    paths = sorted(tmp_path.glob("*_analysis.json"))
    summary = aggregate(paths)
    assert summary["count"] == 3
    assert summary["grades"]["A"] == 2
    assert summary["grades"]["FAIL"] == 1
    assert summary["feature_totals"]["has_memo"] == 2


def test_aggregate_empty_iterable_returns_zero_count() -> None:
    assert aggregate([])["count"] == 0


def test_cli_emits_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "ONE_20260520_analysis.json").write_text(
        json.dumps({"x": 1}), encoding="utf-8"
    )
    (tmp_path / "ONE_20260520.md").write_text(_FULL_REPORT, encoding="utf-8")
    rc = main([str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Scored 1 report(s)." in captured.out
    assert "A=1" in captured.out


def test_cli_returns_nonzero_when_no_files_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main([str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "No saved analyses found" in captured.out
