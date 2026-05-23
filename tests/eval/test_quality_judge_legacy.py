"""Tests for the LEGACY bucket and markdown-dir resolution (Tranche 5, Step 10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.report_quality_judge import (
    _find_markdown_for_json,
    aggregate,
    main,
    score_saved_analysis,
)

_FULL_REPORT = """# 3306.HK: BUY

## Investment Memo — BUY

**Thesis.** JNBY trades at 9.6x P/E with 39% ROIC.

**Variant view.** Market misprices the recurring revenue mix.

**Key numbers.**
- P/E: 9.6

**Valuation.** Bear HKD 8.20 (30%) / Base HKD 11.50 (50%) / Bull HKD 15.00 (20%); weighted HKD 11.31.

**Top risks.**
- China consumer slowdown

**Kill criteria.**
- D/E > 1.0

**Confidence.** Anchored.

**Source confidence.**

| Claim | Source | Confidence |
| --- | --- | --- |
| Core financials | FILING | HIGH |

CONSULTANT_RESOLUTION:
- CONCERN: NONE
- VERDICT: N/A
"""


# ---------- _find_markdown_for_json ----------


def test_find_markdown_uses_sibling_md_first(tmp_path: Path) -> None:
    json_path = tmp_path / "3306.HK_20260518_204009_analysis.json"
    json_path.write_text("{}", encoding="utf-8")
    sibling = tmp_path / "3306.HK_20260518_204009.md"
    sibling.write_text("SIBLING CONTENT", encoding="utf-8")
    # markdown_dir present but sibling wins.
    out = _find_markdown_for_json(json_path, markdown_dir=tmp_path)
    assert out == "SIBLING CONTENT"


def test_find_markdown_falls_back_to_markdown_dir(tmp_path: Path) -> None:
    """Pipeline shape: results/X_analysis.json + scratch/README-X-DATE.md."""
    results = tmp_path / "results"
    scratch = tmp_path / "scratch"
    results.mkdir()
    scratch.mkdir()
    json_path = results / "3306.HK_20260518_204009_analysis.json"
    json_path.write_text("{}", encoding="utf-8")
    md = scratch / "README-3306-HK-2026-05-18.md"
    md.write_text("PIPELINE MD", encoding="utf-8")
    assert _find_markdown_for_json(json_path, markdown_dir=scratch) == "PIPELINE MD"


def test_find_markdown_returns_empty_when_no_match(tmp_path: Path) -> None:
    json_path = tmp_path / "TICKER_20260518_analysis.json"
    json_path.write_text("{}", encoding="utf-8")
    assert _find_markdown_for_json(json_path, markdown_dir=None) == ""
    assert _find_markdown_for_json(json_path, markdown_dir=tmp_path / "missing") == ""


def test_find_markdown_dasher_handles_dot_tickers(tmp_path: Path) -> None:
    """Tickers with dots (3306.HK, 7203.T) map to dashes in the pipeline filename."""
    scratch = tmp_path
    json_path = tmp_path / "7203.T_20260518_120000_analysis.json"
    json_path.write_text("{}", encoding="utf-8")
    md = scratch / "README-7203-T-2026-05-18.md"
    md.write_text("DASHED OK", encoding="utf-8")
    assert _find_markdown_for_json(json_path, markdown_dir=scratch) == "DASHED OK"


# ---------- score_saved_analysis LEGACY bucket ----------


def test_legacy_bucket_for_zero_features_no_markdown(tmp_path: Path) -> None:
    json_path = tmp_path / "OLD_20240101_analysis.json"
    json_path.write_text(
        json.dumps({"final_decision": {"decision": "BUY"}}),
        encoding="utf-8",
    )
    score = score_saved_analysis(json_path)
    assert score is not None
    assert score.overall == "LEGACY"
    assert score.features_present == 0
    assert any("Legacy artifact" in note for note in score.notes)


def test_post_feature_missing_features_is_fail_not_legacy(tmp_path: Path) -> None:
    """A new report rendered but missing features should be FAIL — the markdown
    proves the run is post-Tranche-1 capable."""
    json_path = tmp_path / "RECENT_20260520_analysis.json"
    json_path.write_text(json.dumps({}), encoding="utf-8")
    md = tmp_path / "RECENT_20260520.md"
    md.write_text(
        "# Just a title.\n\nNo memo, no scenarios, nothing.\n", encoding="utf-8"
    )
    score = score_saved_analysis(json_path)
    assert score is not None
    assert score.overall == "FAIL"
    assert score.features_present == 0


def test_paired_markdown_via_markdown_dir_scores_a(tmp_path: Path) -> None:
    results = tmp_path / "results"
    scratch = tmp_path / "scratch"
    results.mkdir()
    scratch.mkdir()
    json_path = results / "3306.HK_20260518_204009_analysis.json"
    json_path.write_text(json.dumps({}), encoding="utf-8")
    (scratch / "README-3306-HK-2026-05-18.md").write_text(
        _FULL_REPORT, encoding="utf-8"
    )
    score = score_saved_analysis(json_path, markdown_dir=scratch)
    assert score is not None
    assert score.overall == "A"


# ---------- aggregate ----------


def test_aggregate_counts_legacy_bucket(tmp_path: Path) -> None:
    # One LEGACY artifact, one A-grade pair, one FAIL (rendered but featureless).
    (tmp_path / "OLD_20240101_analysis.json").write_text(
        json.dumps({"final_decision": {"decision": "BUY"}}),
        encoding="utf-8",
    )
    (tmp_path / "PAIRED_20260518_analysis.json").write_text(
        json.dumps({}), encoding="utf-8"
    )
    (tmp_path / "PAIRED_20260518.md").write_text(_FULL_REPORT, encoding="utf-8")
    (tmp_path / "FAIL_20260518_analysis.json").write_text(
        json.dumps({}), encoding="utf-8"
    )
    (tmp_path / "FAIL_20260518.md").write_text("# only a title", encoding="utf-8")

    summary = aggregate(sorted(tmp_path.glob("*_analysis.json")))
    assert summary["count"] == 3
    assert summary["grades"]["LEGACY"] == 1
    assert summary["grades"]["A"] == 1
    assert summary["grades"]["FAIL"] == 1


def test_aggregate_forwards_markdown_dir(tmp_path: Path) -> None:
    results = tmp_path / "results"
    scratch = tmp_path / "scratch"
    results.mkdir()
    scratch.mkdir()
    (results / "X.HK_20260518_001_analysis.json").write_text(
        json.dumps({}), encoding="utf-8"
    )
    (scratch / "README-X-HK-2026-05-18.md").write_text(_FULL_REPORT, encoding="utf-8")
    summary = aggregate(sorted(results.glob("*_analysis.json")), markdown_dir=scratch)
    assert summary["count"] == 1
    assert summary["grades"]["A"] == 1
    assert summary["grades"]["LEGACY"] == 0


# ---------- CLI flag ----------


def test_cli_markdown_dir_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    scratch = tmp_path / "scratch"
    results.mkdir()
    scratch.mkdir()
    (results / "Z.HK_20260518_001_analysis.json").write_text(
        json.dumps({}), encoding="utf-8"
    )
    (scratch / "README-Z-HK-2026-05-18.md").write_text(_FULL_REPORT, encoding="utf-8")
    rc = main([str(results), "--markdown-dir", str(scratch)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "A=1" in captured.out
    # LEGACY surfaced in the grade summary.
    assert "LEGACY=" in captured.out


def test_cli_summary_includes_legacy_grade(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "OLD_20240101_analysis.json").write_text(
        json.dumps({"final_decision": {"decision": "BUY"}}),
        encoding="utf-8",
    )
    rc = main([str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "LEGACY=1" in captured.out
