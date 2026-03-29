from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.web.ibkr_dashboard.drilldown_service import (
    DrilldownLoadError,
    build_structured_sections,
    find_markdown_artifacts,
    load_analysis_json,
    render_markdown_file,
)
from tests.factories.ibkr import make_analysis


def test_load_analysis_json_reads_valid_file(tmp_path: Path):
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps({"prediction_snapshot": {"ticker": "7203.T"}}), encoding="utf-8"
    )
    payload = load_analysis_json(path)
    assert payload["prediction_snapshot"]["ticker"] == "7203.T"


def test_load_analysis_json_raises_for_malformed_json(tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(DrilldownLoadError):
        load_analysis_json(path)


def test_render_markdown_file_sanitizes_script(tmp_path: Path):
    path = tmp_path / "report.md"
    path.write_text("# Hello\n\n<script>alert(1)</script>", encoding="utf-8")
    html = render_markdown_file(path)
    assert "<script>" not in html
    assert "Hello" in html


def test_render_markdown_file_strips_unsafe_link_scheme(tmp_path: Path):
    path = tmp_path / "report.md"
    path.write_text("[bad](javascript:alert(1))", encoding="utf-8")
    html = render_markdown_file(path)
    assert "javascript:" not in html


def test_find_markdown_artifacts_returns_paths_when_present(tmp_path: Path):
    analysis = make_analysis(ticker="MEGP.L")
    json_path = tmp_path / "MEGP.L_20260328_000000_analysis.json"
    json_path.write_text("{}", encoding="utf-8")
    report_path = json_path.with_suffix(".md")
    report_path.write_text("# Report", encoding="utf-8")
    article_path = tmp_path / "MEGP.L_article.md"
    article_path.write_text("# Article", encoding="utf-8")
    analysis.file_path = str(json_path)
    artifacts = find_markdown_artifacts(analysis)
    assert artifacts["report_markdown_path"] == str(report_path)
    assert artifacts["article_markdown_path"] == str(article_path)


def test_build_structured_sections_handles_missing_payload():
    assert build_structured_sections(None) == {}
