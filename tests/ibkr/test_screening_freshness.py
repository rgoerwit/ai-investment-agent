from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from src.ibkr.screening_freshness import load_screening_freshness


def test_load_screening_freshness_missing_marker(tmp_path: Path):
    summary = load_screening_freshness(tmp_path)
    assert summary.status == "missing"
    assert summary.screening_date is None


def test_load_screening_freshness_invalid_json(tmp_path: Path):
    (tmp_path / ".pipeline_last_run.json").write_text("{bad", encoding="utf-8")
    summary = load_screening_freshness(tmp_path)
    assert summary.status == "missing"


def test_load_screening_freshness_uses_screening_date(tmp_path: Path):
    screening_date = (date.today() - timedelta(days=10)).isoformat()
    (tmp_path / ".pipeline_last_run.json").write_text(
        (
            "{"
            f'"screening_date":"{screening_date}",'
            '"completed_at":"2099-01-01T00:00:00Z",'
            '"candidate_count":312,'
            '"buy_count":15'
            "}"
        ),
        encoding="utf-8",
    )
    summary = load_screening_freshness(tmp_path, stale_after_days=90)
    assert summary.status == "fresh"
    assert summary.age_days == 10
    assert summary.candidate_count == 312
    assert summary.buy_count == 15


def test_load_screening_freshness_stale_marker(tmp_path: Path):
    screening_date = (date.today() - timedelta(days=91)).isoformat()
    (tmp_path / ".pipeline_last_run.json").write_text(
        (
            "{"
            f'"screening_date":"{screening_date}",'
            '"completed_at":"2026-01-01T00:00:00Z",'
            '"candidate_count":0,'
            '"buy_count":0'
            "}"
        ),
        encoding="utf-8",
    )
    summary = load_screening_freshness(tmp_path, stale_after_days=90)
    assert summary.status == "stale"
    assert summary.age_days == 91
