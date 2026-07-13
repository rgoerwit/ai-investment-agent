"""Score-consistency SUSPECT token → HEALTH/GROWTH_SCORE_UNRELIABLE warning flags."""

from __future__ import annotations

from src.validators.red_flag_detector import RedFlagDetector


def test_extract_metrics_reads_score_consistency_tokens() -> None:
    report = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 40% (based on 10 available points)
HEALTH_SCORE_CONSISTENCY: SUSPECT — raw denominator 10 matches neither rubric total 12 nor available points 8
GROWTH_SCORE_CONSISTENCY: SUSPECT — earned points 7 exceed available 6
### --- END DATA_BLOCK ---
"""
    metrics = RedFlagDetector.extract_metrics(report)

    assert metrics["health_score_consistency"] == "SUSPECT"
    assert metrics["growth_score_consistency"] == "SUSPECT"


def test_extract_metrics_ignores_absent_or_garbled_consistency_tokens() -> None:
    report = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 83% (based on 9 available points)
GROWTH_SCORE_CONSISTENCY: SUSPECTED-MAYBE
### --- END DATA_BLOCK ---
"""
    metrics = RedFlagDetector.extract_metrics(report)

    assert metrics["health_score_consistency"] is None
    assert metrics["growth_score_consistency"] is None


def test_suspect_health_score_emits_zero_penalty_review_flag() -> None:
    flags, result = RedFlagDetector.detect_red_flags(
        {"health_score_consistency": "SUSPECT"},
        "AGS.BR",
    )

    unreliable = [f for f in flags if f["type"] == "HEALTH_SCORE_UNRELIABLE"]
    assert result == "PASS"  # data-quality flag never auto-rejects
    assert len(unreliable) == 1
    assert unreliable[0]["severity"] == "WARNING"
    assert unreliable[0]["action"] == "REVIEW"
    assert unreliable[0]["risk_penalty"] == 0.0


def test_both_kinds_suspect_emit_two_flags() -> None:
    flags, _ = RedFlagDetector.detect_red_flags(
        {
            "health_score_consistency": "SUSPECT",
            "growth_score_consistency": "SUSPECT",
        },
        "AGS.BR",
    )

    types = {f["type"] for f in flags}
    assert {"HEALTH_SCORE_UNRELIABLE", "GROWTH_SCORE_UNRELIABLE"} <= types


def test_consistent_scores_emit_no_unreliable_flag() -> None:
    flags, _ = RedFlagDetector.detect_red_flags(
        {"health_score_consistency": None, "growth_score_consistency": None},
        "TEST",
    )

    assert not [f for f in flags if f["type"].endswith("_SCORE_UNRELIABLE")]
