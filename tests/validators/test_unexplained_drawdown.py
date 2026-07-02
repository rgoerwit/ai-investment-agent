"""UNEXPLAINED_DRAWDOWN_NEWS_GAP: large drawdown + empty news read = +0.5 warning."""

from __future__ import annotations

from src.validators.red_flag_detector import RedFlagDetector
from src.validators.supplemental_extractors import (
    extract_drawdown_explanation,
    extract_material_events_status,
)

_TOKEN_NONE_FOUND = "MATERIAL_EVENTS_90D: NONE_FOUND"
# The exact 6831.HK 2026-07-01 phrasing (pre-v5.4 news prompt).
_LEGACY_NO_EVENTS = (
    "**Most Important Event**: No material operational events (e.g., earnings "
    "releases, major M&A, or regulatory filings) have been reported in the "
    "last 90 days."
)


class TestExtractors:
    def test_token_line_wins(self) -> None:
        assert extract_material_events_status("MATERIAL_EVENTS_90D: FOUND") == "FOUND"
        assert extract_material_events_status(_TOKEN_NONE_FOUND) == "NONE_FOUND"

    def test_legacy_prose_fallback_matches_6831hk_phrasing(self) -> None:
        assert extract_material_events_status(_LEGACY_NO_EVENTS) == "NONE_FOUND"

    def test_empty_or_unstated_returns_none(self) -> None:
        assert extract_material_events_status("") is None
        assert extract_material_events_status(None) is None
        assert extract_material_events_status("Earnings beat expectations.") is None

    def test_drawdown_explanation_parsing(self) -> None:
        assert (
            extract_drawdown_explanation(
                "DRAWDOWN_EXPLANATION: profit warning (Kabutan, 6/12)"
            )
            == "profit warning (Kabutan, 6/12)"
        )
        assert extract_drawdown_explanation("DRAWDOWN_EXPLANATION: NOT_FOUND") is None
        assert extract_drawdown_explanation("no such line") is None
        assert extract_drawdown_explanation(None) is None


class TestDetectUnexplainedDrawdownFlags:
    def test_unexplained_drawdown_plus_none_found_fires(self) -> None:
        flags = RedFlagDetector.detect_unexplained_drawdown_flags(
            "UNEXPLAINED_LARGE_DRAWDOWN", _TOKEN_NONE_FOUND, "6831.HK"
        )
        assert len(flags) == 1
        flag = flags[0]
        assert flag["type"] == "UNEXPLAINED_DRAWDOWN_NEWS_GAP"
        assert flag["severity"] == "WARNING"
        assert flag["action"] == "RISK_PENALTY"
        assert flag["risk_penalty"] == 0.5

    def test_legacy_phrasing_fires_flag(self) -> None:
        flags = RedFlagDetector.detect_unexplained_drawdown_flags(
            "LARGE_DRAWDOWN_MACRO_ONLY", _LEGACY_NO_EVENTS, "6831.HK"
        )
        assert len(flags) == 1

    def test_company_specific_classification_does_not_fire(self) -> None:
        flags = RedFlagDetector.detect_unexplained_drawdown_flags(
            "LARGE_DRAWDOWN_COMPANY_SPECIFIC", _TOKEN_NONE_FOUND, "TEST"
        )
        assert flags == []

    def test_explanation_present_does_not_fire(self) -> None:
        report = (
            f"{_TOKEN_NONE_FOUND}\n"
            "DRAWDOWN_EXPLANATION: sector-wide China catering sell-off (Caixin, 6/20)"
        )
        flags = RedFlagDetector.detect_unexplained_drawdown_flags(
            "UNEXPLAINED_LARGE_DRAWDOWN", report, "TEST"
        )
        assert flags == []

    def test_material_events_found_does_not_fire(self) -> None:
        flags = RedFlagDetector.detect_unexplained_drawdown_flags(
            "UNEXPLAINED_LARGE_DRAWDOWN", "MATERIAL_EVENTS_90D: FOUND", "TEST"
        )
        assert flags == []

    def test_no_classification_or_empty_news_does_not_fire(self) -> None:
        assert (
            RedFlagDetector.detect_unexplained_drawdown_flags(
                None, _TOKEN_NONE_FOUND, "TEST"
            )
            == []
        )
        # Absent news artifact must not read as "no events".
        assert (
            RedFlagDetector.detect_unexplained_drawdown_flags(
                "UNEXPLAINED_LARGE_DRAWDOWN", "", "TEST"
            )
            == []
        )
        assert (
            RedFlagDetector.detect_unexplained_drawdown_flags(
                "UNEXPLAINED_LARGE_DRAWDOWN", None, "TEST"
            )
            == []
        )
