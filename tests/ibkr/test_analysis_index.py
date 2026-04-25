"""Collected analysis-index tests extracted from reconciler cases."""

from pathlib import Path

from src.ibkr.analysis_index import _build_analysis_record_from_data
from tests.ibkr.reconciler_cases import (
    TestLoadLatestAnalyses,
    TestParseScoresFromFinalDecision,
)


def test_build_analysis_record_normalizes_legacy_healthcare_sector():
    record = _build_analysis_record_from_data(
        Path("7203.T_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Healthcare",
                "currency": "JPY",
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.sector == "Health Care"


def test_build_analysis_record_normalizes_consumer_cyclical_sector():
    record = _build_analysis_record_from_data(
        Path("2767.T_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "2767.T",
                "analysis_date": "2026-04-25",
                "verdict": "HOLD",
                "sector": "Consumer Cyclical",
                "currency": "JPY",
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.sector == "Consumer Discretionary"
