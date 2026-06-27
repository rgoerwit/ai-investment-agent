"""Tests for the OCF_PERIOD_NORMALIZATION note flag and ocf_period extraction.

A FILING-sourced headline OCF on a sub-annual period (Q*/H*) must surface a
deterministic NOTE (risk_penalty 0.0) so the PM/RM do not compare a single
quarter against TTM net income / FCF / payout ratios. It must NOT add risk.
"""

from __future__ import annotations

from src.validators.financial_rules import detect_red_flags
from src.validators.metric_extractor import extract_metrics


def _flags_for(metrics: dict) -> list[dict]:
    red_flags, _ = detect_red_flags(metrics, ticker="TEST.V")
    return red_flags


def _get(flags: list[dict], flag_type: str) -> dict | None:
    return next((f for f in flags if f.get("type") == flag_type), None)


def test_filing_quarterly_emits_normalization_note():
    flag = _get(
        _flags_for({"ocf_source": "FILING", "ocf_period": "Q4_2025"}),
        "OCF_PERIOD_NORMALIZATION",
    )
    assert flag is not None
    assert flag["risk_penalty"] == 0.0
    assert flag["action"] == "NOTE"
    assert "Q4_2025" in flag["detail"]


def test_filing_half_year_emits_normalization_note():
    flag = _get(
        _flags_for({"ocf_source": "FILING", "ocf_period": "H1 2025"}),
        "OCF_PERIOD_NORMALIZATION",
    )
    assert flag is not None
    assert flag["risk_penalty"] == 0.0


def test_filing_ttm_no_note():
    assert (
        _get(
            _flags_for({"ocf_source": "FILING", "ocf_period": "TTM"}),
            "OCF_PERIOD_NORMALIZATION",
        )
        is None
    )


def test_filing_annual_no_note():
    assert (
        _get(
            _flags_for({"ocf_source": "FILING", "ocf_period": "FY2025"}),
            "OCF_PERIOD_NORMALIZATION",
        )
        is None
    )


def test_junior_source_with_quarter_no_note():
    # Only a FILING headline matters; an aggregator (JUNIOR) OCF is already TTM.
    assert (
        _get(
            _flags_for({"ocf_source": "JUNIOR", "ocf_period": "Q4_2025"}),
            "OCF_PERIOD_NORMALIZATION",
        )
        is None
    )


def test_missing_period_no_note_no_crash():
    assert (
        _get(_flags_for({"ocf_source": "FILING"}), "OCF_PERIOD_NORMALIZATION") is None
    )
    assert (
        _get(
            _flags_for({"ocf_source": "FILING", "ocf_period": None}),
            "OCF_PERIOD_NORMALIZATION",
        )
        is None
    )


def test_extract_metrics_parses_ocf_period():
    data_block = (
        "### --- START DATA_BLOCK ---\n"
        "OPERATING_CASH_FLOW: 10.6M\n"
        "OPERATING_CASH_FLOW_SOURCE: FILING\n"
        "OPERATING_CASH_FLOW_PERIOD: Q4_2025\n"
        "### --- END DATA_BLOCK ---"
    )
    metrics = extract_metrics(data_block)
    assert metrics["ocf_period"] == "Q4_2025"
    assert metrics["ocf_source"] == "FILING"


def test_extract_metrics_ocf_period_na_is_none():
    data_block = (
        "### --- START DATA_BLOCK ---\n"
        "OPERATING_CASH_FLOW_PERIOD: N/A\n"
        "### --- END DATA_BLOCK ---"
    )
    assert extract_metrics(data_block)["ocf_period"] is None
