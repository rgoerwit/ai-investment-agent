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


def test_build_analysis_record_loads_macro_regime_block():
    record = _build_analysis_record_from_data(
        Path("7203.T_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Consumer Cyclical",
                "currency": "JPY",
            },
            "macro_regime_block": {
                "present": True,
                "risk_appetite": "RISK_OFF",
                "shock_type": "ENERGY",
                "shock_phase": "ACUTE",
                "equity_transmission": "EARNINGS_PRESSURE",
                "dip_posture": "WAIT_FOR_CONFIRMATION",
                "confidence": "MEDIUM",
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.macro_regime["risk_appetite"] == "RISK_OFF"


def test_build_analysis_record_legacy_json_defaults_macro_regime_empty():
    record = _build_analysis_record_from_data(
        Path("7203.T_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Consumer Cyclical",
                "currency": "JPY",
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.macro_regime == {}


def test_build_analysis_record_repairs_legacy_currency():
    record = _build_analysis_record_from_data(
        Path("PINFRA.MX_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "PINFRA.MX",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Industrials",
                "currency": "USD",  # Legacy bug: saved as USD
                "fx_rate_to_usd": 1.0,  # Legacy bug: saved as 1.0
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.currency == "MXN"
    assert record.fx_rate_to_usd == 0.049
    assert record.currency_repaired is True
    assert record.currency_repair_reason == "legacy_snapshot_usd_default"


def test_build_analysis_record_repairs_apr_wa_legacy_currency():
    record = _build_analysis_record_from_data(
        Path("APR.WA_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "APR.WA",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Industrials",
                "currency": "USD",
                "fx_rate_to_usd": 1.0,
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.currency == "PLN"
    assert record.fx_rate_to_usd == 0.25
    assert record.currency_repaired is True


def test_build_analysis_record_repairs_apr_ol_legacy_currency():
    record = _build_analysis_record_from_data(
        Path("APR.OL_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "APR.OL",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Industrials",
                "currency": "USD",
                "fx_rate_to_usd": 1.0,
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.currency == "NOK"
    assert record.fx_rate_to_usd == 0.092


def test_build_analysis_record_repairs_deme_br_legacy_currency():
    record = _build_analysis_record_from_data(
        Path("DEME.BR_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "DEME.BR",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Industrials",
                "currency": "USD",
                "fx_rate_to_usd": 1.0,
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.currency == "EUR"
    assert record.fx_rate_to_usd == 1.09


def test_build_analysis_record_preserves_valid_usd():
    record = _build_analysis_record_from_data(
        Path("AAPL_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "AAPL",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Information Technology",
                "currency": "USD",
                "fx_rate_to_usd": 1.0,
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.currency == "USD"
    assert record.fx_rate_to_usd == 1.0
    assert record.currency_repaired is False


def test_build_analysis_record_does_not_repair_bare_apr():
    record = _build_analysis_record_from_data(
        Path("APR_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "APR",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Industrials",
                "currency": "USD",
                "fx_rate_to_usd": 1.0,
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.currency == "USD"
    assert record.currency_repaired is False
