"""Tests for the source-confidence table (Step 5)."""

from __future__ import annotations

import json

import pytest

from src.reporting.source_confidence import (
    build_source_confidence_rows,
    render_source_confidence_markdown,
)

_DATA_BLOCK_FILING = (
    "### --- START DATA_BLOCK ---\n"
    "SECTOR: Consumer Discretionary\n"
    "OPERATING_CASH_FLOW_SOURCE: FILING\n"
    "PE_RATIO_TTM: 9.6\n"
    "### --- END DATA_BLOCK ---\n"
)

_DATA_BLOCK_AGGREGATOR = (
    "### --- START DATA_BLOCK ---\n"
    "SECTOR: Industrials\n"
    "OPERATING_CASH_FLOW_SOURCE: JUNIOR\n"
    "### --- END DATA_BLOCK ---\n"
)


def _ocf_source_discrepancy_flag() -> dict[str, object]:
    return {
        "type": "OCF_SOURCE_DISCREPANCY",
        "severity": "WARNING",
        "detail": "Filing OCF differs from aggregator OCF.",
        "action": "RISK_PENALTY",
        "risk_penalty": 0.5,
    }


def _claim(rows, claim_name):
    matches = [row for row in rows if row[0] == claim_name]
    assert matches, f"no row for claim {claim_name!r}"
    return matches[0]


def test_build_rows_filing_ocf_high_confidence() -> None:
    rows = build_source_confidence_rows({"fundamentals_report": _DATA_BLOCK_FILING})
    claim, source, conf = _claim(rows, "Core financials")
    assert conf == "HIGH"
    assert "Filing" in source


_DATA_BLOCK_FILING_DISCREPANCY = (
    "### --- START DATA_BLOCK ---\n"
    "SECTOR: Basic Materials\n"
    "OPERATING_CASH_FLOW_SOURCE: FILING\n"
    "OCF_FILING_REASON: DISCREPANCY\n"
    "### --- END DATA_BLOCK ---\n"
)


def test_build_rows_filing_ocf_discrepancy_downgraded() -> None:
    # FILING OCF that conflicts with the aggregator is not "ground truth | HIGH".
    rows = build_source_confidence_rows(
        {"fundamentals_report": _DATA_BLOCK_FILING_DISCREPANCY}
    )
    claim, source, conf = _claim(rows, "Core financials")
    assert conf == "MEDIUM"
    assert "conflict" in source.lower()
    assert "ground truth" not in source.lower()


def test_build_rows_ocf_discrepancy_is_never_downgraded_by_consultant_prose() -> None:
    """The "corroborated / period mismatch" row is gone with the suppression it
    depended on: consultant prose alone cannot upgrade a filing/API OCF conflict
    into a corroborated figure."""
    block = (
        "### --- START DATA_BLOCK ---\n"
        "OPERATING_CASH_FLOW: 151.97M PLN\n"
        "OPERATING_CASH_FLOW_SOURCE: FILING\n"
        "OCF_FILING_REASON: DISCREPANCY\n"
        "### --- END DATA_BLOCK ---\n"
    )
    rows = build_source_confidence_rows(
        {
            "fundamentals_report": block,
            "consultant_review": (
                "SPOT_CHECK operatingCashflow: DATA_BLOCK 151.97m PLN FY2025; "
                "FMP 178.06m PLN TTM/Q1 — PERIOD MISMATCH, not a data conflict."
            ),
            "auditor_report": "Operating cash flow: PLN 151.967m",
            "red_flags": [_ocf_source_discrepancy_flag()],
        }
    )
    _, source, conf = _claim(rows, "Core financials")
    assert conf == "MEDIUM"
    assert "Filing/API OCF conflict" in source
    assert "corroborated" not in source


def test_build_rows_major_concerns_do_not_resolve_ocf_period_mismatch() -> None:
    block = (
        "### --- START DATA_BLOCK ---\n"
        "OPERATING_CASH_FLOW: 151.97M PLN\n"
        "OPERATING_CASH_FLOW_SOURCE: FILING\n"
        "OCF_FILING_REASON: DISCREPANCY\n"
        "### --- END DATA_BLOCK ---\n"
    )
    rows = build_source_confidence_rows(
        {
            "fundamentals_report": block,
            "consultant_review": (
                "### CONSULTANT REVIEW: MAJOR_CONCERNS\n"
                "SPOT_CHECK operatingCashflow: DATA_BLOCK 151.97m PLN FY2025; "
                "FMP 178.06m PLN TTM/Q1 — PERIOD MISMATCH, not a data conflict."
            ),
            "auditor_report": "Operating cash flow: PLN 151.967m",
            "red_flags": [_ocf_source_discrepancy_flag()],
        }
    )
    _, source, conf = _claim(rows, "Core financials")
    assert conf == "MEDIUM"
    assert "conflict" in source.lower()
    assert "corroborated" not in source.lower()


def test_build_rows_filing_ocf_no_discrepancy_stays_high() -> None:
    # FILING with API_UNAVAILABLE (single-source, no conflict) keeps HIGH.
    block = _DATA_BLOCK_FILING.replace(
        "OPERATING_CASH_FLOW_SOURCE: FILING\n",
        "OPERATING_CASH_FLOW_SOURCE: FILING\nOCF_FILING_REASON: API_UNAVAILABLE\n",
    )
    rows = build_source_confidence_rows({"fundamentals_report": block})
    _, _, conf = _claim(rows, "Core financials")
    assert conf == "HIGH"


def test_build_rows_aggregator_ocf_medium_confidence() -> None:
    rows = build_source_confidence_rows({"fundamentals_report": _DATA_BLOCK_AGGREGATOR})
    _, source, conf = _claim(rows, "Core financials")
    assert conf == "MEDIUM"
    assert "Aggregator" in source


def test_build_rows_metric_provenance_uses_selected_ttm_cash_flow_key() -> None:
    fundamentals = (
        "### --- START DATA_BLOCK ---\n"
        "OPERATING_CASH_FLOW: 774.37M TWD\n"
        "FREE_CASH_FLOW: 463.42M TWD\n"
        "PE_RATIO_TTM: 5.18\n"
        "OPERATING_CASH_FLOW_SOURCE: JUNIOR\n"
        "### --- END DATA_BLOCK ---\n"
    )
    raw = {
        "operatingCashflow": 792_112_000,
        "_operatingCashflow_source": "extracted_from_statements",
        "operatingCashflow_TTM": 774_369_000,
        "_operatingCashflow_TTM_source": "calculated_from_quarterly",
        "freeCashflow": 361_403_616,
        "_freeCashflow_source": "calculated_from_statements",
        "freeCashflow_TTM": 463_420_000,
        "_freeCashflow_TTM_source": "calculated_from_quarterly",
        "trailingPE": 5.18,
        "_field_sources": {
            "operatingCashflow_TTM": "yfinance",
            "freeCashflow_TTM": "yfinance",
            "trailingPE": "eodhd",
        },
    }
    rows = build_source_confidence_rows(
        {
            "fundamentals_report": fundamentals,
            "raw_fundamentals_data": json.dumps(raw),
        }
    )
    _, ocf_source, _ = _claim(rows, "Metric provenance: OPERATING_CASH_FLOW")
    assert "selected operatingCashflow_TTM" in ocf_source
    assert "derivation calculated_from_quarterly" in ocf_source
    assert "provider yfinance" in ocf_source

    _, fcf_source, _ = _claim(rows, "Metric provenance: FREE_CASH_FLOW")
    assert "selected freeCashflow_TTM" in fcf_source


def test_build_rows_metric_provenance_reports_calculated_pe_fallback() -> None:
    fundamentals = (
        "### --- START DATA_BLOCK ---\n"
        "PE_RATIO_TTM: 5.18\n"
        "OPERATING_CASH_FLOW_SOURCE: JUNIOR\n"
        "### --- END DATA_BLOCK ---\n"
    )
    raw = {
        "marketCap": 85_058_281_472,
        "netIncomeToCommon": 16_418_440_192,
        "_field_sources": {
            "marketCap": "yfinance",
            "netIncomeToCommon": "yfinance",
        },
    }
    rows = build_source_confidence_rows(
        {
            "fundamentals_report": fundamentals,
            "source_artifacts": {"raw_fundamentals_data": json.dumps(raw)},
        }
    )
    _, pe_source, _ = _claim(rows, "Metric provenance: PE_RATIO_TTM")
    assert "selected marketCap/netIncomeToCommon" in pe_source
    assert "calculated_from_market_cap_net_income" in pe_source
    assert "provider yfinance" in pe_source


def test_build_rows_surfaces_quarterly_diagnostics() -> None:
    raw = {
        "_quarterly_diagnostics": [
            {
                "field": "revenueGrowth_TTM",
                "status": "unavailable",
                "reason": "insufficient_quarterly_history",
            },
            {
                "field": "freeCashflow_TTM",
                "status": "unavailable",
                "reason": "missing_cashflow_or_capex_row",
            },
        ]
    }
    rows = build_source_confidence_rows({"raw_fundamentals_data": json.dumps(raw)})
    _, source, conf = _claim(rows, "Quarterly/TTM diagnostics")
    assert conf == "MEDIUM"
    assert "revenueGrowth_TTM: insufficient_quarterly_history" in source
    assert "freeCashflow_TTM: missing_cashflow_or_capex_row" in source


def test_build_rows_no_fundamentals_low_confidence() -> None:
    rows = build_source_confidence_rows({})
    _, source, conf = _claim(rows, "Core financials")
    assert conf == "LOW"
    assert source == "Not available"


def test_build_rows_auditor_successful_high() -> None:
    state = {
        "fundamentals_report": _DATA_BLOCK_FILING,
        "run_summary": {"auditor_completed": True, "auditor_successful": True},
    }
    rows = build_source_confidence_rows(state)
    _, source, conf = _claim(rows, "Forensic check")
    assert conf == "HIGH"
    assert "Auditor" in source


def test_build_rows_auditor_ran_with_caveats_medium() -> None:
    state = {
        "run_summary": {"auditor_completed": True, "auditor_successful": False},
    }
    rows = build_source_confidence_rows(state)
    _, source, conf = _claim(rows, "Forensic check")
    assert conf == "MEDIUM"
    assert "caveats" in source


def test_build_rows_auditor_not_run() -> None:
    rows = build_source_confidence_rows({})
    _, source, conf = _claim(rows, "Forensic check")
    assert source == "Not run"
    assert conf == "—"


def test_build_rows_consultant_successful_high() -> None:
    state = {"run_summary": {"consultant_successful": True}}
    rows = build_source_confidence_rows(state)
    _, source, conf = _claim(rows, "Cross-model review")
    assert conf == "HIGH"


def test_build_rows_consultant_error_low_confidence() -> None:
    state = {
        "run_summary": {
            "consultant_completed": True,
            "consultant_successful": False,
            "consultant_verdict": "ERROR",
        }
    }
    rows = build_source_confidence_rows(state)
    _, source, conf = _claim(rows, "Cross-model review")
    assert conf == "LOW"
    assert "failed validation" in source


def test_build_rows_apac_caution_surfaced() -> None:
    state = {
        "apac_regional_report": (
            "### APAC REGIONAL AUDIT: 7203.T\n"
            "**VERDICT FOR CONSULTANT AND PM**: CAUTION — promoter pledges unresolved.\n"
        )
    }
    rows = build_source_confidence_rows(state)
    _, source, _ = _claim(rows, "Regional context")
    assert "APAC Specialist (CAUTION)" in source


@pytest.mark.parametrize(
    "apac,expected_fragment",
    [
        ("NO_MATERIAL_APAC_CONNECTION", "no material APAC link"),
        ("APAC_SPECIALIST_UNAVAILABLE", "unavailable"),
    ],
)
def test_build_rows_apac_sentinels_handled(apac: str, expected_fragment: str) -> None:
    rows = build_source_confidence_rows({"apac_regional_report": apac})
    _, source, _ = _claim(rows, "Regional context")
    assert expected_fragment in source


def test_build_rows_reads_saved_json_shape() -> None:
    saved = {
        "reports": {
            "fundamentals_report": _DATA_BLOCK_FILING,
            "auditor_report": "audit content",
            "consultant_review": "review content",
            "apac_regional_report": "**VERDICT FOR CONSULTANT AND PM**: SUPPORT — no concern.",
        },
        "run_summary": {
            "auditor_completed": True,
            "auditor_successful": True,
            "consultant_successful": True,
        },
    }
    rows = build_source_confidence_rows(saved)
    assert _claim(rows, "Core financials")[2] == "HIGH"
    assert _claim(rows, "Forensic check")[2] == "HIGH"
    assert _claim(rows, "Cross-model review")[2] == "HIGH"
    assert "SUPPORT" in _claim(rows, "Regional context")[1]


def test_render_markdown_emits_table() -> None:
    rows = [
        ("Core financials", "FILING (OCF source)", "HIGH"),
        ("Forensic check", "Not run", "—"),
    ]
    md = render_source_confidence_markdown(rows)
    assert md.startswith("| Claim | Source | Confidence |")
    assert "| --- | --- | --- |" in md
    assert "| Core financials | FILING (OCF source) | HIGH |" in md
    assert "| Forensic check | Not run | — |" in md


def test_render_markdown_empty_rows_returns_empty_string() -> None:
    assert render_source_confidence_markdown([]) == ""
