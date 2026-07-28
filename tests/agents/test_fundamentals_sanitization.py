"""Regression tests for fundamentals DATA_BLOCK sanitization."""

from __future__ import annotations

import json

import pytest

from src.agents import analyst_nodes
from src.agents.analyst_nodes import (
    _sanitize_fundamentals_output,
    _valuation_input_reliability,
)
from src.agents.fundamentals_reconciler import (
    parse_score_breakdown,
    reconcile_score_consistency,
)
from src.data_block_utils import replace_or_append_block_line
from tests.helpers.frozen_regressions import load_frozen_regression


def test_sanitize_fundamentals_output_forces_missing_horizons_to_na() -> None:
    content = """### --- START DATA_BLOCK ---
REVENUE_GROWTH_FY: 39.8%
REVENUE_GROWTH_TTM: 39.8%
REVENUE_GROWTH_MRQ: 39.8% (as of 2025-12-31)
EARNINGS_GROWTH_TTM: 98.8%
EARNINGS_GROWTH_MRQ: 100.5%
GROWTH_TRAJECTORY: STABLE
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "revenueGrowth": 0.398,
            "revenueGrowth_TTM": None,
            "revenueGrowth_MRQ": None,
            "earningsGrowth": 0.988,
            "earningsGrowth_TTM": None,
            "earningsGrowth_MRQ": None,
            "growth_trajectory": None,
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "2173.T")

    assert "REVENUE_GROWTH_FY: 39.8%" in sanitized
    assert "REVENUE_GROWTH_TTM: N/A" in sanitized
    assert "REVENUE_GROWTH_MRQ: N/A" in sanitized
    assert "EARNINGS_GROWTH_TTM: N/A" in sanitized
    assert "EARNINGS_GROWTH_MRQ: N/A" in sanitized
    assert "GROWTH_TRAJECTORY: N/A" in sanitized


def test_6782_fy_growth_is_preserved_without_relabeling_it_ttm() -> None:
    regression = load_frozen_regression("6782_TW_regression.json")

    sanitized = _sanitize_fundamentals_output(
        regression["fundamentals_output"],
        json.dumps(regression["raw_metrics"]),
        regression["ticker"],
    )

    assert "REVENUE_GROWTH_FY: 15.0%" in sanitized
    assert "REVENUE_GROWTH_FY_SOURCE: ANNUAL_STATEMENTS" in sanitized
    assert "EARNINGS_GROWTH_FY: 39.4%" in sanitized
    assert "EARNINGS_GROWTH_FY_SOURCE: NET_INCOME_STATEMENT_PROXY" in sanitized
    assert "REVENUE_GROWTH_TTM: N/A" in sanitized
    assert "EARNINGS_GROWTH_TTM: N/A" in sanitized


def test_6782_mrq_growth_stays_bound_to_its_statement_date() -> None:
    content = """### --- START DATA_BLOCK ---
REVENUE_GROWTH_MRQ: 16.9% (as of 2026-03-31)
EARNINGS_GROWTH_MRQ: 102.8%
LATEST_QUARTER_DATE: 2026-03-31
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "latest_quarter_date": "2025-12-31",
            "_latest_quarter_date_source": "yfinance_quarterly",
            "revenueGrowth_MRQ": 0.168693,
            "_revenueGrowth_MRQ_source": "calculated_from_quarterly",
            "earningsGrowth_MRQ": 1.028262,
            "_earningsGrowth_MRQ_source": "calculated_from_quarterly",
            "_data_quality_notes": [
                "Newer quarter metadata exists for 2026-03-31, but "
                "statement-derived MRQ metrics remain aligned to 2025-12-31."
            ],
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "6782.TW")

    assert "REVENUE_GROWTH_MRQ: 16.9% (as of 2025-12-31)" in sanitized
    assert "EARNINGS_GROWTH_MRQ: 102.8% (as of 2025-12-31)" in sanitized
    assert "LATEST_QUARTER_DATE: 2025-12-31" in sanitized
    assert "LATEST_QUARTER_DATE: 2026-03-31" not in sanitized
    assert "REVENUE_GROWTH_MRQ: 16.9% (as of 2026-03-31)" not in sanitized
    assert "Newer quarter metadata exists for 2026-03-31" in sanitized
    assert "not the latest reported quarter" in sanitized


def test_mrq_period_is_applied_per_metric_source_not_payload_wide() -> None:
    content = """### --- START DATA_BLOCK ---
REVENUE_GROWTH_MRQ: 12.0% (as of 2026-03-31)
EARNINGS_GROWTH_MRQ: 30.0% (as of 2026-03-31)
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "latest_quarter_date": "2025-12-31",
            "_latest_quarter_date_source": "yfinance_quarterly",
            "revenueGrowth_MRQ": 0.12,
            "_revenueGrowth_MRQ_source": "calculated_from_quarterly",
            "earningsGrowth_MRQ": 0.30,
            "_earningsGrowth_MRQ_source": "provider_metadata",
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "TEST")

    assert "REVENUE_GROWTH_MRQ: 12.0% (as of 2025-12-31)" in sanitized
    assert "EARNINGS_GROWTH_MRQ: 30.0%" in sanitized
    assert "EARNINGS_GROWTH_MRQ: 30.0% (as of" not in sanitized


def test_primary_latest_results_are_promoted_without_relabeling_mrq() -> None:
    content = """### --- START DATA_BLOCK ---
REVENUE_GROWTH_MRQ: 16.9%
EARNINGS_GROWTH_MRQ: 102.8%
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "latest_quarter_date": "2025-12-31",
            "_latest_quarter_date_source": "yfinance_quarterly",
            "revenueGrowth_MRQ": 0.169,
            "_revenueGrowth_MRQ_source": "calculated_from_quarterly",
            "earningsGrowth_MRQ": 1.028,
            "_earningsGrowth_MRQ_source": "calculated_from_quarterly",
        }
    )
    foreign_data = """### --- START LATEST_RESULTS ---
LATEST_RESULTS_PERIOD: Three months ended March 31, 2026
LATEST_RESULTS_PERIOD_END: 2026-03-31
LATEST_RESULTS_PRIOR_PERIOD: Three months ended March 31, 2025
LATEST_RESULTS_PRIOR_PERIOD_END: 2025-03-31
LATEST_RESULTS_PERIOD_MONTHS: 3
LATEST_RESULTS_CURRENCY: New dollars
LATEST_RESULTS_REPORTING_UNIT: thousands
LATEST_RESULTS_REVENUE: 1,500
LATEST_RESULTS_PRIOR_REVENUE: 1,000
LATEST_RESULTS_EARNINGS: 405
LATEST_RESULTS_PRIOR_EARNINGS: 200
LATEST_RESULTS_EARNINGS_SCOPE: Net income attributable to owners of parent
LATEST_RESULTS_SOURCE_URL: https://issuer.example/results
LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY
LATEST_RESULTS_REVENUE_GROWTH_YOY: 50.0%
LATEST_RESULTS_EARNINGS_GROWTH_YOY: 102.5%
### --- END LATEST_RESULTS ---
"""

    sanitized = _sanitize_fundamentals_output(
        content,
        raw_data,
        "TEST",
        foreign_data=foreign_data,
    )

    assert "LATEST_RESULTS_PERIOD_END: 2026-03-31" in sanitized
    assert "LATEST_RESULTS_EARNINGS_GROWTH_YOY: 102.5%" in sanitized
    assert "EARNINGS_GROWTH_MRQ: 102.8% (as of 2025-12-31)" in sanitized
    assert "Newer primary results exist for Three months ended March 31, 2026" in (
        sanitized
    )


def test_unvalidated_latest_period_is_promoted_without_numeric_claims() -> None:
    content = """### --- START DATA_BLOCK ---
EARNINGS_GROWTH_MRQ: 102.8%
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "latest_quarter_date": "2025-12-31",
            "_latest_quarter_date_source": "yfinance_quarterly",
            "earningsGrowth_MRQ": 1.028,
            "_earningsGrowth_MRQ_source": "calculated_from_quarterly",
        }
    )
    foreign_data = """### --- START LATEST_RESULTS ---
LATEST_RESULTS_COVERAGE_STATUS: FOUND
LATEST_RESULTS_PERIOD: Three months ended March 31, 2026
LATEST_RESULTS_PERIOD_END: 2026-03-31
LATEST_RESULTS_PRIOR_PERIOD: Three months ended March 31, 2025
LATEST_RESULTS_PRIOR_PERIOD_END: 2025-03-31
LATEST_RESULTS_PERIOD_MONTHS: 3
LATEST_RESULTS_CURRENCY: New dollars
LATEST_RESULTS_REPORTING_UNIT: thousands
LATEST_RESULTS_REVENUE: 1,500
LATEST_RESULTS_PRIOR_REVENUE: 1,000
LATEST_RESULTS_EARNINGS: 405
LATEST_RESULTS_PRIOR_EARNINGS: 200
LATEST_RESULTS_EARNINGS_SCOPE: Net income attributable to owners of parent
LATEST_RESULTS_SOURCE_URL: https://issuer.example/results
LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED
LATEST_RESULTS_REVENUE_GROWTH_YOY: N/A
LATEST_RESULTS_EARNINGS_GROWTH_YOY: N/A
### --- END LATEST_RESULTS ---
"""

    sanitized = _sanitize_fundamentals_output(
        content,
        raw_data,
        "TEST",
        foreign_data=foreign_data,
    )

    assert "LATEST_RESULTS_PERIOD_END: 2026-03-31" in sanitized
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in sanitized
    assert "LATEST_RESULTS_REVENUE: 1,500" not in sanitized
    assert "newer-period results candidate exists" in sanitized
    assert "Do not present that MRQ period as the latest reported quarter" in sanitized


def test_metadata_only_mrq_date_is_not_described_as_statement_aligned() -> None:
    content = """### --- START DATA_BLOCK ---
EARNINGS_GROWTH_MRQ: 30.0% (as of 2026-03-31)
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "latest_quarter_date": "2026-03-31",
            "_latest_quarter_date_source": "reconciled_most_recent_quarter",
            "earningsGrowth_MRQ": 0.30,
            "_earningsGrowth_MRQ_source": "provider_metadata",
        }
    )
    foreign_data = """### --- START LATEST_RESULTS ---
LATEST_RESULTS_PERIOD: Six months ended June 30, 2026
LATEST_RESULTS_PERIOD_END: 2026-06-30
LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY
### --- END LATEST_RESULTS ---
"""

    sanitized = _sanitize_fundamentals_output(
        content,
        raw_data,
        "TEST",
        foreign_data=foreign_data,
    )

    assert "EARNINGS_GROWTH_MRQ: 30.0%" in sanitized
    assert "EARNINGS_GROWTH_MRQ: 30.0% (as of" not in sanitized
    assert "statement-derived MRQ growth remains aligned" not in sanitized


def test_secondary_capex_evidence_is_preserved_for_policy_gate() -> None:
    content = """### --- START DATA_BLOCK ---
GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=0; ROA_ROE_IMPROVING=0; GROSS_MARGIN=1; GLOBAL_EXPANSION=1; R_AND_D_CAPEX_BACKLOG=1
RAW_GROWTH_SCORE: 4/6
ADJUSTED_GROWTH_SCORE: 66.7% (based on 6 available points)
### --- END DATA_BLOCK ---
"""
    foreign_data = """CAPACITY_UTILIZATION: 95%
CAPACITY_EVIDENCE_STATUS: SECONDARY
R_AND_D_CAPEX_BACKLOG_EVIDENCE: SECONDARY
"""

    sanitized = _sanitize_fundamentals_output(
        content,
        "",
        "6782.TW",
        foreign_data=foreign_data,
    )

    assert "R_AND_D_CAPEX_BACKLOG=1" in sanitized
    assert "R_AND_D_CAPEX_BACKLOG_EVIDENCE: SECONDARY" in sanitized
    assert "R_AND_D_CAPEX_BACKLOG_EVIDENCE_ADJUSTMENT" not in sanitized


def test_forward_and_cash_conversion_provenance_are_code_owned() -> None:
    content = """### --- START DATA_BLOCK ---
CURRENT_PRICE: 183.00
PE_RATIO_FORWARD: 9.37
MOAT_CFO_NI_AVG: 1.48
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "forwardEps": 19.54,
            "forwardPE": 9.365404,
            "moat_cfoToNiAvg": 1.4827,
            "moat_cfoToNiYears": 3,
            "_field_sources": {
                "forwardEps": "yfinance",
                "forwardPE": "yfinance",
                "moat_cfoToNiAvg": "yfinance",
            },
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "6782.TW")

    assert "FORWARD_EPS: 19.54" in sanitized
    assert "FORWARD_EPS_SOURCE: yfinance" in sanitized
    assert "PE_RATIO_FORWARD_SOURCE: yfinance" in sanitized
    assert "MOAT_CFO_NI_YEARS: 3" in sanitized
    assert "MOAT_CFO_NI_SOURCE: yfinance" in sanitized


def test_static_sector_pe_reference_provenance_is_promoted() -> None:
    content = """### --- START DATA_BLOCK ---
SECTOR_MEDIAN_PE: 22
PE_VS_SECTOR: 0.50
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "sectorMedianPE": 22,
            "peVsSector": 0.50,
            "sectorPeReferenceType": "STATIC_POLICY_REFERENCE",
            "sectorPeReferenceAsOf": "N/A",
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "6782.TW")

    assert "SECTOR_PE_REFERENCE_TYPE: STATIC_POLICY_REFERENCE" in sanitized
    assert "SECTOR_PE_REFERENCE_AS_OF: N/A" in sanitized


def test_sanitize_fundamentals_output_extracts_production_raw_payload() -> None:
    content = """### --- START DATA_BLOCK ---
REVENUE_GROWTH_TTM: 20.3%
EARNINGS_GROWTH_TTM: 33.5%
NET_DEBT_EBITDA: -0.01
CASH_TO_ASSETS: 33.1%
### --- END DATA_BLOCK ---
"""
    raw_data = (
        "### TOOL 1: get_financial_metrics\n"
        + json.dumps(
            {
                "revenueGrowth": 0.203,
                "revenueGrowth_TTM": None,
                "earningsGrowth": 0.389,
                "earningsGrowth_TTM": None,
                "totalDebt": 15_579_192_320,
                "cashAndShortTermInvestments": 1_603_617_000,
                "ebitda": 7_168_355_840,
                "marketCap": 82_678_956_032,
                "totalAssets": 48_487_647_000,
                "capital_cashToAssets": 0.0331,
            }
        )
        + "\n### TOOL 2: supplemental search\nNoisy text."
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert "REVENUE_GROWTH_TTM: N/A" in sanitized
    assert "EARNINGS_GROWTH_TTM: N/A" in sanitized
    assert "NET_DEBT_EBITDA: 1.95" in sanitized
    assert "CASH_TO_ASSETS: 3.3%" in sanitized
    assert "GROWTH_DATA_QUALITY_NOTE:" in sanitized
    assert "BALANCE_SHEET_DATA_QUALITY_NOTE:" in sanitized


def test_sanitize_fundamentals_output_extracts_non_first_tool_payload() -> None:
    content = """### --- START DATA_BLOCK ---
NET_DEBT_EBITDA: -0.01
### --- END DATA_BLOCK ---
"""
    raw_data = "### TOOL 2: get_financial_metrics\n" + json.dumps(
        {
            "totalDebt": 15_579_192_320,
            "cashAndShortTermInvestments": 1_603_617_000,
            "ebitda": 7_168_355_840,
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert "NET_DEBT_EBITDA: 1.95" in sanitized


def test_sanitize_fundamentals_output_ignores_unmarked_json() -> None:
    content = """### --- START DATA_BLOCK ---
NET_DEBT_EBITDA: -0.01
### --- END DATA_BLOCK ---
"""
    raw_data = (
        'Unrelated search output {"totalDebt": 1, '
        '"cashAndShortTermInvestments": 0, "ebitda": 1}'
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert sanitized == content


def test_sanitize_fundamentals_output_skips_malformed_marked_payload() -> None:
    content = """### --- START DATA_BLOCK ---
NET_DEBT_EBITDA: -0.01
### --- END DATA_BLOCK ---
"""
    raw_data = "### TOOL 1: get_financial_metrics\n{not valid json"

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert sanitized == content


def test_sanitize_fundamentals_output_handles_zero_ebitda() -> None:
    content = """### --- START DATA_BLOCK ---
NET_DEBT_EBITDA: -0.01
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "totalDebt": 15_579_192_320,
            "cashAndShortTermInvestments": 1_603_617_000,
            "ebitda": 0,
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert "NET_DEBT_EBITDA: N/A" in sanitized


def test_sanitize_fundamentals_output_overrides_na_when_raw_value_computes() -> None:
    content = """### --- START DATA_BLOCK ---
NET_DEBT_EBITDA: N/A
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "totalDebt": 15_579_192_320,
            "cashAndShortTermInvestments": 1_603_617_000,
            "ebitda": 7_168_355_840,
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert "NET_DEBT_EBITDA: 1.95" in sanitized


def test_sanitize_fundamentals_output_prefers_capital_cash_to_assets() -> None:
    content = """### --- START DATA_BLOCK ---
CASH_TO_ASSETS: 33.1%
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "cashAndShortTermInvestments": 50,
            "totalAssets": 100,
            "capital_cashToAssets": 0.0331,
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert "CASH_TO_ASSETS: 3.3%" in sanitized
    assert "CASH_TO_ASSETS: 50.0%" not in sanitized


def test_sanitize_fundamentals_output_accepts_four_hash_datablock() -> None:
    content = """#### --- START DATA_BLOCK ---
NET_DEBT_EBITDA: -0.01
#### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "totalDebt": 15_579_192_320,
            "cashAndShortTermInvestments": 1_603_617_000,
            "ebitda": 7_168_355_840,
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert "NET_DEBT_EBITDA: 1.95" in sanitized


def test_sanitize_fundamentals_output_flags_stale_annual_statements() -> None:
    content = """### --- START DATA_BLOCK ---
REVENUE_GROWTH_FY: 9.9%
EARNINGS_GROWTH_FY: 8.9%
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "revenueGrowth": 0.099,
            "earningsGrowth": 0.089,
            "statements_stale": True,
            "_income_statement_date": "2024-12-31",
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "FRAGUAB.MX")

    assert "GROWTH_DATA_STALE:" in sanitized
    assert "2024-12-31" in sanitized


def test_sanitize_fundamentals_output_no_stale_flag_when_current() -> None:
    content = """### --- START DATA_BLOCK ---
REVENUE_GROWTH_FY: 22.3%
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "revenueGrowth": 0.223,
            "earningsGrowth": 0.198,
            "_income_statement_date": "2025-06-30",
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "4396.T")

    assert "GROWTH_DATA_STALE:" not in sanitized


def test_sanitize_fundamentals_output_reconciles_b3_balance_sheet_fields() -> None:
    content = """### --- START DATA_BLOCK ---
NET_CASH_TO_MARKET_CAP: 1.8%
CASH_TO_ASSETS: 33.1%
NET_DEBT_EBITDA: -0.01
PFIC_ASSET_RATIO: 33.1%
PFIC_CASH_TRAP: YES
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "totalDebt": 15_579_192_320,
            "cashAndShortTermInvestments": 1_603_617_000,
            "ebitda": 7_168_355_840,
            "marketCap": 82_678_956_032,
            "totalAssets": 48_487_647_000,
            "capital_cashToAssets": 0.0331,
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert "NET_DEBT_EBITDA: 1.95" in sanitized
    assert "NET_CASH_TO_MARKET_CAP: -16.9%" in sanitized
    assert "CASH_TO_ASSETS: 3.3%" in sanitized
    assert "PFIC_ASSET_RATIO: 3.3%" in sanitized
    assert "PFIC_CASH_TRAP: NO" in sanitized
    assert "PFIC_CASH_TRAP: YES" not in sanitized


def test_sanitize_fundamentals_output_promotes_pfic_proximity_to_medium() -> None:
    content = """### --- START DATA_BLOCK ---
PFIC_RISK: LOW
CASH_TO_ASSETS: 49.3%
PFIC_ASSET_RATIO: 49.3%
PFIC_CASH_TRAP: NO
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"capital_cashToAssets": 0.4934})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "3393.T")

    assert "PFIC_RISK: MEDIUM" in sanitized
    assert "PFIC_ASSET_RATIO: 49.3%" in sanitized
    assert "PFIC_CASH_TRAP: NO" in sanitized


def test_sanitize_fundamentals_output_appends_missing_pfic_risk_on_proximity() -> None:
    content = """### --- START DATA_BLOCK ---
CASH_TO_ASSETS: 49.3%
PFIC_ASSET_RATIO: 49.3%
PFIC_CASH_TRAP: NO
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "cashAndShortTermInvestments": 4_934,
            "totalAssets": 10_000,
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "3393.T")

    assert "PFIC_RISK: MEDIUM" in sanitized


def test_sanitize_fundamentals_output_keeps_pfic_low_below_proximity() -> None:
    content = """### --- START DATA_BLOCK ---
PFIC_RISK: LOW
CASH_TO_ASSETS: 44.9%
PFIC_ASSET_RATIO: 44.9%
PFIC_CASH_TRAP: NO
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"capital_cashToAssets": 0.449})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "3393.T")

    assert "PFIC_RISK: LOW" in sanitized
    assert "PFIC_RISK: MEDIUM" not in sanitized


def test_sanitize_fundamentals_output_preserves_pfic_high_on_proximity() -> None:
    content = """### --- START DATA_BLOCK ---
PFIC_RISK: HIGH
CASH_TO_ASSETS: 49.3%
PFIC_ASSET_RATIO: 49.3%
PFIC_CASH_TRAP: NO
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"capital_cashToAssets": 0.4934})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "3393.T")

    assert "PFIC_RISK: HIGH" in sanitized
    assert "PFIC_RISK: MEDIUM" not in sanitized


def test_sanitize_fundamentals_output_leaves_pfic_risk_when_basis_unreliable() -> None:
    content = """### --- START DATA_BLOCK ---
PFIC_RISK: LOW
CASH_TO_ASSETS: 49.3%
PFIC_ASSET_RATIO: 49.3%
PFIC_CASH_TRAP: NO
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"totalDebt": 1_000, "ebitda": 500})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "3393.T")

    assert "PFIC_RISK: LOW" in sanitized
    assert "PFIC_RISK: MEDIUM" not in sanitized
    assert "PFIC_ASSET_NOTE:" in sanitized


def test_sanitize_fundamentals_output_downgrades_unreliable_pfic_basis() -> None:
    content = """### --- START DATA_BLOCK ---
CASH_TO_ASSETS: 33.1%
PFIC_ASSET_RATIO: 33.1%
PFIC_CASH_TRAP: YES
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"totalDebt": 1_000, "ebitda": 500})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert "CASH_TO_ASSETS: N/A" in sanitized
    assert "PFIC_ASSET_RATIO: N/A" in sanitized
    assert "PFIC_CASH_TRAP: N/A" in sanitized
    assert "PFIC_ASSET_NOTE:" in sanitized


def test_sanitize_fundamentals_output_appends_coverage_quality_note() -> None:
    content = """### --- START DATA_BLOCK ---
ANALYST_COVERAGE_ENGLISH: 2
ANALYST_COVERAGE_TOTAL_EST: HIGH
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"revenueGrowth_TTM": 0.1})
    foreign_data = "Estimated Local Analysts: HIGH"

    sanitized = _sanitize_fundamentals_output(
        content,
        raw_data,
        "B3SA3.SA",
        foreign_data=foreign_data,
    )

    assert "ANALYST_COVERAGE_DATA_QUALITY_NOTE:" in sanitized


def test_sanitize_fundamentals_output_uses_foreign_coverage_signal() -> None:
    content = """### --- START DATA_BLOCK ---
ANALYST_COVERAGE_ENGLISH: 2
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"revenueGrowth_TTM": 0.1})
    foreign_data = "Estimated Local Analysts: HIGH"

    sanitized = _sanitize_fundamentals_output(
        content,
        raw_data,
        "B3SA3.SA",
        foreign_data=foreign_data,
    )

    assert "ANALYST_COVERAGE_DATA_QUALITY_NOTE:" in sanitized


def test_sanitize_invalidates_home_ticker_adr_routing() -> None:
    content = """### --- START DATA_BLOCK ---
ADR_EXISTS: YES
ADR_TYPE: SPONSORED
ADR_TICKER: POMO4.SA
ADR_EXCHANGE: SAO
ADR_THESIS_IMPACT: MODERATE_CONCERN
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"trailingPE": 6.0})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "POMO4.SA")

    assert "ADR_EXISTS: YES" in sanitized
    assert "ADR_TYPE: SPONSORED" in sanitized
    assert "ADR_TICKER: None" in sanitized
    assert "ADR_EXCHANGE: None" in sanitized
    assert "ADR_THESIS_IMPACT: UNCERTAIN" in sanitized
    assert "ADR_DATA_QUALITY_NOTE: Invalid ADR routing fields removed" in sanitized


def test_sanitize_invalidates_suffix_stripped_home_ticker_as_adr() -> None:
    content = """### --- START DATA_BLOCK ---
ADR_EXISTS: YES
ADR_TYPE: SPONSORED
ADR_TICKER: B3SA3
ADR_EXCHANGE: OTC-OTCQX
ADR_THESIS_IMPACT: MODERATE_CONCERN
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"trailingPE": 17.9})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert "ADR_TICKER: None" in sanitized
    assert "ADR_EXCHANGE: None" in sanitized
    assert "ADR_THESIS_IMPACT: UNCERTAIN" in sanitized
    assert "ADR_DATA_QUALITY_NOTE: Invalid ADR routing fields removed" in sanitized


def test_sanitize_invalidates_non_us_adr_exchange() -> None:
    content = """### --- START DATA_BLOCK ---
ADR_EXISTS: YES
ADR_TYPE: SPONSORED
ADR_TICKER: EXAMPLE
ADR_EXCHANGE: SAO
ADR_THESIS_IMPACT: MODERATE_CONCERN
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"trailingPE": 6.0})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "EXMP3.SA")

    assert "ADR_TICKER: None" in sanitized
    assert "ADR_EXCHANGE: None" in sanitized
    assert "ADR_THESIS_IMPACT: UNCERTAIN" in sanitized


def test_sanitize_preserves_valid_adr_routing() -> None:
    content = """### --- START DATA_BLOCK ---
ADR_EXISTS: YES
ADR_TYPE: SPONSORED
ADR_TICKER: ABEV
ADR_EXCHANGE: NYSE
ADR_THESIS_IMPACT: MODERATE_CONCERN
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "trailingPE": 16.0,
            "revenueGrowth_TTM": 0.1,
            "revenueGrowth_MRQ": 0.1,
            "earningsGrowth_TTM": 0.1,
            "earningsGrowth_MRQ": 0.1,
            "growth_trajectory": "STABLE",
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "ABEV3.SA")

    # Valid ADR routing is preserved untouched.
    for line in (
        "ADR_EXISTS: YES",
        "ADR_TYPE: SPONSORED",
        "ADR_TICKER: ABEV",
        "ADR_EXCHANGE: NYSE",
        "ADR_THESIS_IMPACT: MODERATE_CONCERN",
    ):
        assert line in sanitized
    # The only added line is the reliability contract (no forward fields -> UNAVAILABLE).
    assert "VALUATION_INPUT_RELIABILITY: UNAVAILABLE" in sanitized


def test_sanitize_downgrades_loose_otc_sponsored_claim() -> None:
    content = """### --- START DATA_BLOCK ---
ADR_EXISTS: YES
ADR_TYPE: SPONSORED
ADR_TICKER: BOLSY
ADR_EXCHANGE: OTC-OTCQX
ADR_THESIS_IMPACT: MODERATE_CONCERN
### --- END DATA_BLOCK ---
"""
    raw_data = (
        "### TOOL 2: get_fundamental_analysis\n"
        "Investing.com profile: B3 SA Brasil Bolsa Balcao sponsored ADR BOLSY "
        "trades over the counter. No depositary or SEC sponsorship metadata."
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert "ADR_TYPE: UNCERTAIN" in sanitized
    assert "ADR_THESIS_IMPACT: UNCERTAIN" in sanitized
    assert "ADR_DATA_QUALITY_NOTE: OTC sponsorship claim lacked" in sanitized


def test_sanitize_corrects_explicit_unsponsored_adr() -> None:
    content = """### --- START DATA_BLOCK ---
ADR_EXISTS: YES
ADR_TYPE: SPONSORED
ADR_TICKER: BOLSY
ADR_EXCHANGE: OTC-OTCQX
ADR_THESIS_IMPACT: MODERATE_CONCERN
### --- END DATA_BLOCK ---
"""
    raw_data = (
        "Citi Depositary Receipts notice for BOLSY: "
        "Sponsorship Level: Unsponsored ADR Program."
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert "ADR_TYPE: UNSPONSORED" in sanitized
    assert "ADR_THESIS_IMPACT: EMERGING_INTEREST" in sanitized
    assert "ADR_DATA_QUALITY_NOTE: OTC ADR sponsorship corrected" in sanitized


def test_sanitize_preserves_otc_sponsored_with_authoritative_evidence() -> None:
    content = """### --- START DATA_BLOCK ---
ADR_EXISTS: YES
ADR_TYPE: SPONSORED
ADR_TICKER: EXMPY
ADR_EXCHANGE: OTC-OTCQX
ADR_THESIS_IMPACT: MODERATE_CONCERN
### --- END DATA_BLOCK ---
"""
    raw_data = (
        "Source: https://www.adrbny.com/example\n"
        "The company maintains a sponsored Level I ADR program."
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "EXMP.SA")

    assert sanitized == content


def test_sanitize_preserves_nyse_sponsored_claim() -> None:
    content = """### --- START DATA_BLOCK ---
ADR_EXISTS: YES
ADR_TYPE: SPONSORED
ADR_TICKER: ABEV
ADR_EXCHANGE: NYSE
ADR_THESIS_IMPACT: MODERATE_CONCERN
### --- END DATA_BLOCK ---
"""
    raw_data = "Generic profile text only."

    sanitized = _sanitize_fundamentals_output(content, raw_data, "ABEV3.SA")

    assert sanitized == content


def test_sanitize_preserves_uncertain_otc_claim() -> None:
    content = """### --- START DATA_BLOCK ---
ADR_EXISTS: YES
ADR_TYPE: UNCERTAIN
ADR_TICKER: BOLSY
ADR_EXCHANGE: OTC-OTCQX
ADR_THESIS_IMPACT: UNCERTAIN
### --- END DATA_BLOCK ---
"""
    raw_data = "Generic OTC profile text only."

    sanitized = _sanitize_fundamentals_output(content, raw_data, "B3SA3.SA")

    assert sanitized == content


def test_sanitize_quarantined_low_pe_sets_valuation_to_na() -> None:
    content = """### --- START DATA_BLOCK ---
PE_RATIO_TTM: 1.60
PEG_RATIO: 0.20
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"_pe_low_anomaly_quarantined": True})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "SEER3.SA")

    assert "PE_RATIO_TTM: N/A" in sanitized
    assert "PEG_RATIO: N/A" in sanitized


def test_sanitize_low_pe_flag_only_keeps_valuation_lines() -> None:
    content = """### --- START DATA_BLOCK ---
PE_RATIO_TTM: 4.20
PEG_RATIO: 0.70
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {
            "_pe_low_anomaly_flag": "LOW_PE_REQUIRES_INVESTIGATION",
            "revenueGrowth_TTM": 0.1,
            "revenueGrowth_MRQ": 0.1,
            "earningsGrowth_TTM": 0.1,
            "earningsGrowth_MRQ": 0.1,
            "growth_trajectory": "STABLE",
        }
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "TEST.SA")

    # The low-PE *flag* (not a quarantine marker) must not blank the valuation lines.
    assert "PE_RATIO_TTM: 4.20" in sanitized
    assert "PEG_RATIO: 0.70" in sanitized
    # Reliability contract appended; no forward fields present -> UNAVAILABLE.
    assert "VALUATION_INPUT_RELIABILITY: UNAVAILABLE" in sanitized


# --------------------------------------------------------------------------- #
# VALUATION_INPUT_RELIABILITY classifier + DATA_BLOCK contract
# --------------------------------------------------------------------------- #
def test_valuation_input_reliability_usable_when_forward_present_and_clean() -> None:
    assert _valuation_input_reliability({"forwardPE": 12.0}) == "USABLE"


def test_valuation_input_reliability_quarantined_markers() -> None:
    # Every distrust marker the fetcher/merge layer can set → QUARANTINED.
    assert (
        _valuation_input_reliability({"_split_sensitive_metrics_quarantined": True})
        == "QUARANTINED"
    )
    assert (
        _valuation_input_reliability({"_pe_low_anomaly_quarantined": True})
        == "QUARANTINED"
    )
    assert (
        _valuation_input_reliability({"_pe_unit_error_quarantined": "forward"})
        == "QUARANTINED"
    )
    # Trailing P/E is also a valuation input — contract is valuation-input, not forecast.
    assert (
        _valuation_input_reliability({"_pe_unit_error_quarantined": "trailing"})
        == "QUARANTINED"
    )
    assert (
        _valuation_input_reliability({"_forwardPE_quarantine_reason": "recent split"})
        == "QUARANTINED"
    )


def test_valuation_input_reliability_unavailable_cases() -> None:
    assert _valuation_input_reliability({}) == "UNAVAILABLE"
    assert (
        _valuation_input_reliability(
            {
                "trailingPE": 10.0,
                "forwardPE": None,
                "forwardEps": None,
                "pegRatio": None,
            }
        )
        == "UNAVAILABLE"
    )


def test_valuation_input_reliability_unit_error_only_matches_known_values() -> None:
    # A stray truthy (non-"forward"/"trailing") marker must NOT trip the quarantine
    # branch; with a present forward field the result is USABLE.
    assert (
        _valuation_input_reliability(
            {"_pe_unit_error_quarantined": True, "forwardPE": 9.0}
        )
        == "USABLE"
    )


def test_sanitize_appends_valuation_input_reliability_usable() -> None:
    content = """### --- START DATA_BLOCK ---
PE_RATIO_FORWARD: 10.0
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"forwardPE": 10.0, "forwardEps": 1.0})
    sanitized = _sanitize_fundamentals_output(content, raw_data, "TEST.T")
    assert "VALUATION_INPUT_RELIABILITY: USABLE" in sanitized


def test_sanitize_appends_valuation_input_reliability_quarantined() -> None:
    content = """### --- START DATA_BLOCK ---
PE_RATIO_FORWARD: 10.0
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {"_split_sensitive_metrics_quarantined": True, "forwardPE": 10.0}
    )
    sanitized = _sanitize_fundamentals_output(content, raw_data, "TEST.T")
    assert "VALUATION_INPUT_RELIABILITY: QUARANTINED" in sanitized
    # Appears exactly once (replace-or-append, never duplicated).
    assert sanitized.count("VALUATION_INPUT_RELIABILITY:") == 1


def test_sanitize_corrects_fabricated_pe_ratio_ttm() -> None:
    """A PE_RATIO_TTM that contradicts fetched trailingPE is reconciled to the raw value."""
    content = """### --- START DATA_BLOCK ---
PE_RATIO_TTM: 8.20
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"trailingPE": 11.473684})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "BEC.SI")

    assert "PE_RATIO_TTM: 11.47" in sanitized
    assert "PE_RATIO_TTM: 8.20" not in sanitized
    # The correction is carried by the distinct valuation note (own changed_valuation
    # flag), not the growth-data note.
    assert (
        "VALUATION_DATA_QUALITY_NOTE: Valuation/margin scalars reconciled "
        "to fetched raw metrics." in sanitized
    )


def test_sanitize_corrects_payout_and_margin_scalars() -> None:
    content = """### --- START DATA_BLOCK ---
PAYOUT_RATIO: 50.0%
NET_MARGIN: 25.0%
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"payoutRatio": 0.3685, "profitMargins": 0.05894})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "BEC.SI")

    # 50% vs 36.9% (35.7% rel) and 25% vs 5.9% (>3x) both exceed tolerance.
    assert "PAYOUT_RATIO: 36.9%" in sanitized
    assert "NET_MARGIN: 5.9%" in sanitized
    assert "VALUATION_DATA_QUALITY_NOTE:" in sanitized


def test_sanitize_leaves_small_margin_divergence_within_tolerance() -> None:
    """A sub-threshold margin gap (5.57 vs 5.89) is left to the agent, not over-corrected."""
    content = """### --- START DATA_BLOCK ---
NET_MARGIN: 5.57%
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"profitMargins": 0.05894})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "BEC.SI")

    assert "NET_MARGIN: 5.57%" in sanitized
    assert "VALUATION_DATA_QUALITY_NOTE:" not in sanitized


def test_sanitize_leaves_valuation_within_tolerance_untouched() -> None:
    content = """### --- START DATA_BLOCK ---
PE_RATIO_TTM: 11.60
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"trailingPE": 11.473684})

    sanitized = _sanitize_fundamentals_output(content, raw_data, "BEC.SI")

    assert "PE_RATIO_TTM: 11.60" in sanitized
    assert "VALUATION_DATA_QUALITY_NOTE:" not in sanitized


def test_sanitize_does_not_erase_valuation_when_raw_absent() -> None:
    """A filing-derived value must survive when the raw payload lacks the field."""
    content = """### --- START DATA_BLOCK ---
PE_RATIO_TTM: 8.20
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps({"totalDebt": 100, "ebitda": 50})  # no trailingPE

    sanitized = _sanitize_fundamentals_output(content, raw_data, "BEC.SI")

    assert "PE_RATIO_TTM: 8.20" in sanitized
    assert "PE_RATIO_TTM: N/A" not in sanitized
    assert "VALUATION_DATA_QUALITY_NOTE:" not in sanitized


def test_sanitize_skips_pe_reconciliation_when_quarantined() -> None:
    """The low-PE quarantine path wins: PE goes to N/A, not the raw value."""
    content = """### --- START DATA_BLOCK ---
PE_RATIO_TTM: 8.20
PEG_RATIO: 0.46
### --- END DATA_BLOCK ---
"""
    raw_data = json.dumps(
        {"trailingPE": 11.473684, "_pe_low_anomaly_quarantined": True}
    )

    sanitized = _sanitize_fundamentals_output(content, raw_data, "BEC.SI")

    assert "PE_RATIO_TTM: N/A" in sanitized
    assert "PE_RATIO_TTM: 11.47" not in sanitized


class TestScoreConsistency:
    """reconcile_score_consistency: hybrid correct-vs-flag policy."""

    @staticmethod
    def _body(*lines: str) -> str:
        return "\n".join(lines)

    def test_consistent_scores_untouched(self) -> None:
        body = self._body(
            "RAW_HEALTH_SCORE: 9.5/12",
            "ADJUSTED_HEALTH_SCORE: 79.2% (based on 12 available points)",
            "RAW_GROWTH_SCORE: 4/6",
            "ADJUSTED_GROWTH_SCORE: 67% (based on 6 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not corrected
        assert not suspect

    def test_available_denominator_convention_accepted(self) -> None:
        """RAW x/available (the dominant real-world convention) is not suspect."""
        body = self._body(
            "SECTOR: Financials",
            "RAW_HEALTH_SCORE: 9/11",
            "ADJUSTED_HEALTH_SCORE: 82% (based on 11 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not corrected and not suspect

    def test_arithmetic_error_corrected(self) -> None:
        body = self._body(
            "RAW_HEALTH_SCORE: 8/12",
            "ADJUSTED_HEALTH_SCORE: 50% (based on 10 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert corrected and not suspect
        assert "ADJUSTED_HEALTH_SCORE: 80.0% (based on 10 available points)" in updated
        assert "HEALTH_SCORE_DATA_QUALITY_NOTE:" in updated

    def test_rounding_within_tolerance_no_churn(self) -> None:
        body = self._body(
            "RAW_HEALTH_SCORE: 10/12",
            "ADJUSTED_HEALTH_SCORE: 83% (based on 12 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not corrected and not suspect

    def test_denominator_incoherent_flagged_not_fixed(self) -> None:
        """Fraction disagrees with both rubric total and available points."""
        body = self._body(
            "RAW_HEALTH_SCORE: 4/10",
            "ADJUSTED_HEALTH_SCORE: 40% (based on 8 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert suspect and not corrected
        assert "HEALTH_SCORE_CONSISTENCY: SUSPECT" in updated
        assert "ADJUSTED_HEALTH_SCORE: 40%" in updated  # never rewritten

    def test_earned_exceeds_available_flagged(self) -> None:
        body = self._body(
            "RAW_HEALTH_SCORE: 11/9",
            "ADJUSTED_HEALTH_SCORE: 100% (based on 9 available points)",
        )
        _, corrected, suspect = reconcile_score_consistency(body)
        assert suspect and not corrected

    def test_available_above_rubric_total_flagged(self) -> None:
        body = self._body(
            "RAW_GROWTH_SCORE: 5/8",
            "ADJUSTED_GROWTH_SCORE: 63% (based on 8 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert suspect and not corrected
        assert "GROWTH_SCORE_CONSISTENCY: SUSPECT" in updated

    def test_financials_de_removed_without_denominator_reduction(self) -> None:
        body = self._body(
            "SECTOR: Financial Services",
            "SECTOR_ADJUSTMENTS: Financials (Insurance) - D/E removed; ROE >12%.",
            "RAW_HEALTH_SCORE: 10/12",
            "ADJUSTED_HEALTH_SCORE: 83% (based on 12 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert suspect and not corrected
        assert "HEALTH_SCORE_CONSISTENCY: SUSPECT" in updated
        assert "D/E removed" in updated

    def test_financials_de_removed_with_reduced_denominator_ok(self) -> None:
        """AGS.BR June-29 shape: internally consistent, must not flag."""
        body = self._body(
            "SECTOR: Financials",
            "SECTOR_ADJUSTMENTS: Financials: D/E Ratio and EV/EBITDA removed.",
            "RAW_HEALTH_SCORE: 7.5/9",
            "ADJUSTED_HEALTH_SCORE: 83% (based on 9 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not corrected and not suspect

    def test_malformed_lines_are_skipped(self) -> None:
        body = self._body(
            "RAW_HEALTH_SCORE: strong",
            "ADJUSTED_HEALTH_SCORE: solid pass",
            "RAW_GROWTH_SCORE: 4/0",
            "ADJUSTED_GROWTH_SCORE: 80% (based on 0 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        # Health lines unparseable -> skipped; growth zero denominator -> suspect.
        assert not corrected
        assert suspect
        assert "GROWTH_SCORE_CONSISTENCY: SUSPECT" in updated
        assert "HEALTH_SCORE_CONSISTENCY" not in updated

    def test_missing_adjusted_line_skipped(self) -> None:
        body = self._body("RAW_HEALTH_SCORE: 8/12")
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not corrected and not suspect

    def test_sanitize_runs_score_check_without_raw_data(self) -> None:
        """Empty raw payload no longer short-circuits the DATA_BLOCK-internal check."""
        content = """### --- START DATA_BLOCK ---
RAW_HEALTH_SCORE: 8/12
ADJUSTED_HEALTH_SCORE: 50% (based on 10 available points)
### --- END DATA_BLOCK ---
"""
        sanitized = _sanitize_fundamentals_output(content, "", "AGS.BR")
        assert (
            "ADJUSTED_HEALTH_SCORE: 80.0% (based on 10 available points)" in sanitized
        )

    def test_earned_over_available_parenthetical_accepted(self) -> None:
        """Prompt-example form: '70% (7/10 available)' = earned 7 of 10."""
        body = self._body(
            "RAW_HEALTH_SCORE: 7/12",
            "ADJUSTED_HEALTH_SCORE: 70% (7/10 available)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not corrected and not suspect

    def test_available_statement_parenthetical_accepted(self) -> None:
        """Real-world form: '79% (12/12 available)' with RAW 9.5/12 is a
        denominator statement, not an earned claim — must not flag."""
        body = self._body(
            "RAW_HEALTH_SCORE: 9.5/12",
            "ADJUSTED_HEALTH_SCORE: 79% (12/12 available)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not corrected and not suspect

    def test_conflicting_earned_readings_flagged(self) -> None:
        """Self-consistent earned/available reading that contradicts RAW."""
        body = self._body(
            "RAW_HEALTH_SCORE: 5/12",
            "ADJUSTED_HEALTH_SCORE: 70% (7/10 available)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert suspect and not corrected
        assert "HEALTH_SCORE_CONSISTENCY: SUSPECT" in updated


class TestScoreBreakdown:
    """Per-criterion *_SCORE_BREAKDOWN audit (A5b): numerator becomes checkable."""

    _HEALTH_OK = (
        "HEALTH_SCORE_BREAKDOWN: ROE=1; ROA=0.5; OPERATING_MARGIN=0; DE_RATIO=1; "
        "NET_DEBT_EBITDA=N/A; CURRENT_RATIO=1; OCF_POSITIVE=1; FCF_POSITIVE=1; "
        "FCF_YIELD=N/A; PE_OR_PEG=1; EV_EBITDA=N/A; PB_OR_PS=0"
    )  # numeric sum 6.5, available 9

    @staticmethod
    def _body(*lines: str) -> str:
        return "\n".join(lines)

    def test_consistent_breakdown_passes(self) -> None:
        body = self._body(
            self._HEALTH_OK,
            "RAW_HEALTH_SCORE: 6.5/9",
            "ADJUSTED_HEALTH_SCORE: 72% (based on 9 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not corrected and not suspect

    def test_breakdown_sum_mismatch_flagged(self) -> None:
        body = self._body(
            self._HEALTH_OK,
            "RAW_HEALTH_SCORE: 5/9",
            "ADJUSTED_HEALTH_SCORE: 56% (based on 9 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert suspect and not corrected
        assert "HEALTH_SCORE_CONSISTENCY: SUSPECT" in updated
        assert "breakdown awards sum 6.5 != RAW earned 5" in updated

    def test_breakdown_available_mismatch_flagged(self) -> None:
        body = self._body(
            self._HEALTH_OK,
            "RAW_HEALTH_SCORE: 6.5/10",
            "ADJUSTED_HEALTH_SCORE: 65% (based on 10 available points)",
        )
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect
        assert "breakdown available points 9 != stated available 10" in updated

    def test_financials_removed_and_na_reconcile_denominator(self) -> None:
        body = self._body(
            "SECTOR: Financials",
            "SECTOR_ADJUSTMENTS: Financials: D/E removed; ROE threshold 12%.",
            "HEALTH_SCORE_BREAKDOWN: ROE=1; ROA=1; OPERATING_MARGIN=0.5; "
            "DE_RATIO=REMOVED; NET_DEBT_EBITDA=N/A; CURRENT_RATIO=1; "
            "OCF_POSITIVE=1; FCF_POSITIVE=1; FCF_YIELD=N/A; PE_OR_PEG=1; "
            "EV_EBITDA=REMOVED; PB_OR_PS=1",
            "RAW_HEALTH_SCORE: 7.5/8",
            "ADJUSTED_HEALTH_SCORE: 94% (based on 8 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not corrected and not suspect

    def test_unknown_and_missing_keys_flagged(self) -> None:
        body = self._body(
            "HEALTH_SCORE_BREAKDOWN: ROE=1; MYSTERY_METRIC=1",
            "RAW_HEALTH_SCORE: 2/12",
            "ADJUSTED_HEALTH_SCORE: 17% (based on 12 available points)",
        )
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect
        assert "breakdown keys do not match rubric" in updated
        assert "MYSTERY_METRIC" in updated

    def test_duplicate_keys_flagged(self) -> None:
        body = self._body(
            "HEALTH_SCORE_BREAKDOWN: ROE=1; ROE=0; ROA=1",
            "RAW_HEALTH_SCORE: 2/12",
            "ADJUSTED_HEALTH_SCORE: 17% (based on 12 available points)",
        )
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect
        assert "unparseable or has duplicate criteria" in updated

    def test_malformed_award_values_flagged_not_raised(self) -> None:
        body = self._body(
            "HEALTH_SCORE_BREAKDOWN: ROE=maybe; ROA=strong",
            "RAW_HEALTH_SCORE: 2/12",
            "ADJUSTED_HEALTH_SCORE: 17% (based on 12 available points)",
        )
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect  # nothing parseable -> unparseable reason

    def test_absent_breakdown_line_degrades_to_totals_only(self) -> None:
        """Pre-v9.31 reports: identical behavior to the A2 validator."""
        body = self._body(
            "RAW_HEALTH_SCORE: 9/11",
            "ADJUSTED_HEALTH_SCORE: 82% (based on 11 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not corrected and not suspect

    def test_growth_breakdown_validated_independently(self) -> None:
        body = self._body(
            "GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=0; "
            "ROA_ROE_IMPROVING=0.5; GROSS_MARGIN=1; GLOBAL_EXPANSION=0; "
            "R_AND_D_CAPEX_BACKLOG=1",
            "RAW_GROWTH_SCORE: 4/6",
            "ADJUSTED_GROWTH_SCORE: 67% (based on 6 available points)",
        )
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect  # 3.5 != 4
        assert "GROWTH_SCORE_CONSISTENCY: SUSPECT" in updated


class TestScoreBreakdownObjectiveChecks:
    _BASE = (
        "HEALTH_SCORE_BREAKDOWN: ROE=1; ROA=1; OPERATING_MARGIN=1; DE_RATIO=1; "
        "NET_DEBT_EBITDA=1; CURRENT_RATIO=1; OCF_POSITIVE=1; FCF_POSITIVE=1; "
        "FCF_YIELD=1; PE_OR_PEG=1; EV_EBITDA=1; PB_OR_PS=1"
    )  # sum 12/12

    def _body(self, *extra: str) -> str:
        return "\n".join(
            (
                *extra,
                self._BASE,
                "RAW_HEALTH_SCORE: 12/12",
                "ADJUSTED_HEALTH_SCORE: 100% (based on 12 available points)",
            )
        )

    def test_ocf_award_with_negative_ocf_flagged(self) -> None:
        body = self._body("OPERATING_CASH_FLOW: -¥1.2B")
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect
        assert "OCF_POSITIVE=1 but OPERATING_CASH_FLOW is negative" in updated

    def test_fcf_award_with_negative_fcf_flagged(self) -> None:
        body = self._body("FCF: -$120M")
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect
        assert "FCF_POSITIVE=1 but FCF is negative" in updated

    def test_pe_award_with_clearly_failing_multiples_flagged(self) -> None:
        body = self._body("SECTOR: Industrials", "PE_RATIO_TTM: 25.0", "PEG_RATIO: 1.8")
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect
        assert "PE_OR_PEG" in updated

    def test_pe_check_skipped_for_information_technology(self) -> None:
        """IT has a documented P/S alternative to the P/E gate."""
        body = self._body(
            "SECTOR: Information Technology", "PE_RATIO_TTM: 25.0", "PEG_RATIO: 1.8"
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert not suspect and not corrected

    def test_pe_near_threshold_not_flagged(self) -> None:
        body = self._body(
            "SECTOR: Industrials", "PE_RATIO_TTM: 19.0", "PEG_RATIO: 1.25"
        )
        _, corrected, suspect = reconcile_score_consistency(body)
        assert not suspect and not corrected

    def test_pe_check_needs_both_multiples_present(self) -> None:
        body = self._body("SECTOR: Industrials", "PE_RATIO_TTM: 25.0")
        _, corrected, suspect = reconcile_score_consistency(body)
        assert not suspect and not corrected

    def test_forward_ttm_swap_flagged(self) -> None:
        """145020.KQ quick-mode: award earned on PE_RATIO_FORWARD 14.58 while
        trailing 18.22 and PEG fail — swap signature, plain thresholds."""
        body = self._body(
            "SECTOR: Industrials",
            "PE_RATIO_TTM: 18.22",
            "PE_RATIO_FORWARD: 14.58",
            "PEG_RATIO: 1.5",
        )
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect
        assert "forward P/E basis" in updated

    def test_borderline_ttm_without_forward_tolerated(self) -> None:
        """No rescuing forward value: 18.22 stays inside the 10% margin."""
        body = self._body(
            "SECTOR: Industrials", "PE_RATIO_TTM: 18.22", "PEG_RATIO: 1.5"
        )
        _, corrected, suspect = reconcile_score_consistency(body)
        assert not suspect and not corrected

    def test_failing_forward_is_not_a_swap(self) -> None:
        """Forward also above PE_MAX cannot have rescued the point."""
        body = self._body(
            "SECTOR: Industrials",
            "PE_RATIO_TTM: 18.22",
            "PE_RATIO_FORWARD: 19.0",
            "PEG_RATIO: 1.5",
        )
        _, corrected, suspect = reconcile_score_consistency(body)
        assert not suspect and not corrected

    def test_swap_check_skipped_for_information_technology(self) -> None:
        body = self._body(
            "SECTOR: Information Technology",
            "PE_RATIO_TTM: 18.22",
            "PE_RATIO_FORWARD: 14.58",
            "PEG_RATIO: 1.5",
        )
        _, corrected, suspect = reconcile_score_consistency(body)
        assert not suspect and not corrected

    def test_passing_ttm_with_forward_present_untouched(self) -> None:
        body = self._body(
            "SECTOR: Industrials",
            "PE_RATIO_TTM: 12.0",
            "PE_RATIO_FORWARD: 10.0",
            "PEG_RATIO: 1.5",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not suspect and not corrected

    def test_passing_peg_with_forward_present_untouched(self) -> None:
        """PEG legitimately earns the point even when both P/E bases differ."""
        body = self._body(
            "SECTOR: Industrials",
            "PE_RATIO_TTM: 18.22",
            "PE_RATIO_FORWARD: 14.58",
            "PEG_RATIO: 0.9",
        )
        _, corrected, suspect = reconcile_score_consistency(body)
        assert not suspect and not corrected

    def test_positive_values_not_flagged(self) -> None:
        body = self._body(
            "SECTOR: Industrials",
            "OPERATING_CASH_FLOW: ¥2.9B",
            "FCF: $120M",
            "PE_RATIO_TTM: 12.0",
            "PEG_RATIO: 0.9",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not suspect and not corrected


class TestAgsBrNumeratorTriplet:
    """The three AGS.BR runs (4/10, 7.5/9, 5/9 on identical data) that motivated
    A5b: internally consistent totals pass only when the breakdown supports the
    numerator; a contradicting breakdown lands HEALTH_SCORE_UNRELIABLE-class
    SUSPECT instead of silently flipping the <50% gate."""

    _FIN = (
        "SECTOR: Financials",
        "SECTOR_ADJUSTMENTS: Financials: D/E Ratio and EV/EBITDA removed.",
    )

    def _breakdown(self, awards: dict[str, str]) -> str:
        base = {
            "ROE": "0",
            "ROA": "0",
            "OPERATING_MARGIN": "0",
            "DE_RATIO": "REMOVED",
            "NET_DEBT_EBITDA": "N/A",
            "CURRENT_RATIO": "0",
            "OCF_POSITIVE": "0",
            "FCF_POSITIVE": "0",
            "FCF_YIELD": "0",
            "PE_OR_PEG": "0",
            "EV_EBITDA": "REMOVED",
            "PB_OR_PS": "0",
        }
        base.update(awards)
        return "HEALTH_SCORE_BREAKDOWN: " + "; ".join(
            f"{k}={v}" for k, v in base.items()
        )  # available = 9

    def test_5_of_9_with_supporting_breakdown_passes(self) -> None:
        body = "\n".join(
            (
                *self._FIN,
                self._breakdown(
                    {
                        "ROE": "1",
                        "ROA": "1",
                        "CURRENT_RATIO": "1",
                        "OCF_POSITIVE": "1",
                        "PE_OR_PEG": "1",
                    }
                ),
                "RAW_HEALTH_SCORE: 5/9",
                "ADJUSTED_HEALTH_SCORE: 56% (based on 9 available points)",
            )
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not corrected and not suspect

    def test_5_of_9_total_with_7_5_breakdown_is_suspect(self) -> None:
        """June-29-style numerator (7.5) contradicting a July-2-style total."""
        body = "\n".join(
            (
                *self._FIN,
                self._breakdown(
                    {
                        "ROE": "1",
                        "ROA": "1",
                        "OPERATING_MARGIN": "0.5",
                        "CURRENT_RATIO": "1",
                        "OCF_POSITIVE": "1",
                        "FCF_POSITIVE": "1",
                        "PE_OR_PEG": "1",
                        "PB_OR_PS": "1",
                    }
                ),
                "RAW_HEALTH_SCORE: 5/9",
                "ADJUSTED_HEALTH_SCORE: 56% (based on 9 available points)",
            )
        )
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect
        assert "breakdown awards sum 7.5 != RAW earned 5" in updated

    def test_4_of_10_shape_denominator_still_audited(self) -> None:
        """June-28 shape: breakdown says 9 available but totals claim 10."""
        body = "\n".join(
            (
                *self._FIN,
                self._breakdown(
                    {"ROE": "1", "ROA": "1", "CURRENT_RATIO": "1", "OCF_POSITIVE": "1"}
                ),
                "RAW_HEALTH_SCORE: 4/10",
                "ADJUSTED_HEALTH_SCORE: 40% (based on 10 available points)",
            )
        )
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect
        assert "breakdown available points 9 != stated available 10" in updated


class TestScoreBreakdownHardening:
    """Fix-round regressions: strict item parsing + REMOVED semantics."""

    @staticmethod
    def _body(breakdown_items: str, raw: str, adjusted: str) -> str:
        return "\n".join(
            (
                f"HEALTH_SCORE_BREAKDOWN: {breakdown_items}",
                f"RAW_HEALTH_SCORE: {raw}",
                f"ADJUSTED_HEALTH_SCORE: {adjusted}",
            )
        )

    def test_out_of_vocabulary_numeric_award_is_suspect_not_misparsed(self) -> None:
        """ROE=1.5 must not silently parse as ROE=1 (verified regression)."""
        assert parse_score_breakdown("ROE=1.5; ROA=0.75") is None
        body = self._body(
            "ROE=1.5; ROA=0.75", "2/12", "17% (based on 12 available points)"
        )
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect
        assert "unparseable or has duplicate criteria" in updated

    def test_trailing_period_after_award_tolerated(self) -> None:
        awards = parse_score_breakdown("ROE=1; PB_OR_PS=0.")
        assert awards == {"ROE": "1", "PB_OR_PS": "0"}

    def test_labeled_full_line_parses(self) -> None:
        awards = parse_score_breakdown("HEALTH_SCORE_BREAKDOWN: ROE=1; ROA=0")
        assert awards == {"ROE": "1", "ROA": "0"}

    def test_removed_on_never_removable_criterion_is_suspect(self) -> None:
        """AGS.BR 2026-07-02 run 2 shape: OCF_POSITIVE=REMOVED claims a sector
        mandate that does not exist."""
        body = self._body(
            "ROE=1; ROA=0; OPERATING_MARGIN=0.5; DE_RATIO=REMOVED; "
            "NET_DEBT_EBITDA=REMOVED; CURRENT_RATIO=1; OCF_POSITIVE=REMOVED; "
            "FCF_POSITIVE=REMOVED; FCF_YIELD=REMOVED; PE_OR_PEG=1; "
            "EV_EBITDA=REMOVED; PB_OR_PS=0",
            "3.5/6",
            "58% (based on 6 available points)",
        )
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect
        assert (
            "REMOVED claimed for non-sector-removable criteria: OCF_POSITIVE" in updated
        )

    def test_removed_on_sector_removable_criteria_accepted(self) -> None:
        """FCF-class / D/E / EV-EBITDA REMOVED stays legal (Financials usage)."""
        body = self._body(
            "ROE=1; ROA=1; OPERATING_MARGIN=0.5; DE_RATIO=REMOVED; "
            "NET_DEBT_EBITDA=REMOVED; CURRENT_RATIO=1; OCF_POSITIVE=1; "
            "FCF_POSITIVE=REMOVED; FCF_YIELD=REMOVED; PE_OR_PEG=1; "
            "EV_EBITDA=REMOVED; PB_OR_PS=0.5",
            "6/7",
            "86% (based on 7 available points)",
        )
        updated, corrected, suspect = reconcile_score_consistency(body)
        assert updated == body
        assert not corrected and not suspect

    def test_fcf_sign_check_reads_canonical_free_cash_flow_field(self) -> None:
        body = "\n".join(
            (
                "FREE_CASH_FLOW: -€0.3B",
                "HEALTH_SCORE_BREAKDOWN: ROE=1; ROA=1; OPERATING_MARGIN=1; "
                "DE_RATIO=1; NET_DEBT_EBITDA=1; CURRENT_RATIO=1; OCF_POSITIVE=1; "
                "FCF_POSITIVE=1; FCF_YIELD=1; PE_OR_PEG=1; EV_EBITDA=1; PB_OR_PS=1",
                "RAW_HEALTH_SCORE: 12/12",
                "ADJUSTED_HEALTH_SCORE: 100% (based on 12 available points)",
            )
        )
        updated, _, suspect = reconcile_score_consistency(body)
        assert suspect
        assert "FCF_POSITIVE=1 but FREE_CASH_FLOW is negative" in updated


def test_ags_narrative_conflict_gets_authoritative_warning_without_rewrite() -> None:
    content = """### Financial Health Sheet
- **NetDebt/EBITDA:** -0.90 (Net Cash Position) -> 1 pts

### --- START DATA_BLOCK ---
NET_DEBT_EBITDA: 0.23
CASH_TO_ASSETS: 22.4%
SHAREHOLDER_RETURN_EXECUTION: ANNOUNCED_ONLY
### --- END DATA_BLOCK ---
"""

    sanitized = _sanitize_fundamentals_output(content, "", "AGS.SI")

    assert "AUTHORITATIVE_METRIC_CORRECTION" in sanitized
    assert "preceding labeled narrative=-0.90" in sanitized
    assert "authoritative DATA_BLOCK=0.23" in sanitized
    assert "NetDebt/EBITDA:** -0.90" in sanitized
    assert sanitized.index("AUTHORITATIVE_METRIC_CORRECTION") < sanitized.index(
        "START DATA_BLOCK"
    )


def test_guidance_promotion_withholds_eps_growth_for_temporary_tax_credit() -> None:
    content = """### --- START DATA_BLOCK ---
GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=1; ROA_ROE_IMPROVING=1; GROSS_MARGIN=1; GLOBAL_EXPANSION=0; R_AND_D_CAPEX_BACKLOG=0
RAW_GROWTH_SCORE: 4/6
ADJUSTED_GROWTH_SCORE: 66.7% (based on 6 available points)
### --- END DATA_BLOCK ---
"""
    foreign_data = """### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_DATE: 2026-05-08
SOURCE_URL: https://finance.logmi.jp/articles/384869
GUIDANCE_PERIOD: FY3/27
REVENUE_GUIDANCE: JPY110.0bn
OPERATING_PROFIT_GUIDANCE: JPY12.3bn
ORDINARY_OR_PRETAX_PROFIT_GUIDANCE: JPY12.5bn
NET_INCOME_GUIDANCE: JPY9.0bn
NET_INCOME_YOY: -4%
OPERATING_VS_NET_DIRECTION: OP_UP_NET_DOWN
MATERIAL_NONOPERATING_DRIVER: YES
DRIVER_TYPE: TAX_CREDIT
DRIVER_PERSISTENCE: EXPIRING
DRIVER_MATERIALITY: MATERIAL
DRIVER_AFFECTED_PERIOD: FY3/26
EARNINGS_BASELINE_STATUS: TEMPORARILY_BOOSTED
NORMALIZED_EARNINGS_AVAILABLE: NO
GUIDANCE_BRIDGE_STATUS: RECONCILED
### --- END MANAGEMENT_GUIDANCE ---
"""

    sanitized = _sanitize_fundamentals_output(
        content,
        "",
        "6745.T",
        foreign_data=foreign_data,
    )

    assert "GUIDANCE_SOURCE_URL: https://finance.logmi.jp/articles/384869" in sanitized
    assert "DRIVER_TYPE: TAX_CREDIT" in sanitized
    assert "GUIDANCE_BRIDGE_STATUS: RECONCILED" in sanitized
    assert "EPS_GROWTH=0" in sanitized
    assert "RAW_GROWTH_SCORE: 3/6" in sanitized
    assert "ADJUSTED_GROWTH_SCORE: 50.0%" in sanitized
    assert "EPS_GROWTH_BASELINE_ADJUSTMENT: WITHHELD" in sanitized

    normalized_claim = _sanitize_fundamentals_output(
        content,
        "",
        "6745.T",
        foreign_data=foreign_data.replace(
            "NORMALIZED_EARNINGS_AVAILABLE: NO",
            "NORMALIZED_EARNINGS_AVAILABLE: YES",
        ),
    )
    assert "EPS_GROWTH=0" in normalized_claim
    assert "no code-reconciled normalized growth rate is available" in normalized_claim


def test_durable_normalized_baseline_preserves_eps_growth_credit() -> None:
    content = """### --- START DATA_BLOCK ---
GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=1; ROA_ROE_IMPROVING=1; GROSS_MARGIN=1; GLOBAL_EXPANSION=0; R_AND_D_CAPEX_BACKLOG=0
RAW_GROWTH_SCORE: 4/6
ADJUSTED_GROWTH_SCORE: 66.7% (based on 6 available points)
EARNINGS_BASELINE_STATUS: DURABLE
NORMALIZED_EARNINGS_AVAILABLE: YES
GUIDANCE_BRIDGE_STATUS: NOT_APPLICABLE
### --- END DATA_BLOCK ---
"""

    sanitized = _sanitize_fundamentals_output(content, "", "CONTROL.T")

    assert "EPS_GROWTH=1" in sanitized
    assert "RAW_GROWTH_SCORE: 4/6" in sanitized
    assert "EPS_GROWTH_BASELINE_ADJUSTMENT" not in sanitized


def test_authoritative_warning_is_idempotent() -> None:
    content = """Net Debt/EBITDA: -0.90
### --- START DATA_BLOCK ---
NET_DEBT_EBITDA: 0.23
### --- END DATA_BLOCK ---
"""
    once = _sanitize_fundamentals_output(content, "", "AGS.SI")
    twice = _sanitize_fundamentals_output(once, "", "AGS.SI")

    assert twice.count("AUTHORITATIVE_METRIC_CORRECTION") == 1


def test_sanitizer_rejects_mutation_after_canonical_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = """### --- START DATA_BLOCK ---
PE_RATIO_TTM: 99
### --- END DATA_BLOCK ---
"""
    snapshot = {
        "contract_status": "VALID",
        "claims": {
            "PE_RATIO_TTM": {
                "id": "PE_RATIO_TTM",
                "field": "PE_RATIO_TTM",
                "kind": "FACT",
                "value": "10",
            }
        },
    }

    def mutate_projected_fact(body: str) -> tuple[str, bool]:
        return replace_or_append_block_line(body, "PE_RATIO_TTM", "11"), True

    monkeypatch.setattr(
        analyst_nodes,
        "withhold_eps_growth_for_unusable_baseline",
        mutate_projected_fact,
    )

    with pytest.raises(ValueError, match="POST_PROJECTION_FACT_MUTATION"):
        _sanitize_fundamentals_output(
            content,
            "",
            "TEST",
            canonical_snapshot=snapshot,
        )
