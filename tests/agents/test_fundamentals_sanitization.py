"""Regression tests for fundamentals DATA_BLOCK sanitization."""

from __future__ import annotations

import json

from src.agents.analyst_nodes import (
    _sanitize_fundamentals_output,
    _valuation_input_reliability,
)


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
