"""Tests for the OCF corroboration cross-signal (KTY.WA 2026-06-27).

The forensic auditor computes operating cash flow independently of the
Foreign-Language "filing" value the Senior may have promoted under FILING
AUTHORITY. ``detect_ocf_corroboration_flag`` raises a risk-neutral
``OCF_FILING_VALUE_UNCORROBORATED`` flag when the headline OCF diverges
materially from the auditor's figure, blocking the "elite cash generation"
overclaim without escalating the risk tally.
"""

from __future__ import annotations

from src.validators.financial_rules import (
    detect_ocf_corroboration_flag,
    extract_auditor_ocf,
    parse_ocf_amount,
)


class TestParseOcfAmount:
    def test_datablock_billion_form(self):
        assert parse_ocf_amount("1.148B PLN") == 1.148e9

    def test_auditor_tilde_million_form(self):
        assert parse_ocf_amount("Operating cash flow: ~PLN 971m") == 971e6

    def test_bare_large_number(self):
        assert parse_ocf_amount("920,000,000") == 920_000_000.0

    def test_trillion_and_thousand(self):
        assert parse_ocf_amount("2.5 trillion JPY") == 2.5e12
        assert parse_ocf_amount("800k") == 800_000.0

    def test_none_and_no_number(self):
        assert parse_ocf_amount(None) is None
        assert parse_ocf_amount("not available") is None


class TestExtractAuditorOcf:
    def test_extracts_from_labelled_line(self):
        report = (
            "## FORENSIC ASSESSMENT\n"
            "- Operating cash flow: ~PLN 971m\n"
            "  - NI_TO_OCF = 971 / 592 = 1.64\n"
        )
        assert extract_auditor_ocf(report) == 971e6

    def test_net_cash_from_operating_activities_phrasing(self):
        assert (
            extract_auditor_ocf("Net cash from operating activities: 937 million")
            == 937e6
        )

    def test_none_when_no_cashflow_line(self):
        assert extract_auditor_ocf("Revenue grew 6.8%. No cash flow detail.") is None

    def test_none_input(self):
        assert extract_auditor_ocf(None) is None


class TestDetectOcfCorroborationFlag:
    def test_high_outlier_headline_fires(self):
        # KTY case: filing 1.148B vs auditor ~971M -> ~18% divergence.
        flag = detect_ocf_corroboration_flag(1.148e9, 971e6, ticker="KTY.WA")
        assert flag is not None
        assert flag["type"] == "OCF_FILING_VALUE_UNCORROBORATED"
        assert flag["risk_penalty"] == 0.0  # risk-neutral by design
        assert flag["action"] == "DOWNWEIGHT_CASH_NARRATIVE"
        assert "elite cash generation" in flag["detail"]

    def test_within_band_no_flag(self):
        # 935 vs 971 -> ~3.8% divergence, under the 15% threshold.
        assert detect_ocf_corroboration_flag(935e6, 971e6) is None

    def test_headline_lower_no_overclaim_language(self):
        flag = detect_ocf_corroboration_flag(700e6, 971e6)
        assert flag is not None
        assert "elite cash generation" not in flag["detail"]

    def test_missing_or_zero_signals_no_flag(self):
        assert detect_ocf_corroboration_flag(None, 971e6) is None
        assert detect_ocf_corroboration_flag(1.148e9, None) is None
        assert detect_ocf_corroboration_flag(0, 971e6) is None
        assert detect_ocf_corroboration_flag(1.148e9, 0) is None

    def test_custom_threshold(self):
        # 1.148B vs 971M is ~18%; a 25% threshold suppresses it.
        assert detect_ocf_corroboration_flag(1.148e9, 971e6, threshold=0.25) is None


class TestRedFlagDetectorFacade:
    """The PM node calls these through the RedFlagDetector facade (call convention
    parity with the other detectors), while the logic stays in financial_rules."""

    def test_facade_exposes_ocf_helpers(self):
        from src.validators.red_flag_detector import RedFlagDetector

        assert RedFlagDetector.parse_ocf_amount("1.148B PLN") == 1.148e9
        assert (
            RedFlagDetector.extract_auditor_ocf("Operating cash flow: ~PLN 971m")
            == 971e6
        )
        flag = RedFlagDetector.detect_ocf_corroboration_flag(1.148e9, 971e6, "KTY.WA")
        assert flag is not None and flag["type"] == "OCF_FILING_VALUE_UNCORROBORATED"
