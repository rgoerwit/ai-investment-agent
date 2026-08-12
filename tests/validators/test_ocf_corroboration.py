"""Tests for the OCF corroboration cross-signal (KTY.WA 2026-06-27).

The forensic auditor computes operating cash flow independently of the
Foreign-Language "filing" value the Senior may have promoted under FILING
AUTHORITY. ``detect_ocf_corroboration_flag`` raises a risk-neutral
``OCF_FILING_VALUE_UNCORROBORATED`` flag when the headline OCF diverges
materially from the auditor's figure, blocking the "elite cash generation"
overclaim without escalating the risk tally.
"""

from __future__ import annotations

from datetime import date

from src.validators.financial_rules import (
    OcfObservation,
    detect_ocf_corroboration_flag,
    extract_auditor_ocf,
    extract_auditor_ocf_observation,
    extract_datablock_ocf_observation,
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
            "  - OCF_TO_NI = 971 / 592 = 1.64\n"
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

    def test_current_ttm_is_not_compared_with_prior_audited_fy(self):
        headline = OcfObservation(
            amount=219.9e6,
            period="TTM",
            period_end=date(2026, 3, 31),
            currency="SGD",
            scope="CONSOLIDATED",
            source="DATA_BLOCK",
        )
        audited = OcfObservation(
            amount=157.781e6,
            period="FY",
            period_end=date(2025, 3, 31),
            currency="SGD",
            scope="CONSOLIDATED",
            audit_status="AUDITED",
            source="FORENSIC_AUDITOR",
        )
        flag = detect_ocf_corroboration_flag(headline, audited, ticker="AGS.SI")
        assert flag is not None
        assert flag["type"] == "OCF_PERIOD_NOT_COMPARABLE"
        assert flag["risk_penalty"] == 0.0
        assert "No discrepancy inferred" in flag["detail"]

    def test_same_period_identity_can_raise_discrepancy(self):
        identity = {
            "period": "FY",
            "period_start": date(2025, 4, 1),
            "period_end": date(2026, 3, 31),
            "currency": "SGD",
            "scope": "CONSOLIDATED",
        }
        flag = detect_ocf_corroboration_flag(
            OcfObservation(amount=219.9e6, **identity),
            OcfObservation(amount=157.781e6, **identity),
        )
        assert flag is not None
        assert flag["type"] == "OCF_FILING_VALUE_UNCORROBORATED"

    def test_missing_period_end_is_not_comparable(self):
        flag = detect_ocf_corroboration_flag(
            OcfObservation(
                amount=120e6,
                period="FY",
                currency="SGD",
                scope="CONSOLIDATED",
            ),
            OcfObservation(
                amount=80e6,
                period="FY",
                currency="SGD",
                scope="CONSOLIDATED",
            ),
        )
        assert flag is not None
        assert flag["type"] == "OCF_PERIOD_NOT_COMPARABLE"
        assert "PERIOD_UNVERIFIED" in flag["detail"]

    def test_stub_period_start_mismatch_is_not_comparable(self):
        flag = detect_ocf_corroboration_flag(
            OcfObservation(
                amount=120e6,
                period="FY",
                period_start=date(2025, 1, 1),
                period_end=date(2025, 9, 30),
                currency="SGD",
                scope="CONSOLIDATED",
            ),
            OcfObservation(
                amount=80e6,
                period="FY",
                period_start=date(2024, 10, 1),
                period_end=date(2025, 9, 30),
                currency="SGD",
                scope="CONSOLIDATED",
            ),
        )
        assert flag is not None
        assert "PERIOD_START_MISMATCH" in flag["detail"]


class TestExtractOcfObservations:
    def test_extracts_datablock_period_scope_currency_and_end_date(self):
        report = """
### --- START DATA_BLOCK ---
LATEST_QUARTER_DATE: 2026-03-31
OPERATING_CASH_FLOW: S$219.9M
OPERATING_CASH_FLOW_PERIOD: TTM
METRIC_SCOPE_OCF: CONSOLIDATED
### --- END DATA_BLOCK ---
"""
        observation = extract_datablock_ocf_observation(report)
        assert observation == OcfObservation(
            amount=219.9e6,
            period="TTM",
            period_end=date(2026, 3, 31),
            currency="SGD",
            scope="CONSOLIDATED",
            source="DATA_BLOCK",
        )

    def test_extracts_audited_identity_without_calendar_year_inference(self):
        report = """
Net cash from operating activities: SGD 157.781 million
META: REPORT_DATE=2025-03-31 | PERIOD=FY | CURRENCY=SGD |
SCOPE=CONSOLIDATED | AUDIT_STATUS=AUDITED |
AUDITOR_SIGNATURE_DATE=2025-06-20
"""
        observation = extract_auditor_ocf_observation(report)
        assert observation is not None
        assert observation.period == "FY"
        assert observation.period_end == date(2025, 3, 31)
        assert observation.audit_status == "AUDITED"
        assert observation.auditor_signature_date == date(2025, 6, 20)


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


class TestSuppressionPathIsRetired:
    """The period-mismatch suppression could only ever REMOVE a risk flag, ran on
    bare floats (so no period/currency/scope guard applied), and never fired once
    across the persisted artifact history. It must not come back by accident."""

    def test_no_suppression_symbols_remain(self):
        import src.validators.financial_rules as fr
        from src.validators.red_flag_detector import RedFlagDetector

        for name in (
            "is_ocf_period_mismatch_resolved",
            "reconcile_ocf_period_mismatch_flags",
            "_consultant_resolves_ocf_period_mismatch",
        ):
            assert not hasattr(fr, name), f"{name} must stay retired"
            assert not hasattr(RedFlagDetector, name), f"{name} must stay retired"

    def test_report_stage_no_longer_reconciles_flags(self):
        """get_effective_red_flags is now a passthrough: an OCF_SOURCE_DISCREPANCY
        survives rendering rather than being swapped for a zero-penalty note."""
        from src.reporting.state_access import get_effective_red_flags

        flags = [
            {"type": "OCF_SOURCE_DISCREPANCY", "risk_penalty": 0.5},
            {"type": "LOCAL_COVERAGE_HIGH", "risk_penalty": 0.25},
        ]
        assert get_effective_red_flags({"red_flags": flags}) == flags

    def test_report_stage_tolerates_missing_state(self):
        from src.reporting.state_access import get_effective_red_flags

        assert get_effective_red_flags({}) == []
        assert get_effective_red_flags(None) == []
        assert get_effective_red_flags({"red_flags": "corrupt"}) == []
