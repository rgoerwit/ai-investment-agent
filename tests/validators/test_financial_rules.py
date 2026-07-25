"""Focused financial-rule coverage for the red-flag validator."""

from tests.validators.red_flag_validator_cases import (
    TestCyclicalPeakDetection,
    TestOCFNIRatioCheck,
    TestRealWorldEdgeCases,
    TestRealWorldSectorExamples,
    TestSectorAwareRedFlags,
    TestStrictDetectRedFlags,
    TestThinConsensusFlag,
    TestUnreliablePEG,
    TestUnreliablePEGHighGrowth,
    TestUnsustainableDistribution,
)

__all__ = [
    "TestRealWorldEdgeCases",
    "TestSectorAwareRedFlags",
    "TestRealWorldSectorExamples",
    "TestUnsustainableDistribution",
    "TestCyclicalPeakDetection",
    "TestOCFNIRatioCheck",
    "TestUnreliablePEG",
    "TestThinConsensusFlag",
    "TestStrictDetectRedFlags",
    "TestUnreliablePEGHighGrowth",
]

import pytest

from src.validators.financial_rules import (
    contains_transient_strength_marker,
    requires_normalized_earnings_bridge,
)
from src.validators.red_flag_detector import RedFlagDetector
from tests.helpers.frozen_regressions import load_frozen_regression


def _sparse_metrics(raw_report: str, *, strength: bool = True) -> dict:
    """Minimal metrics dict so only the raw-report-driven rule can fire.

    ``strength=True`` sets reported current strength (revenue growth) so the
    NORMALIZED_EARNINGS_REQUIRED catalyst-framing gate is satisfied.
    """
    keys = (
        "debt_to_equity net_income fcf interest_coverage pe_ratio pb_ratio "
        "payout_ratio dividend_coverage net_margin roic_quality profitability_trend "
        "roa_current roa_5y_avg roe_5y_avg peg_ratio ocf revenue_growth_ttm "
        "adjusted_health_score"
    ).split()
    metrics = dict.fromkeys(keys)
    metrics["_raw_report"] = raw_report
    if strength:
        metrics["revenue_growth_ttm"] = 20.0
    return metrics


class TestNormalizedEarningsBridge:
    """Distortion-before-catalyst: one-time events need a normalized bridge."""

    def test_event_without_bridge_true(self):
        assert requires_normalized_earnings_bridge(
            "EPS was lifted by a gain on sale of a division this year."
        )

    def test_deconsolidation_without_bridge_true(self):
        assert requires_normalized_earnings_bridge(
            "The desincorporation of Grupo Nutrisa lifted reported margins."
        )

    def test_tax_credit_without_bridge_true(self):
        assert requires_normalized_earnings_bridge(
            "Net income benefited from the wage-increase tax credit."
        )

    def test_tax_credit_with_bridge_false(self):
        assert not requires_normalized_earnings_bridge(
            "The tax credit lifted profit, but normalized net income excluding the tax credit was flat."
        )

    def test_event_with_bridge_false(self):
        assert not requires_normalized_earnings_bridge(
            "Gain on sale boosted EPS, but normalized EPS excluding the gain still rose."
        )

    def test_no_event_false(self):
        assert not requires_normalized_earnings_bridge("Steady organic revenue growth.")

    def test_empty_report_false(self):
        assert not requires_normalized_earnings_bridge("")
        assert not requires_normalized_earnings_bridge(None)

    @pytest.mark.parametrize(
        "report",
        (
            "Jednorazowa ulga podatkowa podwyższyła zysk netto.",
            "Insentif cukai tahun lalu meningkatkan keuntungan bersih.",
        ),
    )
    def test_localized_tax_terms_are_transient_markers(self, report):
        assert contains_transient_strength_marker(report)


class TestNormalizedEarningsRequiredFlag:
    """Narrative-only candidates remain evidence gaps, not causal diagnoses."""

    def test_evidence_gap_emitted_for_event_without_structured_support(self):
        flags, result = RedFlagDetector.detect_red_flags(
            _sparse_metrics(
                "Reported profit was flattered by a one-time gain on sale."
            ),
            "TEST.MX",
        )
        gaps = [f for f in flags if f["type"] == "EARNINGS_DRIVER_EVIDENCE_GAP"]
        assert len(gaps) == 1
        assert gaps[0]["risk_penalty"] == 0.0
        assert gaps[0]["blocks_buy"] is True
        assert not [f for f in flags if f["type"] == "TRANSIENT_STRENGTH_DISTORTION"]
        assert not [f for f in flags if f["type"] == "NORMALIZED_EARNINGS_REQUIRED"]
        assert result == "PASS"  # discipline flag, not a hard reject

    def test_gap_still_flags_narrative_even_when_narrative_claims_a_bridge(self):
        flags, _ = RedFlagDetector.detect_red_flags(
            _sparse_metrics(
                "Gain on sale recognized; underlying earnings ex-one-time still grew."
            ),
            "TEST.MX",
        )
        assert [f for f in flags if f["type"] == "EARNINGS_DRIVER_EVIDENCE_GAP"]
        assert not [f for f in flags if f["type"] == "NORMALIZED_EARNINGS_REQUIRED"]

    def test_flag_absent_without_event_or_raw_report(self):
        flags, _ = RedFlagDetector.detect_red_flags(
            _sparse_metrics("Ordinary trading update, organic growth."), "X.T"
        )
        assert not [f for f in flags if f["type"] == "EARNINGS_DRIVER_EVIDENCE_GAP"]
        # Legacy callers that pass no _raw_report must be unaffected
        flags2, _ = RedFlagDetector.detect_red_flags(_sparse_metrics(""), "X.T")
        assert not [f for f in flags2 if f["type"] == "EARNINGS_DRIVER_EVIDENCE_GAP"]

    def test_flag_absent_without_current_strength(self):
        """Distressed/no-strength names discussing one-time options do not flag (catalyst-framing gate)."""
        flags, _ = RedFlagDetector.detect_red_flags(
            _sparse_metrics(
                "Slowing market reduces asset sale options.", strength=False
            ),
            "DISTRESS.T",
        )
        assert not [f for f in flags if f["type"] == "EARNINGS_DRIVER_EVIDENCE_GAP"]

    def test_bare_m_and_a_context_is_not_an_earnings_driver(self):
        """6782 regression: acquisition context alone is not causal evidence."""
        regression = load_frozen_regression("6782_TW_regression.json")
        flags, _ = RedFlagDetector.detect_red_flags(
            _sparse_metrics(regression["m_and_a_narrative"]),
            regression["ticker"],
        )

        flag_types = {flag["type"] for flag in flags}
        assert "EARNINGS_DRIVER_EVIDENCE_GAP" not in flag_types
        assert "TRANSIENT_STRENGTH_DISTORTION" not in flag_types
        assert "NORMALIZED_EARNINGS_REQUIRED" not in flag_types


class TestCanonicalManagementGuidanceFlags:
    def test_expiring_tax_credit_flags_baseline_without_growth_gate(self):
        metrics = _sparse_metrics("", strength=False)
        metrics.update(
            {
                "guidance_coverage_status": "FOUND",
                "operating_vs_net_direction": "OP_UP_NET_DOWN",
                "material_nonoperating_driver": "YES",
                "driver_type": "TAX_CREDIT",
                "driver_persistence": "EXPIRING",
                "driver_materiality": "MATERIAL",
                "earnings_baseline_status": "TEMPORARILY_BOOSTED",
                "normalized_earnings_available": "NO",
            }
        )

        flags, result = RedFlagDetector.detect_red_flags(metrics, "6745.T")
        flag_types = {flag["type"] for flag in flags}

        assert "TRANSIENT_STRENGTH_DISTORTION" in flag_types
        assert "NORMALIZED_EARNINGS_REQUIRED" in flag_types
        assert "OPERATING_NET_GUIDANCE_DIVERGENCE" in flag_types
        assert result == "PASS"

    def test_search_failure_is_not_treated_as_durable_baseline(self):
        metrics = _sparse_metrics("", strength=False)
        metrics["guidance_coverage_status"] = "SEARCH_FAILED"

        flags, _ = RedFlagDetector.detect_red_flags(metrics, "UNKNOWN.T")

        assert "MANAGEMENT_GUIDANCE_EVIDENCE_GAP" in {flag["type"] for flag in flags}
        gap = next(
            flag for flag in flags if flag["type"] == "MANAGEMENT_GUIDANCE_EVIDENCE_GAP"
        )
        assert gap["blocks_buy"] is True

    def test_unresolved_targeted_search_is_not_treated_as_durable_baseline(self):
        metrics = _sparse_metrics("", strength=False)
        metrics["guidance_coverage_status"] = "UNRESOLVED_AFTER_TARGETED_SEARCH"

        flags, _ = RedFlagDetector.detect_red_flags(metrics, "KTY.WA")

        assert "MANAGEMENT_GUIDANCE_EVIDENCE_GAP" in {flag["type"] for flag in flags}
