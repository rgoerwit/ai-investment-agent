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


class TestDecisionCriticalGrowthEvidence:
    @staticmethod
    def _metrics(
        *,
        evidence: str = "SECONDARY",
        authority: str = "THIRD_PARTY",
        projected_growth: str = "14%",
    ) -> dict:
        report = f"""### --- START DATA_BLOCK ---
GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=0; ROA_ROE_IMPROVING=0; GROSS_MARGIN=1; GLOBAL_EXPANSION=1; R_AND_D_CAPEX_BACKLOG=1
RAW_GROWTH_SCORE: 4/6
ADJUSTED_GROWTH_SCORE: 66.7% (based on 6 available points)
R_AND_D_CAPEX_BACKLOG_EVIDENCE: {evidence}
GUIDANCE_SOURCE_TYPE: RESEARCH_REPORT
GUIDANCE_SOURCE_AUTHORITY: {authority}
GUIDANCE_MANAGEMENT_IDENTIFIED: NO
GUIDANCE_NET_INCOME: EPS projection, {projected_growth} YoY
GUIDANCE_NET_INCOME_YOY: {projected_growth}
### --- END DATA_BLOCK ---
"""
        return RedFlagDetector.extract_metrics(report)

    def test_secondary_point_blocks_load_bearing_zone_2_buy(self):
        metrics = self._metrics()

        assert metrics["growth_score_earned"] == 4
        assert metrics["growth_score_available"] == 6
        assert metrics["r_and_d_capex_backlog_score"] == 1
        assert metrics["guidance_source_authority"] == "THIRD_PARTY"

        flags, result = RedFlagDetector.detect_red_flags(metrics, "6782.TW")
        gap = next(
            flag
            for flag in flags
            if flag["type"] == "DECISION_CRITICAL_GROWTH_EVIDENCE_GAP"
        )
        assert gap["blocks_buy"] is True
        assert gap["risk_penalty"] == 0.0
        assert result == "PASS"

    def test_primary_projected_growth_above_15_is_valid_alternative(self):
        flags, _ = RedFlagDetector.detect_red_flags(
            self._metrics(authority="PRIMARY", projected_growth="16%"),
            "TEST.TW",
        )

        assert "DECISION_CRITICAL_GROWTH_EVIDENCE_GAP" not in {
            flag["type"] for flag in flags
        }

    def test_third_party_projection_does_not_satisfy_alternative(self):
        flags, _ = RedFlagDetector.detect_red_flags(
            self._metrics(authority="THIRD_PARTY", projected_growth="20%"),
            "TEST.TW",
        )

        assert "DECISION_CRITICAL_GROWTH_EVIDENCE_GAP" in {
            flag["type"] for flag in flags
        }

    def test_secondary_point_does_not_block_when_growth_still_passes_without_it(self):
        report = """### --- START DATA_BLOCK ---
GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=N/A; ROA_ROE_IMPROVING=1; GROSS_MARGIN=1; GLOBAL_EXPANSION=0.5; R_AND_D_CAPEX_BACKLOG=0.5
RAW_GROWTH_SCORE: 4/6
ADJUSTED_GROWTH_SCORE: 80% (based on 5 available points)
R_AND_D_CAPEX_BACKLOG_EVIDENCE: SECONDARY
### --- END DATA_BLOCK ---
"""
        metrics = RedFlagDetector.extract_metrics(report)
        flags, _ = RedFlagDetector.detect_red_flags(metrics, "TEST.TW")

        assert metrics["growth_score_available"] == 5
        assert "DECISION_CRITICAL_GROWTH_EVIDENCE_GAP" not in {
            flag["type"] for flag in flags
        }


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
    def test_unresolved_bridge_without_canonical_driver_is_evidence_gap(self):
        metrics = _sparse_metrics("", strength=False)
        metrics.update(
            {
                "guidance_coverage_status": "FOUND",
                "guidance_bridge_status": "UNRESOLVED",
                "material_nonoperating_driver": "UNKNOWN",
                "driver_type": "UNKNOWN",
                "normalized_earnings_available": "NO",
            }
        )

        flags, _ = RedFlagDetector.detect_red_flags(metrics, "6782.TW")
        flag_types = {flag["type"] for flag in flags}

        assert "EARNINGS_DRIVER_EVIDENCE_GAP" in flag_types
        assert "NORMALIZED_EARNINGS_REQUIRED" not in flag_types
        gap = next(
            flag for flag in flags if flag["type"] == "EARNINGS_DRIVER_EVIDENCE_GAP"
        )
        assert "does not establish" in gap["detail"]
        assert gap["risk_penalty"] == 0.0
        assert gap["blocks_buy"] is True

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
