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

from src.validators.financial_rules import requires_normalized_earnings_bridge
from src.validators.red_flag_detector import RedFlagDetector


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

    def test_event_with_bridge_false(self):
        assert not requires_normalized_earnings_bridge(
            "Gain on sale boosted EPS, but normalized EPS excluding the gain still rose."
        )

    def test_no_event_false(self):
        assert not requires_normalized_earnings_bridge("Steady organic revenue growth.")

    def test_empty_report_false(self):
        assert not requires_normalized_earnings_bridge("")
        assert not requires_normalized_earnings_bridge(None)


class TestNormalizedEarningsRequiredFlag:
    """detect_red_flags emits NORMALIZED_EARNINGS_REQUIRED (0.0 tally) for unreconciled one-timers."""

    def test_flag_emitted_for_event_without_bridge(self):
        flags, result = RedFlagDetector.detect_red_flags(
            _sparse_metrics(
                "Reported profit was flattered by a one-time gain on sale."
            ),
            "TEST.MX",
        )
        norm = [f for f in flags if f["type"] == "NORMALIZED_EARNINGS_REQUIRED"]
        assert len(norm) == 1
        assert (
            norm[0]["risk_penalty"] == 0.0
        )  # no double-count with transient distortion
        assert result == "PASS"  # discipline flag, not a hard reject

    def test_flag_absent_when_bridge_present(self):
        flags, _ = RedFlagDetector.detect_red_flags(
            _sparse_metrics(
                "Disposal gain recognized; underlying earnings ex-one-time still grew."
            ),
            "TEST.MX",
        )
        assert not [f for f in flags if f["type"] == "NORMALIZED_EARNINGS_REQUIRED"]

    def test_flag_absent_without_event_or_raw_report(self):
        flags, _ = RedFlagDetector.detect_red_flags(
            _sparse_metrics("Ordinary trading update, organic growth."), "X.T"
        )
        assert not [f for f in flags if f["type"] == "NORMALIZED_EARNINGS_REQUIRED"]
        # Legacy callers that pass no _raw_report must be unaffected
        flags2, _ = RedFlagDetector.detect_red_flags(_sparse_metrics(""), "X.T")
        assert not [f for f in flags2 if f["type"] == "NORMALIZED_EARNINGS_REQUIRED"]

    def test_flag_absent_without_current_strength(self):
        """Distressed/no-strength names discussing one-time options do not flag (catalyst-framing gate)."""
        flags, _ = RedFlagDetector.detect_red_flags(
            _sparse_metrics(
                "Slowing market reduces asset sale options.", strength=False
            ),
            "DISTRESS.T",
        )
        assert not [f for f in flags if f["type"] == "NORMALIZED_EARNINGS_REQUIRED"]
