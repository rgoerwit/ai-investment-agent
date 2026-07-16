"""Focused supplemental-flag coverage for consultant-related validator logic."""

from tests.validators.red_flag_validator_cases import (
    TestConsultantConditionEnforcement,
    TestConsultantVerdictVariants,
)

__all__ = ["TestConsultantConditionEnforcement", "TestConsultantVerdictVariants"]

import src.validators.supplemental_flags as supplemental_flags
from src.validators.red_flag_detector import RedFlagDetector
from src.validators.supplemental_extractors import (
    extract_material_unverified_operating_signal,
)


class TestMaterialUnverifiedOperatingSignal:
    """A large unverified operating decline must block BUY (0.0 tally), not auto-reject."""

    def test_extractor_detects_operating_collapse(self):
        sig = extract_material_unverified_operating_signal(
            "Q1 operating profit collapse (-53.2% YoY) reported by local press."
        )
        assert sig is not None
        assert sig["decline_pct"] == 53.2

    def test_extractor_ignores_share_price_move(self):
        assert (
            extract_material_unverified_operating_signal(
                "The share price fell 53% this year"
            )
            is None
        )

    def test_extractor_ignores_sub_threshold(self):
        assert (
            extract_material_unverified_operating_signal("operating profit fell 12%")
            is None
        )

    def test_extractor_ignores_positive_growth(self):
        assert (
            extract_material_unverified_operating_signal("operating profit up 53% YoY")
            is None
        )

    def test_extractor_handles_empty(self):
        assert extract_material_unverified_operating_signal("") is None
        assert extract_material_unverified_operating_signal(None) is None

    def test_extractor_terse_minus_form(self):
        sig = extract_material_unverified_operating_signal(
            "operating profit -53.2% YoY"
        )
        assert sig is not None and sig["decline_pct"] == 53.2

    def test_extractor_abbreviation_form(self):
        sig = extract_material_unverified_operating_signal("OP -47% YoY in Q1")
        assert sig is not None and sig["decline_pct"] == 47.0

    def test_extractor_japanese_negative_marker(self):
        sig = extract_material_unverified_operating_signal("営業利益 ▲53.2%")
        assert sig is not None and sig["decline_pct"] == 53.2

    def test_extractor_japanese_gen_suffix(self):
        sig = extract_material_unverified_operating_signal("営業利益 53.2%減")
        assert sig is not None and sig["decline_pct"] == 53.2

    def test_extractor_terse_positive_not_matched(self):
        assert (
            extract_material_unverified_operating_signal("revenue +20%, OP +33%")
            is None
        )

    def test_extractor_terse_below_threshold_not_matched(self):
        assert (
            extract_material_unverified_operating_signal("operating profit -12% YoY")
            is None
        )

    def test_extractor_op_abbreviation_with_verb(self):
        sig = extract_material_unverified_operating_signal("OP collapsed 53% YoY")
        assert sig is not None and sig["decline_pct"] == 53.0

    def test_extractor_ordinary_profit_decreased(self):
        sig = extract_material_unverified_operating_signal(
            "ordinary profit decreased 40%"
        )
        assert sig is not None and sig["decline_pct"] == 40.0

    def test_extractor_operating_margin_not_matched(self):
        # "operating margin" is not an operating *profit* metric — must not match.
        assert (
            extract_material_unverified_operating_signal(
                "operating margin decreased 40%"
            )
            is None
        )

    def test_flag_blocks_buy_with_zero_tally(self):
        flags = RedFlagDetector.detect_material_operating_signal_flags(
            "Foreign-language filing note: operating income declined 47% in the quarter.",
            "2173.T",
        )
        assert len(flags) == 1
        flag = flags[0]
        assert flag["type"] == "MATERIAL_UNVERIFIED_OPERATING_SIGNAL"
        assert flag["blocks_buy"] is True
        assert flag["risk_penalty"] == 0.0  # unverified is not confirmed risk

    def test_flag_absent_when_no_signal(self):
        assert (
            RedFlagDetector.detect_material_operating_signal_flags(
                "Solid quarter, operating profit up 8%.", "X.T"
            )
            == []
        )


def test_capital_efficiency_skips_base_metric_parse_without_signals(monkeypatch):
    def fake_extract_capital_efficiency_signals(_report: str) -> dict:
        return {}

    def fail_extract_metrics(_report: str) -> dict:
        raise AssertionError("extract_metrics should not run without capital signals")

    monkeypatch.setattr(
        supplemental_flags,
        "extract_capital_efficiency_signals",
        fake_extract_capital_efficiency_signals,
    )
    monkeypatch.setattr(supplemental_flags, "extract_metrics", fail_extract_metrics)

    assert (
        supplemental_flags.detect_capital_efficiency_flags("no structured block") == []
    )


# Verbatim FINAL VERDICT section from results/3393.T_20260704_160529_analysis.json
# (trailing markdown spaces dropped) — the run that charged a false +2.0
# CONSULTANT_MANDATE_BREACH off "No mandate breach triggered".
_3393T_FINAL_VERDICT_SECTION = (
    "### FINAL CONSULTANT VERDICT\n\n"
    "**Overall Assessment**: CONDITIONAL APPROVAL\n\n"
    "**Recommended Action for Portfolio Manager**:\n"
    "- **Proceed only after reconciling the FY2028 management-plan targets "
    "and clarifying FCF definition/source.**\n"
    "- No mandate breach triggered: **PFIC_RISK=MEDIUM**, **CMIC clear**, "
    "health well above Tier-3 warning level.\n\n"
    "**Confidence in Internal Analysis**: Medium\n"
)


class TestConsultantBreachNegationAwareParsing:
    """Negated breach/hard-stop mentions must not raise flags (3393.T 2026-07-04).

    Precedence: structured MANDATE_BREACH:/HARD_STOP: tokens (consultant prompt
    v2.12+) > negation-aware prose scan, scoped to the FINAL CONSULTANT VERDICT
    section when one exists.
    """

    def test_affirmative_breach_without_verdict_section(self):
        conditions = RedFlagDetector.parse_consultant_conditions(
            "MANDATE BREACH: PFIC threshold exceeded"
        )
        assert conditions["has_mandate_breach"] is True

    def test_affirmative_breach_inside_verdict_section(self):
        review = (
            "Body prose with no markers.\n\n### FINAL CONSULTANT VERDICT\n\n"
            "MANDATE BREACH: PFIC threshold exceeded\n"
        )
        conditions = RedFlagDetector.parse_consultant_conditions(review)
        assert conditions["has_mandate_breach"] is True

    def test_hard_stop_dash_form_fires(self):
        conditions = RedFlagDetector.parse_consultant_conditions(
            "HARD STOP — sanctions exposure"
        )
        assert conditions["has_hard_stop"] is True

    def test_real_3393t_verdict_section_no_false_breach(self):
        body = (
            "### CONSULTANT REVIEW: CONDITIONAL APPROVAL\n\n"
            "FCF margin is compressing quarterly, though not yet a mandate "
            "breach.\n\n"
        )
        conditions = RedFlagDetector.parse_consultant_conditions(
            body + _3393T_FINAL_VERDICT_SECTION
        )
        assert conditions["has_mandate_breach"] is False
        assert conditions["has_hard_stop"] is False
        assert conditions["verdict"] == "CONDITIONAL_APPROVAL"

    def test_negated_body_mention_only(self):
        conditions = RedFlagDetector.parse_consultant_conditions(
            "FCF is compressing, though not yet a mandate breach."
        )
        assert conditions["has_mandate_breach"] is False

    def test_negated_without_hard_stop(self):
        conditions = RedFlagDetector.parse_consultant_conditions(
            "The review completed without a hard stop."
        )
        assert conditions["has_hard_stop"] is False

    def test_negated_body_plus_affirmative_verdict_section(self):
        review = (
            "No mandate breach was found during factual verification.\n\n"
            "### FINAL CONSULTANT VERDICT\n\n"
            "MANDATE BREACH: PFIC threshold exceeded after re-check.\n"
        )
        conditions = RedFlagDetector.parse_consultant_conditions(review)
        assert conditions["has_mandate_breach"] is True

    def test_body_affirmative_ignored_when_verdict_section_present(self):
        # Body prose discusses conditions hypothetically; the verdict section
        # is the consultant's own summary judgment and wins.
        review = (
            "A MANDATE BREACH would be triggered if PFIC income passes 75%.\n\n"
            "### FINAL CONSULTANT VERDICT\n\n"
            "**Overall Assessment**: APPROVED\n"
        )
        conditions = RedFlagDetector.parse_consultant_conditions(review)
        assert conditions["has_mandate_breach"] is False

    def test_token_none_clears_despite_prose_mention(self):
        review = (
            "### FINAL CONSULTANT VERDICT\n\n"
            "The PFIC exposure edges toward a mandate breach in spirit.\n"
            "**MANDATE_BREACH**: NONE\n"
            "**HARD_STOP**: NONE\n"
        )
        conditions = RedFlagDetector.parse_consultant_conditions(review)
        assert conditions["has_mandate_breach"] is False
        assert conditions["has_hard_stop"] is False

    def test_token_with_description_fires(self):
        review = (
            "### FINAL CONSULTANT VERDICT\n\n"
            "**MANDATE_BREACH**: PFIC threshold exceeded (75% passive income)\n"
            "**HARD_STOP**: NONE\n"
        )
        conditions = RedFlagDetector.parse_consultant_conditions(review)
        assert conditions["has_mandate_breach"] is True
        assert conditions["has_hard_stop"] is False

    def test_token_wins_only_for_the_flag_it_names(self):
        review = (
            "### FINAL CONSULTANT VERDICT\n\n"
            "**MANDATE_BREACH**: NONE\n"
            "HARD STOP: NS-CMIC listed entity.\n"
        )
        conditions = RedFlagDetector.parse_consultant_conditions(review)
        assert conditions["has_mandate_breach"] is False
        assert conditions["has_hard_stop"] is True

    def test_token_negated_restatement_clears(self):
        review = (
            "### FINAL CONSULTANT VERDICT\n\n"
            "**MANDATE_BREACH**: No breach detected\n"
            "**HARD_STOP**: Not triggered\n"
        )
        conditions = RedFlagDetector.parse_consultant_conditions(review)
        assert conditions["has_mandate_breach"] is False
        assert conditions["has_hard_stop"] is False

    def test_token_n_a_and_parenthetical_none_clear(self):
        review = (
            "### FINAL CONSULTANT VERDICT\n\n"
            "- **MANDATE_BREACH**: NONE (PFIC below threshold)\n"
            "- **HARD_STOP**: N/A\n"
        )
        conditions = RedFlagDetector.parse_consultant_conditions(review)
        assert conditions["has_mandate_breach"] is False
        assert conditions["has_hard_stop"] is False

    def test_malformed_empty_token_falls_back_without_crash(self):
        # Empty token value → prose-scan fallback; the bare marker itself is
        # unnegated, so the flag fires conservatively.
        review = "### FINAL CONSULTANT VERDICT\n\nMANDATE_BREACH:\n"
        conditions = RedFlagDetector.parse_consultant_conditions(review)
        assert conditions["has_mandate_breach"] is True

    def test_empty_review_defaults(self):
        conditions = RedFlagDetector.parse_consultant_conditions("")
        assert conditions["has_mandate_breach"] is False
        assert conditions["has_hard_stop"] is False
        assert conditions["verdict"] == "UNKNOWN"


class TestConsultantVerdictSectionScoping:
    """The verdict is read from the FINAL CONSULTANT VERDICT section, not body prose.

    A stray 'major concern' phrase in the discussion body must not override a
    section-level CONDITIONAL APPROVAL (3771.T 2026-07-12: the whole-review
    first-match scan raised a false CONSULTANT_MAJOR_CONCERNS).
    """

    def test_body_major_concern_does_not_override_section_conditional(self):
        review = (
            "### CONSULTANT REVIEW: CONDITIONAL APPROVAL\n\n"
            "The normalized-earnings bridge is a major concern that the PM must "
            "resolve before sizing.\n\n"
            "### FINAL CONSULTANT VERDICT\n\n"
            "**Overall Assessment**: CONDITIONAL APPROVAL\n"
            "**MANDATE_BREACH**: NONE\n"
            "**HARD_STOP**: NONE\n"
        )
        conditions = RedFlagDetector.parse_consultant_conditions(review)
        assert conditions["verdict"] == "CONDITIONAL_APPROVAL"

    def test_major_concerns_inside_section_still_detected(self):
        review = (
            "Body prose without a verdict phrase.\n\n"
            "### FINAL CONSULTANT VERDICT\n\n"
            "**Overall Assessment**: MAJOR CONCERNS\n"
        )
        conditions = RedFlagDetector.parse_consultant_conditions(review)
        assert conditions["verdict"] == "MAJOR_CONCERNS"

    def test_no_section_falls_back_to_whole_review(self):
        # No FINAL CONSULTANT VERDICT section → whole-review scan (unchanged behavior).
        review = "Overall this warrants MAJOR CONCERNS given the leverage."
        conditions = RedFlagDetector.parse_consultant_conditions(review)
        assert conditions["verdict"] == "MAJOR_CONCERNS"


def _rqf_block(**fields: str) -> str:
    body = "\n".join(f"{k}: {v}" for k, v in fields.items())
    return f"### --- START DATA_BLOCK ---\n{body}\n### --- END DATA_BLOCK ---"


class TestReturnQualityFragility:
    """Deterministic relocation of PM rubric item #11 (RETURN_QUALITY_FRAGILITY)."""

    def test_unstable_trend_fires(self):
        block = _rqf_block(
            PROFITABILITY_TREND="UNSTABLE", ROA_PERCENT="6.0%", ROA_5Y_AVG="6.0%"
        )
        flags = supplemental_flags.detect_return_quality_fragility_flags(block, "X")
        assert len(flags) == 1
        assert flags[0]["type"] == "RETURN_QUALITY_FRAGILITY"
        assert flags[0]["risk_penalty"] == 0.5

    def test_apr_declining_below_5y_avg_does_not_fire(self):
        # The exact APR.WA misfire: DECLINING (not UNSTABLE) + ROA_5Y_AVG 11.1% (not <5%).
        block = _rqf_block(
            PROFITABILITY_TREND="DECLINING", ROA_PERCENT="8.77%", ROA_5Y_AVG="11.12%"
        )
        assert (
            supplemental_flags.detect_return_quality_fragility_flags(block, "APR") == []
        )

    def test_unproven_turnaround_fires(self):
        # current strong but weak 5Y base, with a non-peak trend so blocker stays quiet.
        block = _rqf_block(
            PROFITABILITY_TREND="IMPROVING", ROA_PERCENT="8.0%", ROA_5Y_AVG="4.0%"
        )
        flags = supplemental_flags.detect_return_quality_fragility_flags(block, "X")
        assert len(flags) == 1
        assert flags[0]["risk_penalty"] == 0.5

    def test_peak_distortion_suppresses_double_count(self):
        # UNSTABLE + ROA 12 vs 6 avg (ratio 2.0) -> CYCLICAL_PEAK semantics already
        # cover it; do not double-count.
        block = _rqf_block(
            PROFITABILITY_TREND="UNSTABLE", ROA_PERCENT="12.0%", ROA_5Y_AVG="6.0%"
        )
        assert (
            supplemental_flags.detect_return_quality_fragility_flags(block, "X") == []
        )

    def test_cycle_position_peak_suppresses(self):
        block = _rqf_block(
            PROFITABILITY_TREND="UNSTABLE",
            ROA_PERCENT="6.0%",
            ROA_5Y_AVG="6.0%",
            CYCLE_POSITION="PEAK",
        )
        assert (
            supplemental_flags.detect_return_quality_fragility_flags(block, "X") == []
        )

    def test_missing_fields_no_flag_no_crash(self):
        block = _rqf_block(PROFITABILITY_TREND="STABLE")
        assert (
            supplemental_flags.detect_return_quality_fragility_flags(block, "X") == []
        )
        assert supplemental_flags.detect_return_quality_fragility_flags("", "X") == []

    def test_exposed_on_facade(self):
        block = _rqf_block(
            PROFITABILITY_TREND="UNSTABLE", ROA_PERCENT="6.0%", ROA_5Y_AVG="6.0%"
        )
        flags = RedFlagDetector.detect_return_quality_fragility_flags(block, "X")
        assert flags and flags[0]["type"] == "RETURN_QUALITY_FRAGILITY"
