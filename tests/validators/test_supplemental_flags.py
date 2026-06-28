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
