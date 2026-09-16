"""Typed liquidity assessment and graph fast-fail contract tests."""

import math
from types import SimpleNamespace

from src.liquidity_assessment import (
    LiquidityAssessment,
    latest_liquidity_assessment,
    liquidity_fast_fail_update,
)


def test_liquidity_block_round_trip_preserves_hard_fail() -> None:
    original = LiquidityAssessment(
        status="FAIL_INSUFFICIENT_LIQUIDITY",
        average_daily_turnover_usd=86_436.0,
        average_daily_volume=12_345.0,
        zero_volume_days_pct=0.0,
        flat_price_days_pct=3.2,
        reason="below minimum",
    )

    decoded = LiquidityAssessment.from_tool_output(original.render_block())

    assert decoded == original
    assert decoded is not None and decoded.hard_fail is True


def test_latest_assessment_ignores_other_tools() -> None:
    expected = LiquidityAssessment(
        status="MARGINAL", average_daily_turnover_usd=150_000
    )
    records = [
        SimpleNamespace(tool_name="other", content="not liquidity"),
        SimpleNamespace(
            tool_name="calculate_liquidity_metrics",
            content=expected.render_block(),
        ),
    ]

    assert latest_liquidity_assessment(records) == expected


def test_liquidity_hard_fail_uses_existing_reject_contract() -> None:
    assessment = LiquidityAssessment(
        status="FAIL_INSUFFICIENT_LIQUIDITY",
        average_daily_turnover_usd=86_436,
    )

    update = liquidity_fast_fail_update(assessment.to_dict())

    assert update["pre_screening_result"] == "REJECT"
    assert update["red_flags"] == [
        {
            "type": "LIQUIDITY_HARD_FAIL",
            "severity": "CRITICAL",
            "action": "AUTO_REJECT",
            "blocks_buy": True,
            "risk_penalty": 0.0,
            "detail": (
                "Measured average daily turnover $86,436; "
                "status=FAIL_INSUFFICIENT_LIQUIDITY; minimum=$100,000."
            ),
        }
    ]


def test_liquidity_uncertainty_and_marginal_status_do_not_fast_fail() -> None:
    assessments = (
        LiquidityAssessment(status="ERROR"),
        LiquidityAssessment(status="INSUFFICIENT_DATA"),
        LiquidityAssessment(status="MARGINAL", average_daily_turnover_usd=100_000),
        LiquidityAssessment(status="PASS", average_daily_turnover_usd=250_000),
    )
    for assessment in assessments:
        assert liquidity_fast_fail_update(assessment.to_dict()) == {}


def test_typed_decoder_rejects_bad_schema_status_and_nonfinite_numbers() -> None:
    assert (
        LiquidityAssessment.from_dict({"schema_version": 2, "status": "PASS"}) is None
    )
    assert (
        LiquidityAssessment.from_dict({"schema_version": 1, "status": "MODEL_SAYS_BUY"})
        is None
    )
    decoded = LiquidityAssessment.from_dict(
        {
            "schema_version": 1,
            "status": "PASS",
            "average_daily_turnover_usd": 250_000,
            "average_daily_volume": True,
            "zero_volume_days_pct": math.inf,
        }
    )
    assert decoded is not None
    assert decoded.average_daily_turnover_usd == 250_000
    assert decoded.average_daily_volume is None
    assert decoded.zero_volume_days_pct is None


def test_serialized_hard_fail_boolean_cannot_override_status() -> None:
    decoded = LiquidityAssessment.from_dict(
        {
            "schema_version": 1,
            "status": "PASS",
            "average_daily_turnover_usd": 250_000,
            "hard_fail": True,
        }
    )

    assert decoded is not None
    assert decoded.hard_fail is False


def test_decoder_rejects_status_measurement_contradictions() -> None:
    inconsistent = (
        LiquidityAssessment(status="PASS", average_daily_turnover_usd=249_999),
        LiquidityAssessment(status="MARGINAL", average_daily_turnover_usd=99_999),
        LiquidityAssessment(
            status="FAIL_INSUFFICIENT_LIQUIDITY",
            average_daily_turnover_usd=100_000,
        ),
        LiquidityAssessment(
            status="FAIL_IRREGULAR_TRADING",
            average_daily_turnover_usd=300_000,
            zero_volume_days_pct=15.0,
            flat_price_days_pct=30.0,
        ),
    )

    for assessment in inconsistent:
        assert LiquidityAssessment.from_dict(assessment.to_dict()) is None
        assert LiquidityAssessment.from_tool_output(assessment.render_block()) is None


def test_decoder_rejects_negative_or_out_of_range_measurements() -> None:
    malformed = LiquidityAssessment(
        status="PASS",
        average_daily_turnover_usd=250_000,
        average_daily_volume=-1,
        zero_volume_days_pct=101,
        flat_price_days_pct=-0.1,
    )

    decoded = LiquidityAssessment.from_dict(malformed.to_dict())
    parsed_block = LiquidityAssessment.from_tool_output(malformed.render_block())

    assert decoded is not None
    assert decoded.average_daily_volume is None
    assert decoded.zero_volume_days_pct is None
    assert decoded.flat_price_days_pct is None
    assert parsed_block == decoded
