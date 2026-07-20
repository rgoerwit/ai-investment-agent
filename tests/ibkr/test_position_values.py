"""Unit and anomaly matrix for broker position-value normalization."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from src.ibkr.models import NormalizedPosition
from src.ibkr.position_values import normalize_position_values
from src.ibkr.ticker import Ticker


def _normalize(**overrides):
    values = {
        "quantity": 100.0,
        "current_price_local": 90.0,
        "avg_cost_local": 100.0,
        "raw_market_value": 9_000.0,
        "raw_unrealized_pnl": -1_000.0,
        "currency": "JPY",
    }
    values.update(overrides)
    return normalize_position_values(**values)


@pytest.mark.parametrize(
    ("currency", "rate"),
    [
        ("JPY", 0.0067),
        ("KRW", 0.00075),
        ("TWD", 0.032),
        ("HKD", 0.128),
        ("EUR", 1.09),
        ("GBP", 1.27),
        ("CHF", 1.13),
    ],
)
def test_local_values_are_converted_exactly_once(currency, rate):
    result = _normalize(currency=currency)

    assert result.valuation_valid is True
    assert result.market_value_basis == "LOCAL_CONVERTED"
    assert result.unrealized_pnl_basis == "LOCAL_CONVERTED"
    assert result.market_value_usd == pytest.approx(9_000.0 * rate)
    assert result.unrealized_pnl_usd == pytest.approx(-1_000.0 * rate)


@pytest.mark.parametrize(
    ("currency", "rate"),
    [
        ("JPY", 0.0067),
        ("KRW", 0.00075),
        ("TWD", 0.032),
        ("HKD", 0.128),
        ("EUR", 1.09),
        ("GBP", 1.27),
    ],
)
def test_values_already_in_usd_are_not_converted_again(currency, rate):
    market_usd = 9_000.0 * rate
    pnl_usd = -1_000.0 * rate

    result = _normalize(
        currency=currency,
        raw_market_value=market_usd,
        raw_unrealized_pnl=pnl_usd,
    )

    assert result.valuation_valid is True
    assert result.market_value_basis == "BROKER_USD"
    assert result.unrealized_pnl_basis == "BROKER_USD"
    assert result.market_value_usd == pytest.approx(market_usd)
    assert result.unrealized_pnl_usd == pytest.approx(pnl_usd)


def test_market_and_pnl_units_are_classified_independently():
    local_market_usd_pnl = _normalize(raw_unrealized_pnl=-6.7)
    usd_market_local_pnl = _normalize(
        raw_market_value=60.3,
        raw_unrealized_pnl=-1_000.0,
    )

    assert local_market_usd_pnl.market_value_basis == "LOCAL_CONVERTED"
    assert local_market_usd_pnl.unrealized_pnl_basis == "BROKER_USD"
    assert usd_market_local_pnl.market_value_basis == "BROKER_USD"
    assert usd_market_local_pnl.unrealized_pnl_basis == "LOCAL_CONVERTED"


def test_usd_values_have_unambiguous_identity_conversion_without_price_context():
    result = _normalize(
        currency="USD",
        quantity=0.0,
        current_price_local=0.0,
        avg_cost_local=0.0,
        raw_market_value=500.0,
        raw_unrealized_pnl=25.0,
    )

    assert result.valuation_valid is True
    assert result.fx_rate_to_usd == 1.0
    assert result.market_value_basis == "BROKER_USD"
    assert result.market_value_usd == 500.0
    assert result.unrealized_pnl_usd == 25.0


@pytest.mark.parametrize(
    ("quantity", "current_price"),
    [(0.0, 90.0), (100.0, 0.0), (0.0, 0.0)],
)
def test_non_usd_market_value_without_unit_context_fails_closed(
    quantity,
    current_price,
):
    result = _normalize(quantity=quantity, current_price_local=current_price)

    assert result.valuation_valid is False
    assert result.market_value_usd == 0.0
    assert "market value" in (result.valuation_issue or "")


def test_missing_cost_basis_keeps_market_value_but_withholds_pnl():
    result = _normalize(avg_cost_local=0.0)

    assert result.valuation_valid is True
    assert result.market_value_usd == pytest.approx(60.3)
    assert result.unrealized_pnl_usd == 0.0
    assert result.unrealized_pnl_basis == "UNAVAILABLE"


def test_missing_pnl_is_not_fabricated():
    result = _normalize(raw_unrealized_pnl=None)

    assert result.valuation_valid is True
    assert result.unrealized_pnl_usd == 0.0
    assert result.unrealized_pnl_basis == "UNAVAILABLE"


def test_small_breakeven_pnl_noise_is_tolerated():
    result = _normalize(
        current_price_local=100.0,
        raw_market_value=10_000.0,
        raw_unrealized_pnl=20.0,
    )

    assert result.valuation_valid is True
    assert result.unrealized_pnl_basis == "LOCAL_CONVERTED"
    assert result.unrealized_pnl_usd == pytest.approx(0.134)


def test_large_pnl_at_breakeven_is_quarantined():
    result = _normalize(
        current_price_local=100.0,
        raw_market_value=10_000.0,
        raw_unrealized_pnl=1_000.0,
    )

    assert result.valuation_valid is False
    assert result.unrealized_pnl_usd == 0.0
    assert "unrealized P&L" in (result.valuation_issue or "")


def test_market_value_tolerance_accepts_feed_noise_but_rejects_unit_anomaly():
    accepted = _normalize(raw_market_value=9_000.0 * 1.34)
    rejected = _normalize(raw_market_value=9_000.0 * 1.36)

    assert accepted.valuation_valid is True
    assert accepted.market_value_basis == "LOCAL_CONVERTED"
    assert rejected.valuation_valid is False


def test_pnl_mismatch_quarantines_otherwise_valid_market_value():
    result = _normalize(raw_unrealized_pnl=-500.0)

    assert result.valuation_valid is False
    assert result.market_value_usd == 0.0
    assert result.market_value_basis == "UNAVAILABLE"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quantity", math.nan),
        ("current_price_local", math.inf),
        ("avg_cost_local", -math.inf),
        ("raw_market_value", math.nan),
        ("raw_unrealized_pnl", math.inf),
    ],
)
def test_non_finite_broker_values_fail_closed(field, value):
    result = _normalize(**{field: value})

    assert result.valuation_valid is False
    assert result.market_value_usd == 0.0
    assert result.unrealized_pnl_usd == 0.0
    assert "non-finite" in (result.valuation_issue or "")


def test_unknown_currency_never_becomes_usd_by_default():
    result = _normalize(currency="ZZZ")

    assert result.valuation_valid is False
    assert result.fx_rate_to_usd is None
    assert result.market_value_basis == "UNAVAILABLE"
    assert "ZZZ" in (result.valuation_issue or "")


def test_short_position_signs_are_preserved_during_conversion():
    result = _normalize(
        quantity=-100.0,
        raw_market_value=-9_000.0,
        raw_unrealized_pnl=1_000.0,
    )

    assert result.valuation_valid is True
    assert result.market_value_usd == pytest.approx(-60.3)
    assert result.unrealized_pnl_usd == pytest.approx(6.7)


def test_position_model_rejects_unknown_value_basis_token():
    with pytest.raises(ValidationError):
        NormalizedPosition(
            conid=1,
            ticker=Ticker.from_yf("7203.T", currency="JPY"),
            quantity=100,
            market_value_basis="LOCAL",  # type: ignore[arg-type]
        )
