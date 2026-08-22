"""Unit and anomaly matrix for broker position-value normalization."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from src.ibkr.models import NormalizedPosition
from src.ibkr.position_values import normalize_position_values
from src.ibkr.ticker import Ticker


def _normalize(**overrides):
    # normalize_position_values() is a pure function — fx_rate is caller-supplied
    # (see FxRateCache in src/fx_normalization.py), not looked up here. This
    # default is a fixed test fixture, independent of the real fallback table.
    values = {
        "quantity": 100.0,
        "current_price_local": 90.0,
        "avg_cost_local": 100.0,
        "raw_market_value": 9_000.0,
        "raw_unrealized_pnl": -1_000.0,
        "currency": "JPY",
        "fx_rate": 0.0067,
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
    result = _normalize(currency=currency, fx_rate=rate)

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
        fx_rate=rate,
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
    """A USD value needs no unit inference when the price context is missing.

    Narrowed 2026-08-21. This previously asserted the same for `quantity=0.0`,
    but a row claiming no shares beside a material value is a contradiction, not
    an identity conversion: `market_value_usd` is unambiguous while *whether
    anything is held* is not, and the reconciler went on to generate orders from
    it. That shape is now a data-quality review (see
    TestContradictoryClosedRowFailsClosedEverywhere); the identity conversion
    this test exists for is preserved with a real share count.
    """
    result = _normalize(
        currency="USD",
        quantity=25.0,
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
    # The caller (FxRateCache) is what determines a currency is unresolvable
    # and passes fx_rate=None — this function just fails closed on it.
    result = _normalize(currency="ZZZ", fx_rate=None)

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


class TestFlatPositionIsValidatedNotFailed:
    """A closed position is a settled fact, not a valuation failure.

    Regression (2026-08-19): a sold JPY/MXN holding still in the broker snapshot
    had quantity 0, so the unit anchor `quantity * price` was 0 and
    `_classify_basis` fell back to None for every non-USD currency. That marked
    the position `valuation_valid=False`, which let it slip past the evaluator's
    closed-position skip and surface as an *urgent* DATA_QUALITY analysis
    refresh for stock the operator no longer owned.
    """

    @pytest.mark.parametrize(
        ("currency", "rate"),
        [("JPY", 0.0067), ("MXN", 0.052), ("GBp", 0.0135), ("USD", 1.0)],
    )
    def test_closed_position_is_flat_and_valid_in_every_currency(self, currency, rate):
        # Parametrized across currencies deliberately: the defect was invisible
        # because the USD case alone behaved correctly.
        result = _normalize(
            quantity=0.0,
            raw_market_value=0.0,
            raw_unrealized_pnl=0.0,
            currency=currency,
            fx_rate=rate,
        )

        assert result.position_flat is True
        assert result.valuation_valid is True
        assert result.valuation_issue is None
        assert result.market_value_usd == 0.0

    def test_closed_position_needs_no_fx_rate(self):
        """The second, independent door: the FX guard runs before classification."""
        result = _normalize(
            quantity=0.0,
            raw_market_value=0.0,
            raw_unrealized_pnl=0.0,
            currency="ZZZ",
            fx_rate=None,
        )

        assert result.position_flat is True
        assert result.valuation_valid is True

    def test_closed_position_without_pnl_reported_is_still_flat(self):
        result = _normalize(quantity=0.0, raw_market_value=0.0, raw_unrealized_pnl=None)

        assert result.position_flat is True
        assert result.valuation_valid is True

    @pytest.mark.parametrize(("currency", "rate"), [("JPY", 0.0067), ("USD", 1.0)])
    def test_zero_quantity_with_material_value_is_not_flat(self, currency, rate):
        """A broker inconsistency must stay on the data-quality path."""
        result = _normalize(
            quantity=0.0,
            raw_market_value=5_000.0,
            raw_unrealized_pnl=0.0,
            currency=currency,
            fx_rate=rate,
        )

        assert result.position_flat is False

    def test_zero_quantity_with_material_pnl_is_not_flat(self):
        result = _normalize(
            quantity=0.0, raw_market_value=0.0, raw_unrealized_pnl=250.0
        )

        assert result.position_flat is False

    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    def test_non_finite_legs_are_never_flat(self, value):
        assert (
            _normalize(
                quantity=0.0, raw_market_value=value, raw_unrealized_pnl=0.0
            ).position_flat
            is False
        )
        assert (
            _normalize(
                quantity=value, raw_market_value=0.0, raw_unrealized_pnl=0.0
            ).position_flat
            is False
        )

    def test_held_position_is_never_flat(self):
        assert _normalize().position_flat is False

    def test_usd_identity_fallback_survives_a_missing_price(self):
        """IBKR routinely omits mktPrice; a USD value needs no unit inference.

        Guards the asymmetry in `_classify_basis`: for USD the no-anchor
        fallback is an identity, while for any other currency it is None
        because local-vs-USD is genuinely undecidable.
        """
        result = _normalize(
            currency="USD",
            fx_rate=1.0,
            quantity=10.0,
            current_price_local=0.0,
            avg_cost_local=0.0,
            raw_market_value=1_800.0,
            raw_unrealized_pnl=0.0,
        )

        assert result.valuation_valid is True
        assert result.position_flat is False
        assert result.market_value_basis == "BROKER_USD"
        assert result.market_value_usd == pytest.approx(1_800.0)


class TestContradictoryClosedRowFailsClosedEverywhere:
    """Zero shares beside a material value is a broker inconsistency, not a holding.

    Found 2026-08-21 after the flat-position work: the flat predicate correctly
    declined these rows, but they then fell through to `_classify_basis`, whose
    no-anchor fallback is an identity for USD. So a USD row claiming 0 shares
    and a $5,000 value was accepted as a valid holding and could reach the
    order path. The distinguishing signal is the *quantity*, not the anchor —
    an ordinary holding whose `mktPrice` is absent also has a zero anchor, and
    that case must stay valid (guarded below).
    """

    @pytest.mark.parametrize(
        ("currency", "rate"),
        [("USD", 1.0), ("JPY", 0.0067), ("MXN", 0.052), ("GBp", 0.0135)],
    )
    def test_material_market_value_without_shares_is_invalid(self, currency, rate):
        result = _normalize(
            quantity=0.0,
            raw_market_value=5_000.0,
            raw_unrealized_pnl=0.0,
            currency=currency,
            fx_rate=rate,
        )

        assert result.position_flat is False
        assert result.valuation_valid is False
        assert result.market_value_usd == 0.0
        assert "no shares held" in (result.valuation_issue or "")

    @pytest.mark.parametrize("rate", [None, math.nan, math.inf])
    def test_unavailable_rate_is_reported_as_none_not_a_zero_sentinel(self, rate):
        """Matches the FX-missing branch immediately below it in the module."""
        result = _normalize(
            quantity=0.0,
            raw_market_value=5_000.0,
            raw_unrealized_pnl=0.0,
            currency="JPY",
            fx_rate=rate,
        )

        assert result.valuation_valid is False
        assert result.fx_rate_to_usd is None

    def test_material_pnl_without_shares_is_invalid(self):
        result = _normalize(
            quantity=0.0,
            current_price_local=0.0,
            avg_cost_local=0.0,
            raw_market_value=0.0,
            raw_unrealized_pnl=250.0,
            currency="USD",
            fx_rate=1.0,
        )

        assert result.valuation_valid is False

    def test_ordinary_holding_with_absent_price_is_still_valid(self):
        """The regression guard for the fix above.

        IBKR routinely omits `mktPrice`, which zeroes the anchor for a real
        holding. Rejecting the whole no-anchor branch (an earlier attempt at
        this fix) broke exactly this shape.
        """
        result = _normalize(
            quantity=10.0,
            current_price_local=0.0,
            avg_cost_local=0.0,
            raw_market_value=1_800.0,
            raw_unrealized_pnl=0.0,
            currency="USD",
            fx_rate=1.0,
        )

        assert result.valuation_valid is True
        assert result.market_value_basis == "BROKER_USD"
        assert result.market_value_usd == pytest.approx(1_800.0)


class TestFlatRowDropsAnUnusableFxRate:
    """A flat row has no FX dependency, so it must not propagate a bad rate."""

    @pytest.mark.parametrize("rate", [math.nan, math.inf, -math.inf, None, 0.0, -1.0])
    def test_unusable_rate_becomes_none(self, rate):
        result = _normalize(
            quantity=0.0,
            raw_market_value=0.0,
            raw_unrealized_pnl=0.0,
            currency="JPY",
            fx_rate=rate,
        )

        assert result.position_flat is True
        assert result.valuation_valid is True
        assert result.fx_rate_to_usd is None

    def test_usable_rate_is_preserved(self):
        result = _normalize(
            quantity=0.0, raw_market_value=0.0, raw_unrealized_pnl=0.0, fx_rate=0.0067
        )

        assert result.fx_rate_to_usd == pytest.approx(0.0067)
