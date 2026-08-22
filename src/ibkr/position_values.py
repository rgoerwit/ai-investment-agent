"""Normalize IBKR position values without assuming their reported currency unit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

ValueBasis = Literal["BROKER_USD", "LOCAL_CONVERTED", "UNAVAILABLE"]

_MARKET_VALUE_TOLERANCE = 0.35
_PNL_TOLERANCE = 0.35
_MIN_EXPECTED_VALUE = 1e-9
# Broker payloads round; _MIN_EXPECTED_VALUE sits below their reporting precision,
# so "materially zero" needs its own, looser constant. Do not merge the two.
_FLAT_VALUE_EPSILON = 1e-6


@dataclass(frozen=True)
class NormalizedPositionValues:
    """USD values plus provenance needed to judge whether they are trustworthy."""

    market_value_usd: float
    unrealized_pnl_usd: float
    fx_rate_to_usd: float | None
    market_value_basis: ValueBasis
    unrealized_pnl_basis: ValueBasis
    valuation_valid: bool
    valuation_issue: str | None = None
    # A genuinely closed position, established before any FX or unit work.
    position_flat: bool = False


def _is_flat_position(
    *,
    quantity: float,
    raw_market_value: float,
    raw_unrealized_pnl: float | None,
) -> bool:
    """True when the broker reports a genuinely closed position.

    Every value leg must agree that nothing is held. A zero quantity reported
    beside a material market value is a broker inconsistency, not a flat
    position, and must stay on the data-quality path.

    Parse validity is guaranteed by the caller: ``normalize_positions`` routes
    any row with ``malformed_fields`` straight to an invalid result without
    reaching this function, so an unparseable quantity (which also arrives as
    0.0) can never be mistaken for a closed one. Finiteness is still checked
    here because ``float("inf")`` parses successfully.
    """
    if not math.isfinite(quantity) or quantity != 0.0:
        return False
    if not math.isfinite(raw_market_value):
        return False
    if abs(raw_market_value) > _FLAT_VALUE_EPSILON:
        return False
    if raw_unrealized_pnl is None:
        return True
    return (
        math.isfinite(raw_unrealized_pnl)
        and abs(raw_unrealized_pnl) <= _FLAT_VALUE_EPSILON
    )


def _is_contradictory_closed_row(
    *,
    quantity: float,
    raw_market_value: float,
    raw_unrealized_pnl: float | None,
) -> bool:
    """True when the broker reports no shares but a material value anyway.

    The distinguishing signal is the *quantity*, not the anchor. An ordinary
    holding whose ``mktPrice`` is simply absent from the payload also yields a
    zero anchor (``quantity * price``), and for USD the no-anchor fallback
    correctly resolves it as an identity. A row claiming zero shares beside a
    material value is a different thing: the two legs contradict each other, and
    no currency can classify it. Checked before the FX guard and the identity
    fallback so it fails closed everywhere rather than only outside USD.
    """
    if not math.isfinite(quantity) or quantity != 0.0:
        return False
    if math.isfinite(raw_market_value) and abs(raw_market_value) > _FLAT_VALUE_EPSILON:
        return True
    return (
        raw_unrealized_pnl is not None
        and math.isfinite(raw_unrealized_pnl)
        and abs(raw_unrealized_pnl) > _FLAT_VALUE_EPSILON
    )


def normalize_position_values(
    *,
    quantity: float,
    current_price_local: float,
    avg_cost_local: float,
    raw_market_value: float,
    raw_unrealized_pnl: float | None,
    currency: str,
    fx_rate: float | None,
) -> NormalizedPositionValues:
    """Classify broker values as local or USD and convert them exactly once.

    IBKR payloads encountered by this project are not uniform: some position
    values are denominated in the contract currency while others are already
    in the account's base currency. Quantity and local prices provide an
    independent unit check. Values that match neither convention fail closed.

    fx_rate must be resolved by the caller (see FxRateCache in
    src/fx_normalization.py) — this function is a pure computation and does
    not fetch rates itself.
    """
    normalized_currency = currency.strip().upper() or "USD"
    # A closed position is settled before any unit question arises: zero converts
    # to zero under either convention, and no FX rate is needed to say so. This
    # must precede the FX guard below, which would otherwise fail a flat position
    # in a currency whose rate could not be resolved.
    if _is_flat_position(
        quantity=quantity,
        raw_market_value=raw_market_value,
        raw_unrealized_pnl=raw_unrealized_pnl,
    ):
        # A flat row has no FX dependency, so report a rate only when it is
        # usable; passing an unusable one through would let a non-finite value
        # reach serializers and logs for a position worth nothing either way.
        flat_rate = 1.0 if normalized_currency == "USD" else fx_rate
        if flat_rate is not None and not (math.isfinite(flat_rate) and flat_rate > 0):
            flat_rate = None
        return NormalizedPositionValues(
            market_value_usd=0.0,
            unrealized_pnl_usd=0.0,
            fx_rate_to_usd=flat_rate,
            market_value_basis="UNAVAILABLE",
            unrealized_pnl_basis="UNAVAILABLE",
            valuation_valid=True,
            position_flat=True,
        )
    if _is_contradictory_closed_row(
        quantity=quantity,
        raw_market_value=raw_market_value,
        raw_unrealized_pnl=raw_unrealized_pnl,
    ):
        return _invalid_result(
            # None, not a 0.0 sentinel: this branch runs before the FX guard, so
            # the rate may legitimately be absent, and every other unavailable
            # rate in this module is reported as None.
            fx_rate if fx_rate is not None and math.isfinite(fx_rate) else None,
            (
                "Broker reports no shares held but a material market value or "
                f"P&L ({normalized_currency}) — quantity and value legs disagree"
            ),
        )
    if normalized_currency == "USD":
        fx_rate = 1.0
    if fx_rate is None or fx_rate <= 0:
        return NormalizedPositionValues(
            market_value_usd=0.0,
            unrealized_pnl_usd=0.0,
            fx_rate_to_usd=None,
            market_value_basis="UNAVAILABLE",
            unrealized_pnl_basis="UNAVAILABLE",
            valuation_valid=False,
            valuation_issue=(
                f"No local-to-USD FX rate is available for {normalized_currency}"
            ),
        )
    numeric_inputs = (
        quantity,
        current_price_local,
        avg_cost_local,
        raw_market_value,
    )
    if not all(math.isfinite(value) for value in numeric_inputs) or (
        raw_unrealized_pnl is not None and not math.isfinite(raw_unrealized_pnl)
    ):
        return _invalid_result(
            fx_rate,
            f"Broker position contains a non-finite numeric value ({normalized_currency})",
        )

    expected_market_local = quantity * current_price_local
    market_basis = _classify_basis(
        observed=raw_market_value,
        expected_local=expected_market_local,
        fx_rate_to_usd=fx_rate,
        tolerance=_MARKET_VALUE_TOLERANCE,
        fallback_basis="BROKER_USD" if normalized_currency == "USD" else None,
    )
    if market_basis is None:
        return _invalid_result(
            fx_rate,
            f"Broker market value is inconsistent with quantity and local price ({normalized_currency})",
        )

    market_value_usd = _to_usd(raw_market_value, market_basis, fx_rate)

    lacks_pnl_context = quantity == 0 or current_price_local <= 0 or avg_cost_local <= 0
    pnl_basis: ValueBasis | None
    if raw_unrealized_pnl is None or (
        normalized_currency != "USD" and lacks_pnl_context
    ):
        pnl_basis = "UNAVAILABLE"
        unrealized_pnl_usd = 0.0
    elif normalized_currency == "USD" and lacks_pnl_context:
        pnl_basis = "BROKER_USD"
        unrealized_pnl_usd = raw_unrealized_pnl
    else:
        expected_pnl_local = quantity * (current_price_local - avg_cost_local)
        if abs(expected_pnl_local) <= _MIN_EXPECTED_VALUE:
            observed_pnl_local = (
                raw_unrealized_pnl / fx_rate
                if market_basis == "BROKER_USD"
                else raw_unrealized_pnl
            )
            noise_tolerance_local = max(abs(expected_market_local) * 0.005, 0.01)
            pnl_basis = (
                market_basis
                if abs(observed_pnl_local) <= noise_tolerance_local
                else None
            )
        else:
            pnl_basis = _classify_basis(
                observed=raw_unrealized_pnl,
                expected_local=expected_pnl_local,
                fx_rate_to_usd=fx_rate,
                tolerance=_PNL_TOLERANCE,
                fallback_basis=market_basis,
            )
        if pnl_basis is None:
            return _invalid_result(
                fx_rate,
                f"Broker unrealized P&L is inconsistent with quantity and local prices ({normalized_currency})",
            )
        unrealized_pnl_usd = _to_usd(raw_unrealized_pnl, pnl_basis, fx_rate)

    return NormalizedPositionValues(
        market_value_usd=market_value_usd,
        unrealized_pnl_usd=unrealized_pnl_usd,
        fx_rate_to_usd=fx_rate,
        market_value_basis=market_basis,
        unrealized_pnl_basis=pnl_basis,
        valuation_valid=True,
    )


def _classify_basis(
    *,
    observed: float,
    expected_local: float,
    fx_rate_to_usd: float,
    tolerance: float,
    fallback_basis: ValueBasis | None = "LOCAL_CONVERTED",
) -> ValueBasis | None:
    """Return the unit convention whose independently expected value is closest.

    With no usable anchor the caller's fallback decides, and the asymmetry there
    is deliberate: for USD the fallback is an identity (a USD position's value is
    USD by construction, so there is no unit question to answer), while for any
    other currency it is None because local-vs-USD is genuinely undecidable
    without an anchor. IBKR routinely omits mktPrice, so this branch is the
    normal path for an ordinary USD holding — not a degenerate one. A closed
    position never reaches it at all (see _is_flat_position).
    """
    if abs(expected_local) <= _MIN_EXPECTED_VALUE:
        return fallback_basis

    local_error = _relative_error(observed, expected_local)
    usd_error = _relative_error(observed, expected_local * fx_rate_to_usd)
    best_error = min(local_error, usd_error)
    if best_error > tolerance:
        return None
    return "LOCAL_CONVERTED" if local_error < usd_error else "BROKER_USD"


def _relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / max(abs(expected), _MIN_EXPECTED_VALUE)


def _to_usd(value: float, basis: ValueBasis, fx_rate_to_usd: float) -> float:
    if basis == "LOCAL_CONVERTED":
        return value * fx_rate_to_usd
    return value


def _invalid_result(
    fx_rate_to_usd: float | None,
    issue: str,
) -> NormalizedPositionValues:
    return NormalizedPositionValues(
        market_value_usd=0.0,
        unrealized_pnl_usd=0.0,
        fx_rate_to_usd=fx_rate_to_usd,
        market_value_basis="UNAVAILABLE",
        unrealized_pnl_basis="UNAVAILABLE",
        valuation_valid=False,
        valuation_issue=issue,
    )
