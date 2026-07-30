"""Normalize IBKR position values without assuming their reported currency unit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

ValueBasis = Literal["BROKER_USD", "LOCAL_CONVERTED", "UNAVAILABLE"]

_MARKET_VALUE_TOLERANCE = 0.35
_PNL_TOLERANCE = 0.35
_MIN_EXPECTED_VALUE = 1e-9


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
    if raw_unrealized_pnl is None or (
        normalized_currency != "USD" and lacks_pnl_context
    ):
        pnl_basis: ValueBasis = "UNAVAILABLE"
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
    """Return the unit convention whose independently expected value is closest."""
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
    fx_rate_to_usd: float,
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
