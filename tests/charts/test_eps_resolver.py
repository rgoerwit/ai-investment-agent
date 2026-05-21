"""Tests for the EPS_TTM resolver (Tranche 5, Step 3).

Real DATA_BLOCKs from the wild rarely carry an EPS_TTM field; the resolver
must derive EPS from ``current_price / PE_RATIO_TTM`` so scenario valuation
can fire on production data.
"""

from __future__ import annotations

import pytest

from src.charts.extractors.valuation import resolve_eps_ttm


def _data_block(**fields: str) -> str:
    lines = ["### --- START DATA_BLOCK ---"]
    for k, v in fields.items():
        lines.append(f"{k}: {v}")
    lines.append("### --- END DATA_BLOCK ---")
    return "\n".join(lines)


def test_direct_eps_ttm_wins() -> None:
    block = _data_block(EPS_TTM="3.50", CURRENT_PRICE="100", PE_RATIO_TTM="20")
    # Resolver must NOT recompute from price/PE when direct EPS is valid.
    assert resolve_eps_ttm(block) == 3.50


def test_trailing_eps_alias_used_when_eps_ttm_missing() -> None:
    block = _data_block(TRAILING_EPS="2.10")
    assert resolve_eps_ttm(block) == 2.10


def test_derived_from_price_and_pe_when_eps_absent() -> None:
    """Most common production path — DATA_BLOCK lacks EPS field."""
    block = _data_block(CURRENT_PRICE="100.00", PE_RATIO_TTM="12.5")
    assert resolve_eps_ttm(block) == 8.0  # 100 / 12.5


def test_derived_rounded_to_4_decimals() -> None:
    block = _data_block(CURRENT_PRICE="100", PE_RATIO_TTM="3")
    # 100 / 3 = 33.33333... → rounded
    assert resolve_eps_ttm(block) == 33.3333


def test_returns_none_when_no_inputs() -> None:
    block = _data_block(SECTOR="Industrials")
    assert resolve_eps_ttm(block) is None


def test_returns_none_when_only_pe_present() -> None:
    block = _data_block(PE_RATIO_TTM="15")
    assert resolve_eps_ttm(block) is None


def test_returns_none_when_only_price_present() -> None:
    block = _data_block(CURRENT_PRICE="50")
    assert resolve_eps_ttm(block) is None


@pytest.mark.parametrize(
    "field_overrides",
    [
        {"CURRENT_PRICE": "100", "PE_RATIO_TTM": "0"},  # zero PE
        {"CURRENT_PRICE": "100", "PE_RATIO_TTM": "-5"},  # negative PE
        {"CURRENT_PRICE": "-10", "PE_RATIO_TTM": "12"},  # negative price
        {"CURRENT_PRICE": "0", "PE_RATIO_TTM": "12"},  # zero price
    ],
)
def test_returns_none_on_invalid_numerics(field_overrides) -> None:
    assert resolve_eps_ttm(_data_block(**field_overrides)) is None


def test_returns_none_for_unparseable_numerics() -> None:
    block = _data_block(EPS_TTM="N/A", CURRENT_PRICE="bad", PE_RATIO_TTM="x")
    assert resolve_eps_ttm(block) is None


def test_handles_commas_in_numerics() -> None:
    """Defensive: large-number formatting with commas still parses."""
    block = _data_block(CURRENT_PRICE="1,000.00", PE_RATIO_TTM="20")
    assert resolve_eps_ttm(block) == 50.0


def test_empty_or_none_inputs_return_none() -> None:
    assert resolve_eps_ttm("") is None
    assert resolve_eps_ttm(None) is None
