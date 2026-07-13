"""Unit coverage for inventory_turnover_trend() (COGS/inventory, latest-first)."""

from __future__ import annotations

import pandas as pd

from src.data.metric_extraction import inventory_turnover_trend


def _fin(cogs_latest_first: list[float], row: str = "Cost Of Revenue") -> pd.DataFrame:
    cols = [str(2025 - i) for i in range(len(cogs_latest_first))]
    return pd.DataFrame(
        {c: [v] for c, v in zip(cols, cogs_latest_first, strict=True)}, index=[row]
    )


def _bs(inv_latest_first: list[float], row: str = "Inventory") -> pd.DataFrame:
    cols = [str(2025 - i) for i in range(len(inv_latest_first))]
    return pd.DataFrame(
        {c: [v] for c, v in zip(cols, inv_latest_first, strict=True)}, index=[row]
    )


def test_rising_turnover():
    # COGS climbs, inventory flat -> turnover rising.
    res = inventory_turnover_trend(_fin([300, 260, 220]), _bs([100, 100, 100]))
    assert res == ("RISING", 3.0)


def test_falling_turnover():
    # Inventory building far faster than COGS -> turnover falling.
    res = inventory_turnover_trend(_fin([300, 290, 280]), _bs([300, 150, 100]))
    assert res is not None and res[0] == "FALLING"


def test_stable_turnover():
    res = inventory_turnover_trend(_fin([300, 290, 280]), _bs([100, 98, 96]))
    assert res is not None and res[0] == "STABLE"


def test_cogs_alias_cost_of_goods_sold():
    res = inventory_turnover_trend(
        _fin([300, 220], row="Cost Of Goods Sold"), _bs([100, 100])
    )
    assert res is not None and res[0] == "RISING"


def test_fewer_than_two_periods_returns_none():
    assert inventory_turnover_trend(_fin([300]), _bs([100])) is None


def test_missing_inventory_row_returns_none():
    bs = pd.DataFrame({"2025": [1.0], "2024": [1.0]}, index=["Other"])
    assert inventory_turnover_trend(_fin([300, 220]), bs) is None


def test_zero_inventory_period_skipped():
    # A zero-inventory period is skipped; remaining single period -> None (no crash).
    assert inventory_turnover_trend(_fin([300, 220]), _bs([0, 100])) is None


def test_negative_inventory_skipped():
    assert inventory_turnover_trend(_fin([300, 220]), _bs([-50, 100])) is None


def test_empty_frames_return_none():
    assert inventory_turnover_trend(pd.DataFrame(), pd.DataFrame()) is None
