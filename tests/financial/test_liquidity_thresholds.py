"""
Regression tests for liquidity threshold alignment.

PASS threshold: >$250k (aligned with PM/Market Analyst/Trader prompts)
MARGINAL:       $100k–$250k
FAIL:           <$100k

Prior to this fix the tool used $500k as the PASS boundary, causing stocks
at $250k–$500k (e.g. SCL.NZ at $358k) to be labelled MARGINAL by the tool
while the PM treated them as PASS, producing contradictory report text.
"""

from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from src.liquidity_calculation_tool import calculate_liquidity_metrics


def _mock_hist(mean_price: float, volume: int, n: int = 60) -> pd.DataFrame:
    """Return a price/volume DataFrame with deterministic alternating variation.

    Uses ±0.5% alternating prices so the mean is always exactly mean_price and
    no two consecutive prices are equal (avoids the flat-price irregularity flag).
    """
    prices = np.array([mean_price * (1.005 if i % 2 == 0 else 0.995) for i in range(n)])
    return pd.DataFrame({"Close": prices, "Volume": [volume] * n})


async def _run(ticker: str, mean_price: float, volume: int) -> str:
    """Invoke calculate_liquidity_metrics with mocked data sources at FX=1.0 (USD)."""
    with (
        patch(
            "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
            return_value=_mock_hist(mean_price, volume),
        ),
        patch(
            "src.liquidity_calculation_tool.market_data_fetcher.get_financial_metrics",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "src.liquidity_calculation_tool.get_fx_rate",
            new=AsyncMock(return_value=(1.0, "test")),
        ),
    ):
        return await calculate_liquidity_metrics.ainvoke({"ticker": ticker})


class TestLiquidityThresholdAlignment:
    """PASS=$250k, MARGINAL=$100k–$250k, FAIL<$100k."""

    @pytest.mark.asyncio
    async def test_above_250k_is_pass(self):
        """$300k daily turnover → PASS."""
        result = await _run("TEST.US", mean_price=6.0, volume=50_000)
        assert "Status: PASS" in result

    @pytest.mark.asyncio
    async def test_exactly_250k_is_pass(self):
        """$250k daily turnover → PASS (boundary, inclusive)."""
        result = await _run("TEST.US", mean_price=5.0, volume=50_000)
        assert "Status: PASS" in result

    @pytest.mark.asyncio
    async def test_just_below_250k_is_marginal(self):
        """$200k daily turnover → MARGINAL."""
        result = await _run("TEST.US", mean_price=4.0, volume=50_000)
        assert "Status: MARGINAL" in result

    @pytest.mark.asyncio
    async def test_exactly_100k_is_marginal(self):
        """$100k daily turnover → MARGINAL (lower boundary, inclusive)."""
        result = await _run("TEST.US", mean_price=2.0, volume=50_000)
        assert "Status: MARGINAL" in result

    @pytest.mark.asyncio
    async def test_just_below_100k_is_fail(self):
        """$75k daily turnover → FAIL (Insufficient Liquidity)."""
        result = await _run("TEST.US", mean_price=1.5, volume=50_000)
        assert "FAIL (Insufficient Liquidity)" in result

    @pytest.mark.asyncio
    async def test_output_string_references_250k(self):
        """Threshold description in tool output must reference $250,000, not $500,000."""
        result = await _run("TEST.US", mean_price=6.0, volume=50_000)
        assert "$250,000" in result
        assert "$500,000" not in result
