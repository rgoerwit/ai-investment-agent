"""
Tests for Liquidity Tool FX Integration

Tests the minimal currency normalization fix: liquidity tool now converts
prices to USD using dynamic FX rates before checking liquidity threshold.

This is the ONLY integration of fx_normalization.py into the main codebase.
"""

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.liquidity_calculation_tool import (
    EXCHANGE_CURRENCY_MAP,
    calculate_liquidity_metrics,
)


# Helper to call the LangChain tool
async def call_tool(ticker):
    """Helper to call the liquidity tool."""
    return await calculate_liquidity_metrics.ainvoke({"ticker": ticker})


class TestLiquidityFXConversion:
    """Test that liquidity tool correctly converts currencies to USD."""

    @pytest.mark.asyncio
    async def test_usd_stock_no_conversion(self):
        """Test US stock (AAPL) requires no FX conversion."""
        mock_hist = pd.DataFrame(
            {
                "Close": [150.0, 151.0, 152.0],
                "Volume": [50_000_000, 51_000_000, 49_000_000],
            }
        )

        with patch(
            "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
            return_value=mock_hist,
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(1.0, "identity")),
            ):
                result = await call_tool("AAPL")

        assert "Status: PASS" in result
        assert "USD" in result
        assert "1.000000" in result

    @pytest.mark.asyncio
    async def test_hk_stock_converts_to_usd(self):
        """Test Hong Kong stock converts HKD to USD."""
        mock_hist = pd.DataFrame(
            {"Close": [60.0, 61.0, 59.0], "Volume": [10_000_000, 11_000_000, 9_000_000]}
        )

        with patch(
            "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
            return_value=mock_hist,
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(0.128, "yfinance")),
            ):
                result = await call_tool("0005.HK")

        assert "Status: PASS" in result
        assert "HKD" in result
        assert "0.128" in result

    @pytest.mark.asyncio
    async def test_japan_stock_converts_to_usd(self):
        """Test Japanese stock converts JPY to USD."""
        mock_hist = pd.DataFrame(
            {
                "Close": [2500.0, 2520.0, 2480.0],
                "Volume": [5_000_000, 5_200_000, 4_800_000],
            }
        )

        with patch(
            "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
            return_value=mock_hist,
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(0.0067, "yfinance")),
            ):
                result = await call_tool("7203.T")

        assert "Status: PASS" in result
        assert "JPY" in result

    @pytest.mark.asyncio
    async def test_low_liquidity_fails(self):
        """Test low liquidity international stock fails threshold."""
        mock_hist = pd.DataFrame(
            {"Close": [10.0, 10.5, 9.5], "Volume": [10_000, 11_000, 9_000]}
        )

        with patch(
            "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
            return_value=mock_hist,
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(0.128, "yfinance")),
            ):
                result = await call_tool("XXXX.HK")

        assert "Status: FAIL" in result

    @pytest.mark.asyncio
    async def test_fx_fallback_rates(self):
        """Test fallback to static rates when yfinance fails."""
        mock_hist = pd.DataFrame({"Close": [2500.0], "Volume": [5_000_000]})

        with patch(
            "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
            return_value=mock_hist,
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(0.0067, "fallback")),
            ):
                result = await call_tool("7203.T")

        assert "fallback" in result


class TestExchangeCurrencyMapping:
    """Test EXCHANGE_CURRENCY_MAP covers all major exchanges."""

    def test_major_asian_exchanges_present(self):
        """Test major Asian exchanges are in currency map."""
        assert "HK" in EXCHANGE_CURRENCY_MAP
        assert "T" in EXCHANGE_CURRENCY_MAP
        assert "TW" in EXCHANGE_CURRENCY_MAP
        assert "KS" in EXCHANGE_CURRENCY_MAP

    def test_currency_codes_are_strings(self):
        """Test all currency codes are strings (not tuples)."""
        for suffix, currency in EXCHANGE_CURRENCY_MAP.items():
            assert isinstance(currency, str), f"{suffix} maps to {type(currency)}"


class TestBackwardsCompatibility:
    """Test that changes don't break existing behavior."""

    @pytest.mark.asyncio
    async def test_empty_history_returns_fail(self):
        """Test empty history returns FAIL status."""
        empty_hist = pd.DataFrame()

        with patch(
            "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
            return_value=empty_hist,
        ):
            result = await call_tool("XXXX.XX")

        assert "Status: FAIL" in result
        assert "Insufficient Data" in result

    @pytest.mark.asyncio
    async def test_exception_returns_error(self):
        """Test exception handling returns ERROR status."""
        with patch(
            "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
            side_effect=Exception("Network error"),
        ):
            result = await call_tool("TEST.US")

        assert "Status: ERROR" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
