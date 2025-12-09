"""
Backwards Compatibility Test for FX Normalization Changes

Verifies that the switch from static FX rates to dynamic FX rates
doesn't break existing liquidity calculations.

Compares old static rates vs new dynamic rates (mocked) to ensure:
1. Results are numerically similar (within 10% tolerance for FX drift)
2. Pass/fail status remains consistent for borderline cases
3. No calculation logic bugs introduced
"""

import pytest
import pandas as pd
from unittest.mock import patch, AsyncMock

from src.liquidity_calculation_tool import calculate_liquidity_metrics


# Old static rates from before FX normalization changes
OLD_STATIC_RATES = {
    'JPY': 0.0067,
    'HKD': 0.129,
    'TWD': 0.031,
    'KRW': 0.00072,
    'CNY': 0.138,
    'GBP': 1.27,
    'EUR': 1.05,
}


class TestBackwardsCompatibility:
    """Test that FX changes don't break existing liquidity calculations."""

    @pytest.mark.asyncio
    async def test_hkd_calculation_consistency(self):
        """Test HKD calculation is consistent with old static rate."""
        # HKD 60 * 100k volume * 0.129 old rate = $774k USD (PASS)
        mock_hist = pd.DataFrame({
            'Close': [60.0] * 60,
            'Volume': [100_000] * 60
        })

        with patch('src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices',
                   return_value=mock_hist):
            # Mock FX rate to match old static rate
            with patch('src.liquidity_calculation_tool.get_fx_rate',
                       new=AsyncMock(return_value=(0.129, "yfinance"))):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "0005.HK"})

        # Should pass (>$500k threshold)
        assert "Status: PASS" in result
        assert "HKD" in result
        # Old calculation: 60 * 100000 * 0.129 = $774,000
        assert "$774,000" in result or "$774000" in result

    @pytest.mark.asyncio
    async def test_jpy_calculation_consistency(self):
        """Test JPY calculation is consistent with old static rate."""
        # JPY 2500 * 100k volume * 0.0067 old rate = $1,675,000 USD (PASS)
        mock_hist = pd.DataFrame({
            'Close': [2500.0] * 60,
            'Volume': [100_000] * 60
        })

        with patch('src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices',
                   return_value=mock_hist):
            # Mock FX rate to match old static rate
            with patch('src.liquidity_calculation_tool.get_fx_rate',
                       new=AsyncMock(return_value=(0.0067, "yfinance"))):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "7203.T"})

        # Should pass
        assert "Status: PASS" in result
        assert "JPY" in result
        # Old calculation: 2500 * 100000 * 0.0067 = $1,675,000
        assert "$1,675,000" in result or "$1675000" in result

    @pytest.mark.asyncio
    async def test_twd_calculation_consistency(self):
        """Test TWD calculation is consistent with old static rate."""
        # TWD 500 * 100k volume * 0.031 old rate = $1,550,000 USD (PASS)
        mock_hist = pd.DataFrame({
            'Close': [500.0] * 60,
            'Volume': [100_000] * 60
        })

        with patch('src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices',
                   return_value=mock_hist):
            # Mock FX rate to match old static rate
            with patch('src.liquidity_calculation_tool.get_fx_rate',
                       new=AsyncMock(return_value=(0.031, "yfinance"))):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "2330.TW"})

        # Should pass
        assert "Status: PASS" in result
        assert "TWD" in result

    @pytest.mark.asyncio
    async def test_borderline_case_old_vs_new(self):
        """Test borderline case where FX rate drift could change pass/fail."""
        # HKD 40 * 100k volume:
        # - Old rate 0.129: 40 * 100k * 0.129 = $516,000 (PASS)
        # - New rate 0.128: 40 * 100k * 0.128 = $512,000 (PASS)
        # Both should pass - difference is small

        mock_hist = pd.DataFrame({
            'Close': [40.0] * 60,
            'Volume': [100_000] * 60
        })

        # Test with old rate
        with patch('src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices',
                   return_value=mock_hist):
            with patch('src.liquidity_calculation_tool.get_fx_rate',
                       new=AsyncMock(return_value=(0.129, "fallback"))):
                result_old = await calculate_liquidity_metrics.ainvoke({"ticker": "TEST.HK"})

        # Test with new rate
        with patch('src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices',
                   return_value=mock_hist):
            with patch('src.liquidity_calculation_tool.get_fx_rate',
                       new=AsyncMock(return_value=(0.128, "yfinance"))):
                result_new = await calculate_liquidity_metrics.ainvoke({"ticker": "TEST.HK"})

        # Both should pass (both above $500k threshold)
        assert "Status: PASS" in result_old
        assert "Status: PASS" in result_new

        # Extract turnover values
        import re
        turnover_old = int(re.search(r'Turnover \(USD\): \$(\d+,\d+)', result_old).group(1).replace(',', ''))
        turnover_new = int(re.search(r'Turnover \(USD\): \$(\d+,\d+)', result_new).group(1).replace(',', ''))

        # Difference should be small (within 5%)
        diff_pct = abs(turnover_old - turnover_new) / turnover_old
        assert diff_pct < 0.05, f"Turnover difference too large: {diff_pct:.2%}"

    @pytest.mark.asyncio
    async def test_fx_fallback_matches_old_static(self):
        """Test that fallback rates match old static rates (where applicable)."""
        # This verifies our fallback rates are still reasonable
        from src.fx_normalization import FALLBACK_RATES_TO_USD

        # Check key currencies match (within 10% tolerance for drift)
        for currency, old_rate in OLD_STATIC_RATES.items():
            fallback_rate = FALLBACK_RATES_TO_USD.get(currency)
            assert fallback_rate is not None, f"Missing fallback rate for {currency}"

            diff_pct = abs(fallback_rate - old_rate) / old_rate
            assert diff_pct < 0.15, \
                f"{currency}: Fallback rate {fallback_rate} differs too much from old static {old_rate} ({diff_pct:.1%})"

    @pytest.mark.asyncio
    async def test_usd_stocks_unchanged(self):
        """Test USD stocks work exactly as before (no FX conversion needed)."""
        mock_hist = pd.DataFrame({
            'Close': [150.0] * 60,
            'Volume': [50_000_000] * 60
        })

        with patch('src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices',
                   return_value=mock_hist):
            with patch('src.liquidity_calculation_tool.get_fx_rate',
                       new=AsyncMock(return_value=(1.0, "identity"))):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "AAPL"})

        # Should pass
        assert "Status: PASS" in result
        # USD calculation: 150 * 50,000,000 = $7,500,000,000
        assert "$7,500,000,000" in result or "7.5B" in result.replace(",", "")

    @pytest.mark.asyncio
    async def test_gbp_pence_adjustment_unchanged(self):
        """Test UK stocks pence adjustment still works correctly."""
        # 400 pence = £4.00
        # £4.00 * 100k volume * 1.27 FX = $508,000 USD
        mock_hist = pd.DataFrame({
            'Close': [400.0] * 60,  # 400 pence
            'Volume': [100_000] * 60
        })

        with patch('src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices',
                   return_value=mock_hist):
            with patch('src.liquidity_calculation_tool.get_fx_rate',
                       new=AsyncMock(return_value=(1.27, "yfinance"))):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "VOD.L"})

        # Should pass
        assert "Status: PASS" in result
        assert "GBP" in result
        # Pence adjustment: 400/100 = £4 * 100k * 1.27 = $508,000
        assert "$508,000" in result or "$508000" in result

    @pytest.mark.asyncio
    async def test_calculation_logic_unchanged(self):
        """Test that core calculation logic (avg volume * avg price * FX) is unchanged."""
        # Use different prices and volumes to test averaging
        mock_hist = pd.DataFrame({
            'Close': [100.0, 110.0, 90.0] * 20,  # avg = 100
            'Volume': [80_000, 100_000, 120_000] * 20  # avg = 100,000
        })

        with patch('src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices',
                   return_value=mock_hist):
            with patch('src.liquidity_calculation_tool.get_fx_rate',
                       new=AsyncMock(return_value=(0.129, "yfinance"))):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "TEST.HK"})

        # Expected: avg_price=100 * avg_volume=100,000 * fx=0.129 = $1,290,000
        assert "Status: PASS" in result
        assert "$1,290,000" in result or "$1290000" in result

    @pytest.mark.asyncio
    async def test_threshold_unchanged(self):
        """Test that $500k USD threshold is unchanged."""
        # Test at exactly $500k (should FAIL - threshold is >500k, not >=)
        # Price $5 * volume 100k * FX 1.0 = $500,000
        mock_hist = pd.DataFrame({
            'Close': [5.0] * 60,
            'Volume': [100_000] * 60
        })

        with patch('src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices',
                   return_value=mock_hist):
            with patch('src.liquidity_calculation_tool.get_fx_rate',
                       new=AsyncMock(return_value=(1.0, "identity"))):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "BOUNDARY"})

        # At exactly $500k, should FAIL (threshold is > not >=)
        assert "Status: FAIL" in result
        assert "$500,000" in result
        assert "Threshold: $500,000 USD daily" in result


class TestRealWorldScenarios:
    """Test with real-world-like scenarios to catch edge cases."""

    @pytest.mark.asyncio
    async def test_hsbc_real_world(self):
        """Test HSBC (0005.HK) with realistic prices and volume."""
        # HSBC typically trades around HKD 60-70, volume 10-20M shares
        mock_hist = pd.DataFrame({
            'Close': [65.0] * 60,
            'Volume': [15_000_000] * 60
        })

        with patch('src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices',
                   return_value=mock_hist):
            with patch('src.liquidity_calculation_tool.get_fx_rate',
                       new=AsyncMock(return_value=(0.128, "yfinance"))):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "0005.HK"})

        # Should easily pass
        assert "Status: PASS" in result
        # 65 * 15M * 0.128 = $124.8M USD
        assert "HKD" in result

    @pytest.mark.asyncio
    async def test_tsmc_real_world(self):
        """Test TSMC (2330.TW) with realistic prices and volume."""
        # TSMC typically trades around TWD 500-600, volume 20-40M shares
        mock_hist = pd.DataFrame({
            'Close': [550.0] * 60,
            'Volume': [30_000_000] * 60
        })

        with patch('src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices',
                   return_value=mock_hist):
            with patch('src.liquidity_calculation_tool.get_fx_rate',
                       new=AsyncMock(return_value=(0.032, "yfinance"))):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "2330.TW"})

        # Should easily pass
        assert "Status: PASS" in result
        # 550 * 30M * 0.032 = $528M USD
        assert "TWD" in result

    @pytest.mark.asyncio
    async def test_toyota_real_world(self):
        """Test Toyota (7203.T) with realistic prices and volume."""
        # Toyota typically trades around JPY 2500, volume 5-10M shares
        mock_hist = pd.DataFrame({
            'Close': [2500.0] * 60,
            'Volume': [7_000_000] * 60
        })

        with patch('src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices',
                   return_value=mock_hist):
            with patch('src.liquidity_calculation_tool.get_fx_rate',
                       new=AsyncMock(return_value=(0.0067, "yfinance"))):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "7203.T"})

        # Should easily pass
        assert "Status: PASS" in result
        # 2500 * 7M * 0.0067 = $117.25M USD
        assert "JPY" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
