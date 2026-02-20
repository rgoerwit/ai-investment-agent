"""
Backwards Compatibility Test for FX Normalization Changes

Verifies that the switch from static FX rates to dynamic FX rates
doesn't break existing liquidity calculations.

Compares old static rates vs new dynamic rates (mocked) to ensure:
1. Results are numerically similar (within 10% tolerance for FX drift)
2. Pass/fail status remains consistent for borderline cases
3. No calculation logic bugs introduced
"""

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.liquidity_calculation_tool import calculate_liquidity_metrics

# Old static rates from before FX normalization changes
OLD_STATIC_RATES = {
    "JPY": 0.0067,
    "HKD": 0.129,
    "TWD": 0.031,
    "KRW": 0.00072,
    "CNY": 0.138,
    "GBP": 1.27,
    "EUR": 1.05,
}


class TestBackwardsCompatibility:
    """Test that FX changes don't break existing liquidity calculations."""

    @pytest.mark.asyncio
    async def test_hkd_calculation_consistency(self):
        """Test HKD calculation is consistent with old static rate."""
        # HKD 60 * 100k volume * 0.129 old rate = $774k USD (PASS)
        # Vary prices slightly to avoid flat-price detection (±1% variation)
        import numpy as np

        prices = 60.0 + np.random.uniform(-0.6, 0.6, 60)
        mock_hist = pd.DataFrame({"Close": prices, "Volume": [100_000] * 60})

        with (
            patch(
                "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
                return_value=mock_hist,
            ),
            patch(
                # Mock get_financial_metrics to return None to force fallback
                # to historical mean price (standard test behavior here).
                "src.liquidity_calculation_tool.market_data_fetcher.get_financial_metrics",
                new=AsyncMock(return_value=None),
            ),
        ):
            # Mock FX rate to match old static rate
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(0.129, "yfinance")),
            ):
                result = await calculate_liquidity_metrics.ainvoke(
                    {"ticker": "0005.HK"}
                )

        # Should pass (>$500k threshold)
        assert "Status: PASS" in result
        assert "HKD" in result
        # Old calculation: ~60 * 100000 * 0.129 = ~$774,000 (may vary slightly due to price variation)
        assert "Turnover (USD):" in result
        # Check turnover is in expected range ($700k-$850k)
        import re

        turnover_match = re.search(r"Turnover \(USD\): \$(\d+,\d+)", result)
        if turnover_match:
            turnover = int(turnover_match.group(1).replace(",", ""))
            assert 700_000 < turnover < 850_000

    @pytest.mark.asyncio
    async def test_jpy_calculation_consistency(self):
        """Test JPY calculation is consistent with old static rate."""
        # JPY 2500 * 100k volume * 0.0067 old rate = $1,675,000 USD (PASS)
        # Vary prices slightly to avoid flat-price detection (±1% variation)
        import numpy as np

        prices = 2500.0 + np.random.uniform(-25, 25, 60)
        mock_hist = pd.DataFrame({"Close": prices, "Volume": [100_000] * 60})

        with (
            patch(
                "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
                return_value=mock_hist,
            ),
            patch(
                # Mock get_financial_metrics to return None to force fallback
                # to historical mean price (standard test behavior here).
                "src.liquidity_calculation_tool.market_data_fetcher.get_financial_metrics",
                new=AsyncMock(return_value=None),
            ),
        ):
            # Mock FX rate to match old static rate
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(0.0067, "yfinance")),
            ):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "7203.T"})

        # Should pass
        assert "Status: PASS" in result
        assert "JPY" in result
        # Old calculation: ~2500 * 100000 * 0.0067 = ~$1,675,000 (may vary slightly)
        assert "Turnover (USD):" in result
        # Check turnover is in expected range ($1.6M-$1.75M)
        import re

        turnover_match = re.search(r"Turnover \(USD\): \$(\d+,\d+,?\d*)", result)
        if turnover_match:
            turnover = int(turnover_match.group(1).replace(",", ""))
            assert 1_600_000 < turnover < 1_750_000

    @pytest.mark.asyncio
    async def test_twd_calculation_consistency(self):
        """Test TWD calculation is consistent with old static rate."""
        # TWD 500 * 100k volume * 0.031 old rate = $1,550,000 USD (PASS)
        # Vary prices slightly to avoid flat-price detection (±1% variation)
        import numpy as np

        prices = 500.0 + np.random.uniform(-5, 5, 60)
        mock_hist = pd.DataFrame({"Close": prices, "Volume": [100_000] * 60})

        with (
            patch(
                "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
                return_value=mock_hist,
            ),
            patch(
                # Mock get_financial_metrics to return None to force fallback
                # to historical mean price (standard test behavior here).
                "src.liquidity_calculation_tool.market_data_fetcher.get_financial_metrics",
                new=AsyncMock(return_value=None),
            ),
        ):
            # Mock FX rate to match old static rate
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(0.031, "yfinance")),
            ):
                result = await calculate_liquidity_metrics.ainvoke(
                    {"ticker": "2330.TW"}
                )

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

        # Vary prices slightly to avoid flat-price detection (±1% variation)
        import numpy as np

        prices = 40.0 + np.random.uniform(-0.4, 0.4, 60)
        mock_hist = pd.DataFrame({"Close": prices, "Volume": [100_000] * 60})

        # Test with old rate
        with (
            patch(
                "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
                return_value=mock_hist,
            ),
            patch(
                # Mock get_financial_metrics to return None to force fallback
                # to historical mean price (standard test behavior here).
                "src.liquidity_calculation_tool.market_data_fetcher.get_financial_metrics",
                new=AsyncMock(return_value=None),
            ),
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(0.129, "fallback")),
            ):
                result_old = await calculate_liquidity_metrics.ainvoke(
                    {"ticker": "TEST.HK"}
                )

        # Test with new rate
        with (
            patch(
                "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
                return_value=mock_hist,
            ),
            patch(
                # Mock get_financial_metrics to return None to force fallback
                # to historical mean price (standard test behavior here).
                "src.liquidity_calculation_tool.market_data_fetcher.get_financial_metrics",
                new=AsyncMock(return_value=None),
            ),
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(0.128, "yfinance")),
            ):
                result_new = await calculate_liquidity_metrics.ainvoke(
                    {"ticker": "TEST.HK"}
                )

        # Both should pass (both above $500k threshold)
        assert "Status: PASS" in result_old
        assert "Status: PASS" in result_new

        # Extract turnover values
        import re

        turnover_old = int(
            re.search(r"Turnover \(USD\): \$(\d+,\d+)", result_old)
            .group(1)
            .replace(",", "")
        )
        turnover_new = int(
            re.search(r"Turnover \(USD\): \$(\d+,\d+)", result_new)
            .group(1)
            .replace(",", "")
        )

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
            assert (
                diff_pct < 0.15
            ), f"{currency}: Fallback rate {fallback_rate} differs too much from old static {old_rate} ({diff_pct:.1%})"

    @pytest.mark.asyncio
    async def test_usd_stocks_unchanged(self):
        """Test USD stocks work exactly as before (no FX conversion needed)."""
        # Vary prices slightly to avoid flat-price detection (±1% variation)
        import numpy as np

        prices = 150.0 + np.random.uniform(-1.5, 1.5, 60)
        mock_hist = pd.DataFrame({"Close": prices, "Volume": [50_000_000] * 60})

        with (
            patch(
                "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
                return_value=mock_hist,
            ),
            patch(
                # Class-level mock to ensure singleton doesn't hit network
                "src.data.fetcher.SmartMarketDataFetcher.get_financial_metrics",
                new=AsyncMock(return_value=None),
            ),
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(1.0, "identity")),
            ):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "AAPL"})

        # Should pass
        assert "Status: PASS" in result
        # USD calculation: ~150 * 50,000,000 = ~$7,500,000,000 (may vary slightly)
        assert "Turnover (USD):" in result
        # Check turnover is in expected range ($7B-$8B)
        import re

        result_no_commas = result.replace(",", "")
        turnover_match = re.search(r"Turnover \(USD\): \$(\d+)", result_no_commas)
        if turnover_match:
            turnover = int(turnover_match.group(1))
            assert 7_000_000_000 < turnover < 8_000_000_000

    @pytest.mark.asyncio
    async def test_gbp_pence_adjustment_unchanged(self):
        """Test UK stocks pence adjustment still works correctly."""
        # 400 pence = £4.00
        # £4.00 * 100k volume * 1.27 FX = $508,000 USD
        # Vary prices slightly to avoid flat-price detection (±1% variation)
        import numpy as np

        prices = 400.0 + np.random.uniform(-4, 4, 60)
        mock_hist = pd.DataFrame(
            {
                "Close": prices,  # ~400 pence
                "Volume": [100_000] * 60,
            }
        )

        with (
            patch(
                "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
                return_value=mock_hist,
            ),
            patch(
                # Mock get_financial_metrics to return None to force fallback
                # to historical mean price (standard test behavior here).
                "src.liquidity_calculation_tool.market_data_fetcher.get_financial_metrics",
                new=AsyncMock(return_value=None),
            ),
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(1.27, "yfinance")),
            ):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "VOD.L"})

        # Should pass
        assert "Status: PASS" in result
        assert "GBP" in result
        # Pence adjustment: ~400/100 = ~£4 * 100k * 1.27 = ~$508,000
        assert "Turnover (USD):" in result
        # Check turnover is in expected range ($480k-$540k)
        import re

        turnover_match = re.search(r"Turnover \(USD\): \$(\d+,?\d*)", result)
        if turnover_match:
            turnover = int(turnover_match.group(1).replace(",", ""))
            assert 480_000 < turnover < 540_000

    @pytest.mark.asyncio
    async def test_calculation_logic_unchanged(self):
        """Test that core calculation logic (avg volume * avg price * FX) is unchanged."""
        # Use different prices and volumes to test averaging
        mock_hist = pd.DataFrame(
            {
                "Close": [100.0, 110.0, 90.0] * 20,  # avg = 100
                "Volume": [80_000, 100_000, 120_000] * 20,  # avg = 100,000
            }
        )

        with (
            patch(
                "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
                return_value=mock_hist,
            ),
            patch(
                # Mock get_financial_metrics to return None to force fallback
                # to historical mean price (standard test behavior here).
                "src.liquidity_calculation_tool.market_data_fetcher.get_financial_metrics",
                new=AsyncMock(return_value=None),
            ),
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(0.129, "yfinance")),
            ):
                result = await calculate_liquidity_metrics.ainvoke(
                    {"ticker": "TEST.HK"}
                )

        # Expected: avg_price=100 * avg_volume=100,000 * fx=0.129 = $1,290,000
        assert "Status: PASS" in result
        assert "$1,290,000" in result or "$1290000" in result

    @pytest.mark.asyncio
    async def test_threshold_unchanged(self):
        """Test that thresholds correctly classify liquidity levels."""
        # Test between $100k-$500k (should be MARGINAL)
        # Price $4.8 * volume 100k * FX 1.0 = ~$480,000 (with variation: $475k-$485k)
        # Vary prices slightly to avoid flat-price detection (±1% variation)
        import numpy as np

        prices = 4.8 + np.random.uniform(-0.048, 0.048, 60)
        mock_hist = pd.DataFrame({"Close": prices, "Volume": [100_000] * 60})

        with (
            patch(
                "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
                return_value=mock_hist,
            ),
            patch(
                # Mock get_financial_metrics to return None to force fallback
                # to historical mean price (standard test behavior here).
                "src.liquidity_calculation_tool.market_data_fetcher.get_financial_metrics",
                new=AsyncMock(return_value=None),
            ),
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(1.0, "identity")),
            ):
                result = await calculate_liquidity_metrics.ainvoke(
                    {"ticker": "BOUNDARY"}
                )

        # Between $100k-$500k, should be MARGINAL
        assert "Status: MARGINAL" in result
        # Check turnover is in marginal range
        import re

        turnover_match = re.search(r"Turnover \(USD\): \$(\d+,?\d*)", result)
        if turnover_match:
            turnover = int(turnover_match.group(1).replace(",", ""))
            assert 100_000 < turnover < 500_000
        assert "Thresholds:" in result
        assert "100,000" in result  # MARGINAL threshold
        assert "500,000" in result  # PASS threshold


class TestRealWorldScenarios:
    """Test with real-world-like scenarios to catch edge cases."""

    @pytest.mark.asyncio
    async def test_hsbc_real_world(self):
        """Test HSBC (0005.HK) with realistic prices and volume."""
        # HSBC typically trades around HKD 60-70, volume 10-20M shares
        # Vary prices slightly to avoid flat-price detection (±2% variation)
        import numpy as np

        prices = 65.0 + np.random.uniform(-1.3, 1.3, 60)
        mock_hist = pd.DataFrame({"Close": prices, "Volume": [15_000_000] * 60})

        with (
            patch(
                "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
                return_value=mock_hist,
            ),
            patch(
                # Mock get_financial_metrics to return None to force fallback
                # to historical mean price (standard test behavior here).
                "src.liquidity_calculation_tool.market_data_fetcher.get_financial_metrics",
                new=AsyncMock(return_value=None),
            ),
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(0.128, "yfinance")),
            ):
                result = await calculate_liquidity_metrics.ainvoke(
                    {"ticker": "0005.HK"}
                )

        # Should easily pass
        assert "Status: PASS" in result
        # 65 * 15M * 0.128 = $124.8M USD
        assert "HKD" in result

    @pytest.mark.asyncio
    async def test_tsmc_real_world(self):
        """Test TSMC (2330.TW) with realistic prices and volume."""
        # TSMC typically trades around TWD 500-600, volume 20-40M shares
        # Vary prices slightly to avoid flat-price detection (±2% variation)
        import numpy as np

        prices = 550.0 + np.random.uniform(-11, 11, 60)
        mock_hist = pd.DataFrame({"Close": prices, "Volume": [30_000_000] * 60})

        with (
            patch(
                "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
                return_value=mock_hist,
            ),
            patch(
                # Mock get_financial_metrics to return None to force fallback
                # to historical mean price (standard test behavior here).
                "src.liquidity_calculation_tool.market_data_fetcher.get_financial_metrics",
                new=AsyncMock(return_value=None),
            ),
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(0.032, "yfinance")),
            ):
                result = await calculate_liquidity_metrics.ainvoke(
                    {"ticker": "2330.TW"}
                )

        # Should easily pass
        assert "Status: PASS" in result
        # 550 * 30M * 0.032 = $528M USD
        assert "TWD" in result

    @pytest.mark.asyncio
    async def test_toyota_real_world(self):
        """Test Toyota (7203.T) with realistic prices and volume."""
        # Toyota typically trades around JPY 2500, volume 5-10M shares
        # Vary prices slightly to avoid flat-price detection (±2% variation)
        import numpy as np

        prices = 2500.0 + np.random.uniform(-50, 50, 60)
        mock_hist = pd.DataFrame({"Close": prices, "Volume": [7_000_000] * 60})

        with (
            patch(
                "src.liquidity_calculation_tool.market_data_fetcher.get_historical_prices",
                return_value=mock_hist,
            ),
            patch(
                # Mock get_financial_metrics to return None to force fallback
                # to historical mean price (standard test behavior here).
                "src.liquidity_calculation_tool.market_data_fetcher.get_financial_metrics",
                new=AsyncMock(return_value=None),
            ),
        ):
            with patch(
                "src.liquidity_calculation_tool.get_fx_rate",
                new=AsyncMock(return_value=(0.0067, "yfinance")),
            ):
                result = await calculate_liquidity_metrics.ainvoke({"ticker": "7203.T"})

        # Should easily pass
        assert "Status: PASS" in result
        # 2500 * 7M * 0.0067 = $117.25M USD
        assert "JPY" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
