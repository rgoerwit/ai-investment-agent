"""
Comprehensive Test Suite for Currency Normalization

Tests cover:
1. Basic FX rate fetching (yfinance, fallback, graceful failure)
2. Single value normalization (with metadata tracking)
3. Full dict normalization (selective field conversion)
4. Edge cases (None values, missing currencies, zero values)
5. Integration with financial data formats (yfinance/FMP schemas)
"""

from unittest.mock import patch

import pytest

from src.fx_normalization import (
    FALLBACK_RATES_TO_USD,
    get_fx_rate,
    get_fx_rate_fallback,
    get_fx_rate_yfinance,
    normalize_financial_dict,
    normalize_to_usd,
)

# ══════════════════════════════════════════════════════════════════════════════
# TIER 1: FX Rate Fetching Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestFXRateFetching:
    """Test FX rate fetching from yfinance and fallback sources."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_fx_rate_yfinance_jpy(self):
        """Test fetching JPY → USD rate from yfinance (requires network)."""
        rate = await get_fx_rate_yfinance("JPY", "USD")

        # JPY rate should be around 0.0067 (¥150 = $1)
        # Allow wide range for market fluctuations
        assert rate is not None
        assert 0.005 < rate < 0.01, f"JPY rate {rate} out of expected range"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_fx_rate_yfinance_hkd(self):
        """Test fetching HKD → USD rate from yfinance."""
        rate = await get_fx_rate_yfinance("HKD", "USD")

        # HKD is pegged ~7.80:1, so rate ~0.128
        assert rate is not None
        assert 0.12 < rate < 0.14, f"HKD rate {rate} out of expected range"

    @pytest.mark.asyncio
    async def test_get_fx_rate_yfinance_identity(self):
        """Test USD → USD returns 1.0."""
        rate = await get_fx_rate_yfinance("USD", "USD")
        assert rate == 1.0

    @pytest.mark.asyncio
    async def test_get_fx_rate_yfinance_invalid_currency(self):
        """Test invalid currency returns None gracefully."""
        rate = await get_fx_rate_yfinance("ZZZ", "USD")
        assert rate is None

    def test_get_fx_rate_fallback_jpy(self):
        """Test fallback rate lookup for JPY."""
        rate = get_fx_rate_fallback("JPY", "USD")
        assert rate is not None
        assert rate == FALLBACK_RATES_TO_USD["JPY"]

    def test_get_fx_rate_fallback_identity(self):
        """Test fallback USD → USD returns 1.0."""
        rate = get_fx_rate_fallback("USD", "USD")
        assert rate == 1.0

    def test_get_fx_rate_fallback_missing_currency(self):
        """Test fallback returns None for unsupported currency."""
        rate = get_fx_rate_fallback("ZZZ", "USD")
        assert rate is None

    @pytest.mark.asyncio
    async def test_get_fx_rate_unified_yfinance_success(self):
        """Test unified interface uses yfinance when available."""
        rate, source = await get_fx_rate("JPY", "USD")

        # Should succeed with yfinance (or fallback if network issue)
        assert rate is not None
        assert source in ["yfinance", "fallback"]
        assert 0.005 < rate < 0.01

    @pytest.mark.asyncio
    async def test_get_fx_rate_unified_fallback_to_hardcoded(self):
        """Test unified interface falls back to hardcoded rates."""
        # Mock yfinance failure
        with patch("src.fx_normalization.get_fx_rate_yfinance", return_value=None):
            rate, source = await get_fx_rate("JPY", "USD", allow_fallback=True)

            assert rate is not None
            assert source == "fallback"
            assert rate == FALLBACK_RATES_TO_USD["JPY"]

    @pytest.mark.asyncio
    async def test_get_fx_rate_unified_no_fallback(self):
        """Test unified interface respects allow_fallback=False."""
        # Mock yfinance failure
        with patch("src.fx_normalization.get_fx_rate_yfinance", return_value=None):
            rate, source = await get_fx_rate("ZZZ", "USD", allow_fallback=False)

            assert rate is None
            assert source == "unavailable"

    @pytest.mark.asyncio
    async def test_get_fx_rate_case_insensitive(self):
        """Test currency codes are case-insensitive."""
        rate1, _ = await get_fx_rate("jpy", "usd")
        rate2, _ = await get_fx_rate("JPY", "USD")
        rate3, _ = await get_fx_rate("JpY", "UsD")

        assert rate1 == rate2 == rate3


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2: Single Value Normalization Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalizeToUSD:
    """Test single value normalization with metadata tracking."""

    @pytest.mark.asyncio
    async def test_normalize_to_usd_jpy(self):
        """Test JPY → USD normalization."""
        value_jpy = 1000000  # 1 million yen
        value_usd, metadata = await normalize_to_usd(value_jpy, "JPY", "test_value")

        # Should convert to ~$6,700 (using live or fallback rate)
        assert value_usd is not None
        assert 5000 < value_usd < 10000, f"Converted value {value_usd} out of range"

        # Check metadata
        assert metadata["original_value"] == value_jpy
        assert metadata["original_currency"] == "JPY"
        assert metadata["fx_rate"] is not None
        assert metadata["fx_source"] in ["yfinance", "fallback"]
        assert metadata["normalized"] is True

    @pytest.mark.asyncio
    async def test_normalize_to_usd_identity(self):
        """Test USD → USD returns same value."""
        value = 100.0
        result, metadata = await normalize_to_usd(value, "USD")

        assert result == value
        assert metadata["normalized"] is False
        assert metadata["fx_rate"] == 1.0
        assert metadata["fx_source"] == "identity"

    @pytest.mark.asyncio
    async def test_normalize_to_usd_none_value(self):
        """Test None value returns None gracefully."""
        result, metadata = await normalize_to_usd(None, "JPY")

        assert result is None
        assert metadata["original_value"] is None
        assert metadata["normalized"] is False

    @pytest.mark.asyncio
    async def test_normalize_to_usd_zero_value(self):
        """Test zero value normalizes correctly."""
        result, metadata = await normalize_to_usd(0.0, "JPY")

        assert result == 0.0
        assert metadata["normalized"] is True  # Conversion was applied
        assert metadata["fx_rate"] is not None

    @pytest.mark.asyncio
    async def test_normalize_to_usd_negative_value(self):
        """Test negative value normalizes correctly (e.g., losses)."""
        result, metadata = await normalize_to_usd(-1000000, "JPY")

        # Should convert to negative USD value
        assert result is not None
        assert result < 0
        assert -10000 < result < -5000

    @pytest.mark.asyncio
    async def test_normalize_to_usd_unavailable_currency(self):
        """Test unavailable currency returns original value with warning."""
        value = 100.0

        # Mock both yfinance and fallback failure
        with patch("src.fx_normalization.get_fx_rate_yfinance", return_value=None):
            with patch("src.fx_normalization.get_fx_rate_fallback", return_value=None):
                result, metadata = await normalize_to_usd(value, "ZZZ")

                # Should return original value
                assert result == value
                assert metadata["fx_source"] == "unavailable"
                assert metadata["normalized"] is False


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3: Full Dict Normalization Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalizeFinancialDict:
    """Test full financial dict normalization (selective field conversion)."""

    @pytest.mark.asyncio
    async def test_normalize_dict_basic_conversion(self):
        """Test basic dict normalization with HKD data."""
        data = {
            "market_cap": 1.2e12,  # Should convert
            "revenue_ttm": 500e9,  # Should convert
            "pe": 12.5,  # Should NOT convert (ratio)
            "profit_margin": 0.15,  # Should NOT convert (percentage)
            "currency": "HKD",
        }

        # Save original values (dict is modified in place)
        original_market_cap = data["market_cap"]
        original_revenue = data["revenue_ttm"]

        result = await normalize_financial_dict(data)

        # Check currency-dependent fields were converted (HKD rate ~0.128)
        assert result["market_cap"] < original_market_cap  # Smaller in USD
        assert result["market_cap"] < original_market_cap * 0.2  # Much smaller
        assert result["revenue_ttm"] < original_revenue * 0.2

        # Check ratios/percentages unchanged
        assert result["pe"] == 12.5
        assert result["profit_margin"] == 0.15

        # Check currency updated
        assert result["currency"] == "USD"

        # Check metadata added
        assert result["_currency_normalized"] is True
        assert result["_original_currency"] == "HKD"
        assert result["_fx_rate_applied"] is not None
        assert result["_fx_source"] in ["yfinance", "fallback"]

    @pytest.mark.asyncio
    async def test_normalize_dict_already_usd(self):
        """Test dict already in USD is not modified."""
        data = {"market_cap": 100e9, "currency": "USD"}

        result = await normalize_financial_dict(data)

        assert result["market_cap"] == 100e9  # Unchanged
        assert result["_currency_normalized"] is False
        assert result["_original_currency"] == "USD"

    @pytest.mark.asyncio
    async def test_normalize_dict_yfinance_schema(self):
        """Test normalization with yfinance field names."""
        data = {
            "marketCap": 1.2e12,  # yfinance uses camelCase
            "totalRevenue": 500e9,
            "freeCashflow": 50e9,
            "currency": "HKD",
        }

        # Save original values
        original_market_cap = data["marketCap"]
        original_revenue = data["totalRevenue"]
        original_fcf = data["freeCashflow"]

        result = await normalize_financial_dict(data)

        # Check yfinance-style fields were converted (should be ~12.8% of original for HKD)
        assert result["marketCap"] < original_market_cap
        assert (
            result["marketCap"] < original_market_cap * 0.2
        )  # Much smaller than original
        assert result["totalRevenue"] < original_revenue * 0.2
        assert result["freeCashflow"] < original_fcf * 0.2

    @pytest.mark.asyncio
    async def test_normalize_dict_mixed_fields(self):
        """Test dict with both snake_case and camelCase fields."""
        data = {
            "market_cap": 1e12,  # snake_case (FMP style)
            "totalRevenue": 500e9,  # camelCase (yfinance style)
            "currency": "JPY",
        }

        # Save original values
        original_market_cap = data["market_cap"]
        original_revenue = data["totalRevenue"]

        result = await normalize_financial_dict(data)

        # Both should be converted (JPY rate ~0.0067, so much smaller)
        assert result["market_cap"] < original_market_cap
        assert result["market_cap"] < original_market_cap * 0.01  # Much smaller
        assert result["totalRevenue"] < original_revenue * 0.01

    @pytest.mark.asyncio
    async def test_normalize_dict_preserves_nones(self):
        """Test None values are preserved during normalization."""
        data = {"market_cap": None, "revenue_ttm": 500e9, "currency": "JPY"}

        # Save original value
        original_revenue = data["revenue_ttm"]

        result = await normalize_financial_dict(data)

        # None should remain None
        assert result["market_cap"] is None
        # Non-None should convert (JPY rate ~0.0067)
        assert result["revenue_ttm"] < original_revenue
        assert result["revenue_ttm"] < original_revenue * 0.01

    @pytest.mark.asyncio
    async def test_normalize_dict_fx_unavailable(self):
        """Test dict normalization when FX rate unavailable."""
        data = {
            "market_cap": 100e9,
            "currency": "ZZZ",  # Invalid currency
        }

        # Mock FX failure
        with patch(
            "src.fx_normalization.get_fx_rate", return_value=(None, "unavailable")
        ):
            result = await normalize_financial_dict(data)

            # Values should remain unchanged
            assert result["market_cap"] == data["market_cap"]
            assert result["_currency_normalized"] is False
            assert result["_fx_source"] == "unavailable"

    @pytest.mark.asyncio
    async def test_normalize_dict_no_currency_field(self):
        """Test dict without currency field defaults to USD."""
        data = {
            "market_cap": 100e9
            # No "currency" field
        }

        result = await normalize_financial_dict(data)

        # Should assume USD and skip normalization
        assert result["market_cap"] == 100e9
        assert result["_currency_normalized"] is False


# ══════════════════════════════════════════════════════════════════════════════
# TIER 4: Integration Tests (Realistic Scenarios)
# ══════════════════════════════════════════════════════════════════════════════


class TestFXNormalizationIntegration:
    """Test normalization with realistic financial data structures."""

    @pytest.mark.asyncio
    async def test_hsbc_hong_kong_normalization(self):
        """Test realistic HSBC (0005.HK) data normalization."""
        hsbc_data = {
            "market_cap": 1.2e12,  # ~$154B USD
            "revenue_ttm": 500e9,
            "free_cash_flow": 50e9,
            "pe": 12.5,
            "pb": 0.8,
            "profit_margin": 0.15,
            "debt_to_equity": 1.2,
            "currency": "HKD",
        }

        result = await normalize_financial_dict(hsbc_data)

        # Market cap should be ~$154B
        assert 140e9 < result["market_cap"] < 170e9

        # Ratios should be unchanged
        assert result["pe"] == 12.5
        assert result["pb"] == 0.8
        assert result["debt_to_equity"] == 1.2

        # Currency should be updated
        assert result["currency"] == "USD"
        assert result["_original_currency"] == "HKD"

    @pytest.mark.asyncio
    async def test_toyota_japan_normalization(self):
        """Test realistic Toyota (7203.T) data normalization."""
        toyota_data = {
            "marketCap": 45e12,  # yfinance style, ~$300B USD
            "totalRevenue": 40e12,
            "freeCashflow": 5e12,
            "trailingPE": 9.8,
            "priceToBook": 0.9,
            "currency": "JPY",
        }

        result = await normalize_financial_dict(toyota_data)

        # Market cap should be ~$300B
        assert 250e9 < result["marketCap"] < 350e9

        # Ratios unchanged
        assert result["trailingPE"] == 9.8
        assert result["priceToBook"] == 0.9

    @pytest.mark.asyncio
    async def test_tsmc_taiwan_normalization(self):
        """Test realistic TSMC (2330.TW) data normalization."""
        tsmc_data = {
            "market_cap": 16e12,  # ~$520B USD
            "revenue_ttm": 2e12,
            "currency": "TWD",
        }

        result = await normalize_financial_dict(tsmc_data)

        # Market cap should be ~$520B
        assert 450e9 < result["market_cap"] < 600e9

    @pytest.mark.asyncio
    async def test_cross_border_comparison(self):
        """Test that normalization enables apples-to-apples comparison."""
        # Before normalization: looks like TSMC >> Toyota >> HSBC
        hsbc = {"market_cap": 1.2e12, "currency": "HKD"}
        toyota = {"market_cap": 45e12, "currency": "JPY"}
        tsmc = {"market_cap": 16e12, "currency": "TWD"}

        # Normalize all
        hsbc_usd = await normalize_financial_dict(hsbc)
        toyota_usd = await normalize_financial_dict(toyota)
        tsmc_usd = await normalize_financial_dict(tsmc)

        # After normalization: TSMC > Toyota > HSBC (correct order)
        assert (
            tsmc_usd["market_cap"] > toyota_usd["market_cap"] > hsbc_usd["market_cap"]
        )


# ══════════════════════════════════════════════════════════════════════════════
# TIER 5: Edge Case Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestFXNormalizationEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_very_large_numbers(self):
        """Test normalization handles very large market caps."""
        # Apple-scale: ~$3T USD
        data = {"market_cap": 3e12, "currency": "USD"}
        result = await normalize_financial_dict(data)
        assert result["market_cap"] == 3e12

    @pytest.mark.asyncio
    async def test_very_small_numbers(self):
        """Test normalization handles micro-cap stocks."""
        data = {"market_cap": 10e6, "currency": "USD"}  # $10M
        result = await normalize_financial_dict(data)
        assert result["market_cap"] == 10e6

    @pytest.mark.asyncio
    async def test_scientific_notation(self):
        """Test normalization handles scientific notation."""
        data = {"market_cap": 1.5e11, "currency": "JPY"}
        result = await normalize_financial_dict(data)
        assert result["market_cap"] < 1.5e11  # Should convert

    @pytest.mark.asyncio
    async def test_whitespace_in_currency_code(self):
        """Test currency codes with whitespace are normalized."""
        data = {"market_cap": 100e9, "currency": " HKD "}
        result = await normalize_financial_dict(data)
        assert result["_original_currency"] == "HKD"  # Stripped

    @pytest.mark.asyncio
    async def test_lowercase_currency_code(self):
        """Test lowercase currency codes work."""
        data = {"market_cap": 100e9, "currency": "hkd"}
        result = await normalize_financial_dict(data)
        assert result["_currency_normalized"] is True

    @pytest.mark.asyncio
    async def test_dict_with_extra_fields(self):
        """Test dict with non-financial fields is not broken."""
        data = {
            "market_cap": 100e9,
            "currency": "JPY",
            "ticker": "7203.T",  # Should be preserved
            "company_name": "Toyota",  # Should be preserved
            "_source": "yfinance",  # Metadata field should be preserved
        }

        result = await normalize_financial_dict(data)

        assert result["ticker"] == "7203.T"
        assert result["company_name"] == "Toyota"
        assert result["_source"] == "yfinance"

    @pytest.mark.asyncio
    async def test_string_numbers_cause_no_crash(self):
        """Test string values don't cause crashes."""
        data = {"market_cap": "not a number", "currency": "JPY"}

        # Should not crash, but won't convert the field
        result = await normalize_financial_dict(data)
        assert result["market_cap"] == "not a number"  # Unchanged


# ══════════════════════════════════════════════════════════════════════════════
# TIER 6: Performance Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestFXNormalizationPerformance:
    """Test performance characteristics of FX normalization."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_batch_normalization_performance(self):
        """Test normalizing 10 stocks in reasonable time."""
        import time

        test_stocks = [
            {"market_cap": 1e12, "currency": "HKD"},
            {"market_cap": 45e12, "currency": "JPY"},
            {"market_cap": 16e12, "currency": "TWD"},
            {"market_cap": 100e9, "currency": "EUR"},
            {"market_cap": 50e9, "currency": "GBP"},
            {"market_cap": 80e9, "currency": "CHF"},
            {"market_cap": 120e9, "currency": "CAD"},
            {"market_cap": 90e9, "currency": "AUD"},
            {"market_cap": 200e9, "currency": "KRW"},
            {"market_cap": 150e9, "currency": "SGD"},
        ]

        start = time.time()
        results = [await normalize_financial_dict(stock) for stock in test_stocks]
        elapsed = time.time() - start

        # Should complete in <10 seconds (generous - likely <5s with caching)
        assert elapsed < 10.0, f"Batch normalization took {elapsed:.2f}s (too slow)"
        assert len(results) == 10
        assert all(r["_currency_normalized"] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
