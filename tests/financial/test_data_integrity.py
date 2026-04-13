"""
Regression tests for Data Integrity fixes:
1. Search-Based Ticker Resolution (Alpha -> Numeric)
2. 100x Scaling Error Normalization (Sen vs Ringgit / Pence vs Pound)
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.data.fetcher import SmartMarketDataFetcher
from src.ticker_policy import (
    allows_search_resolution,
    is_safe_symbol_crossmatch_base,
    normalize_exchange_specific_base,
    same_exchange,
    split_ticker,
)


@pytest.fixture
def fetcher():
    return SmartMarketDataFetcher()


class TestTickerResolution:
    """Regression tests for Search-Based Ticker Resolution (e.g. PADINI.KL -> 7052.KL)."""

    @pytest.mark.asyncio
    async def test_resolve_ticker_malaysia_success(self, fetcher):
        """Verify successful resolution of a Malaysian alpha ticker to numeric."""
        fetcher.tavily_client = MagicMock()
        # Mock Tavily return containing the mapping in the snippet
        fetcher.tavily_client.search = MagicMock(
            return_value={
                "results": [
                    {
                        "title": "Padini Holdings Berhad (7052.KL) Stock Price, News, Quote",
                        "content": "Padini Holdings Berhad is a Malaysia-based investment holding company... Bursa Malaysia ticker is 7052.KL",
                    }
                ]
            }
        )

        resolved = await fetcher._resolve_ticker_via_search("PADINI.KL")
        assert resolved == "7052.KL"

    @pytest.mark.asyncio
    async def test_resolve_ticker_hk_success(self, fetcher):
        """Verify successful resolution of a Hong Kong alpha ticker to numeric."""
        fetcher.tavily_client = MagicMock()
        fetcher.tavily_client.search = MagicMock(
            return_value={
                "results": [
                    {
                        "title": "Tencent Holdings Limited (0700.HK) Info",
                        "content": "Tencent is listed on HKEX under numeric code 0700.HK",
                    }
                ]
            }
        )

        resolved = await fetcher._resolve_ticker_via_search("TENCENT.HK")
        assert resolved == "0700.HK"

    @pytest.mark.asyncio
    async def test_resolve_ticker_non_asian_ignored(self, fetcher):
        """Ensure we don't attempt resolution for markets that don't use numeric tickers (e.g. US, UK)."""
        fetcher.tavily_client = MagicMock()

        # Should return None without searching
        resolved = await fetcher._resolve_ticker_via_search("AAPL")
        assert resolved is None
        assert not fetcher.tavily_client.search.called

    @pytest.mark.asyncio
    async def test_resolve_ticker_tokyo_disabled(self, fetcher):
        fetcher.tavily_client = MagicMock()

        resolved = await fetcher._resolve_ticker_via_search("262A.T")
        assert resolved is None
        assert not fetcher.tavily_client.search.called

    @pytest.mark.asyncio
    async def test_resolve_ticker_taiwan_requires_exact_suffix(self, fetcher):
        fetcher.tavily_client = MagicMock()
        fetcher.tavily_client.search = MagicMock(
            return_value={
                "results": [
                    {
                        "title": "Cross-listed examples",
                        "content": "2628.HK 2628.TW 2628.T appear together; correct TW code is 2628.TW",
                    }
                ]
            }
        )

        resolved = await fetcher._resolve_ticker_via_search("FOO.TW")
        assert resolved == "2628.TW"

    @pytest.mark.asyncio
    async def test_resolve_ticker_two_requires_exact_suffix(self, fetcher):
        fetcher.tavily_client = MagicMock()
        fetcher.tavily_client.search = MagicMock(
            return_value={
                "results": [
                    {
                        "title": "Cross-listed OTC examples",
                        "content": "1264.TWO and 1264.TW are different listings; exact OTC code is 1264.TWO",
                    }
                ]
            }
        )

        resolved = await fetcher._resolve_ticker_via_search("FOO.TWO")
        assert resolved == "1264.TWO"

    @pytest.mark.asyncio
    async def test_resolve_ticker_does_not_cross_exchange_on_shared_numeric(
        self, fetcher
    ):
        fetcher.tavily_client = MagicMock()
        fetcher.tavily_client.search = MagicMock(
            return_value={
                "results": [
                    {
                        "title": "Shared numeric codes",
                        "content": "2628.HK appears here, but 2628.T is a different Japanese listing.",
                    }
                ]
            }
        )

        resolved = await fetcher._resolve_ticker_via_search("FOO.TW")
        assert resolved is None


class TestTickerPolicy:
    def test_split_ticker_preserves_exchange(self):
        assert split_ticker("262A.T") == ("262A", ".T")
        assert split_ticker("0005.HK") == ("0005", ".HK")

    def test_same_exchange_requires_identical_suffix(self):
        assert same_exchange("2628.HK", "0700.HK") is True
        assert same_exchange("2628.HK", "2628.TW") is False
        assert same_exchange("2628.T", "2628.TW") is False
        assert same_exchange("1264.TWO", "1264.TW") is False

    def test_crossmatch_safe_only_for_non_numeric_bases(self):
        assert is_safe_symbol_crossmatch_base("262A") is True
        assert is_safe_symbol_crossmatch_base("CEK") is True
        assert is_safe_symbol_crossmatch_base("2628") is False

    def test_exchange_specific_normalization_only_pads_hk(self):
        assert normalize_exchange_specific_base("5", ".HK") == "0005"
        assert normalize_exchange_specific_base("262A", ".T") == "262A"
        assert normalize_exchange_specific_base("2330", ".TW") == "2330"

    def test_search_resolution_policy_is_exchange_specific(self):
        assert allows_search_resolution("TENCENT.HK") is True
        assert allows_search_resolution("PADINI.KL") is True
        assert allows_search_resolution("2330.TW") is True
        assert allows_search_resolution("1264.TWO") is True
        assert allows_search_resolution("262A.T") is False


class TestScalingCorrection:
    """Regression tests for 100x scaling error detection (Sen vs Ringgit, etc.)."""

    def test_normalize_sen_to_ringgit(self, fetcher):
        """Source A: 201.0 (Sen), Source B: 2.01 (Ringgit) -> Should return 2.01."""
        corrected = fetcher._normalize_scaling_errors(201.0, 2.01)
        assert corrected == 2.01

    def test_normalize_ringgit_to_sen(self, fetcher):
        """Source A: 2.01 (Ringgit), Source B: 201.0 (Sen) -> Should return 2.01."""
        corrected = fetcher._normalize_scaling_errors(2.01, 201.0)
        assert corrected == 2.01

    def test_no_correction_legit_difference(self, fetcher):
        """Legitimate variance (e.g. 10.0 vs 15.0) should not be touched."""
        corrected = fetcher._normalize_scaling_errors(10.0, 15.0)
        assert corrected == 15.0  # Returns the 'new' candidate value

    def test_normalize_scaling_errors_accepts_minor_unit_tolerance_band(self, fetcher):
        """Values within the shared 10% minor-unit tolerance should normalize."""
        corrected = fetcher._normalize_scaling_errors(95.0, 1.0)
        assert corrected == 1.0

    def test_smart_merge_integrates_scaling(self, fetcher):
        """Verify that _smart_merge_with_quality actually applies the correction."""
        # Setup source results to simulate a replace with scaling correction
        # Order: yahooquery (quality 6) -> fmp (quality 7)
        source_results = {
            "yahooquery": {"currentPrice": 2.01},
            "fmp": {"currentPrice": 201.0},
        }

        merged, meta = fetcher._smart_merge_with_quality(source_results, "TEST.KL")

        # FMP (201.0) should be corrected to match YahooQuery (2.01) because of 100x ratio
        # even though FMP has higher quality ranking.
        assert merged["currentPrice"] == 2.01
