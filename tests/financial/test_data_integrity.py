"""
Regression tests for Data Integrity fixes:
1. Search-Based Ticker Resolution (Alpha -> Numeric)
2. 100x Scaling Error Normalization (Sen vs Ringgit / Pence vs Pound)
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.data.fetcher import SmartMarketDataFetcher


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
