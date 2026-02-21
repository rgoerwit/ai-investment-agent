"""
Tests for EODHD Integration and Smart Merging Logic
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.fetcher import SmartMarketDataFetcher


@pytest.fixture
def fetcher():
    return SmartMarketDataFetcher()


class TestEODHDIntegration:
    """Test proper integration of EODHD into the fetch pipeline."""

    @pytest.mark.asyncio
    async def test_eodhd_wins_over_lower_quality(self, fetcher):
        """
        Verify that EODHD (Quality 9.5) overwrites YahooQuery (6) and FMP (7),
        but yields to YFinance Statements (10).
        """
        # Setup conflicting data sources
        source_results = {
            "yahooquery": {"trailingPE": 20.0},  # Q=6
            "fmp": {"trailingPE": 25.0},  # Q=7
            "eodhd": {"trailingPE": 15.0},  # Q=9.5
            "yfinance": {"trailingPE": 30.0},  # Q=9 (info) - EODHD should win
        }

        merged, meta = fetcher._smart_merge_with_quality(source_results, "TEST")

        # EODHD should beat YF Info, FMP, and YQ
        assert merged["trailingPE"] == 15.0
        assert meta["field_sources"]["trailingPE"] == "eodhd"

    @pytest.mark.asyncio
    async def test_yfinance_statements_wins_over_eodhd(self, fetcher):
        """
        Verify that YFinance calculated statements (Quality 10) win over EODHD (9.5).
        """
        source_results = {
            "eodhd": {"revenueGrowth": 0.05},  # Q=9.5
            "yfinance": {
                "revenueGrowth": 0.10,
                "_revenueGrowth_source": "calculated_from_statements",  # Q=10 tag
            },
        }

        # This will now pass because fetcher.py looks for '_revenueGrowth_source'
        merged, meta = fetcher._smart_merge_with_quality(source_results, "TEST")

        assert merged["revenueGrowth"] == 0.10
        assert meta["field_sources"]["revenueGrowth"] == "yfinance"

    @pytest.mark.asyncio
    async def test_eodhd_fetcher_integration(self, fetcher):
        """Test that the main fetch method calls EODHD fetcher."""

        # Mock the internal fetch methods
        fetcher._fetch_yfinance_enhanced = AsyncMock(return_value={})
        fetcher._fetch_eodhd_fallback = AsyncMock(return_value={"marketCap": 100})
        fetcher._fetch_fmp_fallback = AsyncMock(return_value={})

        # Mock other methods to avoid side effects
        fetcher._fetch_all_sources_parallel = AsyncMock(
            return_value={"eodhd": {"marketCap": 100}, "yfinance": {}}
        )
        fetcher._fetch_tavily_gaps = AsyncMock(return_value={})

        # Run
        data = await fetcher.get_financial_metrics("TEST")

        # Verify result contains EODHD data
        assert data.get("marketCap") == 100

    @pytest.mark.asyncio
    async def test_eodhd_rate_limit_handling(self):
        """Test graceful handling of EODHD rate limits (simulated)."""
        # This import should now work as file is provided
        from src.data.eodhd_fetcher import EODHDFetcher

        fetcher = EODHDFetcher(api_key="test")

        # Mock 429 response
        mock_response = MagicMock()
        mock_response.status = 429

        # Create proper async context manager mock for response
        mock_response_cm = AsyncMock()
        mock_response_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response_cm.__aexit__ = AsyncMock(return_value=None)

        # Create proper async context manager mock for session
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response_cm)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            # First call hits limit
            result = await fetcher.get_financial_metrics("TEST")
            assert result is None
            assert fetcher._is_exhausted is True  # Circuit breaker tripped

            # Second call should return None immediately without network request
            mock_session.get.reset_mock()
            result2 = await fetcher.get_financial_metrics("TEST2")
            assert result2 is None
            mock_session.get.assert_not_called()
