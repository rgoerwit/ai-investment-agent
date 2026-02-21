"""
Tests for Alpha Vantage Integration

Verifies that Alpha Vantage fetcher integrates correctly with the data pipeline.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.fetcher import SmartMarketDataFetcher


class TestAlphaVantageIntegration:
    """Test Alpha Vantage integration with SmartMarketDataFetcher."""

    @pytest.mark.asyncio
    @patch("src.data.fetcher.ALPHA_VANTAGE_AVAILABLE", True)
    @patch("src.data.fetcher.get_av_fetcher")
    async def test_av_fetcher_initialized_when_available(self, mock_get_av_fetcher):
        """Verify Alpha Vantage fetcher is initialized when available."""
        mock_av = MagicMock()
        mock_get_av_fetcher.return_value = mock_av

        fetcher = SmartMarketDataFetcher()

        assert fetcher.av_fetcher is not None
        assert fetcher.av_fetcher == mock_av
        mock_get_av_fetcher.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.data.fetcher.ALPHA_VANTAGE_AVAILABLE", False)
    async def test_av_fetcher_none_when_unavailable(self):
        """Verify Alpha Vantage fetcher is None when unavailable."""
        fetcher = SmartMarketDataFetcher()
        assert fetcher.av_fetcher is None

    @pytest.mark.asyncio
    @patch("src.data.fetcher.ALPHA_VANTAGE_AVAILABLE", True)
    @patch("src.data.fetcher.get_av_fetcher")
    async def test_av_fallback_method_checks_availability(self, mock_get_av_fetcher):
        """Verify _fetch_av_fallback checks circuit breaker before fetching."""
        mock_av = MagicMock()
        mock_av.is_available.return_value = False  # Circuit breaker tripped
        mock_get_av_fetcher.return_value = mock_av

        fetcher = SmartMarketDataFetcher()
        result = await fetcher._fetch_av_fallback("0005.HK")

        # Should return None without calling get_financial_metrics
        assert result is None
        mock_av.is_available.assert_called_once()
        mock_av.get_financial_metrics.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.data.fetcher.ALPHA_VANTAGE_AVAILABLE", True)
    @patch("src.data.fetcher.get_av_fetcher")
    async def test_av_fallback_fetches_data_when_available(self, mock_get_av_fetcher):
        """Verify _fetch_av_fallback successfully fetches when available."""
        mock_av = MagicMock()
        mock_av.is_available.return_value = True
        mock_av.get_financial_metrics = AsyncMock(
            return_value={
                "symbol": "0005.HK",
                "trailingPE": 8.5,
                "priceToBook": 0.9,
                "marketCap": 150e9,
                "_source": "alpha_vantage",
            }
        )
        mock_get_av_fetcher.return_value = mock_av

        fetcher = SmartMarketDataFetcher()
        result = await fetcher._fetch_av_fallback("0005.HK")

        assert result is not None
        assert result["symbol"] == "0005.HK"
        assert result["trailingPE"] == 8.5
        assert result["priceToBook"] == 0.9
        mock_av.get_financial_metrics.assert_called_once_with("0005.HK")
        assert fetcher.stats["sources"]["alpha_vantage"] == 1

    @pytest.mark.asyncio
    @patch("src.data.fetcher.ALPHA_VANTAGE_AVAILABLE", True)
    @patch("src.data.fetcher.get_av_fetcher")
    async def test_av_fallback_returns_none_on_empty_data(self, mock_get_av_fetcher):
        """Verify _fetch_av_fallback returns None when data is empty."""
        mock_av = MagicMock()
        mock_av.is_available.return_value = True
        mock_av.get_financial_metrics = AsyncMock(
            return_value={
                "_source": "alpha_vantage"  # Only metadata, no actual data
            }
        )
        mock_get_av_fetcher.return_value = mock_av

        fetcher = SmartMarketDataFetcher()
        result = await fetcher._fetch_av_fallback("0005.HK")

        # Should return None because no real data fields
        assert result is None
        assert fetcher.stats["sources"]["alpha_vantage"] == 0

    @pytest.mark.asyncio
    @patch("src.data.fetcher.ALPHA_VANTAGE_AVAILABLE", True)
    @patch("src.data.fetcher.get_av_fetcher")
    async def test_av_included_in_parallel_fetch(self, mock_get_av_fetcher):
        """Verify Alpha Vantage is included in parallel source fetching."""
        mock_av = MagicMock()
        mock_av.is_available.return_value = True
        mock_av.get_financial_metrics = AsyncMock(
            return_value={
                "symbol": "AAPL",
                "trailingPE": 25.0,
                "_source": "alpha_vantage",
            }
        )
        mock_get_av_fetcher.return_value = mock_av

        fetcher = SmartMarketDataFetcher()

        # Mock other sources
        with patch.object(
            fetcher, "_fetch_yfinance_enhanced", new=AsyncMock(return_value=None)
        ):
            with patch.object(fetcher, "_fetch_yahooquery_fallback", return_value=None):
                with patch.object(
                    fetcher, "_fetch_fmp_fallback", new=AsyncMock(return_value=None)
                ):
                    with patch.object(
                        fetcher,
                        "_fetch_eodhd_fallback",
                        new=AsyncMock(return_value=None),
                    ):
                        results = await fetcher._fetch_all_sources_parallel("AAPL")

        # Verify Alpha Vantage was included
        assert "alpha_vantage" in results
        assert results["alpha_vantage"] is not None
        assert results["alpha_vantage"]["trailingPE"] == 25.0

    @pytest.mark.asyncio
    @patch("src.data.fetcher.ALPHA_VANTAGE_AVAILABLE", True)
    @patch("src.data.fetcher.get_av_fetcher")
    async def test_av_data_merged_with_correct_quality_score(self, mock_get_av_fetcher):
        """Verify Alpha Vantage data is merged with quality score 9."""
        mock_av = MagicMock()
        mock_av.is_available.return_value = True
        mock_get_av_fetcher.return_value = mock_av

        fetcher = SmartMarketDataFetcher()

        # Simulate source results with different quality
        source_results = {
            "yahooquery": {"trailingPE": 30.0, "priceToBook": 1.5},  # Quality: 6
            "alpha_vantage": {"trailingPE": 25.0, "debtToEquity": 0.5},  # Quality: 9
            "yfinance": {"priceToBook": 1.8, "marketCap": 1e9},  # Quality: 9
        }

        merged, metadata = fetcher._smart_merge_with_quality(source_results, "TEST")

        # Alpha Vantage (quality 9) should override yahooquery (quality 6) for trailingPE
        assert merged["trailingPE"] == 25.0
        assert metadata["field_sources"]["trailingPE"] == "alpha_vantage"
        assert metadata["field_quality"]["trailingPE"] == 9

        # Alpha Vantage should provide debtToEquity
        assert merged["debtToEquity"] == 0.5
        assert metadata["field_sources"]["debtToEquity"] == "alpha_vantage"

        # yfinance should provide priceToBook (processed after alpha_vantage, same quality)
        assert merged["priceToBook"] == 1.8
        assert metadata["field_sources"]["priceToBook"] == "yfinance"


class TestAlphaVantageStatsTracking:
    """Test that Alpha Vantage usage is tracked in fetcher stats."""

    @pytest.mark.asyncio
    @patch("src.data.fetcher.ALPHA_VANTAGE_AVAILABLE", True)
    @patch("src.data.fetcher.get_av_fetcher")
    async def test_alpha_vantage_stats_initialized(self, mock_get_av_fetcher):
        """Verify alpha_vantage stats counter is initialized."""
        mock_get_av_fetcher.return_value = MagicMock()
        fetcher = SmartMarketDataFetcher()

        assert "alpha_vantage" in fetcher.stats["sources"]
        assert fetcher.stats["sources"]["alpha_vantage"] == 0

    @pytest.mark.asyncio
    @patch("src.data.fetcher.ALPHA_VANTAGE_AVAILABLE", True)
    @patch("src.data.fetcher.get_av_fetcher")
    async def test_alpha_vantage_stats_incremented_on_success(
        self, mock_get_av_fetcher
    ):
        """Verify alpha_vantage stats counter increments on successful fetch."""
        mock_av = MagicMock()
        mock_av.is_available.return_value = True
        mock_av.get_financial_metrics = AsyncMock(
            return_value={"symbol": "TEST", "trailingPE": 20.0}
        )
        mock_get_av_fetcher.return_value = mock_av

        fetcher = SmartMarketDataFetcher()
        await fetcher._fetch_av_fallback("TEST")

        assert fetcher.stats["sources"]["alpha_vantage"] == 1

    @pytest.mark.asyncio
    @patch("src.data.fetcher.ALPHA_VANTAGE_AVAILABLE", True)
    @patch("src.data.fetcher.get_av_fetcher")
    async def test_alpha_vantage_stats_not_incremented_on_failure(
        self, mock_get_av_fetcher
    ):
        """Verify alpha_vantage stats counter doesn't increment on failure."""
        mock_av = MagicMock()
        mock_av.is_available.return_value = False  # Circuit breaker
        mock_get_av_fetcher.return_value = mock_av

        fetcher = SmartMarketDataFetcher()
        await fetcher._fetch_av_fallback("TEST")

        assert fetcher.stats["sources"]["alpha_vantage"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
