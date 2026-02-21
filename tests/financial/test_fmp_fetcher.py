"""
Unit tests for FMP (Financial Modeling Prep) API fetcher.

Tests basic functionality:
- API availability checking
- Request construction
- Response parsing
- Error handling (invalid key, rate limits, network errors)
- Data extraction from multiple endpoints
"""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.data.fmp_fetcher import FMPFetcher, get_fmp_fetcher


class TestFMPFetcherInit:
    """Test FMPFetcher initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        fetcher = FMPFetcher(api_key="test-key")

        assert fetcher.api_key == "test-key"
        assert fetcher.base_url == "https://financialmodelingprep.com/stable"
        assert fetcher._key_validated is False

    def test_init_from_environment(self):
        """Test initialization from environment variable via config."""
        # Pydantic Settings loads from .env at import time, so we mock the config getter
        with patch("src.data.fmp_fetcher.config") as mock_config:
            mock_config.get_fmp_api_key.return_value = "env-key"
            fetcher = FMPFetcher()

            assert fetcher.api_key == "env-key"

    def test_init_no_api_key(self):
        """Test initialization without API key."""
        # Mock config to return None (simulates no FMP_API_KEY in environment)
        with patch("src.data.fmp_fetcher.config") as mock_config:
            mock_config.get_fmp_api_key.return_value = None
            fetcher = FMPFetcher()

            assert fetcher.api_key is None

    def test_is_available_with_key(self):
        """Test that is_available returns True when key is present."""
        fetcher = FMPFetcher(api_key="test-key")

        assert fetcher.is_available() is True

    def test_is_available_without_key(self):
        """Test that is_available returns False when key is missing."""
        # Explicit None bypasses config lookup
        fetcher = FMPFetcher(api_key=None)
        # Force api_key to None (bypasses config default behavior)
        fetcher.api_key = None

        assert fetcher.is_available() is False


class TestFMPGet:
    """Test the internal _get method."""

    @pytest.mark.asyncio
    async def test_get_success(self):
        """Test successful API request."""
        fetcher = FMPFetcher(api_key="test-key")

        mock_data = [{"pe": 15.5, "pb": 2.0}]
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_data)

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
            result = await fetcher._get("ratios", {"symbol": "AAPL", "limit": 1})

        assert result == mock_data
        assert fetcher._key_validated is True

    @pytest.mark.asyncio
    async def test_get_without_api_key(self):
        """Test that _get returns None when API key is not available."""
        # Create fetcher with explicit None
        fetcher = FMPFetcher(api_key=None)
        # Force api_key to None (bypasses config default behavior)
        fetcher.api_key = None

        # Early return when no API key - never hits the network
        result = await fetcher._get("ratios", {"symbol": "AAPL"})

        assert result is None

    @pytest.mark.asyncio
    async def test_get_403_invalid_key(self):
        """Test handling of 403 with invalid API key."""
        fetcher = FMPFetcher(api_key="invalid-key")

        mock_response = MagicMock()
        mock_response.status = 403

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
            with pytest.raises(ValueError, match="FMP_API_KEY is invalid"):
                await fetcher._get("ratios", {"symbol": "AAPL"})

    @pytest.mark.asyncio
    async def test_get_403_rate_limit_after_validation(self):
        """Test handling of 403 after key was previously validated."""
        fetcher = FMPFetcher(api_key="test-key")
        fetcher._key_validated = True  # Key was previously validated

        mock_response = MagicMock()
        mock_response.status = 403

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
            # Should return None instead of raising
            result = await fetcher._get("ratios", {"symbol": "AAPL"})
            assert result is None

    @pytest.mark.asyncio
    async def test_get_404_not_found(self):
        """Test handling of 404 (data not found)."""
        fetcher = FMPFetcher(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status = 404

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
            result = await fetcher._get("ratios", {"symbol": "INVALID"})
            assert result is None

    @pytest.mark.asyncio
    async def test_get_network_error(self):
        """Test handling of network errors."""
        fetcher = FMPFetcher(api_key="test-key")

        # Create session mock that raises error when entering context
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientError("Network failed")
        )

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await fetcher._get("ratios", {"symbol": "AAPL"})
            assert result is None

    @pytest.mark.asyncio
    async def test_get_adds_api_key_to_params(self):
        """Test that API key is added to request parameters."""
        fetcher = FMPFetcher(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[])

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
            await fetcher._get("ratios", {"symbol": "AAPL"})

            # Verify API key was added to params
            call_args = mock_session.get.call_args
            params = call_args[1]["params"]
            assert params["apikey"] == "test-key"


class TestGetFinancialMetrics:
    """Test comprehensive financial metrics fetching."""

    @pytest.mark.asyncio
    async def test_get_financial_metrics_all_sources(self):
        """Test fetching from all three endpoints."""
        fetcher = FMPFetcher(api_key="test-key")

        # Mock ratios endpoint
        ratios_data = [
            {
                "priceToEarningsRatio": 25.5,
                "priceToBookRatio": 10.2,
                "priceToEarningsGrowthRatio": 1.5,
                "currentRatio": 1.2,
                "debtToEquityRatio": 1.8,
                "netProfitMargin": 0.25,
                "freeCashFlowPerShare": 5.5,
                "operatingCashFlowPerShare": 6.5,
            }
        ]

        # Mock key-metrics endpoint
        metrics_data = [
            {"returnOnEquity": 0.45, "returnOnAssets": 0.18, "marketCap": 1000000}
        ]

        # Mock growth endpoint
        growth_data = [{"growthRevenue": 0.15, "growthEPS": 0.20}]

        # Create a mock that returns different data based on endpoint
        async def mock_get(endpoint, params):
            if endpoint == "ratios":
                return ratios_data
            elif endpoint == "key-metrics":
                return metrics_data
            elif endpoint == "income-statement-growth":
                return growth_data
            return None

        fetcher._get = AsyncMock(side_effect=mock_get)

        result = await fetcher.get_financial_metrics("AAPL")

        # Verify all fields are populated with correct keys
        assert result["trailingPE"] == 25.5
        assert result["priceToBook"] == 10.2
        assert result["pegRatio"] == 1.5
        assert result["returnOnEquity"] == 0.45
        assert result["returnOnAssets"] == 0.18
        assert result["currentRatio"] == 1.2
        assert result["debtToEquity"] == 1.8
        assert result["profitMargins"] == 0.25
        assert result["revenueGrowth"] == 0.15
        assert result["earningsGrowth"] == 0.20
        assert result["freeCashflow"] == 5.5
        assert result["operatingCashflow"] == 6.5
        assert result["marketCap"] == 1000000
        assert result["_source"] == "fmp"

    @pytest.mark.asyncio
    async def test_get_financial_metrics_partial_data(self):
        """Test handling of partial data from endpoints."""
        fetcher = FMPFetcher(api_key="test-key")

        # Only ratios endpoint returns data
        async def mock_get(endpoint, params):
            if endpoint == "ratios":
                return [{"priceToEarningsRatio": 15.0}]
            return None

        fetcher._get = AsyncMock(side_effect=mock_get)

        result = await fetcher.get_financial_metrics("AAPL")

        # PE should be set, others should be missing
        assert result["trailingPE"] == 15.0
        assert "returnOnEquity" not in result
        assert "revenueGrowth" not in result

    @pytest.mark.asyncio
    async def test_get_financial_metrics_no_data(self):
        """Test handling when no data is available."""
        fetcher = FMPFetcher(api_key="test-key")
        fetcher._get = AsyncMock(return_value=None)

        result = await fetcher.get_financial_metrics("INVALID")

        # Should return None when no data is found
        assert result is None

    @pytest.mark.asyncio
    async def test_get_financial_metrics_empty_arrays(self):
        """Test handling of empty array responses."""
        fetcher = FMPFetcher(api_key="test-key")
        fetcher._get = AsyncMock(return_value=[])

        result = await fetcher.get_financial_metrics("AAPL")

        # Should handle empty arrays gracefully and return None
        assert result is None

    @pytest.mark.asyncio
    async def test_get_financial_metrics_missing_fields(self):
        """Test handling of responses with missing fields."""
        fetcher = FMPFetcher(api_key="test-key")

        # Response has some fields but not others
        async def mock_get(endpoint, params):
            if endpoint == "ratios":
                return [{"priceToEarningsRatio": 20.0}]  # Only PE
            return None

        fetcher._get = AsyncMock(side_effect=mock_get)

        result = await fetcher.get_financial_metrics("AAPL")

        assert result["trailingPE"] == 20.0
        assert result.get("priceToBook") is None  # Missing from response


class TestGlobalFetcher:
    """Test the global fetcher singleton."""

    def test_get_fmp_fetcher_returns_instance(self):
        """Test that get_fmp_fetcher returns an FMPFetcher instance."""
        fetcher = get_fmp_fetcher()

        assert isinstance(fetcher, FMPFetcher)

    def test_get_fmp_fetcher_singleton(self):
        """Test that get_fmp_fetcher returns the same instance."""
        fetcher1 = get_fmp_fetcher()
        fetcher2 = get_fmp_fetcher()

        assert fetcher1 is fetcher2


class TestConvenienceFunction:
    """Test the fetch_fmp_metrics convenience function."""

    @pytest.mark.asyncio
    @patch("src.data.fmp_fetcher.get_fmp_fetcher")
    async def test_fetch_fmp_metrics_available(self, mock_get_fetcher):
        """Test convenience function when FMP is available."""
        from src.data.fmp_fetcher import fetch_fmp_metrics

        # Create proper mock - is_available() is NOT async
        mock_fetcher = MagicMock()
        mock_fetcher.is_available = MagicMock(
            return_value=True
        )  # Regular mock, not async
        mock_fetcher.get_financial_metrics = AsyncMock(
            return_value={"trailingPE": 15.0, "_source": "fmp"}
        )

        mock_get_fetcher.return_value = mock_fetcher

        result = await fetch_fmp_metrics("AAPL")

        assert result["trailingPE"] == 15.0
        mock_fetcher.get_financial_metrics.assert_called_once_with("AAPL")

    @pytest.mark.asyncio
    @patch("src.data.fmp_fetcher.get_fmp_fetcher")
    async def test_fetch_fmp_metrics_unavailable(self, mock_get_fetcher):
        """Test convenience function when FMP is not available."""
        from src.data.fmp_fetcher import fetch_fmp_metrics

        mock_fetcher = MagicMock()
        mock_fetcher.is_available.return_value = False
        mock_get_fetcher.return_value = mock_fetcher

        result = await fetch_fmp_metrics("AAPL")

        assert result is None


class TestErrorScenarios:
    """Test various error scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_json_response(self):
        """Test handling of invalid JSON response.

        Updated: After malformed JSON handling was added, the error no longer
        propagates - it returns None and logs at DEBUG level.
        """
        fetcher = FMPFetcher(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=ValueError("Invalid JSON"))

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
            # Updated: With malformed JSON handling, it should return None gracefully
            result = await fetcher._get("ratios", {"symbol": "AAPL"})
            assert result is None

    @pytest.mark.asyncio
    async def test_unexpected_response_structure(self):
        """Test handling of unexpected response structure."""
        fetcher = FMPFetcher(api_key="test-key")

        # Response is not a list as expected
        fetcher._get = AsyncMock(return_value={"error": "something"})

        result = await fetcher.get_financial_metrics("AAPL")

        # Should handle gracefully and return None (no valid data found)
        assert result is None
