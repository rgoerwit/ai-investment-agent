"""
Unit tests for StockTwits API wrapper.

Tests the basic functionality:
- API request construction
- Response parsing
- Sentiment calculation
- Error handling
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.stocktwits_api import StockTwitsAPI


class TestStockTwitsAPIInit:
    """Test StockTwitsAPI initialization."""

    def test_api_init(self):
        """Test basic API initialization."""
        api = StockTwitsAPI()

        assert api.BASE_URL == "https://api.stocktwits.com/api/2"

    def test_api_is_lightweight(self):
        """Test that API doesn't require heavy initialization."""
        api = StockTwitsAPI()

        # Should have no API key requirement
        assert not hasattr(api, "api_key") or api.api_key is None


class TestGetSentiment:
    """Test sentiment fetching and processing."""

    @pytest.mark.asyncio
    async def test_get_sentiment_success(self):
        """Test successful sentiment fetch with valid response."""
        api = StockTwitsAPI()

        # Mock response data
        mock_messages = [
            {
                "body": "Bullish on AAPL! Great earnings.",
                "user": {"username": "trader1"},
                "entities": {"sentiment": {"basic": "Bullish"}},
            },
            {
                "body": "AAPL looking weak, selling.",
                "user": {"username": "trader2"},
                "entities": {"sentiment": {"basic": "Bearish"}},
            },
            {
                "body": "Just bought more AAPL",
                "user": {"username": "trader3"},
                "entities": {},  # No sentiment tag
            },
        ]

        # 1. Mock the response object
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"messages": mock_messages})

        # 2. Mock the request context manager
        mock_request_cm = AsyncMock()
        mock_request_cm.__aenter__.return_value = mock_response
        mock_request_cm.__aexit__.return_value = None

        # 3. Mock the session context manager
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None

        # 4. Attach get method
        mock_session.get = MagicMock(return_value=mock_request_cm)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await api.get_sentiment("AAPL")

        assert result["source"] == "StockTwits"
        assert result["ticker"] == "AAPL"
        assert result["total_messages_last_30"] == 3
        assert result["bullish_count"] == 1
        assert result["bearish_count"] == 1
        assert result["bullish_pct"] == 50.0
        assert result["bearish_pct"] == 50.0
        assert len(result["messages"]) <= 3

    @pytest.mark.asyncio
    async def test_get_sentiment_strips_suffix(self):
        """Test that ticker suffixes are stripped for StockTwits."""
        api = StockTwitsAPI()

        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"messages": []})

        # Mock request CM
        mock_request_cm = AsyncMock()
        mock_request_cm.__aenter__.return_value = mock_response
        mock_request_cm.__aexit__.return_value = None

        # Mock session CM
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.get = MagicMock(return_value=mock_request_cm)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await api.get_sentiment("AAPL.US")

        # Check that the URL was constructed with clean ticker
        call_args = mock_session.get.call_args
        url = call_args[0][0]
        assert "AAPL.json" in url
        assert "AAPL.US" not in url

    @pytest.mark.asyncio
    async def test_get_sentiment_404_not_found(self):
        """Test handling of symbol not found (404)."""
        api = StockTwitsAPI()

        mock_response = MagicMock()
        mock_response.status = 404

        mock_request_cm = AsyncMock()
        mock_request_cm.__aenter__.return_value = mock_response
        mock_request_cm.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.get = MagicMock(return_value=mock_request_cm)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await api.get_sentiment("INVALID")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_sentiment_429_rate_limit(self):
        """Test handling of rate limit (429)."""
        api = StockTwitsAPI()

        mock_response = MagicMock()
        mock_response.status = 429

        mock_request_cm = AsyncMock()
        mock_request_cm.__aenter__.return_value = mock_response
        mock_request_cm.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.get = MagicMock(return_value=mock_request_cm)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await api.get_sentiment("AAPL")

        assert "error" in result
        assert "rate limit" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_sentiment_500_server_error(self):
        """Test handling of server errors (5xx)."""
        api = StockTwitsAPI()

        mock_response = MagicMock()
        mock_response.status = 500

        mock_request_cm = AsyncMock()
        mock_request_cm.__aenter__.return_value = mock_response
        mock_request_cm.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.get = MagicMock(return_value=mock_request_cm)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await api.get_sentiment("AAPL")

        assert "error" in result
        assert "500" in result["error"]

    @pytest.mark.asyncio
    async def test_get_sentiment_network_error(self):
        """Test handling of network errors."""
        api = StockTwitsAPI()

        # Use a context manager that raises on __aenter__
        # This simulates the error happening during the connection phase
        mock_request_cm = AsyncMock()
        mock_request_cm.__aenter__.side_effect = aiohttp.ClientError(
            "Connection failed"
        )
        mock_request_cm.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.get = MagicMock(return_value=mock_request_cm)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await api.get_sentiment("AAPL")

        assert "error" in result
        assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_get_sentiment_timeout(self):
        """Test handling of request timeout."""
        api = StockTwitsAPI()

        # Use a context manager that raises on __aenter__
        mock_request_cm = AsyncMock()
        mock_request_cm.__aenter__.side_effect = asyncio.TimeoutError()
        mock_request_cm.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.get = MagicMock(return_value=mock_request_cm)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await api.get_sentiment("AAPL")

        assert "error" in result


class TestProcessMessages:
    """Test message processing and sentiment calculation."""

    def test_process_messages_all_bullish(self):
        """Test processing messages that are all bullish."""
        api = StockTwitsAPI()

        messages = [
            {
                "body": "msg1",
                "user": {"username": "u1"},
                "entities": {"sentiment": {"basic": "Bullish"}},
            },
            {
                "body": "msg2",
                "user": {"username": "u2"},
                "entities": {"sentiment": {"basic": "Bullish"}},
            },
        ]

        result = api._process_messages(messages, "AAPL")

        assert result["bullish_count"] == 2
        assert result["bearish_count"] == 0
        assert result["bullish_pct"] == 100.0
        assert result["bearish_pct"] == 0.0

    def test_process_messages_all_bearish(self):
        """Test processing messages that are all bearish."""
        api = StockTwitsAPI()

        messages = [
            {
                "body": "msg1",
                "user": {"username": "u1"},
                "entities": {"sentiment": {"basic": "Bearish"}},
            },
            {
                "body": "msg2",
                "user": {"username": "u2"},
                "entities": {"sentiment": {"basic": "Bearish"}},
            },
        ]

        result = api._process_messages(messages, "AAPL")

        assert result["bullish_count"] == 0
        assert result["bearish_count"] == 2
        assert result["bullish_pct"] == 0.0
        assert result["bearish_pct"] == 100.0

    def test_process_messages_mixed_sentiment(self):
        """Test processing messages with mixed sentiment."""
        api = StockTwitsAPI()

        messages = [
            {
                "body": "msg1",
                "user": {"username": "u1"},
                "entities": {"sentiment": {"basic": "Bullish"}},
            },
            {
                "body": "msg2",
                "user": {"username": "u2"},
                "entities": {"sentiment": {"basic": "Bearish"}},
            },
            {
                "body": "msg3",
                "user": {"username": "u3"},
                "entities": {"sentiment": {"basic": "Bullish"}},
            },
        ]

        result = api._process_messages(messages, "AAPL")

        assert result["bullish_count"] == 2
        assert result["bearish_count"] == 1
        assert round(result["bullish_pct"], 1) == 66.7
        assert round(result["bearish_pct"], 1) == 33.3

    def test_process_messages_no_sentiment_tags(self):
        """Test processing messages with no sentiment tags."""
        api = StockTwitsAPI()

        messages = [
            {"body": "msg1", "user": {"username": "u1"}, "entities": {}},
            {"body": "msg2", "user": {"username": "u2"}, "entities": {}},
        ]

        result = api._process_messages(messages, "AAPL")

        assert result["bullish_count"] == 0
        assert result["bearish_count"] == 0
        assert result["bullish_pct"] == 0.0
        assert result["bearish_pct"] == 0.0

    def test_process_messages_empty_list(self):
        """Test processing empty message list."""
        api = StockTwitsAPI()

        result = api._process_messages([], "AAPL")

        assert result["total_messages_last_30"] == 0
        assert result["bullish_count"] == 0
        assert result["bearish_count"] == 0
        assert result["bullish_pct"] == 0.0
        assert result["bearish_pct"] == 0.0

    def test_process_messages_sample_limit(self):
        """Test that only first 3 messages are sampled."""
        api = StockTwitsAPI()

        messages = [
            {
                "body": f"msg{i}",
                "user": {"username": f"u{i}"},
                "entities": {"sentiment": {"basic": "Bullish"}},
            }
            for i in range(10)
        ]

        result = api._process_messages(messages, "AAPL")

        # Should only have 3 sample messages
        assert len(result["messages"]) == 3

    def test_process_messages_format(self):
        """Test that message samples are formatted correctly."""
        api = StockTwitsAPI()

        messages = [
            {
                "body": "Test message",
                "user": {"username": "testuser"},
                "entities": {"sentiment": {"basic": "Bullish"}},
            }
        ]

        result = api._process_messages(messages, "AAPL")

        sample = result["messages"][0]
        assert "[Bullish]" in sample
        assert "testuser" in sample
        assert "Test message" in sample

    def test_process_messages_missing_user(self):
        """Test processing messages with missing user info."""
        api = StockTwitsAPI()

        messages = [{"body": "Test", "entities": {"sentiment": {"basic": "Bullish"}}}]

        result = api._process_messages(messages, "AAPL")

        # Should handle gracefully
        assert result["bullish_count"] == 1
        assert len(result["messages"]) == 1


class TestURLConstruction:
    """Test URL construction."""

    @pytest.mark.asyncio
    async def test_url_format(self):
        """Test that URL is constructed correctly."""
        api = StockTwitsAPI()

        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"messages": []})

        # Mock request CM
        mock_request_cm = AsyncMock()
        mock_request_cm.__aenter__.return_value = mock_response
        mock_request_cm.__aexit__.return_value = None

        # Mock session CM
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.get = MagicMock(return_value=mock_request_cm)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await api.get_sentiment("AAPL")

        # Verify the URL format
        call_args = mock_session.get.call_args
        url = call_args[0][0]
        assert url == "https://api.stocktwits.com/api/2/streams/symbol/AAPL.json"

    @pytest.mark.asyncio
    async def test_user_agent_header(self):
        """Test that User-Agent header is set."""
        api = StockTwitsAPI()

        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"messages": []})

        # Mock request CM
        mock_request_cm = AsyncMock()
        mock_request_cm.__aenter__.return_value = mock_response
        mock_request_cm.__aexit__.return_value = None

        # Mock session CM
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.get = MagicMock(return_value=mock_request_cm)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await api.get_sentiment("AAPL")

        # Verify headers
        call_kwargs = mock_session.get.call_args[1]
        assert "headers" in call_kwargs
        assert "User-Agent" in call_kwargs["headers"]
