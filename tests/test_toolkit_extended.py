"""Extended toolkit tests focusing on AI/API failure modes and edge cases."""

import json
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src import toolkit


@pytest.fixture
def mock_fetcher():
    """Mock the singleton market_data_fetcher."""
    with patch("src.toolkit.market_data_fetcher") as mock:
        yield mock


@pytest.fixture
def mock_tavily():
    """Mock tavily_tool."""
    with patch("src.toolkit.tavily_tool") as mock:
        mock.ainvoke = AsyncMock()
        yield mock


@pytest.fixture
def mock_stocktwits():
    """
    Mock stocktwits_api.
    """
    with patch("src.toolkit.stocktwits_api") as mock:
        # Setup get_sentiment to be awaitable (AsyncMock)
        mock.get_sentiment = AsyncMock()
        yield mock


# ==================== AI RESPONSE MALFORMATION TESTS ====================


@pytest.mark.asyncio
class TestAIResponseRobustness:
    """Tests for handling malformed AI outputs and API failures."""

    async def test_news_malformed_json(self, mock_tavily):
        """Test handling of malformed JSON from news API."""
        mock_tavily.ainvoke.return_value = "This is not JSON at all"

        result = await toolkit.get_news.ainvoke("AAPL")

        # Should handle gracefully
        assert isinstance(result, str)
        # Code now returns content wrapped in headers even if malformed
        assert "This is not JSON" in result or "No news found" in result

    async def test_news_incomplete_json(self, mock_tavily):
        """Test handling of incomplete JSON structure."""
        mock_tavily.ainvoke.return_value = '{"results": [{"title": "Test"'  # Truncated

        result = await toolkit.get_news.ainvoke("AAPL")

        assert isinstance(result, str)
        # Should just wrap the text
        assert "Test" in result or "No news found" in result

    async def test_news_missing_required_fields(self, mock_tavily):
        """Test handling when API returns JSON but missing required fields."""
        mock_tavily.ainvoke.return_value = {
            "results": [
                {"title": "News 1"},  # Missing 'content', 'url'
                {"content": "Content only"},  # Missing 'title'
            ]
        }

        result = await toolkit.get_news.ainvoke("AAPL")
        assert isinstance(result, str)

    async def test_news_null_values_in_array(self, mock_tavily):
        """Test handling of null values in results array."""
        mock_tavily.ainvoke.return_value = {
            "results": [None, {"title": "Valid", "content": "Content"}, None]
        }

        result = await toolkit.get_news.ainvoke("AAPL")

        assert isinstance(result, str)
        assert "Valid" in result

    async def test_news_unicode_corruption(self, mock_tavily):
        """Test handling of corrupted unicode in news content."""
        mock_tavily.ainvoke.return_value = {
            "results": [
                {
                    "title": "Test\udcffNews",  # Invalid unicode
                    "content": "Content\udcff\udcff",
                    "url": "https://test.com",
                }
            ]
        }

        result = await toolkit.get_news.ainvoke("AAPL")
        assert isinstance(result, str)

    async def test_news_extremely_long_content(self, mock_tavily):
        """Test handling of extremely long news articles."""
        long_content = "A" * 1000000  # 1MB of text
        mock_tavily.ainvoke.return_value = {
            "results": [
                {
                    "title": "Long Article",
                    "content": long_content,
                    "url": "https://test.com",
                }
            ]
        }

        result = await toolkit.get_news.ainvoke("AAPL")

        # Should truncate
        assert isinstance(result, str)
        assert len(result) < 50000

    async def test_news_html_injection(self, mock_tavily):
        """Test handling of HTML/script injection in news content."""
        mock_tavily.ainvoke.return_value = {
            "results": [
                {
                    "title": "<script>alert('xss')</script>Title",
                    "content": "<img src=x onerror=alert(1)>Content",
                    "url": "https://test.com",
                }
            ]
        }

        result = await toolkit.get_news.ainvoke("AAPL")

        # Should sanitize HTML
        assert "<script>" not in result
        assert "&lt;script&gt;" in result or "script" in result

    async def test_news_api_timeout(self, mock_tavily):
        """Test handling of API timeout."""
        mock_tavily.ainvoke.side_effect = TimeoutError("Request timeout")

        result = await toolkit.get_news.ainvoke("AAPL")

        assert isinstance(result, str)
        assert "No news found" in result

    async def test_news_api_rate_limit(self, mock_tavily):
        """Test handling of rate limit errors."""
        mock_tavily.ainvoke.side_effect = Exception("429 Rate Limit Exceeded")

        result = await toolkit.get_news.ainvoke("AAPL")

        assert isinstance(result, str)
        assert "No news found" in result

    async def test_news_network_failure(self, mock_tavily):
        """Test handling of network connection failures."""
        mock_tavily.ainvoke.side_effect = ConnectionError("Network unreachable")

        result = await toolkit.get_news.ainvoke("AAPL")

        assert isinstance(result, str)
        assert "No news found" in result


# ==================== FINANCIAL METRICS EDGE CASES ====================


@pytest.mark.asyncio
class TestFinancialMetricsEdgeCases:
    """Edge cases for financial metrics tool."""

    async def test_metrics_infinity_values(self, mock_fetcher):
        """Test handling of infinity in financial ratios."""
        mock_data = {
            "currentPrice": 100.0,
            "currency": "USD",
            "trailingPE": float("inf"),  # Division by zero scenario
            "priceToBook": float("-inf"),
            "_data_source": "yfinance",
        }
        mock_fetcher.get_financial_metrics = AsyncMock(return_value=mock_data)

        result = await toolkit.get_financial_metrics.ainvoke("BADMATH")

        # Should sanitize infinity to null in JSON output
        assert isinstance(result, str)
        assert "inf" not in result.lower()
        parsed = json.loads(result)
        assert parsed["trailingPE"] is None  # Converted from inf
        assert parsed["priceToBook"] is None  # Converted from -inf
        assert parsed["currentPrice"] == 100.0  # Valid value preserved

    async def test_metrics_nan_values(self, mock_fetcher):
        """Test handling of NaN in metrics."""
        import math

        mock_data = {
            "currentPrice": 100.0,
            "currency": "USD",
            "returnOnEquity": float("nan"),
            "debtToEquity": math.nan,
            "_data_source": "yfinance",
        }
        mock_fetcher.get_financial_metrics = AsyncMock(return_value=mock_data)

        result = await toolkit.get_financial_metrics.ainvoke("NULL_TEST")

        # Should sanitize NaN to null in JSON output
        assert isinstance(result, str)
        assert ": nan" not in result.lower()
        parsed = json.loads(result)
        assert parsed["returnOnEquity"] is None  # Converted from NaN
        assert parsed["debtToEquity"] is None  # Converted from NaN
        assert parsed["currentPrice"] == 100.0  # Valid value preserved

    async def test_metrics_negative_price(self, mock_fetcher):
        """Test handling of corrupted negative price data."""
        mock_data = {
            "currentPrice": -150.0,  # Invalid
            "currency": "USD",
            "_data_source": "corrupt",
        }
        mock_fetcher.get_financial_metrics = AsyncMock(return_value=mock_data)

        result = await toolkit.get_financial_metrics.ainvoke("CORRUPT")

        # Should sanitize negative price to null in JSON output
        assert isinstance(result, str)
        assert "-150" not in result
        parsed = json.loads(result)
        assert parsed["currentPrice"] is None  # Converted from negative
        assert parsed["currency"] == "USD"  # Other fields preserved

    async def test_metrics_missing_currency(self, mock_fetcher):
        """Test handling of missing currency field."""
        mock_data = {
            "currentPrice": 100.0,
            # Missing 'currency'
            "returnOnEquity": 0.25,
            "_data_source": "yfinance",
        }
        mock_fetcher.get_financial_metrics = AsyncMock(return_value=mock_data)

        result = await toolkit.get_financial_metrics.ainvoke("NOCUR")

        # JSON output preserves the raw data - currency will be absent
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["currentPrice"] == 100.0
        assert parsed["returnOnEquity"] == 0.25
        assert "currency" not in parsed  # Missing field is simply absent

    async def test_metrics_unknown_currency(self, mock_fetcher):
        """Test handling of unrecognized currency code."""
        mock_data = {
            "currentPrice": 100.0,
            "currency": "ZZZ",  # Invalid currency code
            "_data_source": "yfinance",
        }
        mock_fetcher.get_financial_metrics = AsyncMock(return_value=mock_data)

        result = await toolkit.get_financial_metrics.ainvoke("BADCUR")

        assert isinstance(result, str)
        assert "ZZZ" in result  # Will show invalid currency but not crash

    async def test_metrics_string_numbers(self, mock_fetcher):
        """Test handling when numbers come as strings (API bug)."""
        mock_data = {
            "currentPrice": "150.00",  # String instead of float
            "currency": "USD",
            "returnOnEquity": "0.25",
            "trailingPE": "20.5",
            "_data_source": "buggy_api",
        }
        mock_fetcher.get_financial_metrics = AsyncMock(return_value=mock_data)

        result = await toolkit.get_financial_metrics.ainvoke("STRNUM")

        # Sanitization should convert strings to numbers
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["currentPrice"] == 150.0  # Converted from string
        assert parsed["returnOnEquity"] == 0.25  # Converted from string
        assert parsed["trailingPE"] == 20.5  # Converted from string
        assert parsed["currency"] == "USD"  # String fields preserved as strings


# ==================== TECHNICAL INDICATORS EDGE CASES ====================


@pytest.mark.asyncio
class TestTechnicalIndicatorsEdgeCases:
    """Edge cases for technical indicators."""

    async def test_indicators_insufficient_data_points(self):
        """Test with insufficient historical data for indicators."""
        # Only 10 days of data (need 50+ for proper MA calculations)
        mock_data = pd.DataFrame({"Close": [100.0] * 10, "Volume": [1000000] * 10})

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_data
            result = await toolkit.get_technical_indicators.ainvoke("NEWSTOCK")

        # Should indicate insufficient data or N/A (NaN handling check)
        assert isinstance(result, str)
        assert "nan" not in result.lower()
        assert "N/A" in result

    async def test_indicators_constant_price(self):
        """Test indicators with completely flat price (no volatility)."""
        mock_data = pd.DataFrame(
            {
                "Close": [100.0] * 200,  # Constant price
                "Volume": [1000000] * 200,
            }
        )

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_data
            result = await toolkit.get_technical_indicators.ainvoke("FLATLINE")

        # Should handle zero volatility
        assert isinstance(result, str)
        assert "100.00" in result  # Price should be shown

    async def test_indicators_extreme_gaps(self):
        """Test indicators with extreme price gaps."""
        prices = [100.0] * 50 + [500.0] * 50 + [100.0] * 50  # Huge gaps
        mock_data = pd.DataFrame({"Close": prices, "Volume": [1000000] * 150})

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_data
            result = await toolkit.get_technical_indicators.ainvoke("GAPPY")

        assert isinstance(result, str)

    async def test_indicators_missing_volume(self):
        """Test indicators when volume data is missing."""
        mock_data = pd.DataFrame(
            {
                "Close": [100.0 + i for i in range(200)]
                # No 'Volume' column
            }
        )

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_data
            result = await toolkit.get_technical_indicators.ainvoke("NOVOL")

        # Should handle missing volume gracefully
        assert isinstance(result, str)


# ==================== SENTIMENT ANALYSIS EDGE CASES ====================


@pytest.mark.asyncio
class TestSentimentEdgeCases:
    """Edge cases for social media sentiment analysis."""

    async def test_sentiment_empty_response(self, mock_stocktwits):
        """Test handling of empty sentiment data."""
        # Setup mock return for async await
        mock_stocktwits.get_sentiment.return_value = {
            "total_messages": 0,
            "messages": [],
        }

        result = await toolkit.get_social_media_sentiment.ainvoke("UNKNOWN")

        assert isinstance(result, str)
        assert "0" in result or "No data" in result

    async def test_sentiment_malformed_messages(self, mock_stocktwits):
        """Test handling of malformed message structure."""
        mock_stocktwits.get_sentiment.return_value = {
            "total_messages": 2,
            "messages": [
                {"invalid": "structure"},  # Missing expected fields
                None,  # Null message
            ],
        }

        result = await toolkit.get_social_media_sentiment.ainvoke("MALFORMED")

        assert isinstance(result, str)

    async def test_sentiment_api_exception(self, mock_stocktwits):
        """Test handling of API exceptions."""
        # Ensure side_effect raises properly in async context
        mock_stocktwits.get_sentiment.side_effect = Exception("API Error")

        result = await toolkit.get_social_media_sentiment.ainvoke("ERROR")

        assert isinstance(result, str)
        assert "error" in result.lower()

    async def test_sentiment_unicode_in_messages(self, mock_stocktwits):
        """Test handling of various unicode in social messages."""
        mock_stocktwits.get_sentiment.return_value = {
            "total_messages": 3,
            "messages": [
                {"body": "🚀🌙 To the moon!", "sentiment": "Bullish"},
                {"body": "测试中文", "sentiment": "Bullish"},
                {"body": "Тест русский", "sentiment": "Bearish"},
            ],
            "bullish_pct": 66.6,
        }

        result = await toolkit.get_social_media_sentiment.ainvoke("UNICODE")

        # Should handle unicode gracefully
        assert isinstance(result, str)

    async def test_sentiment_very_long_messages(self, mock_stocktwits):
        """Test handling of extremely long social messages."""
        long_message = "A" * 10000
        mock_stocktwits.get_sentiment.return_value = {
            "total_messages": 1,
            "messages": [{"body": long_message, "sentiment": "Bullish"}],
            "bullish_pct": 100.0,
        }

        result = await toolkit.get_social_media_sentiment.ainvoke("LONGMSG")

        # Should handle large content
        assert isinstance(result, str)


# ==================== CROSS-TOOL INTEGRATION TESTS ====================


@pytest.mark.asyncio
class TestCrossToolFailures:
    """Test cascading failures across multiple tool calls."""

    async def test_all_tools_fail_gracefully(
        self, mock_fetcher, mock_tavily, mock_stocktwits
    ):
        """Test that if all data sources fail, system doesn't crash."""
        mock_fetcher.get_financial_metrics = AsyncMock(
            side_effect=Exception("FMP Down")
        )
        mock_tavily.ainvoke.side_effect = Exception("Tavily Down")
        mock_stocktwits.get_sentiment.side_effect = Exception("StockTwits Down")

        # Each tool should fail gracefully, returning error string instead of raising
        metrics_result = await toolkit.get_financial_metrics.ainvoke("AAPL")
        news_result = await toolkit.get_news.ainvoke("AAPL")
        sentiment_result = await toolkit.get_social_media_sentiment.ainvoke("AAPL")

        assert all(
            isinstance(r, str) for r in [metrics_result, news_result, sentiment_result]
        )

        # Assert that news_result handled the error gracefully as "No news found"
        assert "No news found" in news_result
        # JSON output uses lowercase "error" key
        assert "error" in metrics_result.lower()
        assert "error" in sentiment_result.lower()

    async def test_partial_data_synthesis(self, mock_fetcher):
        """Test system handles some fields present, others missing."""
        mock_data = {
            "currentPrice": 150.0,
            "currency": "USD",
            # Only 2 fields out of 20+ possible
            "_data_source": "partial",
        }
        mock_fetcher.get_financial_metrics = AsyncMock(return_value=mock_data)

        result = await toolkit.get_financial_metrics.ainvoke("PARTIAL")

        # Should work with minimal data
        assert "150.0" in result
        assert "USD" in result
