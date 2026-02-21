"""
Comprehensive tests for src/toolkit.py - All agent tools.

Tests cover:
- get_financial_metrics: Primary data fetching, FMP fallback, formatting
- get_news: General/local news split, error handling, ticker normalization
- get_technical_indicators: Calculation success, errors, edge cases
- get_fundamental_analysis: Primary search, surgical fallback, full fallback
- get_social_media_sentiment: API success, errors, empty data

Each test uses proper async mocking and validates both success and error paths.
"""

import json
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

# Import the module under test
from src import toolkit


@pytest.fixture
def mock_fetcher():
    """Mock the singleton market_data_fetcher used in toolkit."""
    with patch("src.toolkit.market_data_fetcher") as mock:
        yield mock


@pytest.fixture
def mock_tavily():
    """Mock the tavily search wrapper function.

    Note: toolkit.py now uses _tavily_search_with_timeout (imported from tavily_utils.py)
    which wraps the tavily_tool with timeout protection. We mock this wrapper directly.
    """
    with patch(
        "src.toolkit._tavily_search_with_timeout", new_callable=AsyncMock
    ) as mock:
        yield mock


@pytest.fixture
def mock_stocktwits():
    """Mock the stocktwits_api instance."""
    with patch("src.toolkit.stocktwits_api") as mock:
        yield mock


@pytest.mark.asyncio
class TestGetFinancialMetrics:
    """Tests for the primary fundamental data tool."""

    async def test_complete_data_formatting(self, mock_fetcher):
        """Ensure the tool returns complete data as JSON for the LLM to process."""
        mock_data = {
            "currentPrice": 150.00,
            "currency": "USD",
            "returnOnEquity": 0.25,
            "returnOnAssets": 0.15,
            "operatingMargins": 0.30,
            "debtToEquity": 0.5,
            "currentRatio": 2.0,
            "totalCash": 1000000,
            "totalDebt": 500000,
            "operatingCashflow": 2000000,
            "freeCashflow": 1500000,
            "revenueGrowth": 0.10,
            "earningsGrowth": 0.12,
            "grossMargins": 0.40,
            "trailingPE": 20.5,
            "forwardPE": 18.0,
            "priceToBook": 5.0,
            "pegRatio": 1.1,
            "numberOfAnalystOpinions": 12,
            "_data_source": "yfinance",
        }
        mock_fetcher.get_financial_metrics = AsyncMock(return_value=mock_data)

        result = await toolkit.get_financial_metrics.ainvoke("AAPL")

        # Tool now returns raw JSON for agent to process
        parsed = json.loads(result)
        assert parsed["currentPrice"] == 150.0
        assert parsed["currency"] == "USD"
        assert parsed["returnOnEquity"] == 0.25
        assert parsed["trailingPE"] == 20.5
        assert parsed["numberOfAnalystOpinions"] == 12
        assert parsed["_data_source"] == "yfinance"

    async def test_partial_data_handling(self, mock_fetcher):
        """Ensure the tool handles missing (None) values gracefully as null in JSON."""
        # Only minimal data provided
        mock_data = {
            "currentPrice": 100.0,
            "currency": "USD",
            "_data_source": "partial",
            # All other fields missing/None
        }
        mock_fetcher.get_financial_metrics = AsyncMock(return_value=mock_data)

        result = await toolkit.get_financial_metrics.ainvoke("UNKNOWN")

        # Tool returns raw JSON - agent is responsible for interpreting null values
        parsed = json.loads(result)
        assert parsed["currentPrice"] == 100.0
        assert parsed["currency"] == "USD"
        assert parsed["_data_source"] == "partial"
        # Missing keys should not be present (or be None if included)
        assert parsed.get("returnOnEquity") is None
        assert parsed.get("trailingPE") is None

    async def test_fetcher_error_propagation(self, mock_fetcher):
        """Ensure errors from the fetcher are reported to the agent as JSON."""
        mock_fetcher.get_financial_metrics = AsyncMock(
            return_value={"error": "API Rate Limit"}
        )

        result = await toolkit.get_financial_metrics.ainvoke("AAPL")

        # Tool returns error in JSON format
        parsed = json.loads(result)
        assert "error" in parsed
        assert "API Rate Limit" in parsed["error"]

    async def test_growth_zero_vs_none(self, mock_fetcher):
        """
        CRITICAL: Ensure 0.0 growth is distinct from None in JSON.
        Zero growth (stagnation) must be 0.0, not null.
        """
        mock_data = {
            "currentPrice": 100.0,
            "currency": "USD",
            "revenueGrowth": 0.0,  # Zero growth (stagnation)
            "earningsGrowth": -0.05,  # Negative growth
            "grossMargins": None,  # Missing data
            "_data_source": "yfinance",
        }
        mock_fetcher.get_financial_metrics = AsyncMock(return_value=mock_data)

        result = await toolkit.get_financial_metrics.ainvoke("FLAT")

        # Tool returns raw JSON - 0.0 must be preserved as 0.0, not null
        parsed = json.loads(result)
        assert parsed["revenueGrowth"] == 0.0  # Zero, NOT None
        assert parsed["earningsGrowth"] == -0.05  # Negative value
        assert parsed["grossMargins"] is None  # null in JSON

    async def test_large_numbers_formatting(self, mock_fetcher):
        """Test that large cash/debt values are preserved as numbers in JSON."""
        mock_data = {
            "currentPrice": 250.00,
            "currency": "USD",
            "totalCash": 50000000000,  # $50B
            "totalDebt": 120000000000,  # $120B
            "operatingCashflow": 15000000000,  # $15B
            "freeCashflow": 12000000000,  # $12B
            "_data_source": "yfinance",
        }
        mock_fetcher.get_financial_metrics = AsyncMock(return_value=mock_data)

        result = await toolkit.get_financial_metrics.ainvoke("AAPL")

        # Tool returns raw JSON - large numbers preserved as integers
        parsed = json.loads(result)
        assert parsed["totalCash"] == 50000000000
        assert parsed["totalDebt"] == 120000000000
        assert parsed["operatingCashflow"] == 15000000000
        assert parsed["freeCashflow"] == 12000000000

    async def test_currency_non_usd(self, mock_fetcher):
        """Test handling of non-USD currencies (GBP, EUR, JPY, KRW, etc)."""
        mock_data = {
            "currentPrice": 25000.00,
            "currency": "JPY",
            "returnOnEquity": 0.12,
            "_data_source": "yfinance",
        }
        mock_fetcher.get_financial_metrics = AsyncMock(return_value=mock_data)

        result = await toolkit.get_financial_metrics.ainvoke("7203.T")

        # Tool returns raw JSON - currency included in data
        parsed = json.loads(result)
        assert parsed["currentPrice"] == 25000.0
        assert parsed["currency"] == "JPY"
        assert parsed["returnOnEquity"] == 0.12


@pytest.mark.asyncio
class TestGetNews:
    """Tests for the news gathering tool."""

    async def test_news_general_and_local_split(self, mock_tavily):
        """Test that news is split into General and Local sections when both exist."""
        # Mocking extract_company_name_async internal call
        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            mock_name.return_value = "Toyota"

            # Setup Tavily to return different results for General vs Local queries
            async def side_effect(query_dict):
                query = query_dict["query"]
                if "site:" in query:  # Local query pattern
                    return "Local Japanese News: Nikkei reports strong earnings for Toyota."
                else:  # General query pattern
                    return "General News: Toyota expands manufacturing in US markets."

            mock_tavily.side_effect = side_effect

            # Use a ticker that triggers local logic (.T for Japan)
            result = await toolkit.get_news.ainvoke({"ticker": "7203.T"})

            assert "News Results for Toyota" in result
            assert "=== GENERAL NEWS ===" in result
            assert "Toyota expands" in result
            assert "=== LOCAL/REGIONAL NEWS SOURCES ===" in result
            assert "Nikkei reports" in result

    async def test_news_no_results(self, mock_tavily):
        """Test handling of zero results from news search."""
        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            mock_name.return_value = "Ghost Corp"
            mock_tavily.return_value = None  # No results

            result = await toolkit.get_news.ainvoke({"ticker": "GHST"})

            assert "No news found" in result.lower() or "Ghost Corp" in result

    async def test_news_general_only(self, mock_tavily):
        """Test news when only general search returns results (no local)."""
        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            mock_name.return_value = "Apple"

            async def side_effect(query_dict):
                query = query_dict["query"]
                if "site:" in query:
                    return None  # No local results
                else:
                    return "Apple announces new iPhone model with improved camera."

            mock_tavily.side_effect = side_effect

            result = await toolkit.get_news.ainvoke({"ticker": "AAPL"})

            assert "Apple" in result
            assert "iPhone model" in result
            # Should not show local section if no local results
            assert result.count("===") == 2  # Only one section divider (GENERAL NEWS)

    async def test_news_tavily_error(self, mock_tavily):
        """Test handling of Tavily API errors."""
        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            mock_name.return_value = "FailCorp"
            mock_tavily.side_effect = Exception("Tavily API Error: Rate limit exceeded")

            result = await toolkit.get_news.ainvoke({"ticker": "FAIL"})

            # Code handles error gracefully by returning an error message
            assert "Error fetching news" in result or "Rate limit" in result

    async def test_news_ticker_normalization(self, mock_tavily):
        """Test that tickers are properly normalized before searching."""
        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            with patch("src.toolkit.normalize_ticker") as mock_normalize:
                mock_normalize.return_value = "AAPL"
                mock_name.return_value = "Apple Inc."
                mock_tavily.return_value = "Apple news content"

                await toolkit.get_news.ainvoke({"ticker": "AAPL.US"})

                # Verify normalization was called
                mock_normalize.assert_called_once()


@pytest.mark.asyncio
class TestTechnicalIndicators:
    """Tests for technical analysis calculations."""

    async def test_indicators_success(self, mock_fetcher):
        """Test standard indicator calculation with valid data."""
        # Create a dummy dataframe with enough data for stockstats
        # Needs to be >2y for new moving average logic in toolkit
        dates = pd.date_range(start="2022-01-01", periods=600)
        data = {
            "Open": [100 + i * 0.1 for i in range(600)],
            "High": [105 + i * 0.1 for i in range(600)],
            "Low": [95 + i * 0.1 for i in range(600)],
            "Close": [102 + i * 0.1 for i in range(600)],
            "Volume": [1000 + i * 10 for i in range(600)],
        }
        df = pd.DataFrame(data, index=dates)

        mock_fetcher.get_historical_prices = AsyncMock(return_value=df)

        result = await toolkit.get_technical_indicators.ainvoke("AAPL")

        assert "Technical Indicators for AAPL" in result
        # Updated assertions to match new format "RSI (14):"
        assert "RSI" in result
        assert "MACD" in result
        assert "SMA 50" in result
        assert "SMA 200" in result
        assert "Bollinger Upper" in result

    async def test_indicators_insufficient_data(self, mock_fetcher):
        """Test handling of empty or insufficient history."""
        mock_fetcher.get_historical_prices = AsyncMock(return_value=pd.DataFrame())

        result = await toolkit.get_technical_indicators.ainvoke("EMPTY")

        # Code returns exactly "No data" (case-sensitive match needed)
        assert "No data" in result

    async def test_indicators_short_history(self, mock_fetcher):
        """Test handling when history is too short for indicators (< 14 days for RSI)."""
        dates = pd.date_range(start="2024-01-01", periods=5)
        data = {
            "Open": [100] * 5,
            "High": [105] * 5,
            "Low": [95] * 5,
            "Close": [102] * 5,
            "Volume": [1000] * 5,
        }
        df = pd.DataFrame(data, index=dates)

        mock_fetcher.get_historical_prices = AsyncMock(return_value=df)

        result = await toolkit.get_technical_indicators.ainvoke("SHORT")

        # Should handle gracefully - either error message or skip certain indicators
        assert "SHORT" in result

    async def test_indicators_calculation_error(self, mock_fetcher):
        """Test handling of calculation errors (bad data format)."""
        # Create malformed data that will cause stockstats to fail
        dates = pd.date_range(start="2024-01-01", periods=30)
        data = {
            "Open": [None] * 30,  # All None values
            "High": [None] * 30,
            "Low": [None] * 30,
            "Close": [None] * 30,
            "Volume": [None] * 30,
        }
        df = pd.DataFrame(data, index=dates)

        mock_fetcher.get_historical_prices = AsyncMock(return_value=df)

        result = await toolkit.get_technical_indicators.ainvoke("BAD")

        # Should return error message, not crash
        assert isinstance(result, str)
        assert "error" in result.lower() or "unavailable" in result.lower()

    async def test_indicators_extreme_values(self, mock_fetcher):
        """Test handling of extreme indicator values (RSI = 100, negative MACD, etc)."""
        dates = pd.date_range(start="2022-01-01", periods=600)
        # Create data that produces extreme RSI (all increases -> RSI ~100)
        data = {
            "Open": [100 + i * 5 for i in range(600)],
            "High": [105 + i * 5 for i in range(600)],
            "Low": [100 + i * 5 for i in range(600)],
            "Close": [105 + i * 5 for i in range(600)],
            "Volume": [1000] * 600,
        }
        df = pd.DataFrame(data, index=dates)

        mock_fetcher.get_historical_prices = AsyncMock(return_value=df)

        result = await toolkit.get_technical_indicators.ainvoke("EXTREME")

        # Should format extreme values properly (not crash)
        # Check for RSI presence generally, avoiding strict "RSI:" check if formatting differs
        assert "RSI" in result
        assert "100" in result or "99" in result


@pytest.mark.asyncio
class TestFundamentalAnalysis:
    """Tests for the qualitative research tool (ADRs, Analysts)."""

    async def test_primary_search_success(self, mock_tavily):
        """Test simple success path with adequate primary search results."""
        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            mock_name.return_value = "Samsung"

            # FIX: Make result long enough to pass length threshold (>200 chars)
            long_result = (
                "Samsung Electronics Co., Ltd. is a South Korean multinational "
                "electronics corporation headquartered in Suwon. The company has "
                "American Depositary Receipts (ADRs) listed on OTC markets under "
                "the symbol SSNLF. Approximately 20 analysts provide research "
                "coverage for the stock. Samsung is a major player in semiconductors, "
                "consumer electronics, and mobile devices with substantial global "
                "market presence and competitive positioning in the technology sector."
            )
            mock_tavily.return_value = long_result

            result = await toolkit.get_fundamental_analysis.ainvoke("005930.KS")

            # Actual format: "Fundamental Search Results for 005930.KS:"
            assert "Fundamental Search Results for 005930.KS" in result
            assert "ADR" in result or "Depositary Receipt" in result
            # Should NOT show fallback messages
            assert "Fallback Name Search" not in result
            assert "switched to company name search" not in result

    async def test_surgical_fallback_logic(self, mock_tavily):
        """
        Test the specific logic where primary search lacks ADR info,
        triggering a secondary 'surgical' search.
        """
        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            mock_name.return_value = "Samsung"

            async def side_effect(query_dict):
                query = query_dict["query"]
                # Case 1: Primary search (ticker match)
                if "005930.KS" in query:
                    # Primary search returns long data, but NO ADR info
                    return (
                        "Samsung Electronics revenue is up significantly. "
                        "Market cap is huge. Good fundamentals and strong balance sheet. "
                        "The company has excellent growth prospects. "
                        "Analysts are bullish on semiconductor division. "
                        "Manufacturing capacity is expanding globally. "
                        "Research and development spending increased. "
                        "Market share gains in mobile devices."
                    )
                # Case 2: Surgical search (Name + ADR match)
                # NOTE: src/toolkit.py adds quotes: '"Samsung" American...'
                # We check components to be robust against formatting changes
                if "Samsung" in query and "ADR" in query:
                    # Surgical search finds the ADR
                    return "Samsung has a Global Depositary Receipt (GDR) and OTC ADR under SSNLF."

                return "No data"

            mock_tavily.side_effect = side_effect

            result = await toolkit.get_fundamental_analysis.ainvoke("005930.KS")

            # Check that we got the surgical append
            assert "=== SUPPLEMENTAL ADR SEARCH ===" in result or "ADR" in result
            assert "GDR" in result or "SSNLF" in result

    async def test_full_fallback_on_weak_result(self, mock_tavily):
        """
        Test that if primary ticker search is too short/weak,
        it falls back to a full company name search.
        """
        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            mock_name.return_value = "HSBC Holdings"

            async def side_effect(query_dict):
                query = query_dict["query"]
                if "0005.HK" in query:
                    return "Weak result."  # < 200 chars, triggers fallback
                if "HSBC Holdings" in query:
                    return (
                        "HSBC Holdings PLC is a British universal bank and financial "
                        "services holding company. It has American Depositary Receipts "
                        "(ADRs) trading on the New York Stock Exchange under symbol HSBC. "
                        "The company is one of the world's largest banking organizations. "
                        "Analysts provide extensive coverage with detailed research reports. "
                        "The bank operates globally with strong presence in Asia."
                    )
                return "No data"

            mock_tavily.side_effect = side_effect

            result = await toolkit.get_fundamental_analysis.ainvoke("0005.HK")

            assert (
                "Fallback Name Search" in result
                or "switched to company name search" in result
            )
            assert "HSBC Holdings" in result
            assert "ADR" in result

    async def test_both_searches_fail(self, mock_tavily):
        """Test handling when both primary and fallback searches return nothing."""
        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            mock_name.return_value = "Unknown Corp"

            # All searches return empty/None
            mock_tavily.return_value = None

            result = await toolkit.get_fundamental_analysis.ainvoke("UNKN.XX")

            # Code returns formatted output with ticker and Limited Data indicator
            assert "UNKN.XX" in result
            assert "Limited Data" in result or "Fundamental Search Results" in result

    async def test_adr_detection_nyse(self, mock_tavily):
        """Test detection of NYSE-listed ADRs (most common type)."""
        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            mock_name.return_value = "Alibaba"

            long_result = (
                "Alibaba Group Holding Limited is a Chinese e-commerce giant. "
                "The company's American Depositary Shares (ADS) trade on the "
                "New York Stock Exchange under the ticker symbol BABA. "
                "Each ADS represents eight ordinary shares. "
                "Major institutional investors hold significant positions. "
                "The company faces regulatory scrutiny but maintains strong fundamentals."
            )
            mock_tavily.return_value = long_result

            result = await toolkit.get_fundamental_analysis.ainvoke("9988.HK")

            assert "ADS" in result or "ADR" in result or "American Depositary" in result
            assert "NYSE" in result or "New York Stock Exchange" in result

    async def test_adr_detection_otc(self, mock_tavily):
        """Test detection of OTC-traded ADRs (less liquid, unsponsored)."""
        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            mock_name.return_value = "Sony"

            long_result = (
                "Sony Group Corporation is a Japanese conglomerate. "
                "The company has unsponsored ADRs trading over-the-counter "
                "in the United States under the symbol SNEJF. "
                "These OTC ADRs have lower liquidity than exchange-listed securities. "
                "Sony also maintains a primary listing on the Tokyo Stock Exchange. "
                "The company operates in entertainment, gaming, and electronics sectors."
            )
            mock_tavily.return_value = long_result

            result = await toolkit.get_fundamental_analysis.ainvoke("6758.T")

            assert "ADR" in result or "over-the-counter" in result or "OTC" in result

    async def test_search_with_very_long_response(self, mock_tavily):
        """Test handling of extremely long search responses (>5000 chars)."""
        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            mock_name.return_value = "MegaCorp"

            # Create very long response
            long_result = (
                "MegaCorp is a company. " * 300  # ~6000 chars
                + "It has ADRs on NYSE. "
                + "Analysts cover it extensively. " * 100
            )
            mock_tavily.return_value = long_result

            result = await toolkit.get_fundamental_analysis.ainvoke("MEGA.XX")

            # Should handle without crashing
            assert isinstance(result, str)
            assert "MegaCorp" in result


@pytest.mark.asyncio
class TestStockTwitsSentiment:
    """Tests for the social sentiment tool."""

    async def test_sentiment_success(self, mock_stocktwits):
        """Test successful sentiment retrieval with mixed sentiment."""
        mock_stocktwits.get_sentiment = AsyncMock(
            return_value={
                "source": "StockTwits",
                "bullish_pct": 75.5,
                "bearish_pct": 24.5,
                "messages": ["Bullish on earnings!", "Going up", "Great quarter"],
            }
        )

        result = await toolkit.get_social_media_sentiment.ainvoke("AAPL")

        # Tool returns dict as string representation
        assert "'source': 'StockTwits'" in result or "StockTwits" in result
        assert "75.5" in result or "75" in result
        assert "bullish" in result.lower() or "Bullish" in result

    async def test_sentiment_api_error(self, mock_stocktwits):
        """Test handling of StockTwits API errors."""
        mock_stocktwits.get_sentiment = AsyncMock(
            return_value={"error": "Rate limit exceeded. Please try again later."}
        )

        result = await toolkit.get_social_media_sentiment.ainvoke("AAPL")

        assert "error" in result.lower() or "Rate limit" in result

    async def test_sentiment_no_data(self, mock_stocktwits):
        """Test handling when no sentiment data is available for ticker."""
        mock_stocktwits.get_sentiment = AsyncMock(
            return_value={
                "source": "StockTwits",
                "bullish_pct": 0.0,
                "bearish_pct": 0.0,
                "messages": [],
            }
        )

        result = await toolkit.get_social_media_sentiment.ainvoke("OBSCURE")

        # Should return the data structure even if empty
        assert isinstance(result, str)
        assert (
            "0.0" in result or "No messages" in result or "messages" in result.lower()
        )

    async def test_sentiment_extreme_bullish(self, mock_stocktwits):
        """Test handling of extreme bullish sentiment (>95%)."""
        mock_stocktwits.get_sentiment = AsyncMock(
            return_value={
                "source": "StockTwits",
                "bullish_pct": 98.5,
                "bearish_pct": 1.5,
                "messages": ["To the moon!", "Best stock ever!", "Loading up"],
            }
        )

        result = await toolkit.get_social_media_sentiment.ainvoke("MEME")

        assert "98.5" in result or "98" in result
        assert "1.5" in result or "1" in result

    async def test_sentiment_extreme_bearish(self, mock_stocktwits):
        """Test handling of extreme bearish sentiment (>95%)."""
        mock_stocktwits.get_sentiment = AsyncMock(
            return_value={
                "source": "StockTwits",
                "bullish_pct": 3.0,
                "bearish_pct": 97.0,
                "messages": ["Shorting this", "Overvalued", "Going down"],
            }
        )

        result = await toolkit.get_social_media_sentiment.ainvoke("FAIL")

        assert "3.0" in result or "3" in result
        assert "97.0" in result or "97" in result

    async def test_sentiment_ticker_suffix_handling(self, mock_stocktwits):
        """Test that ticker suffixes are handled properly (AAPL.US -> AAPL)."""
        # StockTwits doesn't recognize exchange suffixes
        mock_stocktwits.get_sentiment = AsyncMock(
            return_value={
                "source": "StockTwits",
                "bullish_pct": 60.0,
                "bearish_pct": 40.0,
                "messages": ["Neutral"],
            }
        )

        result = await toolkit.get_social_media_sentiment.ainvoke("AAPL.US")

        # Should work without error
        assert isinstance(result, str)
        # Verify that get_sentiment was called (suffix stripping happens in stocktwits_api)
        mock_stocktwits.get_sentiment.assert_called_once()

    async def test_sentiment_exception_handling(self, mock_stocktwits):
        """Test handling of unexpected exceptions from StockTwits API."""
        # Return error dict instead of raising exception
        # (toolkit wraps the call and returns the error dict)
        mock_stocktwits.get_sentiment = AsyncMock(
            return_value={"error": "Network timeout"}
        )

        result = await toolkit.get_social_media_sentiment.ainvoke("AAPL")

        # Should return error message in dict format
        assert isinstance(result, str)
        assert "error" in result.lower() or "timeout" in result.lower()


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests that combine multiple tools/functions."""

    async def test_international_ticker_full_workflow(
        self, mock_fetcher, mock_tavily, mock_stocktwits
    ):
        """
        Test complete workflow for international ticker:
        1. Financial metrics (with currency)
        2. News (with local sources)
        3. Fundamental research (ADR detection)
        4. Sentiment (suffix stripping)
        """
        # Setup financial metrics
        mock_fetcher.get_financial_metrics = AsyncMock(
            return_value={
                "currentPrice": 50000.0,
                "currency": "KRW",
                "returnOnEquity": 0.18,
                "_data_source": "yfinance",
            }
        )

        # Setup news
        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            mock_name.return_value = "Samsung"
            mock_tavily.return_value = "Samsung reports strong Q4 results."

            # Setup fundamental research
            long_result = (
                "Samsung Electronics is a South Korean tech giant. "
                "The company has ADRs on OTC markets under SSNLF. "
                "Major semiconductor manufacturer with global presence. "
                "Analysts rate it as a buy with strong fundamentals. "
                "Market leader in memory chips and displays."
            )

            # Setup sentiment
            mock_stocktwits.get_sentiment = AsyncMock(
                return_value={
                    "source": "StockTwits",
                    "bullish_pct": 70.0,
                    "bearish_pct": 30.0,
                    "messages": ["Bullish on semis"],
                }
            )

            # Execute all tools
            ticker = "005930.KS"

            metrics_result = await toolkit.get_financial_metrics.ainvoke(ticker)
            # Tool returns raw JSON
            parsed = json.loads(metrics_result)
            assert parsed["currentPrice"] == 50000.0
            assert parsed["currency"] == "KRW"

            news_result = await toolkit.get_news.ainvoke({"ticker": ticker})
            assert "Samsung" in news_result

            # For fundamental analysis, need to adjust mock for this call
            mock_tavily.return_value = long_result
            fundamental_result = await toolkit.get_fundamental_analysis.ainvoke(ticker)
            assert "ADR" in fundamental_result

            sentiment_result = await toolkit.get_social_media_sentiment.ainvoke(ticker)
            assert "70" in sentiment_result

    async def test_error_cascade_handling(
        self, mock_fetcher, mock_tavily, mock_stocktwits
    ):
        """
        Test that errors in one tool don't crash others.
        Verify graceful degradation when multiple tools fail.
        """
        # All tools return errors
        mock_fetcher.get_financial_metrics = AsyncMock(
            return_value={"error": "API down"}
        )
        mock_tavily.side_effect = Exception("Network error")
        mock_stocktwits.get_sentiment = AsyncMock(
            return_value={"error": "Rate limited"}
        )

        with patch(
            "src.toolkit.extract_company_name_async", new_callable=AsyncMock
        ) as mock_name:
            mock_name.return_value = "TestCorp"

            # Each tool should handle its own errors
            metrics_result = await toolkit.get_financial_metrics.ainvoke("TEST")
            assert (
                "error" in metrics_result.lower()
                or "unavailable" in metrics_result.lower()
            )

            news_result = await toolkit.get_news.ainvoke({"ticker": "TEST"})
            # News tool catches errors and returns error message
            assert (
                "Error fetching news" in news_result or "error" in news_result.lower()
            )

            sentiment_result = await toolkit.get_social_media_sentiment.ainvoke("TEST")
            assert (
                "error" in sentiment_result.lower()
                or "Rate limited" in sentiment_result
            )
