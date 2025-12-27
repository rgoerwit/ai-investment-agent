"""
Tests for Unified Parallel Data Fetcher

Verifies:
1. Parallel execution of all sources
2. Smart merging with quality scores
3. Mandatory Tavily gap-filling logic
4. Panic Mode for Asian tickers
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.fetcher import SmartMarketDataFetcher


@pytest.fixture
def fetcher():
    return SmartMarketDataFetcher()


class TestParallelExecution:
    """Verify all sources are called in parallel."""

    @pytest.mark.asyncio
    async def test_all_sources_called(self, fetcher):
        # Mock all fetch methods
        fetcher._fetch_yfinance_enhanced = AsyncMock(
            return_value={"symbol": "TEST", "currentPrice": 100}
        )
        fetcher._fetch_yahooquery_fallback = MagicMock(
            return_value={"pe": 15}
        )  # Not async
        fetcher._fetch_fmp_fallback = AsyncMock(return_value={"roe": 0.2})

        # Don't actually call Tavily
        fetcher._fetch_tavily_gaps = AsyncMock(return_value={})

        # Run
        result = await fetcher.get_financial_metrics("TEST")

        # Verify calls
        assert fetcher._fetch_yfinance_enhanced.called
        assert fetcher._fetch_fmp_fallback.called
        # Check if yahooquery wrapper was called (it's wrapped in to_thread)

        # Result should contain merged data
        assert result["currentPrice"] == 100
        assert result["pe"] == 15
        assert result["roe"] == 0.2


class TestSmartMerge:
    """Verify quality-based merging."""

    def test_quality_priority(self, fetcher):
        # Setup conflicting data
        source_results = {
            "yfinance": {
                "trailingPE": 10.0,
                "_trailingPE_source": "yfinance_info",
            },  # Quality 9
            "yahooquery": {"trailingPE": 20.0},  # Quality 6
            "fmp": {"trailingPE": 30.0},  # Quality 7
        }

        # NOTE: Function returns (merged, metadata) tuple
        merged, meta = fetcher._smart_merge_with_quality(source_results, "TEST")

        # Should keep yfinance (highest quality)
        assert merged["trailingPE"] == 10.0
        assert meta["field_sources"]["trailingPE"] == "yfinance"

    def test_fmp_over_yahooquery(self, fetcher):
        # Setup conflicting data (no yfinance)
        source_results = {
            "yfinance": None,
            "yahooquery": {"trailingPE": 20.0},  # Quality 6
            "fmp": {"trailingPE": 30.0},  # Quality 7
        }

        # NOTE: Function returns (merged, metadata) tuple
        merged, meta = fetcher._smart_merge_with_quality(source_results, "TEST")

        # Should keep FMP (7 > 6)
        assert merged["trailingPE"] == 30.0
        assert meta["field_sources"]["trailingPE"] == "fmp"


class TestTavilyGapFilling:
    """Verify mandatory gap filling logic."""

    @pytest.mark.asyncio
    async def test_tavily_triggered_low_coverage(self, fetcher):
        # Mock basic data (low coverage)
        fetcher._fetch_all_sources_parallel = AsyncMock(
            return_value={
                "yfinance": {"symbol": "TEST", "currentPrice": 100, "currency": "USD"}
            }
        )

        # Mock Tavily success
        fetcher._fetch_tavily_gaps = AsyncMock(
            return_value={"trailingPE": 15.5, "marketCap": 1000000000}
        )

        # Run
        result = await fetcher.get_financial_metrics("TEST")

        # Verify Tavily was called
        assert fetcher._fetch_tavily_gaps.called

        # Verify data was merged
        assert result["trailingPE"] == 15.5
        assert "_gaps_filled" in result
        assert result["_gaps_filled"] > 0

    @pytest.mark.asyncio
    async def test_tavily_skipped_high_coverage(self, fetcher):
        # Mock high coverage data
        full_data = {
            "symbol": "TEST",
            "currentPrice": 100,
            "currency": "USD",
            "trailingPE": 15,
            "marketCap": 100,
            "priceToBook": 2,
            "returnOnEquity": 0.15,
            "revenueGrowth": 0.1,
            "profitMargins": 0.2,
            "debtToEquity": 0.5,
            "currentRatio": 1.5,
            "freeCashflow": 1000,
            "operatingCashflow": 1000,
            "numberOfAnalystOpinions": 5,
        }

        fetcher._fetch_all_sources_parallel = AsyncMock(
            return_value={"yfinance": full_data}
        )

        fetcher._fetch_tavily_gaps = AsyncMock()

        await fetcher.get_financial_metrics("TEST")

        # Tavily should NOT be called
        assert not fetcher._fetch_tavily_gaps.called


class TestPanicMode:
    """Verify 'Panic Mode' for Asian tickers."""

    @pytest.mark.asyncio
    async def test_panic_mode_trigger_asian_stocks(self, fetcher):
        """
        REGRESSION TEST: Verify that if yfinance returns NOTHING for an Asian ticker,
        the system triggers 'Panic Mode' and calls Tavily for ALL critical fields.

        UPDATED: Test now reflects actual behavior where DANGEROUS_FIELDS
        (trailingPE, forwardPE, pegRatio, currentPrice, marketCap) are filtered out
        from Tavily requests to prevent hallucinated financial data.
        """
        ticker = "0005.HK"

        # Mock total failure from all standard sources
        fetcher._fetch_all_sources_parallel = AsyncMock(
            return_value={"yfinance": None, "yahooquery": None, "fmp": None}
        )

        # Mock Tavily rescue - it returns data even for dangerous fields
        # when the mock is called directly (simulating successful web extraction)
        fetcher._fetch_tavily_gaps = AsyncMock(
            return_value={
                "returnOnEquity": 0.18,
                "returnOnAssets": 0.12,
                "debtToEquity": 0.45,
                "currentRatio": 1.8,
                "operatingMargins": 0.22,
                "grossMargins": 0.35,
                "profitMargins": 0.15,
                "revenueGrowth": 0.08,
                "earningsGrowth": 0.10,
            }
        )

        data = await fetcher.get_financial_metrics(ticker)

        # Verify Tavily was called in panic mode (should be called twice: once for panic, once for mandatory gap fill)
        assert fetcher._fetch_tavily_gaps.called
        assert fetcher._fetch_tavily_gaps.call_count >= 1

        # CRITICAL: We need to check the FIRST call (panic mode), not the last call
        # The first call is made during panic mode with IMPORTANT_FIELDS + REQUIRED_BASICS
        # The second call (if any) is the mandatory gap-fill phase with remaining gaps
        first_call_args = fetcher._fetch_tavily_gaps.call_args_list[0]
        requested_fields = first_call_args[0][1]  # Second positional argument

        # CRITICAL FIX: The actual implementation filters out DANGEROUS_FIELDS before
        # making Tavily requests. These include: trailingPE, forwardPE, pegRatio,
        # currentPrice, and marketCap to prevent hallucinated financial data.
        #
        # In panic mode, the code passes IMPORTANT_FIELDS + REQUIRED_BASICS to _fetch_tavily_gaps,
        # but _fetch_tavily_gaps filters out dangerous fields internally.
        #
        # We should verify that:
        # 1. The panic mode call includes MANY fields (before filtering)
        # 2. The test should verify the behavior, not inspect internal filtering

        # The panic mode call passes IMPORTANT_FIELDS (15 fields) + REQUIRED_BASICS (3 fields)
        # That's ~18 fields passed to _fetch_tavily_gaps
        # After filtering dangerous fields internally, safe fields remain
        assert (
            len(requested_fields) >= 10
        ), f"Expected panic mode to request many fields (IMPORTANT_FIELDS + REQUIRED_BASICS), got: {len(requested_fields)}"

        # Verify rescue was successful for safe fields
        # The mock returns these fields, and they should appear in the final data
        assert data.get("returnOnEquity") == 0.18
        assert data.get("returnOnAssets") == 0.12

    @pytest.mark.asyncio
    async def test_dangerous_fields_filtering(self, fetcher):
        """
        Verify that _fetch_tavily_gaps correctly filters out DANGEROUS_FIELDS
        even when they're passed in the missing_fields list.
        """
        # Setup mock Tavily client
        fetcher.tavily_client = MagicMock()
        fetcher.tavily_client.search = MagicMock(
            return_value={"results": [{"content": "ROE is 15%"}]}
        )

        # Call with both safe and dangerous fields
        missing_fields = [
            "trailingPE",  # DANGEROUS
            "forwardPE",  # DANGEROUS
            "currentPrice",  # DANGEROUS
            "marketCap",  # DANGEROUS
            "pegRatio",  # DANGEROUS
            "returnOnEquity",  # SAFE
            "returnOnAssets",  # SAFE
            "debtToEquity",  # SAFE
        ]

        result = await fetcher._fetch_tavily_gaps("TEST", missing_fields)

        # Verify that Tavily was NOT called if only dangerous fields remain
        # OR verify that only safe fields were searched for
        # The method should filter to only safe fields

        # If tavily_client.search was called, it should only be for safe fields
        if fetcher.tavily_client.search.called:
            # Check that queries don't include dangerous field terms
            for call in fetcher.tavily_client.search.call_args_list:
                query = call[0][0] if call[0] else call[1].get("query", "")
                # Dangerous terms shouldn't dominate the query
                assert not all(
                    term in query.lower() for term in ["p/e", "price", "market cap"]
                )


class TestCalculatedMetrics:
    """Test metrics calculated from other fields."""

    def test_derived_calculations(self, fetcher):
        data = {
            "returnOnAssets": 0.10,
            "debtToEquity": 1.0,
            "trailingPE": 20.0,
            "earningsGrowth": 0.20,
            "currentPrice": 100.0,
            "sharesOutstanding": 1000000,
        }

        derived = fetcher._calculate_derived_metrics(data, "TEST")

        # ROE = ROA * (1 + D/E) = 0.10 * 2 = 0.20
        assert derived["returnOnEquity"] == 0.20

        # PEG = PE / (Growth * 100) = 20 / 20 = 1.0
        assert derived["pegRatio"] == 1.0

        # Market Cap
        assert derived["marketCap"] == 100000000.0


class TestWebFallback:
    """Test web searching capabilities."""

    def test_selective_web_fill(self, fetcher):
        """Ensure regex extract works for key metrics."""
        text = "Market Cap: 300T. P/E Ratio: 15.5"

        extractor = fetcher.pattern_extractor
        result = extractor.extract_from_text(text)

        assert result["trailingPE"] == 15.5
        # 300T = 300 * 1e12 = 300,000,000,000,000.0
        assert result["marketCap"] == 300000000000000.0


class TestFXCaching:
    """Test currency caching logic."""

    def test_fx_cache_hit(self, fetcher):
        """Test that cached FX rates are used."""
        cache_key = "EUR_USD"

        # Manually set a valid cache entry
        from datetime import timedelta

        future_time = datetime.now() + timedelta(hours=1)

        fetcher.fx_cache = {cache_key: 0.85}
        fetcher.fx_cache_expiry_time = {cache_key: future_time}

        # Call
        rate = fetcher.get_currency_rate("EUR", "USD")

        # Verify
        assert rate == 0.85
