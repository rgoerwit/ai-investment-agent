import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.data.fetcher import SmartMarketDataFetcher

@pytest.fixture
def coordinator():
    # Patch the singleton fetchers to avoid real init
    with patch('src.data.fetcher.get_fmp_fetcher') as mock_fmp, \
         patch('src.data.fetcher.get_eodhd_fetcher') as mock_eodhd, \
         patch('src.data.fetcher.get_av_fetcher') as mock_av:

        fetcher = SmartMarketDataFetcher()

        # Mock availability - is_available() is a SYNC method, so use MagicMock
        # The async methods (get_financial_metrics) use AsyncMock
        fetcher.fmp_fetcher = AsyncMock()
        fetcher.fmp_fetcher.is_available = MagicMock(return_value=True)

        fetcher.eodhd_fetcher = AsyncMock()
        fetcher.eodhd_fetcher.is_available = MagicMock(return_value=True)

        fetcher.av_fetcher = AsyncMock()
        fetcher.av_fetcher.is_available = MagicMock(return_value=True)

        return fetcher

@pytest.mark.asyncio
async def test_smart_merge_priority(coordinator):
    """Test that higher quality sources override lower quality ones."""
    
    # Mock source returns
    # FMP (Quality 7)
    coordinator.fmp_fetcher.get_financial_metrics.return_value = {
        'trailingPE': 10.0,
        'marketCap': 1000,
        '_source': 'fmp'
    }
    
    # EODHD (Quality 9.5)
    coordinator.eodhd_fetcher.get_financial_metrics.return_value = {
        'trailingPE': 20.0,  # Should override FMP
        '_source': 'eodhd'
    }
    
    # YFinance (Mocking internal method)
    # Return None to simulate failure, letting EODHD win
    with patch.object(coordinator, '_fetch_yfinance_enhanced', return_value=None), \
         patch.object(coordinator, '_fetch_yahooquery_fallback', return_value=None), \
         patch.object(coordinator, '_fetch_av_fallback', return_value=None):
         
        result = await coordinator.get_financial_metrics("AAPL")
        
        # EODHD (20.0) > FMP (10.0)
        assert result['trailingPE'] == 20.0
        assert result['marketCap'] == 1000  # FMP filled the gap

@pytest.mark.asyncio
async def test_fallback_gap_filling(coordinator):
    """Test that secondary sources fill gaps when primary misses."""
    
    # FMP has P/E but no ROE
    coordinator.fmp_fetcher.get_financial_metrics.return_value = {
        'trailingPE': 15.0,
        'returnOnEquity': None,
        '_source': 'fmp'
    }
    
    # AlphaVantage has ROE
    coordinator.av_fetcher.get_financial_metrics.return_value = {
        'trailingPE': None,
        'returnOnEquity': 0.25,
        '_source': 'alpha_vantage'
    }
    
    # Mock others to None
    with patch.object(coordinator, '_fetch_yfinance_enhanced', return_value=None), \
         patch.object(coordinator, '_fetch_yahooquery_fallback', return_value=None), \
         patch.object(coordinator, '_fetch_eodhd_fallback', return_value=None):
         
        result = await coordinator.get_financial_metrics("MSFT")
        
        assert result['trailingPE'] == 15.0  # From FMP
        assert result['returnOnEquity'] == 0.25  # From AV
