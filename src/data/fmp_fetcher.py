"""
FMP (Financial Modeling Prep) Data Fetcher
Fallback source for financial metrics when yfinance/yahooquery fail

This module provides access to FMP API as a backup data source for ex-US stocks
where Yahoo Finance coverage is insufficient.

Configuration:
    Set FMP_API_KEY in .env file

Error Handling:
    - Invalid API key: Raises ValueError (configuration error - must fix)
    - Data not found: Returns None (expected - log at DEBUG)
    - Network errors: Returns None (transient - log at DEBUG)

Usage:
    from src.data.fmp_fetcher import get_fmp_fetcher
    
    fmp = get_fmp_fetcher()
    if fmp.is_available():
        data = await fmp.get_financial_metrics("005930.KS")
"""

import os
import aiohttp
import logging
import pandas as pd
from typing import Optional, Dict, Any
from src.data.interfaces import FinancialFetcher

logger = logging.getLogger(__name__)


class FMPFetcher(FinancialFetcher):
    """
    Minimal FMP API client for financial metrics.
    
    Implements waterfall fallback:
        yfinance -> yahooquery -> FMP -> partial data
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP fetcher.
        
        Args:
            api_key: FMP API key (or None to read from FMP_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        self.base_url = "https://financialmodelingprep.com/stable"
        self._session = None
        self._key_validated = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close the aiohttp session. Safe to call multiple times."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def is_available(self) -> bool:
        """Check if FMP is configured (API key present)."""
        return self.api_key is not None
    
    async def _get(self, endpoint: str, params: Dict) -> Optional[Any]:
        """
        Make API request with error handling.
        
        Args:
            endpoint: API endpoint (e.g., 'ratios', 'key-metrics')
            params: Query parameters
            
        Returns:
            JSON response data or None if request failed
            
        Raises:
            ValueError: If API key is invalid (403 on first request)
        """
        if not self.is_available():
            return None
        
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        url = f"{self.base_url}/{endpoint}"
        params["apikey"] = self.api_key
        
        try:
            async with self._session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                    except (ValueError, aiohttp.ContentTypeError) as e:
                        logger.debug(f"FMP malformed JSON for {endpoint}: {e}")
                        return None
                    self._key_validated = True
                    return data
                    
                elif response.status == 403:
                    if not self._key_validated:
                        # Key is invalid - this is a configuration error
                        logger.error("FMP API key is invalid (403 Forbidden)")
                        raise ValueError("FMP_API_KEY is invalid or expired. Check your configuration.")
                    else:
                        # Key was valid before, might be rate limit
                        logger.warning(f"FMP 403 error for {endpoint} (possible rate limit)")
                        return None
                        
                else:
                    # Other HTTP errors - log at debug level
                    logger.debug(f"FMP API returned {response.status} for {endpoint}")
                    return None
                    
        except ValueError:
            # Re-raise API key validation errors
            raise
        except aiohttp.ClientError as e:
            # Network errors - log at debug level
            logger.debug(f"FMP network error for {endpoint}: {e}")
            return None
        except Exception as e:
            # Unexpected errors - log at debug level
            logger.debug(f"FMP request failed for {endpoint}: {e}")
            return None

    async def get_price_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Returns OHLC DataFrame with standard columns.
        Currently not implemented for FMP to save API calls.
        """
        return pd.DataFrame()
    
    async def get_financial_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive financial metrics for a symbol.
        
        Fetches data from multiple FMP endpoints and combines them into a single dict.
        Returns standardized keys: trailingPE, priceToBook, returnOnEquity, etc.
        """
        result = {}
        
        # Fetch ratios endpoint (has P/E, P/B, PEG, current ratio, D/E, margins)
        ratios = await self._get("ratios", {"symbol": symbol, "limit": 1})
        if ratios and isinstance(ratios, list) and len(ratios) > 0:
            r = ratios[0]
            # Map to standard interface keys
            result['trailingPE'] = r.get('priceToEarningsRatio')
            result['priceToBook'] = r.get('priceToBookRatio')
            result['pegRatio'] = r.get('priceToEarningsGrowthRatio')
            result['currentRatio'] = r.get('currentRatio')
            result['debtToEquity'] = r.get('debtToEquityRatio')
            result['profitMargins'] = r.get('netProfitMargin')
            result['freeCashflow'] = r.get('freeCashFlowPerShare') # Ratio endpoint often has per share
            result['operatingCashflow'] = r.get('operatingCashFlowPerShare')
        
        # Fetch key-metrics endpoint (has ROE, ROA, Cash Flows)
        metrics = await self._get("key-metrics", {"symbol": symbol, "limit": 1})
        if metrics and isinstance(metrics, list) and len(metrics) > 0:
            m = metrics[0]
            result['returnOnEquity'] = m.get('returnOnEquity')
            result['returnOnAssets'] = m.get('returnOnAssets')
            # Prefer absolute values if available
            if m.get('freeCashFlowPerShare'):
                 result['freeCashflow'] = m.get('freeCashFlowPerShare')
            if m.get('operatingCashFlowPerShare'):
                 result['operatingCashflow'] = m.get('operatingCashFlowPerShare')
            if m.get('marketCap'):
                result['marketCap'] = m.get('marketCap')

        
        # Fetch income statement growth endpoint (has revenue/EPS growth)
        growth = await self._get("income-statement-growth", {"symbol": symbol, "limit": 1})
        if growth and isinstance(growth, list) and len(growth) > 0:
            g = growth[0]
            result['revenueGrowth'] = g.get('growthRevenue')
            result['earningsGrowth'] = g.get('growthEPS')
        
        # Log if we got no data at all
        if not result:
            logger.debug(f"FMP returned no data for {symbol}")
            return None
            
        result['_source'] = 'fmp'
        return result


# Global singleton instance
_fmp_fetcher: Optional[FMPFetcher] = None


def get_fmp_fetcher() -> FMPFetcher:
    """
    Get or create global FMP fetcher instance.
    
    Returns:
        FMPFetcher instance (may not be available if no API key)
    """
    global _fmp_fetcher
    if _fmp_fetcher is None:
        _fmp_fetcher = FMPFetcher()
    return _fmp_fetcher


async def fetch_fmp_metrics(symbol: str) -> Optional[Dict[str, Optional[float]]]:
    """
    Convenience function to fetch FMP metrics.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dict of financial metrics or None if FMP not available
        
    Raises:
        ValueError: If API key is invalid
    """
    fmp = get_fmp_fetcher()
    if not fmp.is_available():
        return None
    
    async with fmp:
        return await fmp.get_financial_metrics(symbol)