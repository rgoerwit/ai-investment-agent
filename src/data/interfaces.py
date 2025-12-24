from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class FinancialFetcher(ABC):
    """
    Abstract Base Class for all financial data providers.

    This interface ensures that all data providers (yfinance, FMP, AlphaVantage, etc.)
    can be used interchangeably by the main data coordinator.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this fetcher is configured and operational.
        Returns True if API key is present and not rate-limited.
        """
        pass

    @abstractmethod
    async def get_financial_metrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Returns a dictionary of standardized financial metrics.
        Must return keys: 'currentPrice', 'marketCap', 'trailingPE', etc.
        Returns None if data fetch fails or rate limit is hit.
        """
        pass

    @abstractmethod
    async def get_price_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Returns OHLC DataFrame with standard columns: Open, High, Low, Close, Volume.
        """
        pass
