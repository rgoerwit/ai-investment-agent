"""
Alpha Vantage Data Fetcher
High-quality fundamental data source with strict rate limit handling.

Free Tier Limits: 25 requests/day, 5 requests/minute.
Strategy: Use until exhausted, then silent fail (Circuit Breaker).
"""

import os
import aiohttp
import asyncio
import structlog
import pandas as pd
from typing import Any
from src.data.interfaces import FinancialFetcher
from src.config import config

logger = structlog.get_logger(__name__)


class AlphaVantageFetcher(FinancialFetcher):
    """
    Async client for Alpha Vantage with automatic rate limit handling.

    Features:
    - Circuit breaker: stops after hitting rate limit
    - Async requests with timeout
    - Field mapping to internal schema
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.get_alpha_vantage_api_key()
        self.base_url = "https://www.alphavantage.co/query"
        self._session = None
        self._is_exhausted = False  # Circuit breaker

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        """Close the aiohttp session. Safe to call multiple times."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def is_available(self) -> bool:
        """Check if configured and quota remaining."""
        has_key = self.api_key is not None and self.api_key != ""
        not_exhausted = not self._is_exhausted

        # Only log unavailability reasons at debug level (respects --quiet flag)
        if not has_key:
            logger.debug("alpha_vantage_unavailable", reason="no_api_key")
        elif self._is_exhausted:
            logger.debug("alpha_vantage_unavailable", reason="rate_limit_exhausted")

        return has_key and not_exhausted

    async def get_price_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical price data.
        Currently not implemented for Alpha Vantage to save API calls for fundamentals.
        Returns empty DataFrame.
        """
        return pd.DataFrame()

    async def get_financial_metrics(self, symbol: str) -> dict[str, Any] | None:
        """
        Fetch company overview (fundamentals).

        Returns None if:
        - API key not configured
        - Rate limit exceeded
        - Invalid symbol
        - Network error
        """
        if not self.is_available():
            return None

        if not self._session:
            self._session = aiohttp.ClientSession()

        # Alpha Vantage uses standard ticker format (0005.HK, AAPL, etc.)
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": self.api_key
        }

        try:
            logger.debug("alpha_vantage_request", symbol=symbol)

            async with self._session.get(
                self.base_url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    # HTTP errors are debug level - not user-facing issues
                    logger.debug("alpha_vantage_http_error",
                                symbol=symbol,
                                status=response.status)
                    return None

                try:
                    data = await response.json()
                except (ValueError, aiohttp.ContentTypeError) as e:
                    logger.debug("alpha_vantage_malformed_json",
                                symbol=symbol,
                                error=str(e))
                    return None

                # Check for API rate limit message
                if "Note" in data:
                    if "higher API call volume" in data["Note"] or "call frequency" in data["Note"]:
                        # Rate limit hit - INFO level (visible but not alarming)
                        logger.info("alpha_vantage_rate_limit_hit",
                                   symbol=symbol,
                                   message="Daily quota exhausted (25 requests/day free tier)")
                        self._is_exhausted = True
                        return None

                # Check for "Information" field (often used for errors)
                if "Information" in data:
                    # API info messages are debug level
                    logger.debug("alpha_vantage_info",
                                symbol=symbol,
                                message=data["Information"])
                    return None

                # Check for valid response (must have Symbol field)
                if not data or "Symbol" not in data:
                    logger.debug("alpha_vantage_no_data", symbol=symbol)
                    return None

                # Success - debug level (only visible when debugging data sources)
                logger.debug("alpha_vantage_success",
                            symbol=symbol,
                            fields_count=len(data))

                return self._parse_overview(data)

        except asyncio.TimeoutError:
            # Timeout - debug level (expected occasionally)
            logger.debug("alpha_vantage_timeout", symbol=symbol)
            return None
        except Exception as e:
            # Unexpected errors - debug level
            logger.debug("alpha_vantage_request_failed",
                        symbol=symbol,
                        error=str(e))
            return None

    def _parse_overview(self, data: dict) -> dict[str, Any]:
        """
        Map Alpha Vantage OVERVIEW fields to internal schema.

        Alpha Vantage field names are documented at:
        https://www.alphavantage.co/documentation/#company-overview
        """
        output = {
            '_source': 'alpha_vantage',

            # Basic info
            'symbol': data.get('Symbol'),
            'currency': data.get('Currency'),

            # Valuation metrics
            'marketCap': self._safe_float(data.get('MarketCapitalization')),
            'trailingPE': self._safe_float(data.get('PERatio')),
            'forwardPE': self._safe_float(data.get('ForwardPE')),
            'priceToBook': self._safe_float(data.get('PriceToBookRatio')),
            'pegRatio': self._safe_float(data.get('PEGRatio')),

            # Profitability
            'returnOnEquity': self._safe_percentage(data.get('ReturnOnEquityTTM')),
            'returnOnAssets': self._safe_percentage(data.get('ReturnOnAssetsTTM')),
            'profitMargins': self._safe_percentage(data.get('ProfitMargin')),
            'operatingMargins': self._safe_percentage(data.get('OperatingMarginTTM')),

            # Growth
            'revenueGrowth': self._safe_percentage(data.get('QuarterlyRevenueGrowthYOY')),
            'earningsGrowth': self._safe_percentage(data.get('QuarterlyEarningsGrowthYOY')),

            # Financial strength
            'debtToEquity': self._safe_float(data.get('DebtToEquity')),
            'currentRatio': self._safe_float(data.get('CurrentRatio')),
            'beta': self._safe_float(data.get('Beta')),

            # Analyst data
            # Note: Alpha Vantage OVERVIEW endpoint doesn't provide analyst count directly
            # Only provides AnalystTargetPrice (a price value, not a count)
            # numberOfAnalystOpinions will be filled by other sources (yfinance, etc.)

            # Metadata
            'sector': data.get('Sector'),
            'industry': data.get('Industry'),
        }

        # Tag each field with source for quality tracking
        tagged_output = {}
        for key, value in output.items():
            if value is not None and key != '_source':
                tagged_output[key] = value
                tagged_output[f'_source_{key}'] = 'alpha_vantage'

        return tagged_output if tagged_output else None

    def _safe_float(self, value: Any) -> float | None:
        """Convert to float, handling None/'None'/'-' gracefully."""
        try:
            if value is None or value == 'None' or value == '-' or value == 'N/A':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    def _safe_percentage(self, value: Any) -> float | None:
        """
        Convert percentage string to decimal float.
        Example: '0.15' -> 0.15 (already decimal)
                 '15%' -> 0.15 (convert from percentage)
        """
        try:
            if value is None or value == 'None' or value == '-' or value == 'N/A':
                return None

            value_str = str(value)

            # Remove percentage sign if present
            if '%' in value_str:
                value_str = value_str.replace('%', '')
                return float(value_str) / 100.0

            # Already decimal (Alpha Vantage returns decimals, not percentages)
            return float(value_str)

        except (ValueError, TypeError):
            return None

    def _safe_int(self, value: Any) -> int | None:
        """Convert to int, handling None/'None'/'-' gracefully."""
        try:
            if value is None or value == 'None' or value == '-' or value == 'N/A':
                return None
            return int(float(value))
        except (ValueError, TypeError):
            return None



# Singleton instance
_av_fetcher = None


def get_av_fetcher() -> AlphaVantageFetcher:
    """Get or create singleton Alpha Vantage fetcher."""
    global _av_fetcher
    if _av_fetcher is None:
        _av_fetcher = AlphaVantageFetcher()
    return _av_fetcher
