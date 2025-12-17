"""
EOD Historical Data (EODHD) Fetcher
High-quality, cost-effective fallback for international/ex-US equities.

Handles:
- Fundamentals (Valuation, Profitability, Growth)
- Smart Ticker Normalization (YFinance -> EODHD format)
- Rate Limit Circuit Breaking (stops requests after 429/402 errors)

Error Codes Handled (per EODHD Docs):
- 401/403: Invalid Token
- 402: Payment Required (Subscription upgrade needed)
- 429: Too Many Requests (Daily limit reached)
"""

import os
import aiohttp
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class EODHDFetcher:
    """
    Async client for EOD Historical Data.
    Maintains state to stop requests if API limits are hit.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('EODHD_API_KEY')
        self.base_url = "https://eodhd.com/api"
        self._session = None
        self._is_exhausted = False  # Circuit breaker for rate limits
        
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
        """Check if configured and not rate-limited."""
        return self.api_key is not None and not self._is_exhausted

    def _normalize_ticker(self, ticker: str) -> str:
        """
        Convert yfinance ticker to EODHD format.
        
        Mappings:
        - US: "AAPL" -> "AAPL.US"
        - London: "BP.L" -> "BP.LSE" (Sometimes .L works, but .LSE is canonical)
        - Others: Generally match (0005.HK -> 0005.HK)
        """
        ticker = ticker.upper()
        
        # Specific overrides for common divergences
        if ticker.endswith('.L'):
            return ticker.replace('.L', '.LSE')
        
        # If no suffix, assume US
        if '.' not in ticker:
            return f"{ticker}.US"
            
        return ticker

    async def get_financial_metrics(self, symbol: str) -> Dict[str, Optional[float]]:
        """
        Fetch fundamentals from EODHD.
        Returns processed dictionary or None if failed.
        """
        if not self.is_available():
            return None

        if not self._session:
            self._session = aiohttp.ClientSession()

        eod_symbol = self._normalize_ticker(symbol)
        url = f"{self.base_url}/fundamentals/{eod_symbol}"
        params = {"api_token": self.api_key, "fmt": "json"}

        try:
            async with self._session.get(url, params=params, timeout=10) as response:

                # --- Error Handling & Circuit Breaking ---
                if response.status == 200:
                    try:
                        data = await response.json()
                    except (ValueError, aiohttp.ContentTypeError) as e:
                        logger.debug(f"EODHD malformed JSON for {eod_symbol}: {e}")
                        return None
                    return self._parse_fundamentals(data)
                
                elif response.status == 429:
                    logger.error(f"EODHD API Limit Exceeded (429). Disabling EODHD for this session.")
                    self._is_exhausted = True
                    return None
                
                elif response.status == 402:
                    logger.warning(f"EODHD Payment Required (402). Access restricted for {eod_symbol}.")
                    # Don't disable globally, might just be this specific exchange
                    return None
                    
                elif response.status == 404:
                    logger.debug(f"EODHD data not found for {eod_symbol}")
                    return None
                    
                else:
                    logger.warning(f"EODHD API error {response.status} for {eod_symbol}")
                    return None
                    
        except Exception as e:
            logger.debug(f"EODHD request failed: {e}")
            return None

    def _parse_fundamentals(self, data: Dict) -> Dict[str, Optional[float]]:
        """Map EODHD JSON structure to internal schema."""
        output = {
            '_source': 'eodhd',
            # Core Valuation
            'marketCap': None, 
            'trailingPE': None, 
            'forwardPE': None,
            'priceToBook': None, 
            'pegRatio': None,
            
            # Profitability
            'returnOnEquity': None, 
            'returnOnAssets': None,
            'profitMargins': None, 
            'operatingMargins': None,
            'grossMargins': None,

            # Growth & Health
            'revenueGrowth': None, 
            'earningsGrowth': None,
            'debtToEquity': None, 
            'currentRatio': None, 
            
            # Cash Flow
            'freeCashflow': None, 
            'operatingCashflow': None,
            
            # Basics
            'currency': None, 
            'currentPrice': None
        }

        try:
            general = data.get('General', {})
            highlights = data.get('Highlights', {})
            valuation = data.get('Valuation', {})
            technicals = data.get('Technicals', {})
            financials = data.get('Financials', {})

            # --- Basics ---
            output['currency'] = general.get('CurrencyCode')
            # EODHD Technicals often has the 50d MA or close
            output['currentPrice'] = self._safe_float(technicals.get('50DayMA'))

            # --- Valuation ---
            output['marketCap'] = self._safe_float(highlights.get('MarketCapitalization'))
            output['trailingPE'] = self._safe_float(highlights.get('PERatio'))
            output['forwardPE'] = self._safe_float(valuation.get('ForwardPE'))
            output['priceToBook'] = self._safe_float(valuation.get('PriceBookMRQ'))
            output['pegRatio'] = self._safe_float(highlights.get('PEGRatio'))

            # --- Profitability ---
            output['returnOnEquity'] = self._safe_float(highlights.get('ReturnOnEquityTTM'))
            output['returnOnAssets'] = self._safe_float(highlights.get('ReturnOnAssetsTTM'))
            output['profitMargins'] = self._safe_float(highlights.get('ProfitMargin'))
            output['operatingMargins'] = self._safe_float(highlights.get('OperatingMarginTTM'))
            output['grossMargins'] = self._safe_float(highlights.get('GrossProfitMarginTTM'))

            # --- Growth ---
            output['revenueGrowth'] = self._safe_float(highlights.get('RevenueTTMYoy'))
            output['earningsGrowth'] = self._safe_float(highlights.get('EarningsShareTTMYoy'))

            # --- Health ---
            # EODHD Debt/Equity is usually absolute, ensure we don't get %.
            output['debtToEquity'] = self._safe_float(highlights.get('DebtToEquity'))
            output['currentRatio'] = self._safe_float(highlights.get('CurrentRatio'))

            # --- Cash Flow ---
            # Parse from Cash_Flow statement to get absolute numbers
            cash_flow_statement = financials.get('Cash_Flow', {}).get('yearly', {})
            if cash_flow_statement:
                # Keys are dates, sort to get most recent
                dates = sorted(cash_flow_statement.keys())
                if dates:
                    last_report = cash_flow_statement[dates[-1]]
                    output['freeCashflow'] = self._safe_float(last_report.get('freeCashFlow'))
                    output['operatingCashflow'] = self._safe_float(last_report.get('totalCashFromOperatingActivities'))

        except Exception as e:
            logger.warning(f"Error parsing EODHD data structure: {e}")

        return output

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert to float."""
        try:
            if value is None or value == 'NA' or value == 'NaN':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

# Singleton Pattern
_eodhd_fetcher = None

def get_eodhd_fetcher() -> EODHDFetcher:
    global _eodhd_fetcher
    if _eodhd_fetcher is None:
        _eodhd_fetcher = EODHDFetcher()
    return _eodhd_fetcher