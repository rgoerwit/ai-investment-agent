from typing import Annotated, Optional
import pandas as pd
import structlog
from langchain_core.tools import tool
from src.ticker_utils import normalize_ticker
from src.data.fetcher import fetcher as market_data_fetcher
from src.fx_normalization import get_fx_rate

logger = structlog.get_logger(__name__)

# COMPREHENSIVE GLOBAL CURRENCY MAP
# format: suffix -> currency_code
# FX rates are fetched dynamically from yfinance, with these as fallback
# Rates approximate as of late 2024/early 2025 (used only if yfinance fails)
EXCHANGE_CURRENCY_MAP = {
    # --- Americas ---
    'US': 'USD',
    'TO': 'CAD',  # Toronto
    'V':  'CAD',  # TSX Venture
    'CN': 'CAD',  # Canadian National
    'MX': 'MXN',  # Mexico
    'SA': 'BRL',  # Brazil (Sao Paulo)
    'BA': 'ARS',  # Buenos Aires (Highly volatile)
    'SN': 'CLP',  # Santiago

    # --- Europe (Eurozone) ---
    'DE': 'EUR',  # Xetra (Germany)
    'F':  'EUR',  # Frankfurt
    'PA': 'EUR',  # Paris
    'AS': 'EUR',  # Amsterdam
    'BR': 'EUR',  # Brussels
    'MC': 'EUR',  # Madrid
    'MI': 'EUR',  # Milan
    'LS': 'EUR',  # Lisbon
    'VI': 'EUR',  # Vienna
    'IR': 'EUR',  # Dublin
    'HE': 'EUR',  # Helsinki
    'AT': 'EUR',  # Athens

    # --- Europe (Non-Euro) ---
    'L':  'GBP',  # London (Pence logic handled in code)
    'SW': 'CHF',  # Switzerland
    'S':  'CHF',  # Switzerland
    'ST': 'SEK',  # Stockholm
    'OL': 'NOK',  # Oslo
    'CO': 'DKK',  # Copenhagen
    'IC': 'ISK',  # Iceland
    'WA': 'PLN',  # Warsaw
    'PR': 'CZK',  # Prague
    'BD': 'HUF',  # Budapest
    'IS': 'TRY',  # Istanbul
    'ME': 'RUB',  # Moscow (Approx/Restricted)

    # --- Asia Pacific ---
    'T':  'JPY',  # Tokyo
    'HK': 'HKD',  # Hong Kong
    'SS': 'CNY',  # Shanghai
    'SZ': 'CNY',  # Shenzhen
    'TW': 'TWD',  # Taiwan
    'TWO':'TWD',  # Taiwan OTC
    'KS': 'KRW',  # Korea KOSPI
    'KQ': 'KRW',  # Korea KOSDAQ
    'SI': 'SGD',  # Singapore
    'KL': 'MYR',  # Kuala Lumpur
    'BK': 'THB',  # Bangkok
    'JK': 'IDR',  # Jakarta
    'VN': 'VND',  # Vietnam
    'PS': 'PHP',  # Philippines
    'BO': 'INR',  # Bombay
    'NS': 'INR',  # NSE India
    'AX': 'AUD',  # Australia
    'NZ': 'NZD',  # New Zealand

    # --- Middle East & Africa ---
    'TA': 'ILS',  # Tel Aviv
    'SR': 'SAR',  # Saudi Arabia
    'QA': 'QAR',  # Qatar
    'AE': 'AED',  # UAE
    'JO': 'ZAR',  # Johannesburg
    'EG': 'EGP',  # Egypt
}

@tool
async def calculate_liquidity_metrics(ticker: Annotated[Optional[str], "Stock ticker symbol"] = None) -> str:
    """
    Calculate liquidity metrics using the robust MarketDataFetcher.
    Checks 3-month average volume and turnover.
    Handles global currency conversion automatically.
    """
    if not ticker:
        return "Error: No ticker symbol provided."

    normalized_symbol = normalize_ticker(ticker)
    
    try:
        # Use the robust fetcher for history
        hist = await market_data_fetcher.get_historical_prices(normalized_symbol, period="3mo")
        
        if hist.empty:
            logger.warning("no_history_found", ticker=ticker)
            return f"""Liquidity Analysis for {ticker}:
Status: FAIL - Insufficient Data
Avg Daily Volume (3mo): N/A
Avg Daily Turnover (USD): N/A
"""

        # Calculate metrics
        avg_volume = hist['Volume'].mean()
        avg_close = hist['Close'].mean()
        
        # Calculate local turnover
        # NOTE: For UK stocks (.L), prices are in Pence, so we must divide by 100 
        # to get Pounds before converting to USD.
        if normalized_symbol.endswith('.L'):
            avg_turnover_local = avg_volume * (avg_close / 100.0)
            logger.info("pence_adjustment_applied", ticker=ticker)
        else:
            avg_turnover_local = avg_volume * avg_close
        
        # Determine currency and FX rate based on suffix
        suffix = 'US'  # Default to US
        if '.' in normalized_symbol:
            suffix = normalized_symbol.split('.')[-1].upper()

        # Look up currency for this exchange
        if suffix in EXCHANGE_CURRENCY_MAP:
            currency = EXCHANGE_CURRENCY_MAP[suffix]
        else:
            # Unknown suffix - assume USD and log warning
            currency = "USD"
            logger.warning("unknown_exchange_suffix", ticker=ticker, suffix=suffix, assumed_currency="USD")

        # Get FX rate dynamically (with fallback to static rates)
        fx_rate, fx_source = await get_fx_rate(currency, "USD", allow_fallback=True)

        if fx_rate is None:
            # Total FX failure - assume 1.0 and flag as uncertain
            fx_rate = 1.0
            fx_source = "assumed"
            logger.warning("fx_rate_unavailable_using_1.0", ticker=ticker, currency=currency)

        logger.info("liquidity_fx_conversion",
                   ticker=ticker,
                   suffix=suffix,
                   currency=currency,
                   fx_rate=fx_rate,
                   source=fx_source)

        avg_turnover_usd = avg_turnover_local * fx_rate

        # Threshold: $500k USD daily turnover is a reasonable floor
        threshold_usd = 500_000
        status = "PASS" if avg_turnover_usd > threshold_usd else "FAIL"

        return f"""Liquidity Analysis for {ticker}:
Status: {status}
Avg Daily Volume (3mo): {int(avg_volume):,}
Avg Daily Turnover (USD): ${int(avg_turnover_usd):,}
Details: {currency} turnover converted at FX rate {fx_rate:.6f} (source: {fx_source})
Threshold: $500,000 USD daily
"""

    except Exception as e:
        logger.error("liquidity_calculation_failed", ticker=ticker, error=str(e), exc_info=True)
        return f"""Liquidity Analysis for {ticker}:
Status: ERROR
Error: {str(e)}
"""