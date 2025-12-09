"""
Pragmatic Currency Normalization for International Stock Analysis

PHILOSOPHY:
- Only normalize metrics that are COMPARED across borders (market cap, revenue, volume)
- Leave ratios and percentages alone (already normalized)
- Use simple, robust fallback chain (yfinance → hardcoded rates → fail gracefully)
- Don't try to be a forex platform - just good enough for research

UPDATED: Dec 2025 - Aligned with modern yfinance patterns
"""

import asyncio
import structlog
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = structlog.get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# TIER 1: Dynamic FX Rates (yfinance - always up-to-date)
# ══════════════════════════════════════════════════════════════════════════════

async def get_fx_rate_yfinance(from_currency: str, to_currency: str = "USD") -> Optional[float]:
    """
    Get live FX rate from yfinance using standard forex pairs.

    Args:
        from_currency: Source currency (e.g., "JPY", "HKD")
        to_currency: Target currency (default "USD")

    Returns:
        Exchange rate as float, or None if unavailable

    Example:
        JPY → USD returns ~0.0067 (1 JPY = $0.0067)
        HKD → USD returns ~0.128 (1 HKD = $0.128)
    """
    if from_currency == to_currency:
        return 1.0

    # yfinance forex ticker format: "JPYUSD=X" (from + to + =X)
    fx_ticker = f"{from_currency}{to_currency}=X"

    try:
        import yfinance as yf

        # Use async thread pool to avoid blocking
        def _fetch_rate():
            ticker = yf.Ticker(fx_ticker)
            # Try fast_info first (faster)
            if hasattr(ticker, 'fast_info') and hasattr(ticker.fast_info, 'last_price'):
                return ticker.fast_info.last_price
            # Fallback to info dict
            info = ticker.info
            return info.get('regularMarketPrice') or info.get('previousClose')

        rate = await asyncio.wait_for(
            asyncio.to_thread(_fetch_rate),
            timeout=3.0  # Quick timeout - we have fallbacks
        )

        if rate and rate > 0:
            logger.debug("fx_rate_fetched", pair=fx_ticker, rate=rate, source="yfinance")
            return float(rate)
        else:
            logger.debug("fx_rate_invalid", pair=fx_ticker, rate=rate)
            return None

    except asyncio.TimeoutError:
        logger.debug("fx_rate_timeout", pair=fx_ticker, timeout_ms=3000)
        return None
    except Exception as e:
        logger.debug("fx_rate_fetch_error", pair=fx_ticker, error=str(e))
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2: Fallback Rates (Hardcoded, updated quarterly)
# ══════════════════════════════════════════════════════════════════════════════

# Last updated: Dec 2025
# Source: ECB reference rates, BoJ, HKMA
FALLBACK_RATES_TO_USD = {
    # Major Asian currencies (your primary use case)
    "JPY": 0.0067,   # Japanese Yen (¥150 = $1)
    "HKD": 0.128,    # Hong Kong Dollar (HK$7.80 = $1)
    "TWD": 0.032,    # Taiwan Dollar (NT$31 = $1)
    "KRW": 0.00075,  # Korean Won (₩1,330 = $1)
    "CNY": 0.14,     # Chinese Yuan (¥7.2 = $1)
    "INR": 0.012,    # Indian Rupee (₹83 = $1)
    "SGD": 0.74,     # Singapore Dollar (S$1.35 = $1)

    # European currencies
    "EUR": 1.09,     # Euro
    "GBP": 1.27,     # British Pound
    "CHF": 1.13,     # Swiss Franc

    # Other major currencies
    "CAD": 0.72,     # Canadian Dollar
    "AUD": 0.64,     # Australian Dollar
    "NZD": 0.60,     # New Zealand Dollar
    "MXN": 0.049,    # Mexican Peso
    "BRL": 0.20,     # Brazilian Real

    # Identity
    "USD": 1.0,
}

def get_fx_rate_fallback(from_currency: str, to_currency: str = "USD") -> Optional[float]:
    """
    Get FX rate from hardcoded fallback table.

    WARNING: These rates are updated manually and may be stale.
    Only use when yfinance is unavailable.
    """
    if from_currency == to_currency:
        return 1.0

    rate = FALLBACK_RATES_TO_USD.get(from_currency)
    if rate:
        logger.warning(
            "fx_rate_using_fallback",
            from_currency=from_currency,
            to_currency=to_currency,
            rate=rate,
            warning="Fallback rate may be stale - update FALLBACK_RATES_TO_USD quarterly"
        )
        return rate

    return None


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3: Unified Interface with Smart Fallback
# ══════════════════════════════════════════════════════════════════════════════

async def get_fx_rate(
    from_currency: str,
    to_currency: str = "USD",
    allow_fallback: bool = True
) -> Tuple[Optional[float], str]:
    """
    Get FX rate with smart fallback chain.

    Fallback order:
    1. yfinance (live rate, preferred)
    2. Hardcoded fallback (if allow_fallback=True)
    3. None (graceful failure)

    Args:
        from_currency: Source currency code (e.g., "JPY")
        to_currency: Target currency code (default "USD")
        allow_fallback: Whether to use hardcoded rates if yfinance fails

    Returns:
        Tuple of (rate, source) where source is "yfinance", "fallback", or "unavailable"

    Example:
        rate, source = await get_fx_rate("JPY", "USD")
        if rate:
            usd_value = jpy_value * rate
    """
    # Normalize currency codes (uppercase, strip whitespace)
    from_currency = from_currency.strip().upper()
    to_currency = to_currency.strip().upper()

    # Identity case
    if from_currency == to_currency:
        return 1.0, "identity"

    # Try yfinance first (preferred - always up-to-date)
    rate = await get_fx_rate_yfinance(from_currency, to_currency)
    if rate is not None:
        return rate, "yfinance"

    # Try fallback rates if allowed
    if allow_fallback:
        rate = get_fx_rate_fallback(from_currency, to_currency)
        if rate is not None:
            return rate, "fallback"

    # Total failure - log and return None
    logger.warning(
        "fx_rate_unavailable",
        from_currency=from_currency,
        to_currency=to_currency,
        tried_sources=["yfinance", "fallback"] if allow_fallback else ["yfinance"]
    )
    return None, "unavailable"


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS: Normalize Specific Metric Types
# ══════════════════════════════════════════════════════════════════════════════

async def normalize_to_usd(
    value: Optional[float],
    currency: str,
    metric_name: str = "value"
) -> Tuple[Optional[float], Dict[str, any]]:
    """
    Normalize a single value to USD with metadata tracking.

    Args:
        value: Numeric value in local currency (or None)
        currency: Source currency code
        metric_name: Name of metric (for logging)

    Returns:
        Tuple of (normalized_value, metadata_dict)

    Metadata includes:
        - original_value: Input value
        - original_currency: Input currency
        - fx_rate: Applied rate (or None)
        - fx_source: Where rate came from
        - normalized: Whether conversion was applied

    Example:
        market_cap_usd, meta = await normalize_to_usd(1.2e12, "HKD", "market_cap")
        # market_cap_usd ≈ 153.6e9 (USD)
        # meta["fx_rate"] ≈ 0.128
    """
    metadata = {
        "original_value": value,
        "original_currency": currency,
        "fx_rate": None,
        "fx_source": None,
        "normalized": False
    }

    # Handle None/missing values
    if value is None:
        return None, metadata

    # Already USD - no conversion needed
    if currency.upper() == "USD":
        metadata["normalized"] = False
        metadata["fx_rate"] = 1.0
        metadata["fx_source"] = "identity"
        return value, metadata

    # Get FX rate
    fx_rate, source = await get_fx_rate(currency, "USD")

    if fx_rate is None:
        logger.warning(
            "fx_normalization_failed",
            metric=metric_name,
            value=value,
            currency=currency,
            reason="No FX rate available"
        )
        # Return original value with warning metadata
        metadata["fx_source"] = "unavailable"
        return value, metadata

    # Apply normalization
    normalized_value = value * fx_rate
    metadata["fx_rate"] = fx_rate
    metadata["fx_source"] = source
    metadata["normalized"] = True

    logger.debug(
        "fx_normalized",
        metric=metric_name,
        original_value=value,
        currency=currency,
        fx_rate=fx_rate,
        normalized_value=normalized_value,
        source=source
    )

    return normalized_value, metadata


async def normalize_financial_dict(
    data: Dict[str, any],
    currency_field: str = "currency"
) -> Dict[str, any]:
    """
    Normalize all currency-dependent fields in a financial data dict.

    This is the main entry point for normalizing fetcher outputs.

    Fields normalized (if present):
        - market_cap
        - revenue_ttm / total_revenue
        - free_cash_flow
        - operating_cash_flow
        - volume (if multiplied by price for liquidity)

    Fields left alone (already normalized):
        - pe, pb, peg (ratios)
        - profit_margin, roa, roe (percentages)
        - debt_to_equity, current_ratio (ratios)
        - revenue_growth, eps_growth (percentages)

    Args:
        data: Dict with financial metrics (from yfinance/FMP/EODHD)
        currency_field: Key containing currency code (default "currency")

    Returns:
        Modified dict with normalized values and added metadata fields:
        - _currency_normalized: True if any conversion happened
        - _fx_rate_applied: The rate used (or None)
        - _fx_source: Where the rate came from
        - _original_currency: Original currency before conversion

    Example:
        data = {
            "market_cap": 1.2e12,
            "currency": "HKD",
            "pe": 12.5  # Not touched - it's a ratio
        }
        normalized = await normalize_financial_dict(data)
        # normalized["market_cap"] ≈ 153.6e9 (USD)
        # normalized["_currency_normalized"] = True
        # normalized["pe"] = 12.5 (unchanged)
    """
    currency = data.get(currency_field, "USD")

    # Strip whitespace from currency code
    if currency:
        currency = currency.strip()

    # If already USD or no currency specified, skip normalization
    if not currency or currency.upper() == "USD":
        data["_currency_normalized"] = False
        data["_original_currency"] = "USD"
        return data

    # Get FX rate once for all fields
    fx_rate, source = await get_fx_rate(currency, "USD")

    if fx_rate is None:
        logger.warning(
            "fx_normalization_skipped",
            currency=currency,
            reason="FX rate unavailable - values remain in local currency"
        )
        data["_currency_normalized"] = False
        data["_fx_rate_applied"] = None
        data["_fx_source"] = "unavailable"
        data["_original_currency"] = currency
        return data

    # Fields that need normalization (absolute currency values)
    currency_dependent_fields = [
        "market_cap",
        "marketCap",  # yfinance variant
        "revenue_ttm",
        "totalRevenue",  # yfinance variant
        "free_cash_flow",
        "freeCashflow",  # yfinance variant
        "operating_cash_flow",
        "operatingCashflow",  # yfinance variant
    ]

    normalized_count = 0
    for field in currency_dependent_fields:
        if field in data and data[field] is not None:
            try:
                original_value = float(data[field])
                data[field] = original_value * fx_rate
                normalized_count += 1
                logger.debug(
                    "field_normalized",
                    field=field,
                    original=original_value,
                    normalized=data[field],
                    currency=currency,
                    fx_rate=fx_rate
                )
            except (ValueError, TypeError) as e:
                logger.warning(
                    "field_normalization_failed",
                    field=field,
                    value=data[field],
                    error=str(e)
                )

    # Add metadata
    data["_currency_normalized"] = normalized_count > 0
    data["_fx_rate_applied"] = fx_rate
    data["_fx_source"] = source
    data["_original_currency"] = currency
    data[currency_field] = "USD"  # Update currency field to reflect normalization

    logger.info(
        "financial_dict_normalized",
        original_currency=currency,
        fields_normalized=normalized_count,
        fx_rate=fx_rate,
        source=source
    )

    return data


# ══════════════════════════════════════════════════════════════════════════════
# TEST HELPERS (for development/debugging)
# ══════════════════════════════════════════════════════════════════════════════

async def test_fx_normalization():
    """Test FX normalization with sample data."""
    print("Testing FX Normalization\n")

    test_cases = [
        ("JPY", 1000000, "Japanese Yen"),
        ("HKD", 1.2e12, "Hong Kong Dollar (HSBC-like)"),
        ("TWD", 16e12, "Taiwan Dollar (TSMC-like)"),
        ("EUR", 100e9, "Euro"),
        ("ZZZ", 100, "Invalid currency"),
    ]

    for currency, value, description in test_cases:
        normalized, meta = await normalize_to_usd(value, currency, description)
        print(f"{description}:")
        print(f"  Original: {value:,.0f} {currency}")
        if normalized and meta['fx_rate'] is not None:
            print(f"  USD: ${normalized:,.0f}")
            print(f"  FX Rate: {meta['fx_rate']:.6f} (from {meta['fx_source']})")
        else:
            print(f"  Normalization failed: {meta['fx_source']}")
            print(f"  Returned value: ${normalized:,.0f} (original)" if normalized else "  Returned: None")
        print()


if __name__ == "__main__":
    asyncio.run(test_fx_normalization())
