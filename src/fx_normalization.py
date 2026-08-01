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
import time
from collections.abc import Iterable
from typing import Any

import structlog

from src.blocking_io import FX_RATE_POLICY, run_blocking_call
from src.error_safety import summarize_exception

logger = structlog.get_logger(__name__)

MINOR_UNIT_CURRENCY_ALIASES = {
    "GBp": "GBP",
    "GBX": "GBP",
}

MINOR_UNIT_SCALE = {
    "GBp": 0.01,
    "GBX": 0.01,
}

# ══════════════════════════════════════════════════════════════════════════════
# TIER 1: Dynamic FX Rates (yfinance - always up-to-date)
# ══════════════════════════════════════════════════════════════════════════════


async def get_fx_rate_yfinance(
    from_currency: str, to_currency: str = "USD"
) -> float | None:
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
    # Normalize to uppercase so "jpy" → "JPY" works for direct callers
    from_currency = from_currency.strip().upper()
    to_currency = to_currency.strip().upper()

    if from_currency == to_currency:
        return 1.0

    # yfinance forex ticker format: "JPYUSD=X" (from + to + =X)
    fx_ticker = f"{from_currency}{to_currency}=X"

    try:
        import yfinance as yf

        from src.yfinance_runtime import YFRateLimitError, configure_yfinance_defaults

        configure_yfinance_defaults()

        # Use async thread pool to avoid blocking
        def _fetch_rate():
            ticker = yf.Ticker(fx_ticker)
            # Try fast_info first (faster), but fall through when it returns None
            # (fast_info.last_price can be None before yfinance warms up its cache)
            if hasattr(ticker, "fast_info") and hasattr(ticker.fast_info, "last_price"):
                fast_price = ticker.fast_info.last_price
                if fast_price:
                    return fast_price
            # Fallback to info dict
            info = ticker.info
            return info.get("regularMarketPrice") or info.get("previousClose")

        rate = await run_blocking_call(
            FX_RATE_POLICY.with_label(f"fx_rate:{fx_ticker}"),
            _fetch_rate,
        )

        if rate and rate > 0:
            logger.debug(
                "fx_rate_fetched", pair=fx_ticker, rate=rate, source="yfinance"
            )
            return float(rate)
        else:
            logger.debug("fx_rate_invalid", pair=fx_ticker, rate=rate)
            return None

    except asyncio.TimeoutError:
        logger.debug(
            "fx_rate_timeout",
            pair=fx_ticker,
            timeout_ms=int(FX_RATE_POLICY.hard_timeout_seconds * 1000),
        )
        return None
    except YFRateLimitError as e:
        logger.debug("fx_rate_rate_limited", pair=fx_ticker, error=str(e))
        return None
    except Exception as e:
        logger.debug("fx_rate_fetch_error", pair=fx_ticker, error=str(e))
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2: Fallback Rates (Hardcoded, updated quarterly)
# ══════════════════════════════════════════════════════════════════════════════

# Last updated: Jul 2026
# Source: Yahoo Finance FX spot quotes (query1.finance.yahoo.com/v8/finance/chart)
FALLBACK_RATES_TO_USD = {
    # Major Asian currencies (your primary use case)
    "JPY": 0.0063,  # Japanese Yen (¥159 = $1)
    "HKD": 0.1275,  # Hong Kong Dollar (HK$7.84 = $1)
    "TWD": 0.0308,  # Taiwan Dollar (NT$32.5 = $1)
    "KRW": 0.0007,  # Korean Won (₩1,421 = $1)
    "CNY": 0.148,  # Chinese Yuan (¥6.74 = $1)
    "INR": 0.0105,  # Indian Rupee (₹95.2 = $1)
    "SGD": 0.780,  # Singapore Dollar (S$1.28 = $1)
    "MYR": 0.245,  # Malaysian Ringgit (MYR 4.08 = $1)
    "THB": 0.030,  # Thai Baht (THB 33.3 = $1)
    "IDR": 0.0000553,  # Indonesian Rupiah (IDR 18,080 = $1)
    "PHP": 0.0163,  # Philippine Peso (PHP 61.3 = $1)
    # European currencies
    "EUR": 1.153,  # Euro
    "GBP": 1.347,  # British Pound
    "CHF": 1.243,  # Swiss Franc
    "SEK": 0.105,  # Swedish Krona (SEK 9.52 = $1)
    "NOK": 0.105,  # Norwegian Krone (NOK 9.52 = $1)
    "DKK": 0.154,  # Danish Krone (DKK 6.49 = $1)
    "PLN": 0.268,  # Polish Zloty (PLN 3.74 = $1)
    "CZK": 0.0477,  # Czech Koruna (CZK 21.0 = $1)
    "HUF": 0.00318,  # Hungarian Forint (HUF 314 = $1)
    # Other major currencies
    "CAD": 0.715,  # Canadian Dollar
    "AUD": 0.703,  # Australian Dollar
    "NZD": 0.588,  # New Zealand Dollar
    "MXN": 0.0576,  # Mexican Peso
    "BRL": 0.197,  # Brazilian Real
    "ZAR": 0.0606,  # South African Rand (ZAR 16.5 = $1)
    "ILS": 0.327,  # Israeli Shekel (ILS 3.06 = $1)
    # Identity
    "USD": 1.0,
}


def normalize_minor_unit_currency(currency: str | None) -> tuple[str | None, float]:
    """Return major-unit currency code and scale factor for minor-unit aliases."""
    if not currency:
        return currency, 1.0
    normalized = MINOR_UNIT_CURRENCY_ALIASES.get(currency)
    if normalized is None:
        return currency, 1.0
    return normalized, MINOR_UNIT_SCALE[currency]


def normalize_minor_unit_amount(
    value: float | None, currency: str | None
) -> tuple[float | None, str | None, float]:
    """Scale a monetary value to the major currency unit when needed."""
    normalized_currency, scale = normalize_minor_unit_currency(currency)
    if value is None:
        return value, normalized_currency, scale
    return value * scale, normalized_currency, scale


def is_near_minor_unit_ratio(ratio: float, tolerance: float = 0.10) -> bool:
    """Return True when a ratio is close to a 100x minor-unit mismatch."""
    return (100 * (1 - tolerance)) < ratio < (100 * (1 + tolerance))


def get_fx_rate_fallback(from_currency: str, to_currency: str = "USD") -> float | None:
    """
    Get FX rate from hardcoded fallback table.

    WARNING: These rates are updated manually and may be stale.
    Only use when yfinance is unavailable. This is a low-level primitive —
    callers that want a deduplicated, actionable log signal when the table
    is used should go through get_fx_rate() (Tier 3) or FxRateCache, both of
    which log once per resolution rather than once per call.

    FALLBACK_RATES_TO_USD is USD-anchored (each entry is "1 unit = $X"), so a
    non-USD to_currency is resolved as a cross-rate through USD
    (from->USD / to->USD), not a direct table lookup — the table has no
    entry for e.g. "EUR->GBP". Returns None (never a mislabeled USD rate)
    when either leg is missing from the table.
    """
    if from_currency == to_currency:
        return 1.0

    normalized_currency, scale = normalize_minor_unit_currency(from_currency)
    if scale != 1.0:
        major_rate = (
            FALLBACK_RATES_TO_USD.get(normalized_currency)
            if normalized_currency is not None
            else None
        )
        from_rate_to_usd = major_rate * scale if major_rate else None
    else:
        from_rate_to_usd = FALLBACK_RATES_TO_USD.get(from_currency)

    if not from_rate_to_usd:
        return None

    if to_currency == "USD":
        fallback_rate = from_rate_to_usd
    else:
        to_rate_to_usd = FALLBACK_RATES_TO_USD.get(to_currency)
        if not to_rate_to_usd:
            logger.debug(
                "fx_rate_fallback_cross_rate_unavailable",
                from_currency=from_currency,
                to_currency=to_currency,
            )
            return None
        fallback_rate = from_rate_to_usd / to_rate_to_usd

    logger.debug(
        "fx_rate_using_fallback",
        from_currency=from_currency,
        to_currency=to_currency,
        rate=fallback_rate,
    )
    return fallback_rate


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3: Unified Interface with Smart Fallback
# ══════════════════════════════════════════════════════════════════════════════


async def get_fx_rate(
    from_currency: str, to_currency: str = "USD", allow_fallback: bool = True
) -> tuple[float | None, str]:
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
            logger.warning(
                "fx_rate_fallback_used",
                currency=from_currency,
                rate=rate,
                msg=(
                    "Live yfinance FX fetch failed — used FALLBACK_RATES_TO_USD "
                    "(src/fx_normalization.py); refresh the table if this persists."
                ),
            )
            return rate, "fallback"

    # Total failure - log and return None
    logger.warning(
        "fx_rate_unavailable",
        from_currency=from_currency,
        to_currency=to_currency,
        tried_sources=["yfinance", "fallback"] if allow_fallback else ["yfinance"],
    )
    return None, "unavailable"


# ══════════════════════════════════════════════════════════════════════════════
# TIER 4: Process-Wide Cache — dedupe repeated lookups of the same currency
# ══════════════════════════════════════════════════════════════════════════════

_FX_RATE_CACHE_TTL_SECONDS = 3600.0  # FX doesn't move enough intraday to matter here


class FxRateCache:
    """Process-wide cache over the live-first get_fx_rate() chain.

    Without this, a portfolio with N positions in the same currency issues N
    independent live-then-fallback lookups for that one currency. This caches
    by currency pair (from_currency, to_currency) — not by position — so each
    pair is resolved at most once per TTL window, and batches a whole currency
    set concurrently.
    """

    def __init__(
        self,
        *,
        cache_ttl_secs: float = _FX_RATE_CACHE_TTL_SECONDS,
        max_concurrency: int = 6,
    ) -> None:
        self._cache_ttl_secs = cache_ttl_secs
        self._max_concurrency = max_concurrency
        # (currency, to_currency) -> (expires_at, rate, source)
        self._cache: dict[tuple[str, str], tuple[float, float, str]] = {}

    def _cached(self, currency: str, to_currency: str) -> tuple[float, str] | None:
        entry = self._cache.get((currency, to_currency))
        if entry is None:
            return None
        expires_at, rate, source = entry
        if time.monotonic() >= expires_at:
            self._cache.pop((currency, to_currency), None)
            return None
        return rate, source

    def _store(self, currency: str, to_currency: str, rate: float, source: str) -> None:
        self._cache[(currency, to_currency)] = (
            time.monotonic() + self._cache_ttl_secs,
            rate,
            source,
        )

    async def _resolve_and_cache(
        self, currency: str, to_currency: str
    ) -> tuple[float | None, str]:
        """Resolve one currency live-first and cache it.

        get_fx_rate() already does live-then-fallback for this single currency
        (and logs one actionable fx_rate_fallback_used signal on fallback), so
        each currency resolves independently — one pair's live outage never
        downgrades another pair that could resolve live.
        """
        rate, source = await get_fx_rate(currency, to_currency)
        if rate is not None:
            self._store(currency, to_currency, rate, source)
        return rate, source

    async def get_rate(
        self, from_currency: str, to_currency: str = "USD"
    ) -> tuple[float | None, str]:
        """Resolve one currency, live-first, using the process-wide cache."""
        from_currency = from_currency.strip().upper()
        to_currency = to_currency.strip().upper()
        if from_currency == to_currency:
            return 1.0, "identity"

        cached = self._cached(from_currency, to_currency)
        if cached is not None:
            return cached

        return await self._resolve_and_cache(from_currency, to_currency)

    async def get_rates(
        self, currencies: Iterable[str], to_currency: str = "USD"
    ) -> dict[str, tuple[float, str]]:
        """Resolve many currencies concurrently, deduped against the cache.

        Each pending currency is resolved independently, live-first, under a
        concurrency bound. There is deliberately no shared preflight/health
        gate: one currency pair's live-FX outage must never force other pairs
        onto the stale fallback table when they could resolve live. During a
        genuine total yfinance outage every pair independently (and
        concurrently) falls back via get_fx_rate(), so the wall-clock cost is
        one timeout per concurrency batch, not one per currency serially.
        """
        to_currency = to_currency.strip().upper()
        unique = {c.strip().upper() for c in currencies if c and c.strip()}
        unique.discard(to_currency)
        if not unique:
            return {}

        resolved: dict[str, tuple[float, str]] = {}
        pending: list[str] = []
        # Sorted so resolution order (and result insertion order) is deterministic
        # regardless of the input set's iteration order.
        for currency in sorted(unique):
            cached = self._cached(currency, to_currency)
            if cached is not None:
                resolved[currency] = cached
            else:
                pending.append(currency)
        if not pending:
            return resolved

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _resolve(currency: str) -> tuple[str, float | None, str]:
            async with semaphore:
                rate, source = await self._resolve_and_cache(currency, to_currency)
                return currency, rate, source

        for currency, rate, source in await asyncio.gather(
            *(_resolve(currency) for currency in pending)
        ):
            if rate is not None:
                resolved[currency] = (rate, source)

        return resolved

    def resolve_rates_sync(
        self, currencies: Iterable[str], to_currency: str = "USD"
    ) -> dict[str, tuple[float, str]]:
        """Sync wrapper for callers outside a running event loop (e.g. the
        IBKR reconciliation path). Must not be called from inside a running
        loop — use get_rates()/get_rate() directly there instead."""
        return asyncio.run(self.get_rates(currencies, to_currency))

    def peek_cached_rate(
        self, currency: str, to_currency: str = "USD"
    ) -> tuple[float, str] | None:
        """Return an already-cached rate with no I/O, or None on a cache miss.

        For sync callers invoked once per ticker (e.g. reconciliation)
        rather than once per batch — avoids paying asyncio.run() startup
        cost on every call once the currency is warm in the cache.
        """
        currency = currency.strip().upper()
        to_currency = to_currency.strip().upper()
        if currency == to_currency:
            return 1.0, "identity"
        return self._cached(currency, to_currency)


# Shared, lazily-constructed singleton so every consumer (IBKR position
# valuation, reconciliation) shares ONE cache — a currency resolved once is
# reused for the rest of the run instead of being fetched per position/ticker.
_fx_rate_cache: FxRateCache | None = None


def get_fx_rate_cache() -> FxRateCache:
    """Return the process-wide shared FX rate cache."""
    global _fx_rate_cache
    if _fx_rate_cache is None:
        _fx_rate_cache = FxRateCache()
    return _fx_rate_cache


def set_fx_rate_cache(cache: FxRateCache | None) -> None:
    """Override (or reset with None) the shared cache — for tests."""
    global _fx_rate_cache
    _fx_rate_cache = cache


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS: Normalize Specific Metric Types
# ══════════════════════════════════════════════════════════════════════════════


async def normalize_to_usd(
    value: float | None, currency: str, metric_name: str = "value"
) -> tuple[float | None, dict[str, Any]]:
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
        "normalized": False,
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
            reason="No FX rate available",
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
        source=source,
    )

    return normalized_value, metadata


async def normalize_financial_dict(
    data: dict[str, Any], currency_field: str = "currency"
) -> dict[str, Any]:
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
    raw_currency = data.get(currency_field, "USD")
    currency = raw_currency if isinstance(raw_currency, str) else "USD"

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
            reason="FX rate unavailable - values remain in local currency",
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
                    fx_rate=fx_rate,
                )
            except (ValueError, TypeError) as e:
                logger.warning(
                    "field_normalization_failed",
                    field=field,
                    value=data[field],
                    **summarize_exception(e, operation="field_normalization_failed"),
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
        source=source,
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
        if normalized and meta["fx_rate"] is not None:
            print(f"  USD: ${normalized:,.0f}")
            print(f"  FX Rate: {meta['fx_rate']:.6f} (from {meta['fx_source']})")
        else:
            print(f"  Normalization failed: {meta['fx_source']}")
            print(
                f"  Returned value: ${normalized:,.0f} (original)"
                if normalized
                else "  Returned: None"
            )
        print()


if __name__ == "__main__":
    asyncio.run(test_fx_normalization())
