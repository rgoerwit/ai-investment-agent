"""Currency resolution for tickers based on suffix and provider metadata."""

from dataclasses import dataclass
from datetime import date
from typing import Literal

from src.exchange_metadata import SUFFIX_TO_CURRENCY_CODE, US_IBKR_EXCHANGES
from src.ticker_policy import get_ticker_suffix

ResolutionSource = Literal[
    "exchange_suffix",  # ".MX" -> MXN - high confidence
    "ibkr_exchange",  # IBKR contract metadata
    "provider_currency",  # yfinance/eodhd reported currency
    "fallback",  # USD default for bare US tickers
    "unresolved",  # could not determine; do NOT silently default
]


@dataclass(frozen=True, slots=True)
class CurrencyResolution:
    """The resolved currency and confidence metadata."""

    code: str | None
    source: ResolutionSource
    confidence: Literal["high", "medium", "low"]
    conflict_warning: str | None = None


def resolve_local_trading_currency(
    *,
    ticker: str | None,
    ibkr_exchange: str | None = None,
    provider_currency: str | None = None,
    as_of: date | None = None,
) -> CurrencyResolution:
    """Resolve the currency a position trades in *locally*.

    Priority for SUFFIXED tickers (e.g. PINFRA.MX):
      1. Exchange-suffix lookup wins — high confidence.
      2. Provider currency is corroborating evidence; if it disagrees with
         the suffix, log a conflict but trust the suffix.
    Priority for BARE tickers (e.g. AAPL, APR):
      1. Provider currency wins (no canonical authority).
      2. Fall back to USD ONLY for known US listings (ibkr_exchange in
         {NYSE, NASDAQ, ARCA, AMEX, IEXG, CBOE, SMART}).
    Unknown / malformed tickers: return ``code=None, source="unresolved"``.
    Callers MUST decide what to do with unresolved — never silently coerce
    to USD downstream.
    """
    if not ticker:
        return CurrencyResolution(code=None, source="unresolved", confidence="low")

    suffix = get_ticker_suffix(ticker)

    # Provider currency normalization
    normalized_provider_currency = (
        provider_currency.upper() if provider_currency else None
    )

    if suffix:
        # 1. SUFFIXED TICKER
        canonical_currency = SUFFIX_TO_CURRENCY_CODE.get(suffix)
        if canonical_currency:
            conflict = None
            if (
                normalized_provider_currency
                and normalized_provider_currency != canonical_currency
            ):
                conflict = (
                    f"Canonical suffix {suffix} implies {canonical_currency}, "
                    f"but provider reported {normalized_provider_currency}"
                )
            return CurrencyResolution(
                code=canonical_currency,
                source="exchange_suffix",
                confidence="high",
                conflict_warning=conflict,
            )

        # Suffix is present but unknown to our canonical map
        # We don't have high confidence, but provider might know
        if normalized_provider_currency:
            return CurrencyResolution(
                code=normalized_provider_currency,
                source="provider_currency",
                confidence="medium",
                conflict_warning=f"Unknown suffix {suffix}, trusting provider currency",
            )

        return CurrencyResolution(code=None, source="unresolved", confidence="low")

    else:
        # 2. BARE TICKER
        if normalized_provider_currency:
            return CurrencyResolution(
                code=normalized_provider_currency,
                source="provider_currency",
                confidence="medium",
            )

        if ibkr_exchange and ibkr_exchange.upper() in US_IBKR_EXCHANGES:
            return CurrencyResolution(
                code="USD",
                source="fallback",
                confidence="low",
            )

        # Bare ticker, no provider info, not a known US exchange -> Unresolved
        return CurrencyResolution(code=None, source="unresolved", confidence="low")
