"""Exchange-qualified ticker policy helpers.

This module keeps a small set of ticker-shape rules in one place so that
callers do not quietly drift into exchange-unsafe assumptions.
"""

from __future__ import annotations

CHINA_SUFFIXES = frozenset({".HK", ".SS", ".SZ"})
KOREA_SUFFIXES = frozenset({".KS", ".KQ"})
TAIWAN_SUFFIXES = frozenset({".TW", ".TWO"})
INDIA_SUFFIXES = frozenset({".NS", ".BO"})
SEARCH_RESOLUTION_SUFFIXES = frozenset({".HK", ".KL", ".TW", ".TWO"})
FRAGILE_EXCHANGE_SUFFIXES = frozenset({".HK", ".TW", ".TWO", ".KS", ".T", ".L"})


def get_ticker_suffix(ticker: str) -> str:
    """Return the final suffix segment, including the leading dot."""
    cleaned = ticker.strip().upper()
    if "." not in cleaned:
        return ""
    return f".{cleaned.rsplit('.', 1)[1]}"


def split_ticker(ticker: str) -> tuple[str, str]:
    """Split a ticker into base symbol and normalized suffix."""
    cleaned = ticker.strip().upper()
    suffix = get_ticker_suffix(cleaned)
    if not suffix:
        return cleaned, ""
    return cleaned[: -len(suffix)], suffix


def is_pure_numeric_base(base: str) -> bool:
    """Return True when the base symbol is all digits."""
    return bool(base) and base.isdigit()


def is_safe_symbol_crossmatch_base(base: str) -> bool:
    """Return True when a base symbol is safe for suffix-agnostic matching."""
    return bool(base) and not is_pure_numeric_base(base)


def allows_search_resolution(ticker: str) -> bool:
    """Return True when search-based symbol rescue is allowed for this ticker."""
    return get_ticker_suffix(ticker) in SEARCH_RESOLUTION_SUFFIXES


def normalize_exchange_specific_base(base: str, suffix: str) -> str:
    """Normalize a base symbol only where exchange rules are well-defined."""
    normalized_base = base.strip().upper()
    normalized_suffix = suffix.strip().upper()
    if normalized_suffix == ".HK" and normalized_base.isdigit():
        return normalized_base.zfill(4)
    return normalized_base


def same_exchange(ticker_a: str, ticker_b: str) -> bool:
    """Return True when two tickers share the same explicit exchange suffix."""
    return get_ticker_suffix(ticker_a) == get_ticker_suffix(ticker_b)


def ticker_in_group(ticker: str, suffixes: frozenset[str]) -> bool:
    """Return True when the ticker's final suffix belongs to the given set."""
    return get_ticker_suffix(ticker) in suffixes
