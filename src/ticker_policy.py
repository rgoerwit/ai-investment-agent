"""Exchange-qualified ticker policy helpers.

This module keeps a small set of ticker-shape rules in one place so that
callers do not quietly drift into exchange-unsafe assumptions.
"""

from __future__ import annotations

_SEARCH_RESOLUTION_SUFFIXES = frozenset({".HK", ".KL", ".TW", ".TWO"})


def split_ticker(ticker: str) -> tuple[str, str]:
    """Split a ticker into base symbol and normalized suffix."""
    cleaned = ticker.strip().upper()
    if "." not in cleaned:
        return cleaned, ""
    base, suffix = cleaned.rsplit(".", 1)
    return base, f".{suffix}"


def is_pure_numeric_base(base: str) -> bool:
    """Return True when the base symbol is all digits."""
    return bool(base) and base.isdigit()


def is_safe_symbol_crossmatch_base(base: str) -> bool:
    """Return True when a base symbol is safe for suffix-agnostic matching."""
    return bool(base) and not is_pure_numeric_base(base)


def allows_search_resolution(ticker: str) -> bool:
    """Return True when search-based symbol rescue is allowed for this ticker."""
    _, suffix = split_ticker(ticker)
    return suffix in _SEARCH_RESOLUTION_SUFFIXES


def normalize_exchange_specific_base(base: str, suffix: str) -> str:
    """Normalize a base symbol only where exchange rules are well-defined."""
    normalized_base = base.strip().upper()
    normalized_suffix = suffix.strip().upper()
    if normalized_suffix == ".HK" and normalized_base.isdigit():
        return normalized_base.zfill(4)
    return normalized_base


def same_exchange(ticker_a: str, ticker_b: str) -> bool:
    """Return True when two tickers share the same explicit exchange suffix."""
    _, suffix_a = split_ticker(ticker_a)
    _, suffix_b = split_ticker(ticker_b)
    return suffix_a == suffix_b
