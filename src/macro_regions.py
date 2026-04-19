"""Canonical macro- and sentiment-region derivation from exchange metadata.

This module exists to stop suffix-to-region logic from drifting across tools,
portfolio helpers, and prompt-injection paths. It owns coarse macro buckets for
cache keying/search hints and finer sentiment-region labels for
``REGION_PLATFORMS`` lookups.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.exchange_metadata import EXCHANGES_BY_SUFFIX
from src.ticker_policy import get_ticker_suffix


@dataclass(frozen=True, slots=True)
class MacroRegionInfo:
    """Normalized region metadata derived from a ticker or exchange suffix."""

    suffix: str
    country: str
    macro_region: str
    sentiment_region: str
    display_region: str
    query_hint: str


@dataclass(frozen=True, slots=True)
class _SuffixOverride:
    country: str
    macro_region: str
    sentiment_region: str
    display_region: str


_COUNTRY_TO_MACRO_REGION: dict[str, str] = {
    "Japan": "JAPAN",
    "Hong Kong": "HONG_KONG",
    "China": "CHINA",
    "Taiwan": "TAIWAN",
    "South Korea": "KOREA",
    "India": "INDIA",
    "Singapore": "SEA",
    "Malaysia": "SEA",
    "Thailand": "SEA",
    "Indonesia": "SEA",
    "Vietnam": "SEA",
    "Australia": "AUSTRALIA",
    "New Zealand": "AUSTRALIA",
    "Canada": "CANADA",
    "UK": "UK",
    "Brazil": "LATAM",
    "Mexico": "LATAM",
    "Germany": "EUROPE",
    "France": "EUROPE",
    "Netherlands": "EUROPE",
    "Belgium": "EUROPE",
    "Portugal": "EUROPE",
    "Italy": "EUROPE",
    "Spain": "EUROPE",
    "Switzerland": "EUROPE",
    "Austria": "EUROPE",
    "Norway": "EUROPE",
    "Sweden": "EUROPE",
    "Finland": "EUROPE",
    "Denmark": "EUROPE",
    "Poland": "EUROPE",
    "Czech Republic": "EUROPE",
    "Hungary": "EUROPE",
    "Romania": "EUROPE",
}

_COUNTRY_TO_SENTIMENT_REGION: dict[str, str] = {
    "Japan": "japan",
    "Hong Kong": "hong_kong",
    "China": "china",
    "Taiwan": "taiwan",
    "South Korea": "south_korea",
    "India": "india",
    "Singapore": "singapore",
    "Malaysia": "malaysia",
    "Thailand": "thailand",
    "Indonesia": "indonesia",
    "Vietnam": "vietnam",
    "Australia": "australia",
    "Canada": "canada",
    "UK": "uk",
    "Brazil": "brazil",
    "Mexico": "mexico",
    "Germany": "germany",
    "France": "france",
    "Netherlands": "unknown",
    "Belgium": "france",
    "Portugal": "portugal",
    "Italy": "unknown",
    "Spain": "spain",
    "Switzerland": "switzerland",
    "Austria": "germany",
    "Norway": "unknown",
    "Sweden": "unknown",
    "Finland": "unknown",
    "Denmark": "denmark",
    "Poland": "poland",
    "Czech Republic": "unknown",
    "Hungary": "unknown",
    "Romania": "unknown",
    "New Zealand": "unknown",
}

_SUFFIX_OVERRIDES: dict[str, _SuffixOverride] = {
    ".VN": _SuffixOverride("Vietnam", "SEA", "vietnam", "Vietnam"),
    ".SR": _SuffixOverride(
        "Saudi Arabia",
        "GLOBAL",
        "middle_east",
        "Middle East",
    ),
    ".QA": _SuffixOverride("Qatar", "GLOBAL", "middle_east", "Middle East"),
    ".AE": _SuffixOverride(
        "United Arab Emirates",
        "GLOBAL",
        "middle_east",
        "Middle East",
    ),
    ".XETRA": _SuffixOverride("Germany", "EUROPE", "germany", "Germany"),
    ".S": _SuffixOverride("Switzerland", "EUROPE", "switzerland", "Switzerland"),
    ".MA": _SuffixOverride("Spain", "EUROPE", "spain", "Spain"),
}

_MACRO_REGION_QUERY_HINTS: dict[str, str] = {
    "JAPAN": "Japan Nikkei BOJ yen JGB inflation wages PMI exports",
    "HONG_KONG": "Hong Kong Hang Seng HKMA HKD property liquidity equity market",
    "CHINA": "China PBOC yuan property credit stimulus PMI growth",
    "TAIWAN": "Taiwan TWD exports electronics central bank PMI",
    "KOREA": "South Korea KOSPI BOK won exports semiconductors PMI credit",
    "INDIA": "India Sensex RBI rupee inflation growth liquidity credit",
    "SEA": "Southeast Asia ASEAN Singapore Malaysia Thailand Indonesia inflation FX trade exports",
    "AUSTRALIA": "Australia ASX RBA AUD iron ore jobs inflation",
    "CANADA": "Canada TSX Bank of Canada CAD oil housing inflation",
    "UK": "UK FTSE Bank of England gilts pound inflation growth",
    "EUROPE": "Europe ECB euro bund growth PMI CPI energy credit",
    "LATAM": "Latin America Brazil Mexico currencies inflation central bank growth",
    "GLOBAL": "global macro central banks inflation growth FX credit liquidity risk appetite equities",
}

MACRO_REGIONS: frozenset[str] = frozenset(_MACRO_REGION_QUERY_HINTS)


def _build_region_info(
    *,
    suffix: str,
    country: str,
    macro_region: str,
    sentiment_region: str,
    display_region: str,
) -> MacroRegionInfo:
    return MacroRegionInfo(
        suffix=suffix,
        country=country,
        macro_region=macro_region,
        sentiment_region=sentiment_region,
        display_region=display_region,
        query_hint=query_hint_for_macro_region(macro_region),
    )


def get_macro_region_info(ticker: str) -> MacroRegionInfo:
    """Return canonical region metadata for a ticker."""
    suffix = get_ticker_suffix(ticker)
    override = _SUFFIX_OVERRIDES.get(suffix)
    if override:
        return _build_region_info(
            suffix=suffix,
            country=override.country,
            macro_region=override.macro_region,
            sentiment_region=override.sentiment_region,
            display_region=override.display_region,
        )

    exchange_info = EXCHANGES_BY_SUFFIX.get(suffix)
    if exchange_info is None:
        return _build_region_info(
            suffix=suffix,
            country="",
            macro_region="GLOBAL",
            sentiment_region="unknown",
            display_region="",
        )

    return _build_region_info(
        suffix=suffix,
        country=exchange_info.country,
        macro_region=_COUNTRY_TO_MACRO_REGION.get(exchange_info.country, "GLOBAL"),
        sentiment_region=_COUNTRY_TO_SENTIMENT_REGION.get(
            exchange_info.country,
            "unknown",
        ),
        display_region=exchange_info.country,
    )


def infer_macro_region(ticker: str) -> str:
    """Return the coarse macro-region bucket for a ticker."""
    return get_macro_region_info(ticker).macro_region


def infer_sentiment_region(ticker: str) -> str:
    """Return the sentiment-region label used by REGION_PLATFORMS lookups."""
    return get_macro_region_info(ticker).sentiment_region


def display_region_for_suffix(suffix: str) -> str:
    """Return a human-readable region label for a raw suffix like '.T'."""
    normalized_suffix = (suffix or "").strip().upper()
    override = _SUFFIX_OVERRIDES.get(normalized_suffix)
    if override:
        return override.display_region

    exchange_info = EXCHANGES_BY_SUFFIX.get(normalized_suffix)
    return exchange_info.country if exchange_info else ""


def query_hint_for_macro_region(region: str) -> str:
    """Return macro search hint terms for a region bucket."""
    normalized_region = (region or "GLOBAL").strip().upper() or "GLOBAL"
    return _MACRO_REGION_QUERY_HINTS.get(
        normalized_region,
        _MACRO_REGION_QUERY_HINTS["GLOBAL"],
    )
