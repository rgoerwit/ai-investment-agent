"""Canonical sector normalization for portfolio-facing workflows."""

from __future__ import annotations

_CANONICAL_GICS_SECTORS = {
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
}

_ALIAS_TO_CANONICAL = {
    "energy": "Energy",
    "materials": "Materials",
    "basic materials": "Materials",
    "industrials": "Industrials",
    "consumer discretionary": "Consumer Discretionary",
    "consumer cyclical": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "consumer defensive": "Consumer Staples",
    "health care": "Health Care",
    "healthcare": "Health Care",
    "financials": "Financials",
    "financial services": "Financials",
    "finance": "Financials",
    "information technology": "Information Technology",
    "information tech": "Information Technology",
    "technology": "Information Technology",
    "tech": "Information Technology",
    "communication services": "Communication Services",
    "telecom": "Communication Services",
    "telecommunications": "Communication Services",
    "utilities": "Utilities",
    "real estate": "Real Estate",
}


def _normalize_lookup_key(raw: str | None) -> str:
    """Collapse formatting drift before exact alias lookup."""
    return " ".join((raw or "").split()).strip().casefold()


def normalize_sector_label(raw: str | None) -> str:
    """Return the canonical GICS sector name for a portfolio-facing sector label."""
    lookup = _normalize_lookup_key(raw)
    if not lookup:
        return "Unknown"
    return _ALIAS_TO_CANONICAL.get(lookup, "Unknown")


def aggregate_sector_weights(
    weights: dict[str, float] | None,
) -> dict[str, float]:
    """Merge sector buckets using canonical sector labels."""
    aggregated: dict[str, float] = {}
    for sector, pct in (weights or {}).items():
        label = normalize_sector_label(sector)
        aggregated[label] = aggregated.get(label, 0.0) + pct
    return aggregated


__all__ = [
    "aggregate_sector_weights",
    "normalize_sector_label",
]
