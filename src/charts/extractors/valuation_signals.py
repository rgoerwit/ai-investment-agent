"""Shared valuation-quality signals used by prompt context and IV math."""

from __future__ import annotations

VALUATION_NORMALIZE_TOKENS = (
    "NORMALIZE EARNINGS",
    "RECURRING PROFIT LOWER THAN REPORTED",
    "NON-RECURRING",
    "ONE-TIME",
    "ONE-OFF",
    "TRANSIENT_STRENGTH_DISTORTION",
)

VALUATION_PEAK_TOKENS = (
    "CYCLICAL PEAK",
    "PEAK-CYCLE",
    "LOW P/E MAY BE PEAK-DISTORTED",
    "CYCLE_POSITION: PEAK",
)

VALUATION_CONTEXT_TOKENS = (
    *VALUATION_NORMALIZE_TOKENS,
    *VALUATION_PEAK_TOKENS,
    "THIN_CONSENSUS",
)


def has_valuation_signal(text: str, tokens: tuple[str, ...]) -> bool:
    """Return True when any valuation-quality token appears in text."""
    upper = text.upper()
    return any(token in upper for token in tokens)
