"""Canonical concentration bucket and projected-breach policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.exchange_metadata import IBKR_TO_YFINANCE
from src.ibkr.models import NormalizedPosition
from src.ibkr.reconciliation_rules import (
    _exchange_from_position,
    _exchange_from_ticker,
)
from src.ibkr.ticker import Ticker
from src.sector_normalization import normalize_sector_label


@dataclass(frozen=True)
class ConcentrationBreach:
    """One dimension's projected-weight limit breach for a candidate."""

    dimension: str
    key: str
    candidate_pct: float
    projected_pct: float
    limit_pct: float


def canonical_exchange_bucket(
    ticker: str | Ticker,
    *,
    analysis_exchange: str | None = None,
    position: NormalizedPosition | None = None,
) -> str:
    """Return the suffix-space exchange key used by portfolio weights."""
    yf_ticker = ticker.yf if isinstance(ticker, Ticker) else ticker
    if "." in yf_ticker:
        return _exchange_from_ticker(yf_ticker)
    if position is not None:
        return _exchange_from_position(position)

    exchange = (analysis_exchange or "").strip().upper()
    mapped = IBKR_TO_YFINANCE.get(exchange)
    if mapped is not None:
        return mapped.lstrip(".") if mapped else "US"
    return exchange.lstrip(".") or "US"


def canonical_sector_bucket(sector: str | None) -> str | None:
    """Return a canonical GICS bucket, or None when sector is unknown."""
    normalized = normalize_sector_label(sector)
    return None if normalized == "Unknown" else normalized


def project_concentration_breaches(
    *,
    exchange_key: str | None,
    sector_key: str | None,
    candidate_pct: float,
    exchange_weights: Mapping[str, float],
    sector_weights: Mapping[str, float],
    exchange_limit_pct: float,
    sector_limit_pct: float,
) -> tuple[ConcentrationBreach, ...]:
    """Return every bucket that a non-negative candidate weight would breach."""
    candidate_pct = max(float(candidate_pct), 0.0)
    breaches: list[ConcentrationBreach] = []
    dimensions = (
        ("exchange", exchange_key, exchange_weights, exchange_limit_pct),
        ("sector", sector_key, sector_weights, sector_limit_pct),
    )
    for dimension, key, weights, limit_pct in dimensions:
        if key is None or not weights:
            continue
        projected_pct = weights.get(key, 0.0) + candidate_pct
        if projected_pct > limit_pct:
            breaches.append(
                ConcentrationBreach(
                    dimension=dimension,
                    key=key,
                    candidate_pct=candidate_pct,
                    projected_pct=projected_pct,
                    limit_pct=limit_pct,
                )
            )
    return tuple(breaches)


def format_concentration_warnings(
    breaches: tuple[ConcentrationBreach, ...],
) -> tuple[str, ...]:
    """Return the evaluator warning fragments used in reconciliation reasons."""
    return tuple(
        (
            f"⚠ {breach.key} → {breach.projected_pct:.0f}% "
            f"(limit {breach.limit_pct:.0f}%)"
            if breach.dimension == "exchange"
            else f"⚠ {breach.key} sector → {breach.projected_pct:.0f}% "
            f"(limit {breach.limit_pct:.0f}%)"
        )
        for breach in breaches
    )
