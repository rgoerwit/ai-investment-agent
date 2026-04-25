"""
Reconciler: Compare IBKR positions vs evaluator recommendations.

The equity evaluator produces one-off BUY/SELL/DNI verdicts per ticker,
unaware of existing positions. This module keeps the orchestration layer that
turns those verdicts into position-aware portfolio actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
)
from src.ibkr.opportunity_finder import find_opportunities
from src.ibkr.position_evaluator import evaluate_positions
from src.ibkr.watchlist_evaluator import evaluate_watchlist
from src.ticker_policy import is_safe_symbol_crossmatch_base, split_ticker

logger = structlog.get_logger(__name__)


@dataclass
class ReconciliationDiagnostics:
    """Diagnostic counters accumulated during reconciliation."""

    cash_blocked_offwatch_buy_count: int = 0


def _build_alpha_base_lookup(
    analyses: dict[str, AnalysisRecord],
) -> tuple[dict[str, AnalysisRecord], dict[str, str]]:
    """Build base-symbol lookup tables for safe alphabetic cross-format matching."""
    alpha_base_lookup: dict[str, AnalysisRecord] = {}
    alpha_base_to_key: dict[str, str] = {}
    for yf_ticker, record in analyses.items():
        base, _suffix = split_ticker(yf_ticker)
        if is_safe_symbol_crossmatch_base(base):
            if "." in yf_ticker:
                alpha_base_lookup[base] = record
                alpha_base_to_key[base] = yf_ticker
            else:
                alpha_base_lookup.setdefault(base, record)
                alpha_base_to_key.setdefault(base, yf_ticker)
    return alpha_base_lookup, alpha_base_to_key


def _load_structural_macro_events() -> list:
    """Best-effort fetch of recent structural macro events for staleness invalidation."""
    structural_events: list = []
    try:
        from datetime import datetime as _dt
        from datetime import timedelta

        from src.memory import create_macro_events_store

        store = create_macro_events_store()
        if store.available:
            structural_events = store.get_structural_events_since(
                (_dt.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            )
    except Exception:
        pass
    return structural_events


def _populate_portfolio_weights(
    positions: list[NormalizedPosition],
    analyses: dict[str, AnalysisRecord],
    portfolio: PortfolioSummary,
    alpha_base_lookup: dict[str, AnalysisRecord],
) -> tuple[dict[str, float], dict[str, float]]:
    """Populate sector/exchange concentration weights on the portfolio."""
    from src.ibkr.reconciliation_rules import _exchange_from_position

    sector_weights: dict[str, float] = {}
    exchange_weights: dict[str, float] = {}
    total_position_value = sum(position.market_value_usd for position in positions)
    if total_position_value > 0:
        for pos in positions:
            current_ticker = pos.ticker.yf
            analysis = analyses.get(current_ticker)
            if (
                analysis is None
                and not pos.ticker.has_suffix
                and not pos.ticker.ibkr.isdigit()
            ):
                best = alpha_base_lookup.get(pos.ticker.ibkr.upper())
                if best and "." in best.ticker:
                    current_ticker = best.ticker
                    analysis = best
            sector = (analysis.sector if analysis else "") or "Unknown"
            exchange = _exchange_from_position(pos)
            weight = pos.market_value_usd / total_position_value * 100
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
            exchange_weights[exchange] = exchange_weights.get(exchange, 0.0) + weight

    portfolio.sector_weights = sector_weights
    portfolio.exchange_weights = exchange_weights
    return sector_weights, exchange_weights


def reconcile(
    positions: list[NormalizedPosition],
    analyses: dict[str, AnalysisRecord],
    portfolio: PortfolioSummary,
    max_age_days: int = 14,
    drift_threshold_pct: float = 15.0,
    overweight_threshold_pct: float = 20.0,
    underweight_threshold_pct: float = 20.0,
    sector_limit_pct: float = 30.0,
    exchange_limit_pct: float = 40.0,
    watchlist_tickers: set[str] | None = None,
    diagnostics: ReconciliationDiagnostics | None = None,
) -> list[ReconciliationItem]:
    """
    Compare IBKR positions against evaluator recommendations.

    Returns position-aware actions while preserving existing reconciliation behavior.
    """
    alpha_base_lookup, alpha_base_to_key = _build_alpha_base_lookup(analyses)
    structural_events = _load_structural_macro_events()
    sector_weights, exchange_weights = _populate_portfolio_weights(
        positions,
        analyses,
        portfolio,
        alpha_base_lookup,
    )

    remaining_cash = portfolio.available_cash_usd

    items, held_tickers, remaining_cash = evaluate_positions(
        positions,
        analyses,
        portfolio,
        alpha_base_lookup=alpha_base_lookup,
        structural_macro_events=structural_events,
        max_age_days=max_age_days,
        drift_threshold_pct=drift_threshold_pct,
        overweight_threshold_pct=overweight_threshold_pct,
        underweight_threshold_pct=underweight_threshold_pct,
        sector_limit_pct=sector_limit_pct,
        exchange_limit_pct=exchange_limit_pct,
        sector_weights=sector_weights,
        exchange_weights=exchange_weights,
        remaining_cash=remaining_cash,
    )

    watchlist_items, held_tickers, remaining_cash = evaluate_watchlist(
        watchlist_tickers,
        held_tickers,
        analyses,
        portfolio,
        alpha_base_lookup=alpha_base_lookup,
        alpha_base_to_key=alpha_base_to_key,
        structural_macro_events=structural_events,
        max_age_days=max_age_days,
        drift_threshold_pct=drift_threshold_pct,
        sector_limit_pct=sector_limit_pct,
        exchange_limit_pct=exchange_limit_pct,
        sector_weights=sector_weights,
        exchange_weights=exchange_weights,
        remaining_cash=remaining_cash,
    )
    items.extend(watchlist_items)

    opportunity_items, remaining_cash = find_opportunities(
        analyses,
        held_tickers,
        portfolio,
        diagnostics=diagnostics,
        structural_macro_events=structural_events,
        max_age_days=max_age_days,
        drift_threshold_pct=drift_threshold_pct,
        sector_limit_pct=sector_limit_pct,
        exchange_limit_pct=exchange_limit_pct,
        sector_weights=sector_weights,
        exchange_weights=exchange_weights,
        remaining_cash=remaining_cash,
    )
    items.extend(opportunity_items)

    urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    items.sort(key=lambda item: urgency_order.get(item.urgency, 9))

    logger.info(
        "reconciliation_complete",
        total_items=len(items),
        sells=sum(1 for item in items if item.action == "SELL"),
        trims=sum(1 for item in items if item.action == "TRIM"),
        adds=sum(1 for item in items if item.action == "ADD"),
        buys=sum(1 for item in items if item.action == "BUY"),
        holds=sum(1 for item in items if item.action == "HOLD"),
        reviews=sum(1 for item in items if item.action == "REVIEW"),
        removes=sum(1 for item in items if item.action == "REMOVE"),
    )

    return items
