"""Phase 2 off-watchlist BUY opportunity discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.ibkr.models import (
    AnalysisRecord,
    PortfolioSummary,
    ReconciliationItem,
)
from src.ibkr.order_builder import calculate_quantity
from src.ibkr.reconciliation_rules import (
    _MIN_ORDER_USD,
    _exchange_from_ticker,
    _normalize_verdict,
    _resolve_fx,
    check_staleness,
)

if TYPE_CHECKING:
    from src.ibkr.reconciler import ReconciliationDiagnostics


def find_opportunities(
    analyses: dict[str, AnalysisRecord],
    held_tickers: set[str],
    portfolio: PortfolioSummary,
    *,
    diagnostics: ReconciliationDiagnostics | None,
    structural_macro_events: list,
    max_age_days: int,
    drift_threshold_pct: float,
    sector_limit_pct: float,
    exchange_limit_pct: float,
    sector_weights: dict[str, float],
    exchange_weights: dict[str, float],
    remaining_cash: float,
) -> tuple[list[ReconciliationItem], float]:
    """Find new BUY recommendations not already held or handled by watchlist."""
    items: list[ReconciliationItem] = []

    for ticker, analysis in analyses.items():
        if ticker in held_tickers:
            continue
        if _normalize_verdict(analysis.verdict or "") != "BUY":
            continue

        is_stale, _stale_reason = check_staleness(
            analysis,
            None,
            max_age_days,
            drift_threshold_pct,
            structural_macro_events=structural_macro_events,
        )
        if is_stale:
            continue

        has_portfolio = portfolio.portfolio_value_usd > 0
        if has_portfolio and remaining_cash <= 0:
            if diagnostics is not None:
                diagnostics.cash_blocked_offwatch_buy_count += 1
            continue

        entry_price = analysis.entry_price or analysis.current_price
        conviction = analysis.conviction or analysis.trade_block.conviction or ""
        size_pct = analysis.trade_block.size_pct or (analysis.position_size or 0)

        fx_rate = _resolve_fx(analysis)
        buy_qty = calculate_quantity(
            available_cash_usd=remaining_cash,
            entry_price_local=entry_price or 0.0,
            fx_rate_to_usd=fx_rate,
            size_pct=size_pct,
            portfolio_value_usd=portfolio.portfolio_value_usd,
            yf_ticker=ticker,
        )
        buy_cost_usd = buy_qty * (entry_price or 0.0) * fx_rate

        if buy_qty > 0 and buy_cost_usd < _MIN_ORDER_USD:
            continue

        remaining_cash -= buy_cost_usd

        buy_reason = f"New BUY ({analysis.analysis_date}) — {conviction} conviction, target {size_pct:.1f}%"
        if portfolio.portfolio_value_usd > 0:
            exch = analysis.exchange or _exchange_from_ticker(ticker)
            sect = analysis.sector or "Unknown"
            projected_weight = buy_cost_usd / portfolio.portfolio_value_usd * 100
            proj_exch = exchange_weights.get(exch, 0.0) + projected_weight
            proj_sect = sector_weights.get(sect, 0.0) + projected_weight
            conc_warns = []
            if proj_exch > exchange_limit_pct:
                conc_warns.append(
                    f"⚠ {exch} → {proj_exch:.0f}% (limit {exchange_limit_pct:.0f}%)"
                )
            if proj_sect > sector_limit_pct:
                conc_warns.append(
                    f"⚠ {sect} sector → {proj_sect:.0f}% (limit {sector_limit_pct:.0f}%)"
                )
            if conc_warns:
                buy_reason += "  " + "; ".join(conc_warns)

        items.append(
            ReconciliationItem(
                ticker=ticker,
                action="BUY",
                reason=buy_reason,
                urgency="MEDIUM",
                analysis=analysis,
                suggested_quantity=buy_qty if buy_qty > 0 else None,
                suggested_price=entry_price,
                suggested_order_type="LMT",
                cash_impact_usd=-buy_cost_usd if buy_cost_usd > 0 else 0.0,
            )
        )

    return items, remaining_cash
