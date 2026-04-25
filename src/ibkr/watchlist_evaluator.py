"""Phase 1.5 watchlist evaluation for reconciliation."""

from __future__ import annotations

import re

from src.ibkr.models import AnalysisRecord, PortfolioSummary, ReconciliationItem
from src.ibkr.order_builder import calculate_quantity
from src.ibkr.reconciliation_rules import (
    _MIN_ORDER_USD,
    _REJECT_VERDICTS,
    _exchange_from_ticker,
    _normalize_verdict,
    _resolve_fx,
    check_staleness,
)


def evaluate_watchlist(
    watchlist_tickers: set[str] | None,
    held_tickers: set[str],
    analyses: dict[str, AnalysisRecord],
    portfolio: PortfolioSummary,
    *,
    alpha_base_lookup: dict[str, AnalysisRecord],
    alpha_base_to_key: dict[str, str],
    structural_macro_events: list,
    max_age_days: int,
    drift_threshold_pct: float,
    sector_limit_pct: float,
    exchange_limit_pct: float,
    sector_weights: dict[str, float],
    exchange_weights: dict[str, float],
    remaining_cash: float,
) -> tuple[list[ReconciliationItem], set[str], float]:
    """Evaluate watchlist tickers not currently held."""
    items: list[ReconciliationItem] = []
    watchlist_set = (watchlist_tickers or set()) - held_tickers
    watchlist_resolved_keys: set[str] = set()

    for ticker in sorted(watchlist_set):
        analysis = analyses.get(ticker)
        watchlist_base = (ticker.rsplit(".", 1)[0] if "." in ticker else ticker).upper()
        if re.match(r"^[A-Z][A-Z0-9]*$", watchlist_base):
            resolved_key = alpha_base_to_key.get(watchlist_base)
            if resolved_key and resolved_key != ticker:
                watchlist_resolved_keys.add(resolved_key)
                resolved_analysis = analyses.get(resolved_key)
                if resolved_analysis:
                    analysis = resolved_analysis
                    ticker = resolved_key
        else:
            for analysis_key in analyses:
                if (
                    "." in analysis_key
                    and analysis_key.split(".")[0].upper() == watchlist_base
                ):
                    watchlist_resolved_keys.add(analysis_key)
                    if analysis is None:
                        resolved_analysis = analyses.get(analysis_key)
                        if resolved_analysis:
                            analysis = resolved_analysis
                            ticker = analysis_key

        if analysis is None:
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="REVIEW",
                    reason="Watchlist: no analysis found",
                    urgency="MEDIUM",
                    is_watchlist=True,
                )
            )
            continue

        is_stale, stale_reason = check_staleness(
            analysis,
            None,
            max_age_days,
            drift_threshold_pct,
            structural_macro_events=structural_macro_events,
        )
        if is_stale:
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="REVIEW",
                    reason=f"Watchlist: stale analysis ({stale_reason})",
                    urgency="MEDIUM",
                    analysis=analysis,
                    is_watchlist=True,
                )
            )
            continue

        verdict_upper = _normalize_verdict(analysis.verdict or "")
        if verdict_upper in _REJECT_VERDICTS:
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="REMOVE",
                    reason=f"Remove from watchlist — verdict → {analysis.verdict}  ({analysis.analysis_date})",
                    urgency="MEDIUM",
                    analysis=analysis,
                    is_watchlist=True,
                )
            )
        elif verdict_upper == "BUY":
            has_portfolio = portfolio.portfolio_value_usd > 0
            if has_portfolio and remaining_cash <= 0:
                items.append(
                    ReconciliationItem(
                        ticker=ticker,
                        action="BUY",
                        reason=f"Watchlist BUY ({analysis.analysis_date}) — no cash available",
                        urgency="MEDIUM",
                        analysis=analysis,
                        is_watchlist=True,
                    )
                )
            else:
                entry_price = analysis.entry_price or analysis.current_price
                conviction = (
                    analysis.conviction or analysis.trade_block.conviction or ""
                )
                size_pct = analysis.trade_block.size_pct or (
                    analysis.position_size or 0
                )
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
                    pass
                else:
                    remaining_cash -= buy_cost_usd
                    buy_reason = f"Watchlist BUY ({analysis.analysis_date}) — {conviction} conviction, target {size_pct:.1f}%"
                    if portfolio.portfolio_value_usd > 0:
                        exch = analysis.exchange or _exchange_from_ticker(ticker)
                        sect = analysis.sector or "Unknown"
                        projected_weight = (
                            buy_cost_usd / portfolio.portfolio_value_usd * 100
                        )
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
                            is_watchlist=True,
                        )
                    )
        else:
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="HOLD",
                    reason=f"Watchlist: monitoring — verdict {analysis.verdict}  ({analysis.analysis_date})",
                    urgency="LOW",
                    analysis=analysis,
                    is_watchlist=True,
                )
            )

    held_tickers.update(watchlist_set)
    held_tickers.update(watchlist_resolved_keys)
    return items, held_tickers, remaining_cash
