"""Phase 1 held-position evaluation for reconciliation."""

from __future__ import annotations

import structlog

from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
)
from src.ibkr.order_builder import round_to_lot_size
from src.ibkr.reconciliation_rules import (
    _MIN_ORDER_USD,
    _REJECT_VERDICTS,
    _classify_sell_type,
    _exchange_from_position,
    _normalize_verdict,
    _settlement_date,
    check_staleness,
    check_stop_breach,
    check_target_hit,
)
from src.ibkr.ticker import Ticker

logger = structlog.get_logger(__name__)


def evaluate_positions(
    positions: list[NormalizedPosition],
    analyses: dict[str, AnalysisRecord],
    portfolio: PortfolioSummary,
    *,
    alpha_base_lookup: dict[str, AnalysisRecord],
    structural_macro_events: list,
    max_age_days: int,
    drift_threshold_pct: float,
    overweight_threshold_pct: float,
    underweight_threshold_pct: float,
    sector_limit_pct: float,
    exchange_limit_pct: float,
    sector_weights: dict[str, float],
    exchange_weights: dict[str, float],
    remaining_cash: float,
) -> tuple[list[ReconciliationItem], set[str], float]:
    """Evaluate currently held positions and return actions plus updated cash/held set."""
    items: list[ReconciliationItem] = []
    held_tickers: set[str] = set()

    for pos in positions:
        if pos.quantity <= 0:
            continue

        yf_key = pos.ticker.yf
        analysis: AnalysisRecord | None = None

        if (
            not pos.ticker.has_suffix
            and pos.ticker.ibkr
            and not pos.ticker.ibkr.isdigit()
        ):
            best = alpha_base_lookup.get(pos.ticker.ibkr.upper())
            if best:
                yf_key = best.ticker
                analysis = best
                logger.debug(
                    "analysis_found_via_alpha_base",
                    pos_yf=pos.ticker.yf,
                    ibkr_symbol=pos.ticker.ibkr,
                    found_as=best.ticker,
                )

        if analysis is None:
            analysis = analyses.get(yf_key)

        if analysis is None and pos.ticker.ibkr and not pos.ticker.ibkr.isdigit():
            analysis = alpha_base_lookup.get(pos.ticker.ibkr.upper())
            if analysis:
                logger.debug(
                    "analysis_found_via_base_symbol",
                    yf_ticker=pos.ticker.yf,
                    ibkr_symbol=pos.ticker.ibkr,
                    found_as=analysis.ticker,
                )

        item_ticker = Ticker.from_yf(yf_key) if yf_key != pos.ticker.yf else pos.ticker
        ticker = yf_key
        held_tickers.add(ticker)

        if "." not in ticker:
            held_base = ticker.upper()
            for analysis_key in analyses:
                if (
                    "." in analysis_key
                    and analysis_key.split(".")[0].upper() == held_base
                ):
                    held_tickers.add(analysis_key)

        if analysis is None:
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="REVIEW",
                    reason="Position held but no evaluator analysis found",
                    urgency="MEDIUM",
                    ibkr_position=pos,
                )
            )
            continue

        current_price = pos.current_price_local

        if check_stop_breach(analysis, current_price):
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="SELL",
                    reason=f"Stop breached: price {current_price:.2f} < stop {analysis.stop_price:.2f}",
                    urgency="HIGH",
                    ibkr_position=pos,
                    analysis=analysis,
                    suggested_quantity=abs(int(pos.quantity)),
                    suggested_price=current_price,
                    suggested_order_type="LMT",
                    cash_impact_usd=pos.market_value_usd,
                    settlement_date=_settlement_date(2),
                    sell_type="STOP_BREACH",
                )
            )
            continue

        verdict_upper = _normalize_verdict(analysis.verdict or "")
        if verdict_upper in _REJECT_VERDICTS:
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="SELL",
                    reason=f"Verdict → {analysis.verdict}  ({analysis.analysis_date})",
                    urgency="HIGH",
                    ibkr_position=pos,
                    analysis=analysis,
                    suggested_quantity=abs(int(pos.quantity)),
                    suggested_order_type="LMT",
                    suggested_price=current_price,
                    cash_impact_usd=pos.market_value_usd,
                    settlement_date=_settlement_date(2),
                    sell_type=_classify_sell_type(analysis, stop_breached=False),
                )
            )
            continue

        if check_target_hit(analysis, current_price):
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="REVIEW",
                    reason=f"Target hit: price {current_price:.2f} >= target {analysis.target_1_price:.2f}",
                    urgency="LOW",
                    ibkr_position=pos,
                    analysis=analysis,
                )
            )
            continue

        is_stale, stale_reason = check_staleness(
            analysis,
            current_price,
            max_age_days,
            drift_threshold_pct,
            structural_macro_events=structural_macro_events,
        )
        if is_stale:
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="REVIEW",
                    reason=f"Stale analysis: {stale_reason}",
                    urgency="MEDIUM",
                    ibkr_position=pos,
                    analysis=analysis,
                )
            )
            continue

        target_size_pct = analysis.trade_block.size_pct or (analysis.position_size or 0)
        if target_size_pct > 0 and portfolio.portfolio_value_usd > 0:
            actual_pct = (pos.market_value_usd / portfolio.portfolio_value_usd) * 100
            excess_pct = actual_pct - target_size_pct
            if excess_pct > overweight_threshold_pct:
                target_value_usd = portfolio.portfolio_value_usd * (
                    target_size_pct / 100
                )
                trim_value_usd = pos.market_value_usd - target_value_usd
                price_usd_per_share = (
                    pos.market_value_usd / abs(pos.quantity)
                    if pos.quantity != 0
                    else 1.0
                )
                trim_qty = round_to_lot_size(
                    int(trim_value_usd / (price_usd_per_share or 1.0)), ticker
                )
                items.append(
                    ReconciliationItem(
                        ticker=item_ticker,
                        action="TRIM",
                        reason=f"Overweight: {actual_pct:.1f}% vs target {target_size_pct:.1f}% (+{excess_pct:.1f}%)",
                        urgency="MEDIUM",
                        ibkr_position=pos,
                        analysis=analysis,
                        suggested_quantity=trim_qty,
                        suggested_price=pos.current_price_local,
                        suggested_order_type="LMT",
                        cash_impact_usd=trim_value_usd,
                        settlement_date=_settlement_date(2),
                    )
                )
                continue

            shortfall_pct = target_size_pct - actual_pct
            verdict_upper = _normalize_verdict(analysis.verdict or "")
            if (
                shortfall_pct > underweight_threshold_pct
                and verdict_upper == "BUY"
                and remaining_cash > 0
            ):
                target_value_usd = portfolio.portfolio_value_usd * (
                    target_size_pct / 100
                )
                add_value_usd = min(
                    target_value_usd - pos.market_value_usd, remaining_cash
                )
                price_usd_per_share = (
                    pos.market_value_usd / abs(pos.quantity)
                    if pos.quantity != 0
                    else 1.0
                )
                add_qty = round_to_lot_size(
                    int(add_value_usd / (price_usd_per_share or 1.0)), ticker
                )
                actual_add_cost = add_qty * price_usd_per_share
                if add_qty == 0 or (add_qty > 0 and actual_add_cost < _MIN_ORDER_USD):
                    pass
                else:
                    remaining_cash -= add_value_usd
                    add_reason = f"Underweight: {actual_pct:.1f}% vs target {target_size_pct:.1f}% (-{shortfall_pct:.1f}%)"
                    exch = analysis.exchange or _exchange_from_position(pos)
                    sect = analysis.sector or "Unknown"
                    projected_weight = (
                        add_value_usd / portfolio.portfolio_value_usd * 100
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
                        add_reason += "  " + "; ".join(conc_warns)
                    items.append(
                        ReconciliationItem(
                            ticker=item_ticker,
                            action="ADD",
                            reason=add_reason,
                            urgency="LOW",
                            ibkr_position=pos,
                            analysis=analysis,
                            suggested_quantity=add_qty,
                            suggested_price=pos.current_price_local
                            or analysis.entry_price,
                            suggested_order_type="LMT",
                            cash_impact_usd=-add_value_usd,
                        )
                    )
                    continue

        status_parts = []
        if analysis.entry_price and current_price:
            gain_pct = (
                (current_price - analysis.entry_price) / analysis.entry_price
            ) * 100
            status_parts.append(
                f"entry {analysis.entry_price:.2f} → {current_price:.2f} ({gain_pct:+.1f}%)"
            )
        if analysis.stop_price:
            status_parts.append(f"stop {analysis.stop_price:.2f}")
        if analysis.target_1_price:
            status_parts.append(f"target {analysis.target_1_price:.2f}")

        items.append(
            ReconciliationItem(
                ticker=item_ticker,
                action="HOLD",
                reason=f"Within targets — {'; '.join(status_parts)}"
                if status_parts
                else "Position OK",
                urgency="LOW",
                ibkr_position=pos,
                analysis=analysis,
            )
        )

    return items, held_tickers, remaining_cash
