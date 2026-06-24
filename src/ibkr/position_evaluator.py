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
    SCREEN_REVIEW_DNI_ZONES,
    _classify_sell_type,
    _exchange_from_position,
    _normalize_verdict,
    _normalize_zone,
    _settlement_date,
    check_staleness,
    check_stop_breach,
    check_target_hit,
    classify_profit_take,
)
from src.ibkr.ticker import Ticker
from src.ticker_policy import sibling_ticker_candidates

logger = structlog.get_logger(__name__)


_PROFIT_TAKE_REASON_LABELS = {
    "capital_idle_cash_severe": "severe idle-cash risk",
    "capital_idle_cash_risk_plus_target_hit": "idle-cash risk plus target hit",
    "capital_idle_cash_risk_plus_large_gain": "idle-cash risk plus large gain",
    "short_term_tax": "short-term tax status",
    "unknown_tax_term": "holding period unknown",
    "unknown_tax_term_severity_override": "holding period unknown; severity override",
}


def _profit_take_reason(reasons: tuple[str, ...], target_hit: bool) -> str:
    labels = [_PROFIT_TAKE_REASON_LABELS.get(reason, reason) for reason in reasons]
    prefix = (
        "Profit take" if "unknown_tax_term" not in reasons else "Profit take review"
    )
    context = "; ".join(labels) if labels else "capital allocation discipline"
    if target_hit and "capital_idle_cash_risk_plus_target_hit" not in reasons:
        context = f"{context}; target hit"
    return f"{prefix}: {context}"


def _same_base_sibling_keys(
    ticker: str, analyses: dict[str, AnalysisRecord]
) -> tuple[str, ...]:
    """Return whitelisted same-market sibling analysis keys for diagnostics only."""
    candidates = set(sibling_ticker_candidates(ticker))
    if not candidates:
        return ()
    siblings = [analysis_key for analysis_key in analyses if analysis_key in candidates]
    return tuple(sorted(siblings))


def _data_vacuum_review_reason(
    analysis: AnalysisRecord, siblings: tuple[str, ...]
) -> str:
    if siblings:
        return (
            f"Data-vacuum DNI for {analysis.ticker} conflicts with sibling analysis "
            f"{siblings[0]}; verify exchange suffix."
        )
    resolved = analysis.data_quality.get("ticker_rescue_resolved")
    original = analysis.data_quality.get("ticker_rescue_original")
    if original and resolved:
        return (
            f"Data-vacuum DNI for {analysis.ticker}; ticker rescue attempted "
            f"{original} -> {resolved}; verify exchange suffix."
        )
    return f"Data-vacuum DNI for {analysis.ticker}; verify data coverage."


_CURRENCY_EQUIVALENTS = {"GBX": "GBP", "GBP": "GBX"}  # same economy, unit scaled


def _base_match_allowed(pos: NormalizedPosition, analysis: AnalysisRecord) -> bool:
    """Whether a base-symbol fallback match is safe for this position.

    Suffix-less positions (IBKR couldn't resolve the exchange) may borrow the
    single base-matched analysis. A SUFFIXED position may only do so when the
    analysis currency agrees — otherwise it is a different listing entirely.
    """
    if not pos.ticker.has_suffix:
        return True
    a_ccy = (getattr(analysis, "currency", "") or "").upper()
    p_ccy = (pos.currency or "").upper()
    if not a_ccy or not p_ccy:
        # Legacy records without a recorded currency: allow — ambiguous-base
        # poisoning in the lookup builder already guards multi-listing bases.
        return True
    return a_ccy == p_ccy or _CURRENCY_EQUIVALENTS.get(a_ccy) == p_ccy


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
            candidate = alpha_base_lookup.get(pos.ticker.ibkr.upper())
            if candidate is not None and not _base_match_allowed(pos, candidate):
                # A suffixed position must never borrow a different exchange's
                # analysis (the AGS.BR-shows-AGS.SI bug) — currency must agree.
                logger.warning(
                    "base_symbol_match_blocked",
                    position_ticker=pos.ticker.yf,
                    candidate_analysis=candidate.ticker,
                    position_currency=pos.currency,
                    analysis_currency=getattr(candidate, "currency", None),
                )
                candidate = None
            if candidate is not None:
                analysis = candidate
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
            zone = _normalize_zone(analysis.zone)
            if verdict_upper == "DO_NOT_INITIATE" and zone in SCREEN_REVIEW_DNI_ZONES:
                items.append(
                    ReconciliationItem(
                        ticker=item_ticker,
                        action="REVIEW",
                        reason=(
                            f"Screen-threshold DNI ({zone} zone) - "
                            f"review held position ({analysis.analysis_date})"
                        ),
                        urgency="MEDIUM",
                        ibkr_position=pos,
                        analysis=analysis,
                        sell_type="SCREEN_REJECT",
                    )
                )
                continue

            if (
                verdict_upper == "DO_NOT_INITIATE"
                and analysis.data_quality.get("data_vacuum") is True
                # Belt-and-suspenders: only block executable SELLs when the
                # saved analysis also lacks its own price anchor.
                and analysis.current_price is None
            ):
                siblings = _same_base_sibling_keys(analysis.ticker, analyses)
                items.append(
                    ReconciliationItem(
                        ticker=item_ticker,
                        action="REVIEW",
                        reason=_data_vacuum_review_reason(analysis, siblings),
                        urgency="HIGH",
                        ibkr_position=pos,
                        analysis=analysis,
                        sell_type="DATA_QUALITY_REVIEW",
                    )
                )
                continue

            if (
                verdict_upper == "DO_NOT_INITIATE"
                and isinstance(analysis.health_adj, int | float)
                and isinstance(analysis.growth_adj, int | float)
                and analysis.health_adj == 0
                and analysis.growth_adj == 0
                and analysis.current_price is None
            ):
                siblings = _same_base_sibling_keys(analysis.ticker, analyses)
                items.append(
                    ReconciliationItem(
                        ticker=item_ticker,
                        action="REVIEW",
                        reason=_data_vacuum_review_reason(analysis, siblings),
                        urgency="HIGH",
                        ibkr_position=pos,
                        analysis=analysis,
                        sell_type="DATA_QUALITY_REVIEW",
                    )
                )
                continue

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

        is_stale, stale_reason = check_staleness(
            analysis,
            current_price,
            max_age_days,
            drift_threshold_pct,
            structural_macro_events=structural_macro_events,
        )

        target_hit = check_target_hit(analysis, current_price)
        # A profit-take is the disciplined EXIT when a winner reaches its target or
        # posts a large gain — both of which necessarily push price >drift_threshold
        # above entry. check_staleness flags large drift in *either* direction, so the
        # very upward move that earns the profit-take would otherwise null it (the
        # capital-allocation SELL was dead in production for any target >threshold above
        # entry). Gate the profit-take on age/macro staleness only — an exit reacts to
        # favorable drift, it is not invalidated by it.
        non_drift_stale, _ = check_staleness(
            analysis,
            current_price,
            max_age_days,
            float("inf"),
            structural_macro_events=structural_macro_events,
        )
        profit_take = (
            classify_profit_take(
                analysis=analysis,
                position=pos,
                target_hit=target_hit,
            )
            if not non_drift_stale
            else None
        )
        if profit_take and profit_take.qualifies:
            executable = profit_take.action == "SELL"
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action=profit_take.action or "REVIEW",
                    reason=_profit_take_reason(profit_take.reasons, target_hit),
                    urgency="LOW" if executable else "MEDIUM",
                    ibkr_position=pos,
                    analysis=analysis,
                    suggested_quantity=abs(int(pos.quantity)) if executable else None,
                    suggested_price=current_price if executable else None,
                    suggested_order_type="LMT",
                    cash_impact_usd=pos.market_value_usd if executable else 0.0,
                    settlement_date=_settlement_date(2) if executable else None,
                    sell_type="PROFIT_TAKE",
                    cost_basis_return_pct=profit_take.cost_basis_return_pct,
                    profit_take_reasons=profit_take.reasons,
                )
            )
            continue

        if target_hit:
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
