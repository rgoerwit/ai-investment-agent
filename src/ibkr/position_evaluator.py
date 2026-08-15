"""Phase 1 held-position evaluation for reconciliation."""

from __future__ import annotations

import structlog

from src.ibkr.buy_stability import PriorVerdict, load_recent_same_ticker_history
from src.ibkr.concentration import (
    canonical_exchange_bucket,
    canonical_sector_bucket,
    format_concentration_warnings,
    project_concentration_breaches,
)
from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
)
from src.ibkr.order_builder import round_to_lot_size
from src.ibkr.portfolio_defaults import (
    DEFAULT_MIN_ACTIONABLE_POSITION_USD,
    DEFAULT_SELL_CONFIRMATION_LOOKBACK_DAYS,
)
from src.ibkr.reconciliation_rules import (
    _MIN_ORDER_USD,
    _REJECT_VERDICTS,
    SCREEN_REVIEW_DNI_ZONES,
    _classify_sell_type,
    _normalize_verdict,
    _normalize_zone,
    _settlement_date,
    analysis_identity_verified,
    base_match_allowed,
    check_base_case_reference_reached,
    check_review_level_breach,
    check_staleness,
    classify_disposition,
    classify_profit_take,
    review_level_note,
)
from src.ibkr.ticker import Ticker
from src.ticker_policy import sibling_ticker_candidates

logger = structlog.get_logger(__name__)


def _load_prior_history(analysis: AnalysisRecord) -> list[PriorVerdict]:
    """Load dated same-ticker verdict history for the SELL confirmation gate."""
    from src.config import config

    return load_recent_same_ticker_history(
        analysis.ticker,
        lookback_days=DEFAULT_SELL_CONFIRMATION_LOOKBACK_DAYS,
        results_dir=str(config.results_dir),
        exclude_path=analysis.file_path or None,
    )


def _is_de_minimis(
    pos: NormalizedPosition,
    analysis: AnalysisRecord,
    min_actionable_position_usd: float,
) -> bool:
    """Position too small to be worth attention — unless compliance-flagged.

    Compliance-class flags (PFIC/VIE/CMIC) carry per-position burdens
    independent of dollar size (a $193 PFIC still costs a Form 8621), so they
    exempt a position from de-minimis suppression entirely.
    """
    if pos.market_value_usd >= min_actionable_position_usd:
        return False
    return not analysis.evidence.compliance_flag_types


def _de_minimis_hold(
    item_ticker: Ticker,
    pos: NormalizedPosition,
    analysis: AnalysisRecord,
    suppressed_reason: str,
) -> ReconciliationItem:
    return ReconciliationItem(
        ticker=item_ticker,
        action="HOLD",
        reason=(
            f"De-minimis (${pos.market_value_usd:,.0f}) — monitor only. "
            f"Suppressed: {suppressed_reason}"
        ),
        urgency="LOW",
        ibkr_position=pos,
        analysis=analysis,
        action_basis="DE_MINIMIS",
    )


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
    context = "; ".join(labels) if labels else "capital allocation discipline"
    if target_hit and "capital_idle_cash_risk_plus_target_hit" not in reasons:
        context = f"{context}; target hit"
    return (
        f"Profit-take review: {context} — advisory; verify lot gains/holding "
        "period in IBKR before selling"
    )


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


# base_match_allowed lives in reconciliation_rules (shared with reconciler
# weight attribution); suffix-less positions are currency-guarded there too.


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
    min_actionable_position_usd: float = DEFAULT_MIN_ACTIONABLE_POSITION_USD,
) -> tuple[list[ReconciliationItem], set[str], float]:
    """Evaluate currently held positions and return actions plus updated cash/held set."""
    items: list[ReconciliationItem] = []
    held_tickers: set[str] = set()

    for pos in positions:
        if pos.quantity <= 0 and pos.valuation_valid:
            continue

        yf_key = pos.ticker.yf
        analysis: AnalysisRecord | None = None

        if (
            not pos.ticker.has_suffix
            and pos.ticker.ibkr
            and not pos.ticker.ibkr.isdigit()
        ):
            best = alpha_base_lookup.get(pos.ticker.ibkr.upper())
            if best is not None and not base_match_allowed(pos, best):
                # A suffix-less position (exchange unresolved) must still agree
                # on currency — an EUR Brussels AGS reported as SMART must not
                # adopt an SGD AGS.SI analysis.
                logger.warning(
                    "base_symbol_match_blocked",
                    position_ticker=pos.ticker.yf,
                    candidate_analysis=best.ticker,
                    position_currency=pos.currency,
                    analysis_currency=getattr(best, "currency", None),
                )
                best = None
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
            if candidate is not None and not base_match_allowed(pos, candidate):
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

        if not pos.valuation_valid:
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="REVIEW",
                    reason=(
                        "Position valuation unavailable — "
                        f"{pos.valuation_issue or 'broker value units could not be verified'}"
                    ),
                    urgency="HIGH",
                    ibkr_position=pos,
                    analysis=analysis,
                    sell_type="DATA_QUALITY_REVIEW",
                    action_basis="DATA_QUALITY",
                )
            )
            continue

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

        # A price break below the analysis-time review level raises urgency but
        # carries NO sell authority (July 2026 retail alignment): a sale needs
        # fundamental evidence — confirmed thesis failure, mandatory flags,
        # tender mechanics — never a price level alone. The breach is threaded
        # into the verdict flow below instead of short-circuiting it.
        stop_breached = check_review_level_breach(analysis, current_price, pos.currency)

        # Identity is verified only when the analysis was found under the
        # position's own exchange-resolved yfinance key. Base-symbol borrows
        # and currency-guessed suffixes are inferred mappings — they may
        # inform reviews but must never authorize an executable exit.
        identity_verified = analysis_identity_verified(pos, analysis)

        verdict_upper = _normalize_verdict(analysis.verdict or "")
        if verdict_upper in _REJECT_VERDICTS:
            zone = _normalize_zone(analysis.zone)
            if verdict_upper == "DO_NOT_INITIATE" and zone in SCREEN_REVIEW_DNI_ZONES:
                if _is_de_minimis(pos, analysis, min_actionable_position_usd):
                    items.append(
                        _de_minimis_hold(
                            item_ticker, pos, analysis, f"screen-threshold DNI ({zone})"
                        )
                    )
                    continue
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
                        action_basis="ENTRY_CONSTRAINT",
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
                        action_basis="DATA_QUALITY",
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
                        action_basis="DATA_QUALITY",
                    )
                )
                continue

            disposition = classify_disposition(
                analysis,
                current_price_local=current_price,
                prior_history=_load_prior_history(analysis),
            )
            sell_type = _classify_sell_type(analysis, stop_breached=False)
            if disposition.basis == "DATA_QUALITY":
                # Score-derived sell_type is meaningless when the scores
                # themselves are unreliable; the DATA_QUALITY_REVIEW token
                # also routes the item into the refresh blocking_now bucket
                # (a SOFT_REJECT stamp would leave it on staleness cadence).
                sell_type = "DATA_QUALITY_REVIEW"
            detail = disposition.detail
            if stop_breached:
                detail += (
                    f"; price {current_price:.2f} broke the analysis review "
                    f"level {analysis.stop_price:.2f}"
                )
            reject_reason = (
                f"Verdict → {analysis.verdict}  ({analysis.analysis_date}) — {detail}"
            )
            if disposition.action == "SELL":
                if not identity_verified:
                    # An exit order on an inferred symbol/exchange mapping can
                    # sell the wrong listing — downgrade to review.
                    items.append(
                        ReconciliationItem(
                            ticker=item_ticker,
                            action="REVIEW",
                            reason=(
                                f"{reject_reason}; security identity unverified "
                                "(exchange/currency mapping) — confirm the "
                                "listing in IBKR before any exit"
                            ),
                            urgency="HIGH",
                            ibkr_position=pos,
                            analysis=analysis,
                            sell_type=sell_type,
                            action_basis=disposition.basis,
                        )
                    )
                    continue
                items.append(
                    ReconciliationItem(
                        ticker=item_ticker,
                        action="SELL",
                        reason=reject_reason,
                        urgency="HIGH",
                        ibkr_position=pos,
                        analysis=analysis,
                        suggested_quantity=abs(int(pos.quantity)),
                        suggested_order_type="LMT",
                        suggested_price=current_price,
                        cash_impact_usd=pos.market_value_usd,
                        settlement_date=_settlement_date(2),
                        sell_type=sell_type,
                        action_basis=disposition.basis,
                    )
                )
                continue

            if _is_de_minimis(pos, analysis, min_actionable_position_usd):
                items.append(
                    _de_minimis_hold(
                        item_ticker,
                        pos,
                        analysis,
                        f"verdict {analysis.verdict} ({disposition.basis})",
                    )
                )
                continue

            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="REVIEW",
                    reason=reject_reason,
                    urgency="HIGH" if stop_breached else "MEDIUM",
                    ibkr_position=pos,
                    analysis=analysis,
                    sell_type=sell_type,
                    action_basis=disposition.basis,
                )
            )
            continue

        if stop_breached:
            if _is_de_minimis(pos, analysis, min_actionable_position_usd):
                items.append(
                    _de_minimis_hold(item_ticker, pos, analysis, "price-drop review")
                )
                continue
            health = analysis.health_adj
            growth = analysis.growth_adj
            if health is not None and growth is not None:
                scores_txt = f"last scored H:{health:.0f}% G:{growth:.0f}%"
            else:
                scores_txt = "scores unavailable"
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="REVIEW",
                    reason=(
                        f"Price {current_price:.2f} broke the analysis review "
                        f"level {analysis.stop_price:.2f} — fundamentals "
                        f"{scores_txt} ({analysis.analysis_date}); refresh the "
                        "analysis. A sale needs fundamental failure evidence, "
                        "not a price level"
                    ),
                    urgency="HIGH",
                    ibkr_position=pos,
                    analysis=analysis,
                    sell_type="STOP_BREACH",
                    action_basis="STOP_LOSS",
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

        target_hit = check_base_case_reference_reached(
            analysis, current_price, pos.currency
        )
        # A capital-allocation review can be useful when a winner reaches its
        # valuation reference or posts a large gain. Upward price drift should
        # not suppress that review, but it never grants sale authority.
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
                    action_basis="CAPITAL_ALLOCATION",
                    cost_basis_return_pct=profit_take.cost_basis_return_pct,
                    profit_take_reasons=profit_take.reasons,
                )
            )
            continue

        if target_hit:
            if _is_de_minimis(pos, analysis, min_actionable_position_usd):
                items.append(_de_minimis_hold(item_ticker, pos, analysis, "target hit"))
                continue
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="REVIEW",
                    reason=(
                        "Base-case valuation reference reached: price "
                        f"{current_price:.2f} >= reference "
                        f"{analysis.target_1_price:.2f}; reassess forward return "
                        "and tax lots before changing the position"
                    ),
                    urgency="LOW",
                    ibkr_position=pos,
                    analysis=analysis,
                    action_basis="CAPITAL_ALLOCATION",
                )
            )
            continue

        if is_stale:
            if _is_de_minimis(pos, analysis, min_actionable_position_usd):
                items.append(
                    _de_minimis_hold(
                        item_ticker, pos, analysis, f"stale analysis ({stale_reason})"
                    )
                )
                continue
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
                # Advisory only (July 2026 retail alignment): a drift trim is a
                # sale — a capital-gains event on an intact position — so the
                # sizing math is surfaced as information, never as an order.
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
                        action="REVIEW",
                        reason=(
                            f"Overweight: {actual_pct:.1f}% vs target "
                            f"{target_size_pct:.1f}% (+{excess_pct:.1f}%) — "
                            f"advisory: trimming ≈{trim_qty} shares "
                            f"(~${trim_value_usd:,.0f}) would restore target; "
                            "verify lot gains/holding period in IBKR first"
                        ),
                        urgency="LOW",
                        ibkr_position=pos,
                        analysis=analysis,
                        action_basis="OVERWEIGHT",
                    )
                )
                continue

            shortfall_pct = target_size_pct - actual_pct
            verdict_upper = _normalize_verdict(analysis.verdict or "")
            if (
                shortfall_pct > underweight_threshold_pct
                and verdict_upper == "BUY"
                # A quick-mode BUY is a screening candidate — it must not drive an ADD
                # to an existing position; the position holds until a full re-run.
                and not getattr(analysis, "is_quick_mode", False)
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
                    projected_weight = (
                        add_value_usd / portfolio.portfolio_value_usd * 100
                    )
                    breaches = project_concentration_breaches(
                        exchange_key=canonical_exchange_bucket(
                            item_ticker,
                            analysis_exchange=analysis.exchange,
                            position=pos,
                        ),
                        sector_key=canonical_sector_bucket(analysis.sector),
                        candidate_pct=projected_weight,
                        exchange_weights=exchange_weights,
                        sector_weights=sector_weights,
                        exchange_limit_pct=exchange_limit_pct,
                        sector_limit_pct=sector_limit_pct,
                    )
                    conc_warns = format_concentration_warnings(breaches)
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
            status_parts.append(f"review-below {analysis.stop_price:.2f}")
            stop_note = review_level_note(analysis, current_price)
            if stop_note:
                status_parts.append(stop_note)
        if analysis.target_1_price:
            status_parts.append(f"base-case reference {analysis.target_1_price:.2f}")

        items.append(
            ReconciliationItem(
                ticker=item_ticker,
                action="HOLD",
                reason=f"Monitoring context — {'; '.join(status_parts)}"
                if status_parts
                else "Position OK",
                urgency="LOW",
                ibkr_position=pos,
                analysis=analysis,
            )
        )

    return items, held_tickers, remaining_cash
