"""Shared builders for portfolio report tests."""

from __future__ import annotations

from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    ReconciliationItem,
    TradeBlockData,
)
from src.ibkr.ticker import Ticker
from tests.ibkr.reconciler_cases import _make_analysis, _make_position

CORRELATED_SELL_EVENT_FLAG = (
    "CORRELATED_SELL_EVENT: 8 positions changed verdict within 7d of 2026-03-05"
    " (80% of held positions) — probable macro event."
    " Do not sell into a correlated drop: exit only on confirmed fundamental"
    " failure; review everything else."
)


def _panic_items() -> list[ReconciliationItem]:
    """
    Pre-demoted panic-day items for format_report isolation tests.

    8 SOFT_REJECT items already demoted to REVIEW (as compute_portfolio_health would do),
    plus one legacy STOP_BREACH that presentation must downgrade and one
    confirmed thesis-failure SELL.
    """
    pos = _make_position(current_price=2100)
    items: list[ReconciliationItem] = []

    for i in range(8):
        items.append(
            ReconciliationItem(
                ticker=f"SOFT{i:02d}.T",
                action="REVIEW",
                urgency="MEDIUM",
                reason=(
                    "Verdict → DO_NOT_INITIATE  (2026-03-05)"
                    "  [MACRO_WATCH: demoted from SELL — correlated event detected]"
                ),
                ibkr_position=pos,
                sell_type="SOFT_REJECT",
            )
        )

    items.append(
        ReconciliationItem(
            ticker="STOP.T",
            action="SELL",
            urgency="HIGH",
            reason="Stop breached: price 1700.00 < stop 1900.00",
            ibkr_position=pos,
            sell_type="STOP_BREACH",
        )
    )
    hard_analysis = _make_analysis(ticker="HARD.T", verdict="DO_NOT_INITIATE")
    # Pin the date absolutely: this analysis renders in date-frozen golden
    # snapshots, and _make_analysis derives analysis_date from real now() —
    # a relative date here makes the goldens drift at midnight.
    hard_analysis.analysis_date = "2026-07-12"
    items.append(
        ReconciliationItem(
            ticker="HARD.T",
            action="SELL",
            urgency="HIGH",
            reason="Verdict → DO_NOT_INITIATE  (2026-03-05)",
            ibkr_position=_make_position(ticker="HARD.T", current_price=2100),
            analysis=hard_analysis,
            sell_type="HARD_REJECT",
            action_basis="CONFIRMED_THESIS_FAILURE",
            suggested_quantity=100,
            suggested_price=2100.0,
            cash_impact_usd=1400.0,
            settlement_date="2026-03-09",
        )
    )
    return items


def _make_dip_item(
    ticker: str,
    health: float,
    growth: float,
    entry: float,
    current_price: float,
    stop: float,
    target: float,
    currency: str = "JPY",
    verdict: str = "BUY",
    zone: str = "MODERATE",
    action: str = "REVIEW",
    sell_type: str | None = "SOFT_REJECT",
    age_days: int = 3,
) -> ReconciliationItem:
    """Create a demoted SOFT_REJECT REVIEW item with a full analysis record."""
    from datetime import datetime, timedelta

    analysis_date = (datetime.now() - timedelta(days=age_days)).strftime("%Y-%m-%d")
    analysis = AnalysisRecord(
        ticker=ticker,
        analysis_date=analysis_date,
        verdict=verdict,
        health_adj=health,
        growth_adj=growth,
        zone=zone,
        entry_price=entry,
        stop_price=stop,
        target_1_price=target,
        currency=currency,
        trade_block=TradeBlockData(
            action=verdict,
            entry_price=entry,
            stop_price=stop,
            target_1_price=target,
        ),
    )
    pos = _make_position(
        ticker=ticker,
        current_price=current_price,
        currency=currency,
    )
    return ReconciliationItem(
        ticker=ticker,
        action=action,
        urgency="MEDIUM",
        reason=(
            f"Verdict → {verdict}  ({analysis_date})"
            "  [MACRO_WATCH: demoted from SELL — correlated event detected]"
        ),
        ibkr_position=pos,
        analysis=analysis,
        sell_type=sell_type,
    )


def _make_sell_item(
    ticker: str = "9201.T",
    action: str = "SELL",
    sell_type: str = "STOP_BREACH",
    reason: str = "Stop breached: price 2700.00 < stop 2780.00",
    quantity: float = 100,
    avg_cost_local: float = 2780.0,
    current_price_local: float = 2700.0,
    unrealized_pnl_usd: float = -89.0,
    suggested_quantity: int | None = None,
) -> ReconciliationItem:
    """Build a SELL ReconciliationItem with a fully-populated NormalizedPosition."""
    pos = NormalizedPosition(
        conid=99999,
        ticker=Ticker.from_yf(ticker),
        quantity=quantity,
        avg_cost_local=avg_cost_local,
        current_price_local=current_price_local,
        unrealized_pnl_usd=unrealized_pnl_usd,
        market_value_usd=abs(current_price_local * quantity / 100),
        currency="JPY",
        ticker_identity_verified=True,
        ticker_resolution_source="exchange_map",
    )
    return ReconciliationItem(
        ticker=ticker,
        action=action,
        urgency="HIGH",
        reason=reason,
        ibkr_position=pos,
        sell_type=sell_type,
        suggested_quantity=suggested_quantity,
    )


def _make_sell_item_with_analysis(
    sell_type: str = "STOP_BREACH",
    reason: str = "Stop breached: price 2700.00 < stop 2780.00",
    health: float = 75.0,
    growth: float = 68.0,
    zone: str = "MODERATE",
    verdict: str = "BUY",
    conviction: str = "High",
    analysis_date: str = "2026-01-15",
) -> ReconciliationItem:
    """SELL item with a fully-populated AnalysisRecord for score-line testing."""
    item = _make_sell_item(sell_type=sell_type, reason=reason)
    item = item.model_copy(
        update={
            "analysis": AnalysisRecord(
                ticker="9201.T",
                analysis_date=analysis_date,
                verdict=verdict,
                health_adj=health,
                growth_adj=growth,
                zone=zone,
                conviction=conviction,
                currency="JPY",
            )
        }
    )
    return item


def _make_order(
    conid: int | None = None,
    ticker: str | None = None,
    side: str = "S",
    remaining_size: int = 100,
    price: float = 2780.0,
    order_type: str = "LMT",
    status: str = "Submitted",
) -> dict:
    """Build a minimal IBKR live-order dict."""
    order: dict = {
        "side": side,
        "remainingSize": remaining_size,
        "price": price,
        "orderType": order_type,
        "status": status,
    }
    if conid is not None:
        order["conid"] = conid
    if ticker is not None:
        order["ticker"] = ticker
    return order


def _make_buy_item(
    ticker: str = "7203.T",
    conviction: str = "High",
    size_pct: float = 4.0,
    suggested_quantity: int | None = 100,
    suggested_price: float | None = 2615.0,
    cash_impact_usd: float = -1752.0,
    analysis_date: str = "2026-03-01",
    analysis: AnalysisRecord | None = None,
    is_watchlist: bool = True,
) -> ReconciliationItem:
    """Build a BUY ReconciliationItem as the reconciler would produce for new buys."""
    if analysis is None:
        tb = TradeBlockData(conviction=conviction, size_pct=size_pct)
        analysis = AnalysisRecord(
            ticker=ticker,
            analysis_date=analysis_date,
            verdict="BUY",
            health_adj=72.0,
            growth_adj=65.0,
            trade_block=tb,
            conviction=conviction,
        )
    return ReconciliationItem(
        ticker=ticker,
        action="BUY",
        urgency="MEDIUM",
        reason=f"Watchlist BUY ({analysis_date}) — {conviction} conviction, target {size_pct:.1f}%",
        ibkr_position=None,
        analysis=analysis,
        suggested_quantity=suggested_quantity,
        suggested_price=suggested_price,
        suggested_order_type="LMT",
        cash_impact_usd=cash_impact_usd,
        is_watchlist=is_watchlist,
    )


def _make_offwatch_buy(
    ticker: str = "WDO.TO", conviction: str = "High"
) -> ReconciliationItem:
    """Build a Phase-2 off-watchlist BUY item (is_watchlist=False)."""
    tb = TradeBlockData(conviction=conviction, size_pct=3.0)
    analysis = AnalysisRecord(
        ticker=ticker,
        analysis_date="2026-03-01",
        verdict="BUY",
        health_adj=70.0,
        growth_adj=62.0,
        trade_block=tb,
        conviction=conviction,
    )
    return ReconciliationItem(
        ticker=ticker,
        action="BUY",
        urgency="MEDIUM",
        reason="Off-watchlist BUY",
        ibkr_position=None,
        analysis=analysis,
        suggested_quantity=100,
        suggested_price=15.0,
        cash_impact_usd=-1500.0,
        is_watchlist=False,
    )


def _make_watchlist_review(ticker: str, *, quick: bool) -> ReconciliationItem:
    """Build a watchlist REVIEW item as the evaluator emits it (quick-mode BUY
    verdicts and stale analyses both arrive as REVIEW, never BUY)."""
    analysis = AnalysisRecord(
        ticker=ticker,
        analysis_date="2026-07-01",
        verdict="BUY" if quick else "HOLD",
        health_adj=70.0,
        growth_adj=60.0,
        trade_block=TradeBlockData(conviction="High", size_pct=3.0),
        conviction="High",
        is_quick_mode=quick,
    )
    reason = (
        "Watchlist quick-mode screening BUY (2026-07-01) — "
        "re-run full analysis before acting"
        if quick
        else "Watchlist: stale analysis (age 30d)"
    )
    return ReconciliationItem(
        ticker=ticker,
        action="REVIEW",
        urgency="LOW",
        reason=reason,
        analysis=analysis,
        is_watchlist=True,
    )
