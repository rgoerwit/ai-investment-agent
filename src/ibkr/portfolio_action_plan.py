"""Authoritative action-plan preparation shared by portfolio output surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from src.ibkr.models import PortfolioSummary, ReconciliationItem
from src.ibkr.order_presentation import find_live_order
from src.ibkr.portfolio_defaults import (
    DEFAULT_EXCHANGE_LIMIT_PCT,
    DEFAULT_SECTOR_LIMIT_PCT,
)
from src.ibkr.portfolio_presentation import (
    PortfolioActionGroups,
    build_action_summary_counts,
    group_portfolio_actions,
)
from src.ibkr.watchlist_optimization import (
    WatchlistOptimization,
    resolve_watchlist_optimization,
    selected_watchlist_buy_ids,
)


@dataclass(frozen=True)
class PortfolioActionPlan:
    """One recommendation plan consumed by CLI, JSON, and dashboard views."""

    groups: PortfolioActionGroups
    optimization: WatchlistOptimization
    executable_buy_ids: frozenset[str]
    in_flight_buys: tuple[ReconciliationItem, ...]
    live_order_items: tuple[ReconciliationItem, ...]
    concentration_withheld_dips: tuple[ReconciliationItem, ...]
    macro_event_active: bool


def has_active_macro_event(health_flags: list[str] | None) -> bool:
    """Return whether current or stored macro-event policy is active."""
    return any(
        "CORRELATED_SELL_EVENT" in flag or "ACTIVE_MACRO_EVENT" in flag
        for flag in (health_flags or [])
    )


def _is_open_buy_match(
    item: ReconciliationItem, live_orders: list[dict] | None
) -> bool:
    match = find_live_order(item, live_orders)
    return bool(
        match is not None
        and match.side == "BUY"
        and match.status.strip().lower() != "filled"
    )


def build_portfolio_action_plan(
    items: list[ReconciliationItem],
    portfolio: PortfolioSummary,
    *,
    watchlist_tickers: set[str] | None,
    watchlist_supplied: bool,
    watchlist_unavailable: bool,
    live_orders: list[dict] | None = None,
    macro_event_active: bool = False,
    exchange_limit_pct: float = DEFAULT_EXCHANGE_LIMIT_PCT,
    sector_limit_pct: float = DEFAULT_SECTOR_LIMIT_PCT,
) -> PortfolioActionPlan:
    """Build the complete, presentation-neutral portfolio recommendation plan."""
    groups = group_portfolio_actions(
        items,
        watchlist_tickers=watchlist_tickers,
        macro_event_active=macro_event_active,
        exchange_weights=portfolio.exchange_weights,
        sector_weights=portfolio.sector_weights,
        exchange_limit_pct=exchange_limit_pct,
        sector_limit_pct=sector_limit_pct,
    )
    # The concentration-withheld dips come from the SAME single evaluation that
    # produced groups.dip_candidates — no second select+screen pass.
    withheld_dips = groups.dip_withheld

    in_flight_buys = tuple(
        item
        for item in items
        if item.action == "BUY"
        and item.ibkr_position is None
        and not item.is_watchlist
        and _is_open_buy_match(item, live_orders)
    )
    in_flight_ids = {item.ticker.yf.upper() for item in in_flight_buys}
    optimization = resolve_watchlist_optimization(
        [item for item in items if item.ticker.yf.upper() not in in_flight_ids],
        groups,
        watchlist_tickers=watchlist_tickers,
        watchlist_supplied=watchlist_supplied,
        watchlist_unavailable=watchlist_unavailable,
        exchange_weights=portfolio.exchange_weights,
        sector_weights=portfolio.sector_weights,
        exchange_limit_pct=exchange_limit_pct,
        sector_limit_pct=sector_limit_pct,
    )
    live_order_items = tuple(
        item
        for item in items
        if item.action == "BUY"
        and item.ibkr_position is None
        and find_live_order(item, live_orders) is not None
    )

    return PortfolioActionPlan(
        groups=groups,
        optimization=optimization,
        executable_buy_ids=frozenset(selected_watchlist_buy_ids(optimization)),
        in_flight_buys=in_flight_buys,
        live_order_items=live_order_items,
        concentration_withheld_dips=tuple(withheld_dips),
        macro_event_active=macro_event_active,
    )


def build_action_plan_counts(
    plan: PortfolioActionPlan,
    items: list[ReconciliationItem],
) -> dict[str, int]:
    """Return the canonical counts used by dashboard and machine output."""
    counts = build_action_summary_counts(plan.groups)
    optimization = plan.optimization
    return {
        "buys": len(optimization.keep),
        "candidates": len(optimization.add),
        "sells": counts.get("SELL", 0),
        "reviews": counts.get("REVIEW", 0),
        "holds": counts.get("HOLD", 0),
        "macro_watch": counts.get("MACRO_WATCH", 0),
        "watchlist": sum(1 for item in items if item.is_watchlist),
        "watchlist_removes": len(optimization.remove),
        "watchlist_withheld": len(optimization.withheld_candidates),
        "watchlist_capacity_limited": len(optimization.capacity_limited_candidates),
        "watchlist_below_conviction": len(optimization.excluded_low_conviction),
        "watchlist_in_flight": len(plan.in_flight_buys),
        "total": len(items),
    }
