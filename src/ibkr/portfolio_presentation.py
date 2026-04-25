from __future__ import annotations

from dataclasses import dataclass

from src.ibkr.dip_watch import select_dip_watch_candidates
from src.ibkr.models import PortfolioSummary, ReconciliationItem
from src.ibkr.refresh_service import AnalysisFreshnessSummary, RefreshActivity

_DEFAULT_DIP_WATCH_LIMIT = 7


@dataclass(frozen=True)
class PortfolioActionGroups:
    stop_sells: tuple[ReconciliationItem, ...]
    hard_sells: tuple[ReconciliationItem, ...]
    soft_sells: tuple[ReconciliationItem, ...]
    macro_reviews: tuple[ReconciliationItem, ...]
    macro_stop_reviews: tuple[ReconciliationItem, ...]
    trims: tuple[ReconciliationItem, ...]
    removes: tuple[ReconciliationItem, ...]
    adds: tuple[ReconciliationItem, ...]
    new_buys: tuple[ReconciliationItem, ...]
    watchlist_candidates: tuple[ReconciliationItem, ...]
    holds_real: tuple[ReconciliationItem, ...]
    holds_watch: tuple[ReconciliationItem, ...]
    reviews: tuple[ReconciliationItem, ...]
    dip_candidates: tuple[ReconciliationItem, ...]


@dataclass(frozen=True)
class CashTimelineEntry:
    ticker_yf: str
    ticker_ibkr: str
    action: str
    quantity: int | None
    cash_impact_usd: float
    settlement_date: str | None


@dataclass(frozen=True)
class CashSummaryView:
    total_cash_usd: float
    settled_cash_usd: float
    available_cash_usd: float
    buffer_reserve_usd: float
    unsettled_cash_usd: float
    recommended_buy_cost_usd: float
    settled_cash_after_recommended_buys_usd: float
    pending_inflows: tuple[CashTimelineEntry, ...]
    pending_inflows_total_usd: float
    conditional_proceeds_usd: float  # soft-sell proceeds (review before acting)
    next_settlement_date: str | None


@dataclass(frozen=True)
class FreshnessOverviewView:
    blocking_now: int
    stale_in_queue: int
    due_soon: int
    candidate_blocked: int
    fresh_count: int
    refreshed_count: int
    failed_count: int
    queued_count: int
    skipped_due_to_limit: int
    skipped_read_only: int


@dataclass(frozen=True)
class PortfolioOverviewView:
    sell_count: int
    review_count: int
    hold_count: int
    macro_watch_count: int
    new_buy_count: int
    candidate_count: int
    total_items: int
    position_count: int
    has_live_positions: bool
    is_candidate_heavy: bool


@dataclass(frozen=True)
class LiveOrderMatch:
    order: dict
    side: str
    quantity: int | None
    price: float | str | None
    order_type: str
    status: str


def group_portfolio_actions(
    items: list[ReconciliationItem],
    *,
    watchlist_tickers: set[str] | None = None,
    dip_watch_limit: int = _DEFAULT_DIP_WATCH_LIMIT,
) -> PortfolioActionGroups:
    stop_sells = tuple(
        item
        for item in items
        if item.action == "SELL" and item.sell_type == "STOP_BREACH"
    )
    hard_sells = tuple(
        item
        for item in items
        if item.action == "SELL" and item.sell_type in (None, "HARD_REJECT")
    )
    soft_sells = tuple(
        item
        for item in items
        if item.action == "SELL" and item.sell_type == "SOFT_REJECT"
    )
    macro_reviews = tuple(
        item
        for item in items
        if item.action == "REVIEW" and item.sell_type == "SOFT_REJECT"
    )
    macro_stop_reviews = tuple(
        item
        for item in items
        if item.action == "REVIEW" and item.sell_type == "STOP_BREACH"
    )
    trims = tuple(item for item in items if item.action == "TRIM")
    removes = tuple(item for item in items if item.action == "REMOVE")
    adds = tuple(item for item in items if item.action == "ADD")
    # BUY remains the internal recommendation type; presentation decides whether
    # that surfaces as a live buy or as an advisory watchlist candidate.
    new_buys = tuple(
        item
        for item in items
        if item.action == "BUY" and item.ibkr_position is None and item.is_watchlist
    )
    buys_offwatch = tuple(
        item
        for item in items
        if item.action == "BUY" and item.ibkr_position is None and not item.is_watchlist
    )
    holds_real = tuple(
        item for item in items if item.action == "HOLD" and not item.is_watchlist
    )
    holds_watch = tuple(
        item for item in items if item.action == "HOLD" and item.is_watchlist
    )
    reviews = tuple(
        item
        for item in items
        if item.action == "REVIEW"
        and item.sell_type not in ("SOFT_REJECT", "STOP_BREACH")
    )

    action_bases = frozenset(
        base_ticker(item) for item in removes + stop_sells + hard_sells + soft_sells
    )
    held_bases = frozenset(
        base_ticker(item) for item in items if item.ibkr_position is not None
    )
    watchlist_bases = frozenset(
        base_ticker_value(ticker) for ticker in (watchlist_tickers or set())
    )
    watchlist_candidates = tuple(
        item
        for item in buys_offwatch
        if base_ticker(item) not in (action_bases | held_bases | watchlist_bases)
    )
    dip_candidates = tuple(
        select_dip_watch_candidates(
            list(macro_reviews),
            limit=dip_watch_limit,
        )
    )

    return PortfolioActionGroups(
        stop_sells=stop_sells,
        hard_sells=hard_sells,
        soft_sells=soft_sells,
        macro_reviews=macro_reviews,
        macro_stop_reviews=macro_stop_reviews,
        trims=trims,
        removes=removes,
        adds=adds,
        new_buys=new_buys,
        watchlist_candidates=watchlist_candidates,
        holds_real=holds_real,
        holds_watch=holds_watch,
        reviews=reviews,
        dip_candidates=dip_candidates,
    )


def build_action_summary_counts(groups: PortfolioActionGroups) -> dict[str, int]:
    counts: dict[str, int] = {}
    if groups.stop_sells or groups.hard_sells or groups.soft_sells:
        counts["SELL"] = (
            len(groups.stop_sells) + len(groups.hard_sells) + len(groups.soft_sells)
        )
    if groups.removes:
        counts["REMOVE"] = len(groups.removes)
    if groups.trims:
        counts["TRIM"] = len(groups.trims)
    if groups.adds:
        counts["ADD"] = len(groups.adds)
    if groups.new_buys:
        counts["BUY"] = len(groups.new_buys)
    if groups.watchlist_candidates:
        counts["CANDIDATES"] = len(groups.watchlist_candidates)
    if groups.holds_real:
        counts["HOLD"] = len(groups.holds_real)
    if groups.reviews or groups.macro_stop_reviews:
        counts["REVIEW"] = len(groups.reviews) + len(groups.macro_stop_reviews)
    if groups.macro_reviews:
        counts["MACRO_WATCH"] = len(groups.macro_reviews)
    return counts


def build_cash_timeline(
    items: list[ReconciliationItem],
) -> tuple[CashTimelineEntry, ...]:
    """Build confirmed pending inflows from sells/trims.

    SOFT_REJECT sells are excluded — they are "review before acting" and
    should not be counted as confirmed liquidity.  Their individual proceeds
    are still shown in the soft-sell display section.
    """
    rows = [
        CashTimelineEntry(
            ticker_yf=item.ticker.yf,
            ticker_ibkr=item.ticker.ibkr,
            action=item.action,
            quantity=item.suggested_quantity,
            cash_impact_usd=item.cash_impact_usd,
            settlement_date=item.settlement_date,
        )
        for item in items
        if item.action in {"SELL", "TRIM"}
        and item.sell_type != "SOFT_REJECT"
        and item.cash_impact_usd > 0
        and item.settlement_date
    ]
    rows.sort(key=lambda row: (row.settlement_date or "", row.ticker_yf))
    return tuple(rows)


def _soft_sell_proceeds_usd(items: list[ReconciliationItem]) -> float:
    """Total USD proceeds from SOFT_REJECT sells (conditional, not confirmed)."""
    return sum(
        item.cash_impact_usd
        for item in items
        if item.action == "SELL"
        and item.sell_type == "SOFT_REJECT"
        and item.cash_impact_usd > 0
    )


def build_cash_summary(
    items: list[ReconciliationItem],
    portfolio: PortfolioSummary,
) -> CashSummaryView:
    settled_cash = portfolio.settled_cash_usd
    available_cash = portfolio.available_cash_usd
    total_cash = portfolio.cash_balance_usd
    buffer_reserve = max(settled_cash - available_cash, 0.0)
    unsettled_cash = max(total_cash - settled_cash, 0.0)
    recommended_buy_cost = sum(
        abs(item.cash_impact_usd)
        for item in items
        if item.action in {"ADD", "BUY"}
        and item.cash_impact_usd < 0
        and (item.action != "BUY" or item.is_watchlist)
    )
    pending_inflows = build_cash_timeline(items)
    return CashSummaryView(
        total_cash_usd=total_cash,
        settled_cash_usd=settled_cash,
        available_cash_usd=available_cash,
        buffer_reserve_usd=buffer_reserve,
        unsettled_cash_usd=unsettled_cash,
        recommended_buy_cost_usd=recommended_buy_cost,
        settled_cash_after_recommended_buys_usd=settled_cash - recommended_buy_cost,
        pending_inflows=pending_inflows,
        pending_inflows_total_usd=sum(row.cash_impact_usd for row in pending_inflows),
        conditional_proceeds_usd=_soft_sell_proceeds_usd(items),
        next_settlement_date=(
            min(
                (row.settlement_date for row in pending_inflows if row.settlement_date),
                default=None,
            )
        ),
    )


def build_freshness_overview(
    freshness_summary: AnalysisFreshnessSummary,
    refresh_activity: RefreshActivity,
) -> FreshnessOverviewView:
    return FreshnessOverviewView(
        blocking_now=len(freshness_summary.blocking_now),
        stale_in_queue=len(freshness_summary.stale_in_queue),
        due_soon=len(freshness_summary.due_soon),
        candidate_blocked=len(freshness_summary.candidate_blocked),
        fresh_count=len(freshness_summary.fresh),
        refreshed_count=len(refresh_activity.refreshed),
        failed_count=len(refresh_activity.failed),
        queued_count=len(refresh_activity.queued),
        skipped_due_to_limit=len(refresh_activity.skipped_due_to_limit),
        skipped_read_only=len(refresh_activity.skipped_read_only),
    )


def build_portfolio_overview(
    items: list[ReconciliationItem],
    portfolio: PortfolioSummary,
    *,
    watchlist_tickers: set[str] | None = None,
) -> PortfolioOverviewView:
    groups = group_portfolio_actions(items, watchlist_tickers=watchlist_tickers)
    counts = build_action_summary_counts(groups)
    candidate_count = counts.get("CANDIDATES", 0)
    new_buy_count = counts.get("BUY", 0)
    position_count = portfolio.position_count
    return PortfolioOverviewView(
        sell_count=counts.get("SELL", 0),
        review_count=counts.get("REVIEW", 0),
        hold_count=counts.get("HOLD", 0),
        macro_watch_count=counts.get("MACRO_WATCH", 0),
        new_buy_count=new_buy_count,
        candidate_count=candidate_count,
        total_items=len(items),
        position_count=position_count,
        has_live_positions=position_count > 0,
        is_candidate_heavy=position_count == 0
        and (candidate_count > 0 or new_buy_count > 0),
    )


def find_live_order(
    item: ReconciliationItem,
    live_orders: list[dict] | None,
) -> LiveOrderMatch | None:
    if not live_orders:
        return None

    pos = item.ibkr_position
    conid = pos.conid if pos else None
    yf_base = item.ticker.ibkr.upper()
    hk_padded = item.ticker.yf.split(".")[0].upper()
    symbol_candidates: set[str] = {yf_base, hk_padded}
    if pos and pos.symbol:
        symbol_candidates.add(pos.symbol.upper())

    for order in live_orders:
        matched = False
        order_conid = order.get("conid")
        order_symbol = (order.get("ticker") or order.get("symbol") or "").upper()
        if conid and order_conid is not None:
            try:
                matched = int(order_conid) == int(conid)
            except (TypeError, ValueError):
                matched = False
        if not matched and order_symbol in symbol_candidates:
            matched = True
        if not matched:
            continue

        raw_quantity = order.get("remainingSize") or order.get("totalSize")
        quantity: int | None
        try:
            quantity = int(raw_quantity) if raw_quantity is not None else None
        except (TypeError, ValueError):
            quantity = None
        side = "SELL" if str(order.get("side", "")).upper() in {"S", "SELL"} else "BUY"
        return LiveOrderMatch(
            order=order,
            side=side,
            quantity=quantity,
            price=order.get("price") or order.get("auxPrice"),
            order_type=str(order.get("orderType") or "LMT"),
            status=str(order.get("status") or ""),
        )
    return None


def build_live_order_note(
    item: ReconciliationItem,
    live_orders: list[dict] | None,
) -> str | None:
    match = find_live_order(item, live_orders)
    if match is None:
        return None

    if isinstance(match.price, int | float):
        price_str = f" @ {float(match.price):.2f}"
    elif match.price:
        price_str = f" @ {match.price}"
    else:
        price_str = ""

    rec_side = "SELL" if item.action in {"SELL", "TRIM"} else "BUY"
    display_qty = match.quantity if match.quantity is not None else "?"
    if match.side == rec_side:
        rec_qty = item.suggested_quantity
        if (
            match.quantity is not None
            and rec_qty is not None
            and match.quantity < rec_qty
        ):
            need = rec_qty - match.quantity
            return (
                f"[PARTIAL ORDER: {match.quantity} of {rec_qty} shares already submitted"
                f" — enter {need} more]"
            )
        return (
            f"[ORDER ALREADY SUBMITTED: {match.side} {display_qty}{price_str}"
            f" {match.order_type} ({match.status}) — do not re-enter]"
        )
    return (
        f"[CONFLICT: live {match.side} order {display_qty}{price_str}"
        f" {match.order_type} ({match.status}) while recommending {rec_side}]"
    )


def base_ticker(item: ReconciliationItem) -> str:
    return base_ticker_value(item.ticker.yf)


def base_ticker_value(ticker: str) -> str:
    return ticker.split(".")[0].upper()
