from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.ibkr.dip_watch import (
    collect_dip_watch_source_items,
    select_dip_watch_candidates,
)
from src.ibkr.models import PortfolioSummary, ReconciliationItem
from src.ibkr.refresh_service import AnalysisFreshnessSummary, RefreshActivity
from src.ibkr.ticker import Ticker

_DEFAULT_DIP_WATCH_LIMIT = 7
TARGET_WATCHLIST_SIZE = 6
WATCHLIST_MIN_CONVICTION = "medium"
_WATCHLIST_CONVICTION_RANK = {"high": 0, "medium": 1, "low": 2}
SELL_RECOMMENDATIONS_TITLE = "SELL RECOMMENDATIONS"
SELL_RELATED_REVIEWS_TITLE = "SELL-RELATED REVIEWS"
SELL_TYPE_LABELS: dict[str | None, str] = {
    "STOP_BREACH": "STOP BREACH",
    "HARD_REJECT": "FUNDAMENTAL FAILURE",
    "SOFT_REJECT": "SOFT REJECTION",
    "SCREEN_REJECT": "SCREEN REVIEW",
    "DATA_QUALITY_REVIEW": "DATA REVIEW",
    "SPECIAL_SITUATION_EXIT": "M&A EXIT",
    "PROFIT_TAKE": "PROFIT TAKE",
    None: "SELL",
}

# Decision-basis labels — preferred over sell_type when a basis is stamped,
# so a confirmation-gated exit reads as what it is, not a generic "failure".
ACTION_BASIS_LABELS: dict[str, str] = {
    "MANDATORY_EXIT": "MANDATORY EXIT",
    "STOP_LOSS": "STOP BREACH",
    "CONFIRMED_THESIS_FAILURE": "CONFIRMED THESIS FAILURE",
    "THESIS_REASSESSMENT": "THESIS REASSESSMENT",
    "ENTRY_CONSTRAINT": "ENTRY CONSTRAINT",
    "SPECIAL_SITUATION_REVIEW": "M&A TENDER REVIEW",
    "DATA_QUALITY": "DATA REVIEW",
    "CAPITAL_ALLOCATION": "PROFIT TAKE",
    "DE_MINIMIS": "DE MINIMIS",
}


@dataclass(frozen=True)
class PortfolioActionGroups:
    stop_sells: tuple[ReconciliationItem, ...]
    hard_sells: tuple[ReconciliationItem, ...]
    profit_take_sells: tuple[ReconciliationItem, ...]
    soft_sells: tuple[ReconciliationItem, ...]
    profit_take_reviews: tuple[ReconciliationItem, ...]
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


class WatchlistOptCase(str, Enum):
    """Operator-facing states for the watchlist optimizer."""

    NO_WATCHLIST = "no_watchlist"
    WATCHLIST_UNAVAILABLE = "watchlist_unavailable"
    NOTHING_ACTIONABLE = "nothing_actionable"
    EMPTY_POOL = "empty_pool"
    PARTIAL_FILL = "partial_fill"
    ALIGNED = "aligned"
    FULL_OPTIMIZE = "full_optimize"


@dataclass(frozen=True)
class WatchlistMove:
    item: ReconciliationItem
    reason: str


@dataclass(frozen=True)
class WatchlistOptimization:
    """Pure watchlist recommendation separated from report rendering.

    ``protected_tickers`` are live holdings or unresolved watchlist entries. They
    are deliberately retained outside the six BUY-ready slots because their
    membership cannot be safely changed from this reconciliation alone.
    """

    case: WatchlistOptCase
    watchlist_supplied: bool
    target_size: int
    optimal: tuple[ReconciliationItem, ...]
    keep: tuple[ReconciliationItem, ...]
    add: tuple[ReconciliationItem, ...]
    remove: tuple[WatchlistMove, ...]
    monitors: tuple[ReconciliationItem, ...]
    reviews: tuple[ReconciliationItem, ...]
    protected_tickers: tuple[str, ...]
    excluded_low_conviction: tuple[ReconciliationItem, ...]
    pool_size: int


def _watchlist_ticker_identity(ticker: str) -> str:
    """Return the exchange-qualified identity already managed by ``Ticker``.

    Never strip a suffix here: same-base symbols can be distinct listings. A
    bare symbol remains bare and is protected if it cannot be matched exactly.
    """
    return Ticker.from_yf(ticker).yf.upper()


def _item_ticker_identity(item: ReconciliationItem) -> str:
    return item.ticker.yf.upper()


def watchlist_candidate_conviction(item: ReconciliationItem) -> str:
    """Return the normalized conviction token used by selection and rendering."""
    analysis = item.analysis
    if analysis is None:
        return ""
    raw = analysis.conviction or analysis.trade_block.conviction or ""
    return raw.strip().lower()


def watchlist_candidate_score(item: ReconciliationItem) -> float:
    """Return the deterministic health-plus-growth rank score."""
    analysis = item.analysis
    if analysis is None:
        return 0.0
    return (analysis.health_adj or 0.0) + (analysis.growth_adj or 0.0)


def resolve_watchlist_optimization(
    items: list[ReconciliationItem],
    groups: PortfolioActionGroups,
    *,
    watchlist_tickers: set[str] | None,
    watchlist_supplied: bool,
    watchlist_unavailable: bool,
    target_size: int = TARGET_WATCHLIST_SIZE,
    min_conviction: str = WATCHLIST_MIN_CONVICTION,
) -> WatchlistOptimization:
    """Select up to ``target_size`` medium-or-higher unheld BUYs safely.

    Membership and deduplication use full yfinance identities, rather than base
    symbols, to avoid collapsing listings that happen to share a ticker.
    """
    if target_size < 0:
        raise ValueError("target_size must be non-negative")

    min_rank = _WATCHLIST_CONVICTION_RANK.get(min_conviction.lower())
    if min_rank is None:
        raise ValueError(f"unsupported watchlist conviction: {min_conviction}")

    raw_watchlist = {
        _watchlist_ticker_identity(ticker) for ticker in (watchlist_tickers or set())
    }
    watchlist_items = tuple(item for item in items if item.is_watchlist)
    watched_item_ids = {_item_ticker_identity(item) for item in watchlist_items}
    held_ids = {
        _item_ticker_identity(item) for item in items if item.ibkr_position is not None
    }
    protected_tickers = tuple(
        sorted((raw_watchlist - watched_item_ids) | (raw_watchlist & held_ids))
    )

    # Retain the first exact identity, preferring a current watchlist member to
    # minimize churn. Full identities intentionally keep BHP.AX and BHP.L apart.
    pooled_by_identity: dict[str, ReconciliationItem] = {}
    pool_candidates = sorted(
        (item for item in items if item.action == "BUY" and item.ibkr_position is None),
        key=lambda item: (not item.is_watchlist, _item_ticker_identity(item)),
    )
    for item in pool_candidates:
        pooled_by_identity.setdefault(_item_ticker_identity(item), item)

    pool = tuple(pooled_by_identity.values())
    eligible: list[ReconciliationItem] = []
    excluded_low_conviction: list[ReconciliationItem] = []
    for item in pool:
        conviction_rank = _WATCHLIST_CONVICTION_RANK.get(
            watchlist_candidate_conviction(item), len(_WATCHLIST_CONVICTION_RANK)
        )
        if conviction_rank > min_rank:
            excluded_low_conviction.append(item)
        else:
            eligible.append(item)

    optimal = tuple(
        sorted(
            eligible,
            key=lambda item: (
                _WATCHLIST_CONVICTION_RANK[watchlist_candidate_conviction(item)],
                -watchlist_candidate_score(item),
                not item.is_watchlist,
                _item_ticker_identity(item),
            ),
        )[:target_size]
    )
    optimal_ids = {_item_ticker_identity(item) for item in optimal}

    if watchlist_unavailable:
        return WatchlistOptimization(
            case=WatchlistOptCase.WATCHLIST_UNAVAILABLE,
            watchlist_supplied=False,
            target_size=target_size,
            optimal=optimal,
            keep=(),
            add=optimal,
            remove=(),
            monitors=(),
            reviews=(),
            protected_tickers=(),
            excluded_low_conviction=tuple(excluded_low_conviction),
            pool_size=len(pool),
        )

    if not watchlist_supplied:
        return WatchlistOptimization(
            case=WatchlistOptCase.NO_WATCHLIST,
            watchlist_supplied=False,
            target_size=target_size,
            optimal=optimal,
            keep=(),
            add=optimal,
            remove=(),
            monitors=(),
            reviews=(),
            protected_tickers=(),
            excluded_low_conviction=tuple(excluded_low_conviction),
            pool_size=len(pool),
        )

    reject_ids = {_item_ticker_identity(item) for item in groups.removes}
    monitor_ids = {_item_ticker_identity(item) for item in groups.holds_watch}
    review_ids = {
        _item_ticker_identity(item) for item in groups.reviews if item.is_watchlist
    }
    keep = tuple(
        item
        for item in optimal
        if item.is_watchlist or _item_ticker_identity(item) in raw_watchlist
    )
    add = tuple(item for item in optimal if item not in keep)
    monitors = tuple(
        item
        for item in groups.holds_watch
        if _item_ticker_identity(item) not in optimal_ids
    )
    reviews = tuple(
        item
        for item in groups.reviews
        if item.is_watchlist and _item_ticker_identity(item) not in optimal_ids
    )
    removals: list[WatchlistMove] = []
    for item in watchlist_items:
        identity = _item_ticker_identity(item)
        if identity in optimal_ids or identity in monitor_ids or identity in review_ids:
            continue
        if identity in reject_ids or item.action == "REMOVE":
            removals.append(WatchlistMove(item, "verdict_reject"))
        elif item.action == "BUY":
            reason = (
                "below_medium_conviction"
                if item in excluded_low_conviction
                else "displaced_by_higher_conviction"
            )
            removals.append(WatchlistMove(item, reason))

    if not optimal and not raw_watchlist and not watchlist_items:
        case = WatchlistOptCase.NOTHING_ACTIONABLE
    elif not optimal:
        case = WatchlistOptCase.EMPTY_POOL
    elif len(optimal) < target_size:
        case = WatchlistOptCase.PARTIAL_FILL
    elif not add and not removals:
        case = WatchlistOptCase.ALIGNED
    else:
        case = WatchlistOptCase.FULL_OPTIMIZE

    return WatchlistOptimization(
        case=case,
        watchlist_supplied=True,
        target_size=target_size,
        optimal=optimal,
        keep=keep,
        add=add,
        remove=tuple(removals),
        monitors=monitors,
        reviews=reviews,
        protected_tickers=protected_tickers,
        excluded_low_conviction=tuple(excluded_low_conviction),
        pool_size=len(pool),
    )


def build_watchlist_optimization_summary(
    optimization: WatchlistOptimization,
) -> dict[str, int]:
    """Return CLI-specific watchlist counts without changing dashboard buckets."""
    counts: dict[str, int] = {}
    if optimization.keep:
        counts["WATCHLIST_KEEP"] = len(optimization.keep)
    if optimization.add:
        counts["WATCHLIST_ADD"] = len(optimization.add)
    if optimization.remove:
        counts["WATCHLIST_REMOVE"] = len(optimization.remove)
    if optimization.monitors:
        counts["WATCHLIST_MONITOR"] = len(optimization.monitors)
    if optimization.reviews:
        counts["WATCHLIST_REVIEW"] = len(optimization.reviews)
    return counts


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


@dataclass(frozen=True)
class ActionDisplaySection:
    key: str
    title: str
    kind: str
    items: tuple[object, ...]


def get_sell_type_label(sell_type: str | None) -> str:
    """Return the canonical human label for a sell_type token."""
    return SELL_TYPE_LABELS.get(sell_type, "SELL")


def get_action_label(item: ReconciliationItem) -> str:
    """Return the display label for an item, preferring its decision basis."""
    basis = getattr(item, "action_basis", None)
    if basis and basis in ACTION_BASIS_LABELS:
        return ACTION_BASIS_LABELS[basis]
    return get_sell_type_label(item.sell_type)


def group_portfolio_actions(
    items: list[ReconciliationItem],
    *,
    watchlist_tickers: set[str] | None = None,
    dip_watch_limit: int = _DEFAULT_DIP_WATCH_LIMIT,
    macro_event_active: bool = False,
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
    profit_take_sells = tuple(
        item
        for item in items
        if item.action == "SELL" and item.sell_type == "PROFIT_TAKE"
    )
    soft_sells = tuple(
        item
        for item in items
        if item.action == "SELL" and item.sell_type == "SOFT_REJECT"
    )
    profit_take_reviews = tuple(
        item
        for item in items
        if item.action == "REVIEW" and item.sell_type == "PROFIT_TAKE"
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
        and item.sell_type not in ("SOFT_REJECT", "STOP_BREACH", "PROFIT_TAKE")
    )

    action_bases = frozenset(
        base_ticker(item)
        for item in removes + stop_sells + hard_sells + profit_take_sells + soft_sells
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
            collect_dip_watch_source_items(items),
            macro_event_active=macro_event_active,
            limit=dip_watch_limit,
        )
    )

    return PortfolioActionGroups(
        stop_sells=stop_sells,
        hard_sells=hard_sells,
        profit_take_sells=profit_take_sells,
        soft_sells=soft_sells,
        profit_take_reviews=profit_take_reviews,
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
    if (
        groups.stop_sells
        or groups.hard_sells
        or groups.profit_take_sells
        or groups.soft_sells
    ):
        counts["SELL"] = (
            len(groups.stop_sells)
            + len(groups.hard_sells)
            + len(groups.profit_take_sells)
            + len(groups.soft_sells)
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
    if groups.reviews or groups.profit_take_reviews or groups.macro_stop_reviews:
        counts["REVIEW"] = (
            len(groups.reviews)
            + len(groups.profit_take_reviews)
            + len(groups.macro_stop_reviews)
        )
    if groups.macro_reviews:
        counts["MACRO_WATCH"] = len(groups.macro_reviews)
    return counts


def build_action_display_sections(
    groups: PortfolioActionGroups,
    *,
    dip_watch_items: tuple[object, ...] | None = None,
) -> tuple[ActionDisplaySection, ...]:
    """Return canonical action sections for operator-facing held-position views."""
    sections: list[ActionDisplaySection] = []

    sell_recommendations = (
        groups.stop_sells
        + groups.hard_sells
        + groups.profit_take_sells
        + groups.soft_sells
    )
    if sell_recommendations:
        sections.append(
            ActionDisplaySection(
                key="sell_recommendations",
                title=SELL_RECOMMENDATIONS_TITLE,
                kind="reconciliation_items",
                items=sell_recommendations,
            )
        )

    sell_related_reviews = (
        groups.macro_stop_reviews + groups.macro_reviews + groups.profit_take_reviews
    )
    if sell_related_reviews:
        sections.append(
            ActionDisplaySection(
                key="sell_related_reviews",
                title=SELL_RELATED_REVIEWS_TITLE,
                kind="reconciliation_items",
                items=sell_related_reviews,
            )
        )

    if groups.adds:
        sections.append(
            ActionDisplaySection(
                key="add",
                title="Adds",
                kind="reconciliation_items",
                items=groups.adds,
            )
        )
    if groups.trims:
        sections.append(
            ActionDisplaySection(
                key="trim",
                title="Trims",
                kind="reconciliation_items",
                items=groups.trims,
            )
        )
    if groups.reviews:
        sections.append(
            ActionDisplaySection(
                key="review",
                title="Review Queue",
                kind="reconciliation_items",
                items=groups.reviews,
            )
        )
    if groups.dip_candidates:
        sections.append(
            ActionDisplaySection(
                key="dip_watch",
                title="Dip Watch",
                kind="dip_watch",
                items=(
                    dip_watch_items
                    if dip_watch_items is not None
                    else groups.dip_candidates
                ),
            )
        )
    if groups.holds_real:
        sections.append(
            ActionDisplaySection(
                key="hold",
                title="Holds",
                kind="reconciliation_items",
                items=groups.holds_real,
            )
        )

    return tuple(sections)


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
    """Total USD *potential exit value* of SOFT_REJECT items — never funds.

    SELL items contribute their estimated proceeds (`cash_impact_usd`).
    REVIEW items carry no order fields by contract (a review is not an
    order), so their potential exit value derives from the held position's
    market value. This is the conditional bucket ("soft-sell reviews") —
    strictly informational, excluded from pending/executable cash by
    `build_cash_timeline`'s SELL/TRIM filter.
    """
    total = 0.0
    for item in items:
        if item.sell_type != "SOFT_REJECT":
            continue
        if item.action == "SELL" and item.cash_impact_usd > 0:
            total += item.cash_impact_usd
        elif item.action == "REVIEW":
            if item.cash_impact_usd > 0:
                total += item.cash_impact_usd
            elif item.ibkr_position and item.ibkr_position.market_value_usd > 0:
                total += item.ibkr_position.market_value_usd
    return total


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


# Orders in a terminal state are not live: Cancelled/Inactive never annotate,
# Filled is surfaced only as historical context when no open order matches.
_TERMINAL_ORDER_STATUSES = frozenset({"cancelled", "inactive", "filled"})


def _build_live_order_match(order: dict) -> LiveOrderMatch:
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


def find_live_order(
    item: ReconciliationItem,
    live_orders: list[dict] | None,
) -> LiveOrderMatch | None:
    """Match the first genuinely open order for the item; a Filled order is
    returned only when no open order matches (open-before-filled — a filled
    order encountered first must not hide a later open cross-side conflict)."""
    if not live_orders:
        return None

    pos = item.ibkr_position
    conid = pos.conid if pos else None
    yf_base = item.ticker.ibkr.upper()
    hk_padded = item.ticker.yf.split(".")[0].upper()
    symbol_candidates: set[str] = {yf_base, hk_padded}
    if pos and pos.symbol:
        symbol_candidates.add(pos.symbol.upper())

    filled_fallback: LiveOrderMatch | None = None
    for order in live_orders:
        matched = False
        order_conid = order.get("conid")
        order_symbol = (order.get("ticker") or order.get("symbol") or "").upper()
        if conid and order_conid is not None:
            try:
                if int(order_conid) != int(conid):
                    # Comparable conids that differ are authoritative — never
                    # fall back to symbol (bare-symbol collisions across
                    # exchanges, e.g. SGX AGS vs Brussels Ageas AGS).
                    continue
                matched = True
            except (TypeError, ValueError):
                matched = False
        if not matched and order_symbol in symbol_candidates:
            matched = True
        if not matched:
            continue

        status = str(order.get("status") or "").strip().lower()
        if status in _TERMINAL_ORDER_STATUSES:
            # Filled is historical context, kept only if no open order
            # matches; Cancelled/Inactive are dead and never annotate.
            if status == "filled" and filled_fallback is None:
                filled_fallback = _build_live_order_match(order)
            continue

        return _build_live_order_match(order)
    return filled_fallback


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
    if match.status.strip().lower() == "filled":
        # Historical information, not a live order — no conflict language and
        # no "do not re-enter" imperative.
        return (
            f"[ORDER FILLED: {match.side} {display_qty}{price_str} {match.order_type}]"
        )
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
