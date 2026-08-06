from __future__ import annotations

import math
from collections.abc import Set
from dataclasses import dataclass

from src.ibkr.dip_watch import (
    collect_dip_watch_source_items,
    select_dip_watch,
)
from src.ibkr.models import NormalizedPosition, PortfolioSummary, ReconciliationItem
from src.ibkr.order_presentation import (
    LiveOrderMatch as LiveOrderMatch,
)
from src.ibkr.order_presentation import (
    base_ticker as base_ticker,
)
from src.ibkr.order_presentation import (
    base_ticker_value as base_ticker_value,
)
from src.ibkr.order_presentation import (
    build_live_order_note as build_live_order_note,
)
from src.ibkr.order_presentation import (
    find_live_order as find_live_order,
)
from src.ibkr.portfolio_defaults import (
    DEFAULT_EXCHANGE_LIMIT_PCT,
    DEFAULT_SECTOR_LIMIT_PCT,
)
from src.ibkr.reconciliation_rules import analysis_identity_verified
from src.ibkr.refresh_service import AnalysisFreshnessSummary, RefreshActivity
from src.ibkr.watchlist_optimization import (
    CONCENTRATION_INCUMBENT_MIN_SCORE as CONCENTRATION_INCUMBENT_MIN_SCORE,
)
from src.ibkr.watchlist_optimization import (
    ConcentrationNote as ConcentrationNote,
)
from src.ibkr.watchlist_optimization import (
    WatchlistMove as WatchlistMove,
)
from src.ibkr.watchlist_optimization import (
    WatchlistOptCase as WatchlistOptCase,
)
from src.ibkr.watchlist_optimization import (
    WatchlistOptimization as WatchlistOptimization,
)
from src.ibkr.watchlist_optimization import (
    build_watchlist_optimization_summary as build_watchlist_optimization_summary,
)
from src.ibkr.watchlist_optimization import (
    concentration_breach_summary as concentration_breach_summary,
)
from src.ibkr.watchlist_optimization import (
    is_executable_buy as is_executable_buy,
)
from src.ibkr.watchlist_optimization import (
    resolve_watchlist_optimization as resolve_watchlist_optimization,
)
from src.ibkr.watchlist_optimization import (
    selected_watchlist_buy_ids as selected_watchlist_buy_ids,
)
from src.ibkr.watchlist_optimization import (
    watchlist_candidate_conviction as watchlist_candidate_conviction,
)
from src.ibkr.watchlist_optimization import (
    watchlist_candidate_score as watchlist_candidate_score,
)

_DEFAULT_DIP_WATCH_LIMIT = 7


def cost_basis_unit_mismatch(position: NormalizedPosition | None) -> bool:
    """Return whether local cost/current prices have a likely 100x unit mismatch."""
    if (
        position is None
        or position.avg_cost_local <= 0
        or position.current_price_local <= 0
    ):
        return False
    ratio = position.current_price_local / position.avg_cost_local
    return ratio > 50.0 or ratio < 0.02


_MAX_PLAUSIBLE_FX_EFFECT_PCT = 50.0


def fx_return_split_diagnostic(
    position,
) -> tuple[tuple[float, float, float] | None, str | None]:
    """Decompose a position's return into (local_pct, fx_pct, usd_pct).

    Local-price return is IBKR local cost basis → current local price (it
    includes valuation and sentiment, so it is labeled "local-price", never
    "business"). USD return derives from observed IBKR USD P&L; the implied
    FX/basis effect is the multiplicative residual:
    (1+usd) = (1+local) × (1+fx). Returns None
    for USD positions or when any observed input is missing/degenerate —
    honest absence beats a fabricated split. The decomposition is available
    only when IBKR supplied P&L in USD; P&L converted from local currency at
    today's rate contains no historical entry-FX information.
    """
    if position is None:
        return None, None
    if not getattr(position, "valuation_valid", True):
        return None, (
            getattr(position, "valuation_issue", None)
            or "Position valuation is unavailable pending a data-quality review"
        )
    currency = (position.currency or "USD").upper()
    if currency in ("USD", ""):
        return None, None
    if getattr(position, "unrealized_pnl_basis", "BROKER_USD") != "BROKER_USD":
        return None, None
    numeric_inputs = (
        position.avg_cost_local,
        position.current_price_local,
        position.market_value_usd,
        position.unrealized_pnl_usd,
    )
    if not all(math.isfinite(value) for value in numeric_inputs):
        return None, "FX decomposition withheld: position contains non-finite values"
    if (
        position.avg_cost_local <= 0
        or position.current_price_local <= 0
        or position.market_value_usd == 0
        or cost_basis_unit_mismatch(position)
    ):
        return None, None
    local_ret = (
        position.current_price_local - position.avg_cost_local
    ) / position.avg_cost_local
    usd_cost_basis = position.market_value_usd - position.unrealized_pnl_usd
    if usd_cost_basis <= 0:
        return None, None
    usd_ret = position.unrealized_pnl_usd / usd_cost_basis
    fx_ret = (1.0 + usd_ret) / (1.0 + local_ret) - 1.0
    split = (local_ret * 100.0, fx_ret * 100.0, usd_ret * 100.0)
    if abs(split[1]) > _MAX_PLAUSIBLE_FX_EFFECT_PCT:
        return None, (
            "FX decomposition withheld: broker values imply an implausible "
            f"{split[1]:+.1f}% residual; verify value units or entry FX"
        )
    return split, None


def fx_return_split(position) -> tuple[float, float, float] | None:
    """Backward-compatible split-only view of :func:`fx_return_split_diagnostic`."""
    return fx_return_split_diagnostic(position)[0]


SELL_RECOMMENDATIONS_TITLE = "SELL RECOMMENDATIONS"
SELL_RELATED_REVIEWS_TITLE = "POSITION REVIEWS"
SELL_TYPE_LABELS: dict[str | None, str] = {
    # Retail framing (July 2026): a price break is a review trigger, not an
    # order class — the label must not read as a standing sell instruction.
    "STOP_BREACH": "PRICE-DROP REVIEW",
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
    "STOP_LOSS": "PRICE-DROP REVIEW",
    "CONFIRMED_THESIS_FAILURE": "CONFIRMED THESIS FAILURE",
    "THESIS_REASSESSMENT": "THESIS REASSESSMENT",
    "ENTRY_CONSTRAINT": "ENTRY CONSTRAINT",
    "SPECIAL_SITUATION_REVIEW": "M&A TENDER REVIEW",
    "DATA_QUALITY": "DATA REVIEW",
    "CAPITAL_ALLOCATION": "CAPITAL ALLOCATION REVIEW",
    "OVERWEIGHT": "OVERWEIGHT",
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
    # Names the concentration screen withheld from dip_candidates — computed in
    # the SAME pass as dip_candidates so the policy is evaluated exactly once.
    dip_withheld: tuple[ReconciliationItem, ...] = ()


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
    conditional_proceeds_usd: float
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


_EXECUTABLE_SELL_BASES = frozenset({"MANDATORY_EXIT", "CONFIRMED_THESIS_FAILURE"})


def _sell_identity_verified(item: ReconciliationItem) -> bool:
    position = item.ibkr_position
    return bool(
        position is not None
        and position.conid > 0
        and position.ticker_identity_verified
        and (
            item.action_basis == "MANDATORY_EXIT"
            or (
                item.analysis is not None
                and analysis_identity_verified(position, item.analysis)
            )
        )
    )


def is_executable_sell(item: ReconciliationItem) -> bool:
    """Return whether an item is a decision-safe sale under retail policy."""
    return bool(
        item.action == "SELL"
        and item.action_basis in _EXECUTABLE_SELL_BASES
        and _sell_identity_verified(item)
        and item.suggested_quantity is not None
        and item.suggested_quantity > 0
        and item.cash_impact_usd > 0
    )


def retail_safe_action(item: ReconciliationItem) -> ReconciliationItem:
    """Downgrade legacy sale/trim records that lack current retail authority.

    Saved bundles can predate the confirmation, identity, and tax-aware policy.
    Normalizing at the shared presentation boundary prevents those records from
    becoming executable again in either the CLI or dashboard.
    """
    if item.action == "SELL" and not is_executable_sell(item):
        if item.sell_type == "STOP_BREACH":
            basis = "STOP_LOSS"
            explanation = "price movement is a review trigger, not sale authority"
        elif item.sell_type == "PROFIT_TAKE":
            basis = "CAPITAL_ALLOCATION"
            explanation = "selling an intact winner requires tax-lot review"
        else:
            basis = item.action_basis or "THESIS_REASSESSMENT"
            if (
                item.action_basis in _EXECUTABLE_SELL_BASES
                and not _sell_identity_verified(item)
            ):
                explanation = "security identity or listing mapping is unverified"
            elif item.action_basis in _EXECUTABLE_SELL_BASES:
                explanation = "executable quantity or proceeds are incomplete"
            else:
                explanation = (
                    "sale lacks confirmed thesis-failure or mandatory-exit evidence"
                )
        return item.model_copy(
            update={
                "action": "REVIEW",
                "action_basis": basis,
                "reason": f"{item.reason} — sale downgraded: {explanation}",
                "urgency": "HIGH" if item.sell_type == "STOP_BREACH" else "MEDIUM",
                "suggested_quantity": None,
                "suggested_price": None,
                "cash_impact_usd": 0.0,
                "settlement_date": None,
            }
        )
    if item.action == "TRIM":
        return item.model_copy(
            update={
                "action": "REVIEW",
                "action_basis": "OVERWEIGHT",
                "reason": (
                    f"{item.reason} — legacy trim downgraded: verify tax lots and "
                    "after-friction benefit before changing an intact position"
                ),
                "urgency": "LOW",
                "suggested_quantity": None,
                "suggested_price": None,
                "cash_impact_usd": 0.0,
                "settlement_date": None,
            }
        )
    return item


def group_portfolio_actions(
    items: list[ReconciliationItem],
    *,
    watchlist_tickers: set[str] | None = None,
    dip_watch_limit: int = _DEFAULT_DIP_WATCH_LIMIT,
    macro_event_active: bool = False,
    exchange_weights: dict[str, float] | None = None,
    sector_weights: dict[str, float] | None = None,
    exchange_limit_pct: float = DEFAULT_EXCHANGE_LIMIT_PCT,
    sector_limit_pct: float = DEFAULT_SECTOR_LIMIT_PCT,
) -> PortfolioActionGroups:
    items = [retail_safe_action(item) for item in items]

    def is_macro_review(item: ReconciliationItem) -> bool:
        return "[MACRO_" in item.reason

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
        if item.action == "REVIEW"
        and item.sell_type == "SOFT_REJECT"
        and is_macro_review(item)
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
        and item.sell_type not in ("STOP_BREACH", "PROFIT_TAKE")
        and not (item.sell_type == "SOFT_REJECT" and is_macro_review(item))
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
    dip_selection = select_dip_watch(
        collect_dip_watch_source_items(items),
        macro_event_active=macro_event_active,
        limit=dip_watch_limit,
        exchange_weights=exchange_weights,
        sector_weights=sector_weights,
        exchange_limit_pct=exchange_limit_pct,
        sector_limit_pct=sector_limit_pct,
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
        dip_candidates=dip_selection.kept,
        dip_withheld=dip_selection.withheld,
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
    """Build pending inflows from confirmed fundamental/mandatory exits only.

    Price reviews, profit-taking reviews, legacy trims, and soft rejections do
    not fund new purchases. They are tax- and friction-sensitive operator
    choices, not executable cash.
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
        if is_executable_sell(item) and item.settlement_date
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
    *,
    watchlist_optimization: WatchlistOptimization | None = None,
    executable_buy_ids: Set[str] | None = None,
) -> CashSummaryView:
    settled_cash = portfolio.settled_cash_usd
    available_cash = portfolio.available_cash_usd
    total_cash = portfolio.cash_balance_usd
    buffer_reserve = max(settled_cash - available_cash, 0.0)
    unsettled_cash = max(total_cash - settled_cash, 0.0)
    if watchlist_optimization is not None and executable_buy_ids is not None:
        raise ValueError("pass watchlist_optimization or executable_buy_ids, not both")
    if executable_buy_ids is not None:
        recommended_buy_cost = sum(
            abs(item.cash_impact_usd)
            for item in items
            if is_executable_buy(item, executable_buy_ids) and item.cash_impact_usd < 0
        )
    elif watchlist_optimization is not None:
        # Optimizer-aware: only buys that survived merit+concentration
        # selection reserve cash (screened/displaced watchlist BUYs still
        # carry a negative cash_impact_usd and must not count).
        selected_ids = selected_watchlist_buy_ids(watchlist_optimization)
        recommended_buy_cost = sum(
            abs(item.cash_impact_usd)
            for item in items
            if is_executable_buy(item, selected_ids) and item.cash_impact_usd < 0
        )
    else:
        # Legacy: every sized watchlist BUY. Kept for callers without an
        # optimization in hand; can overstate when the optimizer would
        # screen or displace names.
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
