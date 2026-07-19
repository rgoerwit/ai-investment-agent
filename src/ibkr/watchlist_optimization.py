"""Watchlist ranking, concentration screening, and executable-buy policy."""

from __future__ import annotations

from collections.abc import Iterable, Set
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from src.ibkr.concentration import (
    ConcentrationBreach,
    canonical_exchange_bucket,
    canonical_sector_bucket,
    project_concentration_breaches,
)
from src.ibkr.models import ReconciliationItem
from src.ibkr.portfolio_defaults import (
    DEFAULT_EXCHANGE_LIMIT_PCT,
    DEFAULT_SECTOR_LIMIT_PCT,
)
from src.ibkr.ticker import Ticker

if TYPE_CHECKING:
    from src.ibkr.portfolio_presentation import PortfolioActionGroups

TARGET_WATCHLIST_SIZE = 6
WATCHLIST_MIN_CONVICTION = "medium"
WATCHLIST_ADDITION_MIN_CONVICTION = "high"
_WATCHLIST_CONVICTION_RANK = {"high": 0, "medium": 1, "low": 2}

# Existing watchlist members get a narrow anti-flap exception. New additions
# must fit current concentration and never use this exception.
CONCENTRATION_INCUMBENT_MIN_SCORE = 135.0


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
class ConcentrationNote:
    """A candidate's concentration decision and every breached dimension."""

    item: ReconciliationItem
    breaches: tuple[ConcentrationBreach, ...]


@dataclass(frozen=True)
class WatchlistMove:
    item: ReconciliationItem
    reason: str
    note: ConcentrationNote | None = None


@dataclass(frozen=True)
class WatchlistOptimization:
    """Pure watchlist recommendation separated from report rendering."""

    case: WatchlistOptCase
    watchlist_supplied: bool
    target_size: int
    current_size: int | None
    available_addition_slots: int
    optimal: tuple[ReconciliationItem, ...]
    keep: tuple[ReconciliationItem, ...]
    add: tuple[ReconciliationItem, ...]
    remove: tuple[WatchlistMove, ...]
    monitors: tuple[ReconciliationItem, ...]
    reviews: tuple[ReconciliationItem, ...]
    protected_tickers: tuple[str, ...]
    excluded_low_conviction: tuple[ReconciliationItem, ...]
    pool_size: int
    withheld_candidates: tuple[ConcentrationNote, ...] = ()
    capacity_limited_candidates: tuple[ReconciliationItem, ...] = ()
    admitted_over_limit: tuple[ConcentrationNote, ...] = ()
    retained_for_watchlist_floor: tuple[WatchlistMove, ...] = ()


def _watchlist_ticker_identity(ticker: str) -> str:
    """Return the exchange-qualified identity already managed by Ticker."""
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


def _candidate_size_pct(item: ReconciliationItem) -> float:
    analysis = item.analysis
    if analysis is None:
        return 0.0
    return float(analysis.trade_block.size_pct or analysis.position_size or 0.0)


def _candidate_exchange(item: ReconciliationItem) -> str:
    analysis = item.analysis
    return canonical_exchange_bucket(
        item.ticker,
        analysis_exchange=analysis.exchange if analysis else None,
        position=item.ibkr_position,
    )


def _candidate_sector(item: ReconciliationItem) -> str | None:
    analysis = item.analysis
    return canonical_sector_bucket(analysis.sector if analysis else None)


def weakest_bucket_incumbents(
    held_items: Iterable[ReconciliationItem],
    *,
    dimension: str,
    key: str,
    limit: int = 3,
) -> list[ReconciliationItem]:
    """Return the lowest-scoring held positions in an over-limit bucket.

    Informational only: when a withheld buy names an overweight bucket, the
    operator may compare it with current holdings. This is not a claim that the
    lowest score is the weakest investment: stale evidence, valuation, thesis
    quality, and tax lots still require review. It never pairs a sale with a
    buy or emits an order.
    """
    in_bucket: list[ReconciliationItem] = []
    seen: set[str] = set()
    for item in held_items:
        if item.ibkr_position is None:
            continue
        identity = _item_ticker_identity(item)
        if identity in seen:
            continue
        bucket = (
            _candidate_exchange(item)
            if dimension == "exchange"
            else _candidate_sector(item)
        )
        if bucket != key:
            continue
        seen.add(identity)
        in_bucket.append(item)
    in_bucket.sort(
        key=lambda item: (
            item.analysis is None,
            watchlist_candidate_score(item),
            item.ticker.yf,
        )
    )
    return in_bucket[:limit]


def _select_with_concentration_headroom(
    ranked: list[ReconciliationItem],
    *,
    target_size: int,
    exchange_weights: dict[str, float] | None,
    sector_weights: dict[str, float] | None,
    exchange_limit_pct: float,
    sector_limit_pct: float,
    allow_over_limit: bool = True,
    accumulate_selected: bool = True,
) -> tuple[list[ReconciliationItem], list[ConcentrationNote], list[ConcentrationNote]]:
    if (not exchange_weights and not sector_weights) or target_size <= 0:
        return ranked[:target_size], [], []
    running_exchange = dict(exchange_weights or {})
    running_sector = dict(sector_weights or {})
    selected: list[ReconciliationItem] = []
    admitted: list[ConcentrationNote] = []
    withheld: list[ConcentrationNote] = []
    for item in ranked:
        if len(selected) >= target_size:
            break
        size_pct = _candidate_size_pct(item)
        exchange_key = _candidate_exchange(item)
        sector_key = _candidate_sector(item)
        breaches = project_concentration_breaches(
            exchange_key=exchange_key,
            sector_key=sector_key,
            candidate_pct=size_pct,
            exchange_weights=running_exchange if exchange_weights else {},
            sector_weights=running_sector if sector_weights else {},
            exchange_limit_pct=exchange_limit_pct,
            sector_limit_pct=sector_limit_pct,
        )
        if breaches:
            note = ConcentrationNote(item=item, breaches=breaches)
            if (
                allow_over_limit
                and watchlist_candidate_conviction(item) == "high"
                and watchlist_candidate_score(item) >= CONCENTRATION_INCUMBENT_MIN_SCORE
            ):
                admitted.append(note)
            else:
                withheld.append(note)
                continue
        selected.append(item)
        if accumulate_selected and exchange_weights:
            running_exchange[exchange_key] = (
                running_exchange.get(exchange_key, 0.0) + size_pct
            )
        if accumulate_selected and sector_weights and sector_key is not None:
            running_sector[sector_key] = running_sector.get(sector_key, 0.0) + size_pct
    return selected, admitted, withheld


def resolve_watchlist_optimization(
    items: list[ReconciliationItem],
    groups: PortfolioActionGroups,
    *,
    watchlist_tickers: set[str] | None,
    watchlist_supplied: bool,
    watchlist_unavailable: bool,
    target_size: int = TARGET_WATCHLIST_SIZE,
    min_conviction: str = WATCHLIST_MIN_CONVICTION,
    addition_min_conviction: str = WATCHLIST_ADDITION_MIN_CONVICTION,
    exchange_weights: dict[str, float] | None = None,
    sector_weights: dict[str, float] | None = None,
    exchange_limit_pct: float = DEFAULT_EXCHANGE_LIMIT_PCT,
    sector_limit_pct: float = DEFAULT_SECTOR_LIMIT_PCT,
) -> WatchlistOptimization:
    """Select up to target_size medium-or-higher unheld BUYs safely."""
    if target_size < 0:
        raise ValueError("target_size must be non-negative")
    min_rank = _WATCHLIST_CONVICTION_RANK.get(min_conviction.lower())
    if min_rank is None:
        raise ValueError(f"unsupported watchlist conviction: {min_conviction}")
    addition_min_rank = _WATCHLIST_CONVICTION_RANK.get(addition_min_conviction.lower())
    if addition_min_rank is None:
        raise ValueError(
            f"unsupported watchlist addition conviction: {addition_min_conviction}"
        )

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

    pooled_by_identity: dict[str, ReconciliationItem] = {}
    pool_candidates = sorted(
        (item for item in items if item.action == "BUY" and item.ibkr_position is None),
        key=lambda item: (not item.is_watchlist, _item_ticker_identity(item)),
    )
    for item in pool_candidates:
        pooled_by_identity.setdefault(_item_ticker_identity(item), item)

    pool = tuple(pooled_by_identity.values())

    def is_incumbent(item: ReconciliationItem) -> bool:
        return item.is_watchlist or _item_ticker_identity(item) in raw_watchlist

    eligible_incumbents: list[ReconciliationItem] = []
    eligible_additions: list[ReconciliationItem] = []
    excluded_low_conviction: list[ReconciliationItem] = []
    for item in pool:
        conviction_rank = _WATCHLIST_CONVICTION_RANK.get(
            watchlist_candidate_conviction(item), len(_WATCHLIST_CONVICTION_RANK)
        )
        required_rank = min_rank if is_incumbent(item) else addition_min_rank
        if conviction_rank > required_rank:
            excluded_low_conviction.append(item)
        elif is_incumbent(item):
            eligible_incumbents.append(item)
        else:
            eligible_additions.append(item)

    def rank_candidates(
        candidates: list[ReconciliationItem],
    ) -> list[ReconciliationItem]:
        return sorted(
            candidates,
            key=lambda item: (
                _WATCHLIST_CONVICTION_RANK[watchlist_candidate_conviction(item)],
                -watchlist_candidate_score(item),
                _item_ticker_identity(item),
            ),
        )

    ranked_incumbents = rank_candidates(eligible_incumbents)
    ranked_additions = rank_candidates(eligible_additions)

    # Existing watchlist BUYs are maintenance decisions. Evaluate each against
    # the portfolio as it exists now; do not pretend every watchlist name has
    # already been bought when assessing the next one.
    selected_incumbents, admitted_incumbents, withheld_incumbents = (
        _select_with_concentration_headroom(
            ranked_incumbents,
            target_size=target_size,
            exchange_weights=exchange_weights,
            sector_weights=sector_weights,
            exchange_limit_pct=exchange_limit_pct,
            sector_limit_pct=sector_limit_pct,
            accumulate_selected=False,
        )
    )

    # New additions are deliberately stricter: they must be high-conviction and
    # fit current portfolio concentration without assuming trims, sales, or any
    # other recommended candidate has already been purchased.
    if target_size > 0:
        qualified_additions, _, withheld_additions = (
            _select_with_concentration_headroom(
                ranked_additions,
                target_size=len(ranked_additions),
                exchange_weights=exchange_weights,
                sector_weights=sector_weights,
                exchange_limit_pct=exchange_limit_pct,
                sector_limit_pct=sector_limit_pct,
                allow_over_limit=False,
                accumulate_selected=False,
            )
        )
    else:
        qualified_additions, withheld_additions = [], []
    current_watchlist_ids = raw_watchlist | watched_item_ids
    available_slots = (
        target_size
        if not watchlist_supplied or watchlist_unavailable
        else max(target_size - len(current_watchlist_ids), 0)
    )
    current_size = (
        None
        if not watchlist_supplied or watchlist_unavailable
        else len(current_watchlist_ids)
    )
    selected_additions = qualified_additions[:available_slots]
    capacity_limited_candidates = tuple(qualified_additions[available_slots:])

    selected = [*selected_incumbents, *selected_additions]
    admitted = admitted_incumbents
    withheld = [*withheld_incumbents, *withheld_additions]
    optimal = tuple(selected)
    admitted_over_limit = tuple(admitted)
    optimal_ids = {_item_ticker_identity(item) for item in optimal}

    if watchlist_unavailable:
        return WatchlistOptimization(
            case=WatchlistOptCase.WATCHLIST_UNAVAILABLE,
            watchlist_supplied=False,
            target_size=target_size,
            current_size=None,
            available_addition_slots=available_slots,
            optimal=optimal,
            keep=(),
            add=optimal,
            remove=(),
            monitors=(),
            reviews=(),
            protected_tickers=(),
            excluded_low_conviction=tuple(excluded_low_conviction),
            pool_size=len(pool),
            withheld_candidates=tuple(withheld),
            capacity_limited_candidates=capacity_limited_candidates,
            admitted_over_limit=admitted_over_limit,
        )
    if not watchlist_supplied:
        return WatchlistOptimization(
            case=WatchlistOptCase.NO_WATCHLIST,
            watchlist_supplied=False,
            target_size=target_size,
            current_size=None,
            available_addition_slots=available_slots,
            optimal=optimal,
            keep=(),
            add=optimal,
            remove=(),
            monitors=(),
            reviews=(),
            protected_tickers=(),
            excluded_low_conviction=tuple(excluded_low_conviction),
            pool_size=len(pool),
            withheld_candidates=tuple(withheld),
            capacity_limited_candidates=capacity_limited_candidates,
            admitted_over_limit=admitted_over_limit,
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
    add = tuple(selected_additions)
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
    concentration_removals = [note for note in withheld if is_incumbent(note.item)]
    withheld_candidates = tuple(
        note for note in withheld if not is_incumbent(note.item)
    )
    concentration_removal_ids = {
        _item_ticker_identity(note.item) for note in concentration_removals
    }
    removals = [
        WatchlistMove(note.item, "concentration_displaced", note=note)
        for note in concentration_removals
    ]
    for item in watchlist_items:
        identity = _item_ticker_identity(item)
        if identity in optimal_ids or identity in monitor_ids or identity in review_ids:
            continue
        if identity in concentration_removal_ids:
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

    # IBKR does not permit an empty watchlist. When every current entry would
    # be removed, retain the strongest one as a non-executable operational floor.
    retained_for_watchlist_floor: tuple[WatchlistMove, ...] = ()
    final_entries = (*optimal, *monitors, *reviews, *protected_tickers)
    if (
        watchlist_supplied
        and not final_entries
        and (raw_watchlist or watchlist_items)
        and removals
    ):
        floor_move = min(
            removals,
            key=lambda move: (
                move.reason == "verdict_reject",
                _WATCHLIST_CONVICTION_RANK.get(
                    watchlist_candidate_conviction(move.item),
                    len(_WATCHLIST_CONVICTION_RANK),
                ),
                -watchlist_candidate_score(move.item),
                _item_ticker_identity(move.item),
            ),
        )
        retained_for_watchlist_floor = (floor_move,)
        removals = [move for move in removals if move is not floor_move]

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
        current_size=current_size,
        available_addition_slots=available_slots,
        optimal=optimal,
        keep=keep,
        add=add,
        remove=tuple(removals),
        monitors=monitors,
        reviews=reviews,
        protected_tickers=protected_tickers,
        excluded_low_conviction=tuple(excluded_low_conviction),
        pool_size=len(pool),
        withheld_candidates=withheld_candidates,
        capacity_limited_candidates=capacity_limited_candidates,
        admitted_over_limit=admitted_over_limit,
        retained_for_watchlist_floor=retained_for_watchlist_floor,
    )


def selected_watchlist_buy_ids(optimization: WatchlistOptimization) -> set[str]:
    return {
        _item_ticker_identity(item)
        for item in optimization.optimal
        if item.is_watchlist
    }


def is_executable_buy(item: ReconciliationItem, selected_ids: Set[str]) -> bool:
    if item.action == "ADD":
        return True
    return (
        item.action == "BUY"
        and item.is_watchlist
        and _item_ticker_identity(item) in selected_ids
    )


def concentration_breach_summary(note: ConcentrationNote) -> str:
    return " + ".join(
        f"overweight {breach.dimension} {breach.key} "
        f"(projected {breach.projected_pct:.1f}% > {breach.limit_pct:.0f}%)"
        for breach in note.breaches
    )


def build_watchlist_optimization_summary(
    optimization: WatchlistOptimization,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, values in (
        ("WATCHLIST_KEEP", optimization.keep),
        ("WATCHLIST_ADD", optimization.add),
        ("WATCHLIST_REMOVE", optimization.remove),
        ("WATCHLIST_MONITOR", optimization.monitors),
        ("WATCHLIST_REVIEW", optimization.reviews),
        ("WATCHLIST_CAPACITY_LIMITED", optimization.capacity_limited_candidates),
    ):
        if values:
            counts[key] = len(values)
    return counts
