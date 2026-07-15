from __future__ import annotations

import pytest

from src.ibkr.models import ReconciliationItem
from src.ibkr.portfolio_presentation import (
    WatchlistOptCase,
    group_portfolio_actions,
    resolve_watchlist_optimization,
)
from tests.ibkr.reconciler_cases import _make_analysis, _make_position


def _buy(
    ticker: str,
    *,
    conviction: str = "High",
    score: float = 135.0,
    is_watchlist: bool = False,
    cash_blocked: bool = False,
) -> ReconciliationItem:
    analysis = _make_analysis(ticker=ticker, conviction=conviction)
    analysis.health_adj = score / 2
    analysis.growth_adj = score / 2
    return ReconciliationItem(
        ticker=ticker,
        action="BUY",
        reason="BUY",
        urgency="MEDIUM",
        analysis=analysis,
        is_watchlist=is_watchlist,
        suggested_quantity=None if cash_blocked else 10,
        suggested_price=100.0,
        cash_impact_usd=0.0 if cash_blocked else -1000.0,
        is_cash_blocked=cash_blocked,
    )


def _resolve(
    items: list[ReconciliationItem],
    watchlist_tickers: set[str] | None,
    *,
    supplied: bool = True,
    target_size: int = 6,
    unavailable: bool = False,
    min_conviction: str = "medium",
):
    groups = group_portfolio_actions(items, watchlist_tickers=watchlist_tickers)
    return resolve_watchlist_optimization(
        items,
        groups,
        watchlist_tickers=watchlist_tickers,
        watchlist_supplied=supplied,
        watchlist_unavailable=unavailable,
        target_size=target_size,
        min_conviction=min_conviction,
    )


def test_exact_exchange_qualified_tickers_are_not_deduplicated_by_base():
    on_watchlist = _buy("BHP.AX", score=140, is_watchlist=True)
    off_watchlist = _buy("BHP.L", score=130)

    optimization = _resolve([on_watchlist, off_watchlist], {"BHP.AX"}, target_size=2)

    assert [item.ticker.yf for item in optimization.optimal] == ["BHP.AX", "BHP.L"]
    assert optimization.keep == (on_watchlist,)
    assert optimization.add == (off_watchlist,)


def test_unresolved_bare_watchlist_symbol_is_protected_not_suffix_matched():
    candidate = _buy("5434.TW")

    optimization = _resolve([candidate], {"5434"}, target_size=1)

    assert optimization.add == (candidate,)
    assert optimization.protected_tickers == ("5434",)
    assert optimization.remove == ()


def test_held_watchlist_member_is_protected_from_optimization_removal():
    held = ReconciliationItem(
        ticker="AAPL",
        action="HOLD",
        reason="Held position",
        urgency="LOW",
        ibkr_position=_make_position(ticker="AAPL"),
    )
    candidate = _buy("7203.T")

    optimization = _resolve([held, candidate], {"AAPL"}, target_size=1)

    assert optimization.protected_tickers == ("AAPL",)
    assert optimization.add == (candidate,)
    assert optimization.remove == ()


def test_reject_removes_but_monitors_and_reviews_remain_protected():
    rejected = ReconciliationItem(
        ticker="7203.T",
        action="REMOVE",
        reason="Verdict rejected",
        urgency="MEDIUM",
        analysis=_make_analysis(ticker="7203.T", verdict="DO_NOT_INITIATE"),
        is_watchlist=True,
    )
    monitor = ReconciliationItem(
        ticker="6758.T",
        action="HOLD",
        reason="Monitor",
        urgency="LOW",
        analysis=_make_analysis(ticker="6758.T", verdict="HOLD"),
        is_watchlist=True,
    )
    review = ReconciliationItem(
        ticker="9432.T",
        action="REVIEW",
        reason="Stale",
        urgency="MEDIUM",
        is_watchlist=True,
    )

    optimization = _resolve(
        [rejected, monitor, review],
        {"7203.T", "6758.T", "9432.T"},
    )

    assert optimization.case is WatchlistOptCase.EMPTY_POOL
    assert [(move.item, move.reason) for move in optimization.remove] == [
        (rejected, "verdict_reject")
    ]
    assert optimization.monitors == (monitor,)
    assert optimization.reviews == (review,)


def test_cash_blocked_medium_or_high_candidate_remains_rankable_and_non_executable():
    candidate = _buy("WDO.TO", cash_blocked=True)

    optimization = _resolve([candidate], set(), target_size=1)

    assert optimization.case is WatchlistOptCase.FULL_OPTIMIZE
    assert optimization.add == (candidate,)
    assert candidate.is_cash_blocked is True
    assert candidate.suggested_quantity is None
    assert candidate.cash_impact_usd == 0.0


def test_exactly_full_watchlist_without_changes_is_aligned():
    items = [_buy(f"{ticker}.T", is_watchlist=True) for ticker in range(1000, 1006)]

    optimization = _resolve(items, {item.ticker.yf for item in items})

    assert optimization.case is WatchlistOptCase.ALIGNED
    assert len(optimization.keep) == 6
    assert optimization.add == ()
    assert optimization.remove == ()


def test_case_1_no_watchlist_supplied_only_suggests_additions_and_cannot_merge():
    candidate = _buy("7203.T")

    optimization = _resolve([candidate], None, supplied=False)

    assert optimization.case is WatchlistOptCase.NO_WATCHLIST
    assert optimization.add == (candidate,)
    assert optimization.keep == ()
    assert optimization.remove == ()
    assert optimization.monitors == ()
    assert optimization.reviews == ()


def test_no_watchlist_supplied_and_no_candidates_returns_no_actions():
    optimization = _resolve([], None, supplied=False)

    assert optimization.case is WatchlistOptCase.NO_WATCHLIST
    assert optimization.optimal == ()
    assert optimization.add == ()
    assert optimization.keep == ()
    assert optimization.remove == ()


def test_case_2_supplied_empty_watchlist_with_no_worthy_tickers_has_nothing_to_recommend():
    low = _buy("7203.T", conviction="Low")

    optimization = _resolve([low], set())

    assert optimization.case is WatchlistOptCase.NOTHING_ACTIONABLE
    assert optimization.optimal == ()
    assert optimization.add == ()
    assert optimization.remove == ()
    assert optimization.excluded_low_conviction == (low,)


def test_supplied_watchlist_with_only_low_conviction_entries_is_empty_pool():
    low = _buy("7203.T", conviction="Low", is_watchlist=True)

    optimization = _resolve([low], {"7203.T"})

    assert optimization.case is WatchlistOptCase.EMPTY_POOL
    assert optimization.optimal == ()
    assert [(move.item, move.reason) for move in optimization.remove] == [
        (low, "below_medium_conviction")
    ]


def test_case_3_partial_fill_keeps_all_available_medium_or_higher_candidates():
    candidates = [
        _buy("7203.T", conviction="High"),
        _buy("6758.T", conviction="Medium"),
    ]

    optimization = _resolve(candidates, set())

    assert optimization.case is WatchlistOptCase.PARTIAL_FILL
    assert optimization.optimal == tuple(candidates)
    assert optimization.add == tuple(candidates)
    assert optimization.keep == ()
    assert optimization.remove == ()


def test_case_4_full_pool_keeps_best_current_names_and_swaps_weaker_ones():
    keep_high = _buy("7203.T", score=200, is_watchlist=True)
    keep_high_2 = _buy("6758.T", score=180, is_watchlist=True)
    displaced_medium = _buy("9432.T", conviction="Medium", score=50, is_watchlist=True)
    displaced_medium_2 = _buy(
        "9984.T", conviction="Medium", score=30, is_watchlist=True
    )
    additions = [
        _buy("8306.T", score=190),
        _buy("8058.T", score=170),
        _buy("7201.T", conviction="Medium", score=200),
        _buy("4063.T", conviction="Medium", score=190),
    ]
    items = [keep_high, keep_high_2, displaced_medium, displaced_medium_2, *additions]

    optimization = _resolve(
        items,
        {"7203.T", "6758.T", "9432.T", "9984.T"},
    )

    assert optimization.case is WatchlistOptCase.FULL_OPTIMIZE
    assert optimization.keep == (keep_high, keep_high_2)
    assert optimization.add == tuple(additions)
    assert [(move.item, move.reason) for move in optimization.remove] == [
        (displaced_medium, "displaced_by_higher_conviction"),
        (displaced_medium_2, "displaced_by_higher_conviction"),
    ]


def test_no_current_equity_is_kept_when_all_worthy_names_are_new_candidates():
    low_current = _buy("7203.T", conviction="Low", is_watchlist=True)
    candidate = _buy("6758.T")

    optimization = _resolve([low_current, candidate], {"7203.T"}, target_size=1)

    assert optimization.keep == ()
    assert optimization.add == (candidate,)
    assert [(move.item, move.reason) for move in optimization.remove] == [
        (low_current, "below_medium_conviction")
    ]


def test_full_empty_watchlist_is_populated_entirely_from_new_candidates():
    candidates = [_buy(f"{ticker}.T") for ticker in range(1000, 1006)]

    optimization = _resolve(candidates, set())

    assert optimization.case is WatchlistOptCase.FULL_OPTIMIZE
    assert optimization.keep == ()
    assert optimization.add == tuple(candidates)
    assert optimization.remove == ()


def test_supplied_watchlist_with_only_monitor_and_review_has_no_removal_recommendation():
    monitor = ReconciliationItem(
        ticker="7203.T",
        action="HOLD",
        reason="Monitor",
        urgency="LOW",
        analysis=_make_analysis(ticker="7203.T", verdict="HOLD"),
        is_watchlist=True,
    )
    review = ReconciliationItem(
        ticker="6758.T",
        action="REVIEW",
        reason="Stale",
        urgency="MEDIUM",
        is_watchlist=True,
    )

    optimization = _resolve([monitor, review], {"7203.T", "6758.T"})

    assert optimization.case is WatchlistOptCase.EMPTY_POOL
    assert optimization.keep == ()
    assert optimization.add == ()
    assert optimization.remove == ()
    assert optimization.monitors == (monitor,)
    assert optimization.reviews == (review,)


def test_unavailable_watchlist_allows_additions_but_never_keep_or_remove_authority():
    candidate = _buy("7203.T")

    optimization = _resolve([candidate], {"7203.T"}, unavailable=True, target_size=1)

    assert optimization.case is WatchlistOptCase.WATCHLIST_UNAVAILABLE
    assert optimization.add == (candidate,)
    assert optimization.keep == ()
    assert optimization.remove == ()
    assert optimization.protected_tickers == ()


def test_exact_duplicate_prefers_the_current_watchlist_item():
    current = _buy("7203.T", is_watchlist=True)
    duplicate_candidate = _buy("7203.T", score=200)

    optimization = _resolve([duplicate_candidate, current], {"7203.T"}, target_size=1)

    assert optimization.optimal == (current,)
    assert optimization.keep == (current,)
    assert optimization.add == ()


def test_equal_rank_tie_breaks_by_exchange_qualified_ticker_deterministically():
    later = _buy("BBB.T")
    earlier = _buy("AAA.T")

    optimization = _resolve([later, earlier], set(), target_size=1)

    assert optimization.optimal == (earlier,)


def test_zero_target_only_removes_buy_ready_rows_and_preserves_monitor_and_review():
    buy = _buy("7203.T", is_watchlist=True)
    monitor = ReconciliationItem(
        ticker="6758.T",
        action="HOLD",
        reason="Monitor",
        urgency="LOW",
        analysis=_make_analysis(ticker="6758.T", verdict="HOLD"),
        is_watchlist=True,
    )
    review = ReconciliationItem(
        ticker="9432.T",
        action="REVIEW",
        reason="Stale",
        urgency="MEDIUM",
        is_watchlist=True,
    )

    optimization = _resolve(
        [buy, monitor, review],
        {"7203.T", "6758.T", "9432.T"},
        target_size=0,
    )

    assert optimization.optimal == ()
    assert [(move.item, move.reason) for move in optimization.remove] == [
        (buy, "displaced_by_higher_conviction")
    ]
    assert optimization.monitors == (monitor,)
    assert optimization.reviews == (review,)


@pytest.mark.parametrize(
    ("target_size", "min_conviction", "message"),
    [(-1, "medium", "target_size"), (1, "certain", "unsupported watchlist conviction")],
)
def test_invalid_optimizer_configuration_fails_closed(
    target_size: int,
    min_conviction: str,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        _resolve(
            [_buy("7203.T")],
            set(),
            target_size=target_size,
            min_conviction=min_conviction,
        )
