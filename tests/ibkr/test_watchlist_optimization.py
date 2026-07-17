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
    size_pct: float = 4.0,
    sector: str = "",
) -> ReconciliationItem:
    analysis = _make_analysis(ticker=ticker, conviction=conviction, size_pct=size_pct)
    analysis.health_adj = score / 2
    analysis.growth_adj = score / 2
    if sector:
        analysis.sector = sector
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
    exchange_weights: dict[str, float] | None = None,
    sector_weights: dict[str, float] | None = None,
    exchange_limit_pct: float = 40.0,
    sector_limit_pct: float = 30.0,
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
        exchange_weights=exchange_weights,
        sector_weights=sector_weights,
        exchange_limit_pct=exchange_limit_pct,
        sector_limit_pct=sector_limit_pct,
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


# ── Concentration screen (symmetric, projected-weight, two-tier hatch) ───────


def test_projected_weight_trigger_accepts_under_and_withholds_over():
    """35% + 4% ≤ 40 accepted; 38% + 4% > 40 withheld (projected basis)."""
    candidate = _buy("7203.T", conviction="Medium", size_pct=4.0)

    under = _resolve([candidate], set(), exchange_weights={"T": 35.0})
    over = _resolve([candidate], set(), exchange_weights={"T": 38.0})

    assert under.optimal == (candidate,)
    assert under.withheld_candidates == ()
    assert over.optimal == ()
    assert [n.item for n in over.withheld_candidates] == [candidate]
    breach = over.withheld_candidates[0].breaches[0]
    assert (breach.dimension, breach.key) == ("exchange", "T")
    assert breach.projected_pct == pytest.approx(42.0)
    assert breach.limit_pct == 40.0


def test_projected_exactly_at_limit_passes():
    candidate = _buy("7203.T", conviction="Medium", size_pct=4.0)
    optimization = _resolve([candidate], set(), exchange_weights={"T": 36.0})
    assert optimization.optimal == (candidate,)
    assert optimization.withheld_candidates == ()


def test_withheld_slot_refills_with_next_ranked_under_limit_name():
    over_limit = _buy("7203.T", conviction="Medium", score=190)
    refill_hk = _buy("0005.HK", conviction="Medium", score=120)
    refill_us = _buy("ACME", conviction="Medium", score=110)

    optimization = _resolve(
        [over_limit, refill_hk, refill_us],
        set(),
        target_size=2,
        exchange_weights={"T": 45.0},
    )

    assert [item.ticker.yf for item in optimization.optimal] == ["0005.HK", "ACME"]
    assert [n.item.ticker.yf for n in optimization.withheld_candidates] == ["7203.T"]


def test_newcomer_escape_hatch_boundaries():
    """Newcomers need HIGH conviction AND score ≥ 150 to claim an over-limit slot."""
    admitted = _buy("7203.T", conviction="High", score=150.0)
    below_bar = _buy("6758.T", conviction="High", score=149.9)
    high_score_medium = _buy("6971.T", conviction="Medium", score=180.0)

    optimization = _resolve(
        [admitted, below_bar, high_score_medium],
        set(),
        exchange_weights={"T": 45.0},
    )

    assert [item.ticker.yf for item in optimization.optimal] == ["7203.T"]
    assert [n.item.ticker.yf for n in optimization.admitted_over_limit] == ["7203.T"]
    assert sorted(n.item.ticker.yf for n in optimization.withheld_candidates) == [
        "6758.T",
        "6971.T",
    ]


def test_incumbent_escape_hatch_lower_bar_and_removal_below_it():
    """Incumbents survive at ≥135; below that they are moved OUT (removed)."""
    survives = _buy("7203.T", conviction="High", score=135.0, is_watchlist=True)
    removed = _buy("6758.T", conviction="High", score=134.9, is_watchlist=True)

    optimization = _resolve(
        [survives, removed],
        {"7203.T", "6758.T"},
        exchange_weights={"T": 45.0},
    )

    assert optimization.optimal == (survives,)
    assert optimization.keep == (survives,)
    assert [n.item.ticker.yf for n in optimization.admitted_over_limit] == ["7203.T"]
    assert [(m.item.ticker.yf, m.reason) for m in optimization.remove] == [
        ("6758.T", "concentration_displaced")
    ]
    assert optimization.remove[0].note is not None
    assert optimization.remove[0].note.breaches[0].key == "T"
    assert optimization.withheld_candidates == ()


def test_hysteresis_band_incumbent_survives_where_identical_newcomer_fails():
    """Same conviction+score: the incumbent (bar 135) stays, the newcomer (bar 150) is withheld."""
    incumbent = _buy("7203.T", conviction="High", score=140.0, is_watchlist=True)
    newcomer = _buy("6758.T", conviction="High", score=140.0)

    optimization = _resolve(
        [incumbent, newcomer],
        {"7203.T"},
        exchange_weights={"T": 45.0},
    )

    assert optimization.optimal == (incumbent,)
    assert [n.item.ticker.yf for n in optimization.withheld_candidates] == ["6758.T"]
    assert optimization.remove == ()


def test_hatch_admit_consumes_headroom_for_next_same_bucket_name():
    """The second name is judged against the post-admit base (42+4, not 38+4)."""
    star = _buy("7203.T", conviction="High", score=168.0, size_pct=4.0)
    follower = _buy("6758.T", conviction="Medium", score=160.0, size_pct=4.0)

    optimization = _resolve([star, follower], set(), exchange_weights={"T": 38.0})

    assert [n.item.ticker.yf for n in optimization.admitted_over_limit] == ["7203.T"]
    follower_note = optimization.withheld_candidates[0]
    assert follower_note.item.ticker.yf == "6758.T"
    assert follower_note.breaches[0].projected_pct == pytest.approx(46.0)


def test_normal_accept_consumes_headroom_and_flips_next_to_withheld():
    """25 + 4 = 29 accepted; the next same-bucket 29 + 4 = 33 > 30 withheld."""
    first = _buy("7203.T", conviction="Medium", score=190, sector="Industrials")
    second = _buy("6758.T", conviction="Medium", score=120, sector="Industrials")

    optimization = _resolve(
        [first, second], set(), sector_weights={"Industrials": 25.0}
    )

    assert [item.ticker.yf for item in optimization.optimal] == ["7203.T"]
    assert [n.item.ticker.yf for n in optimization.withheld_candidates] == ["6758.T"]
    assert optimization.withheld_candidates[0].breaches[0].dimension == "sector"


def test_sector_only_breach_withholds():
    candidate = _buy("7203.T", conviction="Medium", sector="Industrials", size_pct=4.0)

    optimization = _resolve(
        [candidate],
        set(),
        exchange_weights={"T": 10.0},
        sector_weights={"Industrials": 28.0},
    )

    assert optimization.optimal == ()
    note = optimization.withheld_candidates[0]
    assert [b.dimension for b in note.breaches] == ["sector"]
    assert note.breaches[0].projected_pct == pytest.approx(32.0)


def test_both_dimensions_breach_one_note_two_breaches():
    candidate = _buy("7203.T", conviction="Medium", sector="Industrials", size_pct=4.0)

    optimization = _resolve(
        [candidate],
        set(),
        exchange_weights={"T": 45.0},
        sector_weights={"Industrials": 29.0},
    )

    note = optimization.withheld_candidates[0]
    assert sorted(b.dimension for b in note.breaches) == ["exchange", "sector"]


def test_unknown_sector_skips_sector_dimension():
    """No sector on the analysis ⇒ the sector screen cannot attribute — pass."""
    candidate = _buy("7203.T", conviction="Medium", sector="")

    optimization = _resolve(
        [candidate],
        set(),
        exchange_weights={"T": 10.0},
        sector_weights={"Industrials": 45.0},
    )

    assert optimization.optimal == (candidate,)
    assert optimization.withheld_candidates == ()


def test_missing_size_pct_screens_only_when_bucket_already_over():
    zero_sized = _buy("7203.T", conviction="Medium", size_pct=0.0)

    under = _resolve([zero_sized], set(), exchange_weights={"T": 38.0})
    already_over = _resolve([zero_sized], set(), exchange_weights={"T": 45.0})

    assert under.optimal == (zero_sized,)
    assert already_over.optimal == ()
    assert already_over.withheld_candidates[0].breaches[0].candidate_pct == 0.0


def test_empty_and_none_weights_degrade_to_pure_merit():
    items = [
        _buy("7203.T", conviction="Medium", score=190, is_watchlist=True),
        _buy("0005.HK", conviction="High", score=120),
    ]
    legacy = _resolve(items, {"7203.T"})
    with_none = _resolve(items, {"7203.T"}, exchange_weights=None, sector_weights=None)
    with_empty = _resolve(items, {"7203.T"}, exchange_weights={}, sector_weights={})

    assert legacy == with_none == with_empty
    assert legacy.withheld_candidates == ()
    assert legacy.admitted_over_limit == ()


def test_screen_is_deterministic_across_identical_calls():
    def build():
        return [
            _buy("7203.T", conviction="High", score=168, is_watchlist=True),
            _buy("6758.T", conviction="Medium", score=150),
            _buy("0005.HK", conviction="Medium", score=140),
        ]

    first = _resolve(build(), {"7203.T"}, exchange_weights={"T": 38.0})
    second = _resolve(build(), {"7203.T"}, exchange_weights={"T": 38.0})

    assert first == second


def test_concentration_underfill_reports_partial_fill_with_notes():
    withheld = _buy("7203.T", conviction="Medium")
    accepted = _buy("0005.HK", conviction="Medium")

    optimization = _resolve(
        [withheld, accepted], set(), target_size=6, exchange_weights={"T": 45.0}
    )

    assert optimization.case is WatchlistOptCase.PARTIAL_FILL
    assert len(optimization.optimal) == 1
    assert len(optimization.withheld_candidates) == 1


def test_concentration_removal_only_run_is_not_aligned():
    kept = _buy("0005.HK", conviction="Medium", score=120, is_watchlist=True)
    screened = _buy("7203.T", conviction="Medium", score=190, is_watchlist=True)

    optimization = _resolve(
        [kept, screened],
        {"0005.HK", "7203.T"},
        target_size=1,
        exchange_weights={"T": 45.0},
    )

    assert optimization.case is not WatchlistOptCase.ALIGNED
    assert [(m.item.ticker.yf, m.reason) for m in optimization.remove] == [
        ("7203.T", "concentration_displaced")
    ]


def test_screened_incumbent_gets_exactly_one_move():
    # The incumbent ranks FIRST (score 190) so the screen — not merit — is
    # what displaces it; a lower-ranked name then takes the slot. A merely
    # outranked incumbent keeps the legacy displaced_by_higher_conviction.
    screened = _buy("7203.T", conviction="Medium", score=190, is_watchlist=True)
    refill = _buy("0005.HK", conviction="Medium", score=120)

    optimization = _resolve(
        [screened, refill],
        {"7203.T"},
        target_size=1,
        exchange_weights={"T": 45.0},
    )

    moves = [m for m in optimization.remove if m.item.ticker.yf == "7203.T"]
    assert [(m.reason, m.note is not None) for m in moves] == [
        ("concentration_displaced", True)
    ]
    assert all(item.ticker.yf != "7203.T" for item in optimization.optimal)
    assert all(item.ticker.yf != "7203.T" for item in optimization.keep)


def test_early_return_branches_carry_notes_without_removals():
    screened_incumbent = _buy(
        "7203.T", conviction="Medium", score=190, is_watchlist=True
    )

    no_watchlist = _resolve(
        [screened_incumbent], None, supplied=False, exchange_weights={"T": 45.0}
    )
    unavailable = _resolve(
        [screened_incumbent],
        None,
        supplied=False,
        unavailable=True,
        exchange_weights={"T": 45.0},
    )

    for optimization in (no_watchlist, unavailable):
        assert optimization.remove == ()
        assert [n.item.ticker.yf for n in optimization.withheld_candidates] == [
            "7203.T"
        ]


def test_target_size_zero_with_weights_selects_nothing_and_notes_nothing():
    optimization = _resolve(
        [_buy("7203.T")], set(), target_size=0, exchange_weights={"T": 45.0}
    )
    assert optimization.optimal == ()
    assert optimization.withheld_candidates == ()
    assert optimization.admitted_over_limit == ()


def test_bare_ticker_buckets_as_us_exchange():
    candidate = _buy("ACME", conviction="Medium", size_pct=4.0)

    optimization = _resolve([candidate], set(), exchange_weights={"US": 45.0})

    assert optimization.optimal == ()
    assert optimization.withheld_candidates[0].breaches[0].key == "US"


def test_corrupted_snapshot_exchange_cannot_bypass_screen():
    """Suffix-first: a suffixed ticker buckets by its suffix even when the
    persisted snapshot exchange is garbage — an unknown alias key would read
    weight 0.0 and silently skip the screen."""
    candidate = _buy("7203.T", conviction="Medium", size_pct=4.0)
    candidate.analysis.exchange = "TSEJ"  # IBKR code — not the yf-suffix space

    optimization = _resolve([candidate], set(), exchange_weights={"T": 45.0})

    assert optimization.optimal == ()
    assert optimization.withheld_candidates[0].breaches[0].key == "T"


def test_bare_ticker_uses_snapshot_exchange_when_present():
    candidate = _buy("ACME", conviction="Medium", size_pct=4.0)
    candidate.analysis.exchange = "KS"

    optimization = _resolve([candidate], set(), exchange_weights={"KS": 45.0})

    assert optimization.optimal == ()
    assert optimization.withheld_candidates[0].breaches[0].key == "KS"


def test_analysis_none_item_is_excluded_before_screen_without_crash():
    no_analysis = ReconciliationItem(
        ticker="9432.T",
        action="BUY",
        reason="BUY",
        urgency="MEDIUM",
        analysis=None,
    )
    healthy = _buy("7203.T", conviction="Medium")

    optimization = _resolve([no_analysis, healthy], set(), exchange_weights={"T": 10.0})

    assert optimization.optimal == (healthy,)
    assert no_analysis in optimization.excluded_low_conviction
