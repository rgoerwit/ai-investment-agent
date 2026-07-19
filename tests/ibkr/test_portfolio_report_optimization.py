"""Watchlist optimization and executable-buy report behavior."""

from __future__ import annotations

import json
from unittest.mock import patch

from scripts.portfolio_manager import (
    _analysis_command,
    _compute_dip_score,
    _portfolio_manager_command,
    _portfolio_manager_recommend_command,
    format_json,
    format_report,
)
from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
    TradeBlockData,
)
from src.ibkr.portfolio_presentation import (
    build_cash_summary,
    build_live_order_note,
    group_portfolio_actions,
)
from src.ibkr.refresh_service import RefreshActivity
from src.ibkr.screening_freshness import ScreeningFreshnessSummary
from src.ibkr.ticker import Ticker
from src.ibkr.watchlist_optimization import resolve_watchlist_optimization
from tests.ibkr.format_report_cases import (
    CORRELATED_SELL_EVENT_FLAG,
    _make_buy_item,
    _make_dip_item,
    _make_offwatch_buy,
    _make_order,
    _make_sell_item,
    _make_sell_item_with_analysis,
    _make_watchlist_review,
    _panic_items,
)
from tests.ibkr.reconciler_cases import (
    _make_analysis,
    _make_portfolio,
    _make_position,
)


class TestWatchlistOptimizationReporting:
    """The operator-facing text must make each optimization state actionable."""

    @staticmethod
    def _set_score(item: ReconciliationItem, score: float) -> ReconciliationItem:
        assert item.analysis is not None
        item.analysis.health_adj = score / 2
        item.analysis.growth_adj = score / 2
        return item

    def test_case_1_additions_only_explains_that_no_merge_is_possible(self):
        report = format_report(
            [_make_offwatch_buy("7203.T")],
            _make_portfolio(),
            show_recommendations=True,
        )

        assert "WATCHLIST ADDITION REVIEW" in report
        assert "no watchlist loaded; additions evaluated" in report
        assert "  ADD" in report
        assert "REMOVE FROM WATCHLIST" not in report
        assert "[Update IBKR watchlist to]" not in report

    def test_case_2_empty_supplied_watchlist_with_no_worthy_candidates_is_honest(self):
        low = _make_buy_item("7203.T", conviction="Low", is_watchlist=False)
        report = format_report(
            [low],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers=set(),
            watchlist_total=0,
        )

        assert "no additions recommended from current state" in report
        assert "ANALYZED BUYS NOT RECOMMENDED — BELOW CONVICTION BAR:" in report
        assert "7203 (7203.T)" in report
        assert "new additions require High" in report
        assert "ADDITIONS RECOMMENDED NOW: None" in report
        assert "REMOVE FROM WATCHLIST" not in report

    def test_empty_result_does_not_recommend_emptying_supplied_watchlist(self):
        items = [
            self._set_score(
                _make_buy_item(ticker, is_watchlist=True, conviction="High"),
                score,
            )
            for ticker, score in (("3393.T", 100), ("1926.T", 110), ("3762.T", 105))
        ]
        portfolio = _make_portfolio()
        portfolio.exchange_weights = {"T": 45.0}

        report = format_report(
            items,
            portfolio,
            show_recommendations=True,
            watchlist_tickers={item.ticker.yf for item in items},
            watchlist_total=len(items),
        )
        machine = json.loads(
            format_json(
                items,
                portfolio,
                watchlist_tickers={item.ticker.yf for item in items},
                watchlist_total=len(items),
            )
        )

        assert "no additions recommended from current state" in report
        assert (
            "ADMINISTRATIVE WATCHLIST RETENTION — NOT A BUY RECOMMENDATION:" in report
        )
        assert report.count("REMOVE FROM WATCHLIST") == 2
        assert [
            row["ticker_yf"]
            for row in machine["recommendation_plan"]["watchlist"][
                "retained_for_watchlist_floor"
            ]
        ] == ["1926.T"]

    def test_supplied_empty_watchlist_with_no_items_reports_no_recommendation(self):
        report = format_report(
            [],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers=set(),
            watchlist_total=0,
        )

        assert "no additions recommended from current state" in report
        assert "ADDITIONS RECOMMENDED NOW: None" in report
        assert "REMOVE FROM WATCHLIST" not in report

    def test_case_3_partial_fill_reports_capacity_and_each_addition(self):
        first = _make_offwatch_buy("7203.T", conviction="High")
        second = _make_offwatch_buy("6758.T", conviction="Medium")
        report = format_report(
            [first, second],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers=set(),
            watchlist_total=0,
        )

        assert "1 addition recommended from current state" in report
        assert sum(line.startswith("  ADD     ") for line in report.splitlines()) == 1
        assert "7203" in report
        assert "ANALYZED BUYS NOT RECOMMENDED — BELOW CONVICTION BAR:" in report

    def test_case_4_full_optimization_separates_keeps_additions_and_optional_swaps(
        self,
    ):
        keep_one = self._set_score(_make_buy_item("7203.T"), 200)
        keep_two = self._set_score(_make_buy_item("6758.T"), 180)
        replace_one = self._set_score(_make_buy_item("9432.T", conviction="Medium"), 50)
        replace_two = self._set_score(_make_buy_item("9984.T", conviction="Medium"), 30)
        additions = [
            self._set_score(_make_offwatch_buy("8306.T"), 190),
            self._set_score(_make_offwatch_buy("8058.T"), 170),
            self._set_score(_make_offwatch_buy("7201.T", conviction="Medium"), 200),
            self._set_score(_make_offwatch_buy("4063.T", conviction="Medium"), 190),
        ]
        report = format_report(
            [keep_one, keep_two, replace_one, replace_two, *additions],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers={"7203.T", "6758.T", "9432.T", "9984.T"},
            watchlist_total=4,
        )

        assert "2 additions recommended from current state" in report
        assert "KEEPING ACTIVE (4):" in report
        assert sum(line.startswith("  ADD     ") for line in report.splitlines()) == 2
        assert "OPTIONAL OPTIMIZATION" not in report
        assert "REMOVE FROM WATCHLIST" not in report

    def test_must_remove_is_visually_distinct_from_optional_optimization(self):
        rejected = ReconciliationItem(
            ticker="7203.T",
            action="REMOVE",
            reason="Rejected",
            urgency="MEDIUM",
            analysis=_make_analysis(ticker="7203.T", verdict="DO_NOT_INITIATE"),
            is_watchlist=True,
        )
        low = _make_buy_item("6758.T", conviction="Low")
        report = format_report(
            [rejected, low],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers={"7203.T", "6758.T"},
            watchlist_total=2,
        )

        assert "MUST REMOVE (verdict reject):" in report
        assert "OPTIONAL OPTIMIZATION" not in report
        assert (
            "ADMINISTRATIVE WATCHLIST RETENTION — NOT A BUY RECOMMENDATION:" in report
        )
        assert "verdict DO_NOT_INITIATE" in report
        assert "below medium conviction" in report

    def test_safe_replacement_list_preserves_active_monitor_review_and_held_entries(
        self,
    ):
        active = _make_buy_item("7203.T")
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
        held = ReconciliationItem(
            ticker="AAPL",
            action="HOLD",
            reason="Held",
            urgency="LOW",
            ibkr_position=_make_position(ticker="AAPL"),
        )
        report = format_report(
            [active, monitor, review, held],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers={"7203.T", "6758.T", "9432.T", "AAPL"},
            watchlist_total=4,
        )

        assert "KEEPING ACTIVE (1):" in report
        assert "KEEPING MONITORS (1):" in report
        assert "KEEPING REVIEWS (1):" in report
        assert "KEEPING PROTECTED (1): AAPL" in report
        assert "[Update IBKR watchlist to]: 7203, 6758, 9432, AAPL" in report

    def test_exchange_ambiguous_raw_symbols_withhold_replacement_list(self):
        current = _make_buy_item("BHP.AX")
        candidate = _make_offwatch_buy("BHP.L")
        report = format_report(
            [current, candidate],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers={"BHP.AX"},
            watchlist_total=1,
        )

        assert "KEEPING ACTIVE (1):" in report
        assert "  ADD" in report
        assert "raw IBKR symbols are exchange-ambiguous" in report


class TestExecutableBuyAlignment:
    """CASH SUMMARY, ACTION PLAN (TODAY), and Plan turnover all follow the
    merit-selected optimal set via one shared executable-buy predicate — a
    sized-but-displaced watchlist BUY must not reserve cash or get an order."""

    def _report(self, items, watchlist_total):
        return format_report(
            items,
            _make_portfolio(),
            show_recommendations=True,
            watchlist_total=watchlist_total,
        )

    def test_selected_watchlist_buy_in_both_cash_summary_and_today(self):
        report = self._report([_make_buy_item("7203.T")], watchlist_total=1)
        cash_block = report.split("CASH SUMMARY")[1].split("ACTION PLAN")[0]
        today_block = report.split("ACTION PLAN")[1]
        assert "BUY  7203.T  (100 sh):" in cash_block
        assert "→ BUY  7203" in today_block

    def test_displaced_watchlist_buy_excluded_from_cash_today_and_turnover(self):
        selected = [_make_buy_item(f"720{i}.T", conviction="High") for i in range(1, 7)]
        displaced = _make_buy_item("8888.T", conviction="Medium")
        report = self._report([*selected, displaced], watchlist_total=7)

        opt_block = report.split("WATCHLIST ADDITION REVIEW")[1].split("CASH SUMMARY")[
            0
        ]
        assert "OPTIONAL OPTIMIZATION" in opt_block
        assert "REMOVE FROM WATCHLIST  8888 (8888.T)" in opt_block

        cash_block = report.split("CASH SUMMARY")[1].split("ACTION PLAN")[0]
        today_block = report.split("ACTION PLAN")[1]
        for item in selected:
            sym = item.ticker.ibkr
            assert f"BUY  {sym}.T  (100 sh):" in cash_block
            assert f"→ BUY  {sym}" in today_block
        assert "8888" not in cash_block
        assert "8888" not in today_block
        # Turnover counts the six selected buys only (6 × $1,752), not seven.
        assert "buys ~$10,512" in report

    def test_advisory_offwatch_buy_reserves_no_cash_and_no_order(self):
        report = self._report([_make_offwatch_buy("WDO.TO")], watchlist_total=0)
        assert "  ADD" in report
        assert "→ BUY  WDO" not in report
        assert "BUY  WDO  (100 sh):" not in report
        assert "Plan turnover" not in report

    def test_unsized_selected_buy_is_kept_without_phantom_cash(self):
        item = _make_buy_item(suggested_quantity=None, cash_impact_usd=0.0)
        report = self._report([item], watchlist_total=1)
        assert "KEEPING ACTIVE (1): 7203 (7203.T)" in report
        cash_block = report.split("CASH SUMMARY")[1]
        assert "BUY  7203  (" not in cash_block
        assert "ACTION PLAN" not in report

    def test_unsized_displaced_buy_renders_only_as_optional_removal(self):
        """A cash-blocked (unsized) watchlist BUY that also loses its slot must
        surface as an optional removal only — no cash line, no order, no crash."""
        selected = [_make_buy_item(f"720{i}.T", conviction="High") for i in range(1, 7)]
        displaced = _make_buy_item(
            "8888.T",
            conviction="Medium",
            suggested_quantity=None,
            cash_impact_usd=0.0,
        )
        report = self._report([*selected, displaced], watchlist_total=7)
        opt_block = report.split("WATCHLIST ADDITION REVIEW")[1].split("CASH SUMMARY")[
            0
        ]
        assert "REMOVE FROM WATCHLIST  8888 (8888.T)" in opt_block
        assert "8888" not in report.split("CASH SUMMARY")[1]

    def test_turnover_shows_sells_while_excluding_advisory_buys(self):
        """Executable sells still drive the turnover line even when every BUY
        in the run is advisory (off-watch) and contributes $0."""
        sell = _make_sell_item("9201.T").model_copy(
            update={
                "action_basis": "CONFIRMED_THESIS_FAILURE",
                "analysis": _make_analysis(
                    ticker="9201.T",
                    verdict="DO_NOT_INITIATE",
                ),
                "suggested_quantity": 100,
                "cash_impact_usd": 900.0,
                "settlement_date": "2026-07-20",
            }
        )
        report = self._report([sell, _make_offwatch_buy("WDO.TO")], watchlist_total=0)
        assert "executable sells ~$900" in report
        assert "buys ~$0" in report


class TestKeepingReviewsQuickMarker:
    """Quick-mode BUY verdicts surface as watchlist REVIEWs; the KEEPING
    REVIEWS summary line carries a quick count so the 'run full first' signal
    is visible inside the optimization section itself."""

    def test_quick_review_count_annotated(self):
        items = [
            _make_watchlist_review("7203.T", quick=True),
            _make_watchlist_review("6758.T", quick=False),
        ]
        report = format_report(
            items,
            _make_portfolio(),
            show_recommendations=True,
            watchlist_total=2,
        )
        assert "KEEPING REVIEWS (2):" in report
        assert "(1 quick — re-run full)" in report
        # The full instruction still lives in the main REVIEWS section.
        assert "re-run full analysis before acting" in report

    def test_no_suffix_when_no_quick_reviews(self):
        report = format_report(
            [_make_watchlist_review("6758.T", quick=False)],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_total=1,
        )
        assert "KEEPING REVIEWS (1):" in report
        assert "quick — re-run full" not in report

    def test_review_without_analysis_renders_without_crash(self):
        """A 'no analysis found' watchlist REVIEW (analysis=None) must render
        and never count toward the quick suffix."""
        item = ReconciliationItem(
            ticker="6971.T",
            action="REVIEW",
            urgency="MEDIUM",
            reason="Watchlist: no analysis found",
            analysis=None,
            is_watchlist=True,
        )
        report = format_report(
            [item],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_total=1,
        )
        assert "KEEPING REVIEWS (1): 6971 (6971.T)" in report
        assert "quick — re-run full" not in report

    def test_mode_unknown_review_not_counted_quick(self):
        """is_quick_mode is tri-state; None (mode unknown, legacy artifact)
        must not be counted as quick."""
        item = _make_watchlist_review("6758.T", quick=False)
        item.analysis.is_quick_mode = None
        report = format_report(
            [item],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_total=1,
        )
        assert "KEEPING REVIEWS (1):" in report
        assert "quick — re-run full" not in report


class TestBuildCashSummaryOptimizerAware:
    """build_cash_summary counts only merit+concentration-selected buys when
    given the optimization, and preserves legacy behavior exactly when not —
    other callers still depend on the unsafe legacy mode."""

    def _items_and_optimization(self):
        screened = _make_buy_item("7203.T", conviction="Medium")  # T over limit
        selected = _make_buy_item("MEGP.L", conviction="Medium")
        items = [screened, selected]
        groups = group_portfolio_actions(
            items,
            watchlist_tickers={"7203.T", "MEGP.L"},
            exchange_weights={"T": 45.0},
        )
        optimization = resolve_watchlist_optimization(
            items,
            groups,
            watchlist_tickers={"7203.T", "MEGP.L"},
            watchlist_supplied=True,
            watchlist_unavailable=False,
            exchange_weights={"T": 45.0},
        )
        return items, optimization

    def test_optimizer_aware_excludes_screened_buy(self):
        items, optimization = self._items_and_optimization()
        summary = build_cash_summary(
            items, _make_portfolio(), watchlist_optimization=optimization
        )
        assert summary.recommended_buy_cost_usd == 1752.0  # MEGP only

    def test_legacy_none_counts_every_watchlist_buy(self):
        items, _ = self._items_and_optimization()
        legacy = build_cash_summary(items, _make_portfolio())
        assert legacy.recommended_buy_cost_usd == 3504.0  # both, pre-optimizer

    def test_empty_optimal_yields_zero_recommended_cost(self):
        items, _ = self._items_and_optimization()
        groups = group_portfolio_actions(items, watchlist_tickers=set())
        empty_opt = resolve_watchlist_optimization(
            items,
            groups,
            watchlist_tickers=set(),
            watchlist_supplied=True,
            watchlist_unavailable=False,
            target_size=0,
        )
        summary = build_cash_summary(
            items, _make_portfolio(), watchlist_optimization=empty_opt
        )
        assert summary.recommended_buy_cost_usd == 0.0
