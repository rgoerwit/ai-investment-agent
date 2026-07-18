"""Freshness, cash, read-only, and order-authority report behavior."""

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


class TestAnalysisFreshnessReporting:
    def _items(self) -> list[ReconciliationItem]:
        blocking = ReconciliationItem(
            ticker="7203.T",
            action="REVIEW",
            reason="Stale analysis: age 20d > max_age_days 14",
            urgency="MEDIUM",
            ibkr_position=_make_position(ticker="7203.T"),
            analysis=_make_analysis(ticker="7203.T", age_days=20),
        )
        queued = ReconciliationItem(
            ticker="0005.HK",
            action="REVIEW",
            reason="Verdict → DO_NOT_INITIATE  (2026-03-05)",
            urgency="MEDIUM",
            ibkr_position=_make_position(ticker="0005.HK"),
            analysis=_make_analysis(ticker="0005.HK", age_days=20),
            sell_type="SOFT_REJECT",
        )
        due_soon = ReconciliationItem(
            ticker="GTT.PA",
            action="HOLD",
            reason="Position OK",
            urgency="LOW",
            ibkr_position=_make_position(ticker="GTT.PA"),
            analysis=_make_analysis(ticker="GTT.PA", age_days=8),
        )
        return [blocking, queued, due_soon]

    def test_analysis_freshness_section_replaces_split_brain_deadlines(self):
        report = format_report(self._items(), _make_portfolio(), max_age_days=14)
        assert "ANALYSIS FRESHNESS" in report
        assert "Needs review before action:" in report
        assert "Already in refresh queue:" in report
        assert "Due soon:" in report
        assert "Upcoming review deadlines" not in report

    def test_reviews_subtitle_uses_decision_safe_wording(self):
        report = format_report(self._items(), _make_portfolio(), max_age_days=14)
        assert "REVIEWS  (analysis not decision-safe — refresh before acting)" in report

    def test_read_only_refresh_skip_shows_exact_manual_action(self):
        item = ReconciliationItem(
            ticker="7203.T",
            action="REVIEW",
            reason="Stale analysis: age 20d > max_age_days 14",
            urgency="MEDIUM",
            ibkr_position=_make_position(ticker="7203.T"),
            analysis=_make_analysis(ticker="7203.T", age_days=20),
        )
        report = format_report(
            [item],
            _make_portfolio(),
            refresh_activity=RefreshActivity(
                policy="blocking",
                limit=10,
                skipped_read_only=["7203.T"],
            ),
        )
        assert "Skipped (read-only): 7203.T" in report
        assert (
            "User action: read-only mode blocked refresh — run "
            f"{_portfolio_manager_command('--refresh-policy', 'blocking')}"
        ) in report

    def test_successful_refresh_run_can_leave_no_manual_action(self):
        report = format_report(
            [],
            _make_portfolio(),
            refresh_activity=RefreshActivity(
                policy="blocking",
                limit=10,
                refreshed=["7203.T"],
            ),
        )
        assert "Refreshed: 7203.T" in report
        assert "User action: none" in report

    def test_format_json_includes_freshness_summary(self):
        payload = json.loads(
            format_json(
                self._items(),
                _make_portfolio(),
                max_age_days=14,
            )
        )
        summary = payload["analysis_freshness_summary"]
        assert summary["blocking_now_count"] == 1
        assert summary["stale_in_queue_count"] == 1
        assert summary["due_soon_count"] == 1
        assert summary["manual_action_required"] is True

    def test_format_report_shows_stale_screening_freshness(self):
        report = format_report(
            self._items(),
            _make_portfolio(),
            max_age_days=14,
            screening_freshness=ScreeningFreshnessSummary(
                status="stale",
                screening_date="2026-01-05",
                completed_at="2026-01-05T10:30:00Z",
                age_days=90,
                stale_after_days=90,
                candidate_count=245,
                buy_count=12,
            ),
        )
        assert "SCREENING FRESHNESS" in report
        assert "Last completed sweep: 2026-01-05  (90 days ago)" in report
        assert "Candidates screened: 245  ·  BUYs found: 12" in report

    def test_format_report_omits_fresh_screening_freshness(self):
        report = format_report(
            self._items(),
            _make_portfolio(),
            max_age_days=14,
            screening_freshness=ScreeningFreshnessSummary(
                status="fresh",
                screening_date="2026-04-01",
                completed_at="2026-04-01T10:30:00Z",
                age_days=4,
                stale_after_days=90,
                candidate_count=245,
                buy_count=12,
            ),
        )
        assert "SCREENING FRESHNESS" not in report

    def test_format_json_always_includes_screening_freshness(self):
        payload = json.loads(
            format_json(
                self._items(),
                _make_portfolio(),
                max_age_days=14,
                screening_freshness=ScreeningFreshnessSummary(status="missing"),
            )
        )
        assert payload["screening_freshness"]["status"] == "missing"


class TestCashHeaderWording:
    def test_cash_total_notes_when_unsettled_proceeds_are_present(self):
        portfolio = PortfolioSummary(
            account_id="U1234567",
            portfolio_value_usd=52_436,
            cash_balance_usd=1_323,
            settled_cash_usd=500,
            available_cash_usd=0,
            cash_pct=2.5,
            position_count=12,
        )

        report = format_report([], portfolio)

        assert "Cash (total):     $     1,323" in report
        assert "includes $823 of unsettled sale proceeds (not yet spendable)" in report
        assert "Settled cash:     $       500" in report

    def test_cash_total_omits_unsettled_warning_when_all_cash_is_settled(self):
        portfolio = PortfolioSummary(
            account_id="U1234567",
            portfolio_value_usd=52_436,
            cash_balance_usd=1_323,
            settled_cash_usd=1_323,
            available_cash_usd=0,
            cash_pct=2.5,
            position_count=12,
        )

        report = format_report([], portfolio)

        assert "Cash (total):     $     1,323" in report
        assert "all shown cash is settled" in report
        assert "not yet spendable" not in report


class TestCliUiSharedPresentationAlignment:
    def test_shared_live_order_note_matches_report_output(self):
        item = ReconciliationItem(
            ticker="7203.T",
            action="SELL",
            reason="Verdict → DO_NOT_INITIATE",
            urgency="HIGH",
            ibkr_position=_make_position(ticker="7203.T", conid=1234),
            analysis=_make_analysis(ticker="7203.T"),
            suggested_quantity=100,
            suggested_price=1950.0,
            sell_type="HARD_REJECT",
        )
        live_orders = [
            {
                "conid": 1234,
                "ticker": "7203",
                "side": "SELL",
                "remainingSize": 100,
                "price": 1950.0,
                "orderType": "LMT",
                "status": "Submitted",
            }
        ]

        report = format_report(
            [item],
            _make_portfolio(),
            show_recommendations=True,
            live_orders=live_orders,
        )

        assert build_live_order_note(item, live_orders) in report

    def test_shared_cash_summary_matches_report_pending_inflows(self):
        portfolio = PortfolioSummary(
            account_id="U1234567",
            portfolio_value_usd=52_436,
            cash_balance_usd=1_323,
            settled_cash_usd=1_323,
            available_cash_usd=0,
            cash_pct=2.5,
            position_count=12,
        )
        sell_item = ReconciliationItem(
            ticker="7203.T",
            action="SELL",
            reason="Verdict → DO_NOT_INITIATE",
            urgency="HIGH",
            ibkr_position=_make_position(ticker="7203.T"),
            analysis=_make_analysis(ticker="7203.T"),
            suggested_quantity=100,
            suggested_price=1950.0,
            cash_impact_usd=1_300.0,
            settlement_date="2026-03-31",
            sell_type="HARD_REJECT",
        )
        buy_item = ReconciliationItem(
            ticker="ASML.AS",
            action="BUY",
            reason="Watchlist candidate",
            urgency="MEDIUM",
            analysis=_make_analysis(ticker="ASML.AS"),
            suggested_quantity=10,
            suggested_price=50.0,
            cash_impact_usd=-500.0,
            is_watchlist=True,
        )
        shared = build_cash_summary([sell_item, buy_item], portfolio)

        report = format_report(
            [sell_item, buy_item],
            portfolio,
            show_recommendations=True,
        )

        assert f"${shared.pending_inflows_total_usd:>6,.0f}" in report
        assert shared.next_settlement_date in report
        assert f"${shared.settled_cash_after_recommended_buys_usd:>7,.0f}" in report

    def test_soft_sell_excluded_from_pending_inflows(self):
        """SOFT_REJECT sells should not appear in pending_inflows."""
        portfolio = PortfolioSummary(
            account_id="U1234567",
            portfolio_value_usd=50_000,
            cash_balance_usd=500,
            settled_cash_usd=500,
            available_cash_usd=0,
            cash_pct=1.0,
            position_count=10,
        )
        hard_sell = ReconciliationItem(
            ticker="7203.T",
            action="SELL",
            reason="Verdict → DO_NOT_INITIATE",
            urgency="HIGH",
            ibkr_position=_make_position(ticker="7203.T"),
            analysis=_make_analysis(ticker="7203.T"),
            suggested_quantity=100,
            suggested_price=2000.0,
            cash_impact_usd=1_400.0,
            settlement_date="2026-04-21",
            sell_type="HARD_REJECT",
        )
        soft_sell = ReconciliationItem(
            ticker="0005.HK",
            action="SELL",
            reason="Verdict → DO_NOT_INITIATE",
            urgency="HIGH",
            ibkr_position=_make_position(ticker="0005.HK", currency="HKD"),
            analysis=_make_analysis(ticker="0005.HK"),
            suggested_quantity=200,
            suggested_price=50.0,
            cash_impact_usd=1_300.0,
            settlement_date="2026-04-21",
            sell_type="SOFT_REJECT",
        )
        summary = build_cash_summary([hard_sell, soft_sell], portfolio)

        # Only hard sell in pending inflows
        assert summary.pending_inflows_total_usd == 1_400.0
        assert len(summary.pending_inflows) == 1
        assert summary.pending_inflows[0].ticker_yf == "7203.T"
        # Soft sell in conditional
        assert summary.conditional_proceeds_usd == 1_300.0

    def test_hard_sells_and_stops_included_in_pending(self):
        """HARD_REJECT and STOP_BREACH sells appear in pending_inflows."""
        portfolio = PortfolioSummary(
            account_id="U1234567",
            portfolio_value_usd=50_000,
            cash_balance_usd=500,
            settled_cash_usd=500,
            available_cash_usd=0,
            cash_pct=1.0,
            position_count=10,
        )
        hard = ReconciliationItem(
            ticker="7203.T",
            action="SELL",
            reason="test",
            urgency="HIGH",
            ibkr_position=_make_position(ticker="7203.T"),
            analysis=_make_analysis(ticker="7203.T"),
            cash_impact_usd=1_000.0,
            settlement_date="2026-04-21",
            sell_type="HARD_REJECT",
        )
        stop = ReconciliationItem(
            ticker="0005.HK",
            action="SELL",
            reason="test",
            urgency="HIGH",
            ibkr_position=_make_position(ticker="0005.HK", currency="HKD"),
            analysis=_make_analysis(ticker="0005.HK"),
            cash_impact_usd=800.0,
            settlement_date="2026-04-21",
            sell_type="STOP_BREACH",
        )
        summary = build_cash_summary([hard, stop], portfolio)
        assert summary.pending_inflows_total_usd == 1_800.0
        assert len(summary.pending_inflows) == 2
        assert summary.conditional_proceeds_usd == 0.0

    def test_conditional_proceeds_shown_in_report(self):
        """When soft sells exist, report shows conditional proceeds line."""
        portfolio = PortfolioSummary(
            account_id="U1234567",
            portfolio_value_usd=50_000,
            cash_balance_usd=500,
            settled_cash_usd=500,
            available_cash_usd=0,
            cash_pct=1.0,
            position_count=5,
        )
        soft_sell = ReconciliationItem(
            ticker="0005.HK",
            action="SELL",
            reason="Verdict → DO_NOT_INITIATE",
            urgency="HIGH",
            ibkr_position=_make_position(ticker="0005.HK", currency="HKD"),
            analysis=_make_analysis(ticker="0005.HK", verdict="DO_NOT_INITIATE"),
            suggested_quantity=200,
            suggested_price=50.0,
            cash_impact_usd=1_300.0,
            settlement_date="2026-04-21",
            sell_type="SOFT_REJECT",
        )
        report = format_report(
            [soft_sell],
            portfolio,
            show_recommendations=True,
        )
        # Soft sells should NOT be in "Total pending" — instead conditional
        assert (
            "Conditional (soft-sell reviews)" in report
            or "No confirmed sale proceeds" in report
        )


class TestReadOnlyDataNotLoaded:
    """format_report() must not assert own/watchlist status when no IBKR data loaded."""

    def _offwatch_buy(self) -> ReconciliationItem:
        """An off-watchlist BUY candidate sourced from a saved analysis."""
        return ReconciliationItem(
            ticker="7203.T",
            action="BUY",
            urgency="MEDIUM",
            reason="Analysis says BUY",
            ibkr_position=None,
            is_watchlist=False,
            analysis=_make_analysis(ticker="7203.T", verdict="BUY"),
            suggested_price=2100.0,
        )

    def test_read_only_says_status_unknown_not_new_position(self):
        """portfolio_data_loaded=False → banner shown, no 'new position' assertion."""
        report = format_report(
            [self._offwatch_buy()],
            _make_portfolio(),
            show_recommendations=True,
            portfolio_data_loaded=False,
        )
        assert "READ-ONLY" in report
        assert "UNKNOWN" in report
        assert "[own/watchlist status unknown]" in report
        assert "[not on watchlist — new position]" not in report
        # Header/cash relabeled to "not loaded" rather than a misleading $0/N/A.
        assert "Account:          not loaded (read-only)" in report
        assert "Net liquidation:  not loaded" in report
        assert "no watchlist loaded — additions only" in report
        assert "inspect and add to watchlist before acting" not in report

    def test_loaded_default_marks_the_candidate_as_a_watchlist_addition(self):
        """Default (data loaded) presents an explicit addition, not an ownership claim."""
        report = format_report(
            [self._offwatch_buy()],
            _make_portfolio(),
            show_recommendations=True,
        )
        assert "+ ADD" in report
        assert "[own/watchlist status unknown]" not in report
        assert "READ-ONLY — no IBKR connection" not in report
        assert "not loaded (read-only)" not in report
        assert "no watchlist loaded — additions only" in report

    def test_format_json_surfaces_portfolio_data_loaded(self):
        """format_json exposes the flag both ways for machine consumers."""
        offline = json.loads(
            format_json(
                [self._offwatch_buy()],
                _make_portfolio(),
                show_recommendations=True,
                portfolio_data_loaded=False,
            )
        )
        online = json.loads(format_json([self._offwatch_buy()], _make_portfolio()))
        assert offline["portfolio_data_loaded"] is False
        assert online["portfolio_data_loaded"] is True

    def test_read_only_empty_candidates_does_not_imply_loaded_cash(self):
        """Empty-candidate branch must not claim '$0 settled cash within the buffer'
        in read-only mode (cash was never loaded)."""
        report = format_report(
            [],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_candidates_blocked_by_cash=2,
            portfolio_data_loaded=False,
        )
        assert "holdings and cash were not loaded" in report
        assert "within the cash buffer" not in report


class TestOrderMatcherAuthority:
    """Conid is authoritative; terminal statuses are not live orders."""

    def test_differing_conids_never_fall_back_to_symbol(self):
        """The AGS collision: SGX 'AGS' item must not match a Brussels Ageas
        'AGS' order — comparable conids that differ end the comparison."""
        item = _make_sell_item(ticker="9201.T", sell_type="HARD_REJECT")
        order = _make_order(conid=11111, ticker="9201", side="B", price=70.80)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "CONFLICT" not in report
        assert "ORDER" not in report.split("SELL RECOMMENDATIONS")[-1].split("═")[0]

    def test_cancelled_order_never_annotates(self):
        item = _make_sell_item(ticker="9201.T")
        order = _make_order(conid=99999, side="B", status="Cancelled")
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "CONFLICT" not in report
        assert "ORDER ALREADY SUBMITTED" not in report

    def test_inactive_order_never_annotates(self):
        item = _make_sell_item(ticker="9201.T")
        order = _make_order(conid=99999, side="S", status="Inactive")
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "ORDER ALREADY SUBMITTED" not in report
        assert "CONFLICT" not in report

    def test_filled_order_is_historical_note_not_conflict(self):
        """A filled cross-side order is information, not a live conflict —
        and carries no 'do not re-enter' imperative."""
        item = _make_sell_item(ticker="9201.T")
        order = _make_order(conid=99999, side="B", status="Filled", price=279.48)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "ORDER FILLED" in report
        assert "CONFLICT" not in report
        assert "do not re-enter" not in report

    def test_open_order_wins_over_earlier_filled_order(self):
        """A filled order encountered first must not hide a later open
        cross-side conflict."""
        item = _make_sell_item(ticker="9201.T")
        filled = _make_order(conid=99999, side="S", status="Filled")
        open_buy = _make_order(conid=99999, side="B", status="Submitted")
        report = format_report(
            [item], _make_portfolio(), live_orders=[filled, open_buy]
        )
        assert "CONFLICT" in report
        assert "ORDER FILLED" not in report

    def test_non_numeric_conid_falls_back_to_symbol(self):
        item = _make_sell_item(ticker="9201.T")
        order = _make_order(ticker="9201", side="S", remaining_size=100)
        order["conid"] = "not-a-number"
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "ORDER ALREADY SUBMITTED" in report
