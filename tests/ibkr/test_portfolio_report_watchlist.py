"""Watchlist and action-plan report behavior."""

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


class TestWatchlistUnavailableDegradation:
    """When the watchlist (Tier 2 brokerage session) can't be read but holdings
    loaded, off-watchlist BUY candidates are surfaced as direct BUYs rather than
    'add to watchlist' advisories — the run does not abort."""

    def _candidate(self) -> ReconciliationItem:
        return _make_buy_item(ticker="4396.T", conviction="High", is_watchlist=False)

    def test_unavailable_surfaces_buy_candidates_not_watchlist_adds(self):
        report = format_report(
            [self._candidate()],
            _make_portfolio(),
            show_recommendations=True,
            errors={"watchlist": "brokerage session not authenticated"},
            portfolio_data_loaded=True,
        )
        assert "WATCHLIST UNAVAILABLE" in report
        assert "Watchlist filtering is unavailable" in report
        assert "WATCHLIST OPTIMIZATION" in report
        assert "confirm watchlist status and re-check IBKR before acting" in report
        assert "WATCHLIST CANDIDATES" not in report
        assert "ADD TO WATCHLIST" not in report
        assert "NEW BUYS and watchlist filtering are omitted" not in report
        assert "new-buy suggestions are omitted" not in report
        assert "+ ADD" in report

    def test_available_keeps_watchlist_add_framing(self):
        report = format_report(
            [self._candidate()],
            _make_portfolio(),
            show_recommendations=True,
            errors={},
            portfolio_data_loaded=True,
        )
        assert "WATCHLIST UNAVAILABLE" not in report
        assert "WATCHLIST OPTIMIZATION" in report
        assert "+ ADD" in report


class TestWatchlistOptimizationSection:
    """format_report() unified watchlist optimization rendering."""

    def _report(
        self,
        items: list[ReconciliationItem],
        live_orders=None,
        watchlist_name: str | None = "TestWatchlist",
        watchlist_total: int | None = None,
    ) -> str:
        return format_report(
            items,
            _make_portfolio(),
            show_recommendations=True,
            live_orders=live_orders or [],
            watchlist_name=watchlist_name,
            watchlist_total=watchlist_total,
        )

    # ── Line-count / data-completeness ────────────────────────────────────────

    def test_full_data_shows_conviction_and_cost(self):
        """BUY with conviction, quantity, and cash_impact_usd → all 3 detail elements shown."""
        item = _make_buy_item(
            conviction="High",
            suggested_quantity=100,
            suggested_price=2615.0,
            cash_impact_usd=-1752.0,
        )
        report = self._report([item])
        assert "WATCHLIST OPTIMIZATION" in report
        assert "high conviction" in report
        assert "target 4.0%" in report
        assert "Cost:" in report
        assert "use already-settled cash" in report

    def test_no_quantity_hides_cost_line(self):
        """BUY with price but no quantity → no Cost line (can't compute cost without shares)."""
        item = _make_buy_item(
            conviction="Medium",
            suggested_quantity=None,  # no quantity
            suggested_price=2615.0,
            cash_impact_usd=0.0,  # no cost when no quantity
        )
        report = self._report([item])
        assert "medium conviction" in report
        assert "target 4.0%" in report
        assert "Cost:" not in report

    def test_no_price_shows_no_entry_price_indicator(self):
        """BUY with no entry price → '(no entry price' indicator in order line."""
        item = _make_buy_item(
            suggested_quantity=None,
            suggested_price=None,
            cash_impact_usd=0.0,
        )
        report = self._report([item])
        assert "no entry price" in report

    def test_missing_conviction_is_excluded_from_the_buy_ready_pool(self):
        """An analysis without conviction is never made a BUY-ready slot."""
        tb = TradeBlockData(conviction="", size_pct=2.5)
        analysis = AnalysisRecord(
            ticker="6752.T",
            analysis_date="2026-03-01",
            verdict="BUY",
            health_adj=68.0,
            growth_adj=60.0,
            trade_block=tb,
            conviction="",
        )
        item = _make_buy_item(
            ticker="6752.T",
            conviction="",
            analysis=analysis,
            suggested_quantity=None,
            cash_impact_usd=0.0,
        )
        report = self._report([item])
        assert "Excluded below medium conviction: 1" in report
        assert "+ ADD" not in report

    def test_no_analysis_is_excluded_from_the_buy_ready_pool(self):
        """A BUY with no analysis cannot safely occupy a BUY-ready slot."""
        item = ReconciliationItem(
            ticker="9201.T",
            action="BUY",
            urgency="MEDIUM",
            reason="Watchlist BUY",
            ibkr_position=None,
            analysis=None,
            suggested_price=2615.0,
            suggested_order_type="LMT",
            cash_impact_usd=0.0,
            is_watchlist=True,
        )
        report = self._report([item])
        assert "Excluded below medium conviction: 1" in report
        assert "+ ADD" not in report

    def test_header_section_is_always_shown(self):
        """The unified section makes an empty optimization explicit."""
        item = _make_buy_item()
        report = self._report([item])
        assert "WATCHLIST OPTIMIZATION" in report

    def test_title_states_no_watchlist_when_snapshot_metadata_is_absent(self):
        """A name alone is not evidence that a watchlist was actually loaded."""
        item = _make_buy_item()
        report = self._report([item], watchlist_name="MyWatchlist")
        assert "no watchlist loaded — additions only" in report

    def test_auto_discovered_watchlist_uses_total_not_name(self):
        """An auto-discovered watchlist is loaded when its total is present."""
        item = _make_buy_item()
        report = self._report([item], watchlist_name=None, watchlist_total=1)
        assert "optimal BUY-ready set under-filled (1 of 6)" in report
        assert "no watchlist loaded" not in report

    def test_title_reports_the_buy_ready_target_when_watchlist_is_loaded(self):
        """The report uses the optimization target, not a raw watchlist BUY count."""
        items = [_make_buy_item("7203.T"), _make_buy_item("6752.T")]
        report = self._report(items, watchlist_name="MyWatchlist", watchlist_total=15)
        assert "optimal BUY-ready set under-filled (2 of 6)" in report

    def test_title_does_not_treat_a_name_as_loaded_state(self):
        """A missing total remains an additions-only result."""
        item = _make_buy_item()
        report = self._report(
            [item], watchlist_name="MyWatchlist", watchlist_total=None
        )
        assert "no watchlist loaded — additions only" in report

    def test_offwatch_buy_is_an_addition_in_the_unified_section(self):
        """An off-watchlist BUY appears as an addition rather than a parallel list."""
        item = ReconciliationItem(
            ticker="9201.T",
            action="BUY",
            urgency="MEDIUM",
            reason="New BUY (2026-03-01) — High conviction, target 4.0%",
            ibkr_position=None,
            is_watchlist=False,
        )
        report = self._report([item], watchlist_name="MyWatchlist")
        assert "WATCHLIST OPTIMIZATION" in report

    # ── Order annotation for BUY items ────────────────────────────────────────

    def test_buy_with_existing_buy_order_shows_already_submitted(self):
        """BUY item with matching live BUY order → 'ORDER ALREADY SUBMITTED' in report."""
        item = _make_buy_item(ticker="7203.T")
        order = _make_order(ticker="7203", side="B", remaining_size=100, price=2615.0)
        report = self._report([item], live_orders=[order])
        assert "ORDER ALREADY SUBMITTED" in report
        assert "BUY" in report

    def test_buy_with_conflicting_sell_order_shows_conflict(self):
        """BUY recommendation but live SELL order exists → CONFLICT warning."""
        item = _make_buy_item(ticker="7203.T")
        order = _make_order(ticker="7203", side="S", remaining_size=100, price=2615.0)
        report = self._report([item], live_orders=[order])
        assert "CONFLICT" in report
        assert "ORDER ALREADY SUBMITTED" not in report

    def test_buy_with_no_live_orders_no_annotation(self):
        """BUY item with empty live_orders → no order annotation."""
        item = _make_buy_item(ticker="7203.T")
        report = self._report([item], live_orders=[])
        assert "ORDER ALREADY SUBMITTED" not in report
        assert "CONFLICT" not in report

    def test_action_plan_shows_already_submitted_note_for_buy(self):
        """ACTION PLAN (TODAY section) shows 'order already submitted' for BUY already in flight."""
        # A BUY item with settled cash qualifies for funded_today
        item = _make_buy_item(
            ticker="7203.T",
            suggested_quantity=100,
            suggested_price=2615.0,
            cash_impact_usd=-1752.0,
        )
        order = _make_order(ticker="7203", side="B", remaining_size=100, price=2615.0)
        report = self._report([item], live_orders=[order])
        # ACTION PLAN section uses lowercase "order already submitted"
        assert "order already submitted" in report.lower()

    def test_partial_buy_order_shows_remaining_needed(self):
        """BUY item with partial live BUY order → annotation shows how many more shares are needed."""
        item = _make_buy_item(
            ticker="7203.T",
            suggested_quantity=100,
            suggested_price=2615.0,
            cash_impact_usd=-1752.0,
        )
        # Only 50 of 100 shares ordered
        order = _make_order(ticker="7203", side="B", remaining_size=50, price=2615.0)
        report = self._report([item], live_orders=[order])
        assert "PARTIAL ORDER" in report
        assert "50 of" in report
        assert "50 more" in report
        assert "ORDER ALREADY SUBMITTED" not in report


# ── IBKR vs yFinance display symbol tests ──
