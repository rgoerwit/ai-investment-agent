"""Sell detail and live-order report behavior."""

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


class TestPnlLine:
    """Tests for the _pnl_line gain/loss estimate helper embedded in format_report()."""

    # We test _pnl_line indirectly through format_report() output, which is the only
    # public surface. Each test creates minimal SELL items and inspects the report lines.

    def _report_for(self, item: ReconciliationItem) -> str:
        return format_report([item], _make_portfolio(), show_recommendations=False)

    def test_gain_line_shows_positive_pnl(self):
        """Position with current > avg_cost shows 'est. gain:' in local currency."""
        # (2300 - 2000) × 100 shares = +¥30,000
        item = _make_sell_item(
            avg_cost_local=2000.0,
            current_price_local=2300.0,
        )
        report = self._report_for(item)
        assert "est. gain:" in report
        assert "+JPY 30,000" in report

    def test_loss_line_shows_negative_pnl(self):
        """Position with current < avg_cost shows 'est. loss:' in local currency."""
        # (2700 - 2780) × 100 shares = -¥8,000
        item = _make_sell_item(
            avg_cost_local=2780.0,
            current_price_local=2700.0,
        )
        report = self._report_for(item)
        assert "est. loss:" in report
        assert "-JPY 8,000" in report

    def test_gain_line_has_tax_note(self):
        """Gain lines include 'verify holding period in IBKR'."""
        item = _make_sell_item(
            avg_cost_local=2000.0,
            current_price_local=2300.0,
            unrealized_pnl_usd=234.0,
        )
        report = self._report_for(item)
        assert "verify holding period in IBKR" in report

    def test_loss_line_has_no_tax_note(self):
        """Loss lines do NOT include the holding-period note."""
        item = _make_sell_item(
            avg_cost_local=2780.0,
            current_price_local=2700.0,
            unrealized_pnl_usd=-89.0,
        )
        report = self._report_for(item)
        assert "verify holding period in IBKR" not in report

    def test_suppressed_when_no_cost_basis(self):
        """avg_cost_local == 0 → pnl line suppressed (no est. gain/loss in report)."""
        item = _make_sell_item(
            avg_cost_local=0.0,
            current_price_local=2700.0,
            unrealized_pnl_usd=-89.0,
        )
        report = self._report_for(item)
        assert "est. gain:" not in report
        assert "est. loss:" not in report

    def test_suppressed_on_mismatch(self):
        """≥90% price swing → currency-unit mismatch warning, not a dollar figure."""
        # avg_cost_local in JPY raw (e.g. 27.80) vs current in full JPY (2700) → >9600% swing
        item = _make_sell_item(
            avg_cost_local=27.80,
            current_price_local=2700.0,
            unrealized_pnl_usd=500.0,
        )
        report = self._report_for(item)
        assert "currency-unit mismatch" in report
        assert "est. gain:" not in report
        assert "est. loss:" not in report

    def test_partial_sell_prorates_pnl(self):
        """suggested_quantity < quantity → P&L uses sell_qty, not full position qty."""
        # (2200 - 2000) × 50 shares sold = +¥10,000
        item = _make_sell_item(
            quantity=100,
            avg_cost_local=2000.0,
            current_price_local=2200.0,
            suggested_quantity=50,
        )
        item = item.model_copy(
            update={
                "action_basis": "CONFIRMED_THESIS_FAILURE",
                "analysis": _make_analysis(
                    ticker=item.ticker.yf,
                    verdict="DO_NOT_INITIATE",
                ),
                "cash_impact_usd": 1100.0,
                "settlement_date": "2026-07-20",
            }
        )
        report = self._report_for(item)
        assert "+JPY 10,000" in report
        assert "est. gain:" in report

    def test_sell_items_in_report_show_pnl(self):
        """format_report() output for a HARD_REJECT SELL contains est. gain/loss line."""
        item = _make_sell_item(
            sell_type="HARD_REJECT",
            reason="Verdict → DO_NOT_INITIATE  (2026-03-05)",
            avg_cost_local=1500.0,
            current_price_local=1728.0,
            unrealized_pnl_usd=180.0,
        )
        item = item.model_copy(
            update={
                "action_basis": "CONFIRMED_THESIS_FAILURE",
                "analysis": _make_analysis(
                    ticker=item.ticker.yf,
                    verdict="DO_NOT_INITIATE",
                ),
                "suggested_quantity": 100,
                "cash_impact_usd": item.ibkr_position.market_value_usd,
                "settlement_date": "2026-07-20",
            }
        )
        report = format_report([item], _make_portfolio(), show_recommendations=False)
        assert "SELL RECOMMENDATIONS" in report
        assert "[CONFIRMED THESIS FAILURE]" in report
        assert "est. gain:" in report


class TestScoreLine:
    """Tests for the _score_line fundamentals helper embedded in format_report()."""

    def _report_for(self, item: ReconciliationItem) -> str:
        return format_report([item], _make_portfolio(), show_recommendations=False)

    def test_stop_breach_shows_health_and_growth(self):
        """STOP_BREACH SELL includes health and growth scores in output."""
        item = _make_sell_item_with_analysis(health=75.0, growth=68.0)
        report = self._report_for(item)
        assert "Health:75" in report
        assert "Growth:68" in report

    def test_stop_breach_shows_zone_and_verdict(self):
        """STOP_BREACH SELL includes zone and original verdict."""
        item = _make_sell_item_with_analysis(
            zone="MODERATE", verdict="BUY", conviction="High"
        )
        report = self._report_for(item)
        assert "Risk zone:MODERATE" in report
        assert "BUY (High)" in report

    def test_hard_reject_shows_scores(self):
        """HARD_REJECT SELL shows scores — helps confirm it genuinely failed."""
        item = _make_sell_item_with_analysis(
            sell_type="HARD_REJECT",
            reason="Verdict → DO_NOT_INITIATE  (2026-03-05)",
            health=38.0,
            growth=32.0,
            zone="HIGH",
            verdict="DO_NOT_INITIATE",
            conviction="Low",
        )
        item = item.model_copy(
            update={
                "action_basis": "CONFIRMED_THESIS_FAILURE",
                "suggested_quantity": 100,
                "cash_impact_usd": item.ibkr_position.market_value_usd,
                "settlement_date": "2026-07-20",
            }
        )
        report = self._report_for(item)
        assert "Health:38" in report
        assert "Growth:32" in report
        assert "Risk zone:HIGH" in report

    def test_score_line_shows_analysis_date(self):
        """Score line includes analysis date so operator knows how stale the scores are."""
        item = _make_sell_item_with_analysis(analysis_date="2026-01-15")
        report = self._report_for(item)
        assert "analysis:  2026-01-15" in report

    def test_score_line_suppressed_when_no_analysis(self):
        """Item with no analysis → no score line (no Health/Growth in output)."""
        item = _make_sell_item()  # no analysis attached
        report = self._report_for(item)
        assert "Health:" not in report
        assert "Growth:" not in report

    def test_score_line_suppressed_when_no_scores(self):
        """Analysis with no health_adj/growth_adj → score line suppressed."""
        item = _make_sell_item()
        item = item.model_copy(
            update={
                "analysis": AnalysisRecord(
                    ticker="9201.T",
                    analysis_date="2026-01-15",
                    verdict="BUY",
                    # health_adj and growth_adj deliberately omitted
                )
            }
        )
        report = self._report_for(item)
        assert "Health:" not in report
        assert "Growth:" not in report

    def test_score_line_before_pnl_line(self):
        """Score line appears before the P&L line in output."""
        item = _make_sell_item_with_analysis(health=75.0, growth=68.0)
        # Give it a valid cost basis so pnl_line also appears
        pos = item.ibkr_position.model_copy(
            update={"avg_cost_local": 2000.0, "current_price_local": 2300.0}
        )
        item = item.model_copy(update={"ibkr_position": pos})
        report = self._report_for(item)
        health_pos = report.index("Health:75")
        pnl_pos = report.index("est. gain:")
        assert health_pos < pnl_pos

    def test_macro_review_uses_display_data_line_not_score_line(self):
        """Demoted SOFT_REJECT (macro review) uses _display_data_line, not _score_line.
        Scores appear via the existing macro-review data display path."""
        item = _make_dip_item(
            ticker="9201.T",
            health=75,
            growth=68,
            entry=2800,
            current_price=2700,
            stop=2600,
            target=3200,
            verdict="DO_NOT_INITIATE",
        )
        report = format_report(
            [item],
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        # Health/Growth appear through the compact macro-review detail path, not
        # the dated _score_line path.
        assert "H:75%" in report
        assert "G:68%" in report


# ── Order annotation helpers ──────────────────────────────────────────────────


class TestOrderAnnotation:
    """format_report() live-order annotation via live_orders parameter."""

    @staticmethod
    def _confirmed(item: ReconciliationItem) -> ReconciliationItem:
        position = item.ibkr_position
        quantity = abs(int(position.quantity)) if position else 1
        proceeds = position.market_value_usd if position else 1.0
        analysis = item.analysis or _make_analysis(
            ticker=item.ticker.yf,
            verdict="DO_NOT_INITIATE",
        )
        if position is not None:
            analysis = analysis.model_copy(update={"currency": position.currency})
        return item.model_copy(
            update={
                "action_basis": "CONFIRMED_THESIS_FAILURE",
                "analysis": analysis,
                "suggested_quantity": item.suggested_quantity or quantity,
                "cash_impact_usd": proceeds,
                "settlement_date": "2026-07-20",
            }
        )

    def test_sell_with_matching_open_order_shows_note(self):
        """SELL item with matching open SELL order (by conid) → 'ORDER ALREADY SUBMITTED' shown."""
        item = _make_sell_item(
            ticker="9201.T",
            sell_type="STOP_BREACH",
            reason="Stop breached: price 2700.00 < stop 2780.00",
        )
        item = self._confirmed(item)
        # Position conid = 99999 (set by _make_sell_item via NormalizedPosition)
        order = _make_order(conid=99999, side="S", remaining_size=100, price=2780.0)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "ORDER ALREADY SUBMITTED" in report
        assert "SELL" in report
        assert "2780.00" in report

    def test_buy_with_matching_open_order_shows_note(self):
        """BUY item (no position) with open BUY order matched by symbol → note shown."""
        item = ReconciliationItem(
            ticker="9201.T",
            action="BUY",
            urgency="LOW",
            reason="Watchlist BUY",
            ibkr_position=None,
            is_watchlist=True,
        )
        # No conid on a BUY with no position — match by symbol "9201"
        order = _make_order(ticker="9201", side="B", remaining_size=50, price=2750.0)
        report = format_report(
            [item],
            _make_portfolio(),
            live_orders=[order],
            watchlist_name="TestWatchlist",
        )
        assert "ORDER ALREADY SUBMITTED" in report
        assert "BUY" in report

    def test_opposite_side_order_shows_conflict_note(self):
        """Open BUY order when recommending SELL → CONFLICT warning shown."""
        item = _make_sell_item(ticker="9201.T", sell_type="STOP_BREACH")
        item = self._confirmed(item)
        order = _make_order(conid=99999, side="B", remaining_size=100, price=2780.0)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "CONFLICT" in report
        assert "ORDER ALREADY SUBMITTED" not in report

    def test_no_orders_no_annotation(self):
        """Empty live_orders list → no order annotation anywhere in report."""
        item = _make_sell_item()
        item = self._confirmed(item)
        report = format_report([item], _make_portfolio(), live_orders=[])
        assert "ORDER ALREADY SUBMITTED" not in report
        assert "CONFLICT" not in report

    def test_order_for_different_ticker_not_shown(self):
        """Order with a different conid and non-matching symbol → not annotated."""
        item = _make_sell_item(ticker="9201.T")
        item = self._confirmed(item)
        # item.ibkr_position.conid == 99999; order has conid 11111 and ticker "XXXX"
        order = _make_order(conid=11111, ticker="XXXX", side="S", remaining_size=100)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "ORDER ALREADY SUBMITTED" not in report
        assert "CONFLICT" not in report

    def test_live_orders_none_no_annotation(self):
        """live_orders omitted (default None) → no annotation."""
        item = _make_sell_item()
        item = self._confirmed(item)
        report = format_report([item], _make_portfolio())
        assert "ORDER ALREADY SUBMITTED" not in report

    def test_partial_fill_sell_shows_remaining(self):
        """Same-side SELL order for fewer shares than recommended → PARTIAL ORDER note."""
        item = _make_sell_item(
            ticker="9201.T",
            sell_type="STOP_BREACH",
            reason="Stop breached: price 2700.00 < stop 2780.00",
            quantity=100,
            suggested_quantity=100,
        )
        item = self._confirmed(item)
        # Only 40 shares ordered, but 100 recommended
        order = _make_order(conid=99999, side="S", remaining_size=40, price=2780.0)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "PARTIAL ORDER" in report
        assert "40 of" in report
        assert "ORDER ALREADY SUBMITTED" not in report

    def test_full_fill_sell_shows_do_not_reenter(self):
        """Same-side SELL order covers full recommended quantity → do not re-enter note."""
        item = _make_sell_item(
            ticker="9201.T",
            sell_type="STOP_BREACH",
            reason="Stop breached: price 2700.00 < stop 2780.00",
            quantity=100,
            suggested_quantity=100,
        )
        item = self._confirmed(item)
        order = _make_order(conid=99999, side="S", remaining_size=100, price=2780.0)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "ORDER ALREADY SUBMITTED" in report
        assert "PARTIAL ORDER" not in report

    def test_hk_sell_matched_by_unpadded_ibkr_symbol(self):
        """Regression: HK yf ticker '0005.HK' has zero-padded base '0005', but IBKR
        live orders use unpadded '5'.  Symbol fallback must match both forms."""
        item = _make_sell_item(ticker="0005.HK", sell_type="HARD_REJECT")
        item = self._confirmed(item)
        # Simulate IBKR live order using its own unpadded symbol "5"
        order = _make_order(conid=99999, side="S", remaining_size=400, price=55.0)
        # conid=99999 matches → this works even without symbol fallback
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "ORDER ALREADY SUBMITTED" in report

    def test_hk_buy_matched_by_unpadded_ibkr_symbol(self):
        """New BUY for '0005.HK' (no position, no conid): IBKR live order has symbol '5'.
        Symbol fallback must strip leading zeros to match."""
        item = ReconciliationItem(
            ticker="0005.HK",
            action="BUY",
            urgency="LOW",
            reason="Watchlist BUY",
            ibkr_position=None,
            is_watchlist=True,
        )
        # IBKR order: bare "5", not "0005"
        order = _make_order(ticker="5", side="B", remaining_size=400, price=55.0)
        report = format_report(
            [item],
            _make_portfolio(),
            live_orders=[order],
            watchlist_name="TestWatchlist",
        )
        assert (
            "ORDER ALREADY SUBMITTED" in report
        ), "HK BUY order with IBKR symbol '5' should match yf ticker '0005.HK'"

    def test_ibkr_symbol_from_position_used_for_sell_match(self):
        """When a position exists, pos.symbol (IBKR-native) is the authoritative symbol
        candidate.  Even if the yf base and IBKR base differ, conid matching catches it."""
        # This test uses conid (most reliable path) — symbol is a secondary candidate
        pos = NormalizedPosition(
            conid=77777,
            ticker=Ticker.from_yf("0700.HK"),
            quantity=100,
            currency="HKD",
            current_price_local=34000,
            avg_cost_local=30000,
            market_value_usd=434000,
            ticker_identity_verified=True,
            ticker_resolution_source="exchange_map",
        )
        item = ReconciliationItem(
            ticker="0700.HK",
            action="SELL",
            urgency="HIGH",
            reason="DO_NOT_INITIATE",
            ibkr_position=pos,
            sell_type="HARD_REJECT",
        )
        item = self._confirmed(item)
        order = _make_order(conid=77777, side="S", remaining_size=100, price=34000.0)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "ORDER ALREADY SUBMITTED" in report

    def test_divergent_base_symbols_use_conid_not_symbol(self):
        """When IBKR and yFinance use different base symbols (rare), conid is the
        only reliable match.  Symbol fallback will miss — that is documented and acceptable."""
        pos = NormalizedPosition(
            conid=55555,
            ticker=Ticker.from_yf("MELI"),
            quantity=10,
            currency="USD",
            current_price_local=1500,
            avg_cost_local=1200,
            market_value_usd=15000,
            ticker_identity_verified=True,
            ticker_resolution_source="exchange_map",
        )
        item = ReconciliationItem(
            ticker="MELI",
            action="SELL",
            urgency="HIGH",
            reason="STOP_BREACH",
            ibkr_position=pos,
            sell_type="STOP_BREACH",
        )
        item = self._confirmed(item)
        order = _make_order(conid=55555, side="S", remaining_size=10, price=1490.0)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "ORDER ALREADY SUBMITTED" in report
