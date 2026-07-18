"""Candidate display, ticker identity, and output-tightening report behavior."""

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
    _make_sell_item_with_analysis,
    _make_watchlist_review,
    _panic_items,
)
from tests.ibkr.reconciler_cases import (
    _make_analysis,
    _make_portfolio,
    _make_position,
)


class TestIbkrDisplaySymbol:
    """format_report() shows ibkr_symbol in human-visible sections, yf ticker in run commands."""

    def _held_hold(self, yf_ticker: str, ibkr_symbol: str) -> ReconciliationItem:
        pos = _make_position(ticker=yf_ticker, current_price=2100)
        return ReconciliationItem(
            ticker=yf_ticker,
            ibkr_symbol=ibkr_symbol,
            action="HOLD",
            reason="Position OK",
            urgency="LOW",
            ibkr_position=pos,
        )

    def _held_review(self, yf_ticker: str, ibkr_symbol: str) -> ReconciliationItem:
        pos = _make_position(ticker=yf_ticker, current_price=2100)
        return ReconciliationItem(
            ticker=yf_ticker,
            ibkr_symbol=ibkr_symbol,
            action="REVIEW",
            reason="Stale analysis: age 20d > 14d limit",
            urgency="MEDIUM",
            ibkr_position=pos,
        )

    def test_holds_section_shows_ibkr_symbol(self):
        """HOLDS section displays the IBKR symbol, not the yfinance ticker."""
        item = self._held_hold("7203.T", "7203")
        report = format_report([item], _make_portfolio())
        assert "7203   " in report or "7203  " in report  # displayed
        assert "7203.T" not in report.split("HOLDS")[1].split("REVIEWS")[0]

    def test_holds_section_hk_symbol_no_zero_pad(self):
        """HK positions display IBKR symbol '5' not yfinance '0005.HK'."""
        item = self._held_hold("0005.HK", "5")
        report = format_report([item], _make_portfolio())
        holds_block = report.split("HOLDS")[1] if "HOLDS" in report else report
        assert "5     " in holds_block or "5  " in holds_block  # IBKR symbol

    def test_holds_section_korean_symbol_keeps_fixed_width(self):
        """Korean positions display IBKR fixed-width symbol, not stripped base."""
        item = self._held_hold("010130.KS", "010130")
        report = format_report([item], _make_portfolio())
        holds_block = report.split("HOLDS")[1] if "HOLDS" in report else report
        assert "010130" in holds_block
        assert "10130" not in holds_block.split("010130", 1)[0]

    def test_reviews_run_cmd_uses_yf_ticker(self):
        """REVIEWS run command must use yf ticker (with exchange suffix) for --ticker arg."""
        item = self._held_review("7203.T", "7203")
        report = format_report([item], _make_portfolio())
        # Run command in REVIEWS should reference yf ticker
        assert "--ticker 7203.T" in report

    def test_reviews_display_uses_ibkr_symbol(self):
        """REVIEWS label shows IBKR symbol, not yfinance ticker."""
        item = self._held_review("7203.T", "7203")
        report = format_report([item], _make_portfolio())
        reviews_block = report.split("REVIEWS")[1] if "REVIEWS" in report else ""
        # Display part (before the run cmd) uses ibkr symbol
        assert "REVIEW" in reviews_block
        # Ensure the display portion shows "7203" not "7203.T"
        # (the run cmd has "--ticker 7203.T", the label has "7203 ")
        assert "7203  " in reviews_block or "7203 " in reviews_block

    def test_new_buy_without_analysis_is_excluded(self):
        """A missing analysis cannot be rendered as a BUY-ready watchlist row."""
        item = ReconciliationItem(
            ticker="CAG.ST",
            action="BUY",
            reason="Watchlist BUY",
            urgency="MEDIUM",
            is_watchlist=True,
        )
        report = format_report([item], _make_portfolio())
        assert "Excluded below medium conviction: 1" in report

    def test_reviews_run_cmd_uses_analysis_ticker_when_item_ticker_bare(self):
        """When item.ticker has no suffix but analysis.ticker has one, REVIEWS run cmd uses analysis.ticker.

        Scenario: ibkr_symbol_to_yf() couldn't find the IBKR exchange code, so
        pos.yf_ticker = "MEGP" (bare).  _alpha_base_lookup found the analysis stored
        as "MEGP.L", so item.analysis.ticker = "MEGP.L".  The run command must use
        the canonical yfinance ticker, not the bare IBKR-derived one.
        """
        from datetime import datetime, timedelta

        pos = NormalizedPosition(
            conid=22222,
            ticker=Ticker.from_ibkr("MEGP", currency="GBX"),
            quantity=100,
            avg_cost_local=100.0,
            current_price_local=95.0,
            market_value_usd=13000.0,
            currency="GBX",
        )
        item = ReconciliationItem(
            ticker="MEGP",  # bare — ibkr_symbol_to_yf couldn't resolve exchange
            action="REVIEW",
            reason="Stale analysis: age 20d > 14d limit",
            urgency="MEDIUM",
            ibkr_position=pos,
            analysis=AnalysisRecord(
                ticker="MEGP.L",  # canonical yfinance format from analysis file
                analysis_date=(datetime.now() - timedelta(days=20)).strftime(
                    "%Y-%m-%d"
                ),
                verdict="BUY",
                health_adj=72.0,
                growth_adj=65.0,
            ),
        )
        report = format_report([item], _make_portfolio())
        # Run command must reference the canonical yfinance ticker, not the bare symbol
        assert "--ticker MEGP.L" in report
        assert "⚠ verify exchange suffix" not in report

    def test_holds_no_suffix_warning_regardless_of_currency(self):
        """HOLDS section never shows an exchange-suffix warning — IBKR tickers don't have suffixes."""
        for currency in ("GBX", "EUR", "JPY"):
            pos = NormalizedPosition(
                conid=44444,
                ticker=Ticker.from_ibkr("MEGP", currency=currency),
                quantity=200,
                avg_cost_local=100.0,
                current_price_local=95.0,
                market_value_usd=13000.0,
                currency=currency,
            )
            item = ReconciliationItem(
                ticker="MEGP",
                action="HOLD",
                reason="Position OK",
                urgency="LOW",
                ibkr_position=pos,
            )
            report = format_report([item], _make_portfolio())
            assert (
                "exchange" not in report.lower() or "exchange" not in report
            ), f"Unexpected exchange warning in HOLDS for currency={currency}"

    def test_review_suffix_warning_when_ticker_bare(self):
        """REVIEWS run command shows suffix warning when the ticker has no exchange suffix."""
        item = ReconciliationItem(
            ticker="CEK",  # no suffix — exchange unknown
            action="REVIEW",
            reason="Stale analysis: age 20d > 14d limit",
            urgency="MEDIUM",
        )
        report = format_report([item], _make_portfolio())
        # Warning appears in the run command, not on the display ticker line
        assert "exchange unknown" in report
        assert "--ticker CEK" in report

    def test_review_no_suffix_warning_when_ticker_has_suffix(self):
        """REVIEWS run command omits the suffix warning when exchange is known."""
        item = ReconciliationItem(
            ticker="CEK.DE",
            action="REVIEW",
            reason="Stale analysis: age 20d > 14d limit",
            urgency="MEDIUM",
        )
        report = format_report([item], _make_portfolio())
        assert "exchange unknown" not in report
        assert "--ticker CEK.DE" in report


class TestWatchlistCandidatesInFlight:
    """The unified optimizer excludes off-watchlist items with live BUY orders."""

    def _report(self, items, live_orders=None) -> str:
        return format_report(
            items,
            _make_portfolio(),
            show_recommendations=True,
            live_orders=live_orders or [],
        )

    def test_candidate_with_live_buy_order_hidden_from_section(self):
        """An off-watchlist BUY with an open order is not a new addition."""
        item = _make_offwatch_buy("WDO.TO")
        live_order = {"ticker": "WDO", "side": "B", "remainingSize": "100"}
        report = self._report([item], live_orders=[live_order])
        assert "WATCHLIST OPTIMIZATION" in report
        assert "already in flight" in report
        assert "WDO" in report
        assert not any(
            line.startswith("  + ADD") and "WDO" in line for line in report.splitlines()
        )

    def test_candidate_without_live_order_shown_normally(self):
        """Off-watchlist BUY with no live order appears as an addition."""
        item = _make_offwatch_buy("WDO.TO")
        report = self._report([item], live_orders=[])
        assert "WATCHLIST OPTIMIZATION" in report
        assert "WDO" in report
        assert "+ ADD" in report

    def test_empty_section_shown_when_cash_policy_blocked_candidates(self):
        report = format_report(
            [],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_candidates_blocked_by_cash=2,
        )

        assert "WATCHLIST OPTIMIZATION" in report
        assert "Cash-blocked candidates retained for ranking: 2" in report

    def test_in_flight_candidate_excluded_from_watchlist_moves(self):
        """The retired WATCHLIST MOVES block cannot duplicate an in-flight candidate."""
        item = _make_offwatch_buy("WDO.TO", conviction="High")
        live_order = {"ticker": "WDO", "side": "B", "remainingSize": "50"}
        report = self._report([item], live_orders=[live_order])
        assert "WATCHLIST MOVES" not in report
        assert not any(
            line.startswith("  + ADD") and "WDO" in line for line in report.splitlines()
        )

    def test_two_candidates_one_in_flight_other_shown(self):
        """When one candidate is in-flight and another is not, only the latter appears."""
        inflight = _make_offwatch_buy("WDO.TO", conviction="High")
        pending = _make_offwatch_buy("TOTL.TO", conviction="Medium")
        live_order = {"ticker": "WDO", "side": "B", "remainingSize": "100"}
        report = self._report([inflight, pending], live_orders=[live_order])
        assert "WATCHLIST OPTIMIZATION" in report
        assert "TOTL" in report
        assert any(
            line.startswith("  + ADD") and "TOTL" in line
            for line in report.splitlines()
        )
        assert not any(
            line.startswith("  + ADD") and "WDO" in line for line in report.splitlines()
        )
        assert "already in flight" in report


class TestExchangeQualifiedCandidateSafety:
    """Exchange-qualified candidates are never suppressed by a base-symbol match."""

    def _make_sell_item(
        self, ticker: str, sell_type: str | None = "HARD_REJECT"
    ) -> ReconciliationItem:
        pos = _make_position(ticker=ticker, current_price=35.50)
        return ReconciliationItem(
            ticker=ticker,
            action="SELL",
            reason="Verdict → DO_NOT_INITIATE",
            urgency="HIGH",
            ibkr_position=pos,
            sell_type=sell_type,
            suggested_quantity=20,
            suggested_price=35.50,
        )

    def test_sell_does_not_block_different_exchange_same_base_candidate(self):
        """A bare DLG sell must not suppress an exchange-qualified DLG.MI listing."""
        sell = self._make_sell_item("DLG")
        buy_cand = _make_offwatch_buy("DLG.MI")
        report = format_report(
            [sell, buy_cand], _make_portfolio(), show_recommendations=True
        )
        assert "SELL" in report
        assert "WATCHLIST OPTIMIZATION" in report
        assert "DLG" in report
        assert "+ ADD" in report

    def test_stop_breach_does_not_block_different_exchange_same_base_candidate(self):
        """Stop handling remains scoped to the exact exchange-qualified security."""
        sell = self._make_sell_item("DLG", sell_type="STOP_BREACH")
        buy_cand = _make_offwatch_buy("DLG.MI")
        report = format_report(
            [sell, buy_cand], _make_portfolio(), show_recommendations=True
        )
        assert "WATCHLIST OPTIMIZATION" in report
        assert "+ ADD" in report

    def test_different_base_candidate_not_blocked(self):
        """SELL DLG does not suppress a candidate with a different base symbol."""
        sell = self._make_sell_item("DLG")
        buy_cand = _make_offwatch_buy("WDO.TO")
        report = format_report(
            [sell, buy_cand], _make_portfolio(), show_recommendations=True
        )
        assert "WATCHLIST OPTIMIZATION" in report
        assert "WDO" in report


class TestWatchlistTickerIdentity:
    """The optimizer only deduplicates exact exchange-qualified identities."""

    def test_suffixed_watchlist_ticker_keeps_same_candidate(self):
        """An exact watchlist ticker becomes KEEP, not a duplicate addition."""
        buy_cand = _make_offwatch_buy("5434.TW")
        report = format_report(
            [buy_cand],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers={"5434.TW"},
            watchlist_total=1,
        )
        assert "KEEPING ACTIVE" in report
        assert "+ ADD" not in report

    def test_bare_watchlist_ticker_protects_itself_but_not_a_suffixed_listing(self):
        """A failed bare resolution cannot be assumed to equal 5434.TW."""
        buy_cand = _make_offwatch_buy("5434.TW")
        report = format_report(
            [buy_cand],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers={"5434"},
            watchlist_total=1,
        )
        assert "KEEPING PROTECTED" in report
        assert "+ ADD" in report

    def test_none_watchlist_does_not_suppress(self):
        """watchlist_tickers=None → no watchlist filter applied."""
        buy_cand = _make_offwatch_buy("5434.TW")
        report = format_report(
            [buy_cand],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers=None,
        )
        assert "WATCHLIST OPTIMIZATION" in report
        assert "5434" in report

    def test_different_base_not_blocked(self):
        """watchlist_tickers={'5434'} does not suppress a different base candidate."""
        buy_cand = _make_offwatch_buy("WDO.TO")
        report = format_report(
            [buy_cand],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers={"5434"},
        )
        assert "WATCHLIST OPTIMIZATION" in report
        assert "WDO" in report


class TestPortfolioManagerOutputTightening:
    def test_buy_without_quantity_is_labeled_as_incomplete(self):
        item = _make_buy_item(
            suggested_quantity=None,
            suggested_price=2615.0,
            cash_impact_usd=0.0,
        )
        report = format_report([item], _make_portfolio(), show_recommendations=True)
        assert "quantity unavailable — inspect before placing order" in report

    def test_macro_review_labels_analysis_entry_and_ibkr_cost_basis(self):
        item = _make_dip_item(
            ticker="9201.T",
            health=75,
            growth=68,
            entry=2800,
            current_price=2700,
            stop=2600,
            target=3200,
        )
        report = format_report(
            [item],
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        assert "thesis:    entry JPY 2,800.00 -> now JPY 2,700.00" in report
        assert "P/L:" in report
        assert "vs JPY 2,000.00" in report

    def test_soft_rejection_detail_lines_stay_readable_width(self):
        item = _make_dip_item(
            ticker="9201.T",
            health=95,
            growth=88,
            entry=2800,
            current_price=2700,
            stop=2600,
            target=3600,
        )
        item.suggested_quantity = 350
        item.suggested_price = 2700.0
        item.suggested_order_type = "LMT"
        item.cash_impact_usd = 1907.0
        item.settlement_date = "2026-03-17"

        report = format_report(
            [item],
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
            show_recommendations=True,
        )

        in_soft_section = False
        for line in report.splitlines():
            if "SELL RECOMMENDATIONS" in line:
                in_soft_section = True
                continue
            if in_soft_section and line.startswith("═" * 54):
                break
            if in_soft_section and line.startswith("             "):
                assert len(line) <= 96

    def test_concentration_merges_healthcare_labels(self):
        portfolio = _make_portfolio()
        portfolio.sector_weights = {
            "Healthcare": 10.1,
            "Health Care": 1.5,
            "Industrials": 21.7,
        }
        report = format_report([], portfolio)
        assert "Healthcare" not in report
        assert "Health Care" in report
        assert "11.6%" in report

    def test_concentration_merges_multiple_legacy_sector_labels(self):
        portfolio = _make_portfolio()
        portfolio.sector_weights = {
            "Technology": 8.0,
            "Information Technology": 1.5,
            "Basic Materials": 4.0,
            "Materials": 2.0,
            "Consumer Cyclical": 3.0,
            "Consumer Discretionary": 1.0,
            "Consumer Defensive": 2.5,
            "Consumer Staples": 0.5,
        }
        report = format_report([], portfolio)
        assert "  Technology              " not in report
        assert "  Basic Materials         " not in report
        assert "  Consumer Cyclical       " not in report
        assert "  Consumer Defensive      " not in report
        assert "Information Technology" in report
        assert "Materials" in report
        assert "Consumer Discretionary" in report
        assert "Consumer Staples" in report
        assert "9.5%" in report
        assert "6.0%" in report
        assert "4.0%" in report
        assert "3.0%" in report

    def test_watchlist_moves_are_advisory_not_past_tense(self):
        high = _make_offwatch_buy("TOTL.JK", conviction="High")
        report = format_report([high], _make_portfolio(), show_recommendations=True)
        assert "WATCHLIST OPTIMIZATION" in report
        assert any(
            line.startswith("  + ADD") and "TOTL" in line
            for line in report.splitlines()
        )
        assert "ADDED TO WATCHLIST" not in report
