"""Concentration-aware portfolio report behavior."""

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


class TestWatchlistConcentration:
    """Concentration screen rendering: removals carry the overweight reason,
    hatch admits are flagged with the tier that applied, withheld off-watch
    names get a footer, and screened names never reach cash/order surfaces."""

    def test_screened_incumbent_optional_remove_and_excluded_from_cash_and_today(
        self,
    ):
        screened = _make_buy_item("7203.T", conviction="Medium")
        kept = _make_buy_item("MEGP.L", conviction="Medium")
        portfolio = _make_portfolio()
        portfolio.exchange_weights = {"T": 45.0}
        report = format_report(
            [screened, kept],
            portfolio,
            show_recommendations=True,
            watchlist_total=2,
            watchlist_tickers={"7203.T", "MEGP.L"},
        )

        assert "− REMOVE FROM WATCHLIST  7203 (7203.T)" in report
        assert "overweight exchange T (projected 49.0% > 40%)" in report
        assert "below retention bar" in report
        assert "Exact replacement list withheld — decide optional removals" in report
        cash_block = report.split("CASH SUMMARY")[1].split("ACTION PLAN")[0]
        today_block = report.split("ACTION PLAN")[1]
        assert "BUY  MEGP  (100 sh):" in cash_block
        assert "→ BUY  MEGP" in today_block
        assert "7203" not in cash_block
        assert "7203" not in today_block

    def test_incumbent_hatch_admit_renders_incumbent_tier_and_keeps(self):
        # Default _make_buy_item analysis: High conviction, 72+65=137 ≥ 135.
        item = _make_buy_item("7203.T")
        portfolio = _make_portfolio()
        portfolio.exchange_weights = {"T": 45.0}
        report = format_report(
            [item],
            portfolio,
            show_recommendations=True,
            watchlist_total=1,
            watchlist_tickers={"7203.T"},
        )

        assert "⚠ over-limit admit  7203 (7203.T)" in report
        assert "score 137/200 ≥ 135 (incumbent)" in report
        assert "KEEPING ACTIVE (1): 7203 (7203.T)" in report
        assert "[Update IBKR watchlist to]: 7203" in report

    def test_newcomer_hatch_admit_renders_newcomer_tier(self):
        item = _make_offwatch_buy("WDO.TO")
        item.analysis.health_adj = 80.0
        item.analysis.growth_adj = 75.0
        portfolio = _make_portfolio()
        portfolio.exchange_weights = {"TO": 45.0}
        report = format_report(
            [item],
            portfolio,
            show_recommendations=True,
            watchlist_total=0,
        )

        assert "⚠ over-limit admit  WDO (WDO.TO)" in report
        assert "score 155/200 ≥ 150" in report
        assert "(incumbent)" not in report

    def test_sector_breach_renders_sector_dimension(self):
        item = _make_buy_item("7203.T", conviction="Medium")
        item.analysis.sector = "Industrials"
        portfolio = _make_portfolio()
        portfolio.sector_weights = {"Industrials": 45.0}
        report = format_report(
            [item],
            portfolio,
            show_recommendations=True,
            watchlist_total=1,
            watchlist_tickers={"7203.T"},
        )

        assert "overweight sector Industrials (projected 49.0% > 30%)" in report

    def test_offwatch_withheld_footer_lists_breach(self):
        item = _make_offwatch_buy("WDO.TO")  # High conviction, 132 < 150
        portfolio = _make_portfolio()
        portfolio.exchange_weights = {"TO": 45.0}
        report = format_report(
            [item],
            portfolio,
            show_recommendations=True,
            watchlist_total=0,
        )

        assert "Withheld by concentration (1):" in report
        assert "WDO (WDO.TO) [exchange TO 48.0% > 40%]" in report
        assert "+ ADD" not in report

    def test_empty_weights_render_no_concentration_lines(self):
        report = format_report(
            [_make_buy_item("7203.T", conviction="Medium")],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_total=1,
            watchlist_tickers={"7203.T"},
        )

        assert "overweight" not in report
        assert "Withheld by concentration" not in report
        assert "over-limit admit" not in report

    def test_read_only_run_emits_no_concentration_lines(self):
        report = format_report(
            [_make_buy_item("7203.T", conviction="Medium", is_watchlist=False)],
            _make_portfolio(),
            show_recommendations=True,
            portfolio_data_loaded=False,
        )

        assert "overweight" not in report
        assert "Withheld by concentration" not in report

    def test_dip_screen_withholds_sub_star3_and_keeps_star3(self):
        """End-to-end dip check: a sub-★★★ dip in the overweight bucket leaves
        DIP WATCH (with a transparency note); a ★★★ dip in the same bucket
        stays."""
        withheld = _make_dip_item(
            "9201.T",
            health=60,
            growth=60,
            entry=2100,
            current_price=1800,
            stop=1700,
            target=2600,
            action="HOLD",
            sell_type=None,
        )
        kept = _make_dip_item(
            "6758.T",
            health=75,
            growth=72,
            entry=2100,
            current_price=1800,
            stop=1700,
            target=2600,
            action="HOLD",
            sell_type=None,
        )
        portfolio = _make_portfolio()
        portfolio.exchange_weights = {"T": 45.0}
        report = format_report([withheld, kept], portfolio)

        assert "DIP WATCH" in report
        # The section title sits between its own dividers; rows follow the
        # trailing divider and precede the transparency note (which names 9201).
        dip_block = report.split("DIP WATCH")[1].split("═" * 54)[1]
        dip_rows = dip_block.split("(1 dip candidate")[0]
        assert "6758" in dip_rows
        assert "9201" not in dip_rows
        assert (
            "(1 dip candidate withheld — overweight bucket, below ★★★: 9201)" in report
        )

    def test_dip_screen_inactive_without_weights(self):
        item = _make_dip_item(
            "9201.T",
            health=60,
            growth=60,
            entry=2100,
            current_price=1800,
            stop=1700,
            target=2600,
            action="HOLD",
            sell_type=None,
        )
        report = format_report([item], _make_portfolio())

        assert "9201" in report.split("DIP WATCH")[1].split("═" * 54)[1]
        assert "withheld — overweight bucket" not in report
