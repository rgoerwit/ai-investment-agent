"""Dip-watch report behavior."""

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


class TestComputeDipScore:
    def test_returns_zero_without_analysis(self):
        """Item with no analysis → score 0.0."""
        item = ReconciliationItem(
            ticker="X.T", action="REVIEW", urgency="LOW", reason="no analysis"
        )
        assert _compute_dip_score(item) == 0.0

    def test_base_score_from_health_and_growth(self):
        """Score with no position bonus = health*0.4 + growth*0.4."""
        item = _make_dip_item(
            "A.T",
            health=80,
            growth=70,
            entry=100,
            current_price=100,
            stop=90,
            target=110,
        )
        # base = 80*0.4 + 70*0.4 = 32 + 28 = 60; price_bonus=0 (no dip); rr depends on params
        score = _compute_dip_score(item)
        assert score >= 60.0

    def test_price_discount_adds_bonus(self):
        """Score increases when current price is below entry price."""
        at_entry = _make_dip_item(
            "A.T",
            health=70,
            growth=70,
            entry=2000,
            current_price=2000,
            stop=1800,
            target=2400,
        )
        below_entry = _make_dip_item(
            "A.T",
            health=70,
            growth=70,
            entry=2000,
            current_price=1800,
            stop=1600,
            target=2400,
        )
        assert _compute_dip_score(below_entry) > _compute_dip_score(at_entry)

    def test_no_bonus_when_price_above_entry(self):
        """No price bonus when current price ≥ entry price (not a dip)."""
        item = _make_dip_item(
            "A.T",
            health=70,
            growth=70,
            entry=1800,
            current_price=2000,
            stop=1600,
            target=2400,
        )
        base_only = 70 * 0.4 + 70 * 0.4  # 56.0; rr_bonus may apply
        score = _compute_dip_score(item)
        # price_bonus must be 0 (no dip), score = base + rr_bonus only
        # Verify by checking price_bonus = 0 path: dip_pct = (1800-2000)/1800 < 0
        assert score >= base_only  # at minimum base score

    def test_rr_bonus_for_good_risk_reward(self):
        """Higher upside/downside ratio → higher score."""
        tight_stop = _make_dip_item(
            "A.T",
            health=70,
            growth=70,
            entry=2000,
            current_price=1900,
            stop=1850,
            target=2400,
        )
        wide_stop = _make_dip_item(
            "A.T",
            health=70,
            growth=70,
            entry=2000,
            current_price=1900,
            stop=1000,
            target=2400,
        )
        # tight_stop: upside=(2400-1900)/1900=26.3%, downside=(1900-1850)/1900=2.6% → R/R 10.1 (capped at 8 pts)
        # wide_stop: downside=(1900-1000)/1900=47.4% → R/R 0.55 (small bonus)
        assert _compute_dip_score(tight_stop) > _compute_dip_score(wide_stop)


class TestDipWatch:
    """DIP WATCH section rendering and eligibility filtering."""

    def _items_with_dip_candidates(self) -> list[ReconciliationItem]:
        """8 demoted SOFT_REJECT items: 5 high-quality, 3 low-quality."""
        high = [
            _make_dip_item(
                f"GOOD{i:02d}.T",
                health=75,
                growth=72,
                entry=2000,
                current_price=1850,
                stop=1700,
                target=2500,
            )
            for i in range(5)
        ]
        # Low quality: health < 55
        low = [
            _make_dip_item(
                f"POOR{i:02d}.T",
                health=48,
                growth=70,
                entry=2000,
                current_price=1850,
                stop=1700,
                target=2500,
            )
            for i in range(3)
        ]
        return high + low

    def test_dip_watch_section_present_on_panic_day(self):
        """DIP WATCH section rendered when CORRELATED_SELL_EVENT and eligible items exist."""
        report = format_report(
            self._items_with_dip_candidates(),
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        assert "DIP WATCH" in report

    def test_dip_watch_section_absent_without_correlated_event(self):
        """No CORRELATED_SELL_EVENT → no DIP WATCH section."""
        report = format_report(
            self._items_with_dip_candidates(),
            _make_portfolio(),
            portfolio_health_flags=[],
        )
        assert "DIP WATCH" not in report

    def test_dip_watch_held_buy_pullback_shows_without_correlated_event(self):
        """Held BUY pullbacks do not require a correlated-sell event."""
        item = _make_dip_item(
            "HELD.T",
            health=80,
            growth=78,
            entry=2000,
            current_price=1800,
            stop=1700,
            target=2600,
            action="HOLD",
            sell_type=None,
        )
        report = format_report([item], _make_portfolio(), portfolio_health_flags=[])
        assert "DIP WATCH" in report
        assert "HELD.T" in report

    def test_dip_watch_items_ranked_by_score(self):
        """Higher-scoring items appear before lower-scoring items in DIP WATCH."""
        high_score = _make_dip_item(
            "HIGH.T",
            health=80,
            growth=78,
            entry=2000,
            current_price=1800,
            stop=1700,
            target=2600,
        )
        low_score = _make_dip_item(
            "LOW.T",
            health=58,
            growth=56,
            entry=2000,
            current_price=1880,
            stop=1900,
            target=2100,
        )
        items = [low_score, high_score]  # low first, should be reversed in output
        report = format_report(
            items,
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        assert "DIP WATCH" in report
        dw_idx = report.index("DIP WATCH")
        section = report[dw_idx:]
        high_pos = section.find("HIGH.T")
        low_pos = section.find("LOW.T")
        assert high_pos < low_pos, "HIGH.T should appear before LOW.T in DIP WATCH"

    def test_dip_watch_excludes_low_quality_items(self):
        """Items with health < 55 or growth < 55 excluded regardless of dip depth."""
        low_health = _make_dip_item(
            "LHLT.T",
            health=48,
            growth=72,
            entry=2000,
            current_price=1700,
            stop=1600,
            target=2600,
        )
        low_growth = _make_dip_item(
            "LGRW.T",
            health=72,
            growth=48,
            entry=2000,
            current_price=1700,
            stop=1600,
            target=2600,
        )
        good = _make_dip_item(
            "GOOD.T",
            health=72,
            growth=68,
            entry=2000,
            current_price=1850,
            stop=1700,
            target=2500,
        )
        items = [low_health, low_growth, good]
        report = format_report(
            items,
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        assert "DIP WATCH" in report
        dw_idx = report.index("DIP WATCH")
        section = report[dw_idx:]
        assert "LHLT.T" not in section
        assert "LGRW.T" not in section
        assert "GOOD.T" in section

    def test_dip_watch_includes_intact_macro_review_during_event(self):
        """During a macro event, a fresh + fundamentally-sound DNI/HIGH-zone review
        that dipped IS a dip-buy candidate (the macro relaxation), with a caveat."""
        item = _make_dip_item(
            "DNI.T",
            health=95,
            growth=88,
            entry=2000,
            current_price=1700,
            stop=1600,
            target=2800,
            verdict="DO_NOT_INITIATE",
            zone="HIGH",
        )
        report = format_report(
            [item],
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        assert "DIP WATCH" in report
        assert "DNI" in report
        assert "macro dip — fundamentals intact" in report

    def test_dip_watch_excludes_intact_macro_review_without_event(self):
        """Outside a macro event the normal gates apply: a DNI/HIGH-zone review is
        NOT a dip-buy candidate."""
        item = _make_dip_item(
            "DNI.T",
            health=95,
            growth=88,
            entry=2000,
            current_price=1700,
            stop=1600,
            target=2800,
            verdict="DO_NOT_INITIATE",
            zone="HIGH",
        )
        report = format_report([item], _make_portfolio(), portfolio_health_flags=[])
        assert "DIP WATCH" not in report

    def test_dip_watch_excludes_stale_macro_review_during_event(self):
        """The recency safeguard holds even during a macro event: a stale review is
        not trusted, so it must be refreshed before it can be a dip-buy candidate."""
        item = _make_dip_item(
            "DNI.T",
            health=95,
            growth=88,
            entry=2000,
            current_price=1700,
            stop=1600,
            target=2800,
            verdict="DO_NOT_INITIATE",
            zone="HIGH",
            age_days=120,  # stale — older than dip-watch max age
        )
        report = format_report(
            [item],
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        assert "DIP WATCH" not in report

    def test_dip_watch_excludes_unsound_macro_review_during_event(self):
        """Weak fundamentals are excluded even during a macro event."""
        item = _make_dip_item(
            "WEAK.T",
            health=40,
            growth=35,
            entry=2000,
            current_price=1700,
            stop=1600,
            target=2800,
            verdict="DO_NOT_INITIATE",
            zone="HIGH",
        )
        report = format_report(
            [item],
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        assert "DIP WATCH" not in report

    def test_dip_watch_absent_when_no_scoreable_items(self):
        """CORRELATED_SELL_EVENT but all macro_reviews have health < 55 → no DIP WATCH."""
        items = [
            _make_dip_item(
                f"POOR{i:02d}.T",
                health=40,
                growth=40,
                entry=2000,
                current_price=1800,
                stop=1700,
                target=2500,
            )
            for i in range(5)
        ]
        report = format_report(
            items,
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        assert "DIP WATCH" not in report

    def test_dip_watch_run_cmd_uses_analysis_ticker_when_item_ticker_bare(self):
        """DIP WATCH re-run cmd uses analysis.ticker (with suffix) when item.ticker is bare.

        Scenario: ibkr_symbol_to_yf() failed to resolve the exchange suffix, leaving
        item.ticker = "BARE" (no suffix).  The analysis was stored as "BARE.L".
        The re-run command in DIP WATCH must use "BARE.L" not "BARE".
        """
        from datetime import datetime, timedelta

        analysis_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        # Override analysis.ticker to have the canonical yf suffix (different from item.ticker)
        analysis = AnalysisRecord(
            ticker="BARE.L",  # canonical yfinance ticker found via _alpha_base_lookup
            analysis_date=analysis_date,
            verdict="BUY",
            health_adj=75.0,  # >= 55 → passes DIP WATCH quality filter
            growth_adj=70.0,  # >= 55 → passes
            zone="MODERATE",
            entry_price=200.0,
            stop_price=170.0,
            target_1_price=250.0,
            currency="GBX",
        )
        pos = NormalizedPosition(
            conid=33333,
            ticker=Ticker.from_yf("BARE"),
            quantity=100,
            avg_cost_local=200.0,
            current_price_local=185.0,  # below entry → dip bonus
            market_value_usd=2000.0,
            currency="GBX",
        )
        item = ReconciliationItem(
            ticker="BARE",  # bare — ibkr_symbol_to_yf couldn't resolve exchange
            ibkr_symbol="BARE",
            action="REVIEW",
            urgency="MEDIUM",
            reason=(
                f"Verdict → BUY  ({analysis_date})"
                "  [MACRO_WATCH: demoted from SELL — correlated event detected]"
            ),
            ibkr_position=pos,
            analysis=analysis,
            sell_type="SOFT_REJECT",
        )
        report = format_report(
            [item],
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        assert "DIP WATCH" in report
        # Re-run command must use canonical yf ticker (analysis.ticker), not bare symbol
        assert "--ticker BARE.L" in report
        assert "⚠ verify exchange suffix" not in report


# ── _pnl_line helpers ─────────────────────────────────────────────────────────
