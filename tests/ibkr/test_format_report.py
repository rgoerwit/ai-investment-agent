"""Tests for format_report() output formatting — macro-panic day and normal day."""

from __future__ import annotations

from scripts.portfolio_manager import _compute_dip_score, format_report
from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    ReconciliationItem,
    TradeBlockData,
)
from tests.ibkr.test_reconciler import _make_portfolio, _make_position

# A realistic CORRELATED_SELL_EVENT flag string (matches what compute_portfolio_health emits)
_CORR_FLAG = (
    "CORRELATED_SELL_EVENT: 8 positions changed verdict on 2026-03-05"
    " (80% of held positions) — probable macro event."
    " Execute stop-breach SELLs only; review verdict-change SELLs before acting."
)


def _panic_items() -> list[ReconciliationItem]:
    """
    Pre-demoted panic-day items for format_report isolation tests.

    8 SOFT_REJECT items already demoted to REVIEW (as compute_portfolio_health would do),
    plus 1 STOP_BREACH and 1 HARD_REJECT that remain as SELL.
    """
    pos = _make_position(current_price=2100)
    items: list[ReconciliationItem] = []

    for i in range(8):
        items.append(
            ReconciliationItem(
                ticker=f"SOFT{i:02d}.T",
                action="REVIEW",
                urgency="MEDIUM",
                reason=(
                    "Verdict → DO_NOT_INITIATE  (2026-03-05)"
                    "  [MACRO_WATCH: demoted from SELL — correlated event detected]"
                ),
                ibkr_position=pos,
                sell_type="SOFT_REJECT",
            )
        )

    items.append(
        ReconciliationItem(
            ticker="STOP.T",
            action="SELL",
            urgency="HIGH",
            reason="Stop breached: price 1700.00 < stop 1900.00",
            ibkr_position=pos,
            sell_type="STOP_BREACH",
        )
    )
    items.append(
        ReconciliationItem(
            ticker="HARD.T",
            action="SELL",
            urgency="HIGH",
            reason="Verdict → DO_NOT_INITIATE  (2026-03-05)",
            ibkr_position=pos,
            sell_type="HARD_REJECT",
        )
    )
    return items


class TestFormatReportPanicDay:
    """format_report() on a panic day — CORRELATED_SELL_EVENT flag set, SOFT_REJECTs demoted."""

    def _report_with_flag(self) -> str:
        return format_report(
            _panic_items(),
            _make_portfolio(),
            portfolio_health_flags=[_CORR_FLAG],
        )

    def _report_without_flag(self) -> str:
        return format_report(
            _panic_items(),
            _make_portfolio(),
            portfolio_health_flags=[],
        )

    def test_macro_banner_present_when_flag_set(self):
        """CORRELATED_SELL_EVENT in health_flags → MACRO ALERT banner rendered."""
        assert "MACRO ALERT" in self._report_with_flag()

    def test_macro_banner_absent_when_no_flag(self):
        """Empty health_flags → no MACRO ALERT banner, even if items are present."""
        assert "MACRO ALERT" not in self._report_without_flag()

    def test_stop_breached_section_present(self):
        """STOP_BREACH item → SELLS — STOP BREACHED section always rendered."""
        assert "SELLS — STOP BREACHED" in self._report_with_flag()

    def test_fundamental_failure_section_present(self):
        """HARD_REJECT item → SELLS — FUNDAMENTAL FAILURE section rendered."""
        assert "SELLS — FUNDAMENTAL FAILURE" in self._report_with_flag()

    def test_soft_rejection_section_present(self):
        """Demoted macro_reviews → SELLS — SOFT REJECTION section rendered."""
        assert "SELLS — SOFT REJECTION" in self._report_with_flag()

    def test_soft_rejection_section_shows_demoted_items(self):
        """Demoted items (action=REVIEW, sell_type=SOFT_REJECT) listed in SOFT REJECTION."""
        report = self._report_with_flag()
        lines = report.split("\n")
        soft_rej_idx = next(
            (i for i, ln in enumerate(lines) if "SELLS — SOFT REJECTION" in ln), None
        )
        assert soft_rej_idx is not None, "SELLS — SOFT REJECTION section missing"
        section_content = "\n".join(lines[soft_rej_idx:])
        assert "SOFT00.T" in section_content

    def test_reviews_section_excludes_soft_reject_items(self):
        """SOFT00.T appears only in SOFT REJECTION — not in the regular REVIEWS section.

        The regular REVIEWS section uses 'python -m src.main' command suggestions.
        SOFT REJECTION section does not. Checking for this distinguishes the two.
        """
        report = self._report_with_flag()
        lines = report.split("\n")
        soft00_in_reviews_format = any(
            "SOFT00.T" in line and "python -m src.main" in line for line in lines
        )
        assert not soft00_in_reviews_format

    def test_deferred_actions_excludes_demoted_items(self):
        """TODAY action list only shows SELL/TRIM items — demoted REVIEW items excluded."""
        report = self._report_with_flag()
        # TODAY action lines have the form '    → ACTION  TICKER...'
        today_action_lines = [
            ln for ln in report.split("\n") if ln.startswith("    → ")
        ]
        assert not any("SOFT00.T" in ln for ln in today_action_lines)

    def test_summary_line_counts_macro_watch_not_review_for_demoted(self):
        """Summary shows 8 MACRO_WATCH (not 8 REVIEW/SELL) — demoted items are counted separately."""
        report = self._report_with_flag()
        summary_line = next(
            (ln for ln in report.split("\n") if ln.strip().startswith("Summary:")), ""
        )
        assert summary_line, "Summary line missing from report"
        assert "8 MACRO_WATCH" in summary_line
        assert "8 SELL" not in summary_line
        assert "8 REVIEW" not in summary_line


class TestFormatReportNormalDay:
    """format_report() on a normal day — no correlated event, standard item rendering."""

    def test_no_banner_on_normal_day(self):
        """No health flags → no MACRO ALERT banner."""
        pos = _make_position(current_price=2100)
        items = [
            ReconciliationItem(
                ticker="7203.T",
                action="HOLD",
                urgency="LOW",
                reason="Position OK",
                ibkr_position=pos,
            )
        ]
        report = format_report(items, _make_portfolio(), portfolio_health_flags=[])
        assert "MACRO ALERT" not in report

    def test_soft_sell_on_normal_day_is_sell_not_review(self):
        """SOFT_REJECT with action=SELL (not demoted) shows [SELL] in SOFT REJECTION section."""
        pos = _make_position(current_price=2100)
        items = [
            ReconciliationItem(
                ticker="7203.T",
                action="SELL",
                urgency="HIGH",
                reason="Verdict → DO_NOT_INITIATE  (2026-03-05)",
                ibkr_position=pos,
                sell_type="SOFT_REJECT",
            )
        ]
        report = format_report(items, _make_portfolio(), portfolio_health_flags=[])
        assert "SELLS — SOFT REJECTION" in report
        lines = report.split("\n")
        soft_rej_idx = next(
            (i for i, ln in enumerate(lines) if "SELLS — SOFT REJECTION" in ln), None
        )
        assert soft_rej_idx is not None
        # Check the next ~15 lines for the SELL action label and the ticker
        section_lines = lines[soft_rej_idx : soft_rej_idx + 15]
        assert any("SELL" in ln and "7203.T" in ln for ln in section_lines)

    def test_no_soft_rejection_section_when_no_soft_items(self):
        """Only STOP_BREACH items → SOFT REJECTION section not rendered."""
        pos = _make_position(current_price=1700)
        items = [
            ReconciliationItem(
                ticker="STOP.T",
                action="SELL",
                urgency="HIGH",
                reason="Stop breached: price 1700.00 < stop 1900.00",
                ibkr_position=pos,
                sell_type="STOP_BREACH",
            )
        ]
        report = format_report(items, _make_portfolio(), portfolio_health_flags=[])
        assert "SELLS — SOFT REJECTION" not in report

    def test_no_sell_sections_when_no_sells(self):
        """All HOLD items → no SELL section headings in output."""
        pos = _make_position(current_price=2100)
        items = [
            ReconciliationItem(
                ticker="7203.T",
                action="HOLD",
                urgency="LOW",
                reason="Position OK",
                ibkr_position=pos,
            )
        ]
        report = format_report(items, _make_portfolio(), portfolio_health_flags=[])
        assert "SELLS —" not in report


# ── DIP WATCH helpers ────────────────────────────────────────────────────────


def _make_dip_item(
    ticker: str,
    health: float,
    growth: float,
    entry: float,
    current_price: float,
    stop: float,
    target: float,
    currency: str = "JPY",
) -> ReconciliationItem:
    """Create a demoted SOFT_REJECT REVIEW item with a full analysis record."""
    from datetime import datetime, timedelta

    analysis_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    analysis = AnalysisRecord(
        ticker=ticker,
        analysis_date=analysis_date,
        verdict="DO_NOT_INITIATE",
        health_adj=health,
        growth_adj=growth,
        zone="MODERATE",
        entry_price=entry,
        stop_price=stop,
        target_1_price=target,
        currency=currency,
        trade_block=TradeBlockData(
            action="DO_NOT_INITIATE",
            entry_price=entry,
            stop_price=stop,
            target_1_price=target,
        ),
    )
    pos = _make_position(
        ticker=ticker,
        current_price=current_price,
        currency=currency,
    )
    return ReconciliationItem(
        ticker=ticker,
        action="REVIEW",
        urgency="MEDIUM",
        reason=(
            f"Verdict → DO_NOT_INITIATE  ({analysis_date})"
            "  [MACRO_WATCH: demoted from SELL — correlated event detected]"
        ),
        ibkr_position=pos,
        analysis=analysis,
        sell_type="SOFT_REJECT",
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
            portfolio_health_flags=[_CORR_FLAG],
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
            current_price=1980,
            stop=1900,
            target=2100,
        )
        items = [low_score, high_score]  # low first, should be reversed in output
        report = format_report(
            items,
            _make_portfolio(),
            portfolio_health_flags=[_CORR_FLAG],
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
            portfolio_health_flags=[_CORR_FLAG],
        )
        assert "DIP WATCH" in report
        dw_idx = report.index("DIP WATCH")
        section = report[dw_idx:]
        assert "LHLT.T" not in section
        assert "LGRW.T" not in section
        assert "GOOD.T" in section

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
            portfolio_health_flags=[_CORR_FLAG],
        )
        assert "DIP WATCH" not in report


# ── _pnl_line helpers ─────────────────────────────────────────────────────────


def _make_sell_item(
    ticker: str = "9201.T",
    action: str = "SELL",
    sell_type: str = "STOP_BREACH",
    reason: str = "Stop breached: price 2700.00 < stop 2780.00",
    quantity: float = 100,
    avg_cost_local: float = 2780.0,
    current_price_local: float = 2700.0,
    unrealized_pnl_usd: float = -89.0,
    suggested_quantity: int | None = None,
) -> ReconciliationItem:
    """Build a SELL ReconciliationItem with a fully-populated NormalizedPosition."""
    pos = NormalizedPosition(
        conid=99999,
        yf_ticker=ticker,
        symbol=ticker.split(".")[0],
        quantity=quantity,
        avg_cost_local=avg_cost_local,
        current_price_local=current_price_local,
        unrealized_pnl_usd=unrealized_pnl_usd,
        market_value_usd=abs(current_price_local * quantity / 100),
        currency="JPY",
    )
    return ReconciliationItem(
        ticker=ticker,
        action=action,
        urgency="HIGH",
        reason=reason,
        ibkr_position=pos,
        sell_type=sell_type,
        suggested_quantity=suggested_quantity,
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
        assert "+¥30,000" in report

    def test_loss_line_shows_negative_pnl(self):
        """Position with current < avg_cost shows 'est. loss:' in local currency."""
        # (2700 - 2780) × 100 shares = -¥8,000
        item = _make_sell_item(
            avg_cost_local=2780.0,
            current_price_local=2700.0,
        )
        report = self._report_for(item)
        assert "est. loss:" in report
        assert "-¥8,000" in report

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
        report = self._report_for(item)
        assert "+¥10,000" in report
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
        report = format_report([item], _make_portfolio(), show_recommendations=False)
        assert "SELLS — FUNDAMENTAL FAILURE" in report
        assert "est. gain:" in report
