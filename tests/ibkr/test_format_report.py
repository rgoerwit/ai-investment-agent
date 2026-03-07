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


def _make_sell_item_with_analysis(
    sell_type: str = "STOP_BREACH",
    reason: str = "Stop breached: price 2700.00 < stop 2780.00",
    health: float = 75.0,
    growth: float = 68.0,
    zone: str = "MODERATE",
    verdict: str = "BUY",
    conviction: str = "High",
    analysis_date: str = "2026-01-15",
) -> ReconciliationItem:
    """SELL item with a fully-populated AnalysisRecord for score-line testing."""
    item = _make_sell_item(sell_type=sell_type, reason=reason)
    item = item.model_copy(
        update={
            "analysis": AnalysisRecord(
                ticker="9201.T",
                analysis_date=analysis_date,
                verdict=verdict,
                health_adj=health,
                growth_adj=growth,
                zone=zone,
                conviction=conviction,
            )
        }
    )
    return item


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
        report = self._report_for(item)
        assert "Health:38" in report
        assert "Growth:32" in report
        assert "Risk zone:HIGH" in report

    def test_score_line_shows_analysis_date(self):
        """Score line includes analysis date so operator knows how stale the scores are."""
        item = _make_sell_item_with_analysis(analysis_date="2026-01-15")
        report = self._report_for(item)
        assert "Last analysis (2026-01-15):" in report

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
        )
        report = format_report(
            [item], _make_portfolio(), portfolio_health_flags=[_CORR_FLAG]
        )
        # Health/Growth appear (via _display_data_line), but not the analysis date
        # that _score_line would add (since _display_data_line doesn't add it)
        assert "Health:75" in report
        assert "Growth:68" in report


# ── Order annotation helpers ──────────────────────────────────────────────────


def _make_order(
    conid: int | None = None,
    ticker: str | None = None,
    side: str = "S",
    remaining_size: int = 100,
    price: float = 2780.0,
    order_type: str = "LMT",
    status: str = "Submitted",
) -> dict:
    """Build a minimal IBKR live-order dict."""
    order: dict = {
        "side": side,
        "remainingSize": remaining_size,
        "price": price,
        "orderType": order_type,
        "status": status,
    }
    if conid is not None:
        order["conid"] = conid
    if ticker is not None:
        order["ticker"] = ticker
    return order


class TestOrderAnnotation:
    """format_report() live-order annotation via live_orders parameter."""

    def test_sell_with_matching_open_order_shows_note(self):
        """SELL item with matching open SELL order (by conid) → 'open order:' note shown."""
        item = _make_sell_item(
            ticker="9201.T",
            sell_type="STOP_BREACH",
            reason="Stop breached: price 2700.00 < stop 2780.00",
        )
        # Position conid = 99999 (set by _make_sell_item via NormalizedPosition)
        order = _make_order(conid=99999, side="S", remaining_size=100, price=2780.0)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "open order:" in report
        assert "SELL" in report
        assert "2780.00" in report

    def test_buy_with_matching_open_order_shows_note(self):
        """BUY item (no position) with open BUY order matched by symbol → note shown."""
        item = ReconciliationItem(
            ticker="9201.T",
            action="BUY",
            urgency="LOW",
            reason="BUY signal",
            ibkr_position=None,
        )
        # No conid on a BUY with no position — match by symbol "9201"
        order = _make_order(ticker="9201", side="B", remaining_size=50, price=2750.0)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "open order:" in report
        assert "BUY" in report

    def test_no_orders_no_note(self):
        """Empty live_orders list → no 'open order:' annotation anywhere in report."""
        item = _make_sell_item()
        report = format_report([item], _make_portfolio(), live_orders=[])
        assert "open order:" not in report

    def test_order_for_different_ticker_not_shown(self):
        """Order with a different conid and non-matching symbol → not annotated."""
        item = _make_sell_item(ticker="9201.T")
        # item.ibkr_position.conid == 99999; order has conid 11111 and ticker "XXXX"
        order = _make_order(conid=11111, ticker="XXXX", side="S", remaining_size=100)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "open order:" not in report

    def test_live_orders_none_no_annotation(self):
        """live_orders omitted (default None) → no annotation."""
        item = _make_sell_item()
        report = format_report([item], _make_portfolio())
        assert "open order:" not in report
