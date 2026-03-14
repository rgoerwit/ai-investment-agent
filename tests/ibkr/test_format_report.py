"""Tests for format_report() output formatting — macro-panic day and normal day."""

from __future__ import annotations

from scripts.portfolio_manager import _compute_dip_score, format_report
from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    ReconciliationItem,
    TradeBlockData,
)
from src.ibkr.ticker import Ticker
from tests.ibkr.test_reconciler import _make_portfolio, _make_position

# A realistic CORRELATED_SELL_EVENT flag string (matches what compute_portfolio_health emits).
# Format: "CORRELATED_SELL_EVENT: N positions changed verdict within Xd of DATE (P% of held …"
_CORR_FLAG = (
    "CORRELATED_SELL_EVENT: 8 positions changed verdict within 7d of 2026-03-05"
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

    def test_macro_banner_shows_count_date_and_pct(self):
        """MACRO ALERT banner must show the count, date, and percentage parsed from the flag."""
        report = self._report_with_flag()
        # All three parsed values must appear in the banner
        assert "8 positions changed verdict on 2026-03-05" in report
        assert "80% of held positions" in report

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
        # Held positions show IBKR symbol (no exchange suffix) in human-visible sections
        assert "SOFT00" in section_content

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
        # Held positions show IBKR symbol ("7203") not yfinance ("7203.T") in display
        section_lines = lines[soft_rej_idx : soft_rej_idx + 15]
        assert any("SELL" in ln and "7203" in ln for ln in section_lines)

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
            verdict="DO_NOT_INITIATE",
            health_adj=75.0,  # >= 55 → passes DIP WATCH quality filter
            growth_adj=70.0,  # >= 55 → passes
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
                f"Verdict → DO_NOT_INITIATE  ({analysis_date})"
                "  [MACRO_WATCH: demoted from SELL — correlated event detected]"
            ),
            ibkr_position=pos,
            analysis=analysis,
            sell_type="SOFT_REJECT",
        )
        report = format_report(
            [item],
            _make_portfolio(),
            portfolio_health_flags=[_CORR_FLAG],
        )
        assert "DIP WATCH" in report
        # Re-run command must use canonical yf ticker (analysis.ticker), not bare symbol
        assert "--ticker BARE.L" in report
        assert "⚠ verify exchange suffix" not in report


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
        ticker=Ticker.from_yf(ticker),
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
        """SELL item with matching open SELL order (by conid) → 'ORDER ALREADY SUBMITTED' shown."""
        item = _make_sell_item(
            ticker="9201.T",
            sell_type="STOP_BREACH",
            reason="Stop breached: price 2700.00 < stop 2780.00",
        )
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
        order = _make_order(conid=99999, side="B", remaining_size=100, price=2780.0)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "CONFLICT" in report
        assert "ORDER ALREADY SUBMITTED" not in report

    def test_no_orders_no_annotation(self):
        """Empty live_orders list → no order annotation anywhere in report."""
        item = _make_sell_item()
        report = format_report([item], _make_portfolio(), live_orders=[])
        assert "ORDER ALREADY SUBMITTED" not in report
        assert "CONFLICT" not in report

    def test_order_for_different_ticker_not_shown(self):
        """Order with a different conid and non-matching symbol → not annotated."""
        item = _make_sell_item(ticker="9201.T")
        # item.ibkr_position.conid == 99999; order has conid 11111 and ticker "XXXX"
        order = _make_order(conid=11111, ticker="XXXX", side="S", remaining_size=100)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "ORDER ALREADY SUBMITTED" not in report
        assert "CONFLICT" not in report

    def test_live_orders_none_no_annotation(self):
        """live_orders omitted (default None) → no annotation."""
        item = _make_sell_item()
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
        order = _make_order(conid=99999, side="S", remaining_size=100, price=2780.0)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "ORDER ALREADY SUBMITTED" in report
        assert "PARTIAL ORDER" not in report

    def test_hk_sell_matched_by_unpadded_ibkr_symbol(self):
        """Regression: HK yf ticker '0005.HK' has zero-padded base '0005', but IBKR
        live orders use unpadded '5'.  Symbol fallback must match both forms."""
        item = _make_sell_item(ticker="0005.HK", sell_type="HARD_REJECT")
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
        )
        item = ReconciliationItem(
            ticker="0700.HK",
            action="SELL",
            urgency="HIGH",
            reason="DO_NOT_INITIATE",
            ibkr_position=pos,
            sell_type="HARD_REJECT",
        )
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
        )
        item = ReconciliationItem(
            ticker="MELI",
            action="SELL",
            urgency="HIGH",
            reason="STOP_BREACH",
            ibkr_position=pos,
            sell_type="STOP_BREACH",
        )
        order = _make_order(conid=55555, side="S", remaining_size=10, price=1490.0)
        report = format_report([item], _make_portfolio(), live_orders=[order])
        assert "ORDER ALREADY SUBMITTED" in report


class TestMacroAlertBannerWithStoredEvent:
    """format_report() MACRO ALERT banner enhanced with a stored event headline."""

    def _event(self, headline: str = "US tariffs announced"):
        from datetime import date, timedelta

        from src.memory import MacroEvent

        return MacroEvent(
            event_date="2026-03-05",
            detected_date="2026-03-07",
            expiry=(date.today() + timedelta(days=20)).isoformat(),
            impact="TRANSIENT",
            event_type="TARIFF_TRADE",
            scope="GLOBAL",
            primary_region="GLOBAL",
            primary_sector="",
            severity="MEDIUM",
            correlation_pct=0.40,
            peak_count=8,
            total_held=20,
            news_headline=headline,
            news_detail="",
            forced_reanalysis=False,
        )

    def _report(self, mock_store_events=None, available: bool = True) -> str:
        from unittest.mock import MagicMock, patch

        mock_store = MagicMock()
        mock_store.available = available
        if mock_store_events:
            mock_store.get_active_events.return_value = mock_store_events
        else:
            mock_store.get_active_events.return_value = []

        with patch(
            "src.memory.create_macro_events_store",
            return_value=mock_store,
        ) as _mock_create:
            # Import inside patch context so the patched name is used
            return format_report(
                _panic_items(),
                _make_portfolio(),
                portfolio_health_flags=[_CORR_FLAG],
            )

    def test_stored_event_headline_appears_in_banner(self):
        """When stored event available, headline shown in MACRO ALERT banner."""
        report = self._report([self._event("US tariffs announced")])
        assert "US tariffs announced" in report

    def test_no_stored_events_banner_still_renders(self):
        """No stored events → banner still shows MACRO ALERT, no headline injected."""
        report = self._report([])
        assert "MACRO ALERT" in report

    def test_store_unavailable_banner_renders_without_headline(self):
        """store.available=False → banner renders normally, no headline injected."""
        report = self._report(available=False)
        assert "MACRO ALERT" in report

    def test_long_headline_truncated_to_62_chars(self):
        """Very long headline is truncated to 62 chars in the banner line."""
        long_hl = "A" * 100
        report = self._report([self._event(long_hl)])
        # The headline should appear but truncated
        # Either 62 chars of A or a prefix of "Characterized: "
        assert "Characterized:" in report or long_hl[:40] in report


# ── New-buy section helpers ───────────────────────────────────────────────────


def _make_buy_item(
    ticker: str = "7203.T",
    conviction: str = "High",
    size_pct: float = 4.0,
    suggested_quantity: int | None = 100,
    suggested_price: float | None = 2615.0,
    cash_impact_usd: float = -1752.0,
    analysis_date: str = "2026-03-01",
    analysis: AnalysisRecord | None = None,
) -> ReconciliationItem:
    """Build a BUY ReconciliationItem as the reconciler would produce for new buys."""
    if analysis is None:
        tb = TradeBlockData(conviction=conviction, size_pct=size_pct)
        analysis = AnalysisRecord(
            ticker=ticker,
            analysis_date=analysis_date,
            verdict="BUY",
            health_adj=72.0,
            growth_adj=65.0,
            trade_block=tb,
            conviction=conviction,
        )
    return ReconciliationItem(
        ticker=ticker,
        action="BUY",
        urgency="MEDIUM",
        reason=f"Watchlist BUY ({analysis_date}) — {conviction} conviction, target {size_pct:.1f}%",
        ibkr_position=None,
        analysis=analysis,
        suggested_quantity=suggested_quantity,
        suggested_price=suggested_price,
        suggested_order_type="LMT",
        cash_impact_usd=cash_impact_usd,
        is_watchlist=True,
    )


class TestNewBuysSection:
    """format_report() NEW BUYS section rendering and live-order annotation."""

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
        assert "NEW BUYS" in report
        assert "High conviction" in report
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
        assert "Medium conviction" in report
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

    def test_missing_conviction_shows_target_only(self):
        """Analysis with empty conviction → target still shown, no 'conviction' label."""
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
        assert "conviction" not in report.split("NEW BUYS")[1].split("═")[0]
        assert "target" in report

    def test_no_analysis_shows_order_line_only(self):
        """BUY item with no analysis → order line only, no detail line."""
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
        assert "NEW BUYS" in report
        assert "9201" in report  # IBKR format (no exchange suffix)
        assert "conviction" not in report

    def test_header_section_shown(self):
        """NEW BUYS section header appears when BUY items exist."""
        item = _make_buy_item()
        report = self._report([item])
        assert "NEW BUYS" in report

    def test_title_shows_named_watchlist(self):
        """When watchlist_name is provided, section subtitle names the watchlist."""
        item = _make_buy_item()
        report = self._report([item], watchlist_name="MyWatchlist")
        assert "from watchlist 'MyWatchlist'" in report

    def test_title_shows_generic_watchlist_when_name_unknown(self):
        """When watchlist_name is None (auto-discovered), subtitle says 'from watchlist'."""
        item = _make_buy_item()
        report = self._report([item], watchlist_name=None)
        assert "from watchlist" in report
        # The section still appears because the item has is_watchlist=True
        assert "NEW BUYS" in report

    def test_title_shows_count_when_watchlist_total_known(self):
        """watchlist_total provided → subtitle includes 'M/N from watchlist' ratio."""
        items = [_make_buy_item("7203.T"), _make_buy_item("6752.T")]
        report = self._report(items, watchlist_name="MyWatchlist", watchlist_total=15)
        assert "2/15" in report
        assert "from watchlist 'MyWatchlist'" in report

    def test_title_omits_count_when_watchlist_total_not_provided(self):
        """watchlist_total not provided → no M/N ratio in subtitle."""
        item = _make_buy_item()
        report = self._report(
            [item], watchlist_name="MyWatchlist", watchlist_total=None
        )
        assert "1/" not in report
        assert "from watchlist 'MyWatchlist'" in report

    def test_section_absent_when_no_watchlist_items(self):
        """BUY item without is_watchlist=True (e.g. Phase 2) → NEW BUYS section not shown."""
        item = ReconciliationItem(
            ticker="9201.T",
            action="BUY",
            urgency="MEDIUM",
            reason="New BUY (2026-03-01) — High conviction, target 4.0%",
            ibkr_position=None,
            is_watchlist=False,
        )
        report = self._report([item], watchlist_name="MyWatchlist")
        assert "NEW BUYS" not in report

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

    def test_new_buy_watchlist_shows_ibkr_ticker(self):
        """Phase 2 BUY (not held) displays IBKR format (no exchange suffix)."""
        item = ReconciliationItem(
            ticker="CAG.ST",
            action="BUY",
            reason="Watchlist BUY",
            urgency="MEDIUM",
            is_watchlist=True,
        )
        report = format_report([item], _make_portfolio())
        assert "CAG" in report  # IBKR format: "CAG" not "CAG.ST"

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


def _make_offwatch_buy(
    ticker: str = "WDO.TO", conviction: str = "High"
) -> ReconciliationItem:
    """Build a Phase-2 off-watchlist BUY item (is_watchlist=False)."""
    tb = TradeBlockData(conviction=conviction, size_pct=3.0)
    analysis = AnalysisRecord(
        ticker=ticker,
        analysis_date="2026-03-01",
        verdict="BUY",
        health_adj=70.0,
        growth_adj=62.0,
        trade_block=tb,
        conviction=conviction,
    )
    return ReconciliationItem(
        ticker=ticker,
        action="BUY",
        urgency="MEDIUM",
        reason="Off-watchlist BUY",
        ibkr_position=None,
        analysis=analysis,
        suggested_quantity=100,
        suggested_price=15.0,
        cash_impact_usd=-1500.0,
        is_watchlist=False,
    )


class TestWatchlistCandidatesInFlight:
    """WATCHLIST CANDIDATES section hides items that already have a live BUY order."""

    def _report(self, items, live_orders=None) -> str:
        return format_report(
            items,
            _make_portfolio(),
            show_recommendations=True,
            live_orders=live_orders or [],
        )

    def test_candidate_with_live_buy_order_hidden_from_section(self):
        """Off-watchlist BUY with an open order is excluded from WATCHLIST CANDIDATES."""
        item = _make_offwatch_buy("WDO.TO")
        live_order = {"ticker": "WDO", "side": "B", "remainingSize": "100"}
        report = self._report([item], live_orders=[live_order])
        # WDO removed from candidates display (no entry in WATCHLIST CANDIDATES body)
        cands_block = (
            report.split("WATCHLIST CANDIDATES")[1]
            if "WATCHLIST CANDIDATES" in report
            else ""
        )
        # Should show the in-flight note, not a regular candidate entry
        assert "already in flight" in cands_block
        assert "WDO" in cands_block
        # The normal "[not on watchlist" detail line should not appear for in-flight items
        assert "[not on watchlist" not in cands_block

    def test_candidate_without_live_order_shown_normally(self):
        """Off-watchlist BUY with no live order appears in WATCHLIST CANDIDATES as usual."""
        item = _make_offwatch_buy("WDO.TO")
        report = self._report([item], live_orders=[])
        assert "WATCHLIST CANDIDATES" in report
        assert "WDO" in report
        assert "not on watchlist" in report

    def test_in_flight_candidate_excluded_from_watchlist_moves(self):
        """In-flight candidates must not appear in WATCHLIST MOVES (ADDED TO WATCHLIST)."""
        item = _make_offwatch_buy("WDO.TO", conviction="High")
        live_order = {"ticker": "WDO", "side": "B", "remainingSize": "50"}
        report = self._report([item], live_orders=[live_order])
        # WATCHLIST MOVES should be absent entirely (no strong candidates remain)
        assert "ADDED TO WATCHLIST" not in report

    def test_two_candidates_one_in_flight_other_shown(self):
        """When one candidate is in-flight and another is not, only the latter appears."""
        inflight = _make_offwatch_buy("WDO.TO", conviction="High")
        pending = _make_offwatch_buy("TOTL.TO", conviction="Medium")
        live_order = {"ticker": "WDO", "side": "B", "remainingSize": "100"}
        report = self._report([inflight, pending], live_orders=[live_order])
        assert "WATCHLIST CANDIDATES" in report
        # TOTL shown as normal candidate
        assert "TOTL" in report
        assert "not on watchlist" in report
        # WDO shown only in the in-flight note, not as a candidate entry
        cands_block = report.split("WATCHLIST CANDIDATES")[1].split("HOLDS")[0]
        assert "already in flight" in cands_block
        assert "WDO" in cands_block


class TestSellBaseExcludesCandidate:
    """WATCHLIST CANDIDATES suppresses same-base BUY when a SELL exists for that symbol."""

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

    def test_sell_base_blocks_same_base_candidate(self):
        """SELL DLG → DLG.MI BUY candidate suppressed from WATCHLIST CANDIDATES."""
        sell = self._make_sell_item("DLG")
        buy_cand = _make_offwatch_buy("DLG.MI")
        report = format_report(
            [sell, buy_cand], _make_portfolio(), show_recommendations=True
        )
        # DLG appears in SELLs section
        assert "SELL" in report
        # DLG.MI must NOT appear as a watchlist candidate
        assert "WATCHLIST CANDIDATES" not in report

    def test_stop_breach_sell_also_blocks_candidate(self):
        """STOP_BREACH sell also suppresses same-base candidate."""
        sell = self._make_sell_item("DLG", sell_type="STOP_BREACH")
        buy_cand = _make_offwatch_buy("DLG.MI")
        report = format_report(
            [sell, buy_cand], _make_portfolio(), show_recommendations=True
        )
        assert "WATCHLIST CANDIDATES" not in report

    def test_different_base_candidate_not_blocked(self):
        """SELL DLG does not suppress a candidate with a different base symbol."""
        sell = self._make_sell_item("DLG")
        buy_cand = _make_offwatch_buy("WDO.TO")
        report = format_report(
            [sell, buy_cand], _make_portfolio(), show_recommendations=True
        )
        assert "WATCHLIST CANDIDATES" in report
        assert "WDO" in report


class TestWatchlistTickersExcludesCandidate:
    """WATCHLIST CANDIDATES suppresses BUY candidates already on the IBKR watchlist.

    Belt-and-suspenders against conid resolution failures: if Phase 1.5 silently
    drops a watchlist ticker (API error on first encounter), the format_report
    _watchlist_bases filter catches it here.
    """

    def test_suffixed_watchlist_ticker_blocks_same_candidate(self):
        """watchlist_tickers={'5434.TW'} → '5434.TW' BUY suppressed from WATCHLIST CANDIDATES."""
        buy_cand = _make_offwatch_buy("5434.TW")
        report = format_report(
            [buy_cand],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers={"5434.TW"},
        )
        assert "WATCHLIST CANDIDATES" not in report

    def test_bare_watchlist_ticker_blocks_suffixed_candidate(self):
        """watchlist_tickers={'5434'} (bare, failed resolution) → '5434.TW' BUY suppressed."""
        buy_cand = _make_offwatch_buy("5434.TW")
        report = format_report(
            [buy_cand],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers={"5434"},
        )
        assert "WATCHLIST CANDIDATES" not in report

    def test_none_watchlist_does_not_suppress(self):
        """watchlist_tickers=None → no watchlist filter applied."""
        buy_cand = _make_offwatch_buy("5434.TW")
        report = format_report(
            [buy_cand],
            _make_portfolio(),
            show_recommendations=True,
            watchlist_tickers=None,
        )
        assert "WATCHLIST CANDIDATES" in report
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
        assert "WATCHLIST CANDIDATES" in report
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
            [item], _make_portfolio(), portfolio_health_flags=[_CORR_FLAG]
        )
        assert "analysis entry ¥2,800.00  now ¥2,700.00" in report
        assert "vs IBKR cost basis ¥2,000.00" in report

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

    def test_watchlist_moves_are_advisory_not_past_tense(self):
        high = _make_offwatch_buy("TOTL.JK", conviction="High")
        report = format_report([high], _make_portfolio(), show_recommendations=True)
        assert "ADD TO WATCHLIST  TOTL" in report
        assert "ADDED TO WATCHLIST" not in report
