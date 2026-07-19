"""Core and macro-event report behavior."""

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


class TestRuntimeCommandHints:
    def test_analysis_command_uses_python_in_container(self, monkeypatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setenv("INVESTMENT_AGENT_CONTAINER", "1")
        assert _analysis_command("7203.T") == "python -m src.main --ticker 7203.T"

    def test_analysis_command_uses_poetry_on_host_even_with_virtual_env(
        self, monkeypatch
    ):
        monkeypatch.setenv("VIRTUAL_ENV", "/tmp/fake-venv")
        monkeypatch.delenv("INVESTMENT_AGENT_CONTAINER", raising=False)
        with patch("scripts.portfolio_manager.Path.exists", return_value=False):
            assert (
                _analysis_command("7203.T")
                == "poetry run python -m src.main --ticker 7203.T"
            )

    def test_portfolio_manager_command_falls_back_to_poetry_on_host(self, monkeypatch):
        monkeypatch.setenv("VIRTUAL_ENV", "/tmp/fake-venv")
        monkeypatch.delenv("INVESTMENT_AGENT_CONTAINER", raising=False)
        with patch("scripts.portfolio_manager.Path.exists", return_value=False):
            assert (
                _portfolio_manager_command("--recommend")
                == "poetry run python scripts/portfolio_manager.py --recommend"
            )

    def test_portfolio_manager_recommend_command_preserves_watchlist(self, monkeypatch):
        monkeypatch.setenv("VIRTUAL_ENV", "/tmp/fake-venv")
        monkeypatch.delenv("INVESTMENT_AGENT_CONTAINER", raising=False)
        with patch("scripts.portfolio_manager.Path.exists", return_value=False):
            assert (
                _portfolio_manager_recommend_command(watchlist_name="watchlist-2026")
                == 'poetry run python scripts/portfolio_manager.py --recommend --watchlist-name "watchlist-2026"'
            )


class TestFormatReportPanicDay:
    """format_report() on a panic day — CORRELATED_SELL_EVENT flag set, SOFT_REJECTs demoted."""

    def _report_with_flag(self) -> str:
        return format_report(
            _panic_items(),
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
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
        # All three parsed values must appear in the banner. Wording is
        # trigger-neutral ("impacted") because the flag may be verdict-flip
        # or drawdown-breadth evidence.
        assert "8 positions impacted (as of 2026-03-05)" in report
        assert "80% of held positions" in report

    def test_macro_banner_absent_when_no_flag(self):
        """Empty health_flags → no MACRO ALERT banner, even if items are present."""
        assert "MACRO ALERT" not in self._report_without_flag()

    def test_macro_banner_uses_current_event_over_store(self):
        """Fix B: banner reflects the event detected THIS run (current_macro_event),
        not a stale/unrelated stored active event, and needs no Chroma round-trip."""
        from src.memory import MacroEvent

        current = MacroEvent(
            event_date="2026-03-05",
            detected_date="2026-03-05",
            expiry="2026-04-02",
            impact="STRUCTURAL",
            event_type="CONTAGION_SPREAD",
            scope="GLOBAL",
            primary_region="GLOBAL",
            primary_sector="",
            severity="HIGH",
            correlation_pct=0.80,
            peak_count=8,
            total_held=10,
            news_headline="Regional bank contagion spreads",
            news_detail="detail",
            forced_reanalysis=True,
        )
        report = format_report(
            _panic_items(),
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
            current_macro_event=current,
        )
        assert "Macro driver: CONTAGION_SPREAD" in report
        assert "Impact: STRUCTURAL" in report

    def test_stop_breached_section_present(self):
        """A legacy STOP_BREACH is downgraded into position reviews."""
        report = self._report_with_flag()
        assert "POSITION REVIEWS" in report
        assert "[PRICE-DROP REVIEW]" in report

    def test_fundamental_failure_section_present(self):
        """A confirmation-gated thesis failure remains an executable sale."""
        report = self._report_with_flag()
        assert "SELL RECOMMENDATIONS" in report
        assert "[CONFIRMED THESIS FAILURE]" in report

    def test_soft_rejection_section_present(self):
        """Demoted macro_reviews render in the position-review section."""
        report = self._report_with_flag()
        assert "POSITION REVIEWS" in report
        assert "[SOFT REJECTION]" in report

    def test_soft_rejection_section_shows_demoted_items(self):
        """Demoted REVIEW + SOFT_REJECT items are listed in position reviews."""
        report = self._report_with_flag()
        lines = report.split("\n")
        soft_rej_idx = next(
            (i for i, ln in enumerate(lines) if "POSITION REVIEWS" in ln), None
        )
        assert soft_rej_idx is not None, "POSITION REVIEWS section missing"
        section_content = "\n".join(lines[soft_rej_idx:])
        # Held positions show IBKR symbol (no exchange suffix) in human-visible sections
        assert "SOFT00" in section_content

    def test_reviews_section_excludes_soft_reject_items(self):
        """SOFT00.T appears only in SOFT REJECTION — not in the regular REVIEWS section.

        The regular REVIEWS section uses 'poetry run python -m src.main' command suggestions.
        SOFT REJECTION section does not. Checking for this distinguishes the two.
        """
        report = self._report_with_flag()
        lines = report.split("\n")
        soft00_in_reviews_format = any(
            "SOFT00.T" in line and "poetry run python -m src.main" in line
            for line in lines
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


class TestFormatReportMacroStopReview:
    """format_report() bucket/rendering logic for STOP_BREACH items demoted during macro events."""

    def _make_demoted_stop_item(self, ticker: str = "STOP01.T") -> ReconciliationItem:
        """A STOP_BREACH item already demoted to REVIEW (as compute_portfolio_health would do)."""
        pos = _make_position(current_price=1500)
        analysis = _make_analysis(ticker=ticker, verdict="BUY", age_days=0)
        analysis.health_adj = 70.0
        analysis.growth_adj = 65.0
        return ReconciliationItem(
            ticker=ticker,
            action="REVIEW",
            urgency="MEDIUM",
            reason=(
                "Stop breached: price 1500.00 < stop 1800.00"
                "  [MACRO_STOP: stop breach during correlated event"
                " — fundamentals intact (health 70%, growth 65%); review before executing]"
            ),
            ibkr_position=pos,
            analysis=analysis,
            sell_type="STOP_BREACH",
        )

    def test_demoted_stop_appears_in_stop_breaches_under_review_section(self):
        """REVIEW + STOP_BREACH item renders in position reviews."""
        item = self._make_demoted_stop_item()
        report = format_report(
            [item],
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        assert "POSITION REVIEWS" in report
        assert "[PRICE-DROP REVIEW]" in report

    def test_demoted_stop_not_in_mechanical_stop_breached_section(self):
        """Demoted STOP_BREACH (action=REVIEW) must not appear in the mechanical 'STOP BREACHED' section."""
        item = self._make_demoted_stop_item()
        report = format_report(
            [item],
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        lines = report.split("\n")
        # Identify which section the ticker appears in
        stop_breach_idx = next(
            (i for i, ln in enumerate(lines) if "SELL RECOMMENDATIONS" in ln), None
        )
        # Mechanical STOP BREACHED section must be absent (no items with action=SELL)
        assert (
            stop_breach_idx is None
        ), "Mechanical STOP BREACHED section should not appear"

    def test_demoted_stop_not_in_regular_reviews_section(self):
        """REVIEW + STOP_BREACH must not appear in the regular REVIEWS section."""
        item = self._make_demoted_stop_item()
        # Add an unrelated REVIEW item so the regular REVIEWS section renders
        pos = _make_position(ticker="7203.T", current_price=2100)
        regular_review = ReconciliationItem(
            ticker="7203.T",
            action="REVIEW",
            urgency="MEDIUM",
            reason="Needs re-analysis",
            ibkr_position=pos,
            sell_type=None,
        )
        report = format_report(
            [item, regular_review],
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        lines = report.split("\n")
        # Find where regular REVIEWS section starts
        reviews_idx = next(
            (
                i
                for i, ln in enumerate(lines)
                if ln.strip().startswith("REVIEWS") and "STOP" not in ln
            ),
            None,
        )
        if reviews_idx is not None:
            section_content = "\n".join(lines[reviews_idx:])
            assert "STOP01" not in section_content

    def test_macro_stop_annotation_stripped_from_display(self):
        """The [MACRO_STOP: ...] annotation must not appear in the rendered output."""
        item = self._make_demoted_stop_item()
        report = format_report(
            [item],
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        assert "[MACRO_STOP:" not in report

    def test_legacy_stop_sell_is_downgraded_even_without_macro_demotion(self):
        """Presentation safety does not depend on the macro-demotion path."""
        pos = _make_position(current_price=2100)
        # Only a regular SELL + STOP_BREACH (not demoted) — no macro_stop_reviews
        items = [
            ReconciliationItem(
                ticker="WEAK.T",
                action="SELL",
                urgency="HIGH",
                reason="Stop breached: price 1500.00 < stop 1800.00",
                ibkr_position=pos,
                sell_type="STOP_BREACH",
            )
        ]
        report = format_report(
            items,
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        assert "POSITION REVIEWS" in report
        assert "SELL RECOMMENDATIONS" not in report
        assert "[PRICE-DROP REVIEW]" in report

    def test_summary_counts_demoted_stop_as_review_not_sell(self):
        """Demoted STOP_BREACH (action=REVIEW) contributes to REVIEW count, not SELL count."""
        item = self._make_demoted_stop_item()
        report = format_report(
            [item],
            _make_portfolio(),
            portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
        )
        summary_line = next(
            (ln for ln in report.split("\n") if ln.strip().startswith("Summary:")), ""
        )
        assert summary_line, "Summary line missing"
        assert "1 REVIEW" in summary_line
        assert "SELL" not in summary_line


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

    def test_legacy_soft_sell_on_normal_day_is_review(self):
        """A soft rejection lacks confirmed fundamental sale authority."""
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
        assert "REVIEWS" in report
        assert "SELL RECOMMENDATIONS" not in report
        assert "[THESIS REASSESSMENT]" in report
        lines = report.split("\n")
        soft_rej_idx = next(
            (i for i, ln in enumerate(lines) if ln.strip().startswith("REVIEWS")), None
        )
        assert soft_rej_idx is not None
        # Check the next ~15 lines for the REVIEW action label and the ticker
        # Held positions show IBKR symbol ("7203") not yfinance ("7203.T") in display
        section_lines = lines[soft_rej_idx : soft_rej_idx + 15]
        assert any("REVIEW" in ln and "7203" in ln for ln in section_lines)

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
        assert "SOFT REJECTION" not in report

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
