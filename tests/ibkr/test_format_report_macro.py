"""Stored macro-event banner behavior."""

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
        from unittest.mock import MagicMock

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
                portfolio_health_flags=[CORRELATED_SELL_EVENT_FLAG],
            )

    def test_stored_event_headline_appears_in_banner(self):
        """When stored event available, headline shown in MACRO ALERT banner."""
        report = self._report([self._event("US tariffs announced")])
        assert "Macro driver: TARIFF_TRADE" in report
        assert "Impact: TRANSIENT" in report
        assert "Headline:" in report
        assert "US tariffs announced" in report

    def test_no_stored_events_banner_still_renders(self):
        """No stored events → banner still shows MACRO ALERT, no headline injected."""
        report = self._report([])
        assert "MACRO ALERT" in report
        assert "Headline:" not in report

    def test_store_unavailable_banner_renders_without_headline(self):
        """store.available=False → banner renders normally, no headline injected."""
        report = self._report(available=False)
        assert "MACRO ALERT" in report
        assert "Headline:" not in report

    def test_long_headline_marked_as_truncated(self):
        """Very long banner headline is truncated explicitly, not with a fragmentary ellipsis."""
        long_hl = "A" * 100
        report = self._report([self._event(long_hl)])
        assert "Headline:" in report
        assert "[truncated]" in report
        assert "Characterized:" not in report

    def test_banner_lines_fit_box_width(self):
        """Injected event metadata should stay within the banner width."""
        report = self._report(
            [
                self._event(
                    "Stock Market News, March 3, 2026: Dow Pares Early as Middle East Conflict Escalates Further"
                )
            ]
        )
        for line in report.splitlines():
            if line.startswith("║"):
                assert len(line) <= 58


# ── New-buy section helpers ───────────────────────────────────────────────────
