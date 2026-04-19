"""Tests for News Analyst macro-context injection helpers."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _active_event(
    event_date: str = "2026-03-05",
    impact: str = "TRANSIENT",
    scope: str = "GLOBAL",
    primary_region: str = "GLOBAL",
    news_headline: str = "Markets fall on tariff fears",
    news_detail: str = "Global sell-off on trade war concerns.",
):
    from src.memory import MacroEvent

    expiry = (date.fromisoformat(event_date) + timedelta(days=28)).isoformat()
    return MacroEvent(
        event_date=event_date,
        detected_date="2026-03-07",
        expiry=expiry,
        impact=impact,
        event_type="TARIFF_TRADE",
        scope=scope,
        primary_region=primary_region,
        primary_sector="",
        severity="MEDIUM",
        correlation_pct=0.30,
        peak_count=6,
        total_held=20,
        news_headline=news_headline,
        news_detail=news_detail,
        forced_reanalysis=False,
    )


class TestNewsAnalystMacroInjection:
    def test_event_only_block_present(self):
        from src.agents.analyst_nodes import _build_news_macro_extra_context

        with patch("src.memory.create_macro_events_store") as mock_create:
            mock_store = MagicMock()
            mock_store.available = True
            mock_store.get_active_events.return_value = [_active_event()]
            mock_create.return_value = mock_store

            extra_context = _build_news_macro_extra_context("7203.T", None)

        assert "### PORTFOLIO MACRO EVENT" in extra_context
        assert "Markets fall on tariff fears" in extra_context
        assert "### REGIONAL MACRO CONTEXT" not in extra_context

    def test_brief_only_block_present(self):
        from src.agents.analyst_nodes import _build_news_macro_extra_context

        with patch("src.memory.create_macro_events_store") as mock_create:
            mock_store = MagicMock()
            mock_store.available = True
            mock_store.get_active_events.return_value = []
            mock_create.return_value = mock_store

            context = SimpleNamespace(
                macro_context_report="### RATES & LIQUIDITY\n- Summary: BOJ still supportive.",
                macro_context_region="JAPAN",
                macro_context_status="generated",
            )
            extra_context = _build_news_macro_extra_context("7203.T", context)

        assert "### PORTFOLIO MACRO EVENT" not in extra_context
        assert "### REGIONAL MACRO CONTEXT" in extra_context
        assert "Region: JAPAN" in extra_context

    def test_event_and_brief_appear_in_deterministic_order(self):
        from src.agents.analyst_nodes import _build_news_macro_extra_context

        with patch("src.memory.create_macro_events_store") as mock_create:
            mock_store = MagicMock()
            mock_store.available = True
            mock_store.get_active_events.return_value = [_active_event()]
            mock_create.return_value = mock_store

            context = SimpleNamespace(
                macro_context_report="### EQUITY REGIME\n- Summary: Risk appetite improving.",
                macro_context_region="JAPAN",
                macro_context_status="cached",
            )
            extra_context = _build_news_macro_extra_context("7203.T", context)

        portfolio_idx = extra_context.index("### PORTFOLIO MACRO EVENT")
        regional_idx = extra_context.index("### REGIONAL MACRO CONTEXT")
        assert portfolio_idx < regional_idx

    def test_regional_macro_injection_log_includes_audit_fields(self):
        from src.agents.analyst_nodes import _build_news_macro_extra_context

        with (
            patch("src.memory.create_macro_events_store") as mock_create,
            patch("src.agents.analyst_nodes.logger") as mock_logger,
        ):
            mock_store = MagicMock()
            mock_store.available = True
            mock_store.get_active_events.return_value = [_active_event()]
            mock_create.return_value = mock_store

            context = SimpleNamespace(
                macro_context_report="### EQUITY REGIME\n- Summary: Risk appetite improving.",
                macro_context_region="JAPAN",
                macro_context_status="generated",
            )
            _build_news_macro_extra_context("7203.T", context)

        mock_logger.info.assert_any_call(
            "macro_context_injected",
            ticker="7203.T",
            region="JAPAN",
            status="generated",
            report_len=len(context.macro_context_report),
            agent="news_analyst",
            portfolio_macro_event_present=True,
            regional_macro_context_present=True,
        )

    def test_no_context_produces_empty_string(self):
        from src.agents.analyst_nodes import _build_news_macro_extra_context

        with patch("src.memory.create_macro_events_store") as mock_create:
            mock_store = MagicMock()
            mock_store.available = True
            mock_store.get_active_events.return_value = []
            mock_create.return_value = mock_store

            extra_context = _build_news_macro_extra_context("7203.T", None)

        assert extra_context == ""

    def test_empty_report_skips_regional_block(self):
        from src.agents.analyst_nodes import _build_news_macro_extra_context

        with patch("src.memory.create_macro_events_store") as mock_create:
            mock_store = MagicMock()
            mock_store.available = True
            mock_store.get_active_events.return_value = []
            mock_create.return_value = mock_store

            context = SimpleNamespace(
                macro_context_report="",
                macro_context_region="JAPAN",
                macro_context_status="generated",
            )
            extra_context = _build_news_macro_extra_context("7203.T", context)

        assert "### REGIONAL MACRO CONTEXT" not in extra_context

    def test_store_unavailable_does_not_block_regional_brief(self):
        from src.agents.analyst_nodes import _build_news_macro_extra_context

        with patch("src.memory.create_macro_events_store") as mock_create:
            mock_store = MagicMock()
            mock_store.available = False
            mock_create.return_value = mock_store

            context = SimpleNamespace(
                macro_context_report="### FX & FLOWS\n- Summary: Yen remains weak.",
                macro_context_region="JAPAN",
                macro_context_status="generated_fallback",
            )
            extra_context = _build_news_macro_extra_context("7203.T", context)

        assert "### PORTFOLIO MACRO EVENT" not in extra_context
        assert "### REGIONAL MACRO CONTEXT" in extra_context

    def test_region_suffix_passed_to_macro_event_lookup(self):
        from src.agents.analyst_nodes import _build_news_macro_extra_context

        with patch("src.memory.create_macro_events_store") as mock_create:
            mock_store = MagicMock()
            mock_store.available = True
            mock_store.get_active_events.return_value = []
            mock_create.return_value = mock_store

            _build_news_macro_extra_context("7203.T", None)

        call_kwargs = mock_store.get_active_events.call_args
        region = call_kwargs.kwargs.get("region_filter") if call_kwargs else None
        assert region == ".T"

    def test_event_block_keeps_innocent_bystander_framing(self):
        from src.agents.analyst_nodes import _build_news_macro_extra_context

        with patch("src.memory.create_macro_events_store") as mock_create:
            mock_store = MagicMock()
            mock_store.available = True
            mock_store.get_active_events.return_value = [_active_event()]
            mock_create.return_value = mock_store

            extra_context = _build_news_macro_extra_context("7203.T", None)

        assert "Innocent Bystander" in extra_context
        assert "Structurally Impaired" in extra_context
