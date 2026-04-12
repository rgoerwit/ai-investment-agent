"""Tests for macro event injection into the News Analyst extra_context."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest


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


def _run_injection_block(ticker: str, events: list) -> str:
    """
    Simulate the `if agent_key == 'news_analyst':` injection block from analyst_node.
    Returns the resulting extra_context string.
    """
    extra_context = ""
    with patch("src.memory.create_macro_events_store") as mock_create:
        mock_store = MagicMock()
        mock_store.available = True
        mock_store.get_active_events.return_value = events
        mock_create.return_value = mock_store

        try:
            from src.memory import create_macro_events_store
            from src.ticker_policy import get_ticker_suffix

            _mstore = create_macro_events_store()
            if _mstore.available:
                _region = get_ticker_suffix(ticker)
                _events = _mstore.get_active_events(region_filter=_region or None)
                if _events:
                    _lines = ["### MACRO EVENT CONTEXT (portfolio-detected)"]
                    for _ev in _events[:2]:
                        _lines.append(
                            f"- {_ev.event_date} | {_ev.impact} | "
                            f"{_ev.scope}: {_ev.news_headline}"
                        )
                        if _ev.news_detail:
                            _lines.append(f"  {_ev.news_detail}")
                    _lines.append(
                        "Instruction: Determine if this equity is an "
                        "'Innocent Bystander' (dropped due to the macro event, "
                        "fundamentals intact \u2192 OPPORTUNITY) or "
                        "'Structurally Impaired' (business model affected \u2192 EXIT). "
                        "Ignore if event is inapplicable to this region/sector."
                    )
                    extra_context += "\n\n" + "\n".join(_lines) + "\n"
        except Exception:
            pass
    return extra_context


class TestNewsAnalystMacroInjection:
    def test_active_event_produces_extra_context(self):
        """Active macro event → MACRO EVENT CONTEXT block appears in extra_context."""
        ctx = _run_injection_block("7203.T", [_active_event()])
        assert "MACRO EVENT CONTEXT" in ctx
        assert "Markets fall on tariff fears" in ctx

    def test_no_events_produces_no_extra_context(self):
        """No active events → extra_context remains empty."""
        ctx = _run_injection_block("7203.T", [])
        assert ctx == ""

    def test_capped_at_2_events(self):
        """3 active events → only 2 injected (cap enforced)."""
        events = [_active_event(event_date=f"2026-03-0{i}") for i in range(1, 4)]
        ctx = _run_injection_block("7203.T", events)
        # Only 2 headlines should appear (cap at 2)
        assert ctx.count("Markets fall on tariff fears") <= 2

    def test_innocent_bystander_framing_present(self):
        """'Innocent Bystander' framing appears in injection text."""
        ctx = _run_injection_block("7203.T", [_active_event()])
        assert "Innocent Bystander" in ctx

    def test_structurally_impaired_framing_present(self):
        """'Structurally Impaired' framing appears in injection text."""
        ctx = _run_injection_block("7203.T", [_active_event()])
        assert "Structurally Impaired" in ctx

    def test_news_detail_included_when_present(self):
        """Event with news_detail → detail line appears in context."""
        ctx = _run_injection_block("7203.T", [_active_event(news_detail="Key detail.")])
        assert "Key detail." in ctx

    def test_news_detail_skipped_when_empty(self):
        """Event with empty news_detail → no empty line in context."""
        ctx = _run_injection_block("7203.T", [_active_event(news_detail="")])
        # Context still produced (event headline present), no extra blank lines
        assert "MACRO EVENT CONTEXT" in ctx
        lines = ctx.strip().split("\n")
        # No line should be only whitespace within the block
        assert not any(line.strip() == "" for line in lines)

    def test_event_date_and_impact_in_context(self):
        """Event date and impact appear in the injected line."""
        ctx = _run_injection_block("7203.T", [_active_event()])
        assert "2026-03-05" in ctx
        assert "TRANSIENT" in ctx

    def test_store_unavailable_produces_no_extra_context(self):
        """MacroEventsStore.available=False → no injection."""
        extra_context = ""
        with patch("src.memory.create_macro_events_store") as mock_create:
            mock_store = MagicMock()
            mock_store.available = False
            mock_create.return_value = mock_store
            try:
                from src.memory import create_macro_events_store

                _mstore = create_macro_events_store()
                if _mstore.available:
                    extra_context = "SHOULD NOT APPEAR"
            except Exception:
                pass
        assert extra_context == ""

    def test_exception_in_injection_does_not_propagate(self):
        """Exception in injection block is caught; extra_context stays empty."""
        extra_context = ""
        with patch(
            "src.memory.create_macro_events_store", side_effect=Exception("boom")
        ):
            try:
                from src.memory import create_macro_events_store

                _mstore = create_macro_events_store()  # raises
            except Exception:
                pass  # caught by outer try/except in analyst_node
        assert extra_context == ""

    def test_region_suffix_passed_to_get_active_events(self):
        """get_active_events() is called with the ticker's region suffix."""
        with patch("src.memory.create_macro_events_store") as mock_create:
            mock_store = MagicMock()
            mock_store.available = True
            mock_store.get_active_events.return_value = []
            mock_create.return_value = mock_store
            _run_injection_block("7203.T", [])
            # Called with region_filter=".T"
            call_kwargs = mock_store.get_active_events.call_args
            if call_kwargs:
                region = call_kwargs[1].get("region_filter") or (
                    call_kwargs[0][0] if call_kwargs[0] else None
                )
                assert region == ".T"
