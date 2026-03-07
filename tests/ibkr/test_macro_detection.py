"""Tests for macro event detection helpers in scripts/portfolio_manager.py."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from scripts.portfolio_manager import (
    _EVENT_TYPE_RULES,
    _EXPIRY_DAYS,
    _characterize_macro_event,
    _store_macro_event_if_detected,
)
from src.ibkr.models import AnalysisRecord, ReconciliationItem, TradeBlockData
from tests.ibkr.test_reconciler import _make_analysis, _make_position


def _make_sell_item(ticker: str, sell_type: str = "SOFT_REJECT") -> ReconciliationItem:
    pos = _make_position(ticker=ticker)
    a = _make_analysis(ticker=ticker, verdict="DO_NOT_INITIATE", age_days=0)
    a.health_adj = 40.0
    a.growth_adj = 35.0
    return ReconciliationItem(
        ticker=ticker,
        action="SELL",
        reason="test",
        urgency="HIGH",
        ibkr_position=pos,
        analysis=a,
        sell_type=sell_type,
    )


class TestCharacterizeMacroEventScope:
    def test_all_same_exchange_yields_regional_scope(self):
        items = [_make_sell_item(f"{i:04d}.T") for i in range(5)]
        scope, region, sector, *_ = _characterize_macro_event("2026-03-05", items, 0.30)
        assert scope == "REGIONAL"
        assert region == ".T"

    def test_all_same_sector_yields_sector_scope(self):
        # Items on DIFFERENT exchanges to avoid regional dominance (top_region_pct < 0.60)
        # 4 different exchanges → each ≤ 25%, all same sector → sector scope
        items = (
            [_make_sell_item(f"{i:04d}.T") for i in range(1)]
            + [_make_sell_item(f"{i:04d}.HK") for i in range(1)]
            + [_make_sell_item(f"{i:04d}.L") for i in range(1)]
            + [_make_sell_item(f"{i:04d}.KS") for i in range(1)]
            + [_make_sell_item(f"{i:04d}.AS") for i in range(1)]
        )
        for item in items:
            item.analysis.sector = "Technology"
        scope, region, sector, *_ = _characterize_macro_event("2026-03-05", items, 0.30)
        assert scope == "SECTOR"
        assert sector == "Technology"

    def test_mixed_regions_yields_global_scope(self):
        items = (
            [_make_sell_item(f"{i:04d}.T") for i in range(3)]
            + [_make_sell_item(f"{i:04d}.HK") for i in range(3)]
            + [_make_sell_item(f"{i:04d}.L") for i in range(3)]
        )
        scope, *_ = _characterize_macro_event("2026-03-05", items, 0.30)
        assert scope == "GLOBAL"

    def test_empty_sell_items_yields_global_scope(self):
        scope, *_ = _characterize_macro_event("2026-03-05", [], 0.25)
        assert scope == "GLOBAL"

    def test_60pct_regional_threshold(self):
        """Exactly 60% same region → REGIONAL."""
        items = [_make_sell_item(f"{i:04d}.T") for i in range(6)] + [
            _make_sell_item(f"{i:04d}.HK") for i in range(4)
        ]
        scope, *_ = _characterize_macro_event("2026-03-05", items, 0.30)
        assert scope == "REGIONAL"


class TestCharacterizeMacroEventLLM:
    def test_llm_classification_used_when_available(self):
        """Valid LLM JSON response → event_type and impact from LLM."""
        llm_response = MagicMock()
        llm_response.content = (
            '{"event_type": "TARIFF_TRADE", "impact": "TRANSIENT", '
            '"opportunity_prior": "HIGH", "reasoning": "Trade policy"}'
        )
        items = [_make_sell_item("7203.T")]
        # Patch at source module (local import inside function)
        with patch("src.llms.create_quick_thinking_llm") as mock_create:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = llm_response
            mock_create.return_value = mock_llm
            with patch("tavily.TavilyClient") as mock_tav:
                mock_tav.return_value.search.return_value = {
                    "results": [
                        {"title": "Tariffs announced", "content": "US tariffs."}
                    ]
                }
                with patch("src.config.config") as mock_cfg:
                    mock_cfg.get_tavily_api_key.return_value = "test_key"
                    scope, region, sector, impact, event_type, headline, detail = (
                        _characterize_macro_event(
                            "2026-03-05", items, 0.30, peak_count=6
                        )
                    )
        assert event_type == "TARIFF_TRADE"
        assert impact == "TRANSIENT"

    def test_no_tavily_key_returns_uncertain_unknown(self):
        """Tavily API key absent → headline='unknown', impact=UNCERTAIN, type=UNKNOWN."""
        items = [_make_sell_item("7203.T")]
        with patch("src.config.config") as mock_cfg:
            mock_cfg.get_tavily_api_key.return_value = ""
            *_, impact, event_type, headline, _ = _characterize_macro_event(
                "2026-03-05", items, 0.30
            )
        assert headline == "unknown"
        assert impact == "UNCERTAIN"
        assert event_type == "UNKNOWN"

    def test_llm_failure_falls_back_to_keywords(self):
        """LLM raises → keyword rules produce correct classification."""
        items = [_make_sell_item("7203.T")]
        with patch(
            "src.llms.create_quick_thinking_llm", side_effect=Exception("LLM fail")
        ):
            with patch("tavily.TavilyClient") as mock_tav:
                mock_tav.return_value.search.return_value = {
                    "results": [
                        {
                            "title": "New trade tariffs announced",
                            "content": "tariff hike",
                        }
                    ]
                }
                with patch("src.config.config") as mock_cfg:
                    mock_cfg.get_tavily_api_key.return_value = "test_key"
                    *_, impact, event_type, headline, _ = _characterize_macro_event(
                        "2026-03-05", items, 0.30
                    )
        assert event_type == "TARIFF_TRADE"
        assert impact == "TRANSIENT"

    def test_structural_keyword_classified_correctly(self):
        """Keyword 'legislation' → REGULATORY_SHIFT / STRUCTURAL when LLM fails."""
        items = [_make_sell_item("7203.T")]
        with patch(
            "src.llms.create_quick_thinking_llm", side_effect=ImportError("no llm")
        ):
            with patch("tavily.TavilyClient") as mock_tav:
                mock_tav.return_value.search.return_value = {
                    "results": [
                        {
                            "title": "New legislation passed",
                            "content": "new law enacted",
                        }
                    ]
                }
                with patch("src.config.config") as mock_cfg:
                    mock_cfg.get_tavily_api_key.return_value = "test_key"
                    *_, impact, event_type, _, _ = _characterize_macro_event(
                        "2026-03-05", items, 0.30
                    )
        assert event_type == "REGULATORY_SHIFT"
        assert impact == "STRUCTURAL"

    def test_llm_malformed_json_falls_back_to_keywords(self):
        """LLM returns non-JSON → JSONDecodeError → keyword fallback."""
        llm_response = MagicMock()
        llm_response.content = "Sorry, I cannot classify this."
        items = [_make_sell_item("7203.T")]
        with patch("src.llms.create_quick_thinking_llm") as mock_create:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = llm_response
            mock_create.return_value = mock_llm
            with patch("tavily.TavilyClient") as mock_tav:
                mock_tav.return_value.search.return_value = {
                    "results": [
                        {"title": "War escalates", "content": "military conflict"}
                    ]
                }
                with patch("src.config.config") as mock_cfg:
                    mock_cfg.get_tavily_api_key.return_value = "test_key"
                    *_, impact, event_type, _, _ = _characterize_macro_event(
                        "2026-03-05", items, 0.30
                    )
        # Keyword fallback: "war" → GEOPOLITICAL
        assert event_type == "GEOPOLITICAL"


class TestEventTypeRules:
    def test_tariff_keywords_match_tariff_trade(self):
        keywords, etype, eimp, _ = _EVENT_TYPE_RULES[0]
        assert etype == "TARIFF_TRADE"
        assert "tariff" in keywords

    def test_geopolitical_keywords_include_war(self):
        matching = [
            (kw, et) for kw, et, *_ in _EVENT_TYPE_RULES if et == "GEOPOLITICAL"
        ]
        assert any("war" in kw for kw, _ in matching)

    def test_regulatory_shift_is_structural(self):
        for _, etype, eimp, _ in _EVENT_TYPE_RULES:
            if etype == "REGULATORY_SHIFT":
                assert eimp == "STRUCTURAL"
                break
        else:
            pytest.fail("REGULATORY_SHIFT not found in _EVENT_TYPE_RULES")

    def test_tariff_trade_is_transient(self):
        for _, etype, eimp, _ in _EVENT_TYPE_RULES:
            if etype == "TARIFF_TRADE":
                assert eimp == "TRANSIENT"
                break


class TestExpiry:
    def test_tariff_trade_expiry_is_28_days(self):
        assert _EXPIRY_DAYS["TARIFF_TRADE"] == 28

    def test_geopolitical_expiry_is_180_days(self):
        assert _EXPIRY_DAYS["GEOPOLITICAL"] == 180

    def test_liquidity_panic_shortest_expiry(self):
        assert _EXPIRY_DAYS["LIQUIDITY_PANIC"] == 14

    def test_macro_recession_longest_expiry(self):
        assert _EXPIRY_DAYS["MACRO_RECESSION"] == 180

    def test_all_event_types_have_expiry(self):
        required = [
            "TARIFF_TRADE",
            "LIQUIDITY_PANIC",
            "CONTAGION_SPREAD",
            "POLITICAL_EVENT",
            "MONETARY_PIVOT",
            "COMMODITY_SHOCK",
            "GEOPOLITICAL",
            "REGULATORY_SHIFT",
            "CREDIT_CONTAGION",
            "MACRO_RECESSION",
            "EXOGENOUS_SHOCK",
            "UNKNOWN",
        ]
        for etype in required:
            assert etype in _EXPIRY_DAYS, f"{etype} missing from _EXPIRY_DAYS"


class TestStoreMacroEventIfDetected:
    _CORR_FLAG = (
        "CORRELATED_SELL_EVENT: 8 positions changed verdict within 7d of 2026-03-05"
        " (40% of held positions) — probable macro event."
        " Execute stop-breach SELLs only; review verdict-change SELLs before acting."
    )

    def test_no_flag_returns_without_side_effects(self):
        """No CORRELATED_SELL_EVENT in flags → nothing stored."""
        with patch("src.memory.create_macro_events_store") as mock_fn:
            _store_macro_event_if_detected(["SOME_OTHER_FLAG"], [])
        mock_fn.assert_not_called()

    def test_flag_present_calls_store_event(self):
        """CORRELATED_SELL_EVENT flag → store.store_event() called."""
        mock_store = MagicMock()
        mock_store.available = True
        with patch("src.memory.create_macro_events_store", return_value=mock_store):
            with patch(
                "scripts.portfolio_manager._characterize_macro_event",  # module-level fn
                return_value=(
                    "GLOBAL",
                    "GLOBAL",
                    "",
                    "TRANSIENT",
                    "TARIFF_TRADE",
                    "headline",
                    "",
                ),
            ):
                _store_macro_event_if_detected([self._CORR_FLAG], [])
        mock_store.store_event.assert_called_once()

    def test_forced_reanalysis_true_for_high_structural(self):
        """correlation ≥ 0.40 AND STRUCTURAL → forced_reanalysis=True."""
        mock_store = MagicMock()
        mock_store.available = True
        with patch("src.memory.create_macro_events_store", return_value=mock_store):
            with patch(
                "scripts.portfolio_manager._characterize_macro_event",  # module-level fn
                return_value=(
                    "GLOBAL",
                    "GLOBAL",
                    "",
                    "STRUCTURAL",
                    "REGULATORY_SHIFT",
                    "h",
                    "",
                ),
            ):
                _store_macro_event_if_detected([self._CORR_FLAG], [])
        stored_event = mock_store.store_event.call_args[0][0]
        assert stored_event.forced_reanalysis is True

    def test_forced_reanalysis_false_for_transient(self):
        """TRANSIENT impact → forced_reanalysis=False even if correlation ≥ 0.40."""
        mock_store = MagicMock()
        mock_store.available = True
        with patch("src.memory.create_macro_events_store", return_value=mock_store):
            with patch(
                "scripts.portfolio_manager._characterize_macro_event",  # module-level fn
                return_value=(
                    "GLOBAL",
                    "GLOBAL",
                    "",
                    "TRANSIENT",
                    "TARIFF_TRADE",
                    "h",
                    "",
                ),
            ):
                _store_macro_event_if_detected([self._CORR_FLAG], [])
        stored_event = mock_store.store_event.call_args[0][0]
        assert stored_event.forced_reanalysis is False

    def test_forced_reanalysis_false_for_low_structural_correlation(self):
        """STRUCTURAL but correlation < 0.40 → forced_reanalysis=False."""
        low_corr_flag = (
            "CORRELATED_SELL_EVENT: 5 positions changed verdict within 7d of 2026-03-05"
            " (25% of held positions) — probable macro event."
            " Execute stop-breach SELLs only; review verdict-change SELLs before acting."
        )
        mock_store = MagicMock()
        mock_store.available = True
        with patch("src.memory.create_macro_events_store", return_value=mock_store):
            with patch(
                "scripts.portfolio_manager._characterize_macro_event",  # module-level fn
                return_value=(
                    "GLOBAL",
                    "GLOBAL",
                    "",
                    "STRUCTURAL",
                    "GEOPOLITICAL",
                    "h",
                    "",
                ),
            ):
                _store_macro_event_if_detected([low_corr_flag], [])
        stored_event = mock_store.store_event.call_args[0][0]
        assert stored_event.forced_reanalysis is False

    def test_unparseable_flag_logs_warning_does_not_raise(self):
        """Malformed CORRELATED_SELL_EVENT string → logs warning, returns cleanly."""
        with patch("scripts.portfolio_manager.logger") as mock_log:
            _store_macro_event_if_detected(["CORRELATED_SELL_EVENT: bad format"], [])
        mock_log.warning.assert_called_once()

    def test_store_unavailable_returns_gracefully(self):
        """store.available=False → returns without raising or calling store_event."""
        mock_store = MagicMock()
        mock_store.available = False
        with patch("src.memory.create_macro_events_store", return_value=mock_store):
            with patch(
                "scripts.portfolio_manager._characterize_macro_event",  # module-level fn
                return_value=(
                    "GLOBAL",
                    "GLOBAL",
                    "",
                    "TRANSIENT",
                    "TARIFF_TRADE",
                    "h",
                    "",
                ),
            ):
                _store_macro_event_if_detected([self._CORR_FLAG], [])
        mock_store.store_event.assert_not_called()

    def test_severity_high_when_correlation_above_40pct(self):
        """Correlation ≥ 0.40 → severity='HIGH' in stored MacroEvent."""
        mock_store = MagicMock()
        mock_store.available = True
        with patch("src.memory.create_macro_events_store", return_value=mock_store):
            with patch(
                "scripts.portfolio_manager._characterize_macro_event",  # module-level fn
                return_value=(
                    "GLOBAL",
                    "GLOBAL",
                    "",
                    "TRANSIENT",
                    "TARIFF_TRADE",
                    "h",
                    "",
                ),
            ):
                _store_macro_event_if_detected([self._CORR_FLAG], [])
        stored_event = mock_store.store_event.call_args[0][0]
        assert stored_event.severity == "HIGH"

    def test_event_date_parsed_correctly(self):
        """Event date from flag string is extracted correctly."""
        mock_store = MagicMock()
        mock_store.available = True
        with patch("src.memory.create_macro_events_store", return_value=mock_store):
            with patch(
                "scripts.portfolio_manager._characterize_macro_event",  # module-level fn
                return_value=(
                    "GLOBAL",
                    "GLOBAL",
                    "",
                    "TRANSIENT",
                    "TARIFF_TRADE",
                    "h",
                    "",
                ),
            ):
                _store_macro_event_if_detected([self._CORR_FLAG], [])
        stored_event = mock_store.store_event.call_args[0][0]
        assert stored_event.event_date == "2026-03-05"

    def test_actual_reconciler_flag_format_parses_correctly(self):
        """The exact format produced by compute_portfolio_health() is parseable.

        reconciler.py uses :.0% formatting → '40%' (no decimal); this verifies
        the regex in _store_macro_event_if_detected handles it correctly.
        """
        # Mirror exact f-string from reconciler.py compute_portfolio_health()
        peak_count, window_days, total = 8, 7, 20
        actual_flag = (
            f"CORRELATED_SELL_EVENT: {peak_count} positions changed verdict"
            f" within {window_days}d of 2026-03-05"
            f" ({peak_count / total:.0%} of held"
            f" positions) — probable macro event. Execute stop-breach SELLs"
            f" only; review verdict-change SELLs before acting."
        )
        mock_store = MagicMock()
        mock_store.available = True
        with patch("src.memory.create_macro_events_store", return_value=mock_store):
            with patch(
                "scripts.portfolio_manager._characterize_macro_event",
                return_value=(
                    "GLOBAL",
                    "GLOBAL",
                    "",
                    "TRANSIENT",
                    "TARIFF_TRADE",
                    "h",
                    "",
                ),
            ):
                _store_macro_event_if_detected([actual_flag], [])
        mock_store.store_event.assert_called_once()
        stored = mock_store.store_event.call_args[0][0]
        assert stored.event_date == "2026-03-05"
        assert stored.peak_count == 8
        assert abs(stored.correlation_pct - 0.40) < 0.01

    def test_severity_medium_when_correlation_below_40pct(self):
        """Correlation 25% (< 0.40) → severity='MEDIUM', not 'HIGH'."""
        flag_25pct = (
            "CORRELATED_SELL_EVENT: 5 positions changed verdict within 7d of 2026-03-05"
            " (25% of held positions) — probable macro event."
            " Execute stop-breach SELLs only; review verdict-change SELLs before acting."
        )
        mock_store = MagicMock()
        mock_store.available = True
        with patch("src.memory.create_macro_events_store", return_value=mock_store):
            with patch(
                "scripts.portfolio_manager._characterize_macro_event",
                return_value=(
                    "GLOBAL",
                    "GLOBAL",
                    "",
                    "TRANSIENT",
                    "TARIFF_TRADE",
                    "h",
                    "",
                ),
            ):
                _store_macro_event_if_detected([flag_25pct], [])
        stored = mock_store.store_event.call_args[0][0]
        assert stored.severity == "MEDIUM"
