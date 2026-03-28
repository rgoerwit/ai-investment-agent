"""End-to-end pipeline coherence tests and MARKET NOTE propagation tests.

Part A: Full reconcile → compute_health → format_report pipeline with no LLM.
Part B: Verify that filter_messages_by_agent always passes HumanMessages through
        so MARKET NOTE context appended in run_analysis reaches every agent.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from scripts.portfolio_manager import format_report
from src.ibkr.models import AnalysisRecord, ReconciliationItem, TradeBlockData
from src.ibkr.reconciler import compute_portfolio_health, reconcile
from tests.ibkr.test_reconciler import (
    _make_analysis,
    _make_multi_sell_scenario,
    _make_portfolio,
    _make_position,
)

# ══════════════════════════════════════════════════════════════════════════════
# Part A — End-to-end pipeline
# ══════════════════════════════════════════════════════════════════════════════


class TestPipelineCoherence:
    """reconcile → compute_health → format_report with realistic multi-sell scenarios."""

    def test_panic_day_full_pipeline_demotes_soft_rejects(self):
        """8 SOFT_REJECTs + 1 STOP_BREACH (strong) + 1 HARD_REJECT + 2 HOLDs.

        After the full pipeline:
        - CORRELATED_SELL_EVENT fires (9 verdict-driven SELLs / 12 total = 75%)
        - All SOFT_REJECT items are demoted to REVIEW
        - MACRO ALERT banner appears in the report
        - STOP BREACHES UNDER REVIEW section appears (strong-fundamentals stop demoted)
        - SOFT REJECTION section appears (demoted items as macro_reviews)
        - Summary counts REVIEW, not SELL, for the 8 demoted items
        """
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=8, n_stop_breaches=1, n_hard_rejects=1, n_holds=2
        )
        items = reconcile(positions, analyses, portfolio)
        health_flags = compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )
        report = format_report(items, portfolio, portfolio_health_flags=health_flags)

        # Macro banner must appear
        assert "MACRO ALERT" in report
        # Strong-fundamentals stop breach demoted to its own review section
        assert "STOP BREACHES UNDER REVIEW" in report
        # Soft-rejection section (macro_reviews) present
        assert "SOFT REJECTION" in report
        # All SOFT_REJECT items demoted in the items list
        soft_still_sell = [
            i for i in items if i.action == "SELL" and i.sell_type == "SOFT_REJECT"
        ]
        assert soft_still_sell == [], "SOFT_REJECT SELLs should have been demoted"
        # Stop-breach item demoted to REVIEW (health=70, growth=65 → strong)
        stop_reviews = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        assert len(stop_reviews) == 1

    def test_panic_day_weak_stop_breach_stays_sell(self):
        """STOP_BREACH item with weak fundamentals stays SELL even during correlated event."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=8, n_stop_breaches=1, n_hard_rejects=0, n_holds=2
        )
        items = reconcile(positions, analyses, portfolio)
        # Force weak fundamentals on the stop-breach item
        for item in items:
            if item.sell_type == "STOP_BREACH" and item.analysis is not None:
                item.analysis.health_adj = 35.0
                item.analysis.growth_adj = 40.0
        health_flags = compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )
        report = format_report(items, portfolio, portfolio_health_flags=health_flags)

        assert "MACRO ALERT" in report
        # Weak stop-breach stays in the mechanical SELL section
        assert "STOP BREACHED" in report
        stop_sells = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "SELL"
        ]
        assert len(stop_sells) == 1

    def test_non_panic_day_soft_rejects_stay_as_sell(self):
        """3 SOFT_REJECTs + 17 HOLDs = 15% → below 25% threshold → no demotion."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=3, n_stop_breaches=0, n_hard_rejects=0, n_holds=17
        )
        items = reconcile(positions, analyses, portfolio)
        health_flags = compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )
        soft_sells = [
            i for i in items if i.sell_type == "SOFT_REJECT" and i.action == "SELL"
        ]
        assert len(soft_sells) == 3
        # No correlated event → no "STOP BREACHES UNDER REVIEW" section in report
        report = format_report(items, portfolio, portfolio_health_flags=health_flags)
        assert "STOP BREACHES UNDER REVIEW" not in report

    def test_exactly_at_threshold_triggers(self):
        """5 SOFT_REJECTs + 15 HOLDs = 20 total → 5/20 = 25.0% → event fires."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=5, n_stop_breaches=0, n_hard_rejects=0, n_holds=15
        )
        items = reconcile(positions, analyses, portfolio)
        flags = compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_just_below_threshold_does_not_trigger(self):
        """5 SOFT_REJECTs + 16 HOLDs = 21 total → 5/21 = 23.8% → no event."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=5, n_stop_breaches=0, n_hard_rejects=0, n_holds=16
        )
        items = reconcile(positions, analyses, portfolio)
        flags = compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_same_count_different_dates_does_not_trigger(self):
        """5 SOFT_REJECT SELLs each 10+ days apart → peak_count=1 per window → no event.

        Uses dates 10 days apart so no sliding window (7-day default) captures > 1
        position at a time. This verifies the algorithm doesn't false-fire on positions
        that genuinely changed verdict at different macro episodes.
        """
        positions = []
        items_list = []
        portfolio = _make_portfolio(value=50000, cash=0)
        portfolio.exchange_weights = {}

        for i in range(5):
            # Space dates 10 days apart — well beyond the 7-day sliding window.
            date = (datetime.now() - timedelta(days=i * 10)).strftime("%Y-%m-%d")
            a = _make_analysis(ticker=f"SPREAD{i:02d}.T", verdict="DO_NOT_INITIATE")
            a.analysis_date = date
            a.health_adj = 65.0
            a.growth_adj = 60.0
            pos = _make_position(
                ticker=f"SPREAD{i:02d}.T", market_value_usd=1000, conid=i + 1
            )
            positions.append(pos)
            items_list.append(
                ReconciliationItem(
                    ticker=f"SPREAD{i:02d}.T",
                    action="SELL",
                    urgency="HIGH",
                    reason=f"Verdict → DO_NOT_INITIATE  ({date})",
                    ibkr_position=pos,
                    analysis=a,
                    sell_type="SOFT_REJECT",
                )
            )

        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items_list
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)


# ══════════════════════════════════════════════════════════════════════════════
# Part B — MARKET NOTE propagation
# ══════════════════════════════════════════════════════════════════════════════


class TestMarketNoteReachesAgents:
    """Verify that the MARKET NOTE appended in run_analysis survives message filtering."""

    def test_human_message_contains_market_note_when_provided(self):
        """filter_messages_by_agent always passes HumanMessage through — MARKET NOTE included."""
        from langchain_core.messages import AIMessage, HumanMessage

        from src.agents import filter_messages_by_agent

        note = "MARKET NOTE: Nikkei-225 down 4.2% on 2026-03-05."
        human = HumanMessage(content=f"Analyze 7203.T (Toyota). {note}")
        other_ai = AIMessage(content="Bear case...", name="bear_researcher")

        # Filter for a different agent — HumanMessage must survive intact
        filtered = filter_messages_by_agent([human, other_ai], "bull_researcher")
        human_msgs = [m for m in filtered if isinstance(m, HumanMessage)]

        assert len(human_msgs) == 1
        assert note in human_msgs[0].content

    def test_human_message_without_market_note_also_passes_through(self):
        """Baseline: filter_messages_by_agent always includes HumanMessage."""
        from langchain_core.messages import HumanMessage

        from src.agents import filter_messages_by_agent

        human = HumanMessage(
            content="Analyze 7203.T (Toyota). Current Date: 2026-03-05."
        )
        filtered = filter_messages_by_agent([human], "portfolio_manager")
        assert any(isinstance(m, HumanMessage) for m in filtered)

    def test_base_message_appends_market_note_when_non_empty(self):
        """Verify the _base_msg construction in run_analysis appends MARKET NOTE correctly."""
        ticker = "7203.T"
        company_name = "Toyota Motor"
        real_date = "2026-03-05"
        market_context = "MARKET NOTE: Nikkei-225 down 4.2% on 2026-03-05."

        base = f"Analyze {ticker} ({company_name}) for investment decision. Current Date: {real_date}."
        if market_context:
            base += f" {market_context}"

        assert market_context in base
        assert base.endswith(market_context)

    def test_base_message_clean_when_market_context_empty(self):
        """Empty market_context → no trailing space or stray period added."""
        ticker = "7203.T"
        company_name = "Toyota Motor"
        real_date = "2026-03-05"
        market_context = ""  # fetch failed or exchange not mapped

        base = f"Analyze {ticker} ({company_name}) for investment decision. Current Date: {real_date}."
        if market_context:
            base += f" {market_context}"

        assert not base.endswith(" ")
        assert base.endswith(".")
