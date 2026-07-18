"""Tests for the held-position disposition layer (July 2026).

Covers the pure classifier (`classify_disposition` / `reject_confirmed`), the
de-minimis materiality floor, the basis-aware macro-evidence predicate with the
MODEL_REGIME_SHIFT discriminator, and the stop-staleness advisories.

Contract under test: a portfolio action requires portfolio evidence — a
stock-level rejection may trigger refresh, review, replacement analysis, or an
exit, but it must not decide among those alone.
"""

from datetime import datetime, timedelta

from src.ibkr.buy_stability import PriorVerdict
from src.ibkr.models import PortfolioEvidence
from src.ibkr.portfolio_health import (
    compute_portfolio_health,
    is_macro_event_evidence,
)
from src.ibkr.reconciler import reconcile
from src.ibkr.reconciliation_rules import classify_disposition, reject_confirmed
from tests.ibkr.reconciler_cases import (
    _make_analysis,
    _make_hold_item_for_health,
    _make_portfolio,
    _make_position,
    _make_sell_item_on_date,
)


def _prior(verdict: str, days_ago: int, quick: bool = False) -> PriorVerdict:
    return PriorVerdict(
        verdict=verdict,
        analysis_dt=datetime.now() - timedelta(days=days_ago),
        is_quick_mode=quick,
        file_path="prior.json",
    )


def _reject_analysis(**kwargs):
    a = _make_analysis(verdict="DO_NOT_INITIATE", age_days=0, **kwargs)
    a.health_adj = 70.0
    a.growth_adj = 65.0
    return a


class TestClassifyDisposition:
    def test_mandatory_exit_flags_force_executable_sell(self):
        a = _reject_analysis()
        a.evidence = PortfolioEvidence(
            mandatory_exit_flag_types=("OPERATOR_RESTRICTED",)
        )
        d = classify_disposition(a, current_price_local=2100, prior_history=[])
        assert (d.action, d.basis, d.executable) == ("SELL", "MANDATORY_EXIT", True)

    def test_active_tender_is_special_situation_review(self):
        a = _reject_analysis()
        a.m_and_a_status = "ACTIVE_TENDER"
        d = classify_disposition(a, current_price_local=2100, prior_history=[])
        assert (d.action, d.basis) == ("REVIEW", "SPECIAL_SITUATION_REVIEW")

    def test_buy_blocking_evidence_reviews_even_when_confirmed(self):
        """An unreliable gate score cannot confirm-sell — data quality outranks
        the confirmation gate (never punish the stock for the model's doubt)."""
        a = _reject_analysis()
        a.evidence = PortfolioEvidence(
            buy_blocking_flag_types=("GROWTH_SCORE_UNRELIABLE",)
        )
        d = classify_disposition(
            a,
            current_price_local=2100,
            prior_history=[_prior("DO_NOT_INITIATE", 20)],
        )
        assert (d.action, d.basis) == ("REVIEW", "DATA_QUALITY")

    def test_intact_appreciated_is_entry_constraint(self):
        a = _reject_analysis()  # entry 2100
        d = classify_disposition(a, current_price_local=2200, prior_history=[])
        assert (d.action, d.basis) == ("REVIEW", "ENTRY_CONSTRAINT")

    def test_intact_below_entry_with_dni_marker_is_entry_constraint(self):
        a = _reject_analysis()
        a.evidence = PortfolioEvidence(dni_review_candidate=True)
        d = classify_disposition(a, current_price_local=1950, prior_history=[])
        assert d.basis == "ENTRY_CONSTRAINT"

    def test_intact_below_entry_without_marker_is_reassessment(self):
        a = _reject_analysis()
        d = classify_disposition(a, current_price_local=1950, prior_history=[])
        assert d.basis == "THESIS_REASSESSMENT"

    def test_weak_fundamentals_appreciated_is_reassessment_not_constraint(self):
        a = _reject_analysis()
        a.growth_adj = 30.0
        d = classify_disposition(a, current_price_local=2200, prior_history=[])
        assert d.basis == "THESIS_REASSESSMENT"


class TestRejectConfirmed:
    def test_spacing_at_minimum_confirms(self):
        a = _reject_analysis()  # analysis_date = today
        assert reject_confirmed(a, [_prior("DO_NOT_INITIATE", 7)])

    def test_spacing_below_minimum_does_not_confirm(self):
        a = _reject_analysis()
        assert not reject_confirmed(a, [_prior("DO_NOT_INITIATE", 6)])

    def test_intervening_full_mode_non_reject_breaks_confirmation(self):
        """The most recent full-mode prior is a BUY — the thesis recovered
        between rejects, so nothing is confirmed."""
        a = _reject_analysis()
        history = [_prior("DO_NOT_INITIATE", 30), _prior("BUY", 10)]
        assert not reject_confirmed(a, history)

    def test_quick_mode_priors_are_invisible(self):
        """A quick screening BUY between two full rejects neither confirms
        nor breaks — quick runs are screening noise to this gate."""
        a = _reject_analysis()
        history = [_prior("DO_NOT_INITIATE", 20), _prior("BUY", 2, quick=True)]
        assert reject_confirmed(a, history)

    def test_unparseable_analysis_date_never_confirms(self):
        a = _reject_analysis()
        a.analysis_date = "not-a-date"
        assert not reject_confirmed(a, [_prior("DO_NOT_INITIATE", 20)])

    def test_empty_history_never_confirms(self):
        assert not reject_confirmed(_reject_analysis(), [])


class TestDeMinimis:
    def test_sub_floor_soft_reject_becomes_de_minimis_hold(self):
        pos = _make_position(current_price=2100, market_value_usd=200)
        a = _reject_analysis()
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action == "HOLD"
        assert items[0].action_basis == "DE_MINIMIS"
        assert "monitor only" in items[0].reason

    def test_sub_floor_stop_breach_still_sells(self):
        pos = _make_position(current_price=1700, market_value_usd=200)
        a = _make_analysis(stop_price=1900)
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action == "SELL"
        assert items[0].action_basis == "STOP_LOSS"

    def test_compliance_flag_exempts_from_suppression(self):
        """A $200 PFIC position still costs a Form 8621 — per-position burden
        is independent of dollar size, so it stays a visible REVIEW."""
        pos = _make_position(current_price=2100, market_value_usd=200)
        a = _reject_analysis()
        a.evidence = PortfolioEvidence(compliance_flag_types=("PFIC_PROBABLE",))
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action == "REVIEW"

    def test_sub_floor_stale_analysis_becomes_de_minimis_hold(self):
        pos = _make_position(current_price=2100, market_value_usd=200)
        a = _make_analysis(verdict="BUY", age_days=60)
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action == "HOLD"
        assert items[0].action_basis == "DE_MINIMIS"


class TestMacroEventEvidence:
    def test_stop_loss_and_confirmed_failure_are_evidence(self):
        for basis in ("STOP_LOSS", "CONFIRMED_THESIS_FAILURE"):
            item = _make_sell_item_on_date("A.T", "2026-07-01")
            item.action_basis = basis
            assert is_macro_event_evidence(item)

    def test_entry_constraint_is_never_evidence(self):
        item = _make_sell_item_on_date("A.T", "2026-07-01", current_price=2200)
        item.action = "REVIEW"
        item.action_basis = "ENTRY_CONSTRAINT"
        assert not is_macro_event_evidence(item)

    def test_reassessment_counts_only_below_entry(self):
        down = _make_sell_item_on_date("A.T", "2026-07-01", current_price=1950)
        down.action = "REVIEW"
        down.action_basis = "THESIS_REASSESSMENT"
        assert is_macro_event_evidence(down)

        up = _make_sell_item_on_date("B.T", "2026-07-01", current_price=2200)
        up.action = "REVIEW"
        up.action_basis = "THESIS_REASSESSMENT"
        assert not is_macro_event_evidence(up)

    def test_legacy_item_without_basis_uses_sell_predicate(self):
        legacy = _make_sell_item_on_date("A.T", "2026-07-01")
        assert legacy.action_basis is None
        assert is_macro_event_evidence(legacy)
        legacy.action = "REVIEW"
        assert not is_macro_event_evidence(legacy)


class TestModelRegimeShift:
    @staticmethod
    def _health_flags(sell_prices: list[float]):
        recent = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        items = [
            _make_sell_item_on_date(
                f"S{i}.T", recent, conid=100 + i, current_price=price
            )
            for i, price in enumerate(sell_prices)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(4)]
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        return flags, items

    def test_majority_price_up_flips_to_model_shift(self):
        """6 verdict flips, all at/above entry (2100) → analyzer re-rated, the
        market did not move: MODEL_REGIME_SHIFT, no macro event, no demotion."""
        flags, items = self._health_flags([2100, 2150, 2200, 2100, 2300, 2250])
        assert any("MODEL_REGIME_SHIFT" in f for f in flags)
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)
        # No demotion pass ran — SELL items untouched
        assert all(i.action == "SELL" for i in items if i.sell_type == "SOFT_REJECT")

    def test_even_split_stays_correlated(self):
        """50/50 price direction is not majority-up — conservative: keep the
        macro event interpretation."""
        flags, _ = self._health_flags([2100, 2150, 2200, 1900, 1850, 1800])
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)
        assert not any("MODEL_REGIME_SHIFT" in f for f in flags)

    def test_price_down_book_stays_correlated(self):
        flags, _ = self._health_flags([1800] * 6)
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)


class TestRefreshConvergence:
    """The confirmation gate only converges if unconfirmed rejects get the
    re-run that can confirm them — they must reach the TOP refresh bucket."""

    @staticmethod
    def _classify(items):
        from src.ibkr.refresh_service import AnalysisRefreshService

        return AnalysisRefreshService().classify(items, max_age_days=14)

    def test_unconfirmed_reject_review_is_blocking_now(self):
        pos = _make_position(current_price=2100)
        a = _reject_analysis()
        a.growth_adj = 30.0  # weak → THESIS_REASSESSMENT, sell_type HARD_REJECT
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action == "REVIEW"
        summary = self._classify(items)
        assert [r.run_ticker for r in summary.blocking_now] == ["7203.T"]

    def test_entry_constraint_review_is_not_blocking(self):
        """A winner rejected by the entry screen must not burn urgent refresh
        budget — it refreshes on the normal staleness cadence."""
        pos = _make_position(current_price=2200)
        a = _reject_analysis()  # intact + appreciated → ENTRY_CONSTRAINT
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action_basis == "ENTRY_CONSTRAINT"
        summary = self._classify(items)
        assert not summary.blocking_now

    def test_de_minimis_hold_consumes_no_refresh_budget(self):
        pos = _make_position(current_price=2100, market_value_usd=200)
        a = _make_analysis(verdict="BUY", age_days=60)  # stale
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action_basis == "DE_MINIMIS"
        summary = self._classify(items)
        assert not summary.blocking_now
        assert not summary.stale_in_queue


class TestReportRendering:
    """End-to-end format_report smoke over the new item population."""

    @staticmethod
    def _render(items, flags=None):
        from scripts.portfolio_manager import format_report

        return format_report(
            items, _make_portfolio(), portfolio_health_flags=flags or []
        )

    def test_confirmed_sell_renders_with_confirmed_label(self):
        from unittest.mock import patch

        pos = _make_position(current_price=2100)
        a = _reject_analysis()
        a.growth_adj = 30.0
        with patch(
            "src.ibkr.position_evaluator._load_prior_history",
            return_value=[_prior("DO_NOT_INITIATE", 20)],
        ):
            items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        report = self._render(items)
        assert "CONFIRMED THESIS FAILURE" in report

    def test_unconfirmed_review_renders_with_refresh_hint(self):
        pos = _make_position(current_price=2100)
        a = _reject_analysis()
        a.growth_adj = 30.0
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        report = self._render(items)
        assert "refresh analysis before exiting".lower() in report.lower()

    def test_de_minimis_hold_renders_as_monitor_only(self):
        pos = _make_position(current_price=2100, market_value_usd=200)
        a = _reject_analysis()
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        report = self._render(items)
        assert "monitor only" in report

    def test_model_regime_shift_banner_renders(self):
        flags = [
            "MODEL_REGIME_SHIFT: 6 of 6 verdict-flip positions trade at/above"
            " their analysis entry — flips are analyzer-side (re-rating)."
        ]
        report = self._render([], flags=flags)
        assert "MODEL RE-RATING DETECTED" in report
        assert "MACRO ALERT" not in report

    def test_macro_banner_without_stop_sells_does_not_say_execute_stops(self):
        flags = [
            "CORRELATED_SELL_EVENT: 8 positions changed verdict within 7d of"
            " 2026-03-05 (80% of held positions) — probable macro event."
        ]
        report = self._render([], flags=flags)
        assert "Execute stops" not in report
        assert "No executable SELLs" in report

    def test_currency_concentration_table_renders(self):
        portfolio = _make_portfolio()
        portfolio.currency_weights = {"JPY": 60.0, "USD": 40.0}
        from scripts.portfolio_manager import format_report

        report = format_report([], portfolio, portfolio_health_flags=[])
        # Currency block was removed July 2026 — sector + exchange suffice.
        assert "Currency:" not in report


class TestSerializerAdditive:
    def test_action_basis_serialized_for_dashboard(self):
        from src.web.ibkr_dashboard.serializers import serialize_item

        item = _make_sell_item_on_date("7203.T", "2026-07-01")
        item.action_basis = "THESIS_REASSESSMENT"
        payload = serialize_item(item)
        assert payload["action_basis"] == "THESIS_REASSESSMENT"
        assert payload["sell_type"] == "SOFT_REJECT"  # legacy field untouched

    def test_legacy_item_serializes_none_basis(self):
        from src.web.ibkr_dashboard.serializers import serialize_item

        item = _make_sell_item_on_date("7203.T", "2026-07-01")
        assert serialize_item(item)["action_basis"] is None


class TestDeMinimisBoundary:
    def test_exactly_at_floor_is_not_de_minimis(self):
        pos = _make_position(current_price=2100, market_value_usd=300)
        a = _reject_analysis()
        a.growth_adj = 30.0
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action == "REVIEW"

    def test_zero_market_value_does_not_crash(self):
        pos = _make_position(current_price=2100, market_value_usd=0.0)
        a = _reject_analysis()
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action_basis == "DE_MINIMIS"


class TestStopStalenessAdvisory:
    def test_stop_inside_noise_band_is_flagged(self):
        pos = _make_position(current_price=1950)
        a = _make_analysis(verdict="BUY", stop_price=1900)  # 2.6% below
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        hold = items[0]
        assert hold.action == "HOLD"
        assert "inside noise range" in hold.reason

    def test_unrevised_stop_after_big_run_is_ratchet_candidate(self):
        pos = _make_position(current_price=2600)  # +23.8% vs entry 2100
        a = _make_analysis(verdict="BUY", stop_price=1900, target_1=3000)
        # widen drift so the +23.8% run does not trip staleness before HOLD
        items = reconcile(
            [pos], {"7203.T": a}, _make_portfolio(), drift_threshold_pct=30.0
        )
        hold = items[0]
        assert hold.action == "HOLD"
        assert "ratchet candidate" in hold.reason

    def test_normal_stop_gets_no_advisory(self):
        pos = _make_position(current_price=2150)
        a = _make_analysis(verdict="BUY", stop_price=1900)
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert "⚠ stop" not in items[0].reason


class TestDispositionControlMatrix:
    """One row per exit control — the anti-churn contract pinned end to end.

    An intact-score reject loses AUTOMATIC sell authority, nothing more:
    mandatory exits, stop breaches, tender review, data-quality quarantine,
    and confirmed below-gate deterioration retain theirs, and confirmation
    itself requires provably full-mode analyses on both sides.
    """

    # ── executable authority retained ────────────────────────────────────
    def test_broken_full_full_spaced_reject_sells(self):
        a = _reject_analysis()
        a.growth_adj = 30.0  # below gate → thesis-failure class
        d = classify_disposition(
            a,
            current_price_local=1950,
            prior_history=[_prior("DO_NOT_INITIATE", 20)],
        )
        assert (d.action, d.basis, d.executable) == (
            "SELL",
            "CONFIRMED_THESIS_FAILURE",
            True,
        )

    def test_stop_breach_retains_executable_authority(self):
        """Stop-breach handling precedes disposition classification and keeps
        its executable authority regardless of intact scores."""
        pos = _make_position(current_price=1850)  # below stop 1900
        a = _make_analysis(verdict="BUY")
        a.health_adj = 80.0
        a.growth_adj = 70.0
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action == "SELL"
        assert items[0].sell_type == "STOP_BREACH"
        assert items[0].action_basis == "STOP_LOSS"

    def test_missing_scores_are_not_intact_so_confirmation_is_reachable(self):
        a = _reject_analysis()
        a.health_adj = None
        a.growth_adj = None
        d = classify_disposition(
            a,
            current_price_local=1950,
            prior_history=[_prior("DO_NOT_INITIATE", 20)],
        )
        assert d.action == "SELL"

    # ── intact scores strip automatic sell authority ─────────────────────
    def test_intact_at_entry_with_confirming_history_stays_review(self):
        a = _reject_analysis()
        d = classify_disposition(
            a,
            current_price_local=2100,  # exactly at entry
            prior_history=[_prior("DO_NOT_INITIATE", 20)],
        )
        assert (d.action, d.basis) == ("REVIEW", "ENTRY_CONSTRAINT")

    def test_intact_above_entry_with_confirming_history_stays_review(self):
        a = _reject_analysis()
        d = classify_disposition(
            a,
            current_price_local=2400,
            prior_history=[_prior("DO_NOT_INITIATE", 20)],
        )
        assert (d.action, d.basis) == ("REVIEW", "ENTRY_CONSTRAINT")

    def test_intact_below_entry_with_confirming_history_is_price_weakness(self):
        """A price drop alone must not trigger a sell — the stop-loss path
        governs downside for intact names."""
        a = _reject_analysis()
        d = classify_disposition(
            a,
            current_price_local=1700,
            prior_history=[_prior("DO_NOT_INITIATE", 20)],
        )
        assert (d.action, d.basis) == ("REVIEW", "THESIS_REASSESSMENT")
        assert "price weakness" in d.detail
        assert "stop-loss governs" in d.detail

    def test_gate_boundary_exactly_50_counts_as_intact(self):
        a = _reject_analysis()
        a.health_adj = 50.0
        a.growth_adj = 50.0
        d = classify_disposition(
            a,
            current_price_local=2200,
            prior_history=[_prior("DO_NOT_INITIATE", 20)],
        )
        assert d.action == "REVIEW"

    # ── mode authority: provably full on BOTH sides ──────────────────────
    def test_quick_current_never_confirms(self):
        a = _reject_analysis()
        a.growth_adj = 30.0
        a.is_quick_mode = True
        assert not reject_confirmed(a, [_prior("DO_NOT_INITIATE", 20)])
        d = classify_disposition(
            a,
            current_price_local=1950,
            prior_history=[_prior("DO_NOT_INITIATE", 20)],
        )
        assert d.action == "REVIEW"

    def test_unknown_current_never_confirms(self):
        a = _reject_analysis()
        a.growth_adj = 30.0
        a.is_quick_mode = None
        assert not reject_confirmed(a, [_prior("DO_NOT_INITIATE", 20)])

    def test_quick_only_history_never_confirms(self):
        a = _reject_analysis()
        a.growth_adj = 30.0
        assert not reject_confirmed(a, [_prior("DO_NOT_INITIATE", 20, quick=True)])

    def test_unknown_mode_prior_never_confirms(self):
        a = _reject_analysis()
        a.growth_adj = 30.0
        unknown = PriorVerdict(
            verdict="DO_NOT_INITIATE",
            analysis_dt=datetime.now() - timedelta(days=20),
            is_quick_mode=None,  # legacy artifact, run_summary absent
            file_path="prior.json",
        )
        assert not reject_confirmed(a, [unknown])

    # ── refresh routing ──────────────────────────────────────────────────
    def test_data_quality_review_reaches_blocking_now(self):
        """'Re-run before acting' items must get the priority refresh — a
        score-derived SOFT_REJECT stamp would strand them on stale cadence."""
        from src.ibkr.refresh_service import AnalysisRefreshService

        pos = _make_position(current_price=2100)
        a = _reject_analysis()  # intact scores — the trap case
        a.evidence = PortfolioEvidence(
            buy_blocking_flag_types=("HEALTH_SCORE_UNRELIABLE",)
        )
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action_basis == "DATA_QUALITY"
        assert items[0].sell_type == "DATA_QUALITY_REVIEW"
        summary = AnalysisRefreshService().classify(items, max_age_days=14)
        assert [r.run_ticker for r in summary.blocking_now] == ["7203.T"]

    def test_intact_price_down_review_stays_on_staleness_cadence(self):
        from src.ibkr.refresh_service import AnalysisRefreshService

        pos = _make_position(current_price=1950)
        a = _reject_analysis()  # intact, below entry → price weakness
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action_basis == "THESIS_REASSESSMENT"
        assert items[0].sell_type == "SOFT_REJECT"
        summary = AnalysisRefreshService().classify(items, max_age_days=14)
        assert not summary.blocking_now

    # ── macro evidence non-regression ────────────────────────────────────
    def test_intact_price_down_review_still_counts_as_macro_evidence(self):
        """Step 1 must not starve the macro detector: price-down reviews are
        distress evidence even when scores are intact."""
        pos = _make_position(current_price=1950)
        a = _reject_analysis()
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert is_macro_event_evidence(items[0])

    def test_entry_constraint_review_is_never_macro_evidence(self):
        pos = _make_position(current_price=2400)
        a = _reject_analysis()
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action_basis == "ENTRY_CONSTRAINT"
        assert not is_macro_event_evidence(items[0])


class TestReviewRowPresentation:
    """Reviews are not orders: no qty/price/LMT segment, no misleading
    placeholder — a holding line carries the position context instead."""

    @staticmethod
    def _render(items):
        from scripts.portfolio_manager import format_report

        return format_report(items, _make_portfolio(), portfolio_health_flags=[])

    def _review_items(self, current_price=2400):
        pos = _make_position(current_price=current_price)
        a = _reject_analysis()
        return reconcile([pos], {"7203.T": a}, _make_portfolio())

    def test_review_row_has_holding_line_and_no_order_segment(self):
        items = self._review_items()
        assert items[0].action == "REVIEW"
        report = self._render(items)
        assert "holding:   100 shares" in report
        assert "potential exit ~$" in report
        assert "no entry price — re-run analysis" not in report
        review_line = next(
            line for line in report.splitlines() if "REVIEW  7203" in line
        )
        assert "LMT" not in review_line

    def test_entry_constraint_winner_gets_ratchet_advisory(self):
        items = self._review_items(current_price=2600)  # +23.8% over entry
        assert items[0].action_basis == "ENTRY_CONSTRAINT"
        report = self._render(items)
        assert "ratchet candidate" in report

    def test_conditional_bucket_uses_position_market_value(self):
        from src.ibkr.portfolio_presentation import _soft_sell_proceeds_usd

        items = self._review_items()
        assert items[0].cash_impact_usd == 0  # reviews carry no cash impact
        assert _soft_sell_proceeds_usd(items) == 1400.0  # position market value

    def test_hard_reject_review_excluded_from_soft_conditional_bucket(self):
        from src.ibkr.portfolio_presentation import _soft_sell_proceeds_usd

        pos = _make_position(current_price=1950)
        a = _reject_analysis()
        a.growth_adj = 30.0  # broken → HARD_REJECT review
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action == "REVIEW"
        assert items[0].sell_type == "HARD_REJECT"
        assert _soft_sell_proceeds_usd(items) == 0.0

    def test_review_never_enters_cash_timeline(self):
        from src.ibkr.portfolio_presentation import build_cash_timeline

        items = self._review_items()
        assert build_cash_timeline(items) == ()

    def test_watchlist_review_without_position_renders_without_crash(self):
        from src.ibkr.models import ReconciliationItem

        item = ReconciliationItem(
            ticker="9201.T",
            action="REVIEW",
            urgency="MEDIUM",
            reason="Watchlist: stale analysis (age 21d > 14d limit)",
            ibkr_position=None,
            is_watchlist=True,
        )
        report = self._render([item])
        assert "9201" in report
        assert "holding:" not in report


class TestPlanTurnoverLine:
    @staticmethod
    def _render(items, portfolio=None):
        from scripts.portfolio_manager import format_report

        return format_report(
            items, portfolio or _make_portfolio(), portfolio_health_flags=[]
        )

    def test_turnover_line_sums_executable_only(self):
        from unittest.mock import patch

        pos = _make_position(current_price=2100)
        sell_a = _reject_analysis()
        sell_a.growth_adj = 30.0
        with patch(
            "src.ibkr.position_evaluator._load_prior_history",
            return_value=[_prior("DO_NOT_INITIATE", 20)],
        ):
            items = reconcile([pos], {"7203.T": sell_a}, _make_portfolio())
        assert items[0].action == "SELL"
        report = self._render(items)
        assert "Plan turnover:" in report
        assert "executable sells ~$1,400" in report
        assert "% of NAV" in report

    def test_review_notional_excluded_from_turnover(self):
        pos = _make_position(current_price=2400)
        a = _reject_analysis()
        items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        assert items[0].action == "REVIEW"
        report = self._render(items)
        assert "Plan turnover:" not in report

    def test_zero_nav_does_not_crash(self):
        from unittest.mock import patch

        pos = _make_position(current_price=2100)
        a = _reject_analysis()
        a.growth_adj = 30.0
        with patch(
            "src.ibkr.position_evaluator._load_prior_history",
            return_value=[_prior("DO_NOT_INITIATE", 20)],
        ):
            items = reconcile([pos], {"7203.T": a}, _make_portfolio())
        portfolio = _make_portfolio(value=0)
        report = self._render(items, portfolio)
        assert "Plan turnover:" in report
        assert "% of NAV" not in report
