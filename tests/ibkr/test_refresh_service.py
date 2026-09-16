from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from src.ibkr.models import PortfolioEvidence, ReconciliationItem
from src.ibkr.reconciler import reconcile
from src.ibkr.refresh_service import (
    AnalysisFreshnessRow,
    AnalysisFreshnessSummary,
    AnalysisRefreshService,
    RefreshActivity,
    RefreshExecutionOptions,
    RefreshPlanOptions,
    _UnrepairedRefresh,
)
from tests.ibkr.reconciler_cases import _make_analysis, _make_portfolio, _make_position


def _make_review_item(
    ticker: str,
    *,
    age_days: int = 20,
    reason: str = "Analysis too old",
    sell_type: str | None = None,
    action_basis: str | None = None,
    held: bool = True,
) -> ReconciliationItem:
    analysis = _make_analysis(ticker=ticker, age_days=age_days)
    if action_basis == "DATA_QUALITY":
        analysis.evidence = PortfolioEvidence(
            buy_blocking_flag_types=("TEST_EVIDENCE_UNAVAILABLE",)
        )
    return ReconciliationItem(
        ticker=ticker,
        action="REVIEW",
        reason=reason,
        urgency="MEDIUM",
        ibkr_position=_make_position(ticker=ticker) if held else None,
        analysis=analysis,
        sell_type=sell_type,
        action_basis=action_basis,
    )


def _freshness_row(
    ticker: str, *, bucket: str, urgency: str = "MEDIUM"
) -> AnalysisFreshnessRow:
    return AnalysisFreshnessRow(
        display_ticker=ticker,
        run_ticker=ticker,
        bucket=bucket,
        reason_family="test",
        reason_text="test",
        action="REVIEW",
        action_basis=None,
        urgency=urgency,
        age_days=20,
        expires_date=None,
        days_until_due=None,
    )


def _plan_options(
    *,
    policy: Literal["off", "blocking", "proactive"] = "proactive",
    limit: int = 10,
    scheduler_state_path: Path | None = None,
) -> RefreshPlanOptions:
    return RefreshPlanOptions(
        policy=policy,
        limit=limit,
        show_recommendations=False,
        read_only=False,
        max_age_days=14,
        scheduler_state_path=scheduler_state_path,
    )


class TestResolvePolicy:
    def test_explicit_policy_wins(self):
        service = AnalysisRefreshService()
        assert (
            service.resolve_policy(
                explicit_policy="proactive",
                refresh_stale=False,
                recommend=False,
                read_only=False,
            )
            == "proactive"
        )

    def test_refresh_stale_maps_to_blocking(self):
        service = AnalysisRefreshService()
        assert (
            service.resolve_policy(
                explicit_policy=None,
                refresh_stale=True,
                recommend=False,
                read_only=False,
            )
            == "blocking"
        )

    def test_recommend_defaults_to_weighted_proactive(self):
        service = AnalysisRefreshService()
        assert (
            service.resolve_policy(
                explicit_policy=None,
                refresh_stale=False,
                recommend=True,
                read_only=False,
            )
            == "proactive"
        )

    def test_recommend_read_only_stays_off(self):
        service = AnalysisRefreshService()
        assert (
            service.resolve_policy(
                explicit_policy=None,
                refresh_stale=False,
                recommend=True,
                read_only=True,
            )
            == "off"
        )

    def test_default_is_off(self):
        service = AnalysisRefreshService()
        assert (
            service.resolve_policy(
                explicit_policy=None,
                refresh_stale=False,
                recommend=False,
                read_only=False,
            )
            == "off"
        )


class TestClassify:
    def test_generic_stale_review_uses_normal_cycle(self):
        service = AnalysisRefreshService()
        summary = service.classify([_make_review_item("7203.T")], max_age_days=14)
        assert summary.blocking_now == []
        assert [row.run_ticker for row in summary.due_soon] == ["7203.T"]

    def test_stale_sell_goes_to_stale_in_queue(self):
        service = AnalysisRefreshService()
        item = ReconciliationItem(
            ticker="5285.T",
            action="SELL",
            reason="Stop breach",
            urgency="HIGH",
            ibkr_position=_make_position(ticker="5285.T"),
            analysis=_make_analysis(ticker="5285.T", age_days=21),
            sell_type="STOP_BREACH",
        )
        summary = service.classify([item], max_age_days=14)
        assert [row.run_ticker for row in summary.stale_in_queue] == ["5285.T"]

    def test_stale_trim_goes_to_stale_in_queue(self):
        service = AnalysisRefreshService()
        item = ReconciliationItem(
            ticker="6758.T",
            action="TRIM",
            reason="Reduce concentration",
            urgency="MEDIUM",
            ibkr_position=_make_position(ticker="6758.T"),
            analysis=_make_analysis(ticker="6758.T", age_days=21),
        )
        summary = service.classify([item], max_age_days=14)
        assert [row.run_ticker for row in summary.stale_in_queue] == ["6758.T"]

    def test_near_expiry_hold_goes_to_due_soon(self):
        service = AnalysisRefreshService()
        item = ReconciliationItem(
            ticker="GTT.PA",
            action="HOLD",
            reason="Position OK",
            urgency="LOW",
            ibkr_position=_make_position(ticker="GTT.PA"),
            analysis=_make_analysis(ticker="GTT.PA", age_days=8),
        )
        summary = service.classify([item], max_age_days=14)
        assert [row.run_ticker for row in summary.due_soon] == ["GTT.PA"]

    def test_unheld_review_candidate_goes_to_candidate_blocked(self):
        service = AnalysisRefreshService()
        summary = service.classify(
            [_make_review_item("7203.T", held=False)],
            max_age_days=14,
        )
        assert [row.run_ticker for row in summary.candidate_blocked] == ["7203.T"]

    def test_soft_reject_excluded_from_blocking(self):
        service = AnalysisRefreshService()
        summary = service.classify(
            [
                _make_review_item(
                    "7203.T",
                    reason="Verdict → DO_NOT_INITIATE  (2026-03-05)",
                    sell_type="SOFT_REJECT",
                )
            ],
            max_age_days=14,
        )
        assert summary.blocking_now == []
        assert [row.run_ticker for row in summary.stale_in_queue] == ["7203.T"]

    def test_fresh_soft_reject_is_visible_but_not_auto_refreshed(self):
        service = AnalysisRefreshService()
        summary = service.classify(
            [
                _make_review_item(
                    "7203.T",
                    age_days=0,
                    reason="Price weakness with intact fundamentals",
                    sell_type="SOFT_REJECT",
                    action_basis="THESIS_REASSESSMENT",
                )
            ],
            max_age_days=14,
        )

        assert summary.blocking_now == []
        assert [row.run_ticker for row in summary.operator_review] == ["7203.T"]

    @pytest.mark.parametrize(
        ("action_basis", "sell_type"),
        (
            ("ENTRY_CONSTRAINT", None),
            ("CAPITAL_ALLOCATION", None),
            ("THESIS_REASSESSMENT", "SOFT_REJECT"),
        ),
    )
    def test_operator_decision_point_joins_normal_cycle_when_due(
        self, action_basis, sell_type
    ):
        service = AnalysisRefreshService()
        summary = service.classify(
            [
                _make_review_item(
                    "7203.T",
                    age_days=7,
                    action_basis=action_basis,
                    sell_type=sell_type,
                )
            ],
            max_age_days=14,
        )

        assert [row.run_ticker for row in summary.due_soon] == ["7203.T"]
        assert summary.operator_review == []

    def test_fresh_hold_goes_to_fresh(self):
        service = AnalysisRefreshService()
        item = ReconciliationItem(
            ticker="ASML.AS",
            action="HOLD",
            reason="Position OK",
            urgency="LOW",
            ibkr_position=_make_position(ticker="ASML.AS"),
            analysis=_make_analysis(ticker="ASML.AS", age_days=1),
        )
        summary = service.classify([item], max_age_days=14)
        assert [row.run_ticker for row in summary.fresh] == ["ASML.AS"]


class TestPlan:
    def test_policy_off_skips_all(self):
        service = AnalysisRefreshService()
        summary = service.classify([_make_review_item("7203.T")], max_age_days=14)
        activity = service.plan(
            summary,
            options=RefreshPlanOptions(
                policy="off",
                limit=10,
                show_recommendations=False,
                read_only=False,
                max_age_days=14,
            ),
        )
        assert activity.queued == []
        assert activity.skipped_due_to_policy == ["7203.T"]

    def test_proactive_includes_due_soon(self):
        service = AnalysisRefreshService()
        item = ReconciliationItem(
            ticker="GTT.PA",
            action="HOLD",
            reason="Position OK",
            urgency="LOW",
            ibkr_position=_make_position(ticker="GTT.PA"),
            analysis=_make_analysis(ticker="GTT.PA", age_days=8),
        )
        summary = service.classify([item], max_age_days=14)
        activity = service.plan(
            summary,
            options=RefreshPlanOptions(
                policy="proactive",
                limit=10,
                show_recommendations=False,
                read_only=False,
                max_age_days=14,
            ),
        )
        assert activity.queued == ["GTT.PA"]

    def test_blocking_orders_due_soon_behind_urgent_work(self):
        """Blocking policy orders the streams; it does not drop the cycle one.

        The earlier contract asserted due-soon work was *excluded* whenever any
        urgent row existed, which idled the rest of the budget every run.
        """
        service = AnalysisRefreshService()
        due_soon_item = ReconciliationItem(
            ticker="GTT.PA",
            action="HOLD",
            reason="Position OK",
            urgency="LOW",
            ibkr_position=_make_position(ticker="GTT.PA"),
            analysis=_make_analysis(ticker="GTT.PA", age_days=8),
        )
        summary = service.classify(
            [
                _make_review_item("7203.T", action_basis="THESIS_REASSESSMENT"),
                due_soon_item,
            ],
            max_age_days=14,
        )
        activity = service.plan(
            summary,
            options=RefreshPlanOptions(
                policy="blocking",
                limit=10,
                show_recommendations=False,
                read_only=False,
                max_age_days=14,
            ),
        )
        assert activity.queued == ["7203.T", "GTT.PA"]

    def test_blocking_falls_back_to_due_soon_when_no_urgent_work_exists(self):
        service = AnalysisRefreshService()
        item = ReconciliationItem(
            ticker="GTT.PA",
            action="HOLD",
            reason="Position OK",
            urgency="LOW",
            ibkr_position=_make_position(ticker="GTT.PA"),
            analysis=_make_analysis(ticker="GTT.PA", age_days=8),
        )
        summary = service.classify([item], max_age_days=14)

        activity = service.plan(
            summary,
            options=RefreshPlanOptions(
                policy="blocking",
                limit=10,
                show_recommendations=False,
                read_only=False,
                max_age_days=14,
            ),
        )

        assert activity.queued == ["GTT.PA"]

    def test_stale_soft_reject_review_is_refreshed(self):
        """A CORRELATED_SELL_EVENT demotes a stale SOFT_REJECT sell to REVIEW; the
        analysis is still stale, so it must be refreshed (the macro demotion must
        not silently suppress refresh)."""
        service = AnalysisRefreshService()
        summary = service.classify(
            [
                _make_review_item(
                    "7203.T",
                    reason="Verdict → DO_NOT_INITIATE  (2026-03-05)",
                    sell_type="SOFT_REJECT",
                )
            ],
            max_age_days=14,
        )
        assert [row.run_ticker for row in summary.stale_in_queue] == ["7203.T"]
        activity = service.plan(
            summary,
            options=RefreshPlanOptions(
                policy="blocking",
                limit=10,
                show_recommendations=False,
                read_only=False,
                max_age_days=14,
            ),
        )
        assert activity.queued == ["7203.T"]

    def test_read_only_moves_queue_to_skipped(self):
        service = AnalysisRefreshService()
        summary = service.classify(
            [_make_review_item("7203.T", action_basis="THESIS_REASSESSMENT")],
            max_age_days=14,
        )
        activity = service.plan(
            summary,
            options=RefreshPlanOptions(
                policy="blocking",
                limit=10,
                show_recommendations=False,
                read_only=True,
                max_age_days=14,
            ),
        )
        assert activity.queued == []
        assert activity.skipped_read_only == ["7203.T"]

    def test_limit_preserves_deterministic_order(self):
        service = AnalysisRefreshService()
        summary = service.classify(
            [
                _make_review_item("7203.T", action_basis="THESIS_REASSESSMENT"),
                _make_review_item("6758.T", action_basis="THESIS_REASSESSMENT"),
            ],
            max_age_days=14,
        )
        activity = service.plan(
            summary,
            options=RefreshPlanOptions(
                policy="blocking",
                limit=1,
                show_recommendations=False,
                read_only=False,
                max_age_days=14,
            ),
        )
        assert activity.queued == ["6758.T"]
        assert activity.skipped_due_to_limit == ["7203.T"]

    def test_ticker_subset_filters_candidates(self):
        service = AnalysisRefreshService()
        summary = service.classify(
            [
                _make_review_item("7203.T", action_basis="THESIS_REASSESSMENT"),
                _make_review_item("6758.T", action_basis="THESIS_REASSESSMENT"),
            ],
            max_age_days=14,
        )
        activity = service.plan(
            summary,
            options=RefreshPlanOptions(
                policy="blocking",
                limit=10,
                show_recommendations=False,
                read_only=False,
                max_age_days=14,
                ticker_subset=frozenset({"6758.T"}),
            ),
        )
        assert activity.queued == ["6758.T"]

    def test_candidates_only_when_show_recommendations(self):
        service = AnalysisRefreshService()
        summary = service.classify(
            [_make_review_item("7203.T", held=False)],
            max_age_days=14,
        )
        activity = service.plan(
            summary,
            options=RefreshPlanOptions(
                policy="blocking",
                limit=10,
                show_recommendations=False,
                read_only=False,
                max_age_days=14,
            ),
        )
        assert activity.queued == []

    @pytest.mark.parametrize("analysis_available", [True, False])
    def test_broker_data_quality_review_does_not_schedule_stock_analysis(
        self, analysis_available
    ):
        service = AnalysisRefreshService()
        position = _make_position(ticker="2173.T").model_copy(
            update={
                "valuation_valid": False,
                "valuation_issue": "broker value units could not be verified",
            }
        )
        items = reconcile(
            [position],
            ({"2173.T": _make_analysis(ticker="2173.T")} if analysis_available else {}),
            _make_portfolio(),
        )

        summary = service.classify(items, max_age_days=14)
        activity = service.plan(summary, options=_plan_options(policy="blocking"))

        assert [row.run_ticker for row in summary.operator_review] == ["2173.T"]
        assert summary.blocking_now == []
        assert activity.queued == []

    @pytest.mark.parametrize("broker_defect", ["short", "invalid_valuation"])
    @pytest.mark.parametrize("settled", [True, False])
    def test_broker_defect_precedes_aging_analysis_refresh(
        self, broker_defect, settled
    ):
        service = AnalysisRefreshService()
        position = _make_position(ticker="2173.T")
        if broker_defect == "short":
            position.quantity = -1
        else:
            position.valuation_valid = False
            position.valuation_issue = "broker value units could not be verified"
        analysis = _make_analysis(ticker="2173.T", age_days=30)
        flag = "LIQUIDITY_HARD_FAIL" if settled else "TEST_EVIDENCE_UNAVAILABLE"
        analysis.evidence = PortfolioEvidence(
            buy_blocking_flag_types=(flag,),
            settled_reject_flag_types=(flag,) if settled else (),
        )
        items = reconcile([position], {"2173.T": analysis}, _make_portfolio())
        summary = service.classify(items, max_age_days=14)
        activity = service.plan(summary, options=_plan_options(policy="blocking"))

        assert items[0].action_basis == "DATA_QUALITY"
        assert service._refresh_repairable(items[0]) is False
        assert [row.run_ticker for row in summary.operator_review] == ["2173.T"]
        assert summary.due_soon == []
        assert summary.blocking_now == []
        assert activity.queued == []

    def test_fresh_settled_liquidity_reject_does_not_schedule_paid_analysis(self):
        """The 2173.T regression: a fresh measured failure is not an evidence gap.

        LIQUIDITY_HARD_FAIL is minted AUTO_REJECT from a measured turnover, so
        no further research can change it. Treating every blocks_buy flag as
        repairable put this position in the urgent stream on every invocation
        and — under the blocking policy, which only consults the cycle stream
        when urgent is empty — starved all normal-cycle refreshes behind it.
        """
        service = AnalysisRefreshService()
        analysis = _make_analysis(
            ticker="2173.T", verdict="DO_NOT_INITIATE", age_days=0
        )
        analysis.health_adj = 30.0
        analysis.growth_adj = 20.0
        analysis.evidence = PortfolioEvidence(
            buy_blocking_flag_types=("LIQUIDITY_HARD_FAIL",),
            settled_reject_flag_types=("LIQUIDITY_HARD_FAIL",),
        )
        items = reconcile(
            [_make_position(ticker="2173.T")], {"2173.T": analysis}, _make_portfolio()
        )
        assert items[0].action_basis == "DATA_QUALITY"

        summary = service.classify(items, max_age_days=14)
        cycle_summary = AnalysisFreshnessSummary(
            blocking_now=list(summary.blocking_now),
            operator_review=list(summary.operator_review),
            due_soon=[_freshness_row("CYCLE.A", bucket="due_soon")],
        )
        activity = service.plan(cycle_summary, options=_plan_options(policy="blocking"))

        assert service._refresh_repairable(items[0]) is False
        assert [row.run_ticker for row in summary.operator_review] == ["2173.T"]
        assert summary.blocking_now == []
        # The freed urgent stream no longer starves normal-cycle work.
        assert activity.queued == ["CYCLE.A"]

    def test_aging_settled_liquidity_reject_rejoins_normal_cycle(self):
        service = AnalysisRefreshService()
        analysis = _make_analysis(
            ticker="2173.T", verdict="DO_NOT_INITIATE", age_days=8
        )
        analysis.health_adj = 30.0
        analysis.growth_adj = 20.0
        analysis.evidence = PortfolioEvidence(
            buy_blocking_flag_types=("LIQUIDITY_HARD_FAIL",),
            settled_reject_flag_types=("LIQUIDITY_HARD_FAIL",),
        )
        items = reconcile(
            [_make_position(ticker="2173.T")],
            {"2173.T": analysis},
            _make_portfolio(),
        )

        summary = service.classify(items, max_age_days=14)
        activity = service.plan(summary, options=_plan_options(policy="blocking"))

        assert [row.run_ticker for row in summary.due_soon] == ["2173.T"]
        assert summary.operator_review == []
        assert activity.queued == ["2173.T"]

    def test_settled_and_indeterminate_flags_together_remain_repairable(self):
        """A settled flag must not mask a real gap sitting beside it."""
        service = AnalysisRefreshService()
        analysis = _make_analysis(
            ticker="2173.T", verdict="DO_NOT_INITIATE", age_days=0
        )
        analysis.health_adj = 30.0
        analysis.growth_adj = 20.0
        analysis.evidence = PortfolioEvidence(
            buy_blocking_flag_types=(
                "LIQUIDITY_HARD_FAIL",
                "MANAGEMENT_GUIDANCE_EVIDENCE_GAP",
            ),
            settled_reject_flag_types=("LIQUIDITY_HARD_FAIL",),
        )
        items = reconcile(
            [_make_position(ticker="2173.T")], {"2173.T": analysis}, _make_portfolio()
        )

        summary = service.classify(items, max_age_days=14)
        activity = service.plan(summary, options=_plan_options(policy="blocking"))

        assert service._refresh_repairable(items[0]) is True
        assert activity.queued == ["2173.T"]

    def test_operator_short_position_review_does_not_schedule_stock_analysis(self):
        service = AnalysisRefreshService()
        position = _make_position(ticker="2173.T", quantity=-10)
        items = reconcile(
            [position],
            {"2173.T": _make_analysis(ticker="2173.T")},
            _make_portfolio(),
        )

        summary = service.classify(items, max_age_days=14)
        activity = service.plan(summary, options=_plan_options(policy="blocking"))

        assert items[0].action_basis == "DATA_QUALITY"
        assert [row.run_ticker for row in summary.operator_review] == ["2173.T"]
        assert activity.queued == []

    def test_blocking_policy_tops_up_spare_capacity_with_cycle_work(self):
        """Strict priority must bound ordering, not throughput.

        Passing only the urgent stream wasted every slot the urgent queue did
        not fill, so a portfolio with one permanently urgent row refreshed
        exactly one analysis per run while 64 due-soon rows waited forever.
        """
        service = AnalysisRefreshService()
        summary = AnalysisFreshnessSummary(
            blocking_now=[_freshness_row("URGENT.A", bucket="blocking_now")],
            due_soon=[
                _freshness_row("CYCLE.A", bucket="due_soon"),
                _freshness_row("CYCLE.B", bucket="due_soon"),
            ],
        )

        activity = service.plan(summary, options=_plan_options(policy="blocking"))

        assert activity.queued == ["URGENT.A", "CYCLE.A", "CYCLE.B"]

    def test_blocking_policy_still_serves_urgent_first_under_a_tight_limit(self):
        service = AnalysisRefreshService()
        summary = AnalysisFreshnessSummary(
            blocking_now=[_freshness_row("URGENT.A", bucket="blocking_now")],
            due_soon=[_freshness_row("CYCLE.A", bucket="due_soon")],
        )

        activity = service.plan(
            summary, options=_plan_options(policy="blocking", limit=1)
        )

        assert activity.queued == ["URGENT.A"]
        assert activity.skipped_due_to_limit == ["CYCLE.A"]

    def test_work_conserving_when_one_stream_is_empty(self):
        service = AnalysisRefreshService()
        only_cycle = AnalysisFreshnessSummary(
            due_soon=[_freshness_row("CYCLE.A", bucket="due_soon")]
        )
        only_urgent = AnalysisFreshnessSummary(
            blocking_now=[_freshness_row("URGENT.A", bucket="blocking_now")]
        )

        assert service.plan(only_cycle, options=_plan_options(limit=1)).queued == [
            "CYCLE.A"
        ]
        assert service.plan(only_urgent, options=_plan_options(limit=1)).queued == [
            "URGENT.A"
        ]

    @pytest.mark.asyncio
    async def test_failed_urgent_candidate_cools_down_without_blocking_cycle(
        self, tmp_path
    ):
        service = AnalysisRefreshService()
        state_path = tmp_path / "refresh-state.json"
        summary = AnalysisFreshnessSummary(
            blocking_now=[_freshness_row("URGENT.A", bucket="blocking_now")],
            due_soon=[_freshness_row("CYCLE.A", bucket="due_soon")],
        )
        initial = service.plan(
            summary, options=_plan_options(limit=1, scheduler_state_path=state_path)
        )
        assert initial.queued == ["URGENT.A"]

        async def failing_runner(**kwargs):
            return None

        await service.execute(
            initial,
            execution=RefreshExecutionOptions(quick_mode=True),
            run_analysis_fn=failing_runner,
            save_results_fn=lambda *args, **kwargs: Path("unused.json"),
        )

        retry = service.plan(
            summary, options=_plan_options(limit=1, scheduler_state_path=state_path)
        )
        assert retry.queued == ["CYCLE.A"]
        assert retry.skipped_due_to_cooldown == ["URGENT.A"]
        assert "URGENT.A" in retry.skipped_due_to_failure_backoff
        assert retry.skipped_due_to_unrepaired == {}

    @pytest.mark.asyncio
    async def test_corrupt_scheduler_state_is_rebuilt_after_the_next_attempt(
        self, tmp_path, monkeypatch
    ):
        """A clobbered cursor must recover from a clean service position and
        be atomically replaced after the next attempted refresh."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr("src.persistence._maybe_save_rejection_record", AsyncMock())
        service = AnalysisRefreshService()
        state_path = tmp_path / "refresh-state.json"
        state_path.write_text("not json", encoding="utf-8")
        summary = AnalysisFreshnessSummary(
            blocking_now=[_freshness_row("URGENT.A", bucket="blocking_now")],
            due_soon=[_freshness_row("CYCLE.A", bucket="due_soon")],
        )

        activity = service.plan(
            summary, options=_plan_options(limit=1, scheduler_state_path=state_path)
        )

        assert activity.queued == ["URGENT.A"]

        async def successful_runner(**kwargs):
            return {"ticker": "URGENT.A"}

        await service.execute(
            activity,
            execution=RefreshExecutionOptions(quick_mode=True),
            run_analysis_fn=successful_runner,
            save_results_fn=lambda *args, **kwargs: tmp_path / "saved.json",
        )

        rebuilt = json.loads(state_path.read_text(encoding="utf-8"))
        assert rebuilt == {
            "next_slot": 1,
            "retry_not_before": {},
            "unrepaired_refresh": {},
            "version": 3,
        }

    def test_read_only_plan_does_not_consume_a_fair_service_slot(self, tmp_path):
        service = AnalysisRefreshService()
        state_path = tmp_path / "refresh-state.json"
        summary = AnalysisFreshnessSummary(
            blocking_now=[_freshness_row("URGENT.A", bucket="blocking_now")],
            due_soon=[_freshness_row("CYCLE.A", bucket="due_soon")],
        )
        read_only = service.plan(
            summary,
            options=RefreshPlanOptions(
                policy="proactive",
                limit=1,
                show_recommendations=False,
                read_only=True,
                max_age_days=14,
                scheduler_state_path=state_path,
            ),
        )

        assert read_only.queued == []
        assert read_only.skipped_read_only == ["URGENT.A"]
        assert not state_path.exists()
        assert service.plan(
            summary, options=_plan_options(limit=1, scheduler_state_path=state_path)
        ).queued == ["URGENT.A"]

    def test_scheduler_state_loads_v1_without_unrepaired_backoffs(self, tmp_path):
        state_path = tmp_path / "refresh-state.json"
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "next_slot": 3,
                    "retry_not_before": {"FAILED.A": future},
                }
            ),
            encoding="utf-8",
        )

        state = AnalysisRefreshService._load_state(state_path)

        assert state.next_slot == 3
        assert state.retry_not_before == {"FAILED.A": future}
        assert state.unrepaired_refresh == {}

        AnalysisRefreshService._save_state(state_path, state)
        migrated = json.loads(state_path.read_text(encoding="utf-8"))
        assert migrated["version"] == 3
        assert migrated["unrepaired_refresh"] == {}

    def test_scheduler_state_rejects_unknown_version(self, tmp_path, caplog):
        state_path = tmp_path / "refresh-state.json"
        state_path.write_text(
            json.dumps(
                {
                    "version": 99,
                    "next_slot": 7,
                    "retry_not_before": {},
                    "unrepaired_refresh": {},
                }
            ),
            encoding="utf-8",
        )

        state = AnalysisRefreshService._load_state(state_path)

        assert state == type(state)()
        assert "refresh_scheduler_state_invalid" in caplog.text

    def test_scheduler_state_prunes_expired_entries_in_memory(self, tmp_path):
        state_path = tmp_path / "refresh-state.json"
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        state_path.write_text(
            json.dumps(
                {
                    "version": 3,
                    "next_slot": 2,
                    "retry_not_before": {"OLD.A": past, "LIVE.A": future},
                    "unrepaired_refresh": {
                        "OLD.A": {"basis": "DATA_QUALITY", "retry_after": past},
                        "BAD.A": {
                            "basis": "DATA_QUALITY",
                            "retry_after": "malformed",
                        },
                        "LIVE.A": {
                            "basis": "DATA_QUALITY",
                            "retry_after": future,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

        state = AnalysisRefreshService._load_state(state_path)

        assert state.retry_not_before == {"LIVE.A": future}
        assert state.unrepaired_refresh == {
            "LIVE.A": _UnrepairedRefresh(basis="DATA_QUALITY", retry_after=future)
        }
        # Loading/planning remains read-only; the next normal scheduler write
        # persists the pruned representation.
        assert "OLD.A" in state_path.read_text(encoding="utf-8")

    def test_scheduler_state_prunes_broker_contract_identifiers(self, tmp_path):
        state_path = tmp_path / "refresh-state.json"
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        state_path.write_text(
            json.dumps(
                {
                    "version": 3,
                    "next_slot": 2,
                    "retry_not_before": {
                        "PEY.TO": future,
                        "IBCID82633947.TO": future,
                    },
                    "unrepaired_refresh": {
                        "IBCID82633947.TO": {
                            "basis": "DATA_QUALITY",
                            "retry_after": future,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        state = AnalysisRefreshService._load_state(state_path)

        assert state.retry_not_before == {"PEY.TO": future}
        assert state.unrepaired_refresh == {}


class TestUserAction:
    def test_user_action_matches_read_only_message(self):
        service = AnalysisRefreshService()
        summary = service.classify([_make_review_item("7203.T")], max_age_days=14)
        activity = RefreshActivity(
            policy="blocking",
            limit=10,
            skipped_read_only=["7203.T"],
        )
        action = service.user_action(
            summary,
            activity,
            show_recommendations=False,
            command_builder=lambda *args: "pm " + " ".join(args),
        )
        assert (
            action
            == "read-only mode blocked refresh — run pm --refresh-policy proactive"
        )

    def test_user_action_respects_failure_backoff(self):
        service = AnalysisRefreshService()
        summary = service.classify([_make_review_item("7203.T")], max_age_days=14)
        activity = RefreshActivity(
            policy="blocking",
            limit=10,
            failed=["7203.T"],
            failed_retry_after={"7203.T": "2026-09-05T02:40:34+00:00"},
        )

        action = service.user_action(
            summary,
            activity,
            show_recommendations=False,
            command_builder=lambda *args: "pm " + " ".join(args),
        )

        assert action == (
            "refresh failed for 7203.T — backoff active; "
            "rerun on a later refresh-enabled run"
        )


class TestExecute:
    @pytest.mark.asyncio
    async def test_weighted_service_cursor_prevents_cycle_starvation_across_runs(
        self, tmp_path, monkeypatch
    ):
        """With limit=1 and a continuously replenished urgent queue, a cycle
        candidate receives the third service opportunity (2 urgent : 1 cycle)."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr("src.persistence._maybe_save_rejection_record", AsyncMock())
        service = AnalysisRefreshService()
        state_path = tmp_path / "refresh-state.json"
        seen: list[str] = []

        async def successful_runner(
            *, ticker: str, quick_mode: bool, skip_charts: bool
        ):
            seen.append(ticker)
            return {"ticker": ticker}

        def save_results(result, ticker: str, *, quick_mode: bool) -> Path:
            return tmp_path / f"{ticker}.json"

        for urgent_ticker, expected in (
            ("URGENT.1", "URGENT.1"),
            ("URGENT.2", "URGENT.2"),
            ("URGENT.3", "CYCLE.1"),
        ):
            summary = AnalysisFreshnessSummary(
                blocking_now=[_freshness_row(urgent_ticker, bucket="blocking_now")],
                due_soon=[_freshness_row("CYCLE.1", bucket="due_soon")],
            )
            activity = service.plan(
                summary,
                options=_plan_options(limit=1, scheduler_state_path=state_path),
            )
            assert activity.queued == [expected]
            await service.execute(
                activity,
                execution=RefreshExecutionOptions(quick_mode=True),
                run_analysis_fn=successful_runner,
                save_results_fn=save_results,
            )

        assert seen == ["URGENT.1", "URGENT.2", "CYCLE.1"]

    @pytest.mark.asyncio
    async def test_execute_calls_runner_and_saver(self):
        service = AnalysisRefreshService()
        calls: list[tuple[str, bool, bool]] = []
        saved: list[tuple[str, bool]] = []

        async def fake_run_analysis(
            *, ticker: str, quick_mode: bool, skip_charts: bool
        ):
            calls.append((ticker, quick_mode, skip_charts))
            return {"ticker": ticker}

        def fake_save_results(result, ticker: str, *, quick_mode: bool) -> Path:
            saved.append((ticker, quick_mode))
            return Path(f"/tmp/{ticker}.json")

        activity = RefreshActivity(policy="blocking", limit=10, queued=["7203.T"])
        updated = await service.execute(
            activity,
            execution=RefreshExecutionOptions(quick_mode=False),
            run_analysis_fn=fake_run_analysis,
            save_results_fn=fake_save_results,
        )

        assert calls == [("7203.T", False, True)]
        assert saved == [("7203.T", False)]
        assert updated.refreshed == ["7203.T"]
        assert updated.failed == []

    @pytest.mark.asyncio
    async def test_execute_records_failures(self):
        service = AnalysisRefreshService()

        async def fake_run_analysis(
            *, ticker: str, quick_mode: bool, skip_charts: bool
        ):
            return None

        def fake_save_results(result, ticker: str, *, quick_mode: bool) -> Path:
            raise AssertionError("save_results_to_file should not be called on failure")

        activity = RefreshActivity(policy="blocking", limit=10, queued=["7203.T"])
        updated = await service.execute(
            activity,
            execution=RefreshExecutionOptions(quick_mode=True),
            run_analysis_fn=fake_run_analysis,
            save_results_fn=fake_save_results,
        )

        assert updated.refreshed == []
        assert updated.failed == ["7203.T"]
        assert "7203.T" in updated.failed_retry_after

    @pytest.mark.asyncio
    async def test_unrepaired_refresh_backs_off_and_price_alone_cannot_requeue(
        self, tmp_path, monkeypatch
    ):
        """Exercise real reconcile rows through plan, execute, and replanning.

        The backoff is keyed on the action basis, so a price move that leaves
        the basis at DATA_QUALITY must not buy a second analysis: price is not
        evidence that more research exists to find. A genuine basis change is.
        """
        from unittest.mock import AsyncMock

        monkeypatch.setattr("src.persistence._maybe_save_rejection_record", AsyncMock())
        monkeypatch.setattr(
            "src.ibkr.position_evaluator._load_prior_history", lambda analysis: []
        )
        service = AnalysisRefreshService()
        state_path = tmp_path / "refresh-state.json"
        position = _make_position(ticker="2173.T", current_price=2100)
        analysis = _make_analysis(
            ticker="2173.T",
            verdict="DO_NOT_INITIATE",
            age_days=0,
        )
        analysis.health_adj = 30.0
        analysis.growth_adj = 20.0
        analysis.evidence = PortfolioEvidence(
            buy_blocking_flag_types=("LEGAL_COUNSEL_UNAVAILABLE",)
        )
        initial_items = reconcile([position], {"2173.T": analysis}, _make_portfolio())
        assert initial_items[0].action_basis == "DATA_QUALITY"
        initial_summary = service.classify(initial_items, max_age_days=14)
        activity = service.plan(
            initial_summary,
            options=_plan_options(limit=1, scheduler_state_path=state_path),
        )
        assert activity.queued == ["2173.T"]

        async def successful_runner(**kwargs):
            return {"ticker": "2173.T"}

        updated = await service.execute(
            activity,
            execution=RefreshExecutionOptions(quick_mode=False),
            run_analysis_fn=successful_runner,
            save_results_fn=lambda *args, **kwargs: tmp_path / "saved.json",
        )

        assert updated.refreshed == ["2173.T"]
        assert service._load_state(state_path).retry_not_before == {}

        # The successfully saved analysis still carries the same evidence gap.
        replanned_items = reconcile([position], {"2173.T": analysis}, _make_portfolio())
        replanned_summary = service.classify(
            replanned_items,
            max_age_days=14,
            already_refreshed=frozenset(updated.refreshed),
        )
        updated = service.record_unrepaired_refreshes(updated, replanned_summary)
        assert "2173.T" in updated.unrepaired_retry_after

        # A new invocation sees the same condition and defers it with timing.
        next_summary = service.classify(replanned_items, max_age_days=14)
        next_activity = service.plan(
            next_summary,
            options=_plan_options(limit=1, scheduler_state_path=state_path),
        )
        assert next_activity.queued == []
        assert next_activity.skipped_due_to_cooldown == ["2173.T"]
        assert "2173.T" in next_activity.skipped_due_to_unrepaired
        assert next_activity.skipped_due_to_failure_backoff == {}

        # The price later crosses the saved review level. The row is still a
        # DATA_QUALITY review, so the paid-analysis backoff still applies: a
        # price hovering near a threshold must not be able to re-authorize
        # spend on an evidence gap the last run already failed to close.
        breached_position = _make_position(ticker="2173.T", current_price=1800)
        stop_items = reconcile(
            [breached_position], {"2173.T": analysis}, _make_portfolio()
        )
        assert "broke the analysis review level" in stop_items[0].reason
        assert stop_items[0].action_basis == "DATA_QUALITY"
        stop_summary = service.classify(stop_items, max_age_days=14)
        stop_activity = service.plan(
            stop_summary,
            options=_plan_options(limit=1, scheduler_state_path=state_path),
        )
        assert stop_activity.queued == []
        assert "2173.T" in stop_activity.skipped_due_to_unrepaired

        # A genuine basis change is new information and clears the backoff.
        repaired = _make_analysis(
            ticker="2173.T", verdict="DO_NOT_INITIATE", age_days=0
        )
        repaired.health_adj = 30.0
        repaired.growth_adj = 20.0
        repaired.evidence = PortfolioEvidence()
        moved_items = reconcile([position], {"2173.T": repaired}, _make_portfolio())
        assert moved_items[0].action_basis != "DATA_QUALITY"
        moved_activity = service.plan(
            service.classify(moved_items, max_age_days=14),
            options=_plan_options(limit=1, scheduler_state_path=state_path),
        )
        assert moved_activity.skipped_due_to_unrepaired == {}

    @pytest.mark.asyncio
    async def test_execute_exception_does_not_strand_later_scheduled_work(
        self, tmp_path, monkeypatch
    ):
        """A provider or persistence exception consumes the failed attempt's
        slot, enters cooldown, and lets the remaining bounded batch continue."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr("src.persistence._maybe_save_rejection_record", AsyncMock())
        service = AnalysisRefreshService()
        state_path = tmp_path / "refresh-state.json"

        async def runner(*, ticker: str, **kwargs):
            if ticker == "FAIL.A":
                raise TimeoutError("provider unavailable")
            return {"ticker": ticker}

        activity = RefreshActivity(
            policy="proactive",
            limit=2,
            queued=["FAIL.A", "CYCLE.A"],
            scheduled_slots=[("FAIL.A", 1), ("CYCLE.A", 2)],
            scheduler_state_path=state_path,
        )
        updated = await service.execute(
            activity,
            execution=RefreshExecutionOptions(quick_mode=True),
            run_analysis_fn=runner,
            save_results_fn=lambda *args, **kwargs: tmp_path / "saved.json",
        )

        assert updated.failed == ["FAIL.A"]
        assert updated.refreshed == ["CYCLE.A"]
        state = service._load_state(state_path)
        assert state.next_slot == 2
        assert "FAIL.A" in state.retry_not_before

    @pytest.mark.asyncio
    async def test_execute_persists_rejection_record_after_save(self, monkeypatch):
        """Portfolio refresh path must mirror src.main: after save_results_fn,
        non-BUY verdicts feed the global lessons_learned collection.
        Honors --no-memory via the existing gate inside
        _maybe_save_rejection_record.
        """
        from unittest.mock import AsyncMock

        rejection_calls: list[tuple] = []

        async def fake_maybe_save_rejection_record(result, args, **kwargs):
            rejection_calls.append(
                (args.ticker, args.quick, getattr(args, "strict", False))
            )

        monkeypatch.setattr(
            "src.persistence._maybe_save_rejection_record",
            AsyncMock(side_effect=fake_maybe_save_rejection_record),
        )

        service = AnalysisRefreshService()

        async def fake_run_analysis(
            *, ticker: str, quick_mode: bool, skip_charts: bool
        ):
            return {"ticker": ticker, "verdict": "HOLD"}

        def fake_save_results(result, ticker: str, *, quick_mode: bool) -> Path:
            return Path(f"/tmp/{ticker}.json")

        activity = RefreshActivity(policy="blocking", limit=10, queued=["7203.T"])
        await service.execute(
            activity,
            execution=RefreshExecutionOptions(quick_mode=True),
            run_analysis_fn=fake_run_analysis,
            save_results_fn=fake_save_results,
        )

        assert rejection_calls == [("7203.T", True, False)], (
            "_maybe_save_rejection_record must be awaited once per refreshed "
            "ticker, after save_results_fn"
        )

    @pytest.mark.asyncio
    async def test_execute_skips_rejection_record_when_run_analysis_returns_none(
        self, monkeypatch
    ):
        """Failed analyses must NOT fire the rejection-record save path."""
        from unittest.mock import AsyncMock

        rejection_mock = AsyncMock()
        monkeypatch.setattr(
            "src.persistence._maybe_save_rejection_record", rejection_mock
        )

        service = AnalysisRefreshService()

        async def fake_run_analysis(**kwargs):
            return None

        def fake_save_results(*args, **kwargs):
            raise AssertionError("save not expected on failure")

        activity = RefreshActivity(policy="blocking", limit=10, queued=["7203.T"])
        await service.execute(
            activity,
            execution=RefreshExecutionOptions(quick_mode=True),
            run_analysis_fn=fake_run_analysis,
            save_results_fn=fake_save_results,
        )

        rejection_mock.assert_not_awaited()


class TestRefreshedThisRunIsNotReAdvertised:
    """A ticker this run refreshed must not carry a rerun command.

    Regression (2026-08-19): RecommendationService re-reconciles after
    executing refreshes, and the urgent classes carry no freshness floor, so
    7047.T and HERDEZ.MX appeared under "Urgent analysis refreshes" with a
    `--ticker` command in the same report whose "Refreshed:" line named them.
    """

    def test_refreshed_rows_leave_every_command_bearing_bucket(self):
        service = AnalysisRefreshService()
        items = [
            _make_review_item("7203.T", action_basis="DATA_QUALITY"),
            _make_review_item("6758.T", action_basis="DATA_QUALITY"),
        ]

        before = service.classify(items, max_age_days=14)
        assert {row.run_ticker for row in before.blocking_now} == {"7203.T", "6758.T"}

        after = service.classify(
            items, max_age_days=14, already_refreshed=frozenset({"7203.T"})
        )

        assert [row.run_ticker for row in after.blocking_now] == ["6758.T"]
        assert [row.run_ticker for row in after.refreshed_this_run] == ["7203.T"]
        assert all(
            row.bucket == "refreshed_this_run" for row in after.refreshed_this_run
        )

    def test_refreshed_rows_are_never_planned(self):
        service = AnalysisRefreshService()
        summary = service.classify(
            [_make_review_item("7203.T", action_basis="DATA_QUALITY")],
            max_age_days=14,
            already_refreshed=frozenset({"7203.T"}),
        )

        activity = service.plan(summary, options=_plan_options(policy="blocking"))

        assert activity.queued == []

    def test_operator_review_is_untouched_because_it_renders_no_command(self):
        service = AnalysisRefreshService()
        summary = service.classify(
            [_make_review_item("7203.T", action_basis="ENTRY_CONSTRAINT", age_days=1)],
            max_age_days=14,
            already_refreshed=frozenset({"7203.T"}),
        )

        assert [row.run_ticker for row in summary.operator_review] == ["7203.T"]
        assert summary.refreshed_this_run == []


class TestUnconfirmedRejectWaitsForItsConfirmationWindow:
    """A refresh that cannot change the disposition is cycle work, not urgent.

    reject_confirmed needs the two rejecting analyses at least
    DEFAULT_SELL_CONFIRMATION_MIN_SPACING_DAYS apart, so a refresh today can
    only confirm when the analysis on disk is already that old. Without this the
    same ticker re-entered the urgent queue on every run for a week.
    """

    @pytest.mark.parametrize("age_days", [0, 1, 6])
    def test_inside_the_window_it_is_operator_review(self, age_days):
        service = AnalysisRefreshService()

        summary = service.classify(
            [
                _make_review_item(
                    "7203.T", action_basis="THESIS_REASSESSMENT", age_days=age_days
                )
            ],
            max_age_days=14,
        )

        assert summary.blocking_now == []
        assert [row.run_ticker for row in summary.operator_review] == ["7203.T"]

    @pytest.mark.parametrize("age_days", [7, 8, 20])
    def test_at_or_past_the_window_it_is_urgent(self, age_days):
        service = AnalysisRefreshService()

        summary = service.classify(
            [
                _make_review_item(
                    "7203.T", action_basis="THESIS_REASSESSMENT", age_days=age_days
                )
            ],
            max_age_days=14,
        )

        assert [row.run_ticker for row in summary.blocking_now] == ["7203.T"]

    def test_soft_reject_is_unaffected_by_the_window(self):
        """A SOFT_REJECT was already on staleness cadence; nothing changes."""
        service = AnalysisRefreshService()

        summary = service.classify(
            [
                _make_review_item(
                    "7203.T",
                    action_basis="THESIS_REASSESSMENT",
                    sell_type="SOFT_REJECT",
                    age_days=1,
                )
            ],
            max_age_days=14,
        )

        assert summary.blocking_now == []
