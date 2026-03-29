from __future__ import annotations

from pathlib import Path

import pytest

from src.ibkr.models import ReconciliationItem
from src.ibkr.refresh_service import (
    AnalysisRefreshService,
    RefreshActivity,
    RefreshExecutionOptions,
    RefreshPlanOptions,
)
from tests.ibkr.test_reconciler import _make_analysis, _make_position


def _make_review_item(
    ticker: str,
    *,
    age_days: int = 20,
    reason: str = "Analysis too old",
    sell_type: str | None = None,
    held: bool = True,
) -> ReconciliationItem:
    return ReconciliationItem(
        ticker=ticker,
        action="REVIEW",
        reason=reason,
        urgency="MEDIUM",
        ibkr_position=_make_position(ticker=ticker) if held else None,
        analysis=_make_analysis(ticker=ticker, age_days=age_days),
        sell_type=sell_type,
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

    def test_recommend_defaults_to_blocking(self):
        service = AnalysisRefreshService()
        assert (
            service.resolve_policy(
                explicit_policy=None,
                refresh_stale=False,
                recommend=True,
                read_only=False,
            )
            == "blocking"
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
    def test_review_holding_goes_to_blocking_now(self):
        service = AnalysisRefreshService()
        summary = service.classify([_make_review_item("7203.T")], max_age_days=14)
        assert [row.run_ticker for row in summary.blocking_now] == ["7203.T"]

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

    def test_blocking_excludes_due_soon(self):
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
            [_make_review_item("7203.T"), due_soon_item],
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
        assert activity.queued == ["7203.T"]

    def test_read_only_moves_queue_to_skipped(self):
        service = AnalysisRefreshService()
        summary = service.classify([_make_review_item("7203.T")], max_age_days=14)
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
            [_make_review_item("7203.T"), _make_review_item("6758.T")],
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
        assert activity.queued == ["7203.T"]
        assert activity.skipped_due_to_limit == ["6758.T"]

    def test_ticker_subset_filters_candidates(self):
        service = AnalysisRefreshService()
        summary = service.classify(
            [_make_review_item("7203.T"), _make_review_item("6758.T")],
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
            == "read-only mode blocked refresh — run pm --refresh-policy blocking"
        )


class TestExecute:
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
