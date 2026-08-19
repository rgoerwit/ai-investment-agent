from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from src.ibkr.models import ReconciliationItem
from src.ibkr.refresh_service import (
    AnalysisFreshnessRow,
    AnalysisFreshnessSummary,
    AnalysisRefreshService,
    RefreshActivity,
    RefreshExecutionOptions,
    RefreshPlanOptions,
)
from tests.ibkr.reconciler_cases import _make_analysis, _make_position


def _make_review_item(
    ticker: str,
    *,
    age_days: int = 20,
    reason: str = "Analysis too old",
    sell_type: str | None = None,
    action_basis: str | None = None,
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

    def test_blocking_keeps_due_soon_behind_urgent_work(self):
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
        assert activity.queued == ["7203.T"]

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
        assert rebuilt == {"next_slot": 1, "retry_not_before": {}, "version": 1}

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
