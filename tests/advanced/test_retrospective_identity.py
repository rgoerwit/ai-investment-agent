"""Step 1: snapshot identity, outcome-scoped dedup, and the evaluation memo.

Three defects with one root — the loop could not say what "this snapshot" *is*,
nor tell "a record exists" from "this snapshot was evaluated":

1. ``_lesson_already_processed`` counted ``prior_rejection`` screening records as
   proof that an outcome lesson existed. Measured on the corpus, 1,698 of 1,790
   tickers carried such a record on a date >= 30 days old, so their most recent
   evaluable snapshot could never produce a lesson.
2. Identity was ``(ticker, analysis_date)``, which collapses two analyses of one
   ticker on one day — exactly the model/prompt-change comparison the system
   exists to support.
3. A snapshot that was priced but did not clear its threshold left no trace, so
   it paid a fresh pair of yfinance round-trips on every subsequent run forever.

The doubles here are behavioural on purpose: a ``MagicMock`` collection returns a
truthy payload from ``.get()`` regardless of the ``where`` clause, so tests for
(1) would pass identically against the pre-fix code.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.retrospective import (
    MEMO_OUTCOME_BELOW_THRESHOLD,
    MEMO_OUTCOME_NO_DATA,
    MEMO_OUTCOME_TRIGGERED,
    OUTCOME_LESSON_TYPES,
    RETROSPECTIVE_REEVALUATION_INTERVAL_DAYS,
    EvaluationMemo,
    _lesson_already_processed,
    _select_within_budget,
    run_retrospective,
    snapshot_identity,
    store_lesson,
)
from tests.advanced.retrospective_fakes import (
    FakeLessonsMemory,
    days_ago,
    make_snapshot,
)


@pytest.fixture(autouse=True)
def _memo_in_tmp(monkeypatch, tmp_path):
    """No test may write the repository's real evaluation memo."""
    monkeypatch.setattr(
        "src.retrospective.DEFAULT_EVALUATION_MEMO_PATH",
        tmp_path / "memo.json",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Outcome-scoped dedup
# ══════════════════════════════════════════════════════════════════════════════


class TestOutcomeScopedDedup:
    def test_prior_rejection_does_not_suppress_an_outcome_lesson(self):
        """The regression. A screening artifact is not an outcome."""
        memory = FakeLessonsMemory()
        memory.seed(
            ticker="2767.T",
            analysis_date="2026-06-16",
            lesson_type="prior_rejection",
        )

        assert _lesson_already_processed(memory, "2767.T", "2026-06-16", None) is False

    def test_an_existing_outcome_lesson_does_suppress(self):
        memory = FakeLessonsMemory()
        memory.seed(
            ticker="2767.T",
            analysis_date="2026-06-16",
            lesson_type="missed_risk",
        )

        assert _lesson_already_processed(memory, "2767.T", "2026-06-16", None) is True

    @pytest.mark.parametrize("lesson_type", sorted(OUTCOME_LESSON_TYPES))
    def test_every_outcome_type_suppresses(self, lesson_type):
        memory = FakeLessonsMemory()
        memory.seed(
            ticker="2767.T", analysis_date="2026-06-16", lesson_type=lesson_type
        )
        assert _lesson_already_processed(memory, "2767.T", "2026-06-16", None) is True

    def test_prior_rejection_is_excluded_from_the_outcome_set(self):
        assert "prior_rejection" not in OUTCOME_LESSON_TYPES
        assert "missed_risk" in OUTCOME_LESSON_TYPES

    def test_another_tickers_lesson_does_not_suppress(self):
        memory = FakeLessonsMemory()
        memory.seed(
            ticker="OTHER.T", analysis_date="2026-06-16", lesson_type="missed_risk"
        )
        assert _lesson_already_processed(memory, "2767.T", "2026-06-16", None) is False

    def test_unavailable_memory_never_suppresses(self):
        memory = FakeLessonsMemory(available=False)
        assert _lesson_already_processed(memory, "2767.T", "2026-06-16", None) is False

    def test_a_raising_collection_degrades_to_not_processed(self):
        memory = FakeLessonsMemory()

        def _boom(*_args, **_kwargs):
            raise RuntimeError("chroma is down")

        memory.situation_collection.get = _boom
        assert _lesson_already_processed(memory, "2767.T", "2026-06-16", None) is False

    def test_non_mapping_metadata_is_skipped_not_raised(self):
        memory = FakeLessonsMemory()
        memory.seed(
            ticker="2767.T", analysis_date="2026-06-16", lesson_type="missed_risk"
        )
        memory.situation_collection.records[0]["metadata"] = {
            "ticker": "2767.T",
            "lesson_type": "missed_risk",
            "analysis_date": "2026-06-16",
        }
        original_get = memory.situation_collection.get

        def _get_with_garbage(*args, **kwargs):
            payload = original_get(*args, **kwargs)
            payload["metadatas"].insert(0, "not-a-mapping")
            return payload

        memory.situation_collection.get = _get_with_garbage
        assert _lesson_already_processed(memory, "2767.T", "2026-06-16", None) is True


class TestBothDedupSitesShareOnePredicate:
    """``store_lesson`` carried a second copy of the same buggy query.

    Fixing only the orchestrator would have changed nothing observable: the
    snapshot would be priced, a lesson generated, and then discarded at the write.
    """

    @pytest.mark.asyncio
    async def test_store_lesson_is_not_blocked_by_a_prior_rejection(self):
        memory = FakeLessonsMemory()
        memory.seed(
            ticker="2767.T",
            analysis_date="2026-06-16",
            lesson_type="prior_rejection",
        )

        stored = await store_lesson(
            "Corroborate cash flow before trusting it.",
            "missed_risk",
            "OPERATIONAL_MISS",
            {"ticker": "2767.T", "analysis_date": "2026-06-16"},
            0.8,
            memory,
        )

        assert stored is True
        assert memory.add_calls == 1

    @pytest.mark.asyncio
    async def test_store_lesson_still_refuses_a_true_duplicate(self):
        memory = FakeLessonsMemory()
        memory.seed(
            ticker="2767.T", analysis_date="2026-06-16", lesson_type="missed_risk"
        )

        stored = await store_lesson(
            "A second lesson for the same analysis.",
            "missed_risk",
            "OPERATIONAL_MISS",
            {"ticker": "2767.T", "analysis_date": "2026-06-16"},
            0.8,
            memory,
        )

        assert stored is False
        assert memory.add_calls == 0

    @pytest.mark.asyncio
    async def test_stored_metadata_carries_the_analysis_id(self):
        memory = FakeLessonsMemory()
        await store_lesson(
            "lesson",
            "missed_risk",
            "OPERATIONAL_MISS",
            {
                "ticker": "2767.T",
                "analysis_date": "2026-06-16",
                "analysis_id": "run-abc",
            },
            0.8,
            memory,
        )
        assert memory.metadatas()[0]["analysis_id"] == "run-abc"

    @pytest.mark.asyncio
    async def test_absent_analysis_id_is_stored_as_empty_string(self):
        """ChromaDB metadata rejects nulls; the field must still be present."""
        memory = FakeLessonsMemory()
        await store_lesson(
            "lesson",
            "missed_risk",
            "OPERATIONAL_MISS",
            {"ticker": "2767.T", "analysis_date": "2026-06-16"},
            0.8,
            memory,
        )
        assert memory.metadatas()[0]["analysis_id"] == ""


# ══════════════════════════════════════════════════════════════════════════════
# Identity
# ══════════════════════════════════════════════════════════════════════════════


class TestSnapshotIdentity:
    def test_analysis_id_wins_when_present(self):
        snapshot = make_snapshot(analysis_id="run-abc")
        assert snapshot_identity(snapshot) == "run-abc"

    def test_source_filename_is_the_legacy_fallback(self):
        snapshot = make_snapshot(
            analysis_id=None, source_file="2767.T_20260216_143000_analysis.json"
        )
        assert snapshot_identity(snapshot) == "2767.T_20260216_143000"

    def test_composite_key_is_the_last_resort(self):
        snapshot = {"ticker": "2767.T", "analysis_date": "2026-06-16"}
        assert snapshot_identity(snapshot) == "2767.T|2026-06-16"

    def test_two_same_day_runs_have_distinct_identities(self):
        """The property the whole step exists for."""
        morning = make_snapshot(analysis_id="run-morning", age_days=180)
        evening = make_snapshot(analysis_id="run-evening", age_days=180)
        assert morning["analysis_date"] == evening["analysis_date"]
        assert snapshot_identity(morning) != snapshot_identity(evening)

    def test_two_same_day_runs_are_distinct_by_filename_too(self):
        morning = make_snapshot(
            analysis_id=None, source_file="2767.T_20260216_090000_analysis.json"
        )
        evening = make_snapshot(
            analysis_id=None, source_file="2767.T_20260216_210000_analysis.json"
        )
        assert snapshot_identity(morning) != snapshot_identity(evening)

    def test_dedup_distinguishes_two_same_day_analyses(self):
        memory = FakeLessonsMemory()
        memory.seed(
            ticker="2767.T",
            analysis_date="2026-06-16",
            analysis_id="run-morning",
            lesson_type="missed_risk",
        )

        assert (
            _lesson_already_processed(memory, "2767.T", "2026-06-16", "run-morning")
            is True
        )
        assert (
            _lesson_already_processed(memory, "2767.T", "2026-06-16", "run-evening")
            is False
        )

    def test_a_legacy_stored_record_still_matches_by_date(self):
        """Records written before analysis_id existed must not be regenerated."""
        memory = FakeLessonsMemory()
        memory.seed(
            ticker="2767.T",
            analysis_date="2026-06-16",
            lesson_type="missed_risk",
        )
        assert (
            _lesson_already_processed(memory, "2767.T", "2026-06-16", "run-modern")
            is True
        )


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation memo
# ══════════════════════════════════════════════════════════════════════════════


class TestEvaluationMemo:
    def test_an_unseen_snapshot_is_evaluated(self, tmp_path):
        memo = EvaluationMemo(tmp_path / "memo.json")
        assert memo.should_evaluate("run-1", 180) is True

    def test_a_recorded_snapshot_is_skipped(self, tmp_path):
        memo = EvaluationMemo(tmp_path / "memo.json")
        memo.record(
            "run-1",
            ticker="2767.T",
            analysis_date=days_ago(180),
            days_elapsed=180,
            outcome=MEMO_OUTCOME_BELOW_THRESHOLD,
        )
        assert memo.should_evaluate("run-1", 180) is False

    def test_re_evaluation_waits_for_the_interval(self, tmp_path):
        memo = EvaluationMemo(tmp_path / "memo.json")
        memo.record(
            "run-1",
            ticker="2767.T",
            analysis_date=days_ago(180),
            days_elapsed=180,
            outcome=MEMO_OUTCOME_BELOW_THRESHOLD,
        )
        just_under = 180 + RETROSPECTIVE_REEVALUATION_INTERVAL_DAYS - 1
        exactly_at = 180 + RETROSPECTIVE_REEVALUATION_INTERVAL_DAYS
        assert memo.should_evaluate("run-1", just_under) is False
        assert memo.should_evaluate("run-1", exactly_at) is True

    def test_a_missing_benchmark_is_retried_immediately(self, tmp_path):
        """A fetch outage is transient — not a verdict that waits 30 days."""
        memo = EvaluationMemo(tmp_path / "memo.json")
        memo.record(
            "run-1",
            ticker="2767.T",
            analysis_date=days_ago(180),
            days_elapsed=180,
            outcome=MEMO_OUTCOME_NO_DATA,
        )
        assert memo.should_evaluate("run-1", 180) is True

    def test_memo_round_trips_through_disk(self, tmp_path):
        path = tmp_path / "memo.json"
        first = EvaluationMemo(path)
        first.record(
            "run-1",
            ticker="2767.T",
            analysis_date=days_ago(180),
            days_elapsed=180,
            outcome=MEMO_OUTCOME_TRIGGERED,
        )
        first.flush()

        second = EvaluationMemo(path)
        assert second.should_evaluate("run-1", 180) is False

    def test_flush_is_a_noop_when_nothing_changed(self, tmp_path):
        path = tmp_path / "memo.json"
        EvaluationMemo(path).flush()
        assert not path.exists()

    def test_corrupt_memo_evaluates_everything(self, tmp_path):
        path = tmp_path / "memo.json"
        path.write_text("{not json at all")
        memo = EvaluationMemo(path)
        assert memo.should_evaluate("run-1", 180) is True

    def test_a_json_list_is_rejected_rather_than_crashing(self, tmp_path):
        path = tmp_path / "memo.json"
        path.write_text("[1, 2, 3]")
        memo = EvaluationMemo(path)
        assert memo.should_evaluate("run-1", 180) is True

    def test_a_non_dict_entry_is_dropped(self, tmp_path):
        path = tmp_path / "memo.json"
        path.write_text(json.dumps({"run-1": "not-a-dict"}))
        memo = EvaluationMemo(path)
        assert memo.should_evaluate("run-1", 180) is True

    def test_a_corrupt_age_forces_re_evaluation(self, tmp_path):
        path = tmp_path / "memo.json"
        path.write_text(
            json.dumps({"run-1": {"evaluated_at_days": "not-a-number", "outcome": "X"}})
        )
        memo = EvaluationMemo(path)
        assert memo.should_evaluate("run-1", 180) is True

    def test_unknown_age_forces_re_evaluation(self, tmp_path):
        memo = EvaluationMemo(tmp_path / "memo.json")
        memo.record(
            "run-1",
            ticker="2767.T",
            analysis_date="",
            days_elapsed=180,
            outcome=MEMO_OUTCOME_BELOW_THRESHOLD,
        )
        assert memo.should_evaluate("run-1", None) is True

    def test_an_unwritable_path_does_not_raise(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file, not a directory")
        memo = EvaluationMemo(blocker / "nested" / "memo.json")
        memo.record(
            "run-1",
            ticker="2767.T",
            analysis_date=days_ago(180),
            days_elapsed=180,
            outcome=MEMO_OUTCOME_TRIGGERED,
        )
        memo.flush()  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# Budget
# ══════════════════════════════════════════════════════════════════════════════


class TestEvaluationBudget:
    def _candidates(self, ages):
        from src.retrospective import _EvaluationCandidate

        return [
            _EvaluationCandidate(
                ticker="2767.T",
                identity=f"run-{age}",
                days_elapsed=age,
                snapshot=make_snapshot(age_days=age, analysis_id=f"run-{age}"),
            )
            for age in ages
        ]

    def test_highest_confidence_band_is_spent_first(self):
        selected, deferred = _select_within_budget(
            self._candidates([40, 180, 500, 200]), 2
        )
        assert [c.days_elapsed for c in selected] == [180, 200]
        assert {c.days_elapsed for c in deferred} == {40, 500}

    def test_selection_is_stable_across_runs(self):
        ages = [40, 180, 500, 200, 95, 260]
        first, _ = _select_within_budget(self._candidates(ages), 3)
        second, _ = _select_within_budget(self._candidates(list(reversed(ages))), 3)
        assert [c.identity for c in first] == [c.identity for c in second]

    def test_a_budget_larger_than_the_corpus_defers_nothing(self):
        selected, deferred = _select_within_budget(self._candidates([100, 200]), 99)
        assert len(selected) == 2
        assert deferred == []

    def test_a_zero_budget_means_unbounded_not_empty(self):
        """0 is 'no ceiling configured', not 'evaluate nothing'."""
        selected, deferred = _select_within_budget(self._candidates([100, 200]), 0)
        assert len(selected) == 2
        assert deferred == []

    def test_an_empty_corpus_is_not_an_error(self):
        assert _select_within_budget([], 10) == ([], [])


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════════


class TestOrchestration:
    @pytest.mark.asyncio
    async def test_a_rejected_ticker_is_still_evaluated(self, tmp_path):
        """End to end: the 1,698-ticker regression."""
        memory = FakeLessonsMemory()
        memory.seed(
            ticker="2767.T",
            analysis_date=days_ago(180),
            lesson_type="prior_rejection",
        )
        snapshots = {"2767.T": [make_snapshot(age_days=180, analysis_id="run-1")]}

        with patch("src.retrospective.load_past_snapshots", return_value=snapshots):
            with patch(
                "src.retrospective.compare_to_reality",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_compare:
                await run_retrospective(
                    "2767.T", Path("/fake"), memory, memo_path=tmp_path / "m.json"
                )

        mock_compare.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_second_run_prices_nothing(self, tmp_path):
        """The memo regression: a non-triggering snapshot is priced once."""
        memory = FakeLessonsMemory()
        snapshots = {
            "2767.T": [
                make_snapshot(age_days=180, analysis_id="run-1"),
                make_snapshot(age_days=200, analysis_id="run-2"),
            ]
        }
        memo_path = tmp_path / "memo.json"

        async def _below_threshold(_snapshot):
            return None

        for expected_calls in (2, 0):
            with (
                patch("src.retrospective.load_past_snapshots", return_value=snapshots),
                patch(
                    "src.retrospective.compare_to_reality",
                    side_effect=_below_threshold,
                ) as mock_compare,
            ):
                await run_retrospective(
                    "2767.T", Path("/fake"), memory, memo_path=memo_path
                )
            assert mock_compare.await_count == expected_calls

    @pytest.mark.asyncio
    async def test_too_recent_snapshots_never_reach_the_network(self, tmp_path):
        memory = FakeLessonsMemory()
        snapshots = {"2767.T": [make_snapshot(age_days=5, analysis_id="run-1")]}

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch(
                "src.retrospective.compare_to_reality",
                new_callable=AsyncMock,
            ) as mock_compare,
        ):
            await run_retrospective(
                "2767.T", Path("/fake"), memory, memo_path=tmp_path / "m.json"
            )

        mock_compare.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_budget_bounds_network_calls(self, tmp_path):
        memory = FakeLessonsMemory()
        snapshots = {
            "2767.T": [
                make_snapshot(age_days=100 + i * 10, analysis_id=f"run-{i}")
                for i in range(10)
            ]
        }

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch(
                "src.retrospective.compare_to_reality",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_compare,
        ):
            await run_retrospective(
                "2767.T",
                Path("/fake"),
                memory,
                memo_path=tmp_path / "m.json",
                max_evaluations=3,
            )

        assert mock_compare.await_count == 3

    @pytest.mark.asyncio
    async def test_a_deferred_snapshot_is_picked_up_next_run(self, tmp_path):
        memory = FakeLessonsMemory()
        snapshots = {
            "2767.T": [
                make_snapshot(age_days=180, analysis_id="run-a"),
                make_snapshot(age_days=181, analysis_id="run-b"),
            ]
        }
        memo_path = tmp_path / "memo.json"
        priced: list[str] = []

        async def _record(snapshot):
            priced.append(snapshot["analysis_id"])
            return None

        for _ in range(2):
            with (
                patch("src.retrospective.load_past_snapshots", return_value=snapshots),
                patch("src.retrospective.compare_to_reality", side_effect=_record),
            ):
                await run_retrospective(
                    "2767.T",
                    Path("/fake"),
                    memory,
                    memo_path=memo_path,
                    max_evaluations=1,
                )

        assert priced == ["run-a", "run-b"]

    @pytest.mark.asyncio
    async def test_a_raising_comparison_does_not_abort_the_run(self, tmp_path):
        memory = FakeLessonsMemory()
        snapshots = {
            "2767.T": [
                make_snapshot(age_days=180, analysis_id="run-a"),
                make_snapshot(age_days=181, analysis_id="run-b"),
            ]
        }
        calls: list[str] = []

        async def _explode(snapshot):
            calls.append(snapshot["analysis_id"])
            raise RuntimeError("yfinance exploded")

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch("src.retrospective.compare_to_reality", side_effect=_explode),
        ):
            lessons = await run_retrospective(
                "2767.T", Path("/fake"), memory, memo_path=tmp_path / "m.json"
            )

        assert len(calls) == 2
        assert lessons == []
