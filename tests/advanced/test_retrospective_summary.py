"""Step 2: the run summary, and the dry run that sizes a run before paying for it.

The orchestrator used to report a start line and a stored count, which cannot
answer the question that has to be settled before widening its inputs: how many
network round-trips will this cost, and where did the rest of the corpus go?

The invariant that makes the totals trustworthy is that every scanned snapshot
lands in exactly one disposition bucket. A summary that merely *looked* plausible
would be worse than none, so ``reconciles`` is asserted on every path here — and
the orchestrator itself logs a warning when it fails.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.retrospective import RetrospectiveRunSummary, run_retrospective
from tests.advanced.retrospective_fakes import FakeLessonsMemory, make_snapshot


def _capture():
    seen: list[RetrospectiveRunSummary] = []
    return seen, seen.append


class TestSummaryShape:
    def test_reconciles_when_every_snapshot_is_dispositioned(self):
        summary = RetrospectiveRunSummary(
            scanned=10,
            skipped_existing_lesson=2,
            skipped_memo=3,
            skipped_too_recent=1,
            deferred_over_budget=1,
            evaluated=3,
        )
        assert summary.reconciles is True

    def test_does_not_reconcile_when_a_bucket_is_missed(self):
        assert RetrospectiveRunSummary(scanned=10, evaluated=3).reconciles is False

    def test_empty_summary_reconciles(self):
        assert RetrospectiveRunSummary().reconciles is True

    def test_to_dict_round_trips_every_field(self):
        summary = RetrospectiveRunSummary(scanned=1, evaluated=1, dry_run=True)
        payload = summary.to_dict()
        assert payload["scanned"] == 1
        assert payload["dry_run"] is True
        assert set(payload) == {
            "scanned",
            "skipped_existing_lesson",
            "skipped_memo",
            "skipped_too_recent",
            "deferred_over_budget",
            "evaluated",
            "unassessed_benchmark",
            "triggered",
            "generated",
            "stored",
            "failed",
            "dry_run",
        }

    def test_summary_is_immutable(self):
        summary = RetrospectiveRunSummary(scanned=1)
        with pytest.raises(AttributeError):
            summary.scanned = 2  # type: ignore[misc]


class TestSummaryFromARealRun:
    @pytest.mark.asyncio
    async def test_totals_reconcile_across_a_mixed_corpus(self, tmp_path):
        memory = FakeLessonsMemory()
        # One already has an outcome lesson, one is too recent, three are live.
        memory.seed(
            ticker="2767.T",
            analysis_date=make_snapshot(age_days=300)["analysis_date"],
            lesson_type="missed_risk",
        )
        snapshots = {
            "2767.T": [
                make_snapshot(age_days=300, analysis_id="run-lessoned"),
                make_snapshot(age_days=5, analysis_id="run-recent"),
                make_snapshot(age_days=180, analysis_id="run-a"),
                make_snapshot(age_days=190, analysis_id="run-b"),
                make_snapshot(age_days=200, analysis_id="run-c"),
            ]
        }
        seen, on_summary = _capture()

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch(
                "src.retrospective.compare_to_reality",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await run_retrospective(
                "2767.T",
                Path("/fake"),
                memory,
                memo_path=tmp_path / "m.json",
                max_evaluations=2,
                on_summary=on_summary,
            )

        summary = seen[-1]
        assert summary.scanned == 5
        assert summary.skipped_existing_lesson == 1
        assert summary.skipped_too_recent == 1
        assert summary.deferred_over_budget == 1
        assert summary.evaluated == 2
        assert summary.reconciles is True
        assert summary.dry_run is False

    @pytest.mark.asyncio
    async def test_memo_skips_are_counted_on_the_second_run(self, tmp_path):
        memory = FakeLessonsMemory()
        snapshots = {"2767.T": [make_snapshot(age_days=180, analysis_id="run-a")]}
        memo_path = tmp_path / "m.json"
        seen, on_summary = _capture()

        for _ in range(2):
            with (
                patch("src.retrospective.load_past_snapshots", return_value=snapshots),
                patch(
                    "src.retrospective.compare_to_reality",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
            ):
                await run_retrospective(
                    "2767.T",
                    Path("/fake"),
                    memory,
                    memo_path=memo_path,
                    on_summary=on_summary,
                )

        assert seen[0].evaluated == 1 and seen[0].skipped_memo == 0
        assert seen[1].evaluated == 0 and seen[1].skipped_memo == 1
        assert all(s.reconciles for s in seen)

    @pytest.mark.asyncio
    async def test_generated_and_stored_are_distinct(self, tmp_path):
        """A generated lesson the store refuses is still generated (and paid for)."""
        memory = FakeLessonsMemory()
        snapshots = {"2767.T": [make_snapshot(age_days=180, analysis_id="run-a")]}
        comparison = make_snapshot(age_days=180, analysis_id="run-a")
        comparison.update({"excess_return_pct": -40.0, "days_elapsed": 180})
        seen, on_summary = _capture()

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch(
                "src.retrospective.compare_to_reality",
                new_callable=AsyncMock,
                return_value=comparison,
            ),
            patch(
                "src.retrospective.generate_lesson",
                new_callable=AsyncMock,
                return_value=("a lesson", "missed_risk", "CYCLICAL_PEAK"),
            ),
            patch(
                "src.retrospective.store_lesson",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await run_retrospective(
                "2767.T",
                Path("/fake"),
                memory,
                memo_path=tmp_path / "m.json",
                on_summary=on_summary,
            )

        assert seen[-1].generated == 1
        assert seen[-1].stored == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("failure", "label"),
        [
            ("generation", "the lesson LLM failed"),
            ("storage", "the Chroma write was refused"),
        ],
    )
    async def test_a_snapshot_whose_lesson_never_landed_is_retried(
        self, tmp_path, failure, label
    ):
        """ "Processed" must mean the work finished, not that it was attempted.

        Recording TRIGGERED at pricing time meant a timed-out lesson call or a
        refused write cost the snapshot its lesson for the whole re-evaluation
        interval, with manual memo deletion as the only recovery. Proving that
        needs a *second* run — a single-run assertion that nothing was stored
        passes either way.
        """
        memory = FakeLessonsMemory()
        snapshots = {"2767.T": [make_snapshot(age_days=180, analysis_id="run-a")]}
        memo_path = tmp_path / "m.json"
        comparison = make_snapshot(age_days=180, analysis_id="run-a")
        comparison.update({"excess_return_pct": -40.0, "days_elapsed": 180})
        priced: list[str] = []

        async def _price(snapshot):
            priced.append(snapshot["analysis_id"])
            return dict(comparison)

        lesson_patch = (
            patch(
                "src.retrospective.generate_lesson",
                new_callable=AsyncMock,
                return_value=None,
            )
            if failure == "generation"
            else patch(
                "src.retrospective.generate_lesson",
                new_callable=AsyncMock,
                return_value=("a lesson", "missed_risk", "CYCLICAL_PEAK"),
            )
        )
        store_patch = patch(
            "src.retrospective.store_lesson",
            new_callable=AsyncMock,
            return_value=False,
        )

        for _ in range(2):
            with (
                patch("src.retrospective.load_past_snapshots", return_value=snapshots),
                patch("src.retrospective.compare_to_reality", side_effect=_price),
                lesson_patch,
                store_patch,
            ):
                await run_retrospective(
                    "2767.T", Path("/fake"), memory, memo_path=memo_path
                )

        assert priced == ["run-a", "run-a"], (
            f"{label}: the snapshot must be retried on the next run, not memoized"
        )

    @pytest.mark.asyncio
    async def test_a_stored_lesson_does_suppress_the_next_run(self, tmp_path):
        """The control: success must still be memoized."""
        memory = FakeLessonsMemory()
        snapshots = {"2767.T": [make_snapshot(age_days=180, analysis_id="run-a")]}
        memo_path = tmp_path / "m.json"
        comparison = make_snapshot(age_days=180, analysis_id="run-a")
        comparison.update({"excess_return_pct": -40.0, "days_elapsed": 180})
        priced: list[str] = []

        async def _price(snapshot):
            priced.append(snapshot["analysis_id"])
            return dict(comparison)

        for _ in range(2):
            with (
                patch("src.retrospective.load_past_snapshots", return_value=snapshots),
                patch("src.retrospective.compare_to_reality", side_effect=_price),
                patch(
                    "src.retrospective.generate_lesson",
                    new_callable=AsyncMock,
                    return_value=("a lesson", "missed_risk", "CYCLICAL_PEAK"),
                ),
                patch(
                    "src.retrospective.store_lesson",
                    new_callable=AsyncMock,
                    return_value=True,
                ),
            ):
                await run_retrospective(
                    "2767.T", Path("/fake"), memory, memo_path=memo_path
                )

        assert priced == ["run-a"], "a stored lesson must not be re-priced"

    @pytest.mark.asyncio
    async def test_a_capped_snapshot_is_memoized_not_re_priced(self, tmp_path):
        """Withheld by MAX_LESSONS_PER_TICKER is policy, not failure."""
        from src.retrospective import MAX_LESSONS_PER_TICKER

        memory = FakeLessonsMemory()
        count = MAX_LESSONS_PER_TICKER + 2
        snapshots = {
            "2767.T": [
                make_snapshot(age_days=180 + i, analysis_id=f"run-{i}")
                for i in range(count)
            ]
        }
        memo_path = tmp_path / "m.json"
        priced: list[str] = []

        async def _price(snapshot):
            priced.append(snapshot["analysis_id"])
            out = dict(snapshot)
            out.update({"excess_return_pct": -40.0, "days_elapsed": 180})
            return out

        for _ in range(2):
            with (
                patch("src.retrospective.load_past_snapshots", return_value=snapshots),
                patch("src.retrospective.compare_to_reality", side_effect=_price),
                patch(
                    "src.retrospective.generate_lesson",
                    new_callable=AsyncMock,
                    return_value=("a lesson", "missed_risk", "CYCLICAL_PEAK"),
                ),
                patch(
                    "src.retrospective.store_lesson",
                    new_callable=AsyncMock,
                    return_value=True,
                ),
            ):
                await run_retrospective(
                    "2767.T", Path("/fake"), memory, memo_path=memo_path
                )

        assert len(priced) == count, (
            "capped snapshots must be memoized; re-pricing them every run only to "
            "cap them again is pure waste"
        )

    @pytest.mark.asyncio
    async def test_a_failed_comparison_is_counted_without_breaking_reconciliation(
        self, tmp_path
    ):
        memory = FakeLessonsMemory()
        snapshots = {"2767.T": [make_snapshot(age_days=180, analysis_id="run-a")]}
        seen, on_summary = _capture()

        async def _explode(_snapshot):
            raise RuntimeError("yfinance exploded")

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch("src.retrospective.compare_to_reality", side_effect=_explode),
        ):
            await run_retrospective(
                "2767.T",
                Path("/fake"),
                memory,
                memo_path=tmp_path / "m.json",
                on_summary=on_summary,
            )

        assert seen[-1].failed == 1
        assert seen[-1].evaluated == 1
        assert seen[-1].reconciles is True


class TestEveryExitPathReports:
    @pytest.mark.asyncio
    async def test_an_empty_corpus_still_emits_a_summary(self, tmp_path):
        seen, on_summary = _capture()
        with patch("src.retrospective.load_past_snapshots", return_value={}):
            await run_retrospective(
                "2767.T",
                Path("/fake"),
                FakeLessonsMemory(),
                memo_path=tmp_path / "m.json",
                on_summary=on_summary,
            )
        assert seen and seen[-1].scanned == 0

    @pytest.mark.asyncio
    async def test_a_memory_init_failure_still_emits_a_summary(self, tmp_path):
        seen, on_summary = _capture()
        with patch(
            "src.memory.FinancialSituationMemory", side_effect=RuntimeError("no chroma")
        ):
            await run_retrospective(
                "2767.T",
                Path("/fake"),
                None,
                memo_path=tmp_path / "m.json",
                on_summary=on_summary,
            )
        assert seen and seen[-1].scanned == 0

    @pytest.mark.asyncio
    async def test_a_raising_callback_never_breaks_the_run(self, tmp_path):
        """Reporting is a diagnostic; it may not cost the run its lessons."""
        memory = FakeLessonsMemory()
        snapshots = {"2767.T": [make_snapshot(age_days=180, analysis_id="run-a")]}

        def _explode(_summary):
            raise RuntimeError("the console is on fire")

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch(
                "src.retrospective.compare_to_reality",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            lessons = await run_retrospective(
                "2767.T",
                Path("/fake"),
                memory,
                memo_path=tmp_path / "m.json",
                on_summary=_explode,
            )

        assert lessons == []


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_prices_nothing_and_writes_nothing(self, tmp_path):
        memory = FakeLessonsMemory()
        snapshots = {
            "2767.T": [
                make_snapshot(age_days=180, analysis_id="run-a"),
                make_snapshot(age_days=190, analysis_id="run-b"),
            ]
        }
        memo_path = tmp_path / "m.json"
        seen, on_summary = _capture()

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch(
                "src.retrospective.compare_to_reality", new_callable=AsyncMock
            ) as mock_compare,
            patch(
                "src.retrospective.generate_lesson", new_callable=AsyncMock
            ) as mock_lesson,
        ):
            lessons = await run_retrospective(
                "2767.T",
                Path("/fake"),
                memory,
                memo_path=memo_path,
                dry_run=True,
                on_summary=on_summary,
            )

        mock_compare.assert_not_awaited()
        mock_lesson.assert_not_awaited()
        assert lessons == []
        assert memory.add_calls == 0
        assert not memo_path.exists(), "a dry run must not persist a memo"

    @pytest.mark.asyncio
    async def test_dry_run_reports_the_projected_cost(self, tmp_path):
        memory = FakeLessonsMemory()
        snapshots = {
            "2767.T": [
                make_snapshot(age_days=180 + i, analysis_id=f"run-{i}")
                for i in range(6)
            ]
        }
        seen, on_summary = _capture()

        with patch("src.retrospective.load_past_snapshots", return_value=snapshots):
            await run_retrospective(
                "2767.T",
                Path("/fake"),
                memory,
                memo_path=tmp_path / "m.json",
                max_evaluations=4,
                dry_run=True,
                on_summary=on_summary,
            )

        summary = seen[-1]
        assert summary.dry_run is True
        assert summary.scanned == 6
        assert summary.evaluated == 4, "projected round-trips"
        assert summary.deferred_over_budget == 2
        assert summary.reconciles is True

    def test_cli_threads_the_flag_and_returns_before_generating(self, monkeypatch):
        """`--retrospective-dry-run` must reach the orchestrator, not just parse."""
        import asyncio
        from types import SimpleNamespace

        from src.main import _run_retrospective_only

        seen_kwargs: dict[str, object] = {}

        async def fake_run_retrospective(**kwargs):
            seen_kwargs.update(kwargs)
            summary = RetrospectiveRunSummary(
                scanned=7, skipped_too_recent=7, dry_run=True
            )
            kwargs["on_summary"](summary)
            return []

        monkeypatch.setattr(
            "src.retrospective.run_retrospective", fake_run_retrospective
        )
        args = SimpleNamespace(
            quiet=True, brief=False, no_memory=False, retrospective_dry_run=True
        )

        assert asyncio.run(_run_retrospective_only(args)) == 0
        assert seen_kwargs["dry_run"] is True

    def test_cli_defaults_to_a_real_run(self, monkeypatch):
        import asyncio
        from types import SimpleNamespace

        from src.main import _run_retrospective_only

        seen_kwargs: dict[str, object] = {}

        async def fake_run_retrospective(**kwargs):
            seen_kwargs.update(kwargs)
            return []

        monkeypatch.setattr(
            "src.retrospective.run_retrospective", fake_run_retrospective
        )
        # No `retrospective_dry_run` attribute at all — the getattr default must
        # hold for callers that predate the flag.
        args = SimpleNamespace(quiet=True, brief=False, no_memory=False)

        assert asyncio.run(_run_retrospective_only(args)) == 0
        assert seen_kwargs["dry_run"] is False

    @pytest.mark.asyncio
    async def test_a_dry_run_does_not_hide_a_snapshot_from_the_next_real_run(
        self, tmp_path
    ):
        """The memo must stay clean — a dry run is not an evaluation."""
        memory = FakeLessonsMemory()
        snapshots = {"2767.T": [make_snapshot(age_days=180, analysis_id="run-a")]}
        memo_path = tmp_path / "m.json"

        with patch("src.retrospective.load_past_snapshots", return_value=snapshots):
            await run_retrospective(
                "2767.T", Path("/fake"), memory, memo_path=memo_path, dry_run=True
            )

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch(
                "src.retrospective.compare_to_reality",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_compare,
        ):
            await run_retrospective(
                "2767.T", Path("/fake"), memory, memo_path=memo_path
            )

        mock_compare.assert_awaited_once()
