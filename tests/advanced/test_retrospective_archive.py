"""Step 3: the retrospective can read archived analyses.

Local retention moves ``*_analysis.json`` out of ``RESULTS_DIR`` at around 120
days. ``TEMPORAL_WEIGHTS`` scores 91-270 days highest, so retention was deleting
the best evidence from the loader's view — measured, ``results/`` covered ages
0-137 d and the archive 139-257 d, i.e. most of the optimum was unreachable and
getting worse daily.

The subtle requirement is that *deduplication must not undo Step 1*: the same
artifact copied into an archive should collapse, while two genuinely distinct
same-day runs must both survive, because that pair is the model-change comparison
the whole plan exists to enable.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.retrospective import load_past_snapshots, run_retrospective
from src.retrospective_sources import (
    parse_archive_dirs,
    resolve_retrospective_sources,
)
from tests.advanced.retrospective_fakes import (
    FakeLessonsMemory,
    make_snapshot,
    write_analysis_artifact,
)


def _cfg(results_dir, archives: str = "") -> SimpleNamespace:
    return SimpleNamespace(results_dir=results_dir, retrospective_archive_dirs=archives)


# ══════════════════════════════════════════════════════════════════════════════
# Source resolution
# ══════════════════════════════════════════════════════════════════════════════


class TestSourceResolution:
    def test_no_configuration_yields_only_the_live_directory(self, tmp_path):
        assert resolve_retrospective_sources(_cfg(tmp_path)) == (tmp_path,)

    def test_archives_follow_the_live_directory(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        sources = resolve_retrospective_sources(_cfg(tmp_path, str(archive)))
        assert sources == (tmp_path, archive), "live results must be scanned first"

    def test_multiple_archives_are_separated_by_the_path_separator(self, tmp_path):
        import os

        first = tmp_path / "a"
        second = tmp_path / "b"
        first.mkdir()
        second.mkdir()
        raw = os.pathsep.join([str(first), str(second)])
        assert resolve_retrospective_sources(_cfg(tmp_path, raw)) == (
            tmp_path,
            first,
            second,
        )

    def test_a_missing_archive_is_dropped_not_raised(self, tmp_path):
        sources = resolve_retrospective_sources(
            _cfg(tmp_path, str(tmp_path / "does-not-exist"))
        )
        assert sources == (tmp_path,)

    def test_the_live_directory_is_never_scanned_twice(self, tmp_path):
        sources = resolve_retrospective_sources(_cfg(tmp_path, str(tmp_path)))
        assert sources == (tmp_path,)

    def test_duplicate_archive_entries_collapse(self, tmp_path):
        import os

        archive = tmp_path / "archive"
        archive.mkdir()
        raw = os.pathsep.join([str(archive), str(archive)])
        assert resolve_retrospective_sources(_cfg(tmp_path, raw)) == (
            tmp_path,
            archive,
        )

    def test_blank_entries_are_ignored(self, tmp_path):
        assert parse_archive_dirs("") == ()
        assert parse_archive_dirs(None) == ()
        assert parse_archive_dirs("  :  ") == ()

    def test_tilde_is_expanded(self):
        (expanded,) = parse_archive_dirs("~/some-archive")
        assert not str(expanded).startswith("~")
        assert str(expanded).endswith("some-archive")


# ══════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════


class TestArchiveLoading:
    def test_snapshots_from_both_trees_are_merged(self, tmp_path):
        live, archive = tmp_path / "results", tmp_path / "archive"
        write_analysis_artifact(live, make_snapshot(age_days=60, analysis_id="run-new"))
        write_analysis_artifact(
            archive, make_snapshot(age_days=200, analysis_id="run-old")
        )

        loaded = load_past_snapshots(None, live, archive_dirs=[archive])

        assert {s["analysis_id"] for s in loaded["2767.T"]} == {"run-new", "run-old"}

    def test_without_archives_the_old_behaviour_is_exact(self, tmp_path):
        live, archive = tmp_path / "results", tmp_path / "archive"
        write_analysis_artifact(live, make_snapshot(age_days=60, analysis_id="run-new"))
        write_analysis_artifact(
            archive, make_snapshot(age_days=200, analysis_id="run-old")
        )

        loaded = load_past_snapshots(None, live)

        assert {s["analysis_id"] for s in loaded["2767.T"]} == {"run-new"}

    def test_the_live_copy_wins_over_an_archived_duplicate(self, tmp_path):
        live, archive = tmp_path / "results", tmp_path / "archive"
        snapshot = make_snapshot(age_days=200, analysis_id="run-a")
        name = snapshot["_source_file"]
        write_analysis_artifact(live, {**snapshot, "verdict": "BUY"}, filename=name)
        write_analysis_artifact(archive, {**snapshot, "verdict": "HOLD"}, filename=name)

        loaded = load_past_snapshots(None, live, archive_dirs=[archive])

        assert len(loaded["2767.T"]) == 1
        assert loaded["2767.T"][0]["verdict"] == "BUY"

    def test_the_same_run_under_two_filenames_collapses_by_identity(self, tmp_path):
        """An archived copy that was renamed still resolves to one analysis."""
        live, archive = tmp_path / "results", tmp_path / "archive"
        snapshot = make_snapshot(age_days=200, analysis_id="run-a")
        write_analysis_artifact(live, snapshot, filename="2767.T_a_analysis.json")
        write_analysis_artifact(
            archive, snapshot, filename="2767.T_a_copy_analysis.json"
        )

        loaded = load_past_snapshots(None, live, archive_dirs=[archive])

        assert len(loaded["2767.T"]) == 1

    def test_two_distinct_same_day_runs_both_survive(self, tmp_path):
        """Dedup must not undo Step 1 — this pair is the model-change comparison."""
        live = tmp_path / "results"
        morning = make_snapshot(age_days=200, analysis_id="run-morning")
        evening = make_snapshot(age_days=200, analysis_id="run-evening")
        write_analysis_artifact(live, morning, filename="2767.T_0900_analysis.json")
        write_analysis_artifact(live, evening, filename="2767.T_2100_analysis.json")

        loaded = load_past_snapshots(None, live)

        assert (
            loaded["2767.T"][0]["analysis_date"] == loaded["2767.T"][1]["analysis_date"]
        )
        assert len(loaded["2767.T"]) == 2

    def test_two_legacy_same_day_runs_also_survive(self, tmp_path):
        """No analysis_id — identity falls back to the (unique) filename."""
        live = tmp_path / "results"
        base = make_snapshot(age_days=200, analysis_id=None)
        write_analysis_artifact(
            live, base, filename="2767.T_20260101_090000_analysis.json"
        )
        write_analysis_artifact(
            live, base, filename="2767.T_20260101_210000_analysis.json"
        )

        loaded = load_past_snapshots(None, live)

        assert len(loaded["2767.T"]) == 2

    def test_a_nonexistent_archive_is_skipped_silently(self, tmp_path):
        live = tmp_path / "results"
        write_analysis_artifact(live, make_snapshot(age_days=60, analysis_id="run-a"))

        loaded = load_past_snapshots(
            None, live, archive_dirs=[tmp_path / "nope", tmp_path / "also-nope"]
        )

        assert len(loaded["2767.T"]) == 1

    def test_a_file_where_a_directory_was_expected_is_skipped(self, tmp_path):
        live = tmp_path / "results"
        write_analysis_artifact(live, make_snapshot(age_days=60, analysis_id="run-a"))
        not_a_dir = tmp_path / "regular-file"
        not_a_dir.write_text("hello")

        loaded = load_past_snapshots(None, live, archive_dirs=[not_a_dir])

        assert len(loaded["2767.T"]) == 1

    def test_malformed_archive_json_is_skipped(self, tmp_path):
        live, archive = tmp_path / "results", tmp_path / "archive"
        write_analysis_artifact(live, make_snapshot(age_days=60, analysis_id="run-a"))
        archive.mkdir()
        (archive / "broken_analysis.json").write_text("{not json")

        loaded = load_past_snapshots(None, live, archive_dirs=[archive])

        assert len(loaded["2767.T"]) == 1

    def test_an_archive_is_readable_even_when_the_live_dir_is_gone(self, tmp_path):
        """Retention could plausibly outrun a re-created results dir."""
        archive = tmp_path / "archive"
        write_analysis_artifact(
            archive, make_snapshot(age_days=200, analysis_id="run-old")
        )

        loaded = load_past_snapshots(
            None, tmp_path / "missing-results", archive_dirs=[archive]
        )

        assert len(loaded["2767.T"]) == 1

    def test_a_missing_live_dir_with_no_archive_still_returns_empty(self, tmp_path):
        assert load_past_snapshots(None, tmp_path / "missing") == {}

    def test_the_ticker_filter_applies_to_archives_too(self, tmp_path):
        live, archive = tmp_path / "results", tmp_path / "archive"
        write_analysis_artifact(live, make_snapshot(age_days=60, analysis_id="run-a"))
        write_analysis_artifact(
            archive,
            make_snapshot(ticker="7203.T", age_days=200, analysis_id="run-other"),
        )
        write_analysis_artifact(
            archive, make_snapshot(age_days=200, analysis_id="run-b")
        )

        loaded = load_past_snapshots("2767.T", live, archive_dirs=[archive])

        assert set(loaded) == {"2767.T"}
        assert len(loaded["2767.T"]) == 2

    def test_progress_totals_span_every_source(self, tmp_path):
        live, archive = tmp_path / "results", tmp_path / "archive"
        write_analysis_artifact(live, make_snapshot(age_days=60, analysis_id="run-a"))
        write_analysis_artifact(
            archive, make_snapshot(age_days=200, analysis_id="run-b")
        )
        updates = []

        load_past_snapshots(None, live, archive_dirs=[archive], progress=updates.append)

        discovered = next(u for u in updates if u.phase == "discovered")
        assert discovered.total_files == 2


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════════


class TestArchivedSnapshotsAreEvaluated:
    @pytest.mark.asyncio
    async def test_an_archived_snapshot_reaches_the_comparison(self, tmp_path):
        live, archive = tmp_path / "results", tmp_path / "archive"
        write_analysis_artifact(live, make_snapshot(age_days=5, analysis_id="run-new"))
        write_analysis_artifact(
            archive, make_snapshot(age_days=200, analysis_id="run-old")
        )
        seen: list[str] = []

        async def _price(snapshot):
            seen.append(snapshot["analysis_id"])
            return None

        with patch("src.retrospective.compare_to_reality", side_effect=_price):
            await run_retrospective(
                "2767.T",
                live,
                FakeLessonsMemory(),
                archive_dirs=[archive],
                memo_path=tmp_path / "m.json",
            )

        assert seen == ["run-old"], "the too-recent live snapshot must not be priced"

    @pytest.mark.asyncio
    async def test_archives_are_off_by_default(self, tmp_path):
        live, archive = tmp_path / "results", tmp_path / "archive"
        write_analysis_artifact(
            archive, make_snapshot(age_days=200, analysis_id="run-old")
        )
        live.mkdir()

        with patch(
            "src.retrospective.compare_to_reality",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_compare:
            await run_retrospective(
                "2767.T", live, FakeLessonsMemory(), memo_path=tmp_path / "m.json"
            )

        mock_compare.assert_not_awaited()


class TestCliWiring:
    def test_both_entry_points_resolve_sources_from_config(self):
        """A config-only archive must reach the loader, not just parse."""
        import inspect

        import src.main as main_module

        source = inspect.getsource(main_module)
        assert source.count("resolve_retrospective_sources(config)") == 2, (
            "both _run_retrospective_only and _maybe_run_ticker_retrospective "
            "must resolve archive dirs; one of them is reading results_dir directly"
        )
        assert source.count("archive_dirs=archive_dirs") == 2

    def test_the_settings_field_is_a_real_override_surface(self):
        from src.config import Settings

        field = Settings.model_fields["retrospective_archive_dirs"]
        assert field.default == "", "archives must be opt-in"
        assert field.validation_alias == "RETROSPECTIVE_ARCHIVE_DIRS"

    def test_default_settings_resolve_to_the_live_directory_only(self):
        from src.config import Settings

        sources = resolve_retrospective_sources(
            SimpleNamespace(
                results_dir=Path("results"),
                retrospective_archive_dirs=Settings.model_fields[
                    "retrospective_archive_dirs"
                ].default,
            )
        )
        assert sources == (Path("results"),)
