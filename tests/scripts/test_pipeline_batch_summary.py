"""Tests for the end-of-batch warning summary.

Contract (per operator request): emit a single line ONLY when failures or
critical errors are present in the batch. Clean batches print nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from _pipeline_log_parsing import LogStats  # noqa: E402
from pipeline_batch_summary import build_summary  # noqa: E402


def _stats(**kwargs) -> LogStats:
    return LogStats(**kwargs)


class TestBuildSummary:
    def test_empty_input_returns_none(self):
        assert build_summary([]) is None

    def test_clean_batch_returns_none(self):
        # 5 clean runs, exit 0, no DNS/timeout/breaker events.
        batch = [_stats(exit_status=0, elapsed_seconds=120) for _ in range(5)]
        assert build_summary(batch) is None

    def test_single_failed_run_surfaces(self):
        batch = [
            _stats(exit_status=0),
            _stats(exit_status=1),
            _stats(exit_status=0),
        ]
        summary = build_summary(batch)
        assert summary is not None
        assert "runs=3" in summary
        assert "failed=1" in summary

    def test_pipeline_child_timeout_surfaces(self):
        batch = [_stats(exit_status=0), _stats(pipeline_child_timeout=True)]
        summary = build_summary(batch)
        assert summary is not None
        assert "child_timeouts=1" in summary

    def test_breaker_trip_surfaces(self):
        batch = [_stats(exit_status=0), _stats(exit_status=0, network_breaker_opened=1)]
        summary = build_summary(batch)
        assert summary is not None
        assert "breaker_trips=1" in summary

    def test_consecutive_dns_cluster_surfaces_even_when_runs_succeed(self):
        # 3 successful runs that each had 5+ DNS retries — graceful
        # degradation, exit 0, but the network was clearly sick. This is
        # the May 2026 outage shape and must NOT be silent.
        batch = [
            _stats(exit_status=0, dns_failures=6),
            _stats(exit_status=0, dns_failures=8),
            _stats(exit_status=0, dns_failures=5),
        ]
        summary = build_summary(batch)
        assert summary is not None
        assert "consecutive_dns_runs=3" in summary
        assert "dominant_failure_kind=dns_resolution" in summary

    def test_two_consecutive_dns_does_not_trigger_cluster(self):
        # 2 elevated runs, not 3 — below the cluster threshold. Pure
        # success path so failed=0, no other signals → silent.
        batch = [
            _stats(exit_status=0, dns_failures=6),
            _stats(exit_status=0, dns_failures=6),
            _stats(exit_status=0, dns_failures=0),
        ]
        assert build_summary(batch) is None

    def test_dominant_failure_kind_picks_top_count(self):
        batch = [
            _stats(exit_status=1, dns_failures=20, timeouts=2),
            _stats(exit_status=0, dns_failures=3, timeouts=8),
        ]
        summary = build_summary(batch)
        assert summary is not None
        # 23 dns vs 10 timeout → dns wins
        assert "dominant_failure_kind=dns_resolution" in summary

    def test_summary_is_single_line(self):
        batch = [_stats(exit_status=1) for _ in range(3)]
        summary = build_summary(batch)
        assert summary is not None
        assert "\n" not in summary


class TestCLIContract:
    """In-process CLI tests. Avoids `subprocess` because earlier tests in
    the full suite can poison fork/exec startup on macOS (ChromaDB /
    tokenizers fork-safety) and the subprocess child can SIGSEGV before
    our script even runs. Calling `main(argv)` directly tests the same
    contract — argv handling, stdout content, return code."""

    def test_clean_dir_emits_nothing(self, tmp_path, capsys):
        from pipeline_batch_summary import main as cli_main

        rc = cli_main(["pipeline_batch_summary.py", str(tmp_path)])
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out == ""

    def test_missing_dir_does_not_crash(self, capsys):
        from pipeline_batch_summary import main as cli_main

        rc = cli_main(["pipeline_batch_summary.py", "/nonexistent/path"])
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out == ""

    def test_dirty_dir_emits_warning_line(self, tmp_path, capsys):
        from pipeline_batch_summary import main as cli_main

        log = tmp_path / "FAIL-LOG-2026-05-08_quick.txt"
        log.write_text(
            "level='error' event='llm_call_failed' failure_kind='dns_resolution'\n" * 8
            + "[pipeline_child_timeout] exceeded 360s\n"
            + "[pipeline_child_exit] status=124 elapsed=360s\n",
            encoding="utf-8",
        )

        rc = cli_main(["pipeline_batch_summary.py", str(tmp_path)])
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.strip().startswith("pipeline_batch_summary ")
        assert "failed=1" in captured.out
        assert "dominant_failure_kind=dns_resolution" in captured.out

    def test_real_outage_fixture_surfaces(self, tmp_path, capsys):
        import shutil

        from pipeline_batch_summary import main as cli_main

        scratch = Path(__file__).resolve().parents[2] / "scratch"
        names = (
            "6272-TW-LOG-2026-05-08_quick.txt",
            "6414-TW-LOG-2026-05-08_quick.txt",
            "6418-T-LOG-2026-05-08_quick.txt",
        )
        if not any((scratch / n).exists() for n in names):
            import pytest

            pytest.skip("Real outage logs not present in scratch/")

        # Isolate the fixtures so we only summarize those three.
        for name in names:
            src = scratch / name
            if src.exists():
                shutil.copy(src, tmp_path / name)

        rc = cli_main(["pipeline_batch_summary.py", str(tmp_path)])
        captured = capsys.readouterr()
        assert rc == 0
        assert "pipeline_batch_summary" in captured.out
        assert "failed=" in captured.out
