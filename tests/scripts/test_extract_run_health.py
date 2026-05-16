"""Tests for the per-run pipeline health-badge extractor.

The badge is the operator-visible signal users wanted (per the May 2026
DNS-outage postmortem). Signal-to-noise rule: nothing surfaces unless
counts cross per-key thresholds. A single transient DNS failure is *not*
news; ten of them is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts/ to sys.path for direct import.
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from _pipeline_log_parsing import (  # noqa: E402
    LogStats,
    format_badge,
    parse_log,
)


def _write_log(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "fake.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


class TestParseLog:
    def test_clean_log_returns_zeros(self, tmp_path):
        log = _write_log(
            tmp_path,
            [
                "[pipeline_child_start] 2026-05-16T10:49:24Z",
                "timestamp='10:49:31' level='info' event='llm_call_success'",
                "[pipeline_child_exit] status=0 elapsed=142s",
            ],
        )
        stats = parse_log(log)
        assert stats.dns_failures == 0
        assert stats.timeouts == 0
        assert stats.exit_status == 0
        assert stats.elapsed_seconds == 142.0
        assert stats.is_clean is True

    def test_counts_dns_and_timeout_events(self, tmp_path):
        log = _write_log(
            tmp_path,
            [
                "level='error' event='llm_call_failed' failure_kind='dns_resolution'",
                "level='error' event='llm_call_failed' failure_kind='dns_resolution'",
                "level='warning' event='llm_call_retry' failure_kind='dns_resolution'",
                "level='warning' event='hard_timeout_exceeded' label='market_context:^N225'",
                "level='warning' event='hard_timeout_exceeded' label='llm:Research Manager'",
                "[pipeline_child_exit] status=0 elapsed=300s",
            ],
        )
        stats = parse_log(log)
        assert stats.dns_failures == 3
        assert stats.timeouts == 2
        assert stats.elapsed_seconds == 300.0
        assert stats.is_clean is False

    def test_pipeline_child_timeout_marker_detected(self, tmp_path):
        log = _write_log(
            tmp_path,
            [
                "[pipeline_child_start] 2026-05-16T10:49:24Z",
                "[pipeline_child_timeout] exceeded 360s; requesting pending-task dump",
                "[pipeline_child_signal] SIGUSR1 pid=12345",
                "level='warning' event='pending_tasks_dump' count=4",
            ],
        )
        stats = parse_log(log)
        assert stats.pipeline_child_timeout is True
        assert stats.pending_task_dumps == 1

    def test_missing_file_returns_empty_stats(self, tmp_path):
        stats = parse_log(tmp_path / "does_not_exist.log")
        assert stats == LogStats()
        assert stats.is_clean is True

    def test_truncated_log_does_not_raise(self, tmp_path):
        # Log ends mid-line — common when a process is SIGTERM'd mid-write.
        log = tmp_path / "trunc.log"
        log.write_text(
            "level='error' event='llm_call_failed' failure_kind='dns_resol",
            encoding="utf-8",
        )
        stats = parse_log(log)
        # No traceback; partial line doesn't match the regex so dns=0.
        assert stats.dns_failures == 0
        assert stats.exit_status is None

    def test_breaker_event_detected(self, tmp_path):
        log = _write_log(
            tmp_path,
            [
                "level='warning' event='network_breaker_opened' failure_kind='dns_resolution' "
                "failures_in_window=4 cool_off_seconds=45.0",
                "[pipeline_child_exit] status=0 elapsed=89s",
            ],
        )
        stats = parse_log(log)
        assert stats.network_breaker_opened == 1
        assert stats.is_clean is False


class TestFormatBadge:
    def test_clean_stats_yields_empty_badge(self):
        assert format_badge(LogStats()) == ""

    def test_below_threshold_dns_is_suppressed(self):
        # 4 DNS failures < threshold (5) → no badge
        assert format_badge(LogStats(dns_failures=4)) == ""

    def test_at_threshold_dns_surfaces(self):
        out = format_badge(LogStats(dns_failures=5, elapsed_seconds=120))
        assert "dns=5" in out
        assert "elapsed=120s" in out
        assert out.startswith("net: ")

    def test_single_timeout_is_suppressed(self):
        # One timeout is the quick-mode slow-tail norm.
        assert format_badge(LogStats(timeouts=1)) == ""

    def test_two_timeouts_surface(self):
        assert "timeout=2" in format_badge(LogStats(timeouts=2, elapsed_seconds=200))

    def test_breaker_trip_always_surfaces(self):
        # Even with no DNS or timeout events crossing threshold.
        out = format_badge(LogStats(network_breaker_opened=1))
        assert "breaker_trip=1" in out

    def test_pipeline_child_timeout_always_surfaces(self):
        out = format_badge(LogStats(pipeline_child_timeout=True, elapsed_seconds=360))
        assert "child_timeout=1" in out
        assert "elapsed=360s" in out

    def test_mixed_signals_combined(self):
        out = format_badge(
            LogStats(
                dns_failures=14,
                timeouts=2,
                network_breaker_opened=1,
                elapsed_seconds=358,
            )
        )
        # Order is dns, timeout, breaker, then elapsed at the end.
        assert out == "net: dns=14 timeout=2 breaker_trip=1 elapsed=358s"


class TestExtractRunHealthCLI:
    """In-process CLI tests. We avoid `subprocess` because the full pytest
    run on macOS can poison fork/exec startup (earlier ChromaDB / tokenizers
    imports vs fork-safety) and the child interpreter can SIGSEGV before
    our script runs. Calling `main(argv)` directly tests the same contract."""

    def test_cli_emits_badge_on_dirty_log(self, tmp_path, capsys):
        from extract_run_health import main as cli_main

        log = _write_log(
            tmp_path,
            ["level='error' event='llm_call_failed' failure_kind='dns_resolution'"] * 6
            + ["[pipeline_child_exit] status=0 elapsed=240s"],
        )
        rc = cli_main(["extract_run_health.py", str(log)])
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.strip().startswith("net: dns=6")

    def test_cli_emits_nothing_on_clean_log(self, tmp_path, capsys):
        from extract_run_health import main as cli_main

        log = _write_log(
            tmp_path,
            ["[pipeline_child_exit] status=0 elapsed=120s"],
        )
        rc = cli_main(["extract_run_health.py", str(log)])
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out == ""

    def test_cli_returns_zero_on_misuse(self, capsys):
        from extract_run_health import main as cli_main

        # Wrong arg count → exit 0 (pipeline must never break).
        rc = cli_main(["extract_run_health.py"])
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out == ""

    def test_cli_returns_zero_on_missing_file(self, tmp_path, capsys):
        from extract_run_health import main as cli_main

        rc = cli_main(["extract_run_health.py", str(tmp_path / "ghost.log")])
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out == ""


class TestRealLogFixtures:
    """Replay the actual failing-ticker logs from the May 2026 DNS outage
    to make sure the badge would have surfaced what was missing then."""

    def test_6418T_log_surfaces_dns_cluster(self):
        # Real log from the outage. We don't ship the fixture; this test
        # is skipped when not present.
        real = (
            Path(__file__).resolve().parents[2]
            / "scratch"
            / "6418-T-LOG-2026-05-08_quick.txt"
        )
        if not real.exists():
            pytest.skip("Real outage log fixture not present in scratch/")
        stats = parse_log(real)
        # The actual run had ~14 DNS errors; threshold is 5.
        assert stats.dns_failures >= 5
        badge = format_badge(stats)
        assert "dns=" in badge
