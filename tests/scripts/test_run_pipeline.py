"""Tests for scripts/run_pipeline.sh — end-to-end screening pipeline."""

import json
import re
from pathlib import Path

import pytest

_RUN_PIPELINE_PATH = Path(__file__).parent.parent.parent / "scripts" / "run_pipeline.sh"


# ============================================================
# TestVerdictExtraction — regex patterns matching report format
# ============================================================
class TestVerdictExtraction:
    """Test the verdict extraction patterns used by run_pipeline.sh.

    The shell script parses headers in either of these forms:
      # TICKER (Company Name): BUY
      # TICKER: DO_NOT_INITIATE
    """

    HEADER_PATTERN = re.compile(r"^# .+: (?P<verdict>[^\r\n]+)$", re.MULTILINE)

    @classmethod
    def _extract_verdict(cls, content: str) -> str | None:
        match = cls.HEADER_PATTERN.search(content)
        return match.group("verdict") if match else None

    def test_buy_detected(self, tmp_path):
        content = "# 8002.T (Marubeni Corporation): BUY\n"
        assert self._extract_verdict(content) == "BUY"

    def test_sell_detected(self, tmp_path):
        content = "# UNTR.JK (PT United Tractors Tbk): SELL\n"
        assert self._extract_verdict(content) == "SELL"

    def test_hold_detected(self):
        content = "# 7740.T (Tamron Co.,Ltd.): HOLD\n"
        assert self._extract_verdict(content) == "HOLD"

    def test_do_not_initiate_detected(self):
        content = "# X.Y (Foo Corp): DO NOT INITIATE\n"
        assert self._extract_verdict(content) == "DO NOT INITIATE"

    def test_no_company_name_header_detected(self):
        content = "# 262A.T: DO NOT INITIATE\n"
        assert self._extract_verdict(content) == "DO NOT INITIATE"

    def test_exchange_qualified_numeric_headers_detected(self):
        assert self._extract_verdict("# 2628.HK (Foo): BUY\n") == "BUY"
        assert self._extract_verdict("# 2628.TW (Bar): HOLD\n") == "HOLD"
        assert self._extract_verdict("# 2628.T (Baz): SELL\n") == "SELL"

    def test_verdict_at_line_10(self):
        """Report has preamble lines before the verdict title."""
        lines = [
            "---",
            "title: Analysis Report",
            "date: 2026-02-20",
            "ticker: 8002.T",
            "mode: quick",
            "model: gemini-3-pro-preview",
            "version: 1.0",
            "thesis: GARP",
            "---",
            "# 8002.T (Marubeni Corporation): BUY",
        ]
        content = "\n".join(lines) + "\n"
        assert self._extract_verdict(content) == "BUY"

    def test_commas_in_company_name(self):
        """Company names with commas and periods should still match."""
        content = "# 7740.T (Tamron Co.,Ltd.): HOLD\n"
        assert self._extract_verdict(content) == "HOLD"

    def test_parentheses_in_company_name(self):
        """Nested parentheses shouldn't break the match."""
        content = "# 2330.TW (Taiwan Semiconductor (TSMC)): BUY\n"
        assert self._extract_verdict(content) == "BUY"

    def test_no_verdict_line(self):
        """File without verdict line should not match."""
        content = "Some random content\nNo verdict here\n"
        assert self._extract_verdict(content) is None

    def test_buy_must_be_exact_end(self):
        """BUY pattern requires exact match at end of line (no trailing text)."""
        content = "# X.Y (Foo): BUY_SOMETHING\n"
        assert self._extract_verdict(content) == "BUY_SOMETHING"

    def test_verdict_with_spaces_detected(self):
        content = "# 0142.HK (First Pacific Company Limited): DO NOT INITIATE\n"
        assert self._extract_verdict(content) == "DO NOT INITIATE"


# ============================================================
# TestTickerToDash — filename convention
# ============================================================
class TestTickerToDash:
    """Test the ticker-to-dash conversion used for output filenames.

    The shell script uses: DASH=$(echo "$ticker" | tr '._' '-')
    """

    @staticmethod
    def _ticker_to_dash(ticker: str) -> str:
        """Python equivalent of: echo "$ticker" | tr '._' '-'"""
        return ticker.translate(str.maketrans("._", "--"))

    def test_dot_suffix(self):
        assert self._ticker_to_dash("0005.HK") == "0005-HK"

    def test_underscore(self):
        assert self._ticker_to_dash("FOO_BAR") == "FOO-BAR"

    def test_complex_ticker(self):
        assert self._ticker_to_dash("PINFRA.MX") == "PINFRA-MX"

    def test_multi_dot(self):
        assert self._ticker_to_dash("A.B.TO") == "A-B-TO"

    def test_no_special_chars(self):
        assert self._ticker_to_dash("AAPL") == "AAPL"

    def test_mixed(self):
        assert self._ticker_to_dash("BRK_B.TO") == "BRK-B-TO"


class TestRunDateDerivation:
    DATE_RE = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})")

    @classmethod
    def _extract_date(cls, path: str) -> str | None:
        match = cls.DATE_RE.search(Path(path).name)
        return match.group(1) if match else None

    @classmethod
    def _derive_run_date(
        cls,
        today: str,
        *,
        skip_scrape: str = "",
        buys_file: str = "",
        run_date: str = "",
    ) -> str:
        if run_date:
            return run_date
        if buys_file:
            extracted = cls._extract_date(buys_file)
            if extracted:
                return extracted
        if skip_scrape:
            extracted = cls._extract_date(skip_scrape)
            if extracted:
                return extracted
        return today

    def test_skip_scrape_date_drives_stage1_resume(self):
        assert (
            self._derive_run_date(
                "2026-03-20", skip_scrape="scratch/gems_2026-03-19.txt"
            )
            == "2026-03-19"
        )

    def test_buys_file_date_drives_stage2_resume(self):
        assert (
            self._derive_run_date("2026-03-20", buys_file="scratch/buys_2026-03-18.txt")
            == "2026-03-18"
        )

    def test_explicit_run_date_wins(self):
        assert (
            self._derive_run_date(
                "2026-03-20",
                skip_scrape="scratch/gems_2026-03-19.txt",
                buys_file="scratch/buys_2026-03-18.txt",
                run_date="2026-03-17",
            )
            == "2026-03-17"
        )

    @staticmethod
    def _detect_stage1_resume_date(
        inferred_date: str,
        today: str,
        completed_counts: dict[str, int],
    ) -> str:
        inferred_count = completed_counts.get(inferred_date, 0)
        today_count = completed_counts.get(today, 0)
        return today if today_count > inferred_count else inferred_date

    def test_stage1_prefers_today_when_more_outputs_exist(self):
        assert (
            self._detect_stage1_resume_date(
                "2026-03-19",
                "2026-03-20",
                {"2026-03-19": 6, "2026-03-20": 149},
            )
            == "2026-03-20"
        )

    def test_stage1_keeps_inferred_date_when_it_has_more_outputs(self):
        assert (
            self._detect_stage1_resume_date(
                "2026-03-19",
                "2026-03-20",
                {"2026-03-19": 149, "2026-03-20": 6},
            )
            == "2026-03-19"
        )


# ============================================================
# TestResumability — skip logic
# ============================================================
class TestResumability:
    """Test the resumability/skip logic in run_pipeline.sh.

    The shell script checks:
      if ! $FORCE && [[ -f "$OUTFILE" ]] && grep -qE '^# .*\\): ' "$OUTFILE"; then
          # SKIP
      fi
    """

    @staticmethod
    def _run_skip_check(outfile: Path, force: bool = False) -> str:
        """Pure-Python reimplementation of ``report_is_complete``-gated skip.

        The shell script skips a ticker only if its report has a verdict header
        AND that verdict is not the ``ANALYSIS FAILED`` sentinel — an
        unpublishable soft-failure that must be retried, not skipped forever.

        Returns "SKIP" or "PROCESS".

        Note: Previously this shelled out to bash via subprocess.run(), but that
        causes segfaults on macOS/Apple Silicon with Python 3.12 due to fork()
        safety issues with loaded C extensions (grpc, numpy, pandas, etc.).
        """
        if not force and outfile.is_file():
            content = outfile.read_text()
            m = re.search(r"^# .+: (?P<verdict>[^\r\n]+)$", content, re.MULTILINE)
            if m and m.group("verdict").strip() != "ANALYSIS FAILED":
                return "SKIP"
        return "PROCESS"

    def test_existing_report_with_verdict_skipped(self, tmp_path):
        outfile = tmp_path / "report.md"
        outfile.write_text("# 7203.T (Toyota Motor Corporation): BUY\n")

        result = self._run_skip_check(outfile, force=False)
        assert result == "SKIP"

    def test_existing_report_without_verdict_not_skipped(self, tmp_path):
        outfile = tmp_path / "report.md"
        outfile.write_text("Partial output, no verdict line\n")

        result = self._run_skip_check(outfile, force=False)
        assert result == "PROCESS"

    def test_force_flag_overrides_skip(self, tmp_path):
        outfile = tmp_path / "report.md"
        outfile.write_text("# 7203.T (Toyota Motor Corporation): BUY\n")

        result = self._run_skip_check(outfile, force=True)
        assert result == "PROCESS"

    def test_missing_file_not_skipped(self, tmp_path):
        outfile = tmp_path / "nonexistent.md"

        result = self._run_skip_check(outfile, force=False)
        assert result == "PROCESS"

    def test_sell_verdict_also_skipped(self, tmp_path):
        """SELL verdict should also trigger skip (any verdict counts)."""
        outfile = tmp_path / "report.md"
        outfile.write_text("# FAIL.T (Bad Corp): SELL\n")

        result = self._run_skip_check(outfile, force=False)
        assert result == "SKIP"

    def test_do_not_initiate_also_skipped(self, tmp_path):
        outfile = tmp_path / "report.md"
        outfile.write_text("# X.Y (Foo Corp): DO NOT INITIATE\n")

        result = self._run_skip_check(outfile, force=False)
        assert result == "SKIP"

    def test_no_company_name_header_also_skipped(self, tmp_path):
        outfile = tmp_path / "report.md"
        outfile.write_text("# 262A.T: DO NOT INITIATE\n")

        result = self._run_skip_check(outfile, force=False)
        assert result == "SKIP"

    def test_analysis_failed_report_is_not_skipped(self, tmp_path):
        """An ANALYSIS FAILED report is a soft failure, not a completed
        analysis — it must be re-run on the next pass, not skipped forever."""
        outfile = tmp_path / "report.md"
        outfile.write_text("# 1999.HK (Man Wah Holdings Limited): ANALYSIS FAILED\n")

        result = self._run_skip_check(outfile, force=False)
        assert result == "PROCESS"

    def test_force_still_processes_analysis_failed(self, tmp_path):
        outfile = tmp_path / "report.md"
        outfile.write_text("# 1999.HK (Man Wah): ANALYSIS FAILED\n")

        assert self._run_skip_check(outfile, force=True) == "PROCESS"


class TestAnalysisFailedHandling:
    """Contract: an ANALYSIS FAILED report (unpublishable soft failure) must be
    retried and surfaced loudly, never counted as a green success."""

    @staticmethod
    def _script_text() -> str:
        return _RUN_PIPELINE_PATH.read_text()

    def test_report_is_complete_helper_exists_and_excludes_failed(self):
        script = self._script_text()
        assert "report_is_complete()" in script
        assert '"$verdict" != "ANALYSIS FAILED"' in script
        # The old unconditional predicate is fully replaced.
        assert "report_has_verdict_header" not in script

    def test_resumability_uses_report_is_complete(self):
        script = self._script_text()
        # Both stage skip checks gate on report_is_complete, not a bare header.
        assert (
            'if ! $FORCE && [[ -f "$OUTFILE" ]] && report_is_complete "$OUTFILE"; then'
            in script
        )

    def test_no_analysis_is_warned_not_succeeded(self):
        script = self._script_text()
        # The exit-0 branch distinguishes ANALYSIS FAILED / empty verdict.
        assert '"$VERDICT" == "ANALYSIS FAILED"' in script
        assert 'warn "NO ANALYSIS: $ticker' in script
        # Both stages tally + report a no-analysis count.
        assert "STAGE1_NOANALYSIS" in script
        assert "STAGE2_NOANALYSIS" in script
        assert "no-analysis (will retry)" in script


class TestTimeoutBudgetContract:
    """The Stage-1 per-ticker watchdog must give the gate-critical APEX seats'
    larger --quick per-call budget room to finish (2026-07-06 fix), and every
    failure must leave a durable, rerun-proof record."""

    @staticmethod
    def _script_text() -> str:
        return _RUN_PIPELINE_PATH.read_text()

    def test_stage1_watchdog_default_raised_to_600(self):
        script = self._script_text()
        assert (
            'STAGE1_TICKER_TIMEOUT_SECONDS="${STAGE1_TICKER_TIMEOUT_SECONDS:-600}"'
            in script
        )
        # The old 360 default (which the collapsed 60s call budget dies well
        # inside, and which was too tight for a standard-tier APEX seat) is gone.
        assert ":-360}" not in script

    def test_stage2_watchdog_default_preserved(self):
        script = self._script_text()
        assert (
            'STAGE2_TICKER_TIMEOUT_SECONDS="${STAGE2_TICKER_TIMEOUT_SECONDS:-2400}"'
            in script
        )

    def test_watchdog_printed_in_previews(self):
        script = self._script_text()
        assert "Per-ticker watchdog: ${STAGE1_TICKER_TIMEOUT_SECONDS}s" in script
        assert "Per-ticker watchdog:   ${STAGE2_TICKER_TIMEOUT_SECONDS}s" in script

    def test_durable_failure_log_helper_exists_and_is_called(self):
        script = self._script_text()
        assert "record_pipeline_failure()" in script
        assert "pipeline_failures-${DATE}.log" in script
        # Recorded on both the exit-0 no-analysis path and the child-exit path,
        # in both stages.
        assert 'record_pipeline_failure "$ticker" 1 no_analysis' in script
        assert 'record_pipeline_failure "$ticker" 1 "child_exit_${status}"' in script
        assert 'record_pipeline_failure "$ticker" 2 no_analysis' in script
        assert 'record_pipeline_failure "$ticker" 2 "child_exit_${status}"' in script


class TestStage1Contract:
    @staticmethod
    def _script_text() -> str:
        return _RUN_PIPELINE_PATH.read_text()

    def test_stage1_preview_mode_no_longer_mentions_strict(self):
        script = self._script_text()
        assert "Mode:                --quick --brief --no-memory" in script
        assert "Mode:                --quick --strict --brief --no-memory" not in script

    def test_stage1_quick_command_is_non_strict(self):
        script = self._script_text()
        assert "--quick --no-charts --quiet --brief --no-memory \\" in script
        assert (
            "--quick --strict --no-charts --quiet --brief --no-memory \\" not in script
        )

    def test_help_text_limits_strict_to_stage2(self):
        script = self._script_text()
        assert "Apply strict mode to Stage 2 full analysis" in script
        assert (
            "Stage 1 screening always runs strict regardless of this flag."
            not in script
        )


class TestPythonRuntimeResolution:
    @staticmethod
    def _script_text() -> str:
        return _RUN_PIPELINE_PATH.read_text()

    def test_active_venv_is_validated_before_use(self):
        script = self._script_text()
        assert 'python -c "import pandas, requests, yfinance"' in script

    def test_missing_active_venv_deps_falls_back_to_poetry(self):
        script = self._script_text()
        assert (
            "Active virtual environment lacks repo dependencies; falling back to Poetry runtime"
            in script
        )
        assert (
            "Deactivate the unrelated venv, or install this project's dependencies into it"
            in script
        )

    def test_default_cooldown_is_ten_seconds(self):
        script = self._script_text()
        assert 'COOLDOWN="${COOLDOWN_SECONDS:-10}"' in script
        assert (
            "--cooldown N        Seconds between ticker analyses (default: 10)"
            in script
        )


class TestPipelineCancellationContract:
    @staticmethod
    def _script_text() -> str:
        return _RUN_PIPELINE_PATH.read_text()

    def test_script_sources_shared_signal_helper(self):
        script = self._script_text()
        assert 'source "${SCRIPT_DIR}/pipeline_signals.sh"' in script
        assert "signal_proxy.py" not in script

    def test_stage_commands_run_through_tracked_child_helper(self):
        script = self._script_text()
        assert 'if run_tracked_child "" "${GEMS_CMD[@]}"; then' in script
        # Stage-1 and stage-2 ticker analyses route through the durable-log
        # wrapper; that wrapper still calls run_tracked_child, preserving the
        # watchdog/cancellation contract while keeping per-run detail logs.
        assert "run_pipeline_child() {" in script
        assert 'run_tracked_child "$detail_logfile" "$@"' in script
        assert (
            'run_pipeline_child "$DETAIL_LOGFILE" "$LOGFILE" "${PYTHON_CMD[@]}" -m src.main \\'
            in script
        )
        assert (
            'DETAIL_LOGFILE="${SCRATCH}/${DASH}-LOG-${DATE}_quick-${PIPELINE_RUN_ID}.txt"'
            in script
        )
        assert (
            'DETAIL_LOGFILE="${SCRATCH}/${DASH}-LOG-${DATE}-${PIPELINE_RUN_ID}.txt"'
            in script
        )
        assert (
            'PIPELINE_TICKER_TIMEOUT_SECONDS="${STAGE1_TICKER_TIMEOUT_SECONDS}"'
            in script
        )
        assert (
            'PIPELINE_TICKER_TIMEOUT_SECONDS="${STAGE2_TICKER_TIMEOUT_SECONDS}"'
            in script
        )

    def test_interrupt_status_aborts_later_stages(self):
        script = self._script_text()
        assert "exit_if_interrupted_status" in script

    def test_dead_pg_tracking_and_setsid_branch_are_removed(self):
        script = self._script_text()
        assert "process_group_id_for_pid" not in script
        assert "setsid" not in script
        assert "PIPELINE_SIGNAL_PROXY" not in script

    def test_help_text_mentions_process_group_force_stop(self):
        script = self._script_text()
        assert (
            "Force stop         Kill the whole process group, not only the shell PID"
            in script
        )


class TestPipelineMarkerPayload:
    @staticmethod
    def _build_marker_payload(
        screening_date: str,
        *,
        ticker_count: int | None,
        buy_count: int | None,
        completed_at: str = "2026-04-05T22:14:19Z",
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "workflow": "run_pipeline",
            "screening_date": screening_date,
            "completed_at": completed_at,
            "candidate_count": ticker_count,
            "buy_count": buy_count,
        }

    def test_zero_buy_completion_still_records_marker(self):
        payload = self._build_marker_payload(
            "2026-04-05",
            ticker_count=312,
            buy_count=0,
        )
        assert payload["screening_date"] == "2026-04-05"
        assert payload["buy_count"] == 0

    def test_stage2_resume_can_leave_candidate_count_unknown(self):
        payload = self._build_marker_payload(
            "2026-03-18",
            ticker_count=None,
            buy_count=12,
            completed_at="2026-04-05T22:14:19Z",
        )
        assert payload["screening_date"] == "2026-03-18"
        assert payload["candidate_count"] is None

    def test_marker_payload_is_json_serializable(self):
        payload = self._build_marker_payload(
            "2026-04-05",
            ticker_count=245,
            buy_count=12,
        )
        rendered = json.dumps(payload)
        assert '"workflow": "run_pipeline"' in rendered
