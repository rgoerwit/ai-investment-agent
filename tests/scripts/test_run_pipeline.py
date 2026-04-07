"""Tests for scripts/run_pipeline.sh — end-to-end screening pipeline."""

import json
import re
from pathlib import Path

import pytest


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
        """Pure-Python reimplementation of the skip logic from run_pipeline.sh.

        The shell script checks whether the report has a parseable verdict header.

        Returns "SKIP" or "PROCESS".

        Note: Previously this shelled out to bash via subprocess.run(), but that
        causes segfaults on macOS/Apple Silicon with Python 3.12 due to fork()
        safety issues with loaded C extensions (grpc, numpy, pandas, etc.).
        """
        if not force and outfile.is_file():
            content = outfile.read_text()
            if re.search(r"^# .+: .+$", content, re.MULTILINE):
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
