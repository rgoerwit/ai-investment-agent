"""Tests for scripts/run_pipeline.sh — end-to-end screening pipeline."""

import re
from pathlib import Path

import pytest


# ============================================================
# TestVerdictExtraction — regex patterns matching report format
# ============================================================
class TestVerdictExtraction:
    """Test the verdict extraction patterns used by run_pipeline.sh.

    The shell script uses grep patterns like:
      grep -qE '^# .*\\): BUY$' "$OUTFILE"
    to determine the verdict from the markdown title line.
    """

    # The same regex patterns used in run_pipeline.sh
    BUY_PATTERN = re.compile(r"^# .*\): BUY$", re.MULTILINE)
    SELL_PATTERN = re.compile(r"^# .*\): SELL", re.MULTILINE)
    HOLD_PATTERN = re.compile(r"^# .*\): HOLD", re.MULTILINE)
    DNI_PATTERN = re.compile(r"^# .*\): DO_NOT_INITIATE", re.MULTILINE)
    # Generic "has verdict" pattern (used for skip logic)
    VERDICT_PATTERN = re.compile(r"^# .*\): ", re.MULTILINE)

    def test_buy_detected(self, tmp_path):
        content = "# 8002.T (Marubeni Corporation): BUY\n"
        assert self.BUY_PATTERN.search(content) is not None

    def test_sell_detected(self, tmp_path):
        content = "# UNTR.JK (PT United Tractors Tbk): SELL\n"
        assert self.BUY_PATTERN.search(content) is None
        assert self.SELL_PATTERN.search(content) is not None

    def test_hold_detected(self):
        content = "# 7740.T (Tamron Co.,Ltd.): HOLD\n"
        assert self.BUY_PATTERN.search(content) is None
        assert self.HOLD_PATTERN.search(content) is not None

    def test_do_not_initiate_detected(self):
        content = "# X.Y (Foo Corp): DO_NOT_INITIATE\n"
        assert self.BUY_PATTERN.search(content) is None
        assert self.DNI_PATTERN.search(content) is not None

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
        assert self.BUY_PATTERN.search(content) is not None

    def test_commas_in_company_name(self):
        """Company names with commas and periods should still match."""
        content = "# 7740.T (Tamron Co.,Ltd.): HOLD\n"
        assert self.VERDICT_PATTERN.search(content) is not None
        assert self.HOLD_PATTERN.search(content) is not None

    def test_parentheses_in_company_name(self):
        """Nested parentheses shouldn't break the match."""
        content = "# 2330.TW (Taiwan Semiconductor (TSMC)): BUY\n"
        assert self.BUY_PATTERN.search(content) is not None

    def test_no_verdict_line(self):
        """File without verdict line should not match."""
        content = "Some random content\nNo verdict here\n"
        assert self.VERDICT_PATTERN.search(content) is None

    def test_buy_must_be_exact_end(self):
        """BUY pattern requires exact match at end of line (no trailing text)."""
        content = "# X.Y (Foo): BUY_SOMETHING\n"
        assert self.BUY_PATTERN.search(content) is None


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

        The shell script checks:
          if ! $FORCE && [[ -f "$OUTFILE" ]] && grep -qE '^# .*\\): ' "$OUTFILE"

        Returns "SKIP" or "PROCESS".

        Note: Previously this shelled out to bash via subprocess.run(), but that
        causes segfaults on macOS/Apple Silicon with Python 3.12 due to fork()
        safety issues with loaded C extensions (grpc, numpy, pandas, etc.).
        """
        if not force and outfile.is_file():
            content = outfile.read_text()
            if re.search(r"^# .*\): ", content, re.MULTILINE):
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
        outfile.write_text("# X.Y (Foo Corp): DO_NOT_INITIATE\n")

        result = self._run_skip_check(outfile, force=False)
        assert result == "SKIP"
