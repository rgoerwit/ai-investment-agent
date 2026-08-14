"""Tests for scripts/scan_batch_health.py — the batch anomaly digest."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Add scripts/ to path so we can import scan_batch_health as a module.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import scan_batch_health as sbh  # noqa: E402


def _write(results_dir: Path, ticker: str, date: str, time: str, **kw) -> Path:
    """Write a minimal analysis JSON with the fields the scanner reads."""
    snapshot = {
        "verdict": kw.get("verdict", "BUY"),
        "is_quick_mode": kw.get("is_quick_mode", False),
    }
    run_summary = {
        "publishable": kw.get("publishable", True),
        "llm_failures": kw.get("llm_failures", 0),
        "consultant_verdict": kw.get("consultant_verdict", "CONDITIONAL"),
        "optional_failures": kw.get("optional_failures", []),
        "required_failures": kw.get("required_failures", []),
    }
    payload = {
        "prediction_snapshot": snapshot,
        "run_summary": run_summary,
        "analysis_validity": kw.get("validity", {}),
    }
    path = results_dir / f"{ticker}_{date}_{time}_analysis.json"
    path.write_text(json.dumps(payload))
    return path


class TestNormalizeVerdict:
    def test_spacing_and_underscore_collapse(self):
        assert sbh._normalize_verdict("DO_NOT_INITIATE") == sbh._normalize_verdict(
            "do not initiate"
        )

    def test_none_is_empty(self):
        assert sbh._normalize_verdict(None) == ""


class TestDetectAnomalies:
    def _rec(self, **kw) -> sbh.Record:
        return sbh.Record(
            ticker=kw.get("ticker", "X.T"),
            date="20260712",
            time="120000",
            path=Path("x"),
            verdict=sbh._normalize_verdict(kw.get("verdict", "HOLD")),
            is_quick=kw.get("is_quick", False),
            run_summary=kw.get("run_summary", {"publishable": True}),
            validity=kw.get("validity", {}),
        )

    def test_clean_record_no_anomalies(self):
        rec = self._rec(
            run_summary={
                "publishable": True,
                "llm_failures": 0,
                "consultant_verdict": "CONDITIONAL",
                "optional_failures": [],
            }
        )
        assert sbh.detect_anomalies(rec, prior="HOLD") == []

    def test_not_publishable(self):
        rec = self._rec(run_summary={"publishable": False})
        assert any("not publishable" in a for a in sbh.detect_anomalies(rec, None))

    def test_llm_failures(self):
        rec = self._rec(run_summary={"publishable": True, "llm_failures": 2})
        assert any("llm_failures=2" in a for a in sbh.detect_anomalies(rec, None))

    def test_consultant_error_and_unparsed(self):
        for bad in ("ERROR", "UNPARSED"):
            rec = self._rec(
                run_summary={"publishable": True, "consultant_verdict": bad}
            )
            assert any(bad in a for a in sbh.detect_anomalies(rec, None))

    def test_consultant_skipped_is_not_flagged(self):
        rec = self._rec(
            run_summary={"publishable": True, "consultant_verdict": "SKIPPED"}
        )
        assert sbh.detect_anomalies(rec, None) == []

    def test_optional_failures(self):
        rec = self._rec(
            run_summary={
                "publishable": True,
                "optional_failures": ["consultant_review"],
            }
        )
        assert any("optional failures" in a for a in sbh.detect_anomalies(rec, None))

    def test_required_failure_from_validity(self):
        rec = self._rec(validity={"required_failures": ["final_trade_decision"]})
        assert any("required failures" in a for a in sbh.detect_anomalies(rec, None))

    def test_verdict_flip(self):
        rec = self._rec(verdict="DO NOT INITIATE")
        flags = sbh.detect_anomalies(rec, prior="BUY")
        assert any("verdict flip" in a for a in flags)

    def test_no_flip_when_prior_matches(self):
        rec = self._rec(verdict="HOLD")
        assert sbh.detect_anomalies(rec, prior="HOLD") == []


class TestPriorVerdictSameMode:
    def _rec(self, date, time, verdict, is_quick):
        return sbh.Record(
            ticker="X.T",
            date=date,
            time=time,
            path=Path("x"),
            verdict=sbh._normalize_verdict(verdict),
            is_quick=is_quick,
            run_summary={},
            validity={},
        )

    def test_quick_prior_ignored_for_full_record(self):
        current = self._rec("20260712", "120000", "HOLD", is_quick=False)
        quick_prior = self._rec("20260708", "120000", "BUY", is_quick=True)
        # Only a quick prior exists → no same-mode comparison (Stage-1→Stage-2 is expected).
        assert sbh.prior_verdict(current, [current, quick_prior]) is None

    def test_full_prior_used_for_full_record(self):
        current = self._rec("20260712", "120000", "BUY", is_quick=False)
        full_prior = self._rec("20260302", "120000", "DO NOT INITIATE", is_quick=False)
        assert sbh.prior_verdict(current, [current, full_prior]) == "DO NOT INITIATE"

    def test_unknown_mode_prior_ignored(self):
        current = self._rec("20260712", "120000", "BUY", is_quick=False)
        unknown_prior = self._rec("20260302", "120000", "HOLD", is_quick=None)
        assert sbh.prior_verdict(current, [current, unknown_prior]) is None

    def test_most_recent_same_mode_wins(self):
        current = self._rec("20260712", "120000", "BUY", is_quick=False)
        old = self._rec("20260101", "120000", "SELL", is_quick=False)
        recent = self._rec("20260601", "120000", "HOLD", is_quick=False)
        assert sbh.prior_verdict(current, [current, old, recent]) == "HOLD"


class TestScanEndToEnd:
    def test_scan_flags_real_error_and_full_flip_only(self, tmp_path: Path):
        # Full flip vs an explicit full prior → flagged.
        _write(tmp_path, "AAA.T", "20260302", "120000", verdict="DO_NOT_INITIATE")
        _write(tmp_path, "AAA.T", "20260712", "120000", verdict="BUY")
        # Quick prior + full current → the flip is expected, NOT flagged.
        _write(
            tmp_path, "BBB.T", "20260708", "120000", verdict="BUY", is_quick_mode=True
        )
        _write(tmp_path, "BBB.T", "20260712", "120000", verdict="HOLD")
        # Clean, no prior, no anomaly.
        _write(tmp_path, "CCC.T", "20260712", "120000", verdict="HOLD")
        # A real error anomaly.
        _write(
            tmp_path,
            "DDD.T",
            "20260712",
            "120000",
            verdict="HOLD",
            consultant_verdict="ERROR",
            llm_failures=1,
        )

        result = sbh.scan(tmp_path, "20260712")
        flagged = {rec.ticker: anoms for rec, anoms in result.flagged}
        assert result.total == 4
        assert "AAA.T" in flagged  # genuine full→full flip
        assert "DDD.T" in flagged  # consultant ERROR + llm_failures
        assert "BBB.T" not in flagged  # quick→full transition suppressed
        assert "CCC.T" not in flagged  # clean

    def test_malformed_json_skipped(self, tmp_path: Path):
        (tmp_path / "BAD.T_20260712_120000_analysis.json").write_text("{not json")
        _write(tmp_path, "OK.T", "20260712", "120000", verdict="HOLD")
        result = sbh.scan(tmp_path, "20260712")
        assert result.total == 1

    def test_non_dict_json_skipped(self, tmp_path: Path):
        # A JSON array (valid JSON, wrong shape) must be skipped, not crash.
        (tmp_path / "ARR.T_20260712_120000_analysis.json").write_text("[1, 2, 3]")
        _write(tmp_path, "OK.T", "20260712", "120000", verdict="HOLD")
        result = sbh.scan(tmp_path, "20260712")
        assert result.total == 1


class TestModifiedSinceSelection:
    """--modified-since scopes the batch by file mtime, not filename date.

    This is the cross-day-resume fix: the analysis filename carries the wall-clock
    date, which need not equal the pipeline's logical run-date, so a stage's own
    output is identified by *when it was written*, not by a date string.
    """

    def test_selects_by_mtime_not_filename_date(self, tmp_path: Path):
        # OLD filename-date but recent... vs. an actually-old file. Both carry an
        # anomaly so they'd surface if selected.
        old = _write(
            tmp_path, "OLD.T", "20260707", "120000", verdict="HOLD", llm_failures=1
        )
        new = _write(
            tmp_path, "NEW.T", "20260707", "130000", verdict="HOLD", llm_failures=1
        )
        os.utime(old, (1000, 1000))
        os.utime(new, (5000, 5000))
        result = sbh.scan(tmp_path, modified_since=3000)
        flagged = {rec.ticker for rec, _ in result.flagged}
        assert result.total == 1
        assert "NEW.T" in flagged
        assert "OLD.T" not in flagged

    def test_boundary_is_inclusive(self, tmp_path: Path):
        f = _write(
            tmp_path, "X.T", "20260712", "120000", verdict="HOLD", llm_failures=1
        )
        os.utime(f, (3000, 3000))
        result = sbh.scan(tmp_path, modified_since=3000.0)
        assert result.total == 1

    def test_date_mode_ignores_mtime(self, tmp_path: Path):
        f = _write(tmp_path, "X.T", "20260712", "120000", verdict="HOLD")
        os.utime(f, (1000, 1000))  # ancient mtime must not matter in date mode
        result = sbh.scan(tmp_path, "20260712")
        assert result.total == 1

    def test_flip_comparison_spans_history_in_mtime_mode(self, tmp_path: Path):
        # Prior full analysis is OLD (excluded from the batch) but must still be
        # available as the flip baseline for the recent record.
        prior = _write(
            tmp_path, "AAA.T", "20260302", "120000", verdict="DO_NOT_INITIATE"
        )
        recent = _write(tmp_path, "AAA.T", "20260712", "120000", verdict="BUY")
        os.utime(prior, (1000, 1000))
        os.utime(recent, (5000, 5000))
        result = sbh.scan(tmp_path, modified_since=3000)
        flagged = {rec.ticker: anoms for rec, anoms in result.flagged}
        assert result.total == 1  # only the recent record is in the batch
        assert any("verdict flip" in a for a in flagged.get("AAA.T", []))


class TestFreshOutputCheck:
    def test_publishable_single_artifact_passes(self, tmp_path: Path):
        artifact = _write(tmp_path, "X.T", "20260712", "120000")
        os.utime(artifact, (5000, 5000))

        check = sbh.check_fresh_ticker_output(tmp_path, "X.T", 3000)

        assert check.status == "PUBLISHABLE"
        assert check.publishable is True
        assert check.path == artifact

    def test_nonpublishable_artifact_is_retained_but_incomplete(self, tmp_path: Path):
        artifact = _write(
            tmp_path,
            "1681.HK",
            "20260712",
            "120000",
            publishable=False,
            required_failures=["fundamentals_report"],
        )
        os.utime(artifact, (5000, 5000))

        check = sbh.check_fresh_ticker_output(tmp_path, "1681.HK", 3000)

        assert check.status == "INCOMPLETE"
        assert check.path == artifact
        assert "not publishable" in check.detail

    def test_multiple_fresh_artifacts_are_ambiguous(self, tmp_path: Path):
        first = _write(tmp_path, "X.T", "20260712", "120000")
        second = _write(tmp_path, "X.T", "20260712", "120001")
        os.utime(first, (5000, 5000))
        os.utime(second, (5001, 5001))

        check = sbh.check_fresh_ticker_output(tmp_path, "X.T", 3000)

        assert check.status == "AMBIGUOUS"
        assert check.publishable is False


class TestDegradedButPublishable:
    """A failed optional cross-check seat leaves required artifacts intact, so the
    run is publishable -- and used to report as an unqualified success. The 2026-08-14
    xAI outage ran three full-mode tickers with no cross-check at all and printed
    'OK' each time. ``detect_anomalies`` had already found it; the result was being
    discarded one line before use."""

    def _degraded(self, results_dir: Path) -> Path:
        artifact = _write(
            results_dir,
            "AGS.BR",
            "20260814",
            "174832",
            publishable=True,
            llm_failures=2,
            consultant_verdict="ERROR",
            optional_failures=["auditor_report", "consultant_review"],
        )
        os.utime(artifact, (5000, 5000))
        return artifact

    def test_optional_failures_are_reported_in_detail(self, tmp_path: Path):
        artifact = self._degraded(tmp_path)

        check = sbh.check_fresh_ticker_output(tmp_path, "AGS.BR", 3000)

        assert check.status == "PUBLISHABLE"
        assert check.path == artifact
        assert "optional failures" in check.detail
        assert "consultant_review" in check.detail
        assert "auditor_report" in check.detail

    def test_degradation_does_not_change_status_or_publishability(self, tmp_path: Path):
        # The publication contract is correct as written -- only its reporting was
        # silent. A degraded run must not start failing batches.
        self._degraded(tmp_path)

        check = sbh.check_fresh_ticker_output(tmp_path, "AGS.BR", 3000)

        assert check.status == "PUBLISHABLE"
        assert check.publishable is True

    def test_clean_run_reports_no_detail(self, tmp_path: Path):
        # The regression that matters most: if every run carried detail, the batch
        # would print "(degraded)" for all of them and the signal would be worthless.
        artifact = _write(tmp_path, "X.T", "20260712", "120000")
        os.utime(artifact, (5000, 5000))

        check = sbh.check_fresh_ticker_output(tmp_path, "X.T", 3000)

        assert check.status == "PUBLISHABLE"
        assert check.detail == ""

    def test_incomplete_run_still_reports_only_validity_failures(self, tmp_path: Path):
        artifact = _write(
            tmp_path,
            "1681.HK",
            "20260712",
            "120000",
            publishable=False,
            required_failures=["fundamentals_report"],
            optional_failures=["consultant_review"],
        )
        os.utime(artifact, (5000, 5000))

        check = sbh.check_fresh_ticker_output(tmp_path, "1681.HK", 3000)

        assert check.status == "INCOMPLETE"
        assert "not publishable" in check.detail
        assert "optional failures" not in check.detail


class TestMainArgHandling:
    def test_run_date_and_modified_since_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            sbh.main(["--run-date", "2026-07-07", "--modified-since", "123"])

    def test_modified_since_requires_number(self):
        with pytest.raises(SystemExit):
            sbh.main(["--modified-since", "not-a-number"])

    def test_modified_since_json_output(self, tmp_path: Path, capsys):
        f = _write(
            tmp_path, "X.T", "20260712", "120000", verdict="HOLD", llm_failures=1
        )
        os.utime(f, (5000, 5000))
        rc = sbh.main(
            ["--modified-since", "3000", "--results-dir", str(tmp_path), "--json"]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["modified_since"] == 3000.0
        assert payload["run_date"] is None
        assert payload["total"] == 1

    def test_strict_exits_nonzero_on_anomaly(self, tmp_path: Path, capsys):
        _write(tmp_path, "X.T", "20260712", "120000", verdict="HOLD", llm_failures=1)
        rc = sbh.main(
            ["--run-date", "2026-07-12", "--results-dir", str(tmp_path), "--strict"]
        )
        capsys.readouterr()
        assert rc == 1
