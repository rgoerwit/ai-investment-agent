"""Tests for scripts/scan_batch_health.py — the batch anomaly digest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
