"""Tests for the longitudinal artifact comparison report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import eval_longitudinal_compare as compare  # noqa: E402


def test_extract_row_surfaces_validity_and_data_quality(tmp_path: Path) -> None:
    artifact = tmp_path / "1681.HK_20260802_190854_analysis.json"
    artifact.write_text(
        json.dumps(
            {
                "metadata": {"ticker": "1681.HK"},
                "run_summary": {
                    "publishable": False,
                    "required_failures": ["fundamentals_report"],
                    "consultant_verdict": "CLEAN",
                    "auditor_status": "PARTIAL_DATA",
                },
                "final_decision": {"decision": ""},
                "structured_inputs": {
                    "raw_financial_metrics": {
                        "payload": {
                            "_quality": {"coverage_pct": 70.5},
                            "_quarterly_diagnostics": [
                                {"field": "revenueGrowth_TTM", "status": "unavailable"},
                                {
                                    "field": "earningsGrowth_TTM",
                                    "status": "unavailable",
                                },
                            ],
                            "_source_conflicts": {"earningsGrowth": {}},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    row = compare.extract_row(artifact)

    assert row.publishable is False
    assert row.required_failures == ["fundamentals_report"]
    assert row.data_coverage_pct == 70.5
    assert row.growth_gap_count == 2
    assert row.source_conflict_count == 1

    rendered = compare.render_timeline_markdown("1681.HK", [row])
    assert "INCOMPLETE: fundamentals_report" in rendered
    assert "coverage 70.5%; growth gaps 2; conflicts 1" in rendered
    assert "2026-08-02 19:08:54" in rendered
    assert f"[artifact]({artifact.resolve().as_uri()})" in rendered


def _artifact(tmp_path, name, *, decision_text, snapshot):
    import json

    payload = {
        "final_decision": {"decision": decision_text},
        "prediction_snapshot": snapshot,
        "run_summary": {},
        "red_flags": [],
    }
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


_PM_TEXT = (
    "PORTFOLIO MANAGER VERDICT: BUY\n"
    "### --- START PM_BLOCK ---\n"
    "VERDICT: BUY\nHEALTH_ADJ: 92\nGROWTH_ADJ: 67\nRISK_TALLY: -0.5\nZONE: LOW\n"
    "### --- END PM_BLOCK ---\n"
)


def test_structured_snapshot_takes_precedence(tmp_path):
    """Persisted structured fields outrank reparsed narrative prose."""
    from scripts.eval_longitudinal_compare import extract_row

    p = _artifact(
        tmp_path,
        "AAA_20260802_161755_analysis.json",
        decision_text=_PM_TEXT,
        snapshot={"risk_tally": -0.5, "health_adj": 92, "growth_adj": 67},
    )
    row = extract_row(p)
    assert row.risk_total == -0.5
    assert row.health_adj == 92


def test_negative_tally_recovered_when_snapshot_is_null(tmp_path):
    """Legacy artifacts lost the value; the canonical parser recovers it."""
    from scripts.eval_longitudinal_compare import extract_row

    p = _artifact(
        tmp_path,
        "AAA_20260802_161755_analysis.json",
        decision_text=_PM_TEXT,
        snapshot={"risk_tally": None, "health_adj": None, "growth_adj": None},
    )
    assert extract_row(p).risk_total == -0.5


def test_absent_or_malformed_snapshot_does_not_crash(tmp_path):
    """`snapshot` is rebound later in extract_row; a non-dict must not reach the picker."""
    from scripts.eval_longitudinal_compare import extract_row

    for snapshot in (None, [], "nope", {}):
        p = _artifact(
            tmp_path,
            "AAA_20260802_161755_analysis.json",
            decision_text=_PM_TEXT,
            snapshot=snapshot,
        )
        assert extract_row(p).risk_total == -0.5


# Real lines from scratch/eval_rerun_20260814_173135/run.log, the batch whose
# review plane was 100% dead (xAI 403s) and which this scanner reported clean.
_AUTH_LOG = (
    "2026-08-14 event='llm_call_failed' context='External Consultant' "
    "provider='xai' model='grok-4.6' runnable_class='_ChatModelBinding' attempt=1 "
    "max_attempts=3 failure_kind='auth_error' host='api.x.ai' retryable=False "
    "error_type='PermissionDeniedError'\n"
    "2026-08-14 event='consultant_node_error' ticker='1681.HK' "
    "operation='consultant_node_error' error_type='PermissionDeniedError' "
    "root_cause_type='HTTPStatusError' failure_kind='auth_error' retryable=False "
    "host='api.x.ai' message_preview=\n"
    "2026-08-14 event='auditor_error' ticker='1681.HK' operation='auditor_error' "
    "error_type='PermissionDeniedError' root_cause_type='HTTPStatusError' "
    "failure_kind='auth_error' retryable=False host='api.x.ai' message_preview=\n"
)


class TestRunLogFailureKindScan:
    """The scanner knew three event names, so a provider failure matched none and
    it printed 'No consultant/auditor structural failures detected' over a log with
    14 auth errors. Keying on ``failure_kind`` uses the vocabulary that
    ``classify_failure`` guarantees for every classified failure."""

    def _scan(self, tmp_path: Path, text: str) -> str:
        log = tmp_path / "run.log"
        log.write_text(text)
        return compare.scan_run_log(log)

    def test_provider_auth_failures_are_reported(self, tmp_path: Path) -> None:
        out = self._scan(tmp_path, _AUTH_LOG)
        assert "auth_error" in out
        assert "llm_call_failed" in out

    def test_the_false_all_clear_is_gone(self, tmp_path: Path) -> None:
        # The specific regression: a confidently wrong answer is worse than none.
        out = self._scan(tmp_path, _AUTH_LOG)
        assert "No consultant/auditor structural failures detected" not in out

    def test_vendor_is_attributed_from_provider_or_host(self, tmp_path: Path) -> None:
        # llm_call_failed carries provider=; the node-level events carry only host=.
        out = self._scan(tmp_path, _AUTH_LOG)
        assert "xai" in out
        assert "api.x.ai" in out

    def test_clean_log_still_reports_the_all_clear(self, tmp_path: Path) -> None:
        out = self._scan(
            tmp_path, "2026-08-14 event='analysis_complete' ticker='X.T'\n"
        )
        assert "No consultant/auditor structural failures detected" in out

    def test_dns_failures_are_not_double_counted(self, tmp_path: Path) -> None:
        # dns_resolution keeps its dedicated per-host breakout; counting it in the
        # kind rollup too would report the same line in two sections.
        dns = (
            "2026-08-14 event='tool_call_failed' operation='stocktwits_fetch' "
            "failure_kind='dns_resolution' "
            "message_preview='Cannot connect to host api.stocktwits.com'\n"
        )
        out = self._scan(tmp_path, dns)
        assert "DNS resolution failures, by operation" in out
        assert "`dns_resolution` (" not in out

    def test_unrelated_kinds_are_still_surfaced(self, tmp_path: Path) -> None:
        # Generality is the point: kinds the scanner was never taught must appear.
        out = self._scan(
            tmp_path,
            "2026-08-14 event='llm_call_failed' provider='google' "
            "failure_kind='provider_safety_block' host='generativelanguage.googleapis.com'\n",
        )
        assert "provider_safety_block" in out

    def test_line_without_failure_kind_is_skipped_not_raised(
        self, tmp_path: Path
    ) -> None:
        out = self._scan(
            tmp_path,
            "2026-08-14 event='llm_call_failed' provider='xai' truncated-mid-l\n",
        )
        assert "No consultant/auditor structural failures detected" in out

    def test_legacy_event_patterns_still_report(self, tmp_path: Path) -> None:
        out = self._scan(
            tmp_path,
            "2026-08-14 event='consultant_invalid_structure' ticker='7740.T'\n",
        )
        assert "consultant_invalid_structure" in out
        assert "7740.T" in out
