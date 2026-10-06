"""Checked real evidence and malformed artifacts exercise offline replay."""

import json
from pathlib import Path

import pytest

from scripts.replay_contracts import replay_contracts


def _artifact(tmp_path, *, evidence_change=None):
    case = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/latest_results_narrative.json"
        ).read_text()
    )
    evidence = {**case["evidence"], "execution_status": "SUCCEEDED", "blocked": False}
    evidence.update(evidence_change or {})
    data = {
        "source_artifacts": {"foreign_language_report": case["report"]},
        "evidence_records": [evidence],
        "reports": {
            "fundamentals_report": "### --- START DATA_BLOCK ---\nOPERATING_CASH_FLOW: NZ$100K\n### --- END DATA_BLOCK ---"
        },
    }
    path = tmp_path / "case_analysis.json"
    path.write_text(json.dumps(data))
    return path


def test_checked_replay_is_deterministic_deduplicates_paths_and_does_not_write(
    tmp_path,
):
    path = _artifact(tmp_path)
    original = path.read_bytes()
    report = replay_contracts([path, path])
    assert report == replay_contracts([path])
    assert report["artifacts"] == {"processed": 1, "seen": 1}
    assert report["ocf"] == {"present": 1}
    assert report["latest_results"] == {"SUPPORTED_SECONDARY": 1}
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "change",
    [
        {"blocked": True},
        {"execution_status": "FAILED"},
        {"execution_status": None},
        {"evidence_status": "AUTH_ERROR"},
        {"evidence_status": None},
    ],
)
def test_replay_never_grants_support_from_blocked_failed_or_unknown_evidence(
    tmp_path, change
):
    path = _artifact(tmp_path, evidence_change=change)
    result = replay_contracts([path])
    assert result["latest_results"] == {"SOURCE_NOT_RETAINED": 1}


@pytest.mark.parametrize(
    "text", ["{bad", "[]", '{"reports": null}', '{"evidence_records": {}}']
)
def test_replay_reports_malformed_artifacts_without_aborting_or_printing_content(
    tmp_path, text, capsys
):
    bad = tmp_path / "bad_analysis.json"
    bad.write_text(text)
    good = _artifact(tmp_path)
    result = replay_contracts([bad, good, tmp_path / "missing_analysis.json"])
    assert result["artifacts"] == {
        "malformed_or_unreadable": 2,
        "processed": 1,
        "seen": 3,
    }
    assert result["latest_results"] == {"SUPPORTED_SECONDARY": 1}
    assert capsys.readouterr().out == ""


def test_empty_replay_is_explicit_and_stable():
    assert replay_contracts([]) == {
        key: {} for key in ("artifacts", "ocf", "latest_results", "pm_trace")
    }


@pytest.mark.parametrize(
    "verdict,valid_after", [("DO_NOT_INITIATE", True), ("BUY", False)]
)
def test_replay_distinguishes_free_gate_cleanup_from_missing_buy_support(
    tmp_path, verdict, valid_after
):
    path = _artifact(tmp_path)
    data = json.loads(path.read_text())
    text = (Path(__file__).parents[1] / "fixtures/pm_empty_trace.txt").read_text()
    data["final_decision"] = {"decision": text.replace("DO_NOT_INITIATE", verdict)}
    data["red_flags"] = [{"type": "COVERAGE_GAP", "blocks_buy": True}]
    path.write_text(json.dumps(data))
    counts = replay_contracts([path])["pm_trace"]
    assert counts["invalid_before"] == 1
    assert counts["valid_after" if valid_after else "invalid_after"] == 1
