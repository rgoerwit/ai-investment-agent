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
        key: {}
        for key in (
            "artifacts",
            "ocf",
            "latest_results",
            "latest_results_acquisition",
            "latest_results_claims",
            "pm_trace",
        )
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


@pytest.mark.parametrize(
    "change,expected",
    [
        ({}, "fetched_but_unbound"),
        ({"blocked": True}, "attempted_failed_or_blocked"),
        ({"execution_status": "FAILED"}, "attempted_failed_or_blocked"),
        ({"evidence_status": "UNAVAILABLE"}, "attempted_failed_or_blocked"),
        ({"tool_name": "search_foreign_sources"}, "approved_citation_not_fetched"),
        (
            {"tool_name": "search_foreign_sources", "authority_by_url": []},
            "citation_authority_unavailable",
        ),
        ({"requested_urls": None}, "fetched_but_unbound"),
    ],
)
def test_replay_measures_attempts_and_authority_without_retrieval(
    tmp_path, change, expected
):
    row = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/latest_results_october6.json"
        ).read_text()
    )[0]
    report = row["report"].replace(
        "LATEST_RESULTS_EARNINGS_SCOPE: Net income",
        "LATEST_RESULTS_EARNINGS_SCOPE: Operating profit",
    )
    record = {**row["evidence_records"][0], **change}
    path = tmp_path / "case_analysis.json"
    path.write_text(
        json.dumps(
            {
                "source_artifacts": {"foreign_language_report": report},
                "evidence_records": [record],
            }
        )
    )
    result = replay_contracts([path])
    assert result["artifacts"] == {"seen": 1, "processed": 1}
    assert result["latest_results_acquisition"] == {expected: 1}
    assert result["latest_results_claims"] == {"complete_figures": 1}
    assert result["latest_results"] != {"SUPPORTED_PRIMARY": 1}


def test_replay_counts_incomplete_found_and_truncation_separately(tmp_path):
    path = _artifact(tmp_path, evidence_change={"content_truncated": True})
    data = json.loads(path.read_text())
    data["source_artifacts"]["foreign_language_report"] = data["source_artifacts"][
        "foreign_language_report"
    ].replace(
        "LATEST_RESULTS_PRIOR_REVENUE: 402.4", "LATEST_RESULTS_PRIOR_REVENUE: N/A"
    )
    path.write_text(json.dumps(data))
    result = replay_contracts([path])
    assert result["latest_results"] == {"INVALID_COMPARATIVE_CLAIM": 1}
    assert result["latest_results_acquisition"] == {"incomplete_found_claim": 1}
    assert result["latest_results_claims"] == {
        "missing_or_invalid_figures": 1,
        "retained_evidence_truncated": 1,
    }


def test_replay_reports_multiple_document_records_without_retrying(tmp_path):
    row = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/latest_results_october6.json"
        ).read_text()
    )[0]
    path = tmp_path / "case_analysis.json"
    path.write_text(
        json.dumps(
            {
                "source_artifacts": {"foreign_language_report": row["report"]},
                "evidence_records": row["evidence_records"] * 2,
            }
        )
    )
    result = replay_contracts([path])
    assert result["latest_results_acquisition"] == {"supported_primary": 1}
    assert result["latest_results_claims"] == {
        "complete_figures": 1,
        "multiple_document_records": 1,
    }


def test_invalid_citation_is_separate_from_incomplete_found(tmp_path):
    path = _artifact(tmp_path)
    data = json.loads(path.read_text())
    foreign = data["source_artifacts"]["foreign_language_report"]
    foreign = foreign.replace(
        "LATEST_RESULTS_SOURCE_URL: https://quartr.com/events/hd-hyundai-marine-engine-co-ltd-071970-q2-2026_FVY3q2Nx",
        "LATEST_RESULTS_SOURCE_URL: N/A",
    ).replace(
        "LATEST_RESULTS_PRIOR_REVENUE: 402.4", "LATEST_RESULTS_PRIOR_REVENUE: N/A"
    )
    data["source_artifacts"]["foreign_language_report"] = foreign
    path.write_text(json.dumps(data))
    result = replay_contracts([path])
    assert result["latest_results"] == {"SOURCE_URL_INVALID": 1}
    assert result["latest_results_acquisition"] == {"invalid_citation": 1}


@pytest.mark.parametrize("status", ["EVIDENCE_FOUND", "RESULTS_FOUND"])
@pytest.mark.parametrize("field_name", ["urls", "requested_urls"])
def test_acquisition_uses_owning_url_normalization_and_positive_status(
    status, field_name
):
    from scripts.replay_contracts import _acquisition_state

    record = {
        "tool_name": "get_official_document",
        "execution_status": "SUCCEEDED",
        "evidence_status": status,
        field_name: [None, {}, "https://example.com/"],
    }
    state, attempts, _ = _acquisition_state(
        [record], [], "https://example.com", "COMPARATIVE_ROWS_UNBOUND", True
    )
    assert state == "fetched_but_unbound"
    assert attempts == [record]


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"tool_name": []},
        {"tool_name": "get_official_document", "urls": None},
        {
            "content": "text",
            "tool_name": "get_official_document",
            "urls": [{}],
            "execution_status": "SUCCEEDED",
            "evidence_status": [],
        },
    ],
)
def test_malformed_individual_records_do_not_drop_valid_replay(tmp_path, bad):
    path = _artifact(tmp_path)
    data = json.loads(path.read_text())
    data["evidence_records"].append(bad)
    path.write_text(json.dumps(data))
    result = replay_contracts([path])
    assert result["artifacts"] == {"processed": 1, "seen": 1}
    assert result["latest_results"] == {"SUPPORTED_SECONDARY": 1}


def test_registry_authority_without_acquisition_is_approved_not_unavailable():
    from scripts.replay_contracts import _acquisition_state
    from src.agents.message_utils import make_tool_evidence_record

    record = make_tool_evidence_record(
        tool_name="search_foreign_sources",
        content="STATUS: EVIDENCE_FOUND",
        urls=["https://example.com"],
        evidence_status="EVIDENCE_FOUND",
        authority_by_url=[("https://example.com", "PRIMARY_REGISTRY")],
    )
    assert (
        _acquisition_state(
            [], [record], "https://example.com", "SOURCE_NOT_RETAINED", True
        )[0]
        == "approved_citation_not_fetched"
    )


@pytest.mark.parametrize("bad_status", [[], {}, None])
@pytest.mark.parametrize("field_name", ["urls", "requested_urls"])
def test_matching_malformed_status_keeps_valid_artifact_and_never_counts_fetch(
    tmp_path, bad_status, field_name
):
    from scripts.replay_contracts import _acquisition_state

    path = _artifact(tmp_path)
    data = json.loads(path.read_text())
    valid = data["evidence_records"][0]
    record = {
        "tool_name": "get_official_document",
        "execution_status": "SUCCEEDED",
        "evidence_status": bad_status,
        field_name: valid["urls"],
        "content": "ignored",
    }
    data["evidence_records"].append(record)
    path.write_text(json.dumps(data))
    result = replay_contracts([path])
    assert result["artifacts"] == {"processed": 1, "seen": 1}
    assert result["latest_results"] == {"SUPPORTED_SECONDARY": 1}
    assert (
        _acquisition_state([record], [], valid["urls"][0], "SOURCE_NOT_RETAINED", True)[
            0
        ]
        == "attempted_failed_or_blocked"
    )
