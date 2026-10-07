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
            "ownership",
            "eps_lineage",
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


def test_ownership_replay_uses_basis_owner_and_keeps_unknown_labels_finite(monkeypatch):
    from collections import Counter

    from scripts.replay_contracts import _replay_ownership
    from src.agents.foreign_language_evidence import _CONTROL_BASIS_TERMS

    monkeypatch.setitem(_CONTROL_BASIS_TERMS, "REVIEWED_BASIS", ("reviewed basis",))
    counts = Counter()
    _replay_ownership(
        "Control Status: unexpected private prose\nControl Basis: unexpected private prose\nOwnership Evidence Status: unexpected private prose",
        "Control Status: CONTROLLED\nControl Basis: REVIEWED_BASIS\nOwnership Evidence Status: VERIFIED_URL",
        counts,
    )
    assert counts["after_basis:REVIEWED_BASIS"] == 1
    assert all(
        counts[f"before_{category}:OTHER"] == 1
        for category in ("control", "basis", "evidence")
    )
    assert all("private prose" not in key for key in counts)


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


@pytest.mark.parametrize("pct", [25, 69.72])
@pytest.mark.parametrize(
    "statement,supported",
    [
        ("Holder and its associates hold {pct}% of issued shares.", False),
        ("Holder owns {pct}%. Holder is not an associate of a director.", False),
        ("Holder owns {pct}% and uses the equity method.", True),
        ("Holder owns {pct}%. Holder may use the equity method.", False),
        (
            'Holder owns {pct}%. The report quotes "Holder uses the equity method".',
            False,
        ),
    ],
)
@pytest.mark.parametrize("coverage", ["ABSENT", "NOT_FOUND"])
def test_ownership_replays_without_found_results_and_uses_shared_semantics(
    tmp_path, pct, statement, supported, coverage
):
    from tests.agents.test_foreign_language_evidence import _report

    source = "https://issuer.example/ownership"
    foreign = _report(
        holder=f"Holder ({pct}%)",
        relationship="equity method",
        status="UNKNOWN",
        basis="UNKNOWN",
        source_url=source,
    )
    if coverage == "NOT_FOUND":
        foreign += "\n### --- START LATEST_RESULTS ---\nLATEST_RESULTS_COVERAGE_STATUS: NOT_FOUND\n### --- END LATEST_RESULTS ---"
    path = tmp_path / "ownership_analysis.json"
    path.write_text(
        json.dumps(
            {
                "source_artifacts": {"foreign_language_report": foreign},
                "evidence_records": [
                    {
                        "tool_name": "get_official_document",
                        "content": statement.format(pct=pct),
                        "urls": [source],
                        "execution_status": "SUCCEEDED",
                        "blocked": False,
                        "evidence_status": "EVIDENCE_FOUND",
                    }
                ],
            }
        )
    )
    original = path.read_bytes()
    result = replay_contracts([path])
    expected = (
        "UNKNOWN"
        if supported and pct > 50
        else "NOT_CONTROLLED"
        if supported
        else "CONTROLLED"
        if pct > 50
        else "UNKNOWN"
    )
    assert result["ownership"][f"after_control:{expected}"] == 1
    assert result["ownership"]["replayed"] == 1
    assert result["latest_results"] == {}
    assert result["eps_lineage"] == {"missing_snapshot": 1}
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "status,blocked", [("FAILED", False), ("SUCCEEDED", True), (None, False)]
)
def test_ownership_replay_never_uses_failed_or_blocked_records(
    tmp_path, status, blocked
):
    from tests.agents.test_foreign_language_evidence import _report

    source = "https://issuer.example/ownership"
    path = tmp_path / "ownership_analysis.json"
    path.write_text(
        json.dumps(
            {
                "source_artifacts": {
                    "foreign_language_report": _report(
                        holder="Holder (69.72%)", source_url=source
                    )
                },
                "evidence_records": [
                    {
                        "tool_name": "get_official_document",
                        "content": "Holder holds 69.72% of issued shares.",
                        "urls": [source],
                        "execution_status": status,
                        "blocked": blocked,
                        "evidence_status": "EVIDENCE_FOUND",
                    }
                ],
            }
        )
    )
    result = replay_contracts([path])
    assert result["ownership"]["after_control:UNKNOWN"] == 1
    assert result["ownership"]["after_evidence:REJECTED"] == 1


def _eps_artifact(tmp_path, source, award="1"):
    from src.analysis_snapshot import build_pre_senior_snapshot
    from src.score_lineage import add_validated_derivations
    from tests.test_analysis_snapshot import _fundamentals, _with_structured_metrics

    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics(
            {}, {"earningsGrowth": 0.3, "_earningsGrowth_source": source}
        )
    )
    report = _fundamentals(
        f"GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=N/A; EPS_GROWTH={award}; ROA_ROE_IMPROVING=N/A; GROSS_MARGIN=N/A; GLOBAL_EXPANSION=N/A; R_AND_D_CAPEX_BACKLOG=N/A",
        f"ADJUSTED_GROWTH_SCORE: {'100' if award == '1' else '0'}%",
    )
    # Simulate a retained legacy positive scorecard that used company-profit support.
    before = add_validated_derivations(snapshot, report)
    card = before["scorecards"]["GROWTH"]
    if source in {
        "calculated_from_quarterly",
        "calculated_from_statement_net_income_proxy",
    }:
        card["decision_eligible"] = True
        card["lineage_gaps"] = []
        card["criteria"]["EPS_GROWTH"]["derived_from"] = [
            key
            for key, value in before["claims"].items()
            if value.get("field") == "EARNINGS_GROWTH_FY"
        ]
    path = tmp_path / "eps_analysis.json"
    path.write_text(
        json.dumps(
            {"reports": {"fundamentals_report": report}, "analysis_snapshot": before}
        )
    )
    return path


@pytest.mark.parametrize(
    "source,lost",
    [
        ("calculated_from_quarterly", True),
        ("calculated_from_statement_net_income_proxy", True),
        ("calculated_from_statement_diluted_eps", False),
        ("calculated_from_statement_basic_eps", False),
        ("legacy-provider", False),
    ],
)
def test_eps_replay_measures_dependencies_and_eligibility_without_risk(
    tmp_path, source, lost
):
    path = _eps_artifact(tmp_path, source)
    original = path.read_bytes()
    result = replay_contracts([path])
    assert result["eps_lineage"]["replayed"] == 1
    assert result["eps_lineage"].get("eps_dependencies_changed", 0) == int(lost)
    assert result["eps_lineage"].get("growth_eligibility_lost", 0) == int(lost)
    assert result["eps_lineage"].get("unsupported_positive_eps_after", 0) == int(lost)
    assert result["eps_lineage"]["before_growth:eligible"] == 1
    assert (
        result["eps_lineage"][f"after_growth:{'ineligible' if lost else 'eligible'}"]
        == 1
    )
    assert path.read_bytes() == original
    assert set(result) == {
        "artifacts",
        "ocf",
        "latest_results",
        "latest_results_acquisition",
        "latest_results_claims",
        "pm_trace",
        "ownership",
        "eps_lineage",
    }


@pytest.mark.parametrize(
    "snapshot,expected",
    [
        (None, "missing_snapshot"),
        ({}, "unavailable_contract_or_report"),
        ([], "malformed_snapshot"),
        ("legacy", "malformed_snapshot"),
        ({"contract_status": "VALID", "claims": []}, "malformed_snapshot"),
        (
            {"contract_status": "VALID", "scorecards": {"GROWTH": []}},
            "malformed_snapshot",
        ),
        (
            {
                "contract_status": "VALID",
                "scorecards": {
                    "GROWTH": {
                        "criteria": {
                            "EPS_GROWTH": {
                                "award": "PRIVATE CONTENT",
                                "derived_from": [],
                            }
                        }
                    }
                },
            },
            "malformed_snapshot",
        ),
        (
            {
                "contract_status": "VALID",
                "scorecards": {
                    "GROWTH": {
                        "criteria": {
                            "EPS_GROWTH": {"award": "1", "derived_from": "claim:x"}
                        }
                    }
                },
            },
            "malformed_snapshot",
        ),
    ],
)
def test_legacy_missing_and_malformed_snapshots_do_not_become_contract_defects_or_emit_content(
    tmp_path, snapshot, expected, capsys
):
    path = _artifact(tmp_path)
    data = json.loads(path.read_text())
    data["analysis_snapshot"] = snapshot
    path.write_text(json.dumps(data))
    result = replay_contracts([path])
    assert result["artifacts"] == {"processed": 1, "seen": 1}
    assert result["eps_lineage"] == {expected: 1}
    assert result["latest_results"] == {"SUPPORTED_SECONDARY": 1}
    assert capsys.readouterr().out == ""


def test_ownership_replay_counts_relationship_only_changes_separately(tmp_path):
    from tests.agents.test_foreign_language_evidence import _report

    source = "https://issuer.example/ownership"
    path = tmp_path / "ownership_analysis.json"
    path.write_text(
        json.dumps(
            {
                "source_artifacts": {
                    "foreign_language_report": _report(
                        holder="Holder (25%)",
                        controller="NONE",
                        status="NOT_CONTROLLED",
                        basis="SIGNIFICANT_INFLUENCE_ONLY",
                        relationship="equity method",
                        source_url=source,
                    )
                },
                "evidence_records": [
                    {
                        "tool_name": "get_official_document",
                        "content": "Holder uses the equity method.",
                        "urls": [source],
                        "execution_status": "SUCCEEDED",
                        "evidence_status": "EVIDENCE_FOUND",
                    }
                ],
            }
        )
    )
    counts = replay_contracts([path])["ownership"]
    assert counts["relationship_or_evidence_only_changed"] == 1
    assert "substantive_control_changed" not in counts
