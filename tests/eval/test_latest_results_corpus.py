"""Frozen, individually adjudicated examples gate authority and consumer behavior."""

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.replay_contracts import replay_contracts
from src.agents import foreign_language_evidence as evidence_module
from src.agents.foreign_language_evidence import (
    _field as field,
)
from src.agents.foreign_language_evidence import (
    normalize_foreign_language_evidence,
    promote_foreign_growth_evidence,
)
from src.agents.message_utils import make_tool_evidence_record
from src.analysis_snapshot import build_analysis_snapshot

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_CORPUS = json.loads((_FIXTURES / "latest_results_corpus.json").read_text())["cases"]


def _materialize(case):
    fixtures = json.loads((_FIXTURES / case["fixture"]).read_text())
    base = copy.deepcopy(
        next(row for row in fixtures if row["case"] == case["fixture_case"])
        if isinstance(fixtures, list)
        else fixtures
    )
    for old, new in case["report_replacements"]:
        assert old in base["report"], case["case"]
        base["report"] = base["report"].replace(old, new)
    if "evidence_records" in base:
        base["evidence"] = (
            base["evidence_records"][0] if base["evidence_records"] else {}
        )
    for old, new in case["content_replacements"]:
        assert old in base["evidence"]["content"], case["case"]
        base["evidence"]["content"] = base["evidence"]["content"].replace(old, new)
    base["evidence"].update(case["record_overrides"])
    base.setdefault("evidence_records", [base["evidence"]])
    return base


def _check(case):
    base = _materialize(case)
    normalized = normalize_foreign_language_evidence(
        base["report"],
        [],
        ticker=base["ticker"],
        additional_records=[
            make_tool_evidence_record(**item)
            for item in base["evidence_records"]
            if item.get("evidence_status")
        ],
    )
    actual = field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY")
    assert actual == case["expected_authority"], case["justification"]
    senior, _ = promote_foreign_growth_evidence("GROWTH_SCORE: 7", normalized)
    assert field(senior, "GROWTH_SCORE") == "7"
    for metric, expected in zip(
        ("REVENUE", "EARNINGS"), case["expected_growth"] or [None, None], strict=True
    ):
        key = f"LATEST_RESULTS_{metric}_GROWTH_YOY"
        assert field(normalized, key) == (expected or "N/A"), case["justification"]
        assert field(senior, key) == (expected or ""), case["justification"]
    return field(normalized, "LATEST_RESULTS_NORMALIZATION_REASON")


@pytest.mark.parametrize("case", _CORPUS, ids=lambda case: case["case"])
def test_adjudicated_authority_and_growth_promotion(case):
    _check(case)


def test_manifest_retains_source_justifications_and_frozen_holdout():
    assert len({case["case"] for case in _CORPUS}) == len(_CORPUS)
    assert {case["split"] for case in _CORPUS} == {
        "development",
        "holdout",
        "regression_control",
    }
    assert all(case["justification"] for case in _CORPUS)
    assert Counter(case["kind"] for case in _CORPUS) == {
        "captured_excerpt": 19,
        "adversarial_mutation": 41,
    }


def test_offline_replay_matches_each_adjudicated_case_and_reports_errors_separately(
    tmp_path,
):
    errors = Counter()
    for case in _CORPUS:
        base = _materialize(case)
        path = tmp_path / "case_analysis.json"
        path.write_text(
            json.dumps(
                {
                    "source_artifacts": {"foreign_language_report": base["report"]},
                    "evidence_records": [
                        {
                            **item,
                            "execution_status": "SUCCEEDED",
                            "blocked": False,
                        }
                        for item in base["evidence_records"]
                    ],
                }
            )
        )
        reasons = replay_contracts([path])["latest_results"]
        assert sum(reasons.values()) == 1
        actual = next(iter(reasons))
        expected = case["expected_authority"]
        supported = actual in {"SUPPORTED_PRIMARY", "SUPPORTED_SECONDARY"}
        if expected == "UNSUPPORTED" and supported:
            errors["false_acceptance"] += 1
        elif expected != "UNSUPPORTED" and not supported:
            errors["false_rejection"] += 1
        elif supported and actual != f"SUPPORTED_{expected}":
            errors["authority_mismatch"] += 1
    assert not errors, dict(errors)


def test_presence_only_defect_is_caught_even_when_all_four_numbers_exist(monkeypatch):
    def presence_only(record, values, decimals):
        flattened = record.content.replace(",", "")
        return all(str(number) in flattened for number in decimals.values())

    monkeypatch.setattr(
        evidence_module, "_latest_results_record_supports", presence_only
    )
    _check(_CORPUS[0])  # Positive control still passes under the defective validator.
    for name in ("wrong_unit", "wrong_scope", "swapped_revenue"):
        case = next(case for case in _CORPUS if case["case"] == name)
        with pytest.raises(AssertionError, match="accounting proof"):
            _check(case)


def test_disabling_supported_reader_is_caught_by_valid_case_gate(monkeypatch):
    monkeypatch.setattr(
        evidence_module, "_two_column_statement_supports", lambda *_: False
    )
    with pytest.raises(AssertionError, match="ordered 2025/2024"):
        _check(_CORPUS[0])
    _check(next(case for case in _CORPUS if case["case"] == "wrong_unit"))


def test_metric_pairs_cannot_be_assembled_across_evidence_records():
    base = _materialize(_CORPUS[0])
    records = []
    for omitted in (
        "Revenue 4 79,068,022 80,650,914",
        "Owners of the Company 4,500,698 3,734,429",
    ):
        item = {
            **base["evidence"],
            "content": base["evidence"]["content"].replace(omitted, ""),
        }
        records.append(make_tool_evidence_record(**item))
    normalized = normalize_foreign_language_evidence(
        base["report"], [], ticker=base["ticker"], additional_records=records
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "UNSUPPORTED"
    assert field(normalized, "LATEST_RESULTS_REVENUE_GROWTH_YOY") == "N/A"


@pytest.mark.parametrize(
    "name",
    [
        "captured_6811.HK",
        "captured_6831.HK",
        "captured_TECK.A.TO",
        "explicit_half_year_secondary_narrative",
        "wrong_unit",
    ],
)
def test_normalized_growth_reaches_canonical_consumers_only_with_primary_authority(
    name,
):
    case = next(case for case in _CORPUS if case["case"] == name)
    base = _materialize(case)
    normalized = normalize_foreign_language_evidence(
        base["report"],
        [],
        ticker=base["ticker"],
        additional_records=[
            make_tool_evidence_record(**item) for item in base["evidence_records"]
        ],
    )
    senior, _ = promote_foreign_growth_evidence("GROWTH_SCORE: 7", normalized)
    snapshot = build_analysis_snapshot(
        {
            "fundamentals_report": "### --- START DATA_BLOCK ---\n"
            + senior
            + "\n### --- END DATA_BLOCK ---"
        },
        [SimpleNamespace(**item, blocked=False) for item in base["evidence_records"]],
        degraded=False,
    )
    claims = {claim["field"]: claim for claim in snapshot["claims"].values()}
    for metric, expected in zip(
        ("REVENUE", "EARNINGS"), case["expected_growth"] or [None, None], strict=True
    ):
        claim = claims[f"LATEST_RESULTS_{metric}_GROWTH_YOY"]
        assert claim["value"] == (expected or "N/A")
        assert claim["decision_eligible"] is (expected is not None)
        assert claim["kind"] == "FACT"
        if expected:
            assert claim["authority"] == "PRIMARY"
            assert claim["period"] == field(base["report"], "LATEST_RESULTS_PERIOD_END")
    assert field(senior, "GROWTH_SCORE") == "7"
    assert not snapshot["conflicts"]


@pytest.mark.parametrize(
    "source_header", ["Annual release", "FY2025", "Results (Q4 2025)"]
)
def test_unknown_source_header_never_proves_period_even_when_claim_labels_match(
    source_header,
):
    base = _materialize(_CORPUS[0])
    record = {
        **base["evidence"],
        "content": base["evidence"]["content"].replace(
            "For the year ended 31 December 2025\n2025 2024",
            source_header + "\n2025 2024",
        ),
    }
    normalized = normalize_foreign_language_evidence(
        base["report"],
        [],
        ticker=base["ticker"],
        additional_records=[make_tool_evidence_record(**record)],
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "UNSUPPORTED"


@pytest.mark.parametrize("symbol", ["US$", "EUR", "RMB"])
def test_cell_display_currency_cannot_contradict_bound_table_currency(symbol):
    base = _materialize(
        next(case for case in _CORPUS if case["case"] == "captured_TECK.A.TO")
    )
    for item in base["evidence_records"]:
        item["content"] = item["content"].replace(
            "Revenue $ 3,943 $ 2,290", f"Revenue {symbol} 3,943 {symbol} 2,290"
        )
    normalized = normalize_foreign_language_evidence(
        base["report"],
        [],
        ticker=base["ticker"],
        additional_records=[
            make_tool_evidence_record(**item) for item in base["evidence_records"]
        ],
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "UNSUPPORTED"


@pytest.mark.parametrize(
    "currency", ["United States dollars", "EUR", "HK$", "Renminbi and USD"]
)
def test_intervening_presentation_currency_cannot_contradict_table(currency):
    base = _materialize(
        next(case for case in _CORPUS if case["case"] == "captured_6831.HK")
    )
    for item in base["evidence_records"]:
        item["content"] = item["content"].replace(
            "(Expressed in Renminbi)", f"(Expressed in {currency})"
        )
    normalized = normalize_foreign_language_evidence(
        base["report"],
        [],
        ticker=base["ticker"],
        additional_records=[
            make_tool_evidence_record(**item) for item in base["evidence_records"]
        ],
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "UNSUPPORTED"


@pytest.mark.parametrize(
    "label",
    [
        "Results (For the year ended December 31, 2024)",
        "Results (Six months ended June 30, 2025)",
        "Results (H1 2025)",
    ],
)
def test_readable_contradiction_inside_unknown_claim_wording_still_rejects(label):
    base = _materialize(_CORPUS[0])
    report = base["report"].replace(
        "LATEST_RESULTS_PERIOD: For the year ended 31 December 2025",
        "LATEST_RESULTS_PERIOD: " + label,
    )
    normalized = normalize_foreign_language_evidence(
        report,
        [],
        ticker=base["ticker"],
        additional_records=[make_tool_evidence_record(**base["evidence"])],
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "UNSUPPORTED"
    assert (
        field(normalized, "LATEST_RESULTS_NORMALIZATION_REASON")
        == "PERIOD_LABEL_MISMATCH"
    )


def test_dollar_display_symbol_cannot_enter_a_yuan_statement():
    base = _materialize(_CORPUS[0])
    record = {
        **base["evidence"],
        "content": base["evidence"]["content"].replace(
            "Revenue 4 79,068,022 80,650,914", "Revenue 4 $ 79,068,022 $ 80,650,914"
        ),
    }
    normalized = normalize_foreign_language_evidence(
        base["report"],
        [],
        ticker=base["ticker"],
        additional_records=[make_tool_evidence_record(**record)],
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "UNSUPPORTED"


@pytest.mark.parametrize(
    "header", ["Note 2025 2024", "Notes 2025 2024", "(RMB in thousands) 2025 2024"]
)
@pytest.mark.parametrize("before", ["Revenue 4", "Profit attributable to:"])
def test_reader_cannot_cross_any_recognized_column_header(header, before):
    base = _materialize(_CORPUS[0])
    record = {
        **base["evidence"],
        "content": base["evidence"]["content"].replace(before, header + "\n" + before),
    }
    normalized = normalize_foreign_language_evidence(
        base["report"],
        [],
        ticker=base["ticker"],
        additional_records=[make_tool_evidence_record(**record)],
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "UNSUPPORTED"


@pytest.mark.parametrize(
    "label",
    [
        "H0 2025",
        "H3 2025",
        "Results (H4 2025)",
        "Q0 2025",
        "Q5 2025",
        "Q01 2025",
        pytest.param("H" + "9" * 4301 + " 2025", id="oversized_half_token"),
    ],
)
def test_malformed_calendar_tokens_fail_closed_without_recursion(label):
    assert evidence_module._header_period_matches(label, "2025-12-31", 12) is False
    base = _materialize(_CORPUS[0])
    report = base["report"].replace(
        "LATEST_RESULTS_PERIOD: For the year ended 31 December 2025",
        "LATEST_RESULTS_PERIOD: " + label,
    )
    normalized = normalize_foreign_language_evidence(
        report,
        [],
        ticker=base["ticker"],
        additional_records=[make_tool_evidence_record(**base["evidence"])],
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "UNSUPPORTED"
    assert (
        field(normalized, "LATEST_RESULTS_NORMALIZATION_REASON")
        == "PERIOD_LABEL_MISMATCH"
    )
