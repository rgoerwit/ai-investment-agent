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
from src.agents.message_utils import (
    evidence_record_to_tool_evidence,
    make_tool_evidence_record,
)
from src.analysis_snapshot import build_analysis_snapshot

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_CORPUS = json.loads((_FIXTURES / "latest_results_corpus.json").read_text())["cases"]


def _record(item):
    return (
        evidence_record_to_tool_evidence(item)
        if "content_sha256" in item
        else make_tool_evidence_record(**item)
    )


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
            _record(item)
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
    reason = field(normalized, "LATEST_RESULTS_NORMALIZATION_REASON")
    if "expected_reason" in case:
        assert reason == case["expected_reason"], case["justification"]
    return reason


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
        "captured_excerpt": 34,
        "adversarial_mutation": 65,
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
                        {"execution_status": "SUCCEEDED", "blocked": False, **item}
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
        "captured_GFT.DE",
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
        additional_records=[_record(item) for item in base["evidence_records"]],
    )
    senior, _ = promote_foreign_growth_evidence("GROWTH_SCORE: 7", normalized)
    snapshot = build_analysis_snapshot(
        {
            "fundamentals_report": "### --- START DATA_BLOCK ---\n"
            + senior
            + "\n### --- END DATA_BLOCK ---"
        },
        [
            SimpleNamespace(**{"blocked": False, **item})
            for item in base["evidence_records"]
        ],
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


def test_primary_url_in_same_record_cannot_promote_secondary_cited_url():
    base = _materialize(_CORPUS[0])
    primary = base["evidence"]["urls"][0]
    secondary = "https://secondary.example/results"
    record = make_tool_evidence_record(
        **{**base["evidence"], "urls": [primary, secondary]}
    )
    assert record.authority == "PRIMARY_REGISTRY"
    report = base["report"].replace(primary, secondary)
    normalized = normalize_foreign_language_evidence(
        report, [], ticker=base["ticker"], additional_records=[record]
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "SECONDARY"
    assert field(normalized, "LATEST_RESULTS_EARNINGS_GROWTH_YOY") == "N/A"


@pytest.mark.parametrize(
    "label,end,months,expected",
    [
        ("Half Year ended June 30, 2026 (H1 2026)", "2026-06-30", 6, True),
        ("Half-year ended 30 June 2026", "2026-06-30", 6, True),
        ("Half-year ended 30 Jun 2026", "2026-06-30", 6, True),
        ("Three Months Ended Jun 30, 2026", "2026-06-30", 3, True),
        ("Year ended 31 Dec 2025", "2025-12-31", 12, True),
        ("Half Year ended June 30, 2026 (H2 2026)", "2026-06-30", 6, False),
        ("Half-year ended 30 June 2026", "2026-06-30", 12, False),
        ("Six months ended 30 June 2026", "2026-06-30", 3, False),
        ("Three Months Ended Feb 30, 2026", "2026-02-28", 3, False),
        ("Q2 FY2026", "2026-06-30", 3, None),
        ("2027年3月期 第1四半期", "2026-06-30", 3, True),
        ("２０２７年３月期 第１四半期", "2026-06-30", 3, True),
        ("2027年3月期 第1四半期", "2027-03-31", 3, False),
        ("2027年3月期 第1四半期", "2026-06-30", 6, False),
        ("2027年3月期 第2四半期", "2026-09-30", 3, None),
        ("3月期 第1四半期", "2026-06-30", 3, None),
    ],
)
def test_period_calendar_proof_and_contradictions(label, end, months, expected):
    assert evidence_module._header_period_matches(label, end, months) is expected


@pytest.mark.parametrize(
    "source", ["Jun 30, 2026", "30 Jun 2026", "２０２７年３月期 第１四半期"]
)
def test_retained_abbreviated_dates_and_fiscal_q1_supply_calendar_proof(source):
    assert evidence_module._source_has_period_end(source, "2026-06-30")
    assert not evidence_module._source_has_period_end(source, "2025-06-30")


@pytest.mark.parametrize(
    "currency,unit,source",
    [
        ("EUR", "thousands", "€’000"),
        ("EUR", "€'000", "EUR thousands"),
        ("JPY", "millions", "百万円"),
        ("JPY", "百万円", "JPY millions"),
        ("USD", "thousands", "KUSD"),
        ("USD", "KUSD", "USD thousands"),
        ("IDR", "Million Rupiah", "millions of Rupiah"),
    ],
)
def test_reporting_unit_equivalence_stays_currency_and_scale_bound(
    currency, unit, source
):
    from src.monetary import reporting_unit_scale

    values = {
        "LATEST_RESULTS_CURRENCY": currency,
        "LATEST_RESULTS_REPORTING_UNIT": unit,
    }
    assert evidence_module._heading_has_currency_and_unit(source, values)
    assert evidence_module._compatible_unit_only_header(source, values)
    assert evidence_module._statement_boundary(source)
    assert reporting_unit_scale(unit) == reporting_unit_scale(source)
    for unrelated in (source + " H1 2026", source + "\nNet income 100 90"):
        assert not evidence_module._compatible_unit_only_header(unrelated, values)
    assert not evidence_module._heading_has_currency_and_unit(
        source + " CHF billions", values
    )


@pytest.mark.parametrize("wording", ["Half-year ended", "Half Year ended"])
def test_complete_half_year_source_header_reaches_primary_without_annual_submatch(
    wording,
):
    base = _materialize(
        next(c for c in _CORPUS if c["case"] == "simple_interim_statement")
    )
    report = base["report"].replace("Six months ended", wording)
    item = base["evidence"]
    content = item["content"].replace("Six months ended", wording)
    normalized = normalize_foreign_language_evidence(
        report,
        [],
        ticker=base["ticker"],
        additional_records=[make_tool_evidence_record(**{**item, "content": content})],
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "PRIMARY"
    assert field(normalized, "LATEST_RESULTS_REVENUE_GROWTH_YOY") == "18.0%"


def test_model_fiscal_labels_cannot_prove_unrelated_source_table():
    base = _materialize(_CORPUS[0])
    report = (
        base["report"]
        .replace("For the year ended 31 December 2025", "2026年9月期 第1四半期")
        .replace("For the year ended 31 December 2024", "2025年9月期 第1四半期")
        .replace("LATEST_RESULTS_PERIOD_MONTHS: 12", "LATEST_RESULTS_PERIOD_MONTHS: 3")
    )
    content = base["evidence"]["content"].replace(
        "For the year ended 31 December 2025", "Unknown reporting period"
    )
    content += "\n2026年9月期 第1四半期\nUnrelated source label"
    normalized = normalize_foreign_language_evidence(
        report,
        [],
        ticker=base["ticker"],
        additional_records=[
            make_tool_evidence_record(**{**base["evidence"], "content": content})
        ],
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "UNSUPPORTED"


def test_gft_net_income_growth_and_advisory_awards_do_not_lift_decision_score():
    from src.agents.verdict_policy import assess_growth_gate
    from src.analysis_snapshot import add_validated_derivations
    from src.decision_inputs import DecisionInputs

    base = _materialize(next(c for c in _CORPUS if c["case"] == "captured_GFT.DE"))
    normalized = normalize_foreign_language_evidence(
        base["report"],
        [],
        ticker=base["ticker"],
        additional_records=[_record(r) for r in base["evidence_records"]],
    )
    senior, _ = promote_foreign_growth_evidence(base["senior_excerpt"], normalized)
    # Reconstruct the pre-projection arithmetic from the retained 2/5 awards.
    # These other consumer facts reproduce saved aggregator snapshot values;
    # they do not claim to be part of the issuer excerpt.
    senior = senior.replace(
        "ADJUSTED_GROWTH_SCORE: 0.0%", "ADJUSTED_GROWTH_SCORE: 40.0%"
    )
    senior += "\nGROSS_MARGIN_PERCENT: 15.0%\nROA_YOY_CHANGE_PERCENT: -27.8%\nROE_YOY_CHANGE_PERCENT: -27.0%"
    report = "### --- START DATA_BLOCK ---\n" + senior + "\n### --- END DATA_BLOCK ---"
    snapshot = build_analysis_snapshot(
        {"fundamentals_report": report},
        [SimpleNamespace(**r) for r in base["evidence_records"]],
        degraded=False,
    )
    snapshot = add_validated_derivations(snapshot, report)
    card = snapshot["scorecards"]["GROWTH"]
    claims = {c["field"]: (cid, c) for cid, c in snapshot["claims"].items()}
    assert card["criteria"]["REVENUE_GROWTH"]["award"] == "0"
    assert card["criteria"]["REVENUE_GROWTH"]["derived_from"] == [
        claims["LATEST_RESULTS_REVENUE_GROWTH_YOY"][0]
    ]
    assert claims["LATEST_RESULTS_REVENUE_GROWTH_YOY"][1]["value"] == "4.8%"
    assert claims["LATEST_RESULTS_EARNINGS_GROWTH_YOY"][1]["value"] == "26.7%"
    assert card["criteria"]["EPS_GROWTH"] == {
        "award": "N/A",
        "max_points": 1.0,
        "derived_from": [],
    }
    assert card["advisory_only_awards"] == ["GLOBAL_EXPANSION", "R_AND_D_CAPEX_BACKLOG"]
    assert card["percentage"] == 0.0
    assert card["advisory_percentage"] == 40.0
    inputs = DecisionInputs.from_metrics_and_snapshot(
        {
            "adjusted_growth_score": 40.0,
            "adjusted_health_score": 45.5,
            "pe_ratio": 18.74,
            "revenue_cagr_3y": 6.8,
        },
        "Information Technology",
        snapshot,
    )
    assert inputs.growth_decision_pct == 0.0
    # Supply the captured health result to isolate the growth gate from unrelated
    # health-score lineage; its independently low value excludes an exception.
    from dataclasses import replace

    gate = assess_growth_gate(
        replace(inputs, health_decision_pct=45.5, health_score_reliable=True)
    )
    assert gate.hard_fail
    assert gate.exception is None


@pytest.mark.parametrize("prior", ["0", "-13.5"])
def test_primary_loss_or_zero_baseline_never_creates_earnings_growth(prior):
    import hashlib

    base = _materialize(next(c for c in _CORPUS if c["case"] == "captured_GFT.DE"))
    report = base["report"].replace(
        "LATEST_RESULTS_PRIOR_EARNINGS: 13.5", f"LATEST_RESULTS_PRIOR_EARNINGS: {prior}"
    )
    item = base["evidence_records"][0]
    content = item["content"].replace(
        "Net income\n17.1\n13.5", f"Net income\n17.1\n{prior}"
    )
    item = {
        **item,
        "content": content,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    normalized = normalize_foreign_language_evidence(
        report, [], ticker=base["ticker"], additional_records=[_record(item)]
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "PRIMARY"
    assert field(normalized, "LATEST_RESULTS_EARNINGS_GROWTH_YOY") == "N/A"
    senior, _ = promote_foreign_growth_evidence(
        "GROWTH_SCORE_BREAKDOWN: EPS_GROWTH=N/A", normalized
    )
    assert field(senior, "GROWTH_SCORE_BREAKDOWN") == "EPS_GROWTH=N/A"


@pytest.mark.parametrize(
    "currency,source,expected",
    [
        ("GBP", "GBP millions", True),
        ("GBp", "GBp millions", True),
        ("GBP", "GBp millions", False),
        ("GBp", "GBP millions", False),
        ("GBX", "GBX millions", True),
    ],
)
def test_reporting_units_preserve_minor_currency_identity(currency, source, expected):
    values = {
        "LATEST_RESULTS_CURRENCY": currency,
        "LATEST_RESULTS_REPORTING_UNIT": "millions",
    }
    assert evidence_module._heading_has_currency_and_unit(source, values) is expected


@pytest.mark.parametrize(
    "token", ["EUR'000000", "EUR'000x", "USDmillionth", "wrongprefixKUSD"]
)
def test_unsupported_reporting_declarations_do_not_match_valid_prefix(token):
    from src.monetary import reporting_unit_declarations

    assert reporting_unit_declarations(token) == ()


@pytest.mark.parametrize("unit", ["USD thousands", "KUSD", "百万円"])
@pytest.mark.parametrize("heading", ["EUR thousands", "EUR\nReporting unit: thousands"])
def test_reporting_unit_currency_cannot_contradict_currency_field(unit, heading):
    values = {"LATEST_RESULTS_CURRENCY": "EUR", "LATEST_RESULTS_REPORTING_UNIT": unit}
    assert not evidence_module._heading_has_currency_and_unit(heading, values)


@pytest.mark.parametrize(
    "claim_unit,source_unit,authority",
    [("Billions", "billion", "SECONDARY"), ("million", "billion", "UNSUPPORTED")],
)
def test_narrative_unit_aliases_preserve_exact_currency_scale_and_clause(
    claim_unit, source_unit, authority
):
    base = _materialize(
        next(
            c for c in _CORPUS if c["case"] == "explicit_half_year_secondary_narrative"
        )
    )
    report = base["report"].replace(
        "LATEST_RESULTS_REPORTING_UNIT: Billion",
        f"LATEST_RESULTS_REPORTING_UNIT: {claim_unit}",
    )
    item = {
        **base["evidence"],
        "content": base["evidence"]["content"].replace("billion", source_unit),
    }
    normalized = normalize_foreign_language_evidence(
        report, [], ticker=base["ticker"], additional_records=[_record(item)]
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == authority
    assert field(normalized, "LATEST_RESULTS_EARNINGS_GROWTH_YOY") == "N/A"


@pytest.mark.parametrize(
    "header",
    [
        "2027年3月期 第1四半期",
        "２０２７年３月期 第１四半期",
        "2027年3月期 第1四半期 (2026-07-04)",
        "2027年3月期 第1四半期 (53-week calendar)",
    ],
)
def test_retained_fiscal_q1_binding_and_conflicting_calendar_headers(header):
    import hashlib

    base = _materialize(next(c for c in _CORPUS if c["case"] == "captured_GFT.DE"))
    report = (
        base["report"]
        .replace("H1 2026", "2027年3月期 第1四半期")
        .replace("H1 2025", "2026年3月期 第1四半期")
        .replace("LATEST_RESULTS_PERIOD_MONTHS: 6", "LATEST_RESULTS_PERIOD_MONTHS: 3")
    )
    item = {
        **base["evidence"],
        "content": base["evidence"]["content"]
        .replace("H1 2026", header)
        .replace("H1 2025", "2026年3月期 第1四半期"),
    }
    item["content_sha256"] = hashlib.sha256(item["content"].encode()).hexdigest()
    normalized = normalize_foreign_language_evidence(
        report, [], ticker=base["ticker"], additional_records=[_record(item)]
    )
    expected = "UNSUPPORTED" if "(" in header else "PRIMARY"
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == expected


@pytest.mark.parametrize("currency", ["CNY", "CHF", "EUR"])
def test_implicit_jpy_scale_cannot_be_cast_to_foreign_currency(currency):
    values = {
        "LATEST_RESULTS_CURRENCY": currency,
        "LATEST_RESULTS_REPORTING_UNIT": "millions",
    }
    source = f"{currency} 百万円"
    assert not evidence_module._heading_has_currency_and_unit(source, values)
    assert not evidence_module._compatible_unit_only_header(source, values)
    assert evidence_module._statement_boundary(source)


@pytest.mark.parametrize(
    "source_unit,authority", [("million", "PRIMARY"), ("百万円", "UNSUPPORTED")]
)
def test_primary_narrative_implicit_unit_currency_cannot_contradict_explicit_prefix(
    source_unit, authority
):
    base = _materialize(
        next(
            c for c in _CORPUS if c["case"] == "explicit_half_year_secondary_narrative"
        )
    )
    report = (
        base["report"]
        .replace("LATEST_RESULTS_CURRENCY: KRW", "LATEST_RESULTS_CURRENCY: CNY")
        .replace(
            "LATEST_RESULTS_REPORTING_UNIT: Billion",
            "LATEST_RESULTS_REPORTING_UNIT: million",
        )
    )
    item = {
        **base["evidence"],
        "tool_name": "get_official_document",
        "evidence_status": "EVIDENCE_FOUND",
        "content": base["evidence"]["content"]
        .replace("KRW", "CNY")
        .replace("billion", source_unit),
        "authority_by_url": [(base["evidence"]["urls"][0], "PRIMARY_ISSUER")],
    }
    normalized = normalize_foreign_language_evidence(
        report, [], ticker=base["ticker"], additional_records=[_record(item)]
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == authority
    if authority == "UNSUPPORTED":
        assert field(normalized, "LATEST_RESULTS_EARNINGS_GROWTH_YOY") == "N/A"


@pytest.mark.parametrize(
    "token,currency,scale",
    [
        ("百万円", "JPY", 1e6),
        ("千円", "JPY", 1e3),
        ("億円", "JPY", 1e8),
        ("人民幣千元", "CNY", 1e3),
        ("백만원", "KRW", 1e6),
    ],
)
def test_currency_bound_reporting_units_agree_across_consumers(token, currency, scale):
    from src.monetary import reporting_unit_declarations, reporting_unit_scale

    values = {
        "LATEST_RESULTS_CURRENCY": currency,
        "LATEST_RESULTS_REPORTING_UNIT": token,
    }
    assert reporting_unit_scale(token) == scale
    assert reporting_unit_declarations(token) == ((currency, scale),)
    assert evidence_module._heading_has_currency_and_unit(token, values)
    assert evidence_module._compatible_unit_only_header(token, values)
    assert evidence_module._statement_boundary(token)
    for other in {"CHF", "CNY", "JPY", "KRW"} - {currency}:
        assert not evidence_module._heading_has_currency_and_unit(
            f"{other} {token}", values
        )
        assert not evidence_module._compatible_unit_only_header(
            f"{other} {token}", values
        )


@pytest.mark.parametrize("currency", ["CNY", "TWD"])
def test_plain_chinese_thousands_needs_source_currency_context(currency):
    from src.monetary import reporting_unit_declarations, reporting_unit_scale

    values = {
        "LATEST_RESULTS_CURRENCY": currency,
        "LATEST_RESULTS_REPORTING_UNIT": "千元",
    }
    assert reporting_unit_scale("千元") == 1e3
    assert reporting_unit_declarations("千元") == ()
    assert not evidence_module._heading_has_currency_and_unit("千元", values)
    assert evidence_module._heading_has_currency_and_unit(f"{currency} 千元", values)
    assert evidence_module._statement_boundary("千元")
    assert not evidence_module._heading_has_currency_and_unit("JPY 千元", values)


@pytest.mark.parametrize(
    "text",
    [
        "Don't",
        "don't",
        "T cells",
        "M investors",
        "a K note",
        "B's",
        "2026 T",
        "USD Don't",
    ],
)
def test_stray_scale_letters_never_bind_reporting_metadata(text):
    from src.monetary import reporting_unit_declarations, reporting_unit_scale

    assert reporting_unit_scale(text) is None
    assert reporting_unit_declarations(text) == ()


@pytest.mark.parametrize(
    "unit,scale", [("K", 1e3), ("M", 1e6), ("B", 1e9), ("T", 1e12)]
)
def test_complete_structured_scale_letters_remain_valid(unit, scale):
    from src.monetary import reporting_unit_scale

    assert reporting_unit_scale(unit) == scale
    values = {"LATEST_RESULTS_CURRENCY": "EUR", "LATEST_RESULTS_REPORTING_UNIT": unit}
    assert evidence_module._heading_has_currency_and_unit(f"EUR {unit}", values)


@pytest.mark.parametrize(
    "currency,source_currency", [("€", "€"), ("€", "EUR"), ("EUR", "€")]
)
def test_symbol_currency_reaches_primary_narrative_through_public_normalizer(
    currency, source_currency
):
    import hashlib

    base = _materialize(
        next(
            c for c in _CORPUS if c["case"] == "explicit_half_year_secondary_narrative"
        )
    )
    report = base["report"].replace(
        "LATEST_RESULTS_CURRENCY: KRW", f"LATEST_RESULTS_CURRENCY: {currency}"
    )
    item = {
        **base["evidence"],
        "tool_name": "get_official_document",
        "evidence_status": "EVIDENCE_FOUND",
        "execution_status": "SUCCEEDED",
        "content": base["evidence"]["content"].replace("KRW", source_currency),
        "authority_by_url": [(base["evidence"]["urls"][0], "PRIMARY_ISSUER")],
    }
    item["content_sha256"] = hashlib.sha256(item["content"].encode()).hexdigest()
    normalized = normalize_foreign_language_evidence(
        report, [], ticker=base["ticker"], additional_records=[_record(item)]
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "PRIMARY"
    assert (
        field(normalized, "LATEST_RESULTS_NORMALIZATION_REASON") == "SUPPORTED_PRIMARY"
    )


@pytest.mark.parametrize(
    "label,end,months,expected",
    [
        ("2027年3月期 第1四半期連結累計期間", "2026-06-30", 3, True),
        ("2027年3月期 第2四半期連結累計期間", "2026-09-30", 6, None),
        ("2027年3月期 第2四半期", "2026-09-30", 3, None),
        ("2027年13月期 第1四半期", "2026-06-30", 3, False),
        ("Three months ended 30 Jun 2026", "2026-06-30", 3, True),
        ("Year ended Jun 30, 2026 and 2025", "2025-06-30", 12, True),
        ("Six months ended Jun 31, 2026", "2026-06-30", 6, False),
    ],
)
def test_consolidated_fiscal_and_date_parser_preserves_calendar_limits(
    label, end, months, expected
):
    assert evidence_module._header_period_matches(label, end, months) is expected


@pytest.mark.parametrize(
    "symbol,currency", evidence_module.CURRENCY_SYMBOL_TO_CODE.items()
)
@pytest.mark.parametrize("reader", ["table", "narrative"])
@pytest.mark.parametrize("claim_symbol", [True, False])
def test_registered_currency_symbols_match_iso_claims_through_both_readers(
    symbol, currency, reader, claim_symbol
):
    import hashlib

    case_name = (
        "captured_GFT.DE"
        if reader == "table"
        else "explicit_half_year_secondary_narrative"
    )
    original_currency = "EUR" if reader == "table" else "KRW"
    base = _materialize(next(c for c in _CORPUS if c["case"] == case_name))
    report = base["report"].replace(
        f"LATEST_RESULTS_CURRENCY: {original_currency}",
        f"LATEST_RESULTS_CURRENCY: {symbol if claim_symbol else currency}",
    )
    # Keep unit-field identity consistent when the table fixture embeds EUR there.
    report = report.replace(
        "LATEST_RESULTS_REPORTING_UNIT: EUR million",
        f"LATEST_RESULTS_REPORTING_UNIT: {currency} million",
    )
    item = {
        **base["evidence"],
        "tool_name": "get_official_document",
        "execution_status": "SUCCEEDED",
        "evidence_status": "EVIDENCE_FOUND",
        "content": base["evidence"]["content"].replace(original_currency, symbol),
        "authority_by_url": [(base["evidence"]["urls"][0], "PRIMARY_ISSUER")],
    }
    item["content_sha256"] = hashlib.sha256(item["content"].encode()).hexdigest()
    normalized = normalize_foreign_language_evidence(
        report, [], ticker=base["ticker"], additional_records=[_record(item)]
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "PRIMARY"
    assert (
        field(normalized, "LATEST_RESULTS_NORMALIZATION_REASON") == "SUPPORTED_PRIMARY"
    )


@pytest.mark.parametrize("currency", ["GBP", "GBp", "GBX"])
@pytest.mark.parametrize("declared", ["GBP", "GBp", "GBX"])
def test_claim_reporting_declarations_preserve_minor_identity_before_casefold(
    currency, declared
):
    values = {
        "LATEST_RESULTS_CURRENCY": currency,
        "LATEST_RESULTS_REPORTING_UNIT": f"{currency} millions",
    }
    expected = currency == declared
    assert (
        evidence_module._heading_has_currency_and_unit(f"{declared} millions", values)
        is expected
    )
    assert (
        evidence_module._compatible_unit_only_header(f"{declared} millions", values)
        is expected
    )
    assert evidence_module._statement_boundary(f"{declared} millions")


@pytest.mark.parametrize("currency", ["GBP", "GBp", "GBX"])
@pytest.mark.parametrize("source_currency", ["GBP", "GBp", "GBX"])
def test_narrative_explicit_currency_retains_minor_unit_contradictions(
    currency, source_currency
):
    import hashlib

    base = _materialize(
        next(
            c for c in _CORPUS if c["case"] == "explicit_half_year_secondary_narrative"
        )
    )
    report = (
        base["report"]
        .replace("LATEST_RESULTS_CURRENCY: KRW", f"LATEST_RESULTS_CURRENCY: {currency}")
        .replace(
            "LATEST_RESULTS_REPORTING_UNIT: Billion",
            f"LATEST_RESULTS_REPORTING_UNIT: {currency} billion",
        )
    )
    item = {
        **base["evidence"],
        "tool_name": "get_official_document",
        "execution_status": "SUCCEEDED",
        "evidence_status": "EVIDENCE_FOUND",
        "content": base["evidence"]["content"].replace("KRW", source_currency),
        "authority_by_url": [(base["evidence"]["urls"][0], "PRIMARY_ISSUER")],
    }
    item["content_sha256"] = hashlib.sha256(item["content"].encode()).hexdigest()
    normalized = normalize_foreign_language_evidence(
        report, [], ticker=base["ticker"], additional_records=[_record(item)]
    )
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == (
        "PRIMARY" if currency == source_currency else "UNSUPPORTED"
    )
    if currency != source_currency:
        assert field(normalized, "LATEST_RESULTS_EARNINGS_GROWTH_YOY") == "N/A"


_OCTOBER6 = next(
    row
    for row in json.loads((_FIXTURES / "latest_results_october6.json").read_text())
    if row["case"] == "1112_HK_captured_flattened_results"
)


def _october6_normalize(report, records):
    return normalize_foreign_language_evidence(
        report, [], ticker="1112.HK", additional_records=[_record(r) for r in records]
    )


def test_captured_october6_flattened_statement_and_promoted_consumers():
    normalized = _october6_normalize(_OCTOBER6["report"], _OCTOBER6["evidence_records"])
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "PRIMARY"
    assert (
        field(normalized, "LATEST_RESULTS_NORMALIZATION_REASON") == "SUPPORTED_PRIMARY"
    )
    senior, _ = promote_foreign_growth_evidence("", normalized)
    for metric, growth in zip(
        ("REVENUE", "EARNINGS"), _OCTOBER6["expected_growth"], strict=True
    ):
        assert field(normalized, f"LATEST_RESULTS_{metric}_GROWTH_YOY") == growth
        assert field(senior, f"LATEST_RESULTS_{metric}_GROWTH_YOY") == growth


@pytest.mark.parametrize(
    "case,accepted",
    [
        ("reversed_columns", True),
        ("no_change", True),
        ("footnote", True),
        ("wrong_currency", False),
        ("mixed_currency", False),
        ("wrong_scale", False),
        ("mixed_scale", False),
        ("reversed_values", False),
        ("missing_cell", False),
        ("duplicate_year", False),
        ("wrong_year", False),
        ("wrong_months", False),
        ("duplicate_revenue", False),
        ("duplicate_earnings", False),
        ("undated_header", False),
        ("scope_attributable", False),
        ("scope_adjusted", False),
        ("scope_arbitrary", False),
        ("adjusted_row", False),
        ("numeric_change", False),
        ("intervening_period", False),
        ("split_records", False),
        ("failed", False),
        ("blocked", False),
        ("unapproved", False),
    ],
)
def test_flattened_statement_adversarial_column_unit_scope_and_authority_binding(
    case, accepted
):
    import hashlib

    report = _OCTOBER6["report"]
    record = copy.deepcopy(_OCTOBER6["evidence_records"][0])
    text = record["content"]
    if case == "reversed_columns":
        text = text.replace("2026\n2025", "2025\n2026")
        for current, prior in (
            ("8,695.9", "7,019.2"),
            ("1,583.3", "1,033.1"),
            ("1,891.9", "1,100.7"),
            ("21.8%", "15.7%"),
            ("610.6", "71.0"),
            ("919.2", "363.0"),
            ("10.6%", "5.2%"),
        ):
            text = text.replace(f"{current}\n{prior}", f"{prior}\n{current}")
    elif case == "no_change":
        import re

        text = text.replace("\nChange\n", "\n")
        text = re.sub(r"(?m)^\+[\d.]+(?:%|pts)\n", "", text)
    elif case == "footnote":
        text = text.replace("Net profit\n", "Net profit†\n")
    elif case == "wrong_currency":
        text = text.replace("RMB million", "HKD million")
    elif case == "mixed_currency":
        text = text.replace("RMB million", "HKD million", 1)
    elif case == "wrong_scale":
        text = text.replace("RMB million", "RMB thousand")
    elif case == "mixed_scale":
        text = text.replace("RMB million", "RMB thousand", 1)
    elif case == "reversed_values":
        text = text.replace("8,695.9\n7,019.2", "7,019.2\n8,695.9")
    elif case == "missing_cell":
        text = text.replace("\n7,019.2", "")
    elif case == "duplicate_year":
        text = text.replace("2026\n2025", "2026\n2026")
    elif case == "wrong_year":
        text = text.replace("2026\n2025", "2026\n2024")
    elif case == "wrong_months":
        text = text.replace("Six months ended", "Year ended")
    elif case == "duplicate_revenue":
        text = text.replace(
            "\nEBITDA*\n", "\nRevenue\n8,695.9\n7,019.2\n+23.9%\nEBITDA*\n"
        )
    elif case == "duplicate_earnings":
        text = text.replace(
            "\nAdjusted Net profit**\n",
            "\nNet profit\n610.6\n71.0\n+760.0%\nAdjusted Net profit**\n",
        )
    elif case == "undated_header":
        text = text.replace("Six months ended 30 June", "Interim results")
    elif case.startswith("scope_"):
        qualifier = {
            "scope_attributable": "attributable to owners",
            "scope_adjusted": "adjusted",
            "scope_arbitrary": "anything else",
        }[case]
        report = report.replace(
            "Net profit (Profit for the period)", f"Net profit ({qualifier})"
        )
    elif case == "adjusted_row":
        text = text.replace("\nNet profit\n", "\nAdjusted Net profit\n")
    elif case == "numeric_change":
        text = text.replace("+760.0%\n", "760.0\n")
    elif case == "intervening_period":
        text = text.replace(
            "\nNet profit\n", "\nSix months ended 30 June 2024\nNet profit\n"
        )
    elif case == "failed":
        record["execution_status"] = "FAILED"
    elif case == "blocked":
        record["blocked"] = True
    elif case == "unapproved":
        record["authority_by_url"] = [[record["urls"][0], "UNSUPPORTED"]]
    record["content"] = text
    record["content_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    records = [record]
    if case == "split_records":
        boundary = text.index("Revenue\n")
        records = [
            {**record, "content": text[:boundary]},
            {**record, "content": text[boundary:]},
        ]
        for r in records:
            r["content_sha256"] = hashlib.sha256(r["content"].encode()).hexdigest()
    normalized = _october6_normalize(report, records)
    assert (
        field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "PRIMARY"
    ) is accepted
    if not accepted:
        assert field(normalized, "LATEST_RESULTS_EARNINGS_GROWTH_YOY") == "N/A"


def test_flattened_table_rejects_conflicting_adjacent_period_labels():
    import hashlib

    record = copy.deepcopy(_OCTOBER6["evidence_records"][0])
    record["content"] = record["content"].replace(
        "Six months ended 30 June", "Three months ended 30 June", 1
    )
    record["content_sha256"] = hashlib.sha256(record["content"].encode()).hexdigest()
    normalized = _october6_normalize(_OCTOBER6["report"], [record])
    assert field(normalized, "LATEST_RESULTS_SOURCE_AUTHORITY") == "UNSUPPORTED"
