"""Retained issuer rows bind amounts, periods, currency/unit, and earnings scope."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from src.agents.foreign_language_evidence import (
    _exact_decimal,
    _header_period_matches,
    normalize_foreign_language_evidence,
    promote_foreign_growth_evidence,
)
from src.agents.message_utils import make_tool_evidence_record

_CASES = json.loads(
    (Path(__file__).parents[1] / "fixtures/latest_results_statements.json").read_text()
)


@pytest.mark.parametrize("structured_claim", [False, True])
def test_malformed_comma_cells_cannot_become_different_nominal_amounts(
    structured_claim,
):
    case = _CASES[0]
    report = case["report"].replace("79068022", "1,5" if structured_claim else "15")
    report = report.replace("80650914", "1,2" if structured_claim else "12")
    evidence = dict(case["evidence"])
    evidence["content"] = evidence["content"].replace(
        "79,068,022 80,650,914", "1,5 1,2"
    )
    normalized = normalize_foreign_language_evidence(
        report,
        [],
        ticker=case["ticker"],
        additional_records=[make_tool_evidence_record(**evidence)],
    )
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" not in normalized
    assert "LATEST_RESULTS_REVENUE_GROWTH_YOY: N/A" in normalized


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1,5", None),
        ("1,2", None),
        ("12,34", None),
        ("1,,234", None),
        ("-1,5", None),
        ("1,234", "1234"),
        ("12,345,678.125", "12345678.125"),
        ("1,23,456.125", "123456.125"),
        ("-1,23,456.125", "-123456.125"),
        ("-1,234.125", "-1234.125"),
        ("12345678901234567890.123456789", "12345678901234567890.123456789"),
    ],
)
def test_evidence_decimal_grouping_matches_money_contract_without_float_loss(
    value, expected
):
    assert _exact_decimal(value) == (
        Decimal(expected) if expected is not None else None
    )


@pytest.mark.parametrize("case", _CASES, ids=lambda row: row["case"])
def test_retained_statement_layout_preserves_semantic_binding(case):
    record = make_tool_evidence_record(**case["evidence"])
    assert record.authority in {"PRIMARY_ISSUER", "PRIMARY_REGISTRY"}
    normalized = normalize_foreign_language_evidence(
        case["report"], [], ticker=case["ticker"], additional_records=[record]
    )
    assert ("LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" in normalized) is case[
        "expect_primary"
    ]
    if not case["expect_primary"]:
        assert "LATEST_RESULTS_EARNINGS_GROWTH_YOY: N/A" in normalized


@pytest.mark.parametrize(
    "label,end,months",
    [
        ("Q2 2026", "2026-06-30", 3),
        ("2026年第2季", "2026-06-30", 3),
        ("2T26", "2026-06-30", 3),
        ("Six months ended 30 June 2026", "2026-06-30", 6),
        ("First half 2026", "2026-06-30", 6),
        ("First half of 2026", "2026-06-30", 6),
        ("2026년 상반기 (H1 2026)", "2026-06-30", 6),
        ("For the year ended 31 December 2025", "2025-12-31", 12),
    ],
)
def test_observed_period_aliases_bind_exact_date_and_duration(label, end, months):
    assert _header_period_matches(label, end, months)
    assert not _header_period_matches(label, end, 9)
    assert not _header_period_matches(label, "2026-09-30", months)


@pytest.mark.parametrize(
    "label,end,months",
    [
        ("First half 2025", "2026-06-30", 6),
        ("First half FY2026", "2026-06-30", 6),
        ("First half 2026", "2026-03-31", 6),
        ("First half 2026", "2026-06-30", 3),
        ("2026년 상반기 (H1 2025)", "2026-06-30", 6),
        ("2026년 상반기 (H2 2026)", "2026-06-30", 6),
        ("2026년 2분기 (상반기 누적)", "2026-06-30", 3),
    ],
)
def test_period_aliases_reject_conflicting_or_fiscally_ambiguous_labels(
    label, end, months
):
    assert not _header_period_matches(label, end, months)


def _narrative_case():
    return json.loads(
        (
            Path(__file__).parents[1] / "fixtures/latest_results_narrative.json"
        ).read_text()
    )


def test_retained_comparative_narrative_is_secondary_and_never_promotes_growth():
    case = _narrative_case()
    normalized = normalize_foreign_language_evidence(
        case["report"],
        [],
        ticker=case["ticker"],
        additional_records=[make_tool_evidence_record(**case["evidence"])],
    )
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: SECONDARY" in normalized
    assert "LATEST_RESULTS_NORMALIZATION_REASON: SUPPORTED_SECONDARY" in normalized
    assert "LATEST_RESULTS_REVENUE_GROWTH_YOY: N/A" in normalized
    assert "LATEST_RESULTS_EARNINGS_GROWTH_YOY: N/A" in normalized
    senior, _ = promote_foreign_growth_evidence("PE_RATIO_TTM: 12", normalized)
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: SECONDARY" in senior
    for field in (
        "REVENUE",
        "PRIOR_REVENUE",
        "EARNINGS",
        "PRIOR_EARNINGS",
        "REVENUE_GROWTH_YOY",
        "EARNINGS_GROWTH_YOY",
    ):
        assert f"LATEST_RESULTS_{field}:" not in senior


@pytest.mark.parametrize(
    "mutation",
    [
        "currency",
        "unit",
        "scope",
        "current_year",
        "prior_year",
        "amount",
        "comma",
        "duplicate",
        "missing_prior",
        "different_result",
        "fiscal",
        "failed_evidence",
        "wrong_date",
        "wrong_duration",
        "unknown_claim_number",
    ],
)
def test_comparative_narrative_rejects_incompatible_or_incomplete_evidence(mutation):
    case = _narrative_case()
    evidence = dict(case["evidence"])
    report = case["report"]
    content = evidence["content"]
    if mutation in {
        "currency",
        "unit",
        "scope",
        "current_year",
        "prior_year",
        "amount",
        "comma",
    }:
        old, new = {
            "currency": ("KRW", "USD"),
            "unit": ("billion", "million"),
            "scope": ("Net income", "Operating income"),
            "current_year": ("H1 2026", "H1 2027"),
            "prior_year": ("H1 2025", "H1 2024"),
            "amount": ("261.6", "261.7"),
            "comma": ("261.6", "261,6"),
        }[mutation]
        content = content.replace(old, new)
    elif mutation == "duplicate":
        content = content.replace(
            "</summary>",
            "\nConsolidated revenue for H1 2026 was KRW 999 billion, down from KRW 402.4 billion in H1 2025.\n</summary>",
        )
    elif mutation == "missing_prior":
        content = content.replace(", down from KRW 402.4 billion in H1 2025", "")
    elif mutation == "different_result":
        content = content.replace(
            " Net income for",
            '</summary>\n</result>\n<result relevance="0.8">\n<url>'
            + evidence["urls"][0]
            + "</url>\n<summary>Net income for",
        )
    elif mutation == "fiscal":
        content = content.replace("<summary>", "<summary>Fiscal year ends March 31.\n")
    elif mutation == "failed_evidence":
        evidence["evidence_status"] = "AUTH_ERROR"
    elif mutation == "wrong_date":
        report = report.replace("2026-06-30", "2026-03-31")
    elif mutation == "wrong_duration":
        report = report.replace("PERIOD_MONTHS: 6", "PERIOD_MONTHS: 3")
    elif mutation == "unknown_claim_number":
        report = report.replace("PRIOR_REVENUE: 402.4", "PRIOR_REVENUE: N/A")
    evidence["content"] = content
    normalized = normalize_foreign_language_evidence(
        report,
        [],
        ticker=case["ticker"],
        additional_records=[make_tool_evidence_record(**evidence)],
    )
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in normalized
    assert "LATEST_RESULTS_NORMALIZATION_REASON:" in normalized
    assert "LATEST_RESULTS_REVENUE_GROWTH_YOY: N/A" in normalized


def test_equivalent_period_wording_binds_the_existing_primary_table():
    case = _CASES[1]
    report = (
        case["report"]
        .replace("Six months ended 30 June 2026", "First half 2026")
        .replace("Six months ended 30 June 2025", "First half 2025")
    )
    normalized = normalize_foreign_language_evidence(
        report,
        [],
        ticker=case["ticker"],
        additional_records=[make_tool_evidence_record(**case["evidence"])],
    )
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" in normalized


@pytest.mark.parametrize(
    "mutation",
    [
        "reverse_columns",
        "wrong_currency",
        "wrong_unit",
        "unit_letter_in_prose",
        "currency_is_not_a_scale",
        "wrong_scope",
        "missing_earnings",
        "changed_number",
        "wrong_source",
        "wrong_duration",
        "incidental_dates_wrong_table_period",
        "adjacent_incompatible_table",
        "adjacent_reporting_unit",
        "adjacent_cjk_period",
        "adjacent_short_unit",
        "adjacent_usd_unit",
        "adjacent_ambiguous_symbol_unit",
    ],
)
def test_real_two_column_statement_rejects_incompatible_claims(mutation):
    case = _CASES[0]
    report, evidence = case["report"], dict(case["evidence"])
    if mutation == "reverse_columns":
        evidence["content"] = evidence["content"].replace("2025 2024", "2024 2025")
    elif mutation == "wrong_currency":
        report = report.replace(
            "LATEST_RESULTS_CURRENCY: RMB", "LATEST_RESULTS_CURRENCY: USD"
        )
    elif mutation == "wrong_unit":
        report = report.replace(
            "LATEST_RESULTS_REPORTING_UNIT: RMB'000",
            "LATEST_RESULTS_REPORTING_UNIT: million",
        )
    elif mutation == "unit_letter_in_prose":
        report = report.replace(
            "LATEST_RESULTS_REPORTING_UNIT: RMB'000", "LATEST_RESULTS_REPORTING_UNIT: M"
        )
    elif mutation == "currency_is_not_a_scale":
        report = report.replace(
            "LATEST_RESULTS_REPORTING_UNIT: RMB'000",
            "LATEST_RESULTS_REPORTING_UNIT: RMB",
        )
    elif mutation == "wrong_scope":
        report = report.replace(
            "Profit attributable to owners of the Company", "Operating profit"
        )
    elif mutation == "missing_earnings":
        evidence["content"] = evidence["content"].split("Profit attributable to:")[0]
    elif mutation == "changed_number":
        report = report.replace("79068022", "79068023")
    elif mutation == "wrong_source":
        evidence["urls"] = ["https://example.com/different"]
    elif mutation == "wrong_duration":
        report = report.replace(
            "LATEST_RESULTS_PERIOD_MONTHS: 12", "LATEST_RESULTS_PERIOD_MONTHS: 6"
        )
    elif mutation == "incidental_dates_wrong_table_period":
        evidence["content"] = (
            "Balance-sheet dates: 31 December 2025 and 31 December 2024\n"
            + evidence["content"].replace(
                "For the year ended 31 December 2025", "Six months ended 30 June 2025"
            )
        )
    elif mutation == "adjacent_incompatible_table":
        evidence["content"] = evidence["content"].replace(
            "Profit attributable to:",
            "SEGMENT RESULTS\nSix months ended 30 June 2025\n2025 2024\n"
            "Note RMB million RMB million\nProfit attributable to:",
        )
    elif mutation in {"adjacent_reporting_unit", "adjacent_cjk_period"}:
        period = (
            "截至2025年6月30日止六個月\n" if mutation == "adjacent_cjk_period" else ""
        )
        evidence["content"] = evidence["content"].replace(
            "Profit attributable to:",
            f"SEGMENT RESULTS\n{period}Note RMB million RMB million\nProfit attributable to:",
        )
    elif mutation in {
        "adjacent_short_unit",
        "adjacent_usd_unit",
        "adjacent_ambiguous_symbol_unit",
    }:
        declaration = {
            "adjacent_short_unit": "RMB M RMB M",
            "adjacent_usd_unit": "USD million USD million",
            "adjacent_ambiguous_symbol_unit": "$ million $ million",
        }[mutation]
        evidence["content"] = evidence["content"].replace(
            "Profit attributable to:",
            f"SEGMENT RESULTS\n{declaration}\nProfit attributable to:",
        )
    normalized = normalize_foreign_language_evidence(
        report,
        [],
        ticker=case["ticker"],
        additional_records=[make_tool_evidence_record(**evidence)],
    )
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in normalized
    assert "LATEST_RESULTS_NORMALIZATION_REASON:" in normalized


@pytest.mark.parametrize(
    "foreign_heading",
    [
        "",
        "SEGMENT RESULTS\nSix months ended 30 June 2025\n2025 2024\nRMB million\n",
        "Six months ended 30 June 2025 (RMB'000)\n",
    ],
)
def test_vertical_reader_does_not_cross_table_before_revenue(foreign_heading):
    case = _CASES[0]
    evidence = dict(case["evidence"])
    evidence["content"] = (
        "RMB'000\nFor the year ended 31 December 2025\nFor the year ended 31 December 2024\n"
        + foreign_heading
        + "Revenue\n79068022\n80650914\nProfit attributable to owners of the Company\n4500698\n3734429\n"
    )
    normalized = normalize_foreign_language_evidence(
        case["report"],
        [],
        ticker=case["ticker"],
        additional_records=[make_tool_evidence_record(**evidence)],
    )
    assert ("LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" in normalized) is (
        not foreign_heading
    )


@pytest.mark.parametrize("note_cell", [True, False])
def test_inline_reader_requires_declared_note_column_for_third_numeric_cell(note_cell):
    case = _CASES[0]
    evidence = dict(case["evidence"])
    evidence["content"] = (
        "At 31 December 2025 and 31 December 2024.\n"
        "For the year ended 31 December 2025\n2025 2024\nRMB'000 RMB'000\n"
        + (
            "Revenue 4 79,068,022 80,650,914\n"
            if note_cell
            else "Revenue 79,068,022 80,650,914\n"
        )
        + "Profit attributable to owners of the Company 4,500,698 3,734,429\n"
    )
    normalized = normalize_foreign_language_evidence(
        case["report"],
        [],
        ticker=case["ticker"],
        additional_records=[make_tool_evidence_record(**evidence)],
    )
    assert ("LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" in normalized) is (not note_cell)


@pytest.mark.parametrize(
    "header",
    [
        "Six months ended 30 June 2025",
        "Q2 2025",
        "H1 2025",
        "114年Q2",
        "2025年第2季",
        "2T25",
    ],
)
def test_inline_reader_stops_foreign_period_even_before_first_row(header):
    case = _CASES[0]
    evidence = dict(case["evidence"])
    evidence["content"] = evidence["content"].replace(
        "Revenue 4", f"{header}\nRevenue 4"
    )
    normalized = normalize_foreign_language_evidence(
        case["report"],
        [],
        ticker=case["ticker"],
        additional_records=[make_tool_evidence_record(**evidence)],
    )
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in normalized
