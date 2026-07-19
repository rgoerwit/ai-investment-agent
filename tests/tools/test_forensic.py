import json

from src.tools.forensic import calculate_forensic_ratios, validate_forensic_evidence


def test_evidence_gate_accepts_aligned_fresh_complete_statements() -> None:
    result = json.loads(
        validate_forensic_evidence.invoke(
            {
                "analysis_date": "2026-07-19",
                "report_date": "2025-12-31",
                "balance_sheet_found": True,
                "income_statement_found": True,
                "cash_flow_statement_found": True,
                "statement_periods": ["FY2025", "FY2025", "FY2025"],
                "statement_scopes": ["consolidated"] * 3,
                "auditor_opinion_found": True,
            }
        )
    )
    assert result == {"age_months": 6, "reasons": [], "status": "COMPLETE"}


def test_evidence_gate_separates_alignment_and_completeness_reasons() -> None:
    result = json.loads(
        validate_forensic_evidence.invoke(
            {
                "analysis_date": "2026-07-19",
                "report_date": "2024-01-01",
                "balance_sheet_found": True,
                "income_statement_found": True,
                "cash_flow_statement_found": False,
                "statement_periods": ["FY2024", "H1-2025"],
                "statement_scopes": ["consolidated", "standalone"],
                "auditor_opinion_found": False,
            }
        )
    )
    assert result["status"] == "INSUFFICIENT_DATA"
    assert set(result["reasons"]) == {
        "STALE_DATA",
        "STATEMENT_TRIAD_INCOMPLETE",
        "PERIOD_MISMATCH",
        "SCOPE_MISMATCH",
    }


def test_evidence_gate_marks_only_missing_auditor_opinion_partial() -> None:
    result = json.loads(
        validate_forensic_evidence.invoke(
            {
                "analysis_date": "2026-07-19",
                "report_date": "2025-12-31",
                "balance_sheet_found": True,
                "income_statement_found": True,
                "cash_flow_statement_found": True,
                "statement_periods": ["FY2025"] * 3,
                "statement_scopes": ["consolidated"] * 3,
                "auditor_opinion_found": False,
            }
        )
    )
    assert result["status"] == "PARTIAL_DATA"
    assert result["reasons"] == ["AUDITOR_OPINION_UNVERIFIED"]


def test_evidence_gate_rejects_invalid_or_future_dates() -> None:
    common = {
        "analysis_date": "2026-07-19",
        "balance_sheet_found": True,
        "income_statement_found": True,
        "cash_flow_statement_found": True,
        "statement_periods": ["FY2025"] * 3,
        "statement_scopes": ["consolidated"] * 3,
        "auditor_opinion_found": True,
    }
    invalid = json.loads(
        validate_forensic_evidence.invoke({**common, "report_date": "unknown"})
    )
    future = json.loads(
        validate_forensic_evidence.invoke({**common, "report_date": "2027-01-01"})
    )
    assert invalid["reasons"] == ["REPORT_DATE_UNVERIFIED"]
    assert future["reasons"] == ["REPORT_DATE_IN_FUTURE"]


def test_calculator_happy_path_uses_supplied_period_and_scope() -> None:
    result = json.loads(
        calculate_forensic_ratios.invoke(
            {
                "period": "FY2025",
                "scope": "consolidated",
                "revenue": 1000,
                "cogs": 600,
                "accounts_receivable": 100,
                "inventory": 120,
                "accounts_payable": 60,
                "net_income": 100,
                "operating_cash_flow": 120,
                "total_assets": 800,
                "ebit": 90,
                "interest_expense": 30,
            }
        )
    )

    assert result["status"] == "CALCULATED"
    assert result["metrics"]["ocf_to_ni"] == 1.2
    assert result["metrics"]["dso"] == 36.5
    assert result["metrics"]["zombie_ratio"] == 3.0


def test_calculator_never_divides_by_zero_or_negative_denominator() -> None:
    result = json.loads(
        calculate_forensic_ratios.invoke(
            {
                "period": "FY2025",
                "scope": "consolidated",
                "revenue": 0,
                "cogs": -1,
                "net_income": 0,
                "interest_expense": 0,
                "accounts_receivable": 10,
                "inventory": 10,
                "ebit": 10,
            }
        )
    )

    assert result["metrics"]["dso"] is None
    assert result["metrics"]["dio"] is None
    assert result["metrics"]["ocf_to_ni"] is None
    assert result["metrics"]["zombie_ratio"] is None


def test_calculator_rejects_unverified_period_or_scope() -> None:
    result = json.loads(
        calculate_forensic_ratios.invoke({"period": "", "scope": "consolidated"})
    )
    assert result == {
        "status": "INSUFFICIENT_DATA",
        "reason": "PERIOD_OR_SCOPE_UNVERIFIED",
    }


def test_calculator_rejects_invalid_period_length() -> None:
    result = json.loads(
        calculate_forensic_ratios.invoke(
            {"period": "Q1", "scope": "standalone", "days_in_period": 0}
        )
    )
    assert result["reason"] == "INVALID_PERIOD_LENGTH"
