"""Tests for the cross-shape state accessors (Tranche 5, Step 1)."""

from __future__ import annotations

import pytest

from src.reporting.state_access import (
    get_apac_regional_report,
    get_auditor_report,
    get_consultant_review,
    get_effective_red_flags,
    get_fundamentals_report,
    get_investment_plan,
    get_pm_output,
    get_valuation_params,
)

# ---------- get_pm_output ----------


def test_get_pm_output_runtime_state() -> None:
    state = {"final_trade_decision": "RUNTIME PM"}
    assert get_pm_output(state) == "RUNTIME PM"


def test_get_pm_output_saved_json_shape() -> None:
    """The bug Tier 1 fixes: persistence writes here, memo couldn't read it."""
    saved = {"final_decision": {"decision": "SAVED PM"}}
    assert get_pm_output(saved) == "SAVED PM"


def test_get_pm_output_runtime_wins_over_saved() -> None:
    both = {
        "final_trade_decision": "RUNTIME",
        "final_decision": {"decision": "SAVED"},
    }
    assert get_pm_output(both) == "RUNTIME"


def test_get_pm_output_falls_back_via_reports_dict() -> None:
    state = {"reports": {"portfolio_manager": "FROM REPORTS"}}
    assert get_pm_output(state) == "FROM REPORTS"


@pytest.mark.parametrize("bad", [None, "", 42, [], {"unrelated": 1}])
def test_get_pm_output_missing_returns_empty(bad) -> None:
    assert get_pm_output(bad) == ""


# ---------- get_investment_plan ----------


def test_get_investment_plan_runtime() -> None:
    assert get_investment_plan({"investment_plan": "RUNTIME PLAN"}) == "RUNTIME PLAN"


def test_get_investment_plan_saved_json_shape() -> None:
    saved = {"investment_analysis": {"investment_plan": "SAVED PLAN"}}
    assert get_investment_plan(saved) == "SAVED PLAN"


def test_get_investment_plan_missing_returns_empty() -> None:
    assert get_investment_plan({}) == ""
    assert get_investment_plan(None) == ""


# ---------- get_fundamentals_report ----------


def test_get_fundamentals_report_runtime() -> None:
    assert get_fundamentals_report({"fundamentals_report": "RUNTIME"}) == "RUNTIME"


def test_get_fundamentals_report_saved_json_shape() -> None:
    saved = {"reports": {"fundamentals_report": "SAVED"}}
    assert get_fundamentals_report(saved) == "SAVED"


# ---------- get_valuation_params ----------


def test_get_valuation_params_runtime() -> None:
    assert get_valuation_params({"valuation_params": "RUNTIME VP"}) == "RUNTIME VP"


def test_get_valuation_params_saved_json_shape() -> None:
    saved = {"reports": {"valuation_params": "SAVED VP"}}
    assert get_valuation_params(saved) == "SAVED VP"


# ---------- Tranche 2/4 readers (auditor, consultant, APAC) ----------


def test_get_auditor_report_both_shapes() -> None:
    assert get_auditor_report({"auditor_report": "RT"}) == "RT"
    assert get_auditor_report({"reports": {"auditor_report": "SV"}}) == "SV"


def test_get_consultant_review_both_shapes() -> None:
    assert get_consultant_review({"consultant_review": "RT"}) == "RT"
    assert get_consultant_review({"reports": {"consultant_review": "SV"}}) == "SV"


def test_get_apac_regional_report_both_shapes() -> None:
    assert get_apac_regional_report({"apac_regional_report": "RT"}) == "RT"
    assert get_apac_regional_report({"reports": {"apac_regional_report": "SV"}}) == "SV"


def test_get_effective_red_flags_ignores_invalid_consultant_resolution() -> None:
    state = {
        "red_flags": [
            {
                "type": "OCF_SOURCE_DISCREPANCY",
                "severity": "WARNING",
                "detail": "OCF differs",
                "risk_penalty": 0.5,
            }
        ],
        "reports": {
            "fundamentals_report": (
                "### --- START DATA_BLOCK ---\n"
                "OPERATING_CASH_FLOW: 151.97M PLN\n"
                "OPERATING_CASH_FLOW_SOURCE: FILING\n"
                "OCF_FILING_REASON: DISCREPANCY\n"
                "### --- END DATA_BLOCK ---\n"
            ),
            "consultant_review": (
                "SPOT_CHECK operatingCashflow: DATA_BLOCK 151.97m PLN FY2025; "
                "FMP 178.06m PLN TTM/Q1 — PERIOD MISMATCH, not a data conflict."
            ),
            "auditor_report": "Operating cash flow: PLN 151.967m",
        },
        "artifact_statuses": {
            "consultant_review": {
                "complete": True,
                "ok": False,
                "message": "Consultant review completed with tool failures",
            }
        },
    }

    flags = get_effective_red_flags(state)

    assert [flag["type"] for flag in flags] == ["OCF_SOURCE_DISCREPANCY"]


# ---------- defensive: non-dict input does not raise ----------


@pytest.mark.parametrize(
    "func",
    [
        get_pm_output,
        get_investment_plan,
        get_fundamentals_report,
        get_valuation_params,
        get_auditor_report,
        get_consultant_review,
        get_apac_regional_report,
    ],
)
def test_accessors_tolerate_non_dict_inputs(func) -> None:
    for bad in (None, 42, "string", [1, 2, 3]):
        assert func(bad) == ""
