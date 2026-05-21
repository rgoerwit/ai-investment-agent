"""Regression tests for Step 4a — persisting optional cross-validation artifacts.

`save_results_to_file()` must surface `auditor_report`, `consultant_review`, and
`valuation_params` under `reports` in the saved JSON, so downstream tools
(source-confidence builder, quality judge, dashboard) can read them without
re-running the graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.persistence import save_results_to_file


def _baseline_result(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "market_report": "market text",
        "sentiment_report": "sentiment text",
        "news_report": "news text",
        "fundamentals_report": "fundamentals text",
        "apac_regional_report": "apac text",
        "final_trade_decision": "VERDICT: BUY",
        "investment_debate_state": {
            "bull_history": "bull",
            "bear_history": "bear",
            "count": 1,
        },
        "investment_plan": "plan",
        "trader_investment_plan": "trader plan",
        "risk_debate_state": {
            "current_risky_response": "r",
            "current_safe_response": "s",
            "current_neutral_response": "n",
        },
        "company_of_interest": "TEST",
    }
    if extra:
        result.update(extra)
    return result


def _save(tmp_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    filepath = save_results_to_file(
        result=result,
        ticker="TEST",
        results_dir=tmp_path,
    )
    return json.loads(Path(filepath).read_text(encoding="utf-8"))


def test_persistence_includes_auditor_consultant_valuation_when_present(
    tmp_path: Path,
) -> None:
    saved = _save(
        tmp_path,
        _baseline_result(
            {
                "auditor_report": "AUDIT: paper profit ratio 0.08, zombie ratio 1.2",
                "consultant_review": "CONSULTANT: APPROVED",
                "valuation_params": (
                    "### --- START VALUATION_PARAMS ---\n"
                    "METHOD: P/E_NORMALIZATION\nSECTOR: Industrials\n"
                    "### --- END VALUATION_PARAMS ---"
                ),
            }
        ),
    )
    reports = saved["reports"]
    assert "auditor_report" in reports
    assert "paper profit ratio" in reports["auditor_report"]
    assert "consultant_review" in reports
    assert "APPROVED" in reports["consultant_review"]
    assert "valuation_params" in reports
    assert "P/E_NORMALIZATION" in reports["valuation_params"]


def test_persistence_defaults_to_empty_string_when_optional_missing(
    tmp_path: Path,
) -> None:
    saved = _save(tmp_path, _baseline_result())
    reports = saved["reports"]
    for key in ("auditor_report", "consultant_review", "valuation_params"):
        assert key in reports, f"{key} should be present even when absent from result"
        assert reports[key] == ""


@pytest.mark.parametrize("missing_field", ["auditor_report", "consultant_review"])
def test_persistence_tolerates_individual_missing_fields(
    tmp_path: Path, missing_field: str
) -> None:
    result = _baseline_result(
        {
            "auditor_report": "AUDIT",
            "consultant_review": "REVIEW",
            "valuation_params": "PARAMS",
        }
    )
    del result[missing_field]
    saved = _save(tmp_path, result)
    assert saved["reports"][missing_field] == ""
