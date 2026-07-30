"""Full-mode regression for management-guidance baseline corrections.

This intentionally models the normal/full evidence and decision path. It is not
a quick-mode smoke: the fixture pins ``quick_mode=False`` and exercises search
formatting, code-owned provenance, Senior promotion/scoring, red flags, and the
deterministic PM BUY floor together.
"""

import pytest

from src.agents.analyst_nodes import (
    _normalize_structured_output,
    _sanitize_fundamentals_output,
)
from src.agents.decision_nodes import create_financial_health_validator_node
from src.agents.output_validation import validate_required_output
from src.agents.verdict_policy import maybe_demote_buy_on_blocking_flags
from src.analysis_snapshot import build_pre_senior_snapshot
from src.runtime_diagnostics import success_artifact
from src.tooling.structured_ingress import build_structured_ingress_record
from src.tools.shared import _format_and_truncate_tavily_result
from src.validators.red_flag_detector import RedFlagDetector, Sector


def _state_with_metrics(**state):
    return {
        **state,
        "structured_inputs": {
            "raw_financial_metrics": build_structured_ingress_record(
                {"trailingPE": 12.5},
                agent_key="junior_fundamentals_analyst",
                tool_name="get_financial_metrics",
            )
        },
    }


def test_hochiki_full_mode_tax_credit_flows_from_search_to_buy_gate() -> None:
    run = {"ticker": "6745.T", "quick_mode": False}
    assert run["quick_mode"] is False

    tax_explanation = (
        "前期は賃上げ促進税制による税額控除の適用がありましたが、"
        "今期は適用がないため当期純利益は減益となる見込みです。"
    )
    search_results = [
        {
            "title": "Generic financial result",
            "url": "https://example.com/generic",
            "content": "A" * 600,
        },
        {
            "title": "Hochiki FY3/26 results briefing transcript",
            "url": "https://finance.logmi.jp/articles/384869",
            "content": "B" * 4300 + tax_explanation + "C" * 4800,
        },
        {
            "title": "Hochiki investor-relations library",
            "url": "https://www.hochiki.co.jp/ir/library/",
            "content": "D" * 700,
        },
    ]
    formatted = _format_and_truncate_tavily_result(
        search_results,
        max_chars=7000,
        query="ホーチキ 決算説明会 賃上げ促進税制 当期純利益",
    )
    assert "https://finance.logmi.jp/articles/384869" in formatted
    assert "賃上げ促進税制" in formatted
    assert formatted.count("</result>") == 3

    fla_draft = """### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_TYPE: TRANSCRIPT
SOURCE_DATE: 2026-05-08
SOURCE_URL: https://finance.logmi.jp/articles/384869
SEARCHES_COMPLETED: model claimed all sources
GUIDANCE_PERIOD: FY3/27
REVENUE_GUIDANCE: JPY110.0bn
OPERATING_PROFIT_GUIDANCE: JPY12.3bn
ORDINARY_OR_PRETAX_PROFIT_GUIDANCE: JPY12.5bn
NET_INCOME_GUIDANCE: JPY9.0bn
NET_INCOME_YOY: -4%
OPERATING_VS_NET_DIRECTION: OP_UP_NET_DOWN
MATERIAL_NONOPERATING_DRIVER: YES
DRIVER_TYPE: TAX_CREDIT
DRIVER_PERSISTENCE: EXPIRING
DRIVER_MATERIALITY: MATERIAL
DRIVER_AFFECTED_PERIOD: FY3/26
MANAGEMENT_IDENTIFIED: YES
EARNINGS_BASELINE_STATUS: DURABLE
NORMALIZED_EARNINGS_AVAILABLE: NO
DRIVER_DESCRIPTION: FY3/26 benefited from the wage-increase promotion tax credit; FY3/27 does not.
### --- END MANAGEMENT_GUIDANCE ---
### --- START LATEST_RESULTS ---
LATEST_RESULTS_COVERAGE_STATUS: NOT_FOUND
LATEST_RESULTS_PERIOD: N/A
LATEST_RESULTS_PERIOD_END: N/A
LATEST_RESULTS_PRIOR_PERIOD: N/A
LATEST_RESULTS_PRIOR_PERIOD_END: N/A
LATEST_RESULTS_PERIOD_MONTHS: N/A
LATEST_RESULTS_CURRENCY: N/A
LATEST_RESULTS_REPORTING_UNIT: N/A
LATEST_RESULTS_REVENUE: N/A
LATEST_RESULTS_PRIOR_REVENUE: N/A
LATEST_RESULTS_EARNINGS: N/A
LATEST_RESULTS_PRIOR_EARNINGS: N/A
LATEST_RESULTS_EARNINGS_SCOPE: N/A
LATEST_RESULTS_SOURCE_URL: N/A
### --- END LATEST_RESULTS ---
"""
    preflight = """#### results_package
STATUS: COMPLETED
#### earnings_bridge
STATUS: COMPLETED
#### statutory_filing_api
STATUS: COMPLETED
"""
    fla_report = _normalize_structured_output(
        "foreign_language_analyst",
        fla_draft,
        run["ticker"],
        management_guidance_evidence=preflight,
    )
    assert validate_required_output("foreign_language_analyst", fla_report)["ok"]
    assert "SEARCH_PROVENANCE: CODE_OWNED_PREFLIGHT" in fla_report
    assert "EARNINGS_BASELINE_STATUS: TEMPORARILY_BOOSTED" in fla_report

    senior_draft = """### --- START DATA_BLOCK ---
SECTOR: Industrials
GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=1; ROA_ROE_IMPROVING=1; GROSS_MARGIN=1; GLOBAL_EXPANSION=0; R_AND_D_CAPEX_BACKLOG=0
RAW_GROWTH_SCORE: 4/6
ADJUSTED_GROWTH_SCORE: 66.7% (based on 6 available points)
### --- END DATA_BLOCK ---
"""
    fundamentals = _sanitize_fundamentals_output(
        senior_draft,
        "",
        run["ticker"],
        foreign_data=fla_report,
        canonical_snapshot=build_pre_senior_snapshot(
            _state_with_metrics(foreign_language_report=fla_report)
        ),
    )
    assert validate_required_output("fundamentals_analyst", fundamentals)["ok"]
    assert "EPS_GROWTH=0" in fundamentals
    assert "RAW_GROWTH_SCORE: 3/6" in fundamentals

    metrics = RedFlagDetector.extract_metrics(fundamentals)
    flags, _ = RedFlagDetector.detect_red_flags(
        metrics,
        ticker=run["ticker"],
        sector=Sector.INDUSTRIALS,
    )
    flag_types = {flag["type"] for flag in flags}
    assert "TRANSIENT_STRENGTH_DISTORTION" in flag_types
    assert "NORMALIZED_EARNINGS_REQUIRED" in flag_types
    assert any(flag.get("blocks_buy") is True for flag in flags)

    pm_buy = """#### PORTFOLIO MANAGER VERDICT: BUY
Action: BUY
### --- START PM_BLOCK ---
VERDICT: BUY
POSITION_SIZE: 3%
### --- END PM_BLOCK ---
"""
    demoted, changed = maybe_demote_buy_on_blocking_flags(
        pm_buy,
        red_flags=flags,
        ticker=run["ticker"],
    )
    assert changed is True
    assert "VERDICT: HOLD" in demoted
    assert "NORMALIZED_EARNINGS_REQUIRED" in demoted


def test_clean_durable_full_mode_control_is_not_demoted() -> None:
    run = {"ticker": "CONTROL.T", "quick_mode": False}
    assert run["quick_mode"] is False
    pm_buy = """#### PORTFOLIO MANAGER VERDICT: BUY
### --- START PM_BLOCK ---
VERDICT: BUY
### --- END PM_BLOCK ---
"""

    unchanged, changed = maybe_demote_buy_on_blocking_flags(
        pm_buy,
        red_flags=[],
        ticker=run["ticker"],
    )

    assert changed is False
    assert unchanged == pm_buy


@pytest.mark.asyncio
async def test_unresolved_guidance_survives_full_contract_lifecycle() -> None:
    fla_report = """### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: UNRESOLVED_AFTER_TARGETED_SEARCH
SOURCE_TYPE: N/A
SOURCE_AUTHORITY: UNKNOWN
SOURCE_DATE: N/A
SOURCE_URL: N/A
SEARCHES_COMPLETED: results_package=SUCCEEDED/RESULTS_FOUND
SEARCH_PROVENANCE: CODE_OWNED_PREFLIGHT
GUIDANCE_PERIOD: N/A
REVENUE_GUIDANCE: N/A
OPERATING_PROFIT_GUIDANCE: N/A
ORDINARY_OR_PRETAX_PROFIT_GUIDANCE: N/A
NET_INCOME_GUIDANCE: N/A
NET_INCOME_YOY: N/A
OPERATING_VS_NET_DIRECTION: UNKNOWN
MATERIAL_NONOPERATING_DRIVER: UNKNOWN
DRIVER_TYPE: UNKNOWN
DRIVER_PERSISTENCE: UNKNOWN
DRIVER_MATERIALITY: UNKNOWN
DRIVER_AFFECTED_PERIOD: UNKNOWN
MANAGEMENT_IDENTIFIED: UNKNOWN
EARNINGS_BASELINE_STATUS: UNKNOWN
NORMALIZED_EARNINGS_AVAILABLE: UNKNOWN
GUIDANCE_BRIDGE_STATUS: UNRESOLVED
### --- END MANAGEMENT_GUIDANCE ---
### --- START LATEST_RESULTS ---
LATEST_RESULTS_COVERAGE_STATUS: NOT_FOUND
LATEST_RESULTS_PERIOD: N/A
LATEST_RESULTS_PERIOD_END: N/A
LATEST_RESULTS_PRIOR_PERIOD: N/A
LATEST_RESULTS_PRIOR_PERIOD_END: N/A
LATEST_RESULTS_PERIOD_MONTHS: N/A
LATEST_RESULTS_CURRENCY: N/A
LATEST_RESULTS_REPORTING_UNIT: N/A
LATEST_RESULTS_REVENUE: N/A
LATEST_RESULTS_PRIOR_REVENUE: N/A
LATEST_RESULTS_EARNINGS: N/A
LATEST_RESULTS_PRIOR_EARNINGS: N/A
LATEST_RESULTS_EARNINGS_SCOPE: N/A
LATEST_RESULTS_SOURCE_URL: N/A
### --- END LATEST_RESULTS ---
"""
    snapshot = build_pre_senior_snapshot(
        _state_with_metrics(foreign_language_report=fla_report)
    )
    senior_draft = """### --- START DATA_BLOCK ---
SECTOR: Industrials
DE_RATIO: 0.2
NET_INCOME: 80
FREE_CASH_FLOW: 100
ADJUSTED_HEALTH_SCORE: 75%
GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=0; EPS_GROWTH=0; ROA_ROE_IMPROVING=0; GROSS_MARGIN=0; GLOBAL_EXPANSION=0; R_AND_D_CAPEX_BACKLOG=0
RAW_GROWTH_SCORE: 0/6
ADJUSTED_GROWTH_SCORE: 0.0% (based on 6 available points)
### --- END DATA_BLOCK ---
"""

    fundamentals = _sanitize_fundamentals_output(
        senior_draft,
        "",
        "CONTROL",
        foreign_data=fla_report,
        canonical_snapshot=snapshot,
    )
    validation = validate_required_output("fundamentals_analyst", fundamentals)
    state = {
        "company_of_interest": "CONTROL",
        "fundamentals_report": fundamentals,
        "analysis_snapshot": snapshot,
        **success_artifact("fundamentals_report", fundamentals),
    }
    validator_result = await create_financial_health_validator_node()(state, {})

    assert validation["ok"] is True
    assert "GUIDANCE_COVERAGE_STATUS: UNRESOLVED_AFTER_TARGETED_SEARCH" in fundamentals
    assert "LATEST_RESULTS_COVERAGE_STATUS: NOT_FOUND" in fundamentals
    assert validator_result["pre_screening_result"] == "PASS"
