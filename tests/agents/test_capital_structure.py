import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.capital_structure import (
    assess_capital_structure_scale,
    normalize_legal_output,
    preload_capital_structure_evidence,
    promote_capital_structure,
)
from src.agents.decision_nodes import _ensure_capital_structure_resolution_block


def _legal_payload(capital_structure: dict) -> str:
    return json.dumps(
        {
            "pfic_status": "N/A",
            "vie_structure": "N/A",
            "capital_structure": capital_structure,
        }
    )


def _capital(**overrides) -> dict:
    value = {
        "coverage_status": "FOUND",
        "exposure_type": "LEASE_COMMITMENT",
        "entity": "Data Center A",
        "amount": "USD 2.0 billion",
        "amount_basis": "UNDISCOUNTED",
        "balance_sheet_status": "UNRECOGNIZED",
        "parent_recourse": "NONE",
        "consolidation_risk": "NONE",
        "materiality": "MATERIAL",
        "source_url": "https://example.com/filing",
        "evidence": "Uncommenced lease commitments were disclosed.",
    }
    value.update(overrides)
    return value


def _supporting_evidence(summary: str) -> str:
    return (
        "#### structures_search\n"
        "STATUS: COMPLETED\n"
        "EXECUTION_STATUS: SUCCEEDED\n"
        "EVIDENCE_STATUS: RESULTS_FOUND\n"
        "<result>"
        "<url>https://example.com/filing</url>"
        f"<summary>{summary}</summary>"
        "</result>"
    )


def test_normalizer_qualifies_ordinary_unrecognized_commitment():
    normalized, contract_present = normalize_legal_output(
        _legal_payload(_capital()),
        _supporting_evidence(
            "USD 2.0 billion of uncommenced lease commitments were disclosed."
        ),
    )

    assert contract_present is True
    capital = json.loads(normalized)["capital_structure"]
    assert capital["classification"] == "QUALIFY_RATIOS"


def test_normalizer_blocks_material_parent_recourse():
    normalized, _ = normalize_legal_output(
        _legal_payload(
            _capital(
                exposure_type="GUARANTEE_BACKSTOP",
                parent_recourse="FULL",
            )
        ),
        _supporting_evidence(
            "The parent provided a full guarantee for USD 2.0 billion."
        ),
    )

    capital = json.loads(normalized)["capital_structure"]
    assert capital["classification"] == "BLOCK_BUY"


def test_normalizer_does_not_double_count_recognized_liability():
    normalized, _ = normalize_legal_output(
        _legal_payload(_capital(balance_sheet_status="RECOGNIZED")),
        _supporting_evidence(
            "USD 2.0 billion of recognized lease liabilities were disclosed."
        ),
    )

    capital = json.loads(normalized)["capital_structure"]
    assert capital["classification"] == "CLEAR"


def test_nonrecourse_unconsolidated_entity_only_qualifies_ratios():
    normalized, _ = normalize_legal_output(
        _legal_payload(
            _capital(
                exposure_type="UNCONSOLIDATED_ENTITY",
                parent_recourse="NONE",
                consolidation_risk="NONE",
            )
        ),
        _supporting_evidence(
            "The unconsolidated entity has USD 2.0 billion of maximum exposure."
        ),
    )

    capital = json.loads(normalized)["capital_structure"]
    assert capital["classification"] == "QUALIFY_RATIOS"


def test_scale_assessment_keeps_small_exposure_below_tripwire():
    scale = assess_capital_structure_scale(
        _capital(
            amount="USD 5 million",
            amount_basis="MAXIMUM_EXPOSURE",
            exposure_type="GUARANTEE_BACKSTOP",
            parent_recourse="FULL",
        ),
        {
            "totalDebt": 100_000_000,
            "debtToEquity": 0.8,
            "revenue_TTM": 1_000_000_000,
            "financialCurrency": "USD",
        },
        leverage_threshold=500,
    )

    assert scale["status"] == "MEASURABLE"
    assert scale["exposure_to_debt_pct"] == 5.0
    assert scale["exposure_to_equity_pct"] == 4.0
    assert scale["adjusted_de_pct"] == 84.0
    assert scale["decision_material"] is False


def test_scale_assessment_trips_on_material_size_or_red_zone_crossing():
    material = assess_capital_structure_scale(
        _capital(amount="USD 30 million", amount_basis="MAXIMUM_EXPOSURE"),
        {
            "totalDebt": 100_000_000,
            "debtToEquity": 0.8,
            "totalRevenue": 1_000_000_000,
            "currency": "USD",
        },
        leverage_threshold=500,
    )
    threshold_crossing = assess_capital_structure_scale(
        _capital(amount="USD 15 million", amount_basis="MAXIMUM_EXPOSURE"),
        {
            "totalDebt": 490_000_000,
            "debtToEquity": 4.9,
            "totalRevenue": 2_000_000_000,
            "currency": "USD",
        },
        leverage_threshold=500,
    )

    assert material["decision_material"] is True
    assert threshold_crossing["crosses_red_zone"] is True
    assert threshold_crossing["decision_material"] is True


@pytest.mark.parametrize(
    ("capital_overrides", "raw_overrides", "expected_reason"),
    [
        ({"amount": "approximately USD 5 million"}, {}, "AMOUNT_OR_BASIS_UNUSABLE"),
        ({"amount": "USD 5-10 million"}, {}, "AMOUNT_OR_BASIS_UNUSABLE"),
        ({"amount_basis": "UNDISCOUNTED"}, {}, "AMOUNT_OR_BASIS_UNUSABLE"),
        ({}, {"financialCurrency": "EUR"}, "CURRENCY_MISMATCH"),
        ({}, {"totalDebt": None}, "DEBT_UNAVAILABLE"),
        ({}, {"totalDebt": float("nan")}, "DEBT_UNAVAILABLE"),
        ({}, {"debtToEquity": float("inf")}, "EQUITY_UNAVAILABLE"),
        ({}, {"debtToEquity": -0.8}, "EQUITY_UNAVAILABLE"),
        ({}, {"debtToEquity": True}, "EQUITY_UNAVAILABLE"),
    ],
)
def test_scale_assessment_degrades_strange_inputs_without_raising(
    capital_overrides,
    raw_overrides,
    expected_reason,
):
    capital_values = {
        "amount": "USD 5 million",
        "amount_basis": "MAXIMUM_EXPOSURE",
        **capital_overrides,
    }
    capital = _capital(**capital_values)
    raw = {
        "totalDebt": 100_000_000,
        "debtToEquity": 0.8,
        "revenue_TTM": 1_000_000_000,
        "financialCurrency": "USD",
        **raw_overrides,
    }

    scale = assess_capital_structure_scale(
        capital,
        raw,
        leverage_threshold=500,
    )

    assert scale == {"status": "UNRESOLVED", "reason": expected_reason}


def test_scale_assessment_supports_zero_debt_with_direct_equity():
    scale = assess_capital_structure_scale(
        _capital(amount="USD 5 million", amount_basis="MAXIMUM_EXPOSURE"),
        {
            "totalDebt": 0,
            "totalStockholderEquity": 100_000_000,
            "totalRevenue": 1_000_000_000,
            "financialCurrency": "USD",
        },
        leverage_threshold=500,
    )

    assert scale["status"] == "MEASURABLE"
    assert scale["exposure_to_debt_pct"] is None
    assert scale["exposure_to_equity_pct"] == 5.0
    assert scale["reported_de_pct"] == 0.0
    assert scale["adjusted_de_pct"] == 5.0
    assert scale["decision_material"] is False


def test_scale_assessment_does_not_require_revenue_when_debt_and_equity_align():
    scale = assess_capital_structure_scale(
        _capital(amount="USD 5 million", amount_basis="MAXIMUM_EXPOSURE"),
        {
            "totalDebt": 100_000_000,
            "debtToEquity": 0.8,
            "financialCurrency": "USD",
        },
        leverage_threshold=500,
    )

    assert scale["status"] == "MEASURABLE"
    assert scale["exposure_to_revenue_pct"] is None
    assert scale["decision_material"] is False


def test_normalizer_preserves_omitted_contract_as_unresolved():
    normalized, contract_present = normalize_legal_output(
        '{"pfic_status":"CLEAN","vie_structure":"N/A"}',
        "#### structures_search\nSTATUS: COMPLETED",
    )

    assert contract_present is False
    capital = json.loads(normalized)["capital_structure"]
    assert capital["coverage_status"] == "UNRESOLVED"
    assert capital["classification"] == "UNRESOLVED"


def test_promotion_qualifies_de_without_inventing_adjusted_ratio():
    normalized, _ = normalize_legal_output(
        _legal_payload(_capital()),
        _supporting_evidence(
            "USD 2.0 billion of uncommenced lease commitments were disclosed."
        ),
    )

    updated, promoted = promote_capital_structure("DEBT_TO_EQUITY: 120%", normalized)

    assert promoted is True
    assert "CAPITAL_STRUCTURE_CLASSIFICATION: QUALIFY_RATIOS" in updated
    assert "DE_RATIO_QUALIFICATION: Reported D/E excludes" in updated
    assert "INCREMENTAL_DEBT_LIKE_EXPOSURE: USD 2.0 billion" in updated
    assert "ADJUSTED_DE_RATIO" not in updated


def test_normalizer_rejects_pledged_assets_relabeled_as_lease_commitment():
    normalized, _ = normalize_legal_output(
        _legal_payload(
            _capital(
                amount="TWD 832.76 million",
                evidence="Pledged assets were disclosed.",
            )
        ),
        _supporting_evidence(
            "TWD 832.76 million of land, buildings, and deposits were pledged "
            "as collateral."
        ),
    )

    capital = json.loads(normalized)["capital_structure"]
    assert capital["classification"] == "UNRESOLVED"
    assert capital["exposure_type"] == "UNKNOWN"
    assert capital["amount"] == "N/A"


@pytest.mark.asyncio
async def test_preflight_runs_bounded_searches_and_official_document_extraction():
    calls = []

    async def execute(call, runner):
        calls.append(call)
        if call.name == "search_foreign_sources":
            return SimpleNamespace(
                value=(
                    "<search_results><url>https://www.sec.gov/filing.htm</url>"
                    "</search_results>"
                ),
                blocked=False,
            )
        return SimpleNamespace(value="official filing evidence", blocked=False)

    tools = {
        name: SimpleNamespace(name=name, ainvoke=AsyncMock())
        for name in (
            "get_official_filings",
            "search_foreign_sources",
            "get_official_document",
        )
    }
    with patch(
        "src.runtime_services.get_current_tool_service",
        return_value=SimpleNamespace(execute=execute),
    ):
        evidence = await preload_capital_structure_evidence(
            "TEST",
            "Test Company",
            tools_by_name=tools,
        )

    assert [call.name for call in calls].count("search_foreign_sources") == 2
    assert [call.name for call in calls].count("get_official_filings") == 1
    assert [call.name for call in calls].count("get_official_document") == 1
    assert "CODE-OWNED CAPITAL STRUCTURE PREFLIGHT" in evidence
    assert "債務保証" in calls[1].args["priority_terms"]


def test_final_pm_block_is_inserted_once_before_pm_contract():
    output = (
        "Decision narrative\n\n### --- START PM_BLOCK ---\n"
        "VERDICT: HOLD\n### --- END PM_BLOCK ---\n"
    )
    flags = [
        {
            "type": "DEBT_LIKE_COMMITMENT",
            "detail": "USD 2 billion undiscounted lease commitment.",
        }
    ]

    updated = _ensure_capital_structure_resolution_block(output, flags)
    repeated = _ensure_capital_structure_resolution_block(updated, flags)

    assert updated == repeated
    assert updated.count("CAPITAL STRUCTURE QUALIFICATION (DETERMINISTIC)") == 1
    assert updated.index("CAPITAL STRUCTURE QUALIFICATION") < updated.index("PM_BLOCK")
