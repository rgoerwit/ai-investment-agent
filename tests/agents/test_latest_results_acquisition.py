"""One post-claim fetch shares the existing research budget and inspection ledger."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.analyst_nodes import _normalize_structured_output
from src.agents.management_guidance import fetch_unresolved_latest_results
from src.forensic_budget import ForeignLanguageBudgetPolicy, ResearchBudgetLedger
from src.runtime_services import RuntimeServices, use_runtime_services
from src.tooling.evidence_recorder import EvidenceRecorder
from src.tooling.inspection_service import INSPECTION_SERVICE
from src.tooling.runtime import ToolExecutionService, ToolInvocation


@pytest.fixture
def acquisition():
    case = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/latest_results_october5.json"
        ).read_text()
    )[0]
    budget = ResearchBudgetLedger(
        ForeignLanguageBudgetPolicy.from_settings(quick_mode=True)
    )
    recorder = EvidenceRecorder()
    services = RuntimeServices(
        tool_service=ToolExecutionService([recorder]),
        inspection_service=INSPECTION_SERVICE,
        evidence_recorder=recorder,
    )
    url = case["evidence_records"][0]["urls"][0]
    services.issuer_authority.register_url(url, provenance="structured_company_profile")
    invoke = AsyncMock(
        return_value=f'STATUS: EVIDENCE_FOUND\nDOCUMENT_METADATA: {{"source_url": "{url}"}}\n'
        + case["evidence_records"][0]["content"]
    )
    tool = SimpleNamespace(name="get_official_document", ainvoke=invoke)
    with (
        use_runtime_services(services),
        patch("src.tools.official_documents.get_official_document", tool),
    ):
        yield case, budget, recorder, services, invoke


async def _fetch(case, budget):
    return await fetch_unresolved_latest_results(
        case["report"], ticker=case["ticker"], company_name="Bossard", budget=budget
    )


@pytest.mark.asyncio
async def test_targeted_fetch_binds_withheld_claim_and_preserves_attempt_for_replay(
    acquisition,
):
    case, budget, recorder, _, invoke = acquisition
    fetched = await _fetch(case, budget)
    assert "latest_results_targeted_document" in fetched
    assert budget.tool_calls == {"get_official_document": 1}
    normalized = _normalize_structured_output(
        "foreign_language_analyst",
        case["report"],
        case["ticker"],
        management_guidance_evidence=fetched,
    )
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" in normalized
    assert "LATEST_RESULTS_EARNINGS_GROWTH_YOY: 41.3%" in normalized
    assert recorder.snapshot()[0].authority_by_url[0][1] == "PRIMARY_ISSUER"
    assert await _fetch(case, budget) == ""
    invoke.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "SUPPORTED_PRIMARY",
        "INVALID_COMPARATIVE_CLAIM",
        "PERIOD_LABEL_MISMATCH",
        "SOURCE_URL_INVALID",
    ],
)
async def test_supported_or_incomplete_claim_does_not_fetch(acquisition, reason):
    case, budget, _, _, invoke = acquisition
    case["report"] = case["report"].replace("COMPARATIVE_ROWS_UNBOUND", reason)
    assert await _fetch(case, budget) == ""
    invoke.assert_not_awaited()
    assert budget.tool_calls == {}


@pytest.mark.asyncio
async def test_unapproved_citation_is_never_fetched(acquisition):
    case, budget, _, _, invoke = acquisition
    case["report"] = case["report"].replace(
        case["evidence_records"][0]["urls"][0], "https://unapproved.example/results.pdf"
    )
    assert await _fetch(case, budget) == ""
    invoke.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["budget", "host", "duplicate"])
async def test_existing_research_budget_and_circuit_breakers_block_targeted_fetch(
    acquisition, failure
):
    case, budget, _, _, invoke = acquisition
    url = case["evidence_records"][0]["urls"][0]
    args = {
        "url": url,
        "keywords": "revenue,net sales,net income,profit",
        "ticker": case["ticker"],
        "company_name": "Bossard",
    }
    if failure == "budget":
        budget.policy = replace(budget.policy, document_calls=0)
        reason = "TOOL_CALL_BUDGET_EXHAUSTED"
    elif failure == "host":
        budget.host_failures["ir.bossard.com"] = budget.policy.host_failure_limit
        reason = "HOST_FAILURE_CIRCUIT_OPEN"
    else:
        assert budget.authorize_tool("get_official_document", args) is None
        reason = "DUPLICATE_TOOL_CALL"
    assert reason in await _fetch(case, budget)
    invoke.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        "STATUS: UNAVAILABLE\nREASON: DOCUMENT_DNS_FAILED",
        "TOOL_BLOCKED: INSPECTION_REJECTED",
    ],
)
async def test_failed_or_blocked_document_is_not_requested_again(acquisition, result):
    case, budget, _, _, invoke = acquisition
    invoke.return_value = result
    assert result in await _fetch(case, budget)
    assert await _fetch(case, budget) == ""
    invoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_valid_secondary_claim_can_fetch_approved_primary_source(acquisition):
    case, budget, _, _, invoke = acquisition
    case["report"] = case["report"].replace(
        "COMPARATIVE_ROWS_UNBOUND", "SUPPORTED_SECONDARY"
    )
    assert "EVIDENCE_FOUND" in await _fetch(case, budget)
    invoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_quarterly_claim_cannot_be_satisfied_by_fetched_annual_document(
    acquisition,
):
    case, budget, _, _, invoke = acquisition
    case["report"] = (
        case["report"]
        .replace("H1 2026", "Q2 2026")
        .replace("H1 2025", "Q2 2025")
        .replace("PERIOD_MONTHS: 6", "PERIOD_MONTHS: 3")
    )
    await _fetch(case, budget)
    normalized = _normalize_structured_output(
        "foreign_language_analyst", case["report"], case["ticker"]
    )
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in normalized
    assert "LATEST_RESULTS_EARNINGS_GROWTH_YOY: N/A" in normalized
    invoke.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["quick", "full", "malformed"])
async def test_analyst_node_fetches_after_normalization_and_saves_budget_and_evidence(
    acquisition, mode
):
    from unittest.mock import MagicMock

    from src.agents.analyst_nodes import create_analyst_node
    from src.config import config as settings
    from src.runtime_config import RuntimeConfig, use_runtime_config

    case, budget, _, _, invoke = acquisition
    report = (
        case["report"]
        + """
### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: SEARCH_FAILED
OPERATING_VS_NET_DIRECTION: UNKNOWN
MATERIAL_NONOPERATING_DRIVER: UNKNOWN
EARNINGS_BASELINE_STATUS: UNKNOWN
NORMALIZED_EARNINGS_AVAILABLE: N/A
### --- END MANAGEMENT_GUIDANCE ---
"""
    )
    if mode == "malformed":
        report = report.replace(
            "LATEST_RESULTS_PRIOR_EARNINGS: 38.7", "LATEST_RESULTS_PRIOR_EARNINGS: N/A"
        )
    node = create_analyst_node(
        MagicMock(),
        "foreign_language_analyst",
        [],
        "foreign_language_report",
        research_budget_policy=budget.policy,
    )
    model = AsyncMock(return_value=SimpleNamespace(content=report, tool_calls=None))
    state = {
        "messages": [],
        "company_of_interest": case["ticker"],
        "company_name": "Bossard",
        "management_guidance_evidence": "#### results_package\nSTATUS: COMPLETED\nEVIDENCE_STATUS: RESULTS_FOUND\n#### earnings_bridge\nSTATUS: COMPLETED\nEVIDENCE_STATUS: RESULTS_FOUND",
    }
    with (
        use_runtime_config(
            RuntimeConfig.from_config(settings).with_overrides(
                quick_mode_active=mode != "full"
            )
        ),
        patch("src.agents.runtime.invoke_with_rate_limit_handling", model),
    ):
        result = await node(
            state,
            {
                "configurable": {
                    "context": SimpleNamespace(
                        ticker=case["ticker"], trade_date="2026-10-05"
                    )
                }
            },
        )
    assert result["artifact_statuses"]["foreign_language_report"]["ok"]
    retained = result["foreign_language_report"]
    telemetry = result["research_budgets"]["foreign_language_analyst"]
    assert telemetry["llm_calls"] == 1
    if mode == "quick":
        assert "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" in retained
        assert "LATEST_RESULTS_REVENUE: 575.7" in retained
        assert "LATEST_RESULTS_EARNINGS_GROWTH_YOY: 41.3%" in retained
        assert (
            "latest_results_targeted_document" in result["management_guidance_evidence"]
        )
        assert telemetry["tool_calls"] == {"get_official_document": 1}
        invoke.assert_awaited_once()
    else:
        assert "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" not in retained
        assert telemetry["tool_calls"] == {}
        invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_targeted_preflight_caps_only_rendered_evidence_using_existing_budget(
    acquisition,
):
    case, budget, recorder, _, invoke = acquisition
    budget.evidence_chars = budget.policy.max_evidence_chars
    fetched = await _fetch(case, budget)
    assert "REASON: EVIDENCE_CHAR_LIMIT" in fetched
    assert "Net sales" not in fetched
    assert budget.evidence_truncated
    assert budget.evidence_chars == budget.policy.max_evidence_chars
    # The normal inspected ledger still retains the source, just as graph tools do.
    assert "Net sales" in recorder.snapshot()[0].content
    invoke.assert_awaited_once()
