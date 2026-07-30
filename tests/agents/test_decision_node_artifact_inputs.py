"""Trader/risk/research prompt assembly must use valid artifact content only.

Failure-artifact stub text (ok=False in artifact_statuses) must never be
injected into downstream agent prompts as if it were a real report; the
"N/A" Data Vacuum sentinel is used instead.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.decision_nodes import (
    create_financial_health_validator_node,
    create_portfolio_manager_node,
    create_risk_debater_node,
    create_trader_node,
)
from src.agents.pm_inputs import DIRECT_PM_INPUT_FIELDS
from src.agents.research_nodes import _build_research_report_bundle

STUB = "[SYSTEM]: market_analyst output missing required structure"


def _failed(field: str, stub: str = STUB) -> dict:
    return {field: {"complete": True, "ok": False, "content": stub}}


def _ok(field: str, content: str) -> dict:
    return {field: {"complete": True, "ok": True, "content": content}}


def _mock_llm() -> SimpleNamespace:
    return SimpleNamespace(model_name="gemini-3-flash-preview")


async def _captured_prompt(node, state) -> str:
    with patch(
        "src.agents.decision_nodes.agent_runtime.invoke_with_rate_limit_handling",
        new=AsyncMock(return_value=SimpleNamespace(content="ok")),
    ) as mock_invoke:
        await node(state, {})
    messages = mock_invoke.call_args.args[1]
    return messages[0].content


class TestTraderNodeInputs:
    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_failed_artifact_stub_excluded(self, mock_get_prompt):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="trader prompt", agent_name="Trader"
        )
        state = {
            "company_of_interest": "0005.HK",
            "market_report": STUB,
            "fundamentals_report": "FUNDAMENTALS BODY TEXT",
            "artifact_statuses": {
                **_failed("market_report"),
                **_ok("fundamentals_report", "FUNDAMENTALS BODY TEXT"),
            },
        }
        prompt = await _captured_prompt(create_trader_node(_mock_llm(), None), state)
        assert STUB not in prompt
        assert "FUNDAMENTALS BODY TEXT" in prompt
        assert "MARKET ANALYST REPORT:\nN/A" in prompt

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_missing_fields_become_na_without_error(self, mock_get_prompt):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="trader prompt", agent_name="Trader"
        )
        prompt = await _captured_prompt(
            create_trader_node(_mock_llm(), None), {"company_of_interest": "0005.HK"}
        )
        assert "MARKET ANALYST REPORT:\nN/A" in prompt
        assert "RESEARCH MANAGER PLAN:\nN/A" in prompt

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_failed_valuation_params_omitted(self, mock_get_prompt):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="trader prompt", agent_name="Trader"
        )
        state = {
            "company_of_interest": "0005.HK",
            "valuation_params": STUB,
            "artifact_statuses": _failed("valuation_params"),
        }
        prompt = await _captured_prompt(create_trader_node(_mock_llm(), None), state)
        assert "VALUATION PARAMETERS" not in prompt

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_do_not_initiate_has_no_numeric_downside_level(self, mock_get_prompt):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="trader prompt", agent_name="Trader"
        )
        response = SimpleNamespace(
            content=(
                "TRADE_BLOCK:\n"
                "ACTION: DO_NOT_INITIATE\n"
                "SIZE: 0.0%\n"
                "CONVICTION: Low\n"
                "ENTRY: N/A\n"
                "STOP: 2.10 (downside review level)\n"
                "TARGET_1: N/A\n"
                "TARGET_2: N/A\n"
                "HORIZON: 24 months\n"
                "SPECIAL: patient execution\n"
            )
        )
        with patch(
            "src.agents.decision_nodes.agent_runtime.invoke_with_rate_limit_handling",
            new=AsyncMock(return_value=response),
        ):
            result = await create_trader_node(_mock_llm(), None)(
                {"company_of_interest": "AGS.SI"}, {}
            )

        plan = result["trader_investment_plan"]
        assert "STOP: N/A (no position; use thesis-break conditions)" in plan
        assert "STOP: 2.10" not in plan


class TestRiskNodeInputs:
    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_failed_consultant_and_trader_plan_excluded(self, mock_get_prompt):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="risk prompt", agent_name="Risky Analyst"
        )
        state = {
            "company_of_interest": "0005.HK",
            "consultant_review": STUB,
            "trader_investment_plan": STUB,
            "artifact_statuses": {
                **_failed("consultant_review"),
                **_failed("trader_investment_plan"),
            },
        }
        node = create_risk_debater_node(_mock_llm(), "risky_analyst")
        prompt = await _captured_prompt(node, state)
        assert STUB not in prompt
        assert "POSITION PLANNER OUTPUT: N/A" in prompt
        assert "N/A (consultant disabled or unavailable)" in prompt

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_valid_inputs_flow_through(self, mock_get_prompt):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="risk prompt", agent_name="Risky Analyst"
        )
        state = {
            "company_of_interest": "0005.HK",
            "consultant_review": "CONSULTANT BODY",
            "trader_investment_plan": "TRADE PLAN BODY",
            "artifact_statuses": {
                **_ok("consultant_review", "CONSULTANT BODY"),
                **_ok("trader_investment_plan", "TRADE PLAN BODY"),
            },
        }
        node = create_risk_debater_node(_mock_llm(), "risky_analyst")
        prompt = await _captured_prompt(node, state)
        assert "CONSULTANT BODY" in prompt
        assert "POSITION PLANNER OUTPUT: TRADE PLAN BODY" in prompt


class TestResearchBundleInputs:
    BUDGETS = {"market": 500, "sentiment": 500, "news": 500, "fundamentals": 500}

    def test_failed_artifact_stub_excluded(self):
        state = {
            "market_report": STUB,
            "news_report": "NEWS BODY",
            "artifact_statuses": _failed("market_report"),
        }
        bundle = _build_research_report_bundle(state, self.BUDGETS)
        assert STUB not in bundle
        assert "MARKET ANALYST REPORT:\nN/A" in bundle
        assert "NEWS BODY" in bundle

    def test_empty_state_is_all_na(self):
        bundle = _build_research_report_bundle({}, self.BUDGETS)
        assert "MARKET ANALYST REPORT:\nN/A" in bundle
        assert "FUNDAMENTALS ANALYST REPORT:\nN/A" in bundle


class TestFinancialValidatorArtifactInputs:
    @pytest.mark.asyncio
    async def test_invalid_fundamentals_fallback_cannot_pass_pre_screening(self):
        fallback = """
### --- START DATA_BLOCK ---
SECTOR: Industrials
DE_RATIO: 0.2
FREE_CASH_FLOW: 100
NET_INCOME: 80
ADJUSTED_HEALTH_SCORE: 75%
### --- END DATA_BLOCK ---
"""
        state = {
            "company_of_interest": "TEST",
            "fundamentals_report": fallback,
            "artifact_statuses": {
                "fundamentals_report": {
                    "complete": True,
                    "ok": False,
                    "content": fallback,
                    "message": (
                        "GUIDANCE_COVERAGE_STATUS=MISSING; expected a domain status"
                    ),
                },
            },
        }

        result = await create_financial_health_validator_node()(state, {})

        assert result["pre_screening_result"] == "REJECT"
        assert result["red_flags"][0]["type"] == "DATA_CONTRACT_INVALID"
        assert "GUIDANCE_COVERAGE_STATUS=MISSING" in result["red_flags"][0]["detail"]


class TestPortfolioManagerArtifactInputs:
    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_pm_uses_structured_coverage_instead_of_raw_fla(
        self, mock_get_prompt
    ):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="portfolio manager prompt", agent_name="Portfolio Manager"
        )
        fundamentals = """
### --- START DATA_BLOCK ---
SECTOR: Financials
PE_RATIO_TTM: 16.55
GUIDANCE_COVERAGE_STATUS: NOT_DISCLOSED_AFTER_TARGETED_SEARCH
### --- END DATA_BLOCK ---
"""
        state = {
            "company_of_interest": "B3SA3.SA",
            "fundamentals_report": fundamentals,
            "foreign_language_report": "RAW_FLA_SENTINEL",
            "investment_plan": "Research cites Foreign Language Analyst.",
            "artifact_statuses": {
                **_ok("fundamentals_report", fundamentals),
                **_ok("foreign_language_report", "RAW_FLA_SENTINEL"),
            },
        }

        prompt = await _captured_prompt(
            create_portfolio_manager_node(_mock_llm(), None), state
        )

        assert "RAW_FLA_SENTINEL" not in prompt
        assert "GUIDANCE_COVERAGE_STATUS=NOT_DISCLOSED_AFTER_TARGETED_SEARCH" in prompt

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_pm_excludes_failed_foreign_language_report(self, mock_get_prompt):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="portfolio manager prompt", agent_name="Portfolio Manager"
        )
        fundamentals = """
### --- START DATA_BLOCK ---
SECTOR: Financials
PE_RATIO_TTM: 16.55
### --- END DATA_BLOCK ---
"""
        state = {
            "company_of_interest": "B3SA3.SA",
            "fundamentals_report": fundamentals,
            "foreign_language_report": STUB,
            "artifact_statuses": {
                **_ok("fundamentals_report", fundamentals),
                **_failed("foreign_language_report"),
            },
        }

        prompt = await _captured_prompt(
            create_portfolio_manager_node(_mock_llm(), None), state
        )

        assert STUB not in prompt
        assert "FOREIGN LANGUAGE / NATIVE-SOURCE ANALYST REPORT" not in prompt

    def test_foreign_language_report_is_not_direct_pm_input(self):
        assert "foreign_language_report" not in DIRECT_PM_INPUT_FIELDS

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_pm_gets_one_semantic_trace_correction(self, mock_get_prompt):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="portfolio manager prompt",
            agent_name="Portfolio Manager",
        )
        response_template = (
            "### PORTFOLIO MANAGER VERDICT: HOLD\n"
            "### THESIS COMPLIANCE SUMMARY\n"
            "Hard Fail Checks: PASS\n"
            "### FINAL EXECUTION PARAMETERS\n"
            "- Action: HOLD\n"
            "### --- START PM_BLOCK ---\n"
            "VERDICT: HOLD\n"
            "RISK_TALLY: 0.5\n"
            "ZONE: MODERATE\n"
            "DECISION_FACTS: {claim_id}\n"
            "DECISION_GATES: NONE\n"
            "### --- END PM_BLOCK ---"
        )
        responses = [
            SimpleNamespace(
                content=response_template.format(claim_id="claim:not-registered")
            ),
            SimpleNamespace(content=response_template.format(claim_id="claim:pe")),
        ]
        fundamentals = (
            "### --- START DATA_BLOCK ---\n"
            "PE_RATIO_TTM: 12.0\n"
            "VALUATION_INPUT_RELIABILITY: USABLE\n"
            "### --- END DATA_BLOCK ---"
        )
        state = {
            "company_of_interest": "TEST",
            "fundamentals_report": fundamentals,
            "pre_screening_result": "PASS",
            "red_flags": [],
            "artifact_statuses": _ok("fundamentals_report", fundamentals),
            "analysis_snapshot": {
                "contract_status": "VALID",
                "claims": {
                    "claim:pe": {
                        "id": "claim:pe",
                        "field": "PE_RATIO_TTM",
                        "value": "12.0",
                        "period": None,
                        "authority": "AGGREGATOR",
                        "coverage": "FOUND",
                        "decision_eligible": True,
                        "decision_role": "SUPPORT",
                    }
                },
            },
        }

        with patch(
            "src.agents.decision_nodes.agent_runtime.invoke_with_rate_limit_handling",
            new=AsyncMock(side_effect=responses),
        ) as mock_invoke:
            result = await create_portfolio_manager_node(_mock_llm(), None)(state, {})

        assert mock_invoke.await_count == 2
        assert result["decision_trace"]["status"] == "VALID"
        assert result["decision_trace"]["decision_facts"] == ["claim:pe"]
