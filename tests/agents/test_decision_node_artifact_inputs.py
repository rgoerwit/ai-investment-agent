"""Trader/risk/research prompt assembly must use valid artifact content only.

Failure-artifact stub text (ok=False in artifact_statuses) must never be
injected into downstream agent prompts as if it were a real report; the
"N/A" Data Vacuum sentinel is used instead.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.decision_nodes import create_risk_debater_node, create_trader_node
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
        assert "TRADER PLAN: N/A" in prompt
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
        assert "TRADER PLAN: TRADE PLAN BODY" in prompt


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
