from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage


def _governance_card() -> dict:
    return {
        "ticker": "009970.KS",
        "canonical_name": "Youngone Holdings Co., Ltd.",
        "entity_role": "INTERMEDIATE_HOLDCO",
        "confidence": "clean",
        "related_listed": [
            {"ticker": "111770.KS", "relationship": "operating_subsidiary"}
        ],
    }


def _state_with_card() -> dict:
    return {
        "company_of_interest": "009970.KS",
        "company_name": "Youngone Holdings Co., Ltd.",
        "entity_governance_card": _governance_card(),
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": (
            "### --- START DATA_BLOCK ---\n"
            "ADJUSTED_HEALTH_SCORE: 70%\n"
            "PE_RATIO_TTM: 12.0\n"
            "OPERATING_CASH_FLOW: $600M\n"
            "### --- END DATA_BLOCK ---"
        ),
        "value_trap_report": "value trap",
        "investment_plan": "research plan",
        "trader_investment_plan": "trader plan",
        "consultant_review": "consultant",
        "apac_regional_report": "apac",
        "risk_debate_state": {"current_safe_response": "safe view"},
        "investment_debate_state": {"bull_history": "bull", "bear_history": "bear"},
    }


def _state_with_governance_card(card: dict) -> dict:
    state = _state_with_card()
    state["entity_governance_card"] = card
    return state


@pytest.mark.asyncio
async def test_researcher_and_manager_receive_governance_card(monkeypatch):
    from src.agents import research_nodes

    prompts: list[str] = []

    async def fake_invoke(_llm, messages, **_kwargs):
        prompts.append(str(messages[0].content))
        return AIMessage(content="argument")

    monkeypatch.setattr(
        research_nodes.agent_runtime, "invoke_with_rate_limit_handling", fake_invoke
    )
    monkeypatch.setattr(
        research_nodes,
        "get_runtime_config",
        lambda _config: SimpleNamespace(enable_memory=False),
    )

    researcher = research_nodes.create_researcher_node(Mock(), None, "bull_researcher")
    manager = research_nodes.create_research_manager_node(Mock(), None)

    await researcher(_state_with_card(), {"configurable": {}})
    await manager(_state_with_card(), {"configurable": {}})

    assert len(prompts) == 2
    assert all("ENTITY GOVERNANCE CARD" in prompt for prompt in prompts)
    assert all("111770.KS" in prompt for prompt in prompts)


@pytest.mark.asyncio
async def test_consultant_receives_governance_card(monkeypatch):
    from src.agents import consultant_nodes

    captured: dict[str, str] = {}

    async def fake_invoke(_llm, messages, **_kwargs):
        captured["prompt"] = str(messages[0].content)
        return AIMessage(content="consultant review")

    monkeypatch.setattr(
        consultant_nodes.agent_runtime, "invoke_with_rate_limit_handling", fake_invoke
    )

    node = consultant_nodes.create_consultant_node(Mock(), tools=[], quick_mode=True)
    await node(_state_with_card(), {"configurable": {}})

    assert "ENTITY GOVERNANCE CARD" in captured["prompt"]
    assert "111770.KS" in captured["prompt"]


@pytest.mark.asyncio
async def test_consultant_reconciliation_directive_only_on_conflict(monkeypatch):
    from src.agents import consultant_nodes

    prompts: list[str] = []

    async def fake_invoke(_llm, messages, **_kwargs):
        prompts.append(str(messages[0].content))
        return AIMessage(content="consultant review")

    monkeypatch.setattr(
        consultant_nodes.agent_runtime, "invoke_with_rate_limit_handling", fake_invoke
    )

    node = consultant_nodes.create_consultant_node(Mock(), tools=[], quick_mode=True)
    await node(_state_with_card(), {"configurable": {}})
    assert "GOVERNANCE RECONCILIATION DIRECTIVE" not in prompts[-1]

    conflict_card = {**_governance_card(), "confidence": "conflict"}
    await node(_state_with_governance_card(conflict_card), {"configurable": {}})
    assert "GOVERNANCE RECONCILIATION DIRECTIVE" in prompts[-1]


@pytest.mark.asyncio
async def test_trader_risk_and_pm_receive_governance_card(monkeypatch):
    from src.agents import decision_nodes

    prompts: list[str] = []

    async def fake_invoke(_llm, messages, **_kwargs):
        prompts.append(str(messages[0].content))
        return AIMessage(content="VERDICT: HOLD\nCONVICTION: LOW\nRISK_TALLY: 0")

    monkeypatch.setattr(
        decision_nodes.agent_runtime, "invoke_with_rate_limit_handling", fake_invoke
    )

    trader = decision_nodes.create_trader_node(Mock(), None)
    risk = decision_nodes.create_risk_debater_node(Mock(), "safe_analyst")
    pm = decision_nodes.create_portfolio_manager_node(Mock(), None)

    await trader(_state_with_card(), {"configurable": {}})
    await risk(_state_with_card(), {"configurable": {}})
    await pm(_state_with_card(), {"configurable": {}})

    assert len(prompts) == 3
    assert all("ENTITY GOVERNANCE CARD" in prompt for prompt in prompts)
    assert all("111770.KS" in prompt for prompt in prompts)


@pytest.mark.asyncio
async def test_pm_vehicle_choice_directive_requires_nonstandard_related(monkeypatch):
    from src.agents import decision_nodes

    prompts: list[str] = []

    async def fake_invoke(_llm, messages, **_kwargs):
        prompts.append(str(messages[0].content))
        return AIMessage(content="VERDICT: HOLD\nCONVICTION: LOW\nRISK_TALLY: 0")

    monkeypatch.setattr(
        decision_nodes.agent_runtime, "invoke_with_rate_limit_handling", fake_invoke
    )

    pm = decision_nodes.create_portfolio_manager_node(Mock(), None)
    await pm(_state_with_card(), {"configurable": {}})
    assert "VEHICLE-CHOICE DIRECTIVE" in prompts[-1]

    standalone_card = {
        **_governance_card(),
        "entity_role": "STANDALONE",
        "related_listed": [],
    }
    await pm(_state_with_governance_card(standalone_card), {"configurable": {}})
    assert "VEHICLE-CHOICE DIRECTIVE" not in prompts[-1]


@pytest.mark.asyncio
async def test_researcher_round1_uses_structured_coverage_not_raw_fla(monkeypatch):
    from src.agents import research_nodes

    captured: dict[str, str] = {}

    async def fake_invoke(_llm, messages, **_kwargs):
        captured["prompt"] = str(messages[0].content)
        return AIMessage(content="argument")

    monkeypatch.setattr(
        research_nodes.agent_runtime, "invoke_with_rate_limit_handling", fake_invoke
    )
    monkeypatch.setattr(
        research_nodes,
        "get_runtime_config",
        lambda _config: SimpleNamespace(enable_memory=False),
    )

    researcher = research_nodes.create_researcher_node(
        Mock(), None, "bull_researcher", round_num=1
    )
    state = _state_with_card()
    state["fundamentals_report"] = state["fundamentals_report"].replace(
        "### --- END DATA_BLOCK ---",
        "GUIDANCE_COVERAGE_STATUS: NOT_DISCLOSED_AFTER_TARGETED_SEARCH\n"
        "### --- END DATA_BLOCK ---",
    )
    state["foreign_language_report"] = (
        "RAW_FLA_SENTINEL: No Value-Up program disclosed in DART filings.\n"
    )

    await researcher(state, {"configurable": {}})

    assert "RAW_FLA_SENTINEL" not in captured["prompt"]
    assert (
        "GUIDANCE_COVERAGE_STATUS=NOT_DISCLOSED_AFTER_TARGETED_SEARCH"
        in captured["prompt"]
    )


@pytest.mark.asyncio
async def test_researcher_round2_omits_fla_anchor(monkeypatch):
    """Round 2 anchor budgets must NOT carry FLA — Senior's DATA_BLOCK is the digest by then."""
    from src.agents import research_nodes

    captured: dict[str, str] = {}

    async def fake_invoke(_llm, messages, **_kwargs):
        captured["prompt"] = str(messages[0].content)
        return AIMessage(content="argument")

    monkeypatch.setattr(
        research_nodes.agent_runtime, "invoke_with_rate_limit_handling", fake_invoke
    )
    monkeypatch.setattr(
        research_nodes,
        "get_runtime_config",
        lambda _config: SimpleNamespace(enable_memory=False),
    )

    researcher = research_nodes.create_researcher_node(
        Mock(), None, "bull_researcher", round_num=2
    )
    state = _state_with_card()
    state["foreign_language_report"] = (
        "FLA_ROUND2_SENTINEL_TOKEN should not appear in anchor-budget prompt"
    )
    state["investment_debate_state"] = {
        "bull_round1": "bull r1",
        "bear_round1": "bear r1",
    }

    await researcher(state, {"configurable": {}})

    assert "FLA_ROUND2_SENTINEL_TOKEN" not in captured["prompt"]
    assert "FOREIGN LANGUAGE / LOCAL FILINGS" not in captured["prompt"]


@pytest.mark.asyncio
async def test_research_manager_uses_structured_coverage_not_raw_fla(monkeypatch):
    from src.agents import research_nodes

    captured: dict[str, str] = {}

    async def fake_invoke(_llm, messages, **_kwargs):
        captured["prompt"] = str(messages[0].content)
        return AIMessage(content="plan")

    monkeypatch.setattr(
        research_nodes.agent_runtime, "invoke_with_rate_limit_handling", fake_invoke
    )

    manager = research_nodes.create_research_manager_node(Mock(), None)
    state = _state_with_card()
    state["fundamentals_report"] = state["fundamentals_report"].replace(
        "### --- END DATA_BLOCK ---",
        "CAPITAL_PLAN_STATUS: NOT_DISCLOSED\n### --- END DATA_BLOCK ---",
    )
    state["foreign_language_report"] = (
        "RAW_FLA_SENTINEL: No mid-term plan published as of 2026-05."
    )

    await manager(state, {"configurable": {}})

    assert "RAW_FLA_SENTINEL" not in captured["prompt"]
    assert "CAPITAL_PLAN_STATUS=NOT_DISCLOSED" in captured["prompt"]


@pytest.mark.asyncio
async def test_research_manager_omits_fla_when_absent(monkeypatch):
    """Sanity: when FLA is N/A, the block must not appear (no spurious heading)."""
    from src.agents import research_nodes

    captured: dict[str, str] = {}

    async def fake_invoke(_llm, messages, **_kwargs):
        captured["prompt"] = str(messages[0].content)
        return AIMessage(content="plan")

    monkeypatch.setattr(
        research_nodes.agent_runtime, "invoke_with_rate_limit_handling", fake_invoke
    )

    manager = research_nodes.create_research_manager_node(Mock(), None)
    state = _state_with_card()
    state["foreign_language_report"] = "N/A"

    await manager(state, {"configurable": {}})

    assert "FOREIGN LANGUAGE / LOCAL FILINGS" not in captured["prompt"]
