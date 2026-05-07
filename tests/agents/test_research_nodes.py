from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.research_nodes import create_researcher_node


def _prompt():
    return SimpleNamespace(
        system_message="You are the debate agent.",
        agent_name="Bull Researcher",
        version="test",
    )


def _base_state(**overrides):
    state = {
        "company_of_interest": "AAPL",
        "company_name": "Apple Inc.",
        "company_name_resolved": True,
        "market_report": "Market report " * 40,
        "sentiment_report": "Sentiment report " * 40,
        "news_report": "News report " * 40,
        "fundamentals_report": "Fundamentals report " * 80,
        "investment_debate_state": {
            "bull_round1": "Bull Researcher (Round 1): Strong margins and cash flow.",
            "bear_round1": "Bear Researcher (Round 1): Valuation and competition risk.",
        },
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_round1_prompt_uses_full_reports_bundle():
    llm = MagicMock()
    captured = {}

    async def fake_invoke(runnable, messages, **kwargs):
        captured["prompt"] = messages[0].content
        return SimpleNamespace(content="Initial thesis")

    with (
        patch("src.prompts.get_prompt", return_value=_prompt()),
        patch(
            "src.agents.research_nodes.agent_runtime.invoke_with_rate_limit_handling",
            side_effect=fake_invoke,
        ),
        patch(
            "src.retrospective.format_lessons_for_injection",
            new=AsyncMock(return_value=""),
        ),
        patch("src.retrospective.create_lessons_memory", return_value=None),
        patch("src.agents.research_nodes.log_output_diagnostics"),
        patch("src.agents.research_nodes.log_truncation_diagnostic"),
    ):
        node = create_researcher_node(
            llm, memory=None, agent_key="bull_researcher", round_num=1
        )
        result = await node(_base_state(), {})

    assert "REPORTS:" in captured["prompt"]
    assert "FACTUAL ANCHORS:" not in captured["prompt"]
    assert "MARKET ANALYST REPORT:" in captured["prompt"]
    assert result["investment_debate_state"]["bull_round1"].endswith("Initial thesis")


@pytest.mark.asyncio
async def test_round2_prompt_uses_factual_anchors_not_full_reports():
    llm = MagicMock()
    captured = {}

    async def fake_invoke(runnable, messages, **kwargs):
        captured["prompt"] = messages[0].content
        return SimpleNamespace(content="Round 2 rebuttal")

    with (
        patch("src.prompts.get_prompt", return_value=_prompt()),
        patch(
            "src.agents.research_nodes.agent_runtime.invoke_with_rate_limit_handling",
            side_effect=fake_invoke,
        ),
        patch(
            "src.retrospective.format_lessons_for_injection",
            new=AsyncMock(return_value=""),
        ),
        patch("src.retrospective.create_lessons_memory", return_value=None),
        patch("src.agents.research_nodes.log_output_diagnostics"),
        patch("src.agents.research_nodes.log_truncation_diagnostic"),
    ):
        node = create_researcher_node(
            llm, memory=None, agent_key="bull_researcher", round_num=2
        )
        result = await node(_base_state(), {})

    prompt = captured["prompt"]
    assert "FACTUAL ANCHORS:" in prompt
    assert "\n\nREPORTS:\n" not in prompt
    assert "YOUR ROUND 1 ARGUMENT:" in prompt
    assert "OPPONENT'S ROUND 1 ARGUMENT (REBUT THIS):" in prompt
    assert "do not introduce unsupported new facts" in prompt.lower()
    assert result["investment_debate_state"]["bull_round2"].endswith("Round 2 rebuttal")


@pytest.mark.asyncio
async def test_round2_prompt_handles_missing_reports_and_unresolved_company():
    llm = MagicMock()
    captured = {}

    async def fake_invoke(runnable, messages, **kwargs):
        captured["prompt"] = messages[0].content
        return SimpleNamespace(content="Fallback rebuttal")

    with (
        patch("src.prompts.get_prompt", return_value=_prompt()),
        patch(
            "src.agents.research_nodes.agent_runtime.invoke_with_rate_limit_handling",
            side_effect=fake_invoke,
        ),
        patch(
            "src.retrospective.format_lessons_for_injection",
            new=AsyncMock(return_value=""),
        ),
        patch("src.retrospective.create_lessons_memory", return_value=None),
        patch("src.agents.research_nodes.log_output_diagnostics"),
        patch("src.agents.research_nodes.log_truncation_diagnostic"),
    ):
        node = create_researcher_node(
            llm, memory=None, agent_key="bear_researcher", round_num=2
        )
        state = _base_state(
            company_name="UNKNOWN",
            company_name_resolved=False,
            market_report="N/A",
            sentiment_report="N/A",
            news_report="N/A",
            fundamentals_report="N/A",
            investment_debate_state={"bear_round1": "", "bull_round1": ""},
        )
        await node(state, {})

    prompt = captured["prompt"]
    assert "FACTUAL ANCHORS:" in prompt
    assert prompt.count("N/A") >= 4
    assert "WARNING: Company name could not be verified" in prompt


@pytest.mark.asyncio
async def test_missing_prompt_returns_existing_error_shape():
    node = create_researcher_node(
        MagicMock(), memory=None, agent_key="bull_researcher", round_num=2
    )

    with patch("src.prompts.get_prompt", return_value=None):
        result = await node(_base_state(), {})

    field_state = result["investment_debate_state"]
    assert "Error - Missing prompt for bull_researcher." in field_state["bull_round2"]


@pytest.mark.asyncio
async def test_invocation_failure_keeps_existing_error_path():
    llm = MagicMock()

    with (
        patch("src.prompts.get_prompt", return_value=_prompt()),
        patch(
            "src.agents.research_nodes.agent_runtime.invoke_with_rate_limit_handling",
            side_effect=RuntimeError("network timeout"),
        ),
        patch(
            "src.retrospective.format_lessons_for_injection",
            new=AsyncMock(return_value=""),
        ),
        patch("src.retrospective.create_lessons_memory", return_value=None),
    ):
        node = create_researcher_node(
            llm, memory=None, agent_key="bull_researcher", round_num=2
        )
        result = await node(_base_state(), {})

    assert "[SYSTEM ERROR]" in result["investment_debate_state"]["bull_round2"]
    assert "network timeout" in result["investment_debate_state"]["bull_round2"]
