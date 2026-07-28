from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langgraph.types import RunnableConfig

from src.agents import consultant_nodes, decision_nodes
from src.data.metric_extraction import calculate_derived_metrics
from src.retrospective import extract_snapshot

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "analysis_quality"
    / "b3sa3_quality_state.json"
)


def _load_b3_state() -> dict:
    return json.loads(FIXTURE.read_text())


async def _capture_node_prompt(node, state: dict) -> str:
    captured: dict[str, str] = {}

    async def fake_invoke(_llm, messages, **_kwargs):
        captured["prompt"] = str(messages[0].content)
        return SimpleNamespace(content="CONSULTANT REVIEW: APPROVED")

    with patch("src.agents.runtime.invoke_with_rate_limit_handling", new=fake_invoke):
        with patch("src.prompts.get_prompt") as mock_get_prompt:
            mock_get_prompt.return_value = SimpleNamespace(
                system_message="test prompt",
                agent_name="Test Agent",
            )
            await node(
                state,
                RunnableConfig(
                    configurable={"context": SimpleNamespace(trade_date="2026-06-13")}
                ),
            )
    return captured["prompt"]


def test_b3_quality_fixture_preserves_sector_relative_metrics() -> None:
    state = _load_b3_state()

    metrics = calculate_derived_metrics(state["raw_metrics"], "B3SA3.SA")

    assert metrics["sectorMedianPE"] == pytest.approx(12.0)
    assert metrics["peVsSector"] == pytest.approx(1.38)


@pytest.mark.asyncio
async def test_b3_quality_fixture_preserves_consultant_decision_inputs() -> None:
    state = _load_b3_state()

    prompt = await _capture_node_prompt(
        consultant_nodes.create_consultant_node(
            SimpleNamespace(model_name="consultant-model"), "consultant"
        ),
        state,
    )

    assert "FOREIGN-LANGUAGE / NATIVE-SOURCE ANALYST" in prompt
    assert "FOREIGN LANGUAGE REPORT" in prompt
    assert "VALUE TRAP DETECTOR" in prompt
    assert "VALUE TRAP REPORT" in prompt


@pytest.mark.asyncio
async def test_b3_quality_fixture_uses_canonical_pm_decision_inputs() -> None:
    state = _load_b3_state()

    prompt = await _capture_node_prompt(
        decision_nodes.create_portfolio_manager_node(
            SimpleNamespace(model_name="pm-model"), None
        ),
        state,
    )

    assert "FOREIGN LANGUAGE / NATIVE-SOURCE ANALYST REPORT" not in prompt
    assert "FOREIGN LANGUAGE REPORT" not in prompt
    assert "FUNDAMENTALS ANALYST REPORT" in prompt
    assert "DETERMINISTIC EVIDENCE CONSTRAINTS" in prompt


def test_b3_quality_fixture_uses_brazil_benchmark() -> None:
    snapshot = extract_snapshot({}, "B3SA3.SA")

    assert snapshot["benchmark_index"] == "^BVSP"
