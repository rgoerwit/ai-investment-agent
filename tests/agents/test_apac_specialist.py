from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage

from src.agents.apac_specialist_node import (
    APAC_NO_MATERIAL_SENTINEL,
    APAC_REPORT_FIELD,
    APAC_UNAVAILABLE_SENTINEL,
    build_apac_specialist_payload,
    create_apac_specialist_node,
)


def test_payload_minimization_excludes_forbidden_fields():
    state = {
        "company_of_interest": "7203.T",
        "company_name": "Toyota Motor",
        "trade_date": "2026-05-17",
        "investment_plan": "Research Manager synthesis",
        "fundamentals_report": "DATA_BLOCK",
        "foreign_language_report": "foreign report",
        "value_trap_report": "value trap",
        "news_report": "news",
        "sentiment_report": "sentiment",
        "red_flags": [{"type": "LOCAL_COVERAGE_HIGH"}],
        "entity_governance_card": {
            "ticker": "7203.T",
            "canonical_name": "Toyota Motor Corporation",
            "entity_role": "STANDALONE",
            "confidence": "clean",
        },
        "messages": ["raw tool transcript with secrets"],
        "past_insights": "memory lesson",
        "retrospective_snapshots": ["prior run"],
    }

    payload = build_apac_specialist_payload(state)
    blob = json.dumps(payload, ensure_ascii=False)

    assert payload["ticker"] == "7203.T"
    assert "ENTITY GOVERNANCE CARD" in payload["entity_governance_card"]
    assert "Toyota Motor Corporation" in payload["entity_governance_card"]
    assert "raw tool transcript" not in blob
    assert "memory lesson" not in blob
    assert "prior run" not in blob


@pytest.mark.asyncio
async def test_apac_ticker_invokes_llm_and_records_prompt(monkeypatch):
    async def fake_invoke(*args, **kwargs):
        return AIMessage(
            content=(
                "### APAC REGIONAL AUDIT: 7203.T\n"
                "**DATA GAPS / UNVERIFIED CLAIMS**: None\n"
                "**VERDICT FOR CONSULTANT AND PM**: SUPPORT - APAC context is consistent."
            )
        )

    monkeypatch.setattr(
        "src.agents.apac_specialist_node.agent_runtime.invoke_with_rate_limit_handling",
        fake_invoke,
    )
    node = create_apac_specialist_node(Mock())

    out = await node(
        {"company_of_interest": "7203.T", "company_name": "Toyota Motor"},
        {"configurable": {}},
    )

    assert "APAC REGIONAL AUDIT" in out[APAC_REPORT_FIELD]
    assert out["prompts_used"][APAC_REPORT_FIELD]["agent_name"] == (
        "APAC Regional Specialist"
    )


@pytest.mark.asyncio
async def test_non_apac_ticker_invokes_llm_and_preserves_silence_verdict(monkeypatch):
    """Silence is the prompt's responsibility, not a deterministic prefilter.

    A non-APAC-listed ticker (e.g. AAPL) may still have APAC supply-chain
    exposure. The specialist must be invoked and the silence verdict, if
    returned, preserved verbatim without being passed through cap_state_value.
    """

    invoke = AsyncMock(return_value=AIMessage(content=APAC_NO_MATERIAL_SENTINEL))
    monkeypatch.setattr(
        "src.agents.apac_specialist_node.agent_runtime.invoke_with_rate_limit_handling",
        invoke,
    )
    node = create_apac_specialist_node(Mock())

    out = await node({"company_of_interest": "AAPL"}, {"configurable": {}})

    invoke.assert_awaited_once()
    assert out[APAC_REPORT_FIELD] == APAC_NO_MATERIAL_SENTINEL
    assert out["artifact_statuses"][APAC_REPORT_FIELD]["ok"] is True


@pytest.mark.asyncio
async def test_apac_llm_failure_degrades_to_failure_artifact(monkeypatch):
    monkeypatch.setattr(
        "src.agents.apac_specialist_node.agent_runtime.invoke_with_rate_limit_handling",
        AsyncMock(side_effect=TimeoutError("deepseek timeout")),
    )
    node = create_apac_specialist_node(Mock())

    out = await node({"company_of_interest": "7203.T"}, {"configurable": {}})

    assert out[APAC_REPORT_FIELD] == APAC_UNAVAILABLE_SENTINEL
    status = out["artifact_statuses"][APAC_REPORT_FIELD]
    assert status["complete"] is True
    assert status["ok"] is False


@pytest.mark.asyncio
async def test_glm_1301_retries_once_without_thinking(monkeypatch):
    invoke = AsyncMock(
        side_effect=[
            RuntimeError("Error code 400: {'code': '1301', 'message': '敏感内容'}"),
            AIMessage(content=APAC_NO_MATERIAL_SENTINEL),
        ]
    )
    monkeypatch.setattr(
        "src.agents.apac_specialist_node.agent_runtime.invoke_with_rate_limit_handling",
        invoke,
    )
    node = create_apac_specialist_node(
        Mock(name="thinking"), fallback_llm=Mock(name="direct")
    )

    out = await node({"company_of_interest": "AGS.SI"}, {"configurable": {}})

    assert invoke.await_count == 2
    assert invoke.await_args_list[1].kwargs["max_attempts"] == 1
    assert out[APAC_REPORT_FIELD] == APAC_NO_MATERIAL_SENTINEL


@pytest.mark.asyncio
async def test_non_1301_policy_error_does_not_retry(monkeypatch):
    invoke = AsyncMock(side_effect=RuntimeError("Error code 400: invalid parameter"))
    monkeypatch.setattr(
        "src.agents.apac_specialist_node.agent_runtime.invoke_with_rate_limit_handling",
        invoke,
    )
    node = create_apac_specialist_node(Mock(), fallback_llm=Mock())

    out = await node({"company_of_interest": "AGS.SI"}, {"configurable": {}})

    assert invoke.await_count == 1
    assert out[APAC_REPORT_FIELD] == APAC_UNAVAILABLE_SENTINEL


@pytest.mark.asyncio
async def test_second_glm_1301_failure_is_not_retried_again(monkeypatch):
    error = RuntimeError("Error code 400: {'code': '1301', 'message': '敏感内容'}")
    invoke = AsyncMock(side_effect=[error, error])
    monkeypatch.setattr(
        "src.agents.apac_specialist_node.agent_runtime.invoke_with_rate_limit_handling",
        invoke,
    )
    node = create_apac_specialist_node(Mock(), fallback_llm=Mock())

    out = await node({"company_of_interest": "AGS.SI"}, {"configurable": {}})

    assert invoke.await_count == 2
    assert out[APAC_REPORT_FIELD] == APAC_UNAVAILABLE_SENTINEL


@pytest.mark.asyncio
async def test_non_glm_content_block_also_retries_without_thinking(monkeypatch):
    """The re-issue is deliberately vendor-agnostic (Aug 2026).

    Its hypothesis — the block fired on the reasoning stream, so the same request
    with thinking disabled may pass — is a property of reasoning models, not of
    GLM. This seat talks to whichever single OpenAI-compatible vendor
    APAC_SPECIALIST_BASE_URL names, and that defaults to DeepSeek, so a GLM-only
    predicate left the default vendor with no re-issue for arbitrary reasons.
    """
    invoke = AsyncMock(
        side_effect=[
            RuntimeError(
                "Error code: 400 - {'error': {'message': 'Content Exists Risk'}}"
            ),
            AIMessage(content=APAC_NO_MATERIAL_SENTINEL),
        ]
    )
    monkeypatch.setattr(
        "src.agents.apac_specialist_node.agent_runtime.invoke_with_rate_limit_handling",
        invoke,
    )
    node = create_apac_specialist_node(Mock(), fallback_llm=Mock())

    out = await node({"company_of_interest": "AGS.SI"}, {"configurable": {}})

    assert invoke.await_count == 2
    assert out[APAC_REPORT_FIELD] == APAC_NO_MATERIAL_SENTINEL


@pytest.mark.asyncio
async def test_retry_telemetry_does_not_assert_a_vendor_code(monkeypatch):
    """The event used to hardcode provider_code=1301, which became a lie the
    moment the branch stopped being GLM-only."""
    import src.agents.apac_specialist_node as mod

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        mod.logger, "warning", lambda evt, **kw: events.append((evt, kw))
    )
    invoke = AsyncMock(
        side_effect=[
            RuntimeError(
                "Error code: 400 - {'error': {'code': 'data_inspection_failed'}}"
            ),
            AIMessage(content=APAC_NO_MATERIAL_SENTINEL),
        ]
    )
    monkeypatch.setattr(
        "src.agents.apac_specialist_node.agent_runtime.invoke_with_rate_limit_handling",
        invoke,
    )
    node = create_apac_specialist_node(Mock(), fallback_llm=Mock())

    await node({"company_of_interest": "AGS.SI"}, {"configurable": {}})

    retry = [kw for evt, kw in events if evt == "apac_policy_block_direct_retry"]
    assert retry, "the re-issue must still be logged"
    assert "provider_code" not in retry[0]
    assert "payload_sha256" in retry[0]  # payload stays hashed, never logged


def test_specialist_prompt_loads_with_required_terms():
    from src.prompts import get_prompt

    prompt = get_prompt("apac_regional_specialist")
    assert prompt.agent_name == "APAC Regional Specialist"
    for term in (
        "NO_MATERIAL_APAC_CONNECTION",
        "UNVERIFIED",
        "系列",
        "재벌",
        "关联交易",
        "Trung Quốc + 1",
        "Bumiputera",
        "PLI",
    ):
        assert term in prompt.system_message


def test_consultant_prompt_treats_unverified_apac_as_question():
    from src.prompts import get_prompt

    prompt = get_prompt("consultant")
    assert "APAC Regional Specialist" in prompt.system_message
    assert "UNVERIFIED APAC claims as questions to resolve" in prompt.system_message


def test_portfolio_manager_prompt_treats_apac_silence_as_neutral():
    from src.prompts import get_prompt

    prompt = get_prompt("portfolio_manager")
    assert "NO_MATERIAL_APAC_CONNECTION" in prompt.system_message
    assert "neutral with no risk-tally or conviction impact" in prompt.system_message


def test_apac_llm_factory_gates_quick_disabled_and_missing_key(monkeypatch):
    from src import llms

    monkeypatch.setattr(llms.config, "enable_apac_specialist", False)
    assert llms.create_apac_specialist_llm() is None

    monkeypatch.setattr(llms.config, "enable_apac_specialist", True)
    assert llms.create_apac_specialist_llm(quick_mode=True) is None

    monkeypatch.setattr(
        type(llms.config),
        "get_apac_specialist_api_key",
        lambda self: "",
    )
    assert llms.create_apac_specialist_llm() is None


def test_apac_llm_factory_passes_deepseek_kwargs(monkeypatch):
    from src import llms

    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = ModuleType("langchain_openai")
    fake_module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    monkeypatch.setattr(llms.config, "enable_apac_specialist", True)
    monkeypatch.setattr(
        type(llms.config),
        "get_apac_specialist_api_key",
        lambda self: "sk-test",
    )
    monkeypatch.setattr(llms.config, "apac_specialist_model", "deepseek-v4-pro")
    monkeypatch.setattr(
        llms.config, "apac_specialist_base_url", "https://api.deepseek.com"
    )

    llm = llms.create_apac_specialist_llm(max_completion_tokens=2048)

    assert isinstance(llm, FakeChatOpenAI)
    assert captured["model"] == "deepseek-v4-pro"
    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["reasoning_effort"] == "max"
    assert captured["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "temperature" not in captured


def test_apac_direct_retry_factory_disables_thinking_and_sdk_retries(monkeypatch):
    from src import llms

    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = ModuleType("langchain_openai")
    fake_module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    monkeypatch.setattr(llms.config, "enable_apac_specialist", True)
    monkeypatch.setattr(
        type(llms.config), "get_apac_specialist_api_key", lambda self: "sk-test"
    )

    llms.create_apac_specialist_llm(thinking_enabled=False)

    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured["max_retries"] == 0
    assert "reasoning_effort" not in captured
