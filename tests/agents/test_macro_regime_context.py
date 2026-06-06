from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

_MACRO_REPORT = """### RATES & LIQUIDITY
- Signal: BEARISH
- Direction: worsening
- Summary: Oil shock is tightening liquidity.

### REGIME SUMMARY
- Regional equities face higher discount-rate and earnings-pressure risk.

MACRO_REGIME_BLOCK:
RISK_APPETITE: RISK_OFF
SHOCK_TYPE: ENERGY
SHOCK_PHASE: ACUTE
EQUITY_TRANSMISSION: EARNINGS_PRESSURE
DIP_POSTURE: WAIT_FOR_CONFIRMATION
CONFIDENCE: MEDIUM
"""


def _context(report: str = _MACRO_REPORT, status: str = "generated"):
    return SimpleNamespace(
        ticker="7203.T",
        trade_date="2026-06-05",
        macro_context_report=report,
        macro_context_region="JAPAN",
        macro_context_status=status,
    )


def _config(report: str = _MACRO_REPORT, status: str = "generated"):
    return {"configurable": {"context": _context(report, status)}}


def _fundamentals_block() -> str:
    return (
        "### --- START DATA_BLOCK ---\n"
        "SECTOR: Consumer Discretionary\n"
        "ADJUSTED_HEALTH_SCORE: 72%\n"
        "ADJUSTED_GROWTH_SCORE: 68%\n"
        "PFIC_RISK: LOW\n"
        "PE_RATIO_TTM: 11.2\n"
        "PEG_RATIO: 0.8\n"
        "### --- END DATA_BLOCK ---"
    )


def _decision_state() -> dict:
    return {
        "company_of_interest": "7203.T",
        "company_name": "Toyota Motor",
        "company_name_resolved": True,
        "market_report": "Market report",
        "sentiment_report": "Sentiment report",
        "news_report": "News report",
        "fundamentals_report": _fundamentals_block(),
        "value_trap_report": "",
        "investment_plan": "Research plan",
        "consultant_review": "",
        "apac_regional_report": "",
        "auditor_report": "",
        "valuation_params": "",
        "trader_investment_plan": "TRADE_BLOCK:\nACTION: BUY",
        "risk_debate_state": {
            "current_risky_response": "Risky view",
            "current_safe_response": "Safe view",
            "current_neutral_response": "Neutral view",
        },
        "red_flags": [],
        "pre_screening_result": "PASS",
        "artifact_statuses": {},
        "messages": [],
    }


def test_format_macro_context_for_news_and_decision():
    from src.agents.support import format_macro_context_for_agent

    news = format_macro_context_for_agent(_context(), audience="news")
    assert "### REGIONAL MACRO CONTEXT" in news
    assert "Region: JAPAN" in news
    assert "### RATES & LIQUIDITY" in news

    decision = format_macro_context_for_agent(_config(), audience="decision")
    assert "### MACRO REGIME SIGNAL" in decision
    assert "### REGIME SUMMARY" in decision
    assert "MACRO_REGIME_BLOCK:" in decision
    assert "DIP_POSTURE: WAIT_FOR_CONFIRMATION" in decision
    assert "cannot override hard fails" not in decision


def test_news_context_does_not_include_decision_policy():
    from src.agents.support import format_macro_context_for_agent

    news = format_macro_context_for_agent(_context(), audience="news")

    assert "### REGIONAL MACRO CONTEXT" in news
    assert "cannot override hard fails" not in news
    assert "bounded pre-mortem context" not in news


def test_format_macro_context_accepts_cached_fallback_and_case_insensitive_audience():
    from src.agents.support import format_macro_context_for_agent

    rendered = format_macro_context_for_agent(
        _context(status="generated_fallback"),
        audience="Decision",
    )

    assert "### MACRO REGIME SIGNAL" in rendered
    assert "SHOCK_PHASE: ACUTE" in rendered


def test_format_macro_context_skips_absent_failed_and_unstructured_reports():
    from src.agents.support import format_macro_context_for_agent

    assert format_macro_context_for_agent(None, audience="decision") == ""
    assert format_macro_context_for_agent(_context(status="failed")) == ""
    assert format_macro_context_for_agent(_context(status="disabled")) == ""
    assert (
        format_macro_context_for_agent(
            _context(report="### REGIME SUMMARY\n- Unstructured only."),
            audience="decision",
        )
        == ""
    )


def test_format_macro_context_handles_trailing_heading_after_regime_block():
    from src.agents.support import format_macro_context_for_agent

    rendered = format_macro_context_for_agent(
        _context(report=f"{_MACRO_REPORT}\n\n### NOTES\n- extra"),
        audience="decision",
    )

    assert "CONFIDENCE: MEDIUM" in rendered
    assert "### NOTES" not in rendered


def test_format_macro_context_caps_oversized_report():
    from src.agents.support import format_macro_context_for_agent

    report = _MACRO_REPORT + ("\nEXTRA: " + ("x" * 5000))
    rendered = format_macro_context_for_agent(
        _context(report=report),
        audience="decision",
        max_chars=300,
    )

    assert len(rendered) <= 312
    assert "[TRUNCATED]" in rendered


@pytest.mark.asyncio
async def test_bear_researcher_receives_macro_regime_signal_round1_and_round2(
    monkeypatch,
):
    from src.agents import research_nodes

    captured: list[str] = []

    async def fake_invoke(_llm, messages, **_kwargs):
        captured.append(messages[0].content)
        return AIMessage(content="Bear case")

    monkeypatch.setattr(
        research_nodes.agent_runtime,
        "invoke_with_rate_limit_handling",
        fake_invoke,
    )
    monkeypatch.setattr(research_nodes, "log_output_diagnostics", lambda **_: None)
    monkeypatch.setattr(research_nodes, "log_truncation_diagnostic", lambda **_: None)

    for round_num in (1, 2):
        node = research_nodes.create_researcher_node(
            MagicMock(),
            None,
            "bear_researcher",
            round_num=round_num,
        )
        await node(_decision_state(), _config())

    assert len(captured) == 2
    assert all("### MACRO REGIME SIGNAL" in prompt for prompt in captured)
    assert all("## MACRO REGIME SIGNAL POLICY" in prompt for prompt in captured)


@pytest.mark.asyncio
async def test_bull_researcher_does_not_receive_macro_regime_signal(monkeypatch):
    from src.agents import research_nodes

    captured: dict[str, str] = {}

    async def fake_invoke(_llm, messages, **_kwargs):
        captured["prompt"] = messages[0].content
        return AIMessage(content="Bull case")

    monkeypatch.setattr(
        research_nodes.agent_runtime,
        "invoke_with_rate_limit_handling",
        fake_invoke,
    )
    monkeypatch.setattr(research_nodes, "log_output_diagnostics", lambda **_: None)
    monkeypatch.setattr(research_nodes, "log_truncation_diagnostic", lambda **_: None)

    node = research_nodes.create_researcher_node(
        MagicMock(),
        None,
        "bull_researcher",
        round_num=1,
    )
    await node(_decision_state(), _config())

    assert "### MACRO REGIME SIGNAL" not in captured["prompt"]


@pytest.mark.asyncio
async def test_decision_nodes_receive_macro_regime_signal(monkeypatch):
    from src.agents import decision_nodes

    prompts: list[str] = []

    async def fake_invoke(_llm, messages, **_kwargs):
        prompts.append(messages[0].content)
        return AIMessage(
            content=(
                "### PORTFOLIO MANAGER VERDICT: BUY\n"
                "### --- START PM_BLOCK ---\n"
                "VERDICT: BUY\n"
                "HEALTH_ADJ: 72\n"
                "GROWTH_ADJ: 68\n"
                "RISK_TALLY: 0.5\n"
                "ZONE: LOW\n"
                "SHOW_VALUATION_CHART: YES\n"
                "VALUATION_DISCOUNT: 1.0\n"
                "POSITION_SIZE: 2.0\n"
                "VALUATION_CONTEXT: STANDARD\n"
                "### --- END PM_BLOCK ---"
            )
        )

    monkeypatch.setattr(
        decision_nodes.agent_runtime,
        "invoke_with_rate_limit_handling",
        fake_invoke,
    )
    monkeypatch.setattr(decision_nodes, "log_output_diagnostics", lambda **_: None)
    monkeypatch.setattr(decision_nodes, "log_truncation_diagnostic", lambda **_: None)

    trader = decision_nodes.create_trader_node(MagicMock(), None)
    risk = decision_nodes.create_risk_debater_node(MagicMock(), "safe_analyst")
    pm = decision_nodes.create_portfolio_manager_node(MagicMock(), None)

    state = _decision_state()
    await trader(state, _config())
    await risk(state, _config())
    await pm(state, _config())

    assert len(prompts) == 3
    assert all("### MACRO REGIME SIGNAL" in prompt for prompt in prompts)
    assert all("DIP_POSTURE: WAIT_FOR_CONFIRMATION" in prompt for prompt in prompts)
    assert all("## MACRO REGIME SIGNAL POLICY" in prompt for prompt in prompts)

    trader_prompt, _risk_prompt, pm_prompt = prompts
    assert trader_prompt.index("MACRO REGIME SIGNAL") < trader_prompt.index(
        "APAC REGIONAL SPECIALIST"
    )
    assert pm_prompt.index("MACRO REGIME SIGNAL") < pm_prompt.index(
        "RESEARCH MANAGER RECOMMENDATION"
    )
    assert pm_prompt.index("MACRO REGIME SIGNAL") < pm_prompt.rindex(
        "RISK TEAM DEBATE:"
    )
