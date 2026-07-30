from fractions import Fraction

from src.llm_budgets import (
    AGENT_OUTPUT_BUDGET_FRACTIONS,
    get_agent_output_budget,
    get_generation_budget,
)


def test_agent_budgets_scale_from_base_cap():
    assert get_agent_output_budget("Sentiment Analyst", 32768) == 1024
    assert get_agent_output_budget("Portfolio Manager", 32768) == 16384
    assert get_agent_output_budget("Research Manager", 32768) == 8192


def test_foreign_language_analyst_budget_matches_news_analyst_tier():
    # FLA's prompt now spans 5-6 structured evidence blocks (segment breakdown,
    # ownership, filing cash flow, management guidance, capital structure,
    # R&D/capex backlog) — same scope tier as News Analyst/Trader/Consultant,
    # not the simpler single-block Legal Counsel/Value Trap Detector agents.
    assert get_agent_output_budget("Foreign Language Analyst", 32768) == 4096
    assert get_agent_output_budget(
        "Foreign Language Analyst", 32768
    ) == get_agent_output_budget("News Analyst", 32768)


def test_simple_single_block_agents_unaffected_by_fla_bump():
    assert AGENT_OUTPUT_BUDGET_FRACTIONS["Legal Counsel"] == Fraction(1, 16)
    assert AGENT_OUTPUT_BUDGET_FRACTIONS["Value Trap Detector"] == Fraction(1, 16)


def test_agent_budgets_scale_when_base_cap_doubles():
    assert get_agent_output_budget("Sentiment Analyst", 65536) == 2048
    assert get_agent_output_budget("Portfolio Manager", 65536) == 32768
    assert get_agent_output_budget("Fundamentals Analyst", 65536) == 21846


def test_unknown_agents_default_to_global_base_cap():
    assert get_agent_output_budget("Some Unbudgeted Agent", 32768) == 32768


def test_generation_budget_default_reserve():
    budget = get_generation_budget(
        intent_tokens=2048,
        reserve_class="default",
        reserve_enabled=True,
        default_reserve_tokens=2048,
        deep_reserve_tokens=8192,
    )

    assert budget.intent_tokens == 2048
    assert budget.reserve_tokens == 2048
    assert budget.api_cap_tokens == 4096


def test_generation_budget_deep_reserve():
    budget = get_generation_budget(
        intent_tokens=2048,
        reserve_class="deep",
        reserve_enabled=True,
        default_reserve_tokens=2048,
        deep_reserve_tokens=8192,
    )

    assert budget.intent_tokens == 2048
    assert budget.reserve_tokens == 8192
    assert budget.api_cap_tokens == 10240
