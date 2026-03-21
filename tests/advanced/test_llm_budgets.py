from src.llm_budgets import get_agent_output_budget


def test_agent_budgets_scale_from_base_cap():
    assert get_agent_output_budget("Sentiment Analyst", 32768) == 1024
    assert get_agent_output_budget("Portfolio Manager", 32768) == 16384
    assert get_agent_output_budget("Research Manager", 32768) == 8192


def test_agent_budgets_scale_when_base_cap_doubles():
    assert get_agent_output_budget("Sentiment Analyst", 65536) == 2048
    assert get_agent_output_budget("Portfolio Manager", 65536) == 32768
    assert get_agent_output_budget("Fundamentals Analyst", 65536) == 21846


def test_unknown_agents_default_to_global_base_cap():
    assert get_agent_output_budget("Retry Agent (Deep)", 32768) == 32768
