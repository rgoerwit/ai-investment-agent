"""Regression guards for 1308.HK-driven prompt alignment fixes.

These tests assert that the prompt policies added for the 1308.HK analysis
remain present in the live prompt registry loaded from JSON overrides.
"""

from __future__ import annotations

from src.prompts import get_prompt


class TestFundamentalsPromptAlignment:
    def test_fundamentals_prompt_tracks_ocf_reason_and_growth_guard(self):
        prompt = get_prompt("fundamentals_analyst")
        assert prompt is not None
        major, minor = map(int, prompt.version.split(".")[:2])
        assert (major, minor) >= (9, 9)
        assert "OCF_FILING_REASON" in prompt.system_message
        assert "EARNINGS_GROWTH_TTM < -5%" in prompt.system_message
        assert "acceleration bonus suppressed" in prompt.system_message.lower()


class TestResearchPromptAlignment:
    def test_bull_prompt_allows_unsponsored_adr_and_uses_consultant(self):
        prompt = get_prompt("bull_researcher")
        assert prompt is not None
        assert "Unsponsored ADRs are acceptable" in prompt.system_message
        assert "CONSULTANT INTEGRATION" in prompt.system_message

    def test_bear_prompt_treats_unsponsored_adr_as_nonfatal(self):
        prompt = get_prompt("bear_researcher")
        assert prompt is not None
        assert (
            "Unsponsored ADRs are not an automatic violation" in prompt.system_message
        )
        assert "Sponsored ADR exists on NYSE/NASDAQ/OTC" in prompt.system_message
        assert "CONSULTANT INTEGRATION" in prompt.system_message


class TestRiskPromptAlignment:
    def test_risky_prompt_has_consultant_guardrails(self):
        prompt = get_prompt("risky_analyst")
        assert prompt is not None
        assert "CONSULTANT INTEGRATION" in prompt.system_message
        assert "avoid 8%+ sizing" in prompt.system_message

    def test_safe_prompt_has_consultant_guardrails(self):
        prompt = get_prompt("safe_analyst")
        assert prompt is not None
        assert "CONSULTANT INTEGRATION" in prompt.system_message
        assert "tool-coverage gaps" in prompt.system_message

    def test_neutral_prompt_has_consultant_guardrails(self):
        prompt = get_prompt("neutral_analyst")
        assert prompt is not None
        assert "CONSULTANT INTEGRATION" in prompt.system_message
        assert "consultant-challenged qualitative claims" in prompt.system_message


class TestPortfolioManagerPromptAlignment:
    def test_pm_prompt_has_independence_rule(self):
        prompt = get_prompt("portfolio_manager")
        assert prompt is not None
        assert "INDEPENDENCE RULE" in prompt.system_message
        assert (
            "Another in-system analyst's narrative is not independent verification"
            in prompt.system_message
        )
        assert "structured tool evidence" in prompt.system_message

    def test_pm_prompt_addresses_local_coverage_and_low_us_exposure(self):
        prompt = get_prompt("portfolio_manager")
        assert prompt is not None
        assert "LOCAL_COVERAGE_HIGH" in prompt.system_message
        assert (
            "Not disclosed but business model/geographic mix clearly indicates intra-regional or domestic exposure"
            in prompt.system_message
        )


class TestNewsPromptAlignment:
    def test_news_prompt_blocks_fleet_to_newbuild_inflation(self):
        prompt = get_prompt("news_analyst")
        assert prompt is not None
        assert "Do NOT convert fleet size" in prompt.system_message
        assert "Operates 100 vessels" in prompt.system_message


class TestValueTrapPromptAlignment:
    def test_value_trap_prompt_limits_insider_language(self):
        prompt = get_prompt("value_trap_detector")
        assert prompt is not None
        major, minor = map(int, prompt.version.split(".")[:2])
        assert (major, minor) >= (1, 5)
        assert "NET_SELLER" in prompt.system_message
        assert "cluster selling by CEO/CFO/Chairman" in prompt.system_message
        assert "named executives and dates" in prompt.system_message
