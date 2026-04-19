"""Regression guards for prompt content with behavior-critical instructions.

These tests fail the moment someone accidentally removes or corrupts the
specific prompt instructions that underpin deterministic review assumptions.
No LLM is called; only the prompt JSON on disk is inspected.
"""

from __future__ import annotations

from src.prompts import get_prompt


class TestResearchManagerPromptContent:
    """Guard the TRANSIENT/STRUCTURAL risk-duration instruction in research_manager."""

    def test_version_is_5_2_or_higher(self):
        """Prompt version must be ≥5.2 (the version that introduced TRANSIENT tags).

        Bump the assertion when the prompt is intentionally revised to a new version.
        """
        prompt = get_prompt("research_manager")
        assert prompt is not None, "research_manager prompt not found in registry"
        major, minor = map(int, prompt.version.split(".")[:2])
        assert (major, minor) >= (5, 2), f"Expected ≥5.2, got {prompt.version}"

    def test_transient_duration_tag_present(self):
        """STEP 4 must instruct the LLM to classify risks as [TRANSIENT ...]."""
        prompt = get_prompt("research_manager")
        assert "[TRANSIENT" in prompt.system_message, (
            "TRANSIENT risk-duration tag missing from research_manager prompt. "
            "The LLM needs this to classify short-lived macro risks at 0.5× weight."
        )

    def test_half_weight_instruction_present(self):
        """The 0.5× tally reduction for TRANSIENT risks must be stated explicitly."""
        prompt = get_prompt("research_manager")
        assert "0.5" in prompt.system_message, (
            "0.5× weight instruction missing from research_manager prompt. "
            "Without it the LLM cannot downgrade transient macro risks."
        )

    def test_structural_tag_present(self):
        """STRUCTURAL tag must be present alongside TRANSIENT for contrast."""
        prompt = get_prompt("research_manager")
        assert (
            "STRUCTURAL" in prompt.system_message
        ), "STRUCTURAL risk-duration tag missing from research_manager prompt."

    def test_geopolitical_example_present(self):
        """Concrete geopolitical example helps the LLM classify risks correctly."""
        prompt = get_prompt("research_manager")
        assert "eopolitical" in prompt.system_message, (
            "Geopolitical example missing from research_manager prompt. "
            "Examples anchor the LLM's classification of TRANSIENT vs STRUCTURAL."
        )


class TestFundamentalsPromptContent:
    """Guard prompt rules for multi-horizon growth fidelity."""

    def test_fundamentals_prompt_forbids_fy_to_ttm_copying(self):
        prompt = get_prompt("fundamentals_analyst")
        assert (
            "do not copy FY values into TTM or MRQ fields" in prompt.system_message
        ), "Fundamentals prompt must forbid copying FY growth into TTM/MRQ labels."

    def test_fundamentals_prompt_mentions_event_driven_normalization(self):
        prompt = get_prompt("fundamentals_analyst")
        assert "event-driven normalization" in prompt.system_message, (
            "Fundamentals prompt must distinguish named one-time-event normalization "
            "from generic cyclical decline."
        )

    def test_fundamentals_prompt_has_idle_cash_fields(self):
        prompt = get_prompt("fundamentals_analyst")
        msg = prompt.system_message
        assert "NET_CASH_TO_MARKET_CAP" in msg
        assert "CAPITAL_PLAN_STATUS" in msg


class TestPortfolioManagerPromptContent:
    """Guard PM handling of consultant no-coverage cases."""

    def test_portfolio_manager_prompt_makes_no_coverage_neutral(self):
        prompt = get_prompt("portfolio_manager")
        assert (
            "should be noted without adding +0.25" in prompt.system_message
        ), "PM prompt must keep plain ex-US no-coverage consultant cases neutral."

    def test_portfolio_manager_prompt_mentions_idle_cash_leniency(self):
        prompt = get_prompt("portfolio_manager")
        assert (
            "CAPITAL_PLAN_STATUS" in prompt.system_message
        ), "PM prompt must distinguish idle cash with no plan from justified cash buffers."


class TestFundamentalsEbitdaAnnualization:
    """Guard EBITDA annualization rule (v9.10+).

    Prevents partial-period EBITDA bug: when only H1 filing EBITDA is available,
    the LLM was using it raw as the denominator, doubling apparent leverage
    (observed on RIC.AX post-IPF acquisition — 5.1× reported vs ~2.5–3.5× actual).
    """

    def test_version_is_9_10_or_higher(self):
        prompt = get_prompt("fundamentals_analyst")
        major, minor = map(int, prompt.version.split(".")[:2])
        assert (major, minor) >= (9, 10), f"Expected ≥9.10, got {prompt.version}"

    def test_ebitda_annualization_rule_present(self):
        prompt = get_prompt("fundamentals_analyst")
        assert "annualize" in prompt.system_message, (
            "Fundamentals prompt must instruct LLM to annualize partial-period EBITDA. "
            "Without this, H1 EBITDA is used raw, doubling apparent leverage (RIC.AX bug)."
        )

    def test_net_debt_ebitda_period_field_in_datablock(self):
        prompt = get_prompt("fundamentals_analyst")
        assert (
            "NET_DEBT_EBITDA_PERIOD" in prompt.system_message
        ), "DATA_BLOCK must include NET_DEBT_EBITDA_PERIOD for audit trail of annualization."

    def test_net_debt_ebitda_field_in_datablock(self):
        prompt = get_prompt("fundamentals_analyst")
        assert (
            "NET_DEBT_EBITDA:" in prompt.system_message
        ), "DATA_BLOCK must include NET_DEBT_EBITDA field."


class TestFundamentalsRevenueBacklog:
    """Guard revenue backlog scoring rule (v9.10+).

    Prevents order-book blindness: project-based businesses (construction,
    infrastructure) with large contracted backlogs were scoring Growth:33
    because the rubric had no pathway to credit forward revenue visibility.
    Observed on BEC.SI (BRC Asia, S$2.2B order book, ~1.4× trailing revenue).
    """

    def test_revenue_backlog_coverage_field_in_datablock(self):
        prompt = get_prompt("fundamentals_analyst")
        assert "REVENUE_BACKLOG_COVERAGE" in prompt.system_message, (
            "DATA_BLOCK must include REVENUE_BACKLOG_COVERAGE. "
            "Without it, contracted order books cannot influence growth scoring (BEC.SI bug)."
        )

    def test_revenue_backlog_field_in_datablock(self):
        prompt = get_prompt("fundamentals_analyst")
        assert (
            "REVENUE_BACKLOG:" in prompt.system_message
        ), "DATA_BLOCK must include REVENUE_BACKLOG field."

    def test_backlog_coverage_credits_expansion_point(self):
        """Backlog ≥1.0× trailing revenue must appear adjacent to '1 pt' in the rubric."""
        prompt = get_prompt("fundamentals_analyst")
        msg = prompt.system_message
        backlog_idx = msg.find("REVENUE_BACKLOG_COVERAGE ≥")
        assert (
            backlog_idx != -1
        ), "REVENUE_BACKLOG_COVERAGE threshold must appear in growth scoring rubric."
        nearby = msg[backlog_idx : backlog_idx + 60]
        assert (
            "1 pt" in nearby
        ), "Revenue backlog criterion must award 1 pt when coverage ≥1.0× trailing revenue."


class TestForeignLanguageOrderBook:
    """Guard order book search in Foreign Language Analyst (v1.5+).

    Without Search E, the FLA never looks for revenue backlog data, so it
    can never reach the DATA_BLOCK or influence growth scoring.
    """

    def test_version_is_1_5_or_higher(self):
        prompt = get_prompt("foreign_language_analyst")
        major, minor = map(int, prompt.version.split(".")[:2])
        assert (major, minor) >= (1, 5), f"Expected ≥1.5, got {prompt.version}"

    def test_search_e_order_book_present(self):
        prompt = get_prompt("foreign_language_analyst")
        assert (
            "order book" in prompt.system_message.lower()
        ), "Foreign Language Analyst must include Search E for revenue backlog/order book data."

    def test_revenue_backlog_output_block_present(self):
        prompt = get_prompt("foreign_language_analyst")
        assert (
            "REVENUE BACKLOG" in prompt.system_message
        ), "FLA output format must include REVENUE BACKLOG section to pass data downstream."

    def test_capital_policy_output_block_present(self):
        prompt = get_prompt("foreign_language_analyst")
        assert (
            "CAPITAL POLICY" in prompt.system_message
        ), "FLA output format must include CAPITAL POLICY so capital-allocation evidence reaches Fundamentals."

    def test_ownership_change_search_present(self):
        prompt = get_prompt("foreign_language_analyst")
        msg = prompt.system_message
        assert "Search G: Ownership Changes / Insider Dealings" in msg
        assert "director dealings" in msg
        assert "股權披露" in msg
        assert "大量保有報告書" in msg
        assert "PDMR" in msg

    def test_ownership_change_output_block_present(self):
        prompt = get_prompt("foreign_language_analyst")
        msg = prompt.system_message
        assert "Recent Ownership Changes" in msg
        assert "Insider/Director Dealings" in msg
