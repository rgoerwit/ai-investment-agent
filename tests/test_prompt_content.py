"""Regression guards for prompt content with behavior-critical instructions.

These tests fail the moment someone accidentally removes or corrupts the
specific prompt instructions that underpin deterministic review assumptions.
No LLM is called; only the prompt JSON on disk is inspected.
"""

from __future__ import annotations

import re

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

    def test_fundamentals_prompt_mentions_growth_quality_unproven_cap(self):
        prompt = get_prompt("fundamentals_analyst")
        msg = prompt.system_message
        assert "GROWTH_QUALITY_UNPROVEN" in msg
        assert "Cap Revenue Growth scoring component at 0.5 pts" in msg

    def test_fundamentals_prompt_requires_guidance_baseline_promotion(self):
        msg = get_prompt("fundamentals_analyst").system_message
        for field in (
            "GUIDANCE_COVERAGE_STATUS",
            "OPERATING_VS_NET_DIRECTION",
            "DRIVER_PERSISTENCE",
            "EARNINGS_BASELINE_STATUS",
            "NORMALIZED_EARNINGS_AVAILABLE",
        ):
            assert field in msg
        assert "do not award the EPS_GROWTH point" in msg


class TestForeignLanguageGuidancePromptContent:
    def test_latest_results_and_tax_baseline_search_is_mandatory(self):
        msg = get_prompt("foreign_language_analyst").system_message
        assert "Search K: Management Guidance & Earnings Baseline (MANDATORY)" in msg
        assert "賃上げ促進税制" in msg
        assert "MANAGEMENT_GUIDANCE" in msg
        assert "NOT_DISCLOSED_AFTER_TARGETED_SEARCH" in msg

    def test_ownership_contract_separates_influence_from_control(self):
        msg = get_prompt("foreign_language_analyst").system_message
        for field in (
            "Largest Shareholder",
            "Control Status",
            "Control Basis",
            "Ownership Source URL",
            "Ownership As Of",
        ):
            assert field in msg
        assert "Largest shareholder is not a synonym for controlling shareholder" in msg
        assert "20–50% interest normally indicates significant influence" in msg
        assert "Never infer a ticker from memory" in msg
        assert "PENDING_VALIDATION is an asserted pre-validation state" in msg
        assert "Ownership Evidence Status: NOT_FOUND" in msg
        assert "do not materialize the other ownership fields" in msg

    def test_latest_results_snapshot_requires_inspected_same_statement_values(self):
        msg = get_prompt("foreign_language_analyst").system_message
        assert "Search L: Latest Official Results Snapshot (MANDATORY)" in msg
        assert "call `get_official_document`" in msg
        assert "current and year-ago comparative revenue and earnings" in msg
        assert "same statement presentation" in msg
        assert "this agent runs in parallel" in msg
        assert "LATEST_RESULTS_PERIOD_END" in msg


class TestWriterAnalyticalIntegrityPromptContent:
    def test_writer_scopes_common_semantic_overclaims(self):
        msg = get_prompt("article_writer").system_message.casefold()
        for instruction in (
            "one accounting segment does not establish one product",
            "related-party transaction does not establish value transfer",
            "acquisition-led growth requires",
            "no_catalyst_detected means no identified",
            "aggregator analyst-opinion count",
        ):
            assert instruction in msg


class TestPortfolioManagerPromptContent:
    """Guard PM handling of consultant no-coverage cases."""

    def test_portfolio_manager_prompt_makes_no_coverage_neutral(self):
        prompt = get_prompt("portfolio_manager")
        assert (
            "Missing evidence and internal-agent disagreement are not issuer risk"
            in prompt.system_message
        ), "PM prompt must keep no-coverage and internal disagreements neutral."

    def test_portfolio_manager_unverifiable_conflicts_are_zero_weight(self):
        msg = get_prompt("portfolio_manager").system_message
        assert "UNVERIFIABLE (+0.00, ANALYSIS_QUALITY only" in msg
        assert "entity mismatch as ANALYSIS_QUALITY (+0.00)" in msg

    def test_portfolio_manager_prompt_mentions_idle_cash_leniency(self):
        prompt = get_prompt("portfolio_manager")
        assert (
            "CAPITAL_PLAN_STATUS" in prompt.system_message
        ), "PM prompt must distinguish idle cash with no plan from justified cash buffers."

    def test_portfolio_manager_prompt_blocks_override_on_unproven_strength(self):
        prompt = get_prompt("portfolio_manager")
        msg = prompt.system_message
        assert "GROWTH_QUALITY_UNPROVEN" in msg
        assert "TRANSIENT_STRENGTH_DISTORTION" in msg
        assert "BUY override is not allowed" in msg

    def test_portfolio_manager_prompt_forbids_unverifiable_override_support(self):
        prompt = get_prompt("portfolio_manager")
        assert "UNVERIFIABLE consultant-resolution claims" in prompt.system_message

    def test_portfolio_manager_prompt_separates_data_quality_from_thesis_risk(self):
        """Internal-pipeline disagreements must be DATA_QUALITY_REVIEW (0.0 tally), not thesis risk.

        Guards the fix for the over-rejection failure mode where moving-average /
        price-feed / extractor mismatches were scored as Tier C1/C2 investment risk.
        """
        msg = get_prompt("portfolio_manager").system_message
        assert "DATA-QUALITY REVIEW vs THESIS RISK" in msg
        assert "DATA_QUALITY_REVIEW" in msg
        assert "never Tier C2" in msg

    def test_portfolio_manager_prompt_suppresses_moat_bonus_under_peak(self):
        """Moat/capital-efficiency bonuses must be scored 0.0 under a peak/transient flag."""
        msg = get_prompt("portfolio_manager").system_message
        assert "MOAT_BONUS_SUPPRESSED_PEAK_TRANSIENT" in msg
        assert "do NOT re-introduce the negative bonus" in msg

    def test_portfolio_manager_prompt_enforces_distortion_before_catalyst(self):
        """One-time items must be classified as distortions before being credited as catalysts."""
        msg = get_prompt("portfolio_manager").system_message
        assert "Distortion-before-catalyst" in msg
        assert "NORMALIZED_EARNINGS_REQUIRED" in msg
        assert "tax credit/incentive" in msg
        assert "MANAGEMENT_GUIDANCE_EVIDENCE_GAP" in msg

    def test_portfolio_manager_prompt_blocks_buy_on_material_unverified_signal(self):
        """A large unverified operating decline must block BUY pending verification (not auto-SELL)."""
        msg = get_prompt("portfolio_manager").system_message
        assert "MATERIAL_UNVERIFIED_OPERATING_SIGNAL" in msg
        assert "BLOCK BUY" in msg

    def test_portfolio_manager_blocks_load_bearing_secondary_growth_evidence(self):
        msg = get_prompt("portfolio_manager").system_message
        assert "DECISION_CRITICAL_GROWTH_EVIDENCE_GAP" in msg
        assert "secondary or unsupported R&D/capex evidence" in msg


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


class TestKoreanPromptAnchors:
    """Guard Korean accounting, disclosure, and governance prompt anchors."""

    def test_foreign_language_prompt_has_korean_disclosure_and_fcf_terms(self):
        prompt = get_prompt("foreign_language_analyst")
        msg = prompt.system_message
        for term in [
            "DART",
            "사업보고서",
            "기업가치 제고 계획",
            "영업활동현금흐름",
            "잉여현금흐름",
        ]:
            assert term in msg

    def test_auditor_prompt_has_korean_audit_opinion_terms(self):
        prompt = get_prompt("global_forensic_auditor")
        msg = prompt.system_message
        for term in ["적정의견", "한정의견", "부적정의견", "의견거절"]:
            assert term in msg

    def test_auditor_prompt_has_korean_accounting_risk_terms(self):
        prompt = get_prompt("global_forensic_auditor")
        msg = prompt.system_message
        for term in ["분식회계", "대손충당금", "특수관계자 거래"]:
            assert term in msg

    def test_apac_and_value_trap_prompts_have_korean_market_terms(self):
        apac = get_prompt("apac_regional_specialist")
        value_trap = get_prompt("value_trap_detector")
        assert "코리아 디스카운트" in apac.system_message
        assert "밸류업" in apac.system_message
        assert "자사주 소각" in apac.system_message
        assert "기업지배구조보고서" in value_trap.system_message


def _version_ok(version: str) -> bool:
    return bool(re.match(r"^\d+\.\d+$", version))


class TestOcfSamePeriodRule:
    """Fundamentals prompt must forbid comparing a sub-annual filing OCF to TTM."""

    def test_version_valid(self):
        assert _version_ok(get_prompt("fundamentals_analyst").version)

    def test_ocf_same_period_rule_present(self):
        sm = get_prompt("fundamentals_analyst").system_message
        assert "OCF same-period rule" in sm
        assert "ESTIMATED" in sm


class TestForeignLanguageSearchCFreshness:
    """Search C must prefer the latest quarter, not the annual report."""

    def test_version_valid(self):
        assert _version_ok(get_prompt("foreign_language_analyst").version)

    def test_prefers_latest_quarter(self):
        sm = get_prompt("foreign_language_analyst").system_message
        assert "most recent quarterly" in sm
        assert "trailing-4-quarter" in sm

    def test_en_fallback_not_annual_only(self):
        sm = get_prompt("foreign_language_analyst").system_message
        assert "cash flow from operations annual report" not in sm
        assert "latest quarterly results operating cash flow" in sm


class TestUndiscoveredCoverageGuard:
    """Coverage caveat must bind against unqualified 'undiscovered' framing."""

    def test_research_manager_binding_rule(self):
        rm = get_prompt("research_manager")
        assert _version_ok(rm.version)
        assert "Undiscovered framing is binding" in rm.system_message
        assert "ANALYST_COVERAGE_DATA_QUALITY_NOTE" in rm.system_message

    def test_sentiment_qualifier(self):
        s = get_prompt("sentiment_analyst")
        assert _version_ok(s.version)
        assert "low Western / English-language" in s.system_message

    def test_no_hardcoded_analyst_count(self):
        # The disputed "5 analysts" claim must not be encoded anywhere.
        for key in ("research_manager", "sentiment_analyst"):
            sm = get_prompt(key).system_message
            assert "5 analysts" not in sm


class TestPmPositionPrecedence:
    """PM position parameters must be binding over upstream reference levels."""

    def test_version_valid(self):
        assert _version_ok(get_prompt("portfolio_manager").version)

    def test_precedence_rule_present(self):
        sm = get_prompt("portfolio_manager").system_message
        assert "these FINAL POSITION PARAMETERS supersede" in sm
        assert "supersede" in sm
