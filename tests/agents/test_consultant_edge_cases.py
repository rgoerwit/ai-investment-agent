"""
Comprehensive edge case tests for consultant integration.

Tests various failure modes, data format edge cases, and system robustness
to ensure the consultant doesn't break existing functionality under stress.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest
from langgraph.types import RunnableConfig

from src.agents import create_consultant_node
from src.agents.consultant_nodes import _canonicalize_forensic_auditor_output
from src.report_generator import QuietModeReporter


class TestDataFormatEdgeCases:
    """Test consultant handling of unusual data formats."""

    def test_auditor_output_is_canonicalized_to_raw_verdict_field(self):
        content = (
            "FORENSIC_DATA_BLOCK:\n"
            "STATUS: INSUFFICIENT_DATA\n"
            "**Verdict:** Unable to complete forensic audit.\n"
        )

        normalized = _canonicalize_forensic_auditor_output(content)

        assert "VERDICT: Unable to complete forensic audit." in normalized
        assert "**Verdict:**" not in normalized

    def test_auditor_output_injects_recoverable_fallback_verdict(self):
        content = (
            "## FORENSIC AUDITOR REPORT\n\n"
            "**STATUS**: INSUFFICIENT_DATA\n\n"
            "FORENSIC_DATA_BLOCK:\n"
            "STATUS: INSUFFICIENT_DATA\n"
            "META: N/A\n"
        )

        normalized = _canonicalize_forensic_auditor_output(content)

        assert "STATUS: INSUFFICIENT_DATA" in normalized
        assert (
            "VERDICT: Unable to perform comprehensive forensic audit from "
            "verified primary source documents."
        ) in normalized

    def test_auditor_output_expands_inline_stub(self):
        content = (
            "FORENSIC_DATA_BLOCK: STATUS=INSUFFICIENT_DATA, "
            "REASON=STALE_DATA, REPORT_DATE=2025-06-30, AGE=9 months"
        )

        normalized = _canonicalize_forensic_auditor_output(content)

        assert "FORENSIC_DATA_BLOCK:" in normalized
        assert "STATUS: INSUFFICIENT_DATA" in normalized
        assert "REASON: STALE_DATA" in normalized
        assert "META: REPORT_DATE=2025-06-30 | AGE=9 months" in normalized

    def test_auditor_output_normalizes_block_and_status_labels(self):
        content = (
            "FORENSIC BLOCK:\n"
            "STATUS: **UNAVAILABLE**\n"
            "META: CONTEXT_LIMIT_EXCEEDED\n"
        )

        normalized = _canonicalize_forensic_auditor_output(content)

        assert "FORENSIC_DATA_BLOCK:" in normalized
        assert "STATUS: UNAVAILABLE" in normalized

    @pytest.mark.asyncio
    async def test_consultant_handles_empty_reports(self):
        """Test consultant gracefully handles empty analyst reports."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: MAJOR CONCERNS - Missing data"

        async def mock_invoke(*args, **kwargs):
            return mock_response

        with patch(
            "src.agents.runtime.invoke_with_rate_limit_handling", new=mock_invoke
        ):
            with patch("src.prompts.get_prompt") as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                # All reports empty
                state = {
                    "company_of_interest": "TEST",
                    "company_name": "Test Co",
                    "market_report": "",
                    "sentiment_report": "",
                    "news_report": "",
                    "fundamentals_report": "",
                    "investment_debate_state": {},
                    "investment_plan": "",
                    "red_flags": [],
                    "pre_screening_result": "PASS",
                }

                config = RunnableConfig(
                    configurable={"context": Mock(trade_date="2025-12-13")}
                )

                result = await consultant_node(state, config)

                assert "consultant_review" in result
                assert result["consultant_review"]  # Should still return something

    @pytest.mark.asyncio
    async def test_consultant_handles_missing_debate_state(self):
        """Test consultant handles missing investment_debate_state gracefully and logs diagnostic."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: APPROVED"

        # Track what was passed to invoke
        invoke_calls = []

        async def mock_invoke(*args, **kwargs):
            invoke_calls.append(args)
            return mock_response

        with patch(
            "src.agents.runtime.invoke_with_rate_limit_handling", new=mock_invoke
        ):
            with patch("src.prompts.get_prompt") as mock_get_prompt:
                with patch("src.agents.consultant_nodes.logger") as mock_logger:
                    mock_prompt = Mock()
                    mock_prompt.system_message = "You are a consultant."
                    mock_prompt.agent_name = "External Consultant"
                    mock_get_prompt.return_value = mock_prompt

                    consultant_node = create_consultant_node(mock_llm, "consultant")

                    # investment_debate_state is None instead of dict
                    state = {
                        "company_of_interest": "TEST",
                        "company_name": "Test Co",
                        "market_report": "Report",
                        "sentiment_report": "Report",
                        "news_report": "Report",
                        "fundamentals_report": "Report",
                        "investment_debate_state": None,  # None instead of dict
                        "investment_plan": "BUY",
                        "red_flags": [],
                        "pre_screening_result": "PASS",
                    }

                    config = RunnableConfig(
                        configurable={"context": Mock(trade_date="2025-12-13")}
                    )

                    result = await consultant_node(state, config)

                    assert "consultant_review" in result
                    # Should not crash - consultant may still provide review even with missing data
                    assert len(result["consultant_review"]) > 0

                    # Verify diagnostic logging was triggered
                    mock_logger.error.assert_called_once()
                    error_call = mock_logger.error.call_args
                    assert error_call[0][0] == "consultant_received_none_debate_state"
                    assert error_call[1]["ticker"] == "TEST"

                    # Verify diagnostic message was included in context passed to LLM
                    assert len(invoke_calls) > 0
                    llm_messages = invoke_calls[0][1]
                    message_content = llm_messages[0].content
                    assert "SYSTEM DIAGNOSTIC" in message_content
                    assert "Debate state unexpectedly None" in message_content

    @pytest.mark.asyncio
    async def test_consultant_handles_list_instead_of_string(self):
        """Test consultant handles list accumulation from state reducers."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: APPROVED"

        async def mock_invoke(*args, **kwargs):
            return mock_response

        with patch(
            "src.agents.runtime.invoke_with_rate_limit_handling", new=mock_invoke
        ):
            with patch("src.prompts.get_prompt") as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                # Reports are lists (from reducer accumulation bug)
                state = {
                    "company_of_interest": "TEST",
                    "company_name": "Test Co",
                    "market_report": ["Report 1", "Report 2"],  # List instead of string
                    "sentiment_report": "Normal string report",
                    "news_report": "Report",
                    "fundamentals_report": "Report",
                    "investment_debate_state": {"history": "Debate"},
                    "investment_plan": "BUY",
                    "red_flags": [],
                    "pre_screening_result": "PASS",
                }

                config = RunnableConfig(
                    configurable={"context": Mock(trade_date="2025-12-13")}
                )

                result = await consultant_node(state, config)

                assert "consultant_review" in result
                # Should not crash - lists should be handled


class TestConfigurationEdgeCases:
    """Test consultant configuration and initialization edge cases."""

    def test_consultant_with_invalid_model_name(self):
        """Test consultant handles invalid model names gracefully."""
        # Skip if langchain-openai not installed
        try:
            import langchain_openai
        except ImportError:
            pytest.skip("langchain-openai not installed (optional dependency)")

        from src.llms import create_consultant_llm

        with patch("langchain_openai.ChatOpenAI") as mock_chatgpt:
            mock_llm = Mock()
            mock_chatgpt.return_value = mock_llm

            with patch("src.llms.config") as mock_config:
                mock_config.enable_consultant = True
                mock_config.consultant_model = "invalid-model-name-12345"
                mock_config.get_openai_api_key.return_value = "test-key"
                llm = create_consultant_llm()

                # Should still create LLM (OpenAI will validate model name)
                assert llm is not None
                call_kwargs = mock_chatgpt.call_args[1]
                assert call_kwargs["model"] == "invalid-model-name-12345"

    def test_consultant_with_empty_api_key(self):
        """Test consultant with empty string API key (not missing)."""
        import src.llms
        from src.llms import get_consultant_llm

        # Reset singleton to force re-evaluation
        src.llms._consultant_llm_instance = None

        with patch("src.llms.config") as mock_config:
            mock_config.enable_consultant = True
            mock_config.get_openai_api_key.return_value = ""
            llm = get_consultant_llm()

            # Should return None (empty key treated same as missing)
            assert llm is None

        # Reset for other tests
        src.llms._consultant_llm_instance = None

    def test_consultant_enable_flag_disabled(self):
        """Test that consultant is disabled when enable_consultant=False."""
        import src.llms
        from src.llms import get_consultant_llm

        # Reset singleton to force re-evaluation
        src.llms._consultant_llm_instance = None

        with patch("src.llms.config") as mock_config:
            mock_config.enable_consultant = False
            llm = get_consultant_llm()
            assert llm is None, "Should be disabled when enable_consultant=False"

        # Reset for other tests
        src.llms._consultant_llm_instance = None


class TestErrorPropagation:
    """Test error handling and propagation through the system."""

    @pytest.mark.asyncio
    async def test_consultant_llm_timeout_error(self):
        """Test consultant handles LLM timeout gracefully."""
        mock_llm = Mock()

        async def mock_invoke_timeout(*args, **kwargs):
            raise TimeoutError("OpenAI API request timed out after 120s")

        with patch(
            "src.agents.runtime.invoke_with_rate_limit_handling",
            new=mock_invoke_timeout,
        ):
            with patch("src.prompts.get_prompt") as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                state = {
                    "company_of_interest": "TEST",
                    "company_name": "Test Co",
                    "market_report": "Report",
                    "sentiment_report": "Report",
                    "news_report": "Report",
                    "fundamentals_report": "Report",
                    "investment_debate_state": {"history": "Debate"},
                    "investment_plan": "BUY",
                }

                config = RunnableConfig(
                    configurable={"context": Mock(trade_date="2025-12-13")}
                )

                result = await consultant_node(state, config)

                assert "consultant_review" in result
                assert result["consultant_review"] == ""
                status = result["artifact_statuses"]["consultant_review"]
                assert status["ok"] is False
                assert status["error_kind"] == "timeout"

    @pytest.mark.asyncio
    async def test_consultant_wall_clock_timeout_cuts_off_stalled_invoke(self):
        """A hanging consultant call should be bounded by the node timeout budget."""
        mock_llm = Mock()

        async def mock_invoke_hang(*args, **kwargs):
            await asyncio.sleep(1.0)
            raise AssertionError("consultant invoke should have been cancelled first")

        with patch(
            "src.agents.runtime.invoke_with_rate_limit_handling",
            new=mock_invoke_hang,
        ):
            with patch("src.prompts.get_prompt") as mock_get_prompt:
                with patch(
                    "src.agents.consultant_nodes.CONSULTANT_CALL_TIMEOUT_SECONDS", 0.05
                ):
                    with patch(
                        "src.agents.consultant_nodes.CONSULTANT_TOTAL_TIMEOUT_SECONDS",
                        0.08,
                    ):
                        mock_prompt = Mock()
                        mock_prompt.system_message = "You are a consultant."
                        mock_prompt.agent_name = "External Consultant"
                        mock_get_prompt.return_value = mock_prompt

                        consultant_node = create_consultant_node(mock_llm, "consultant")
                        state = {
                            "company_of_interest": "TEST",
                            "company_name": "Test Co",
                            "market_report": "Report",
                            "sentiment_report": "Report",
                            "news_report": "Report",
                            "fundamentals_report": "Report",
                            "investment_debate_state": {"history": "Debate"},
                            "investment_plan": "BUY",
                        }
                        config = RunnableConfig(
                            configurable={"context": Mock(trade_date="2025-12-13")}
                        )

                        started = time.monotonic()
                        result = await consultant_node(state, config)
                        elapsed = time.monotonic() - started

        assert elapsed < 0.5
        assert result["consultant_review"] == ""
        status = result["artifact_statuses"]["consultant_review"]
        assert status["ok"] is False
        assert status["error_kind"] == "timeout"

    @pytest.mark.asyncio
    async def test_consultant_rate_limit_error(self):
        """Test consultant handles OpenAI rate limit errors."""
        mock_llm = Mock()

        async def mock_invoke_rate_limit(*args, **kwargs):
            raise Exception("Rate limit exceeded. Please retry after 60s.")

        with patch(
            "src.agents.runtime.invoke_with_rate_limit_handling",
            new=mock_invoke_rate_limit,
        ):
            with patch("src.prompts.get_prompt") as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                state = {
                    "company_of_interest": "TEST",
                    "company_name": "Test Co",
                    "market_report": "Report",
                    "sentiment_report": "Report",
                    "news_report": "Report",
                    "fundamentals_report": "Report",
                    "investment_debate_state": {"history": "Debate"},
                    "investment_plan": "BUY",
                }

                config = RunnableConfig(
                    configurable={"context": Mock(trade_date="2025-12-13")}
                )

                result = await consultant_node(state, config)

                assert "consultant_review" in result
                assert result["consultant_review"] == ""
                status = result["artifact_statuses"]["consultant_review"]
                assert status["ok"] is False
                assert status["error_kind"] == "rate_limit"


class TestReportGeneration:
    """Test report generation with consultant review."""

    def test_report_includes_consultant_review(self):
        """Test that generated report includes consultant section."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis here",
            "sentiment_report": "Sentiment analysis",
            "news_report": "News analysis",
            "fundamentals_report": "Fundamentals",
            "investment_plan": "BUY recommendation",
            "consultant_review": "CONSULTANT REVIEW: APPROVED\n\nAnalysis is sound.",
            "trader_investment_plan": "Trading plan",
            "final_trade_decision": "FINAL DECISION: BUY\n\nRationale: Good fundamentals.",
        }

        report = reporter.generate_report(result, brief_mode=False)

        assert "External Consultant Review" in report
        assert "CONSULTANT REVIEW: APPROVED" in report

    def test_report_excludes_consultant_error(self):
        """Test that report excludes consultant review if it's an error."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis",
            "fundamentals_report": "Fundamentals",
            "investment_plan": "BUY recommendation",
            "consultant_review": "Consultant Review Error: OpenAI API timeout",
            "final_trade_decision": "FINAL DECISION: BUY",
        }

        report = reporter.generate_report(result, brief_mode=False)

        # Should NOT include consultant section if it's an error
        assert "External Consultant Review" not in report

    def test_report_excludes_consultant_na(self):
        """Test that report excludes consultant review if N/A (disabled)."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis",
            "fundamentals_report": "Fundamentals",
            "investment_plan": "BUY recommendation",
            "consultant_review": "N/A (consultant disabled or unavailable)",
            "final_trade_decision": "FINAL DECISION: BUY",
        }

        report = reporter.generate_report(result, brief_mode=False)

        # Should NOT include consultant section if N/A
        assert "External Consultant Review" not in report

    def test_report_handles_missing_consultant_field(self):
        """Test report generation when consultant_review field missing entirely."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis",
            "fundamentals_report": "Fundamentals",
            "investment_plan": "BUY recommendation",
            # consultant_review field missing entirely
            "final_trade_decision": "FINAL DECISION: BUY",
        }

        report = reporter.generate_report(result, brief_mode=False)

        # Should not crash, should generate valid report
        assert "BUY" in report
        assert "TEST" in report
        assert "External Consultant Review" not in report

    def test_rendered_report_strips_reasoning_dict_repr_single_quotes(self):
        """Single-quote Python repr of a reasoning dict is stripped from rendered output (A3)."""
        leaked_line = "{'id': 'rs_abc123', 'summary': [], 'type': 'reasoning'}"
        review = f"Analysis is thorough and well-supported.\n{leaked_line}\nRisk is acceptable."
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")
        result = {
            "company_of_interest": "TEST",
            "consultant_review": review,
            "final_trade_decision": "FINAL DECISION: BUY",
        }
        report = reporter.generate_report(result, brief_mode=False)
        assert "rs_abc123" not in report
        assert "Analysis is thorough" in report
        assert "Risk is acceptable" in report

    def test_rendered_report_strips_reasoning_dict_repr_double_quotes(self):
        """JSON-style double-quote repr of a reasoning dict is stripped from rendered output (A3)."""
        leaked_line = '{"id": "rs_xyz999", "summary": [], "type": "reasoning"}'
        review = f"Cross-validation complete.\n{leaked_line}\nNo anomalies found."
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")
        result = {
            "company_of_interest": "TEST",
            "consultant_review": review,
            "final_trade_decision": "FINAL DECISION: BUY",
        }
        report = reporter.generate_report(result, brief_mode=False)
        assert "rs_xyz999" not in report
        assert "Cross-validation complete" in report
        assert "No anomalies found" in report

    def test_rendered_report_preserves_legitimate_review_text(self):
        """Legitimate review prose is preserved when a leaked reasoning line is stripped."""
        leaked_line = "{'id': 'rs_abc123', 'summary': [], 'type': 'reasoning'}"
        review = (
            "The analysis correctly identifies key risks.\n"
            f"{leaked_line}\n"
            "Valuation metrics support the BUY recommendation."
        )
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")
        result = {
            "company_of_interest": "TEST",
            "consultant_review": review,
            "final_trade_decision": "FINAL DECISION: BUY",
        }
        report = reporter.generate_report(result, brief_mode=False)
        assert "correctly identifies key risks" in report
        assert "Valuation metrics support" in report

    def test_sanitizer_does_not_strip_unrelated_dict_like_lines(self):
        """Lines that look like dicts but lack rs_ prefix and reasoning type are preserved."""
        review = (
            'Analysis summary: {"key": "value", "score": 95}\nConclusion: strong BUY.'
        )
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")
        result = {
            "company_of_interest": "TEST",
            "consultant_review": review,
            "final_trade_decision": "FINAL DECISION: BUY",
        }
        report = reporter.generate_report(result, brief_mode=False)
        assert '{"key": "value"' in report or '"key": "value"' in report
        assert "Conclusion: strong BUY" in report

    def test_report_omits_research_manager_recommendation_when_pm_decision_exists(self):
        """The public report should not publish a second recommendation section."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis",
            "fundamentals_report": "Fundamentals",
            "investment_plan": "INVESTMENT RECOMMENDATION: HOLD",
            "final_trade_decision": "PORTFOLIO MANAGER VERDICT: DO NOT INITIATE",
        }

        report = reporter.generate_report(result, brief_mode=False)

        assert "Investment Recommendation" not in report
        assert "PORTFOLIO MANAGER VERDICT: DO NOT INITIATE" in report

    def test_report_surfaces_verification_caveats_before_executive_summary(self):
        """Consultant disputes should be elevated before the main writeup."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis",
            "fundamentals_report": "Fundamentals",
            "consultant_review": (
                "CONSULTANT REVIEW: CONDITIONAL\n\n"
                "The insider-selling claim is unsubstantiated.\n"
                "The 100 new vessels claim is likely wrong."
            ),
            "artifact_statuses": {
                "consultant_review": {"complete": True, "ok": False},
            },
            "final_trade_decision": "FINAL DECISION: HOLD",
        }

        report = reporter.generate_report(result, brief_mode=False)

        assert "## Verification Caveats" in report
        assert "insider-selling claim is unsubstantiated" in report
        assert report.index("## Verification Caveats") < report.index(
            "## Executive Summary"
        )

    def test_report_rewrites_false_consultant_unavailable_claim(self):
        """Public report should not claim the consultant was unavailable when review exists."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis",
            "fundamentals_report": "Fundamentals",
            "consultant_review": "CONSULTANT REVIEW: CONDITIONAL APPROVAL\n\nCoverage gaps remain.",
            "artifact_statuses": {
                "consultant_review": {"complete": True, "ok": False},
            },
            "final_trade_decision": (
                'DECISION RATIONALE: The pre-screening flagged a "Consultant Conditional" '
                "warning, but as the external consultant was unavailable to provide "
                "specific conditions, the verified `DATA_BLOCK` fundamentals and moat "
                "signals take absolute precedence."
            ),
        }

        report = reporter.generate_report(result, brief_mode=False)

        assert (
            "external consultant was unavailable to provide specific conditions"
            not in report
        )
        assert "tool-coverage gaps" in report

    def test_report_repairs_glued_structured_block_boundary_before_demoting_headers(
        self,
    ):
        """Rendered reports should clean older glued block boundaries without other changes."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "fundamentals_report": (
                "### --- START DATA_BLOCK ---\n"
                "SECTOR: Energy\n"
                "### --- END DATA_BLOCK ---### FINANCIAL HEALTH DETAIL\n"
                "**Score**: 9/12\n"
            ),
            "final_trade_decision": "PORTFOLIO MANAGER VERDICT: HOLD",
        }

        report = reporter.generate_report(result, brief_mode=False)

        # Both the DATA_BLOCK content and the prose section must survive
        assert "SECTOR: Energy" in report
        assert "FINANCIAL HEALTH DETAIL" in report
        # DATA_BLOCK is repositioned to end of section — prose appears before it
        assert report.index("FINANCIAL HEALTH DETAIL") < report.index(
            "START DATA_BLOCK"
        )


class TestBackwardsCompatibility:
    """Test that existing code works with or without consultant."""

    def test_state_dict_without_consultant_field(self):
        """Test AgentState dict access pattern without consultant_review."""
        state = {
            "company_of_interest": "TEST",
            "market_report": "Report",
            "fundamentals_report": "Report",
        }

        # Should not raise KeyError
        consultant_review = state.get("consultant_review", "")
        assert consultant_review == ""

    def test_portfolio_manager_handles_missing_consultant(self):
        """Test Portfolio Manager context building handles missing consultant."""
        # Simulate Portfolio Manager context assembly
        state = {
            "market_report": "Market",
            "sentiment_report": "Sentiment",
            "news_report": "News",
            "fundamentals_report": "Fundamentals",
            "investment_plan": "BUY",
            "consultant_review": "",  # Empty
            "trader_investment_plan": "Trader",
            "risk_debate_state": {"current_risky_response": "Risk assessment"},
        }

        consultant = state.get("consultant_review", "")

        # Should handle empty consultant gracefully
        consultant_section = f"\n\nEXTERNAL CONSULTANT REVIEW:\n{consultant if consultant else 'N/A (consultant disabled or unavailable)'}"

        assert "N/A (consultant disabled or unavailable)" in consultant_section


class TestTokenTracking:
    """Test token tracking for consultant usage."""

    def test_token_tracker_has_openai_pricing(self):
        """Test that token tracker includes OpenAI model pricing."""
        from src.token_tracker import TokenUsage

        # Test gpt-4o pricing
        usage = TokenUsage(
            timestamp="2025-12-13",
            agent_name="Consultant",
            model_name="gpt-4o",
            prompt_tokens=4000,
            completion_tokens=800,
            total_tokens=4800,
        )

        cost = usage.estimated_cost_usd

        # gpt-4o: $2.50/1M input, $10.00/1M output
        # 4000 * 2.50 / 1M + 800 * 10.00 / 1M = 0.01 + 0.008 = 0.018
        expected_cost = (4000 * 2.50 / 1_000_000) + (800 * 10.00 / 1_000_000)
        assert abs(cost - expected_cost) < 0.001

    def test_token_tracker_has_gpt4o_mini_pricing(self):
        """Test token tracker pricing for gpt-4o-mini."""
        from src.token_tracker import TokenUsage

        usage = TokenUsage(
            timestamp="2025-12-13",
            agent_name="Consultant",
            model_name="gpt-4o-mini",
            prompt_tokens=100000,
            completion_tokens=50000,
            total_tokens=150000,
        )

        cost = usage.estimated_cost_usd

        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        # 100k * 0.15 / 1M + 50k * 0.60 / 1M = 0.015 + 0.030 = 0.045
        expected_cost = (100000 * 0.15 / 1_000_000) + (50000 * 0.60 / 1_000_000)
        assert abs(cost - expected_cost) < 0.001


class TestLargeContextHandling:
    """Test consultant handling of very large reports."""

    @pytest.mark.asyncio
    async def test_consultant_with_very_large_reports(self):
        """Test consultant handles very large input context."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: APPROVED"

        async def mock_invoke(*args, **kwargs):
            return mock_response

        with patch(
            "src.agents.runtime.invoke_with_rate_limit_handling", new=mock_invoke
        ):
            with patch("src.prompts.get_prompt") as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                # Create very large reports (simulate comprehensive analysis)
                large_report = "DATA_BLOCK\n" + ("Financial metric: 123.45\n" * 1000)

                state = {
                    "company_of_interest": "TEST",
                    "company_name": "Test Co",
                    "market_report": "Market analysis " * 500,
                    "sentiment_report": "Sentiment " * 500,
                    "news_report": "News " * 500,
                    "fundamentals_report": large_report,
                    "investment_debate_state": {"history": "Debate history " * 1000},
                    "investment_plan": "BUY recommendation " * 100,
                }

                config = RunnableConfig(
                    configurable={"context": Mock(trade_date="2025-12-13")}
                )

                result = await consultant_node(state, config)

                assert "consultant_review" in result
                # Should handle large context without crashing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
