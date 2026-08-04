"""
Comprehensive edge case tests for consultant integration.

Tests various failure modes, data format edge cases, and system robustness
to ensure the consultant doesn't break existing functionality under stress.
"""

import asyncio
import copy
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from langgraph.types import RunnableConfig

from src.agents import create_consultant_node
from src.agents.consultant_nodes import (
    _create_openai_responses_fallback_llm,
    _select_quick_consultant_profile,
)
from src.agents.forensic_repair import canonicalize_forensic_auditor_output
from src.report_generator import QuietModeReporter


class TestDataFormatEdgeCases:
    """Test consultant handling of unusual data formats."""

    @pytest.mark.asyncio
    async def test_final_turn_tool_call_triggers_fallback_synthesis(self):
        """3679.T 2026-07-11: when the final allowed iteration still returns a
        tool call, its serialized function_call block must not become the
        review — content extracts empty and the forced-synthesis fallback
        produces real prose instead."""
        from types import SimpleNamespace

        from src.agents.consultant_tool_loop import (
            ConsultantToolLoopPolicy,
            run_bounded_consultant_loop,
        )

        fc_block = {
            "arguments": '{"ticker":"3679.T","metric":"netIncomeToCommon"}',
            "call_id": "call_x",
            "name": "spot_check_metric_mcp_fmp",
            "type": "function_call",
            "id": "fc_1",
            "status": "completed",
        }
        tool_response = SimpleNamespace(
            content=[fc_block],
            tool_calls=[{"name": "spot_check_metric", "args": {}, "id": "call_x"}],
        )
        synthesis = SimpleNamespace(
            content="FINAL CONSULTANT VERDICT: CONDITIONAL APPROVAL",
            tool_calls=[],
        )
        invoked = []

        async def fake_invoke(llm, messages):
            invoked.append(llm)
            return tool_response if llm == "active" else synthesis

        policy = ConsultantToolLoopPolicy(
            max_tool_iterations=0,
            max_tool_calls_per_turn=4,
            deadline=time.monotonic() + 60,
            total_timeout=60,
        )
        result = await run_bounded_consultant_loop(
            active_llm="active",
            fallback_llm="fallback",
            messages=[],
            tools_by_name={"spot_check_metric": object()},
            policy=policy,
            invoke_with_deadline=fake_invoke,
            agent_name="External Consultant",
            agent_key="consultant",
            ticker="3679.T",
        )

        assert invoked == ["active", "fallback"]
        assert result.content == "FINAL CONSULTANT VERDICT: CONDITIONAL APPROVAL"
        assert "function_call" not in result.content

    @pytest.mark.asyncio
    async def test_subscription_coverage_gap_does_not_count_as_tool_failure(self):
        from src.agents.consultant_tool_loop import (
            ConsultantToolLoopPolicy,
            run_bounded_consultant_loop,
        )

        responses = iter(
            (
                SimpleNamespace(
                    content="",
                    tool_calls=[
                        {
                            "name": "spot_check_metric_mcp_fmp",
                            "args": {"ticker": "AGS.SI", "metric": "operatingCashflow"},
                            "id": "mcp-1",
                        }
                    ],
                ),
                SimpleNamespace(
                    content="FINAL CONSULTANT VERDICT: CONDITIONAL APPROVAL",
                    tool_calls=[],
                ),
            )
        )

        async def fake_invoke(_llm, _messages):
            return next(responses)

        class CoverageGapTool:
            async def ainvoke(self, _args):
                return json.dumps(
                    {
                        "error": "HTTP 402 Payment Required",
                        "failure_kind": "coverage_gap",
                        "provider": "fmp",
                        "skipped": True,
                        "coverage_gap": True,
                    }
                )

        class PassthroughService:
            async def execute(self, invocation, runner):
                return SimpleNamespace(value=await runner(invocation.args))

        result = await run_bounded_consultant_loop(
            active_llm="active",
            fallback_llm="fallback",
            messages=[],
            tools_by_name={"spot_check_metric_mcp_fmp": CoverageGapTool()},
            policy=ConsultantToolLoopPolicy(
                max_tool_iterations=1,
                max_tool_calls_per_turn=4,
                deadline=time.monotonic() + 60,
                total_timeout=60,
            ),
            invoke_with_deadline=fake_invoke,
            tool_service_getter=lambda: PassthroughService(),
            agent_name="External Consultant",
            agent_key="consultant",
            ticker="AGS.SI",
        )

        assert result.had_tool_errors is False
        assert result.tool_failure_count == 0
        assert result.tool_call_count == 1

    def test_auditor_output_is_canonicalized_to_raw_verdict_field(self):
        content = (
            "FORENSIC_DATA_BLOCK:\n"
            "STATUS: INSUFFICIENT_DATA\n"
            "**Verdict:** Unable to complete forensic audit.\n"
        )

        normalized = canonicalize_forensic_auditor_output(content)

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

        normalized = canonicalize_forensic_auditor_output(content)

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

        normalized = canonicalize_forensic_auditor_output(content)

        assert "FORENSIC_DATA_BLOCK:" in normalized
        assert "STATUS: INSUFFICIENT_DATA" in normalized
        assert "REASON: STALE_DATA" in normalized
        assert "META: REPORT_DATE=2025-06-30 | AGE=9 months" in normalized

    def test_auditor_output_normalizes_block_and_status_labels(self):
        content = (
            "FORENSIC BLOCK:\nSTATUS: **UNAVAILABLE**\nMETA: CONTEXT_LIMIT_EXCEEDED\n"
        )

        normalized = canonicalize_forensic_auditor_output(content)

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
    async def test_quick_consultant_no_tools_does_not_fallback_on_empty_content(self):
        """Quick/no-tool Consultant should not spend a second LLM call on empty output."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = ""
        mock_response.tool_calls = [{"name": "spot_check_metric_alt", "args": {}}]
        invoke_calls = []

        async def mock_invoke(*args, **kwargs):
            invoke_calls.append((args, kwargs))
            return mock_response

        with patch(
            "src.agents.runtime.invoke_with_rate_limit_handling", new=mock_invoke
        ):
            with patch("src.prompts.get_prompt") as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(
                    mock_llm, "consultant", tools=[], quick_mode=True
                )
                state = {
                    "company_of_interest": "TEST",
                    "company_name": "Test Co",
                    "market_report": "Report",
                    "sentiment_report": "Report",
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

        assert len(invoke_calls) == 1
        status = result["artifact_statuses"]["consultant_review"]
        assert status["ok"] is False
        assert status["error_kind"] == "application_error"

    @pytest.mark.asyncio
    async def test_quick_consultant_prompt_addendum_and_context_caps(self):
        """Quick prompt keeps evidence channels while using smaller section caps."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = (
            "### CONSULTANT REVIEW: APPROVED\n\n### FINAL CONSULTANT VERDICT\nAPPROVED"
        )
        invoke_messages = []
        summarize_calls = []

        async def mock_invoke(_llm, messages, **kwargs):
            invoke_messages.extend(messages)
            return mock_response

        def fake_summarize(value, section, budget):
            summarize_calls.append((section, budget))
            return f"{section}:{budget}:{value}"

        with patch(
            "src.agents.runtime.invoke_with_rate_limit_handling", new=mock_invoke
        ):
            with patch("src.prompts.get_prompt") as mock_get_prompt:
                with patch(
                    "src.agents.consultant_nodes.support.summarize_for_pm",
                    side_effect=fake_summarize,
                ):
                    mock_prompt = Mock()
                    mock_prompt.system_message = "You are a consultant."
                    mock_prompt.agent_name = "External Consultant"
                    mock_get_prompt.return_value = mock_prompt

                    consultant_node = create_consultant_node(
                        mock_llm, "consultant", tools=[], quick_mode=True
                    )
                    state = {
                        "company_of_interest": "TEST",
                        "company_name": "Test Co",
                        "market_report": "Market",
                        "sentiment_report": "Sentiment",
                        "news_report": "News",
                        "fundamentals_report": "Fundamentals",
                        "investment_debate_state": {"history": "Debate"},
                        "investment_plan": "Research",
                        "auditor_report": "Auditor",
                        "red_flags": [],
                        "pre_screening_result": "PASS",
                    }
                    config = RunnableConfig(
                        configurable={"context": Mock(trade_date="2025-12-13")}
                    )

                    result = await consultant_node(state, config)

        prompt = invoke_messages[0].content
        assert "QUICK SCREENING MODE" in prompt
        assert (
            "Do not request tools unless tool results are explicitly available"
            in prompt
        )
        assert "CONSULTANT REVIEW" in prompt
        assert "FINAL CONSULTANT VERDICT" in prompt
        assert "MARKET ANALYST REPORT:" in prompt
        assert "SENTIMENT ANALYST REPORT:" in prompt
        assert "NEWS ANALYST REPORT:" in prompt
        assert "FUNDAMENTALS ANALYST REPORT:" in prompt
        assert "BULL/BEAR DEBATE HISTORY" in prompt
        assert "RESEARCH MANAGER SYNTHESIS" in prompt
        assert "INDEPENDENT FORENSIC AUDIT" in prompt
        assert ("market", 900) in summarize_calls
        assert ("fundamentals", 2500) in summarize_calls
        assert ("research", 1400) in summarize_calls
        assert ("auditor", 1200) in summarize_calls
        assert result["consultant_quick_profile"] == "quick_standard"

    def test_quick_consultant_profile_expands_for_borderline_inputs(self):
        assert _select_quick_consultant_profile({"red_flags": [{"type": "PFIC"}]}) == (
            "quick_expanded"
        )
        assert (
            _select_quick_consultant_profile(
                {"artifact_statuses": {"auditor_report": {"ok": False}}}
            )
            == "quick_expanded"
        )
        assert (
            _select_quick_consultant_profile(
                {"investment_plan": "Recommendation: HOLD"}
            )
            == "quick_expanded"
        )
        assert _select_quick_consultant_profile({"investment_plan": "BUY"}) == (
            "quick_standard"
        )

    @pytest.mark.asyncio
    async def test_quick_consultant_expanded_profile_adds_evidence_index(self):
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = (
            "### CONSULTANT REVIEW: APPROVED\n\n### FINAL CONSULTANT VERDICT\nAPPROVED"
        )
        invoke_messages = []
        summarize_calls = []

        async def mock_invoke(_llm, messages, **kwargs):
            invoke_messages.extend(messages)
            return mock_response

        def fake_summarize(value, section, budget):
            summarize_calls.append((section, budget))
            return f"{section}:{budget}:{value}"

        with patch(
            "src.agents.runtime.invoke_with_rate_limit_handling", new=mock_invoke
        ):
            with patch("src.prompts.get_prompt") as mock_get_prompt:
                with patch(
                    "src.agents.consultant_nodes.support.summarize_for_pm",
                    side_effect=fake_summarize,
                ):
                    mock_prompt = Mock()
                    mock_prompt.system_message = "You are a consultant."
                    mock_prompt.agent_name = "External Consultant"
                    mock_get_prompt.return_value = mock_prompt

                    consultant_node = create_consultant_node(
                        mock_llm, "consultant", tools=[], quick_mode=True
                    )
                    result = await consultant_node(
                        {
                            "company_of_interest": "TEST",
                            "company_name": "Test Co",
                            "market_report": "Market",
                            "sentiment_report": "Sentiment",
                            "news_report": "News",
                            "fundamentals_report": "Fundamentals",
                            "investment_debate_state": {"history": "Debate"},
                            "investment_plan": "HOLD due to liquidity risk",
                            "auditor_report": "Auditor",
                            "red_flags": [{"type": "LIQUIDITY", "detail": "thin"}],
                            "pre_screening_result": "PASS",
                        },
                        RunnableConfig(
                            configurable={"context": Mock(trade_date="2025-12-13")}
                        ),
                    )

        prompt = invoke_messages[0].content
        assert "DECISION-CRITICAL EVIDENCE INDEX" in prompt
        assert "LIQUIDITY" in prompt
        assert "MARKET ANALYST REPORT:" in prompt
        assert ("fundamentals", 3600) in summarize_calls
        assert ("research", 2200) in summarize_calls
        assert result["consultant_quick_profile"] == "quick_expanded"

    async def _consultant_prompt(self, state: dict, *, quick_mode: bool) -> str:
        """Assemble a consultant prompt and return it verbatim."""
        mock_response = Mock()
        mock_response.content = (
            "### CONSULTANT REVIEW: APPROVED\n\n### FINAL CONSULTANT VERDICT\nAPPROVED"
        )
        invoke_messages: list = []

        async def mock_invoke(_llm, messages, **kwargs):
            invoke_messages.extend(messages)
            return mock_response

        with (
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling", new=mock_invoke
            ),
            patch("src.prompts.get_prompt") as mock_get_prompt,
        ):
            mock_prompt = Mock()
            mock_prompt.system_message = "You are a consultant."
            mock_prompt.agent_name = "External Consultant"
            mock_get_prompt.return_value = mock_prompt

            node = create_consultant_node(
                Mock(), "consultant", tools=[], quick_mode=quick_mode
            )
            await node(
                {
                    "company_of_interest": "TEST",
                    "company_name": "Test Co",
                    "investment_debate_state": {"history": "Debate"},
                    **state,
                },
                RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")}),
            )
        return invoke_messages[0].content


class TestConsultantRedFlagRendering:
    """The consultant reads flags through the canonical renderer, not dict repr.

    ``prompts/consultant.json`` grants this seat veto authority keyed on the
    literal token ``CMIC_FLAGGED``, which is a red-flag ``type`` value. It used to
    arrive only inside a Python ``repr`` of ``list[dict]`` — and, in quick mode,
    inside a copy clipped at 220 chars that dropped ``risk_penalty`` and
    ``blocks_buy`` first, because ``detail``/``rationale`` are long strings
    ordered ahead of them.
    """

    CMIC_FLAG = {
        "type": "CMIC_FLAGGED",
        "severity": "HIGH",
        "detail": "Company appears on NS-CMIC list. Named in an OFAC listing.",
        "action": "RISK_PENALTY",
        "risk_penalty": 2.0,
        "blocks_buy": True,
        "rationale": "US Executive Orders prohibit US persons from investing in "
        "NS-CMIC listed companies. Verify current OFAC status before investing. "
        "Restrictions may be modified by future executive orders.",
    }

    @pytest.mark.asyncio
    async def test_veto_token_is_legible_with_its_penalty(self):
        prompt = await TestDataFormatEdgeCases()._consultant_prompt(
            {"red_flags": [self.CMIC_FLAG], "pre_screening_result": "PASS"},
            quick_mode=False,
        )
        assert "CMIC_FLAGGED [risk_penalty +2.00]" in prompt
        assert "Company appears on NS-CMIC list." in prompt

    @pytest.mark.asyncio
    async def test_no_python_dict_repr_reaches_the_prompt(self):
        prompt = await TestDataFormatEdgeCases()._consultant_prompt(
            {"red_flags": [self.CMIC_FLAG], "pre_screening_result": "PASS"},
            quick_mode=False,
        )
        assert "{'type':" not in prompt
        assert "'blocks_buy'" not in prompt

    @pytest.mark.asyncio
    async def test_quick_mode_renders_the_flag_set_once(self):
        """Counts the *rendered* flag line, not the bare token.

        These tests mock the system message, so a whole-prompt token count would
        not measure the real prompt — `prompts/consultant.json` itself names
        `CMIC_FLAGGED` once in its veto rule, so the real assembled prompt
        legitimately contains the token twice (rule + rendered flag). What must
        not recur is the flag *rendering*, which quick mode used to duplicate.
        """
        prompt = await TestDataFormatEdgeCases()._consultant_prompt(
            {
                "red_flags": [self.CMIC_FLAG],
                "pre_screening_result": "PASS",
                "investment_plan": "HOLD",
            },
            quick_mode=True,
        )
        assert prompt.count("CMIC_FLAGGED [risk_penalty") == 1

    @pytest.mark.asyncio
    async def test_consultant_is_not_given_the_pm_tally_contract(self):
        """`TOTAL RISK COUNT` is a Portfolio Manager output field.

        `prompts/consultant.json` never mentions it, so shipping the PM's tally
        instruction here would ask this seat to produce a field it does not own.
        """
        prompt = await TestDataFormatEdgeCases()._consultant_prompt(
            {"red_flags": [self.CMIC_FLAG], "pre_screening_result": "PASS"},
            quick_mode=False,
        )
        assert "CODE-COMPUTED RISK SUBTOTAL" in prompt
        assert "TOTAL RISK COUNT" not in prompt

    @pytest.mark.asyncio
    async def test_penalties_survive_a_long_flag_list(self):
        """The truncation regression: every penalty must reach the model."""
        flags = [
            {
                "type": f"FLAG_{i}",
                "detail": "d" * 400,
                "rationale": "r" * 400,
                "risk_penalty": 0.5,
            }
            for i in range(8)
        ]
        prompt = await TestDataFormatEdgeCases()._consultant_prompt(
            {"red_flags": flags, "pre_screening_result": "PASS"}, quick_mode=True
        )
        for i in range(8):
            assert f"FLAG_{i} [risk_penalty +0.50]" in prompt
        assert (
            "CODE-COMPUTED RISK SUBTOTAL (deterministic, already weighted): +4.00"
            in (prompt)
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("red_flags", [None, [], "not-a-list", [None, 3]])
    async def test_malformed_flags_still_produce_a_prompt(self, red_flags):
        prompt = await TestDataFormatEdgeCases()._consultant_prompt(
            {"red_flags": red_flags, "pre_screening_result": "PASS"}, quick_mode=False
        )
        assert "RED FLAGS (Pre-Screening Results)" in prompt

    @pytest.mark.asyncio
    async def test_state_red_flags_are_not_mutated(self):
        flags = [dict(self.CMIC_FLAG)]
        snapshot = copy.deepcopy(flags)
        await TestDataFormatEdgeCases()._consultant_prompt(
            {"red_flags": flags, "pre_screening_result": "PASS"}, quick_mode=False
        )
        assert flags == snapshot

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
        from src.llms import get_consultant_llm

        with patch("src.llms.config") as mock_config:
            mock_config.enable_consultant = True
            mock_config.get_openai_api_key.return_value = ""
            llm = get_consultant_llm()

            # Should return None (empty key treated same as missing)
            assert llm is None

    def test_consultant_enable_flag_disabled(self):
        """Test that consultant is disabled when enable_consultant=False."""
        from src.llms import get_consultant_llm

        with patch("src.llms.config") as mock_config:
            mock_config.enable_consultant = False
            llm = get_consultant_llm()
            assert llm is None, "Should be disabled when enable_consultant=False"

    def test_auditor_openai_fallback_rejects_non_openai_primary(self):
        """The fallback path must not apply OpenAI-only params to Gemini clients."""
        mock_llm = Mock()
        mock_llm.model_name = "gemini-3-flash-preview"

        with pytest.raises(ValueError, match="OpenAI"):
            _create_openai_responses_fallback_llm(mock_llm)

    def test_openai_fallback_uses_custom_base_and_chat_completions(self):
        """A custom OPENAI_API_BASE routes the fallback to Chat Completions."""
        from pydantic import SecretStr

        from src.agents import consultant_nodes as cn

        fake_cls = MagicMock()
        with (
            patch.object(cn.support, "infer_provider_name", return_value="openai"),
            patch.object(cn.support, "get_model_name", return_value="kimi-k2"),
            patch.object(
                cn.settings_config, "openai_api_base", "https://api.moonshot.cn/v1"
            ),
            patch.object(cn.settings_config, "openai_api_key", SecretStr("k")),
            patch("langchain_openai.ChatOpenAI", fake_cls),
        ):
            _create_openai_responses_fallback_llm(Mock())

        _, kwargs = fake_cls.call_args
        assert kwargs["base_url"] == "https://api.moonshot.cn/v1"
        assert kwargs["api_key"] == "k"
        assert "use_responses_api" not in kwargs
        assert "output_version" not in kwargs

    def test_openai_fallback_default_uses_responses_api(self):
        """With no custom base the fallback keeps the OpenAI Responses API."""
        from pydantic import SecretStr

        from src.agents import consultant_nodes as cn

        fake_cls = MagicMock()
        with (
            patch.object(cn.support, "infer_provider_name", return_value="openai"),
            patch.object(cn.support, "get_model_name", return_value="gpt-5.4"),
            patch.object(cn.settings_config, "openai_api_base", ""),
            patch.object(cn.settings_config, "openai_api_key", SecretStr("k")),
            patch("langchain_openai.ChatOpenAI", fake_cls),
        ):
            _create_openai_responses_fallback_llm(Mock())

        _, kwargs = fake_cls.call_args
        assert "base_url" not in kwargs
        assert kwargs["use_responses_api"] is True
        assert kwargs["output_version"] == "responses/v1"


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
        """Consultant deadlines should be passed into the shared LLM runtime."""
        mock_llm = Mock()
        seen_timeouts: list[float] = []

        async def mock_invoke_hang(*args, **kwargs):
            seen_timeouts.append(kwargs["overall_timeout_seconds"])
            raise TimeoutError("shared runtime timeout")

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
        assert seen_timeouts
        assert seen_timeouts[0] <= 0.05
        status = result["artifact_statuses"]["consultant_review"]
        assert status["error_kind"] == "timeout"
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
                "consultant_review": {"complete": True, "ok": True},
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


class TestConsultantQuickEnvelope:
    """P1-5: Quick-mode Consultant total-deadline envelope is tight (35s default)."""

    def test_default_quick_total_timeout_is_35s(self):
        """The shipped default must be 35s — earlier 60s value hid hung calls.
        Operators can still override via CONSULTANT_QUICK_TOTAL_TIMEOUT_SECONDS.

        Reads the declared field default rather than instantiating ``Settings()``:
        instantiation loads the developer's ``.env``, so an operator who sets the
        documented override made this test fail on a *correct* configuration.
        Mirrors ``TestApiRetryAttemptsDefault._field_default``.
        """
        from src.config import Settings

        default = Settings.model_fields[
            "consultant_quick_total_timeout_seconds"
        ].default
        assert default == 35.0

    @pytest.mark.asyncio
    async def test_quick_mode_deadline_flows_into_runtime(self, monkeypatch):
        """The Consultant node passes the quick-mode total budget through as
        the per-call timeout ceiling to invoke_with_rate_limit_handling."""
        from src.agents import consultant_nodes as cn_mod
        from src.config import config as config_singleton

        monkeypatch.setattr(
            config_singleton, "consultant_quick_total_timeout_seconds", 35.0
        )
        # Force the per-call cap above the total so the total is the binding
        # constraint and shows up in the recorded timeout.
        monkeypatch.setattr(cn_mod, "CONSULTANT_CALL_TIMEOUT_SECONDS", 999.0)

        seen_timeouts: list[float] = []

        async def fake_invoke(*args, **kwargs):
            seen_timeouts.append(float(kwargs["overall_timeout_seconds"]))
            raise TimeoutError("stubbed")

        monkeypatch.setattr(
            "src.agents.runtime.invoke_with_rate_limit_handling", fake_invoke
        )

        mock_prompt = Mock()
        mock_prompt.system_message = "You are a consultant."
        mock_prompt.agent_name = "External Consultant"
        with patch("src.prompts.get_prompt", return_value=mock_prompt):
            node = create_consultant_node(Mock(), "consultant", quick_mode=True)
            state = {
                "company_of_interest": "TST",
                "company_name": "Test Co",
                "market_report": "x",
                "sentiment_report": "x",
                "news_report": "x",
                "fundamentals_report": "x",
                "investment_debate_state": {"history": "x"},
                "investment_plan": "BUY",
            }
            config = RunnableConfig(
                configurable={"context": Mock(trade_date="2026-05-15")}
            )
            await node(state, config)

        assert seen_timeouts, "consultant node never invoked the LLM runtime"
        # First call: the full quick budget is still available, so the
        # passed timeout should be at-or-below the 35s envelope.
        assert seen_timeouts[0] <= 35.0
        assert seen_timeouts[0] > 0.0

    @pytest.mark.asyncio
    async def test_env_override_takes_precedence(self, monkeypatch):
        """If CONSULTANT_QUICK_TOTAL_TIMEOUT_SECONDS is set lower at runtime,
        the consultant node honors it (the deadline shrinks, not grows)."""
        from src.agents import consultant_nodes as cn_mod
        from src.config import config as config_singleton

        monkeypatch.setattr(
            config_singleton, "consultant_quick_total_timeout_seconds", 10.0
        )
        monkeypatch.setattr(cn_mod, "CONSULTANT_CALL_TIMEOUT_SECONDS", 999.0)

        seen_timeouts: list[float] = []

        async def fake_invoke(*args, **kwargs):
            seen_timeouts.append(float(kwargs["overall_timeout_seconds"]))
            raise TimeoutError("stubbed")

        monkeypatch.setattr(
            "src.agents.runtime.invoke_with_rate_limit_handling", fake_invoke
        )

        mock_prompt = Mock()
        mock_prompt.system_message = "x"
        mock_prompt.agent_name = "External Consultant"
        with patch("src.prompts.get_prompt", return_value=mock_prompt):
            node = create_consultant_node(Mock(), "consultant", quick_mode=True)
            await node(
                {
                    "company_of_interest": "TST",
                    "company_name": "Test Co",
                    "market_report": "x",
                    "sentiment_report": "x",
                    "news_report": "x",
                    "fundamentals_report": "x",
                    "investment_debate_state": {"history": "x"},
                    "investment_plan": "BUY",
                },
                RunnableConfig(configurable={"context": Mock(trade_date="2026-05-15")}),
            )

        assert seen_timeouts[0] <= 10.0


class TestTruncatedFinalResponse:
    """1088.HK 2026-08-02: a review cut off at the token cap is a fragment.

    The model burned its whole completion budget on hidden reasoning and
    returned 46 characters of preamble. A fragment is truthy, so the existing
    empty-content fallback could not see it and the fragment was persisted as a
    complete consultant review (``ok=True``). The provider says so in
    ``finish_reason``; the loop now reads it.
    """

    @staticmethod
    def _loop_kwargs(fake_invoke, **overrides):
        from src.agents.consultant_tool_loop import ConsultantToolLoopPolicy

        kwargs = {
            "active_llm": "active",
            "fallback_llm": "fallback",
            "messages": [],
            "tools_by_name": {"spot_check_metric": object()},
            "policy": ConsultantToolLoopPolicy(
                max_tool_iterations=1,
                max_tool_calls_per_turn=4,
                deadline=time.monotonic() + 60,
                total_timeout=60,
            ),
            "invoke_with_deadline": fake_invoke,
            "agent_name": "External Consultant",
            "agent_key": "consultant",
            "ticker": "1088.HK",
        }
        kwargs.update(overrides)
        return kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metadata",
        [
            {"finish_reason": "length"},
            {"status": "incomplete", "incomplete_details": {"reason": "max_tokens"}},
        ],
    )
    async def test_capped_fragment_is_replaced_by_a_forced_synthesis(self, metadata):
        from src.agents.consultant_tool_loop import run_bounded_consultant_loop

        fragment = SimpleNamespace(
            content="I'll spot-check the decision-critical conflict",
            tool_calls=[],
            response_metadata=metadata,
        )
        synthesis = SimpleNamespace(
            content="FINAL CONSULTANT VERDICT: CONDITIONAL APPROVAL",
            tool_calls=[],
            response_metadata={"finish_reason": "stop"},
        )
        invoked = []

        async def fake_invoke(llm, _messages):
            invoked.append(llm)
            return fragment if llm == "active" else synthesis

        result = await run_bounded_consultant_loop(**self._loop_kwargs(fake_invoke))

        assert invoked == ["active", "fallback"]
        assert result.content == "FINAL CONSULTANT VERDICT: CONDITIONAL APPROVAL"

    @pytest.mark.asyncio
    async def test_fragment_is_kept_when_the_retry_returns_nothing(self):
        """Never trade a usable fragment for an empty response."""
        from src.agents.consultant_tool_loop import run_bounded_consultant_loop

        fragment = SimpleNamespace(
            content="Partial review text",
            tool_calls=[],
            response_metadata={"finish_reason": "length"},
        )
        empty = SimpleNamespace(
            content="", tool_calls=[], response_metadata={"finish_reason": "stop"}
        )

        async def fake_invoke(llm, _messages):
            return fragment if llm == "active" else empty

        result = await run_bounded_consultant_loop(**self._loop_kwargs(fake_invoke))

        assert result.content == "Partial review text"

    @pytest.mark.asyncio
    async def test_clean_finish_does_not_pay_for_a_second_call(self):
        from src.agents.consultant_tool_loop import run_bounded_consultant_loop

        clean = SimpleNamespace(
            content="FINAL CONSULTANT VERDICT: APPROVED",
            tool_calls=[],
            response_metadata={"finish_reason": "stop"},
        )
        invoked = []

        async def fake_invoke(llm, _messages):
            invoked.append(llm)
            return clean

        result = await run_bounded_consultant_loop(**self._loop_kwargs(fake_invoke))

        assert invoked == ["active"]
        assert result.content == "FINAL CONSULTANT VERDICT: APPROVED"

    @pytest.mark.asyncio
    async def test_failed_resynthesis_keeps_the_fragment(self):
        """The re-ask is strictly additive — it must not cost us the fragment.

        Widening the trigger from "empty" to "empty or truncated" means a
        deadline-exhausted re-ask could otherwise turn a degraded-but-usable
        review into a failed artifact.
        """
        from src.agents.consultant_tool_loop import run_bounded_consultant_loop

        fragment = SimpleNamespace(
            content="Partial but usable review",
            tool_calls=[],
            response_metadata={"finish_reason": "length"},
        )

        async def fake_invoke(llm, _messages):
            if llm == "active":
                return fragment
            raise TimeoutError("consultant deadline exhausted")

        result = await run_bounded_consultant_loop(**self._loop_kwargs(fake_invoke))

        assert result.content == "Partial but usable review"

    @pytest.mark.asyncio
    async def test_failed_resynthesis_still_raises_when_there_is_no_content(self):
        """The pre-existing empty-content path keeps propagating failures."""
        from src.agents.consultant_tool_loop import run_bounded_consultant_loop

        empty = SimpleNamespace(content="", tool_calls=[], response_metadata={})

        async def fake_invoke(llm, _messages):
            if llm == "active":
                return empty
            raise TimeoutError("consultant deadline exhausted")

        with pytest.raises(TimeoutError):
            await run_bounded_consultant_loop(**self._loop_kwargs(fake_invoke))

    @pytest.mark.asyncio
    async def test_provider_response_object_is_appended_verbatim(self):
        """Tool turns must replay the provider's own message object.

        Vendor-specific fields (Kimi's ``reasoning_content``, any future
        equivalent) live on the response object. Reconstructing a message from
        ``content`` + ``tool_calls`` would silently drop them, so the contract
        this loop owns is: append what the provider returned, unmodified.
        """
        from src.agents.consultant_tool_loop import run_bounded_consultant_loop

        tool_turn = SimpleNamespace(
            content="",
            tool_calls=[
                {"name": "spot_check_metric", "args": {"ticker": "X"}, "id": "call_1"}
            ],
            response_metadata={"finish_reason": "tool_calls"},
            reasoning_content="hidden chain of thought",
        )
        final = SimpleNamespace(
            content="FINAL CONSULTANT VERDICT: APPROVED",
            tool_calls=[],
            response_metadata={"finish_reason": "stop"},
        )
        responses = iter((tool_turn, final))
        seen_messages = []

        async def fake_invoke(_llm, loop_messages):
            seen_messages.append(list(loop_messages))
            return next(responses)

        class _Tool:
            async def ainvoke(self, _args):
                return "42"

        class PassthroughService:
            async def execute(self, invocation, runner):
                return SimpleNamespace(value=await runner(invocation.args))

        result = await run_bounded_consultant_loop(
            **self._loop_kwargs(
                fake_invoke,
                tools_by_name={"spot_check_metric": _Tool()},
                tool_service_getter=lambda: PassthroughService(),
            )
        )

        assert result.content == "FINAL CONSULTANT VERDICT: APPROVED"
        second_turn = seen_messages[1]
        assert tool_turn in second_turn, "assistant turn was not replayed"
        replayed = second_turn[second_turn.index(tool_turn)]
        assert replayed is tool_turn
        assert replayed.reasoning_content == "hidden chain of thought"
        assert replayed.tool_calls == tool_turn.tool_calls

    @pytest.mark.asyncio
    async def test_metadata_free_response_is_not_treated_as_truncated(self):
        """Synthetic AIMessages (tests, non-streaming providers) stay clean."""
        from src.agents.consultant_tool_loop import run_bounded_consultant_loop

        bare = SimpleNamespace(content="A complete review.", tool_calls=[])
        invoked = []

        async def fake_invoke(llm, _messages):
            invoked.append(llm)
            return bare

        result = await run_bounded_consultant_loop(**self._loop_kwargs(fake_invoke))

        assert invoked == ["active"]
        assert result.content == "A complete review."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
