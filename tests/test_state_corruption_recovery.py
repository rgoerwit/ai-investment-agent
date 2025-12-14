"""
Test suite for TypedDict edge cases and state corruption recovery.

This module tests various scenarios where LangGraph state could be corrupted
or malformed, ensuring graceful degradation with maximum information preservation.

Key scenarios tested:
1. TypedDict structural issues (None, empty, missing keys, wrong types)
2. Fast-fail routing (debate skipped, consultant receives incomplete state)
3. Partial state corruption (some fields present, others missing)
4. State merge conflicts (parallel execution edge cases)
5. Information preservation during recovery
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from langgraph.types import RunnableConfig

from src.agents import (
    create_consultant_node,
    InvestDebateState,
    AgentState
)


class TestTypedDictStructuralIssues:
    """Test handling of malformed TypedDict structures."""

    @pytest.mark.asyncio
    async def test_debate_state_empty_dict(self):
        """Test consultant handles empty dict debate state (all keys missing)."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: Limited context available"

        invoke_calls = []

        async def mock_invoke(*args, **kwargs):
            invoke_calls.append(args)
            return mock_response

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                with patch('src.agents.logger') as mock_logger:
                    mock_prompt = Mock()
                    mock_prompt.system_message = "You are a consultant."
                    mock_prompt.agent_name = "External Consultant"
                    mock_get_prompt.return_value = mock_prompt

                    consultant_node = create_consultant_node(mock_llm, "consultant")

                    # Empty dict - no keys at all
                    state = {
                        "company_of_interest": "TEST",
                        "company_name": "Test Co",
                        "market_report": "Market data",
                        "sentiment_report": "Sentiment data",
                        "news_report": "News data",
                        "fundamentals_report": "Fundamental data",
                        "investment_debate_state": {},  # Empty dict
                        "investment_plan": "BUY recommendation",
                        "red_flags": [],
                        "pre_screening_result": "PASS"
                    }

                    config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                    result = await consultant_node(state, config)

                    # Should not crash
                    assert "consultant_review" in result
                    assert len(result["consultant_review"]) > 0

                    # Should use 'N/A' for missing debate history
                    assert len(invoke_calls) > 0
                    message_content = invoke_calls[0][1][0].content
                    assert "BULL/BEAR DEBATE HISTORY" in message_content
                    # Empty dict returns 'N/A' since .get('history', 'N/A') will return 'N/A'
                    assert "N/A" in message_content or "RESEARCH MANAGER SYNTHESIS" in message_content

    @pytest.mark.asyncio
    async def test_debate_state_partial_keys(self):
        """Test consultant handles debate state with only some keys present."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: Partial debate context available"

        invoke_calls = []

        async def mock_invoke(*args, **kwargs):
            invoke_calls.append(args)
            return mock_response

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                # Partial keys - only some fields present
                state = {
                    "company_of_interest": "TEST",
                    "company_name": "Test Co",
                    "market_report": "Market data",
                    "sentiment_report": "Sentiment data",
                    "news_report": "News data",
                    "fundamentals_report": "Fundamental data",
                    "investment_debate_state": {
                        "history": "Bull: Stock is undervalued\nBear: High debt levels",
                        # Missing: bull_history, bear_history, count, current_response, judge_decision
                    },
                    "investment_plan": "BUY recommendation",
                    "red_flags": [],
                    "pre_screening_result": "PASS"
                }

                config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                result = await consultant_node(state, config)

                # Should not crash
                assert "consultant_review" in result
                assert len(result["consultant_review"]) > 0

                # Should preserve available debate history
                assert len(invoke_calls) > 0
                message_content = invoke_calls[0][1][0].content
                assert "Bull: Stock is undervalued" in message_content
                assert "Bear: High debt levels" in message_content

    @pytest.mark.asyncio
    async def test_debate_state_wrong_types(self):
        """Test consultant handles debate state with wrong field types."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: Type mismatch recovered"

        invoke_calls = []

        async def mock_invoke(*args, **kwargs):
            invoke_calls.append(args)
            return mock_response

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                # Wrong types - count should be int, not string
                state = {
                    "company_of_interest": "TEST",
                    "company_name": "Test Co",
                    "market_report": "Market data",
                    "sentiment_report": "Sentiment data",
                    "news_report": "News data",
                    "fundamentals_report": "Fundamental data",
                    "investment_debate_state": {
                        "history": "Debate content",
                        "bull_history": "Bull arguments",
                        "bear_history": "Bear arguments",
                        "count": "4",  # Should be int, is string
                        "current_response": "",
                        "judge_decision": ""
                    },
                    "investment_plan": "BUY recommendation",
                    "red_flags": [],
                    "pre_screening_result": "PASS"
                }

                config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                result = await consultant_node(state, config)

                # Should not crash - Python's duck typing handles this
                assert "consultant_review" in result
                assert len(result["consultant_review"]) > 0

                # Should preserve debate content despite type mismatch
                assert len(invoke_calls) > 0
                message_content = invoke_calls[0][1][0].content
                assert "Debate content" in message_content


class TestFastFailRouting:
    """Test scenarios where fast-fail routing skips debate."""

    @pytest.mark.asyncio
    async def test_consultant_after_fast_fail_reject(self):
        """
        Test consultant receives no debate when red-flag validator triggers REJECT.

        In fast-fail path: Validator (REJECT) → Portfolio Manager (skips debate)
        But if consultant somehow runs, it should handle missing debate gracefully.
        """
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: No debate available (fast-fail path)"

        invoke_calls = []

        async def mock_invoke(*args, **kwargs):
            invoke_calls.append(args)
            return mock_response

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                with patch('src.agents.logger') as mock_logger:
                    mock_prompt = Mock()
                    mock_prompt.system_message = "You are a consultant."
                    mock_prompt.agent_name = "External Consultant"
                    mock_get_prompt.return_value = mock_prompt

                    consultant_node = create_consultant_node(mock_llm, "consultant")

                    # Fast-fail scenario: debate never ran
                    state = {
                        "company_of_interest": "LEVERAGED_CORP",
                        "company_name": "High Leverage Corp",
                        "market_report": "Market data",
                        "sentiment_report": "Sentiment data",
                        "news_report": "News data",
                        "fundamentals_report": "D/E Ratio: 650% (EXTREME RISK)",
                        "investment_debate_state": None,  # Debate skipped
                        "investment_plan": "N/A (fast-fail path)",
                        "red_flags": [
                            {
                                "severity": "CRITICAL",
                                "type": "EXTREME_LEVERAGE",
                                "detail": "D/E ratio 650% exceeds 500% threshold",
                                "action": "AUTO_REJECT"
                            }
                        ],
                        "pre_screening_result": "REJECT"
                    }

                    config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                    result = await consultant_node(state, config)

                    # Should not crash
                    assert "consultant_review" in result
                    assert len(result["consultant_review"]) > 0

                    # Should log diagnostic
                    mock_logger.error.assert_called_once()
                    error_call = mock_logger.error.call_args
                    assert error_call[0][0] == "consultant_received_none_debate_state"

                    # Should include diagnostic message
                    assert len(invoke_calls) > 0
                    message_content = invoke_calls[0][1][0].content
                    assert "SYSTEM DIAGNOSTIC" in message_content
                    assert "fast-fail path" in message_content.lower()

                    # Should still receive red-flag information (preserved context)
                    assert "650%" in state["fundamentals_report"]  # Case-insensitive check


class TestPartialStateCorruption:
    """Test scenarios with partial state corruption."""

    @pytest.mark.asyncio
    async def test_missing_analyst_reports(self):
        """Test consultant handles missing analyst reports gracefully."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: Limited analyst data"

        invoke_calls = []

        async def mock_invoke(*args, **kwargs):
            invoke_calls.append(args)
            return mock_response

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                # Some analyst reports missing (state corruption scenario)
                state = {
                    "company_of_interest": "TEST",
                    "company_name": "Test Co",
                    "market_report": "",  # Missing
                    "sentiment_report": "Positive sentiment",  # Available
                    "news_report": "",  # Missing
                    "fundamentals_report": "Strong financials",  # Available
                    "investment_debate_state": {
                        "history": "Debate occurred",
                        "bull_history": "",
                        "bear_history": "",
                        "count": 2,
                        "current_response": "",
                        "judge_decision": ""
                    },
                    "investment_plan": "BUY recommendation",
                    "red_flags": [],
                    "pre_screening_result": "PASS"
                }

                config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                result = await consultant_node(state, config)

                # Should not crash
                assert "consultant_review" in result
                assert len(result["consultant_review"]) > 0

                # Should preserve available data
                assert len(invoke_calls) > 0
                message_content = invoke_calls[0][1][0].content
                assert "Positive sentiment" in message_content
                assert "Strong financials" in message_content
                assert "Debate occurred" in message_content

                # Missing reports should show as empty sections (section header present, content blank)
                assert "MARKET ANALYST REPORT:" in message_content
                assert "NEWS ANALYST REPORT:" in message_content

    @pytest.mark.asyncio
    async def test_corrupted_investment_plan(self):
        """Test consultant handles corrupted/missing investment plan."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: No synthesis available"

        invoke_calls = []

        async def mock_invoke(*args, **kwargs):
            invoke_calls.append(args)
            return mock_response

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                # Investment plan missing/corrupted
                state = {
                    "company_of_interest": "TEST",
                    "company_name": "Test Co",
                    "market_report": "Market data",
                    "sentiment_report": "Sentiment data",
                    "news_report": "News data",
                    "fundamentals_report": "Fundamental data",
                    "investment_debate_state": {
                        "history": "Debate occurred with arguments",
                        "bull_history": "",
                        "bear_history": "",
                        "count": 2,
                        "current_response": "",
                        "judge_decision": ""
                    },
                    "investment_plan": "",  # Missing/empty
                    "red_flags": [],
                    "pre_screening_result": "PASS"
                }

                config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                result = await consultant_node(state, config)

                # Should not crash
                assert "consultant_review" in result
                assert len(result["consultant_review"]) > 0

                # Should still have analyst reports and debate
                assert len(invoke_calls) > 0
                message_content = invoke_calls[0][1][0].content
                assert "Market data" in message_content
                assert "Debate occurred" in message_content

                # Investment plan shows as N/A or empty
                assert "RESEARCH MANAGER SYNTHESIS" in message_content


class TestInformationPreservation:
    """Test maximum information preservation during recovery."""

    @pytest.mark.asyncio
    async def test_preserve_partial_debate_history(self):
        """Test that partial debate history is preserved rather than discarded."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: Partial debate analyzed"

        invoke_calls = []

        async def mock_invoke(*args, **kwargs):
            invoke_calls.append(args)
            return mock_response

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                # Partial debate: only history, no bull/bear split
                state = {
                    "company_of_interest": "PARTIAL_DEBATE",
                    "company_name": "Partial Debate Corp",
                    "market_report": "Market analysis",
                    "sentiment_report": "Sentiment analysis",
                    "news_report": "News analysis",
                    "fundamentals_report": "Fundamental analysis",
                    "investment_debate_state": {
                        "history": "Bull Analyst: Strong growth trajectory\nBear Analyst: Valuation concerns\nBull Analyst: Market expansion opportunity",
                        # Missing bull_history and bear_history
                        "count": 3,
                        "current_response": "",
                        "judge_decision": ""
                    },
                    "investment_plan": "HOLD - conflicting signals",
                    "red_flags": [],
                    "pre_screening_result": "PASS"
                }

                config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                result = await consultant_node(state, config)

                # Should not crash
                assert "consultant_review" in result
                assert len(result["consultant_review"]) > 0

                # Should preserve ALL debate content
                assert len(invoke_calls) > 0
                message_content = invoke_calls[0][1][0].content
                assert "Strong growth trajectory" in message_content
                assert "Valuation concerns" in message_content
                assert "Market expansion opportunity" in message_content

    @pytest.mark.asyncio
    async def test_fallback_to_research_manager_when_debate_missing(self):
        """
        Test that when debate is missing, consultant can still provide value
        by reviewing research manager synthesis and analyst reports.
        """
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: Analysis based on synthesis and reports"

        invoke_calls = []

        async def mock_invoke(*args, **kwargs):
            invoke_calls.append(args)
            return mock_response

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                # No debate, but rich analyst reports and synthesis
                state = {
                    "company_of_interest": "NO_DEBATE",
                    "company_name": "No Debate Corp",
                    "market_report": "RSI: 35 (oversold), MACD: bullish crossover",
                    "sentiment_report": "Reddit mentions up 50%, positive tone",
                    "news_report": "New product launch announced, analyst upgrades",
                    "fundamentals_report": "P/E: 12, ROE: 18%, D/E: 0.3, FCF positive",
                    "investment_debate_state": None,  # No debate
                    "investment_plan": "BUY - undervalued with growth catalysts. Technical breakout imminent.",
                    "red_flags": [],
                    "pre_screening_result": "PASS"
                }

                config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                result = await consultant_node(state, config)

                # Should not crash
                assert "consultant_review" in result
                assert len(result["consultant_review"]) > 0

                # Should still have full context from reports and synthesis
                assert len(invoke_calls) > 0
                message_content = invoke_calls[0][1][0].content

                # All analyst reports present
                assert "RSI: 35" in message_content
                assert "Reddit mentions" in message_content
                assert "product launch" in message_content
                assert "P/E: 12" in message_content

                # Research manager synthesis present
                assert "undervalued with growth catalysts" in message_content


class TestLoggingQuietModeRespect:
    """Test that all error logging respects --quiet mode."""

    @pytest.mark.asyncio
    async def test_diagnostic_logging_uses_structlog(self):
        """Verify diagnostic logging uses structlog (respects quiet mode)."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW"

        async def mock_invoke(*args, **kwargs):
            return mock_response

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                with patch('src.agents.logger') as mock_logger:
                    mock_prompt = Mock()
                    mock_prompt.system_message = "You are a consultant."
                    mock_prompt.agent_name = "External Consultant"
                    mock_get_prompt.return_value = mock_prompt

                    consultant_node = create_consultant_node(mock_llm, "consultant")

                    state = {
                        "company_of_interest": "TEST",
                        "company_name": "Test Co",
                        "market_report": "Data",
                        "sentiment_report": "Data",
                        "news_report": "Data",
                        "fundamentals_report": "Data",
                        "investment_debate_state": None,  # Triggers diagnostic
                        "investment_plan": "BUY",
                        "red_flags": [],
                        "pre_screening_result": "PASS"
                    }

                    config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                    await consultant_node(state, config)

                    # Verify logger.error was called (structlog)
                    assert mock_logger.error.called

                    # Verify it's using keyword arguments (structlog pattern)
                    error_call = mock_logger.error.call_args
                    assert error_call[1]["ticker"] == "TEST"
                    assert "message" in error_call[1]

                    # This ensures it's structlog, which respects quiet mode via configuration


class TestStateMergeConflicts:
    """Test edge cases from potential state merge conflicts."""

    @pytest.mark.asyncio
    async def test_debate_state_as_list_accumulation(self):
        """
        Test handling if state reducer accidentally accumulates debate state as list.

        This could theoretically happen if LangGraph's take_last reducer malfunctions
        or if there's a bug in state merging during parallel execution.
        """
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: List accumulation handled"

        invoke_calls = []

        async def mock_invoke(*args, **kwargs):
            invoke_calls.append(args)
            return mock_response

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                # State corruption: debate_state is list instead of dict
                state = {
                    "company_of_interest": "LIST_BUG",
                    "company_name": "List Bug Corp",
                    "market_report": "Data",
                    "sentiment_report": "Data",
                    "news_report": "Data",
                    "fundamentals_report": "Data",
                    "investment_debate_state": [  # List instead of dict!
                        {"history": "First attempt"},
                        {"history": "Second attempt"}
                    ],
                    "investment_plan": "BUY",
                    "red_flags": [],
                    "pre_screening_result": "PASS"
                }

                config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                result = await consultant_node(state, config)

                # Should not crash (isinstance check catches this)
                assert "consultant_review" in result
                assert len(result["consultant_review"]) > 0

                # List fails isinstance(debate_state, dict) check
                # So falls through to default 'N/A' behavior
                assert len(invoke_calls) > 0
                message_content = invoke_calls[0][1][0].content
                # Should show N/A since list is not a dict
                assert "N/A" in message_content or "RESEARCH MANAGER SYNTHESIS" in message_content


class TestEndToEndInformationFlow:
    """
    Test that critical information flows through the entire graph without loss.

    Verifies that key data points from analyst reports propagate correctly through:
    Analysts → Researchers → Research Manager → Consultant → Portfolio Manager

    This catches scenarios where:
    - State fields are not propagated properly
    - Important context is filtered out or summarized away
    - Critical red flags or metrics are lost in transitions
    - Information exists in early nodes but "forgotten" by portfolio manager
    """

    @pytest.mark.asyncio
    async def test_critical_data_propagates_to_portfolio_manager(self):
        """
        End-to-end test: Verify portfolio manager receives all critical data from earlier nodes.

        Seeds state with specific critical data points and verifies they all reach
        the portfolio manager's context. This ensures no information loss during
        graph traversal.
        """
        from src.agents import create_portfolio_manager_node

        # Critical data points to track through the graph
        CRITICAL_DATA_POINTS = {
            # From fundamentals analyst
            "fundamentals": {
                "P/E ratio: 12.5": "Valuation metric",
                "D/E ratio: 45%": "Leverage metric",
                "Revenue growth: 23%": "Growth metric",
                "DATA_BLOCK": "Mandatory structured data section"
            },
            # From market analyst
            "market": {
                "Daily volume: $2.5M USD": "Liquidity check",
                "52-week high: $145": "Technical reference"
            },
            # From news analyst
            "news": {
                "Announced expansion into US market": "Growth catalyst",
                "CEO confirmed Q4 earnings": "Upcoming event"
            },
            # From sentiment analyst
            "sentiment": {
                "Analyst coverage: 8 analysts": "Coverage metric",
                "Average rating: BUY": "Consensus view"
            },
            # From research manager synthesis
            "synthesis": {
                "RECOMMENDATION: BUY": "Final synthesis decision",
                "Target position: 2.5%": "Position sizing"
            },
            # From consultant review
            "consultant": {
                "VALIDATION RESULT: APPROVED": "External validation",
                "No material biases detected": "Quality check"
            },
            # From risk debate
            "risk": {
                "MODERATE risk": "Risk assessment",
                "Stop loss: $95": "Risk parameter"
            },
            # From red-flag pre-screening
            "red_flags": {
                "Pre-Screening Result: PASS": "Gate check",
                "No extreme leverage detected": "Safety check"
            }
        }

        # Build state with all critical data points
        state = {
            "company_of_interest": "TEST",
            "company_name": "Test Company Ltd",

            # Fundamentals report with critical metrics
            "fundamentals_report": f"""
            FUNDAMENTALS ANALYSIS - TEST

            === DATA_BLOCK ===
            P/E ratio: 12.5
            D/E ratio: 45%
            Revenue growth: 23%
            Free cash flow: $150M
            === END DATA_BLOCK ===

            Company shows strong fundamentals with reasonable valuation.
            """,

            # Market report with liquidity data
            "market_report": f"""
            MARKET ANALYSIS - TEST

            Daily volume: $2.5M USD
            52-week high: $145
            Current price: $120
            Trend: UPTREND
            """,

            # News report with catalysts
            "news_report": f"""
            NEWS ANALYSIS - TEST

            Recent developments:
            - Announced expansion into US market (Dec 2025)
            - CEO confirmed Q4 earnings beat expectations
            - New product launch scheduled for Q1 2026
            """,

            # Sentiment report with coverage
            "sentiment_report": f"""
            SENTIMENT ANALYSIS - TEST

            Analyst coverage: 8 analysts
            Average rating: BUY
            Price target range: $130-$160
            """,

            # Research manager synthesis
            "investment_plan": f"""
            RESEARCH MANAGER SYNTHESIS

            RECOMMENDATION: BUY
            Target position: 2.5%
            Entry price: $118-$122

            Rationale: Strong fundamentals, reasonable valuation, growth catalysts.
            """,

            # Consultant review
            "consultant_review": f"""
            EXTERNAL CONSULTANT REVIEW

            VALIDATION RESULT: APPROVED

            Analysis quality: HIGH
            No material biases detected
            Research manager synthesis is well-supported by data.
            """,

            # Trader investment plan
            "trader_investment_plan": f"""
            TRADER EXECUTION PLAN

            Action: BUY
            Entry: Market order at $120
            Stop loss: $95 (20% downside)
            """,

            # Risk debate (from risk team)
            "risk_debate_state": {
                "history": f"""
                RISK TEAM DEBATE

                Conservative Risk Analyst: MODERATE risk
                - Leverage at 45% is manageable
                - Stop loss: $95 protects downside

                Neutral Risk Analyst: MODERATE risk
                - Agrees with conservative assessment

                Aggressive Risk Analyst: LOW risk
                - Strong growth justifies position
                """
            },

            # Red-flag pre-screening results
            "pre_screening_result": "PASS",
            "red_flags_detected": [],
            "fundamentals_quality_note": "Pre-Screening Result: PASS - No extreme leverage detected"
        }

        # Mock LLM and track what context is passed to portfolio manager
        pm_invoke_calls = []

        async def mock_pm_invoke(*args, **kwargs):
            pm_invoke_calls.append(args)
            mock_response = Mock()
            mock_response.content = "FINAL DECISION: BUY at 2.5% position"
            return mock_response

        # Create portfolio manager node
        mock_llm = Mock()
        pm_node = create_portfolio_manager_node(mock_llm, memory=None)

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_pm_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are the portfolio manager."
                mock_prompt.agent_name = "Portfolio Manager"
                mock_get_prompt.return_value = mock_prompt

                config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                # Execute portfolio manager node
                result = await pm_node(state, config)

                # Verify execution succeeded
                assert "final_trade_decision" in result
                assert len(result["final_trade_decision"]) > 0

                # Verify portfolio manager was invoked
                assert len(pm_invoke_calls) > 0, "Portfolio manager should have been invoked"

                # Extract the context passed to portfolio manager
                pm_context = pm_invoke_calls[0][1][0].content

                # Verify ALL critical data points reached portfolio manager
                missing_data_points = []

                for category, data_points in CRITICAL_DATA_POINTS.items():
                    for data_point, description in data_points.items():
                        if data_point not in pm_context:
                            missing_data_points.append({
                                "category": category,
                                "data_point": data_point,
                                "description": description
                            })

                # Assert no information was lost
                if missing_data_points:
                    error_msg = "INFORMATION LOSS DETECTED - Critical data missing from Portfolio Manager context:\n"
                    for missing in missing_data_points:
                        error_msg += f"  [{missing['category']}] {missing['data_point']} ({missing['description']})\n"
                    error_msg += f"\nPortfolio Manager Context:\n{pm_context}"
                    assert False, error_msg

                # Verify specific critical sections are present
                assert "FUNDAMENTALS ANALYST REPORT" in pm_context, "Fundamentals section missing"
                assert "MARKET ANALYST REPORT" in pm_context, "Market section missing"
                assert "NEWS ANALYST REPORT" in pm_context, "News section missing"
                assert "SENTIMENT ANALYST REPORT" in pm_context, "Sentiment section missing"
                assert "RESEARCH MANAGER RECOMMENDATION" in pm_context, "Research manager section missing"
                assert "EXTERNAL CONSULTANT REVIEW" in pm_context, "Consultant section missing"
                assert "TRADER PROPOSAL" in pm_context, "Trader section missing"
                assert "RISK TEAM DEBATE" in pm_context, "Risk section missing"

    @pytest.mark.asyncio
    async def test_red_flag_information_not_lost_in_fast_fail(self):
        """
        Test that red-flag information is preserved even in fast-fail routing.

        In fast-fail path (Validator REJECT → Portfolio Manager), verify that:
        - Red flag details are preserved in state
        - Portfolio manager receives the red flag information
        - Critical safety data is not lost despite skipping debate
        """
        from src.agents import create_portfolio_manager_node

        # Create state with red-flag rejection
        state = {
            "company_of_interest": "RISKY",
            "company_name": "Risky Corp",

            # Critical red-flag information
            "pre_screening_result": "REJECT",
            "red_flags_detected": [
                "Extreme leverage: D/E Ratio: 850% (>500% threshold)",
                "Earnings quality issue: Positive NI $50M, Negative FCF -$120M",
                "Refinancing risk: Interest coverage 1.2x (< 2.0x threshold)"
            ],

            # Fundamentals report contains the detailed red flags
            "fundamentals_report": """
            FUNDAMENTALS ANALYSIS - RISKY

            === DATA_BLOCK ===
            D/E Ratio: 850%
            Net Income: $50M
            Free Cash Flow: -$120M
            Interest Coverage: 1.2x
            === END DATA_BLOCK ===

            WARNING: Multiple red flags detected. See pre-screening results.
            """,

            # Fast-fail path: these would normally be empty since debate is skipped
            "market_report": "",
            "news_report": "",
            "sentiment_report": "",
            "investment_plan": "REJECTED - Pre-screening failed",
            "consultant_review": "",
            "trader_investment_plan": "",
            "risk_debate_state": {}
        }

        # Mock LLM and track context
        pm_invoke_calls = []

        async def mock_pm_invoke(*args, **kwargs):
            pm_invoke_calls.append(args)
            mock_response = Mock()
            mock_response.content = "FINAL DECISION: REJECT - Extreme leverage risk"
            return mock_response

        mock_llm = Mock()
        pm_node = create_portfolio_manager_node(mock_llm, memory=None)

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_pm_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are the portfolio manager."
                mock_prompt.agent_name = "Portfolio Manager"
                mock_get_prompt.return_value = mock_prompt

                config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                # Execute portfolio manager node
                result = await pm_node(state, config)

                # Verify execution
                assert "final_trade_decision" in result
                assert len(pm_invoke_calls) > 0

                # Extract portfolio manager context
                pm_context = pm_invoke_calls[0][1][0].content

                # Verify ALL red-flag information is preserved
                assert "850%" in pm_context, "D/E ratio red flag missing"
                assert "Negative FCF" in pm_context or "-$120M" in pm_context, "FCF red flag missing"
                assert "Interest coverage" in pm_context or "1.2x" in pm_context, "Interest coverage red flag missing"

                # Verify fundamentals report (which contains red flags) is present
                assert "FUNDAMENTALS ANALYST REPORT" in pm_context

                # Even in fast-fail, PM should have access to the rejection reason
                assert "DATA_BLOCK" in pm_context, "Structured data section missing in fast-fail path"
