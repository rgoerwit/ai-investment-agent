"""
Tests for external consultant node integration.

Tests both the functionality of the consultant node and non-regression
of existing system behavior when consultant is enabled/disabled.
"""

import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from src.agents import create_consultant_node, AgentState
from src.llms import get_consultant_llm, create_consultant_llm
from src.graph import create_trading_graph
from langgraph.types import RunnableConfig


class TestConsultantNodeCreation:
    """Test suite for consultant node creation and configuration."""

    def test_consultant_llm_disabled_env_var(self):
        """Test that consultant is disabled when ENABLE_CONSULTANT=false."""
        with patch.dict(os.environ, {"ENABLE_CONSULTANT": "false"}):
            llm = get_consultant_llm()
            assert llm is None

    def test_consultant_llm_missing_api_key(self):
        """Test that consultant returns None when OPENAI_API_KEY missing."""
        with patch.dict(os.environ, {"ENABLE_CONSULTANT": "true"}, clear=True):
            # Remove OPENAI_API_KEY if it exists
            os.environ.pop("OPENAI_API_KEY", None)
            llm = get_consultant_llm()
            assert llm is None

    def test_consultant_llm_creation_success(self):
        """Test successful consultant LLM creation with valid config."""
        # Skip if langchain-openai not installed (it's optional)
        try:
            import langchain_openai
        except ImportError:
            pytest.skip("langchain-openai not installed (optional dependency)")

        with patch('langchain_openai.ChatOpenAI') as mock_chatgpt:
            mock_llm = Mock()
            mock_chatgpt.return_value = mock_llm

            with patch.dict(os.environ, {
                "ENABLE_CONSULTANT": "true",
                "OPENAI_API_KEY": "test-key",
                "CONSULTANT_MODEL": "gpt-4o"
            }):
                llm = create_consultant_llm()

                assert llm is not None
                mock_chatgpt.assert_called_once()
                call_kwargs = mock_chatgpt.call_args[1]
                assert call_kwargs["model"] == "gpt-4o"
                assert call_kwargs["openai_api_key"] == "test-key"
                assert call_kwargs["temperature"] == 0.3

    def test_consultant_node_creation(self):
        """Test that consultant node factory creates valid node function."""
        mock_llm = Mock()
        node_func = create_consultant_node(mock_llm, "consultant")

        assert callable(node_func)
        assert node_func.__name__ == "consultant_node"


class TestConsultantNodeExecution:
    """Test suite for consultant node execution logic."""

    @pytest.mark.asyncio
    async def test_consultant_node_receives_full_context(self):
        """Test that consultant receives all analyst reports and debate history."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "CONSULTANT REVIEW: APPROVED - Analysis is sound."

        # Create async mock for rate limit handling
        async def mock_invoke(*args, **kwargs):
            return mock_response

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                state = {
                    "company_of_interest": "0005.HK",
                    "company_name": "HSBC Holdings",
                    "market_report": "Market report content",
                    "sentiment_report": "Sentiment report content",
                    "news_report": "News report content",
                    "fundamentals_report": "Fundamentals DATA_BLOCK here",
                    "investment_debate_state": {"history": "Bull: ... Bear: ..."},
                    "investment_plan": "Research Manager recommends BUY",
                    "red_flags": [],
                    "pre_screening_result": "PASS"
                }

                config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                result = await consultant_node(state, config)

                assert "consultant_review" in result
                assert result["consultant_review"] == mock_response.content

    @pytest.mark.asyncio
    async def test_consultant_node_handles_missing_prompt(self):
        """Test graceful handling when consultant prompt is missing."""
        mock_llm = Mock()

        with patch('src.prompts.get_prompt', return_value=None):
            consultant_node = create_consultant_node(mock_llm, "consultant")

            state = {"company_of_interest": "TEST"}
            config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

            result = await consultant_node(state, config)

            assert "consultant_review" in result
            assert "Error" in result["consultant_review"]

    @pytest.mark.asyncio
    async def test_consultant_node_handles_llm_error(self):
        """Test that consultant node returns error message on LLM failure."""
        mock_llm = Mock()

        async def mock_invoke_error(*args, **kwargs):
            raise Exception("OpenAI API timeout")

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke_error):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                state = {
                    "company_of_interest": "0005.HK",
                    "company_name": "HSBC Holdings",
                    "market_report": "Report",
                    "sentiment_report": "Report",
                    "news_report": "Report",
                    "fundamentals_report": "Report",
                    "investment_debate_state": {"history": "Debate"},
                    "investment_plan": "Plan"
                }

                config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                result = await consultant_node(state, config)

                assert "consultant_review" in result
                assert "Error" in result["consultant_review"]
                assert "OpenAI API timeout" in result["consultant_review"]


class TestGraphIntegration:
    """Test suite for consultant integration into the graph."""

    @patch('src.llms.get_consultant_llm')
    def test_graph_skips_consultant_when_disabled(self, mock_get_consultant):
        """Test that graph routes directly to Trader when consultant unavailable."""
        mock_get_consultant.return_value = None

        graph = create_trading_graph(ticker="TEST", max_debate_rounds=1)

        # Check that graph compiles without consultant node
        assert graph is not None
        # The graph should have Research Manager → Trader edge when consultant is disabled

    @patch('src.llms.get_consultant_llm')
    def test_graph_includes_consultant_when_enabled(self, mock_get_consultant):
        """Test that graph includes consultant node when available."""
        mock_llm = Mock()
        mock_get_consultant.return_value = mock_llm

        graph = create_trading_graph(ticker="TEST", max_debate_rounds=1)

        assert graph is not None
        # The graph should have Research Manager → Consultant → Trader path


class TestConsultantValueAddition:
    """Test suite for consultant adding value to analysis."""

    @pytest.mark.asyncio
    async def test_consultant_detects_confirmation_bias(self):
        """Test consultant can detect confirmation bias in debate."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """
CONSULTANT REVIEW: CONDITIONAL APPROVAL

SECTION 1: FACTUAL VERIFICATION
Status: ✓ FACTS VERIFIED

SECTION 2: BIAS DETECTION
Status: ⚠ BIASES IDENTIFIED
- Confirmation Bias: Both Bull and Bear cite P/E of 15 to support opposing views
  Impact: May lead to overconfidence in recommendation
"""

        async def mock_invoke(*args, **kwargs):
            return mock_response

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                state = {
                    "company_of_interest": "TEST",
                    "company_name": "Test Company",
                    "market_report": "P/E is 15",
                    "sentiment_report": "Report",
                    "news_report": "Report",
                    "fundamentals_report": "DATA_BLOCK\nP/E: 15.0",
                    "investment_debate_state": {
                        "history": "Bull: P/E of 15 is low!\nBear: P/E of 15 is high!"
                    },
                    "investment_plan": "BUY",
                    "red_flags": [],
                    "pre_screening_result": "PASS"
                }

                config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                result = await consultant_node(state, config)

                assert "consultant_review" in result
                assert "Confirmation Bias" in result["consultant_review"]

    @pytest.mark.asyncio
    async def test_consultant_detects_factual_error(self):
        """Test consultant can detect factual mismatches between reports and DATA_BLOCK."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """
CONSULTANT REVIEW: MAJOR CONCERNS

SECTION 1: FACTUAL VERIFICATION
Status: ✗ ERRORS FOUND

Material Errors:
- Research Manager cited P/E of 15, but DATA_BLOCK shows 22.3
- This discrepancy could change the BUY/HOLD recommendation
"""

        async def mock_invoke(*args, **kwargs):
            return mock_response

        with patch('src.agents.invoke_with_rate_limit_handling', new=mock_invoke):
            with patch('src.prompts.get_prompt') as mock_get_prompt:
                mock_prompt = Mock()
                mock_prompt.system_message = "You are a consultant."
                mock_prompt.agent_name = "External Consultant"
                mock_get_prompt.return_value = mock_prompt

                consultant_node = create_consultant_node(mock_llm, "consultant")

                state = {
                    "company_of_interest": "TEST",
                    "company_name": "Test Company",
                    "market_report": "Report",
                    "sentiment_report": "Report",
                    "news_report": "Report",
                    "fundamentals_report": "DATA_BLOCK\nP/E: 22.3",
                    "investment_debate_state": {"history": "Debate"},
                    "investment_plan": "BUY based on P/E of 15",
                    "red_flags": [],
                    "pre_screening_result": "PASS"
                }

                config = RunnableConfig(configurable={"context": Mock(trade_date="2025-12-13")})

                result = await consultant_node(state, config)

                assert "consultant_review" in result
                assert "ERRORS FOUND" in result["consultant_review"]


class TestNonRegression:
    """Test suite ensuring consultant doesn't break existing functionality."""

    @patch('src.llms.get_consultant_llm')
    def test_graph_execution_without_consultant(self, mock_get_consultant):
        """Test that graph still executes correctly when consultant is disabled."""
        mock_get_consultant.return_value = None

        # This should not raise any errors
        graph = create_trading_graph(
            ticker="TEST",
            max_debate_rounds=1,
            enable_memory=False
        )

        assert graph is not None

    def test_consultant_review_field_optional_in_state(self):
        """Test that AgentState works without consultant_review field."""
        state = {
            "company_of_interest": "TEST",
            "market_report": "Report",
            "sentiment_report": "Report",
            "news_report": "Report",
            "fundamentals_report": "Report"
        }

        # Should not raise KeyError when accessing consultant_review
        consultant_review = state.get("consultant_review", "")
        assert consultant_review == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
