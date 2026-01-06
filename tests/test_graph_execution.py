"""Fixed test_graph_execution.py - removed pytestmark from non-async tests."""

from unittest.mock import MagicMock, patch

import pytest


class TestGraphRouting:
    """Test graph routing functions."""

    def test_should_continue_analyst_with_tools(self):
        """Test routing when analyst has tool calls."""
        from src.graph import should_continue_analyst

        mock_message = MagicMock()
        mock_message.tool_calls = ["tool1"]

        state = {"messages": [mock_message]}
        config = {}

        result = should_continue_analyst(state, config)
        assert result == "tools"

    def test_should_continue_analyst_without_tools(self):
        """Test routing when analyst has no tool calls."""
        from src.graph import should_continue_analyst

        mock_message = MagicMock()
        mock_message.tool_calls = []

        state = {"messages": [mock_message]}
        config = {}

        result = should_continue_analyst(state, config)
        assert result == "continue"

    def test_route_tools_with_sender(self):
        """Test tool routing with sender field."""
        from src.graph import route_tools

        state = {"sender": "market_analyst"}

        result = route_tools(state)
        assert result == "Market Analyst"

    def test_route_tools_fallback(self):
        """Test tool routing fallback logic."""
        from src.graph import route_tools

        state = {
            "sender": "",
            "messages": [],
            "market_report": None,
            "sentiment_report": "exists",
        }

        result = route_tools(state)
        assert result == "Market Analyst"  # Fallback to missing report


class TestDebateRouter:
    """Test debate routing logic."""

    @patch("src.graph.create_agent_tool_node")
    @patch("src.graph.create_analyst_node")
    @patch("src.graph.create_researcher_node")
    @patch("src.graph.create_research_manager_node")
    @patch("src.graph.create_trader_node")
    @patch("src.graph.create_risk_debater_node")
    @patch("src.graph.create_portfolio_manager_node")
    @patch("src.graph.toolkit")
    def test_debate_router_alternation(
        self,
        mock_toolkit,
        mock_pm,
        mock_risk,
        mock_trader,
        mock_res_mgr,
        mock_researcher,
        mock_analyst,
        mock_tool_node,
    ):
        """Test debate router alternates correctly."""
        from src.graph import create_trading_graph

        # Mock all node creation (cleaner nodes removed in parallel refactor)
        mock_analyst.return_value = lambda s, c: {}
        mock_researcher.return_value = lambda s, c: {}
        mock_res_mgr.return_value = lambda s, c: {}
        mock_trader.return_value = lambda s, c: {}
        mock_risk.return_value = lambda s, c: {}
        mock_pm.return_value = lambda s, c: {}
        mock_tool_node.return_value = lambda s, c: {}
        mock_toolkit.get_all_tools.return_value = []

        graph = create_trading_graph(max_debate_rounds=2)

        # Test debate router is compiled into graph
        assert graph is not None


class TestSyncCheckRouter:
    """Test sync_check_router for parallel debate fan-out."""

    @patch("src.graph.config")
    def test_sync_check_returns_end_when_incomplete(self, mock_config):
        """Test router returns __end__ when not all analysts complete."""
        from src.graph import sync_check_router

        mock_config.enable_consultant = False

        state = {
            "market_report": "done",
            "sentiment_report": "",  # Not complete
            "news_report": "done",
            "pre_screening_result": "PASS",
        }
        config = {}

        result = sync_check_router(state, config)
        assert result == "__end__"

    @patch("src.graph.config")
    def test_sync_check_returns_pm_on_reject(self, mock_config):
        """Test router returns Portfolio Manager on REJECT."""
        from src.graph import sync_check_router

        mock_config.enable_consultant = False

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "pre_screening_result": "REJECT",
        }
        config = {}

        result = sync_check_router(state, config)
        assert result == "Portfolio Manager"

    @patch("src.graph.config")
    def test_sync_check_returns_list_for_parallel_r1(self, mock_config):
        """Test router returns list for parallel Bull/Bear R1 on PASS."""
        from src.graph import sync_check_router

        mock_config.enable_consultant = False

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "pre_screening_result": "PASS",
        }
        config = {}

        result = sync_check_router(state, config)
        assert isinstance(result, list)
        assert "Bull Researcher R1" in result
        assert "Bear Researcher R1" in result
        assert len(result) == 2


class TestAuditorIntegration:
    """Test auditor node integration with graph routing."""

    @patch("src.graph.config")
    def test_is_auditor_enabled_when_consultant_disabled(self, mock_config):
        """Test _is_auditor_enabled returns False when consultant disabled."""
        from src.graph import _is_auditor_enabled

        mock_config.enable_consultant = False
        mock_config.get_openai_api_key.return_value = "test-key"

        assert _is_auditor_enabled() is False

    @patch("src.graph.config")
    def test_is_auditor_enabled_when_no_api_key(self, mock_config):
        """Test _is_auditor_enabled returns False when API key missing."""
        from src.graph import _is_auditor_enabled

        mock_config.enable_consultant = True
        mock_config.get_openai_api_key.return_value = None

        assert _is_auditor_enabled() is False

    @patch("src.graph.config")
    def test_is_auditor_enabled_when_all_conditions_met(self, mock_config):
        """Test _is_auditor_enabled returns True when all conditions met."""
        from src.graph import _is_auditor_enabled

        mock_config.enable_consultant = True
        mock_config.get_openai_api_key.return_value = "test-key"

        assert _is_auditor_enabled() is True

    @patch("src.graph._is_auditor_enabled")
    def test_fan_out_includes_auditor_when_enabled(self, mock_auditor_enabled):
        """Test fan_out_to_analysts includes Auditor when enabled."""
        from src.graph import fan_out_to_analysts

        mock_auditor_enabled.return_value = True

        result = fan_out_to_analysts({}, {})
        assert "Auditor" in result
        assert len(result) == 7  # 6 analysts + Auditor

    @patch("src.graph._is_auditor_enabled")
    def test_fan_out_excludes_auditor_when_disabled(self, mock_auditor_enabled):
        """Test fan_out_to_analysts excludes Auditor when disabled."""
        from src.graph import fan_out_to_analysts

        mock_auditor_enabled.return_value = False

        result = fan_out_to_analysts({}, {})
        assert "Auditor" not in result
        assert len(result) == 6

    @patch("src.graph._is_auditor_enabled")
    def test_sync_check_waits_for_auditor_when_enabled(self, mock_auditor_enabled):
        """Test sync_check_router waits for auditor_report when enabled."""
        from src.graph import sync_check_router

        mock_auditor_enabled.return_value = True

        # All reports present except auditor_report
        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "pre_screening_result": "PASS",
            "auditor_report": "",  # Empty = not done
        }

        result = sync_check_router(state, {})
        assert result == "__end__"  # Should wait

    @patch("src.graph._is_auditor_enabled")
    def test_sync_check_proceeds_when_auditor_complete(self, mock_auditor_enabled):
        """Test sync_check_router proceeds when auditor_report complete."""
        from src.graph import sync_check_router

        mock_auditor_enabled.return_value = True

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "pre_screening_result": "PASS",
            "auditor_report": "Forensic audit complete",
        }

        result = sync_check_router(state, {})
        assert isinstance(result, list)
        assert "Bull Researcher R1" in result

    def test_route_tools_for_auditor(self):
        """Test route_tools returns correct node for auditor."""
        from src.graph import route_tools

        state = {"sender": "global_forensic_auditor"}
        result = route_tools(state)
        assert result == "Auditor"


class TestTradingContext:
    """Test TradingContext dataclass."""

    def test_trading_context_creation(self):
        """Test TradingContext creation."""
        from src.graph import TradingContext

        context = TradingContext(
            ticker="AAPL", trade_date="2024-01-01", quick_mode=False, enable_memory=True
        )

        assert context.ticker == "AAPL"
        assert context.trade_date == "2024-01-01"
        assert context.max_debate_rounds == 2
        assert context.max_risk_rounds == 1

    def test_trading_context_quick_mode(self):
        """Test TradingContext in quick mode."""
        from src.graph import TradingContext

        context = TradingContext(
            ticker="AAPL",
            trade_date="2024-01-01",
            quick_mode=True,
            max_debate_rounds=1,  # Quick mode uses 1 round
        )

        assert context.quick_mode is True
        assert context.max_debate_rounds == 1


class TestGraphCompilation:
    """Test graph compilation."""

    @patch("src.graph.create_agent_tool_node")
    @patch("src.graph.create_quick_thinking_llm")
    @patch("src.graph.create_deep_thinking_llm")
    @patch("src.graph.toolkit")
    def test_create_trading_graph(
        self, mock_toolkit, mock_deep_llm_func, mock_quick_llm_func, mock_tool_node
    ):
        """Test trading graph creation."""
        from src.graph import create_trading_graph

        # Mock the LLM creation functions to return mock LLMs
        mock_quick_llm = MagicMock()
        mock_deep_llm = MagicMock()
        mock_quick_llm_func.return_value = mock_quick_llm
        mock_deep_llm_func.return_value = mock_deep_llm

        mock_toolkit.get_technical_tools.return_value = []
        mock_toolkit.get_sentiment_tools.return_value = []
        mock_toolkit.get_news_tools.return_value = []
        mock_toolkit.get_fundamental_tools.return_value = []
        mock_toolkit.get_all_tools.return_value = []

        # Mock create_agent_tool_node to return a dummy function
        mock_tool_node.return_value = lambda s, c: {}

        graph = create_trading_graph(
            max_debate_rounds=2, max_risk_discuss_rounds=1, enable_memory=True
        )

        assert graph is not None
        # Graph should be compiled and ready to invoke


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
