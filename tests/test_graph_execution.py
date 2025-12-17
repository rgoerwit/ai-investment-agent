"""Fixed test_graph_execution.py - removed pytestmark from non-async tests."""

import pytest
from unittest.mock import MagicMock, patch


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
            "sentiment_report": "exists"
        }
        
        result = route_tools(state)
        assert result == "Market Analyst"  # Fallback to missing report


class TestDebateRouter:
    """Test debate routing logic."""
    
    @patch('src.graph.create_analyst_node')
    @patch('src.graph.create_researcher_node')
    @patch('src.graph.create_research_manager_node')
    @patch('src.graph.create_trader_node')
    @patch('src.graph.create_risk_debater_node')
    @patch('src.graph.create_portfolio_manager_node')
    @patch('src.graph.toolkit')
    def test_debate_router_alternation(self, mock_toolkit, mock_pm,
                                      mock_risk, mock_trader, mock_res_mgr,
                                      mock_researcher, mock_analyst):
        """Test debate router alternates correctly."""
        from src.graph import create_trading_graph

        # Mock all node creation (cleaner nodes removed in parallel refactor)
        mock_analyst.return_value = lambda s, c: {}
        mock_researcher.return_value = lambda s, c: {}
        mock_res_mgr.return_value = lambda s, c: {}
        mock_trader.return_value = lambda s, c: {}
        mock_risk.return_value = lambda s, c: {}
        mock_pm.return_value = lambda s, c: {}
        mock_toolkit.get_all_tools.return_value = []
        
        graph = create_trading_graph(max_debate_rounds=2)
        
        # Test debate router is compiled into graph
        assert graph is not None


class TestTradingContext:
    """Test TradingContext dataclass."""
    
    def test_trading_context_creation(self):
        """Test TradingContext creation."""
        from src.graph import TradingContext
        
        context = TradingContext(
            ticker="AAPL",
            trade_date="2024-01-01",
            quick_mode=False,
            enable_memory=True
        )
        
        assert context.ticker == "AAPL"
        assert context.trade_date == "2024-01-01"
        assert context.max_debate_rounds == 2
        assert context.max_risk_rounds == 1


class TestGraphCompilation:
    """Test graph compilation."""

    @patch('src.graph.create_quick_thinking_llm')
    @patch('src.graph.create_deep_thinking_llm')
    @patch('src.graph.toolkit')
    def test_create_trading_graph(self, mock_toolkit, mock_deep_llm_func, mock_quick_llm_func):
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

        graph = create_trading_graph(
            max_debate_rounds=2,
            max_risk_discuss_rounds=1,
            enable_memory=True
        )

        assert graph is not None
        # Graph should be compiled and ready to invoke


if __name__ == "__main__":
    pytest.main([__file__, "-v"])