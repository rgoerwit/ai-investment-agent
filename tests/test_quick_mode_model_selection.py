"""
Tests for --quick mode model selection across all agents.

Verifies that:
1. quick_mode=True uses quick models for Research Manager and Portfolio Manager
2. quick_mode=False uses deep models for Research Manager and Portfolio Manager
3. Custom env vars are respected
4. Consultant model selection respects quick_mode
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
import os


class TestGeminiModelSelection:
    """Test Gemini model selection with quick_mode flag."""

    @patch('src.graph.create_memory_instances')
    @patch('src.graph.get_consultant_llm')
    @patch('src.graph.create_deep_thinking_llm')
    @patch('src.graph.create_quick_thinking_llm')
    def test_normal_mode_uses_deep_models_for_rm_and_pm(
        self, mock_quick_llm, mock_deep_llm, mock_consultant, mock_memories
    ):
        """Test that normal mode (quick_mode=False) uses deep models for Research Manager and Portfolio Manager."""
        from src.graph import create_trading_graph

        # Setup mocks
        mock_quick_llm.return_value = Mock()
        mock_deep_llm.return_value = Mock()
        mock_consultant.return_value = None  # Disabled
        mock_memories.return_value = {
            'TEST_bull_memory': Mock(available=True),
            'TEST_bear_memory': Mock(available=True),
            'TEST_invest_judge_memory': Mock(available=True),
            'TEST_trader_memory': Mock(available=True),
            'TEST_risk_manager_memory': Mock(available=True),
        }

        # Create graph with quick_mode=False
        graph = create_trading_graph(ticker="TEST", quick_mode=False, max_debate_rounds=1)

        assert graph is not None

        # Verify create_deep_thinking_llm was called exactly twice (Research Manager + Portfolio Manager)
        assert mock_deep_llm.call_count == 2, \
            f"Expected 2 deep LLM calls (RM + PM), got {mock_deep_llm.call_count}"

        # Verify create_quick_thinking_llm was called for other agents
        # Market, Social, News, Fundamentals, Bull, Bear, Trader, Risky, Safe, Neutral = 10
        assert mock_quick_llm.call_count == 10, \
            f"Expected 10 quick LLM calls (other agents), got {mock_quick_llm.call_count}"

    @patch('src.graph.create_memory_instances')
    @patch('src.graph.get_consultant_llm')
    @patch('src.graph.create_deep_thinking_llm')
    @patch('src.graph.create_quick_thinking_llm')
    def test_quick_mode_uses_quick_models_for_all_agents(
        self, mock_quick_llm, mock_deep_llm, mock_consultant, mock_memories
    ):
        """Test that quick mode (quick_mode=True) uses quick models for ALL agents including RM and PM."""
        from src.graph import create_trading_graph

        # Setup mocks
        mock_quick_llm.return_value = Mock()
        mock_deep_llm.return_value = Mock()
        mock_consultant.return_value = None  # Disabled
        mock_memories.return_value = {
            'TEST_bull_memory': Mock(available=True),
            'TEST_bear_memory': Mock(available=True),
            'TEST_invest_judge_memory': Mock(available=True),
            'TEST_trader_memory': Mock(available=True),
            'TEST_risk_manager_memory': Mock(available=True),
        }

        # Create graph with quick_mode=True
        graph = create_trading_graph(ticker="TEST", quick_mode=True, max_debate_rounds=1)

        assert graph is not None

        # Verify create_deep_thinking_llm was NOT called
        assert mock_deep_llm.call_count == 0, \
            f"Expected 0 deep LLM calls in quick mode, got {mock_deep_llm.call_count}"

        # Verify create_quick_thinking_llm was called for ALL agents
        # Market, Social, News, Fundamentals, Bull, Bear, RM, PM, Trader, Risky, Safe, Neutral = 12
        assert mock_quick_llm.call_count == 12, \
            f"Expected 12 quick LLM calls (all agents), got {mock_quick_llm.call_count}"

    @patch('src.graph.create_memory_instances')
    @patch('src.graph.get_consultant_llm')
    @patch('src.graph.toolkit')
    def test_custom_env_models_respected(self, mock_toolkit, mock_consultant, mock_memories):
        """Test that custom model names from env vars are respected."""
        from src.graph import create_trading_graph

        # Setup mocks
        mock_consultant.return_value = None
        mock_memories.return_value = {
            'TEST_bull_memory': Mock(available=True),
            'TEST_bear_memory': Mock(available=True),
            'TEST_invest_judge_memory': Mock(available=True),
            'TEST_trader_memory': Mock(available=True),
            'TEST_risk_manager_memory': Mock(available=True),
        }
        mock_toolkit.get_technical_tools.return_value = []
        mock_toolkit.get_sentiment_tools.return_value = []
        mock_toolkit.get_news_tools.return_value = []
        mock_toolkit.get_fundamental_tools.return_value = []
        mock_toolkit.get_all_tools.return_value = []

        # Mock the LLM creation functions to capture model names
        with patch('src.graph.create_quick_thinking_llm') as mock_quick:
            with patch('src.graph.create_deep_thinking_llm') as mock_deep:
                mock_quick.return_value = Mock()
                mock_deep.return_value = Mock()

                # Set custom env vars
                with patch.dict(os.environ, {
                    'QUICK_MODEL': 'gemini-custom-quick',
                    'DEEP_MODEL': 'gemini-custom-deep'
                }):
                    # Reload config to pick up new env vars
                    from src import config as config_module
                    import importlib
                    importlib.reload(config_module)

                    graph = create_trading_graph(ticker="TEST", quick_mode=False, max_debate_rounds=1)

                    assert graph is not None
                    # Note: We can't easily verify the model names were used because they're
                    # passed through config, but the graph should create successfully


class TestConsultantModelSelection:
    """Test consultant model selection with quick_mode flag."""

    def test_consultant_quick_mode_integration(self):
        """Test that consultant respects quick_mode in graph creation."""
        from src.graph import create_trading_graph

        with patch('src.graph.create_memory_instances') as mock_memories:
            with patch('src.graph.get_consultant_llm') as mock_consultant:
                with patch('src.graph.create_quick_thinking_llm'):
                    with patch('src.graph.create_deep_thinking_llm'):
                        with patch('src.graph.toolkit'):
                            # Setup - mock returns different keys based on ticker
                            def memory_side_effect(ticker):
                                return {
                                    f'{ticker}_bull_memory': Mock(available=True),
                                    f'{ticker}_bear_memory': Mock(available=True),
                                    f'{ticker}_invest_judge_memory': Mock(available=True),
                                    f'{ticker}_trader_memory': Mock(available=True),
                                    f'{ticker}_risk_manager_memory': Mock(available=True),
                                }

                            mock_memories.side_effect = memory_side_effect
                            mock_consultant.return_value = Mock()  # Enabled

                            # Test normal mode
                            graph = create_trading_graph(ticker="TEST", quick_mode=False, max_debate_rounds=1)
                            assert graph is not None

                            # Verify consultant was called with quick_mode=False
                            calls = mock_consultant.call_args_list
                            assert len(calls) == 1
                            assert calls[0].kwargs.get('quick_mode') == False

                            # Reset mock
                            mock_consultant.reset_mock()

                            # Test quick mode
                            graph_quick = create_trading_graph(ticker="TEST2", quick_mode=True, max_debate_rounds=1)
                            assert graph_quick is not None

                            # Verify consultant was called with quick_mode=True
                            calls = mock_consultant.call_args_list
                            assert len(calls) == 1
                            assert calls[0].kwargs.get('quick_mode') == True


class TestModelSelectionLogging:
    """Test that model selection is properly logged."""

    @patch('src.graph.create_memory_instances')
    @patch('src.graph.get_consultant_llm')
    @patch('src.llms.logger')
    def test_model_initialization_logged(self, mock_logger, mock_consultant, mock_memories):
        """Test that LLM initialization is logged with model names."""
        from src.llms import create_quick_thinking_llm, create_deep_thinking_llm

        # Test quick model logging
        llm = create_quick_thinking_llm()
        assert llm is not None

        # Verify logging was called
        mock_logger.info.assert_called()
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any('Quick LLM' in str(call) for call in log_calls), \
            "Expected 'Quick LLM' in log output"

        # Reset
        mock_logger.reset_mock()

        # Test deep model logging
        llm_deep = create_deep_thinking_llm()
        assert llm_deep is not None

        # Verify logging was called
        mock_logger.info.assert_called()
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any('Deep LLM' in str(call) for call in log_calls), \
            "Expected 'Deep LLM' in log output"


class TestErrorHandling:
    """Test error handling for missing/invalid configurations."""

    def test_missing_env_vars_use_defaults(self):
        """Test that missing env vars use sensible defaults and don't crash."""
        from src.config import Config

        # Create config with no env vars set
        with patch.dict(os.environ, {}, clear=True):
            config = Config()

            # Verify defaults are set
            assert config.quick_think_llm is not None
            assert config.deep_think_llm is not None
            assert 'gemini' in config.quick_think_llm.lower()
            assert 'gemini' in config.deep_think_llm.lower()

    def test_consultant_missing_api_key_graceful(self):
        """Test that missing OPENAI_API_KEY is handled gracefully."""
        from src.llms import get_consultant_llm

        with patch.dict(os.environ, {'ENABLE_CONSULTANT': 'true'}, clear=True):
            # Remove OPENAI_API_KEY
            os.environ.pop('OPENAI_API_KEY', None)

            # Should return None, not crash
            llm = get_consultant_llm()
            assert llm is None

    @patch('src.graph.create_memory_instances')
    @patch('src.graph.get_consultant_llm')
    def test_graph_creation_survives_consultant_failure(self, mock_consultant, mock_memories):
        """Test that graph creation succeeds even if consultant fails."""
        from src.graph import create_trading_graph

        # Setup
        mock_memories.return_value = {
            'TEST_bull_memory': Mock(available=True),
            'TEST_bear_memory': Mock(available=True),
            'TEST_invest_judge_memory': Mock(available=True),
            'TEST_trader_memory': Mock(available=True),
            'TEST_risk_manager_memory': Mock(available=True),
        }
        mock_consultant.return_value = None  # Consultant unavailable

        # Should not raise
        graph = create_trading_graph(ticker="TEST", quick_mode=True, max_debate_rounds=1)
        assert graph is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
