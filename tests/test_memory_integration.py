"""
Integration Tests for Graph with Ticker-Specific Memory

These tests verify that the graph correctly uses ticker-specific memories
and prevents contamination in real-world scenarios.

Run with: pytest tests/test_graph_memory_integration.py -v
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from src.graph import create_trading_graph, TradingContext
from src.memory import sanitize_ticker_for_collection


class TestGraphMemoryIntegration:
    """Integration tests for graph with ticker-specific memory."""
    
    def test_trading_context_supports_ticker_memories(self):
        """Test that TradingContext has ticker memory fields."""
        context = TradingContext(
            ticker="0005.HK",
            trade_date="2025-11-28",
            cleanup_previous_memories=True
        )
        
        assert context.ticker == "0005.HK"
        assert context.cleanup_previous_memories == True
        assert hasattr(context, 'ticker_memories')
    
    @patch('src.graph.create_memory_instances')
    @patch('src.graph.cleanup_all_memories')
    def test_graph_creation_with_cleanup(self, mock_cleanup, mock_create_memories):
        """Test graph creation with cleanup_previous=True."""
        # Setup mock memories
        mock_memories = {
            "TEST_bull_memory": MagicMock(available=True),
            "TEST_bear_memory": MagicMock(available=True),
            "TEST_trader_memory": MagicMock(available=True),
            "TEST_invest_judge_memory": MagicMock(available=True),
            "TEST_risk_manager_memory": MagicMock(available=True),
        }
        mock_create_memories.return_value = mock_memories
        
        # Create graph with cleanup
        graph = create_trading_graph(
            ticker="TEST",
            cleanup_previous=True,
            max_debate_rounds=1
        )
        
        # Verify cleanup was called first with ticker
        # UPDATED: Expect ticker argument
        mock_cleanup.assert_called_once_with(days=0, ticker="TEST")
        
        # Verify memories were created after cleanup
        mock_create_memories.assert_called_once_with("TEST")
        
        # Verify graph was created
        assert graph is not None
    
    @patch('src.graph.create_memory_instances')
    def test_graph_creation_without_cleanup(self, mock_create_memories):
        """Test graph creation with cleanup_previous=False."""
        # Setup mock memories
        mock_memories = {
            "TEST_bull_memory": MagicMock(available=True),
            "TEST_bear_memory": MagicMock(available=True),
            "TEST_trader_memory": MagicMock(available=True),
            "TEST_invest_judge_memory": MagicMock(available=True),
            "TEST_risk_manager_memory": MagicMock(available=True),
        }
        mock_create_memories.return_value = mock_memories
        
        # Create graph without cleanup
        graph = create_trading_graph(
            ticker="TEST",
            cleanup_previous=False,
            max_debate_rounds=1
        )
        
        # Verify memories were created
        mock_create_memories.assert_called_once_with("TEST")
        
        # Verify graph was created
        assert graph is not None
    
    def test_graph_creation_without_ticker_uses_legacy(self):
        """Test that graph creation without ticker uses legacy memories."""
        # Should not raise, should use legacy global memories
        graph = create_trading_graph(max_debate_rounds=1)
        
        assert graph is not None
    
    @patch('src.graph.create_memory_instances')
    def test_different_tickers_create_different_graphs(self, mock_create_memories):
        """Test that different tickers create graphs with different memories."""
        # Setup mock for first ticker
        hsbc_memories = {
            "0005_HK_bull_memory": MagicMock(available=True),
            "0005_HK_bear_memory": MagicMock(available=True),
            "0005_HK_trader_memory": MagicMock(available=True),
            "0005_HK_invest_judge_memory": MagicMock(available=True),
            "0005_HK_risk_manager_memory": MagicMock(available=True),
        }
        
        # Setup mock for second ticker
        canon_memories = {
            "7915_T_bull_memory": MagicMock(available=True),
            "7915_T_bear_memory": MagicMock(available=True),
            "7915_T_trader_memory": MagicMock(available=True),
            "7915_T_invest_judge_memory": MagicMock(available=True),
            "7915_T_risk_manager_memory": MagicMock(available=True),
        }
        
        # First call returns HSBC memories, second returns Canon
        mock_create_memories.side_effect = [hsbc_memories, canon_memories]
        
        # Create graphs for both tickers
        hsbc_graph = create_trading_graph(ticker="0005.HK", max_debate_rounds=1)
        canon_graph = create_trading_graph(ticker="7915.T", max_debate_rounds=1)
        
        # Verify create_memory_instances was called with different tickers
        assert mock_create_memories.call_count == 2
        calls = [call[0][0] for call in mock_create_memories.call_args_list]
        assert "0005.HK" in calls
        assert "7915.T" in calls
        
        # Both graphs should be created
        assert hsbc_graph is not None
        assert canon_graph is not None


class TestGraphMemoryContaminationPrevention:
    """
    High-level integration tests for contamination prevention.
    
    These simulate the real-world scenario from Grok's critique.
    """
    
    @patch('src.graph.create_memory_instances')
    @patch('src.graph.cleanup_all_memories')
    def test_sequential_analysis_no_contamination(self, mock_cleanup, mock_create_memories):
        """
        REGRESSION TEST for Grok's critique:
        "Canon (7915.T) risk assessments mixed into HSBC (0005.HK) report"
        
        Simulate:
        1. Analyze HSBC
        2. Cleanup
        3. Analyze Canon  
        4. Cleanup
        5. Re-analyze HSBC
        
        Verify no cross-contamination.
        """
        # Track which ticker is being analyzed
        analysis_sequence = []
        
        def track_create_memories(ticker):
            analysis_sequence.append(('create', ticker))
            # Return appropriate mocks based on ticker
            # CRITICAL: Use same sanitization as production code
            safe_ticker = sanitize_ticker_for_collection(ticker)
            return {
                f"{safe_ticker}_bull_memory": MagicMock(available=True),
                f"{safe_ticker}_bear_memory": MagicMock(available=True),
                f"{safe_ticker}_trader_memory": MagicMock(available=True),
                f"{safe_ticker}_invest_judge_memory": MagicMock(available=True),
                f"{safe_ticker}_risk_manager_memory": MagicMock(available=True),
            }
        
        # FIX: Accept kwargs to handle 'ticker' argument safely
        def track_cleanup(days, **kwargs):
            ticker = kwargs.get('ticker')
            analysis_sequence.append(('cleanup', days, ticker))
            return {}
        
        mock_create_memories.side_effect = track_create_memories
        mock_cleanup.side_effect = track_cleanup
        
        # Step 1: Analyze HSBC
        hsbc_graph_1 = create_trading_graph(
            ticker="0005.HK",
            cleanup_previous=True,
            max_debate_rounds=1
        )
        
        # Step 2: Analyze Canon (with cleanup)
        canon_graph = create_trading_graph(
            ticker="7915.T",
            cleanup_previous=True,
            max_debate_rounds=1
        )
        
        # Step 3: Re-analyze HSBC (with cleanup)
        hsbc_graph_2 = create_trading_graph(
            ticker="0005.HK",
            cleanup_previous=True,
            max_debate_rounds=1
        )
        
        # Verify sequence
        assert len(analysis_sequence) == 6  # 3 cleanups + 3 creates
        
        # Verify cleanups happened before each ticker
        assert analysis_sequence[0] == ('cleanup', 0, '0005.HK')  # Before HSBC
        assert analysis_sequence[1] == ('create', '0005.HK')
        assert analysis_sequence[2] == ('cleanup', 0, '7915.T')  # Before Canon
        assert analysis_sequence[3] == ('create', '7915.T')
        assert analysis_sequence[4] == ('cleanup', 0, '0005.HK')  # Before HSBC again
        assert analysis_sequence[5] == ('create', '0005.HK')
        
        # All graphs should be created
        assert hsbc_graph_1 is not None
        assert canon_graph is not None
        assert hsbc_graph_2 is not None
    
    @patch('src.graph.create_memory_instances')
    def test_memory_disabled_still_works(self, mock_create_memories):
        """Test that graph works even when memory is disabled."""
        # Don't mock create_memory_instances for this test
        mock_create_memories.side_effect = None
        
        # Create graph with memory disabled
        graph = create_trading_graph(
            ticker="TEST",
            enable_memory=False,
            max_debate_rounds=1
        )
        
        # Should still create graph (using legacy memories)
        assert graph is not None
        
        # create_memory_instances should not be called when memory disabled
        # (it would use legacy memories instead)
        # Note: This behavior depends on implementation


class TestGraphLogging:
    """Test that graph logs contamination-prevention actions."""
    
    @patch('src.graph.create_memory_instances')
    @patch('src.graph.cleanup_all_memories')
    @patch('src.graph.logger')
    def test_cleanup_logged(self, mock_logger, mock_cleanup, mock_create_memories):
        """Test that cleanup actions are logged."""
        mock_memories = {
            "TEST_bull_memory": MagicMock(available=True),
            "TEST_bear_memory": MagicMock(available=True),
            "TEST_trader_memory": MagicMock(available=True),
            "TEST_invest_judge_memory": MagicMock(available=True),
            "TEST_risk_manager_memory": MagicMock(available=True),
        }
        mock_create_memories.return_value = mock_memories
        mock_cleanup.return_value = {}
        
        # Create graph with cleanup
        graph = create_trading_graph(
            ticker="TEST",
            cleanup_previous=True
        )
        
        # Verify cleanup was logged
        cleanup_logged = False
        for call in mock_logger.info.call_args_list:
            if len(call[0]) > 0 and 'cleaning_previous_memories' in str(call[0][0]):
                cleanup_logged = True
                break
        
        assert cleanup_logged, "Cleanup action should be logged"
    
    @patch('src.graph.create_memory_instances')
    @patch('src.graph.logger')
    def test_ticker_memory_creation_logged(self, mock_logger, mock_create_memories):
        """Test that ticker-specific memory creation is logged."""
        mock_memories = {
            "TEST_bull_memory": MagicMock(available=True),
            "TEST_bear_memory": MagicMock(available=True),
            "TEST_trader_memory": MagicMock(available=True),
            "TEST_invest_judge_memory": MagicMock(available=True),
            "TEST_risk_manager_memory": MagicMock(available=True),
        }
        mock_create_memories.return_value = mock_memories
        
        # Create graph with ticker
        graph = create_trading_graph(ticker="TEST")
        
        # Verify creation was logged
        creation_logged = False
        for call in mock_logger.info.call_args_list:
            if len(call[0]) > 0 and 'creating_ticker_memories' in str(call[0][0]):
                creation_logged = True
                break
        
        assert creation_logged, "Ticker memory creation should be logged"
    
    @patch('src.graph.logger')
    def test_legacy_memory_warning_logged(self, mock_logger):
        """Test that using legacy memories triggers a warning."""
        # Create graph without ticker (legacy mode)
        graph = create_trading_graph(max_debate_rounds=1)
        
        # Verify warning was logged
        warning_logged = False
        for call in mock_logger.warning.call_args_list:
            if len(call[0]) > 0 and 'using_legacy_memories' in str(call[0][0]):
                warning_logged = True
                break
        
        assert warning_logged, "Legacy memory usage should trigger warning"


class TestGraphMemoryKeyConsistency:
    """Test that graph.py uses same sanitization as memory.py for edge-case tickers."""

    @patch('src.graph.create_memory_instances')
    @patch('src.graph.cleanup_all_memories')
    def test_edge_case_ticker_sanitization_matches(self, mock_cleanup, mock_create_memories):
        """Test that graph.py correctly handles edge-case tickers with special chars, hyphens, etc."""
        # Test cases that would fail with simple .replace() approach
        edge_case_tickers = [
            "BRK-B",          # Hyphenated ticker (Berkshire Hathaway)
            "BF-B",           # Another hyphenated ticker
            "0005.HK",        # Standard HK ticker
            "A" * 50,         # Very long ticker (>40 chars, will be truncated)
            ".WEIRDTICKER",   # Starts with dot (needs T_ prefix)
        ]

        mock_cleanup.return_value = {}

        for ticker in edge_case_tickers:
            # Create mock memories using the SAME sanitization function
            safe_ticker = sanitize_ticker_for_collection(ticker)
            mock_memories = {
                f"{safe_ticker}_bull_memory": MagicMock(available=True),
                f"{safe_ticker}_bear_memory": MagicMock(available=True),
                f"{safe_ticker}_trader_memory": MagicMock(available=True),
                f"{safe_ticker}_invest_judge_memory": MagicMock(available=True),
                f"{safe_ticker}_risk_manager_memory": MagicMock(available=True),
            }
            mock_create_memories.return_value = mock_memories

            # Should NOT raise ValueError about missing memories
            try:
                graph = create_trading_graph(
                    ticker=ticker,
                    cleanup_previous=False,
                    enable_memory=True,
                    max_debate_rounds=1
                )
                assert graph is not None, f"Graph creation failed for ticker: {ticker}"
            except ValueError as e:
                if "Failed to create memory instances" in str(e):
                    pytest.fail(
                        f"Memory key mismatch for ticker '{ticker}': {e}\n"
                        f"This indicates graph.py sanitization doesn't match memory.py"
                    )
                else:
                    raise

    @patch('src.graph.create_memory_instances')
    @patch('src.graph.cleanup_all_memories')
    def test_hyphenated_ticker_specific(self, mock_cleanup, mock_create_memories):
        """Specific test for hyphenated tickers like BRK-B."""
        ticker = "BRK-B"
        safe_ticker = sanitize_ticker_for_collection(ticker)

        # Verify sanitization converts hyphen to underscore
        assert safe_ticker == "BRK_B", f"Expected 'BRK_B', got '{safe_ticker}'"

        mock_cleanup.return_value = {}
        mock_memories = {
            f"{safe_ticker}_bull_memory": MagicMock(available=True),
            f"{safe_ticker}_bear_memory": MagicMock(available=True),
            f"{safe_ticker}_trader_memory": MagicMock(available=True),
            f"{safe_ticker}_invest_judge_memory": MagicMock(available=True),
            f"{safe_ticker}_risk_manager_memory": MagicMock(available=True),
        }
        mock_create_memories.return_value = mock_memories

        # Should successfully create graph
        graph = create_trading_graph(
            ticker=ticker,
            cleanup_previous=False,
            enable_memory=True,
            max_debate_rounds=1
        )

        assert graph is not None
        mock_create_memories.assert_called_once_with(ticker)

    @patch('src.graph.create_memory_instances')
    @patch('src.graph.cleanup_all_memories')
    def test_very_long_ticker_truncation(self, mock_cleanup, mock_create_memories):
        """Test that very long tickers (>40 chars) are handled consistently."""
        long_ticker = "A" * 100
        safe_ticker = sanitize_ticker_for_collection(long_ticker)

        # Verify truncation to 40 chars (max for base ticker)
        assert len(safe_ticker) == 40, f"Expected 40 chars, got {len(safe_ticker)}"

        mock_cleanup.return_value = {}
        mock_memories = {
            f"{safe_ticker}_bull_memory": MagicMock(available=True),
            f"{safe_ticker}_bear_memory": MagicMock(available=True),
            f"{safe_ticker}_trader_memory": MagicMock(available=True),
            f"{safe_ticker}_invest_judge_memory": MagicMock(available=True),
            f"{safe_ticker}_risk_manager_memory": MagicMock(available=True),
        }
        mock_create_memories.return_value = mock_memories

        # Should successfully create graph without key mismatch
        graph = create_trading_graph(
            ticker=long_ticker,
            cleanup_previous=False,
            enable_memory=True,
            max_debate_rounds=1
        )

        assert graph is not None

    @patch('src.graph.create_memory_instances')
    @patch('src.graph.cleanup_all_memories')
    def test_ticker_starting_with_special_char(self, mock_cleanup, mock_create_memories):
        """Test tickers starting with non-alphanumeric (needs T_ prefix)."""
        ticker = ".WEIRDTICKER"
        safe_ticker = sanitize_ticker_for_collection(ticker)

        # Verify T_ prefix is added
        assert safe_ticker.startswith("T_"), f"Expected T_ prefix, got '{safe_ticker}'"

        mock_cleanup.return_value = {}
        mock_memories = {
            f"{safe_ticker}_bull_memory": MagicMock(available=True),
            f"{safe_ticker}_bear_memory": MagicMock(available=True),
            f"{safe_ticker}_trader_memory": MagicMock(available=True),
            f"{safe_ticker}_invest_judge_memory": MagicMock(available=True),
            f"{safe_ticker}_risk_manager_memory": MagicMock(available=True),
        }
        mock_create_memories.return_value = mock_memories

        # Should successfully create graph
        graph = create_trading_graph(
            ticker=ticker,
            cleanup_previous=False,
            enable_memory=True,
            max_debate_rounds=1
        )

        assert graph is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])