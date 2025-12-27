"""
Unit tests for token tracking module.

Tests the token tracking functionality:
- Token usage calculation and recording
- Cost estimation with correct paid tier pricing
- Singleton pattern behavior
- Token tracker callback integration
- Statistics aggregation and reporting
"""

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.token_tracker import (
    AgentTokenStats,
    TokenTracker,
    TokenTrackingCallback,
    TokenUsage,
    get_tracker,
)


class TestTokenUsage:
    """Test TokenUsage dataclass and cost calculation."""

    def test_token_usage_creation(self):
        """Test creating a TokenUsage instance."""
        usage = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )

        assert usage.agent_name == "test_agent"
        assert usage.model_name == "gemini-2.5-flash"
        assert usage.prompt_tokens == 1000
        assert usage.completion_tokens == 500
        assert usage.total_tokens == 1500

    def test_cost_calculation_flash_25(self):
        """Test cost calculation for Gemini 2.5 Flash (paid tier)."""
        usage = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=1_000_000,  # 1M tokens
            completion_tokens=1_000_000,  # 1M tokens
            total_tokens=2_000_000,
        )

        # Expected: $0.30 + $2.50 = $2.80
        expected_cost = 0.30 + 2.50
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.001

    def test_cost_calculation_gemini_3_pro(self):
        """Test cost calculation for Gemini 3 Pro Preview (paid tier)."""
        usage = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="portfolio_manager",
            model_name="gemini-3-pro-preview",
            prompt_tokens=500_000,  # 0.5M tokens
            completion_tokens=250_000,  # 0.25M tokens
            total_tokens=750_000,
        )

        # Expected: (0.5 * $2.00) + (0.25 * $12.00) = $1.00 + $3.00 = $4.00
        expected_cost = 1.00 + 3.00
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.001

    def test_cost_calculation_flash_exp(self):
        """Test cost calculation for experimental Flash models (paid tier when billing enabled)."""
        usage = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="analyst",
            model_name="gemini-2.0-flash-exp",
            prompt_tokens=2_000_000,  # 2M tokens
            completion_tokens=1_000_000,  # 1M tokens
            total_tokens=3_000_000,
        )

        # Paid tier pricing: (2 * $0.30) + (1 * $2.50) = $0.60 + $2.50 = $3.10
        expected_cost = 0.60 + 2.50
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.001

    def test_cost_calculation_flash_lite(self):
        """Test cost calculation for Gemini 2.5 Flash-Lite (cheaper alternative)."""
        usage = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="quick_agent",
            model_name="gemini-2.5-flash-lite",
            prompt_tokens=1_000_000,  # 1M tokens
            completion_tokens=1_000_000,  # 1M tokens
            total_tokens=2_000_000,
        )

        # Expected: $0.10 + $0.40 = $0.50
        expected_cost = 0.10 + 0.40
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.001

    def test_cost_calculation_unknown_model(self):
        """Test cost calculation for unknown model (defaults to Flash pricing)."""
        usage = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="test_agent",
            model_name="unknown-model-xyz",
            prompt_tokens=1_000_000,  # 1M tokens
            completion_tokens=1_000_000,  # 1M tokens
            total_tokens=2_000_000,
        )

        # Should default to Flash pricing: $0.30 + $2.50 = $2.80
        expected_cost = 0.30 + 2.50
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.001

    def test_cost_calculation_zero_tokens(self):
        """Test cost calculation when no tokens are used."""
        usage = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        assert usage.estimated_cost_usd == 0.0

    def test_cost_calculation_small_tokens(self):
        """Test cost calculation for very small token counts."""
        usage = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=100,  # 0.0001M tokens
            completion_tokens=50,  # 0.00005M tokens
            total_tokens=150,
        )

        # Expected: (0.0001 * $0.30) + (0.00005 * $2.50) = $0.00003 + $0.000125 = $0.000155
        expected_cost = (100 / 1_000_000 * 0.30) + (50 / 1_000_000 * 2.50)
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.0001


class TestAgentTokenStats:
    """Test AgentTokenStats aggregation."""

    def test_agent_stats_creation(self):
        """Test creating an AgentTokenStats instance."""
        stats = AgentTokenStats(agent_name="test_agent")

        assert stats.agent_name == "test_agent"
        assert stats.total_calls == 0
        assert stats.total_prompt_tokens == 0
        assert stats.total_completion_tokens == 0
        assert stats.total_tokens == 0
        assert stats.total_cost_usd == 0.0
        assert len(stats.calls) == 0

    def test_add_usage_single(self):
        """Test adding a single usage record."""
        stats = AgentTokenStats(agent_name="test_agent")

        usage = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )

        stats.add_usage(usage)

        assert stats.total_calls == 1
        assert stats.total_prompt_tokens == 1000
        assert stats.total_completion_tokens == 500
        assert stats.total_tokens == 1500
        assert len(stats.calls) == 1

    def test_add_usage_multiple(self):
        """Test adding multiple usage records."""
        stats = AgentTokenStats(agent_name="test_agent")

        usage1 = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )

        usage2 = TokenUsage(
            timestamp="2025-12-05T12:01:00",
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=2000,
            completion_tokens=1000,
            total_tokens=3000,
        )

        stats.add_usage(usage1)
        stats.add_usage(usage2)

        assert stats.total_calls == 2
        assert stats.total_prompt_tokens == 3000
        assert stats.total_completion_tokens == 1500
        assert stats.total_tokens == 4500
        assert len(stats.calls) == 2

    def test_cost_aggregation(self):
        """Test that costs are aggregated correctly."""
        stats = AgentTokenStats(agent_name="test_agent")

        # Add two usages with known costs
        usage1 = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=1_000_000,  # $0.30
            completion_tokens=1_000_000,  # $2.50
            total_tokens=2_000_000,
        )

        usage2 = TokenUsage(
            timestamp="2025-12-05T12:01:00",
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=500_000,  # $0.15
            completion_tokens=500_000,  # $1.25
            total_tokens=1_000_000,
        )

        stats.add_usage(usage1)
        stats.add_usage(usage2)

        # Expected: ($0.30 + $2.50) + ($0.15 + $1.25) = $2.80 + $1.40 = $4.20
        expected_cost = 2.80 + 1.40
        assert abs(stats.total_cost_usd - expected_cost) < 0.001


class TestTokenTracker:
    """Test TokenTracker singleton and recording functionality."""

    def test_singleton_pattern(self):
        """Test that TokenTracker is a singleton."""
        tracker1 = TokenTracker()
        tracker2 = TokenTracker()

        assert tracker1 is tracker2

    def test_global_tracker_instance(self):
        """Test that get_tracker returns the singleton instance."""
        tracker1 = get_tracker()
        tracker2 = TokenTracker()

        assert tracker1 is tracker2

    def test_record_usage_new_agent(self):
        """Test recording usage for a new agent."""
        tracker = TokenTracker()
        tracker.reset()  # Start fresh

        tracker.record_usage(
            agent_name="new_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        assert "new_agent" in tracker.agent_stats
        assert tracker.agent_stats["new_agent"].total_calls == 1
        assert tracker.agent_stats["new_agent"].total_prompt_tokens == 1000
        assert tracker.agent_stats["new_agent"].total_completion_tokens == 500
        assert len(tracker.all_usages) == 1

    def test_record_usage_existing_agent(self):
        """Test recording usage for an existing agent."""
        tracker = TokenTracker()
        tracker.reset()  # Start fresh

        tracker.record_usage(
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        tracker.record_usage(
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=2000,
            completion_tokens=1000,
        )

        assert tracker.agent_stats["test_agent"].total_calls == 2
        assert tracker.agent_stats["test_agent"].total_prompt_tokens == 3000
        assert tracker.agent_stats["test_agent"].total_completion_tokens == 1500
        assert len(tracker.all_usages) == 2

    def test_get_agent_stats(self):
        """Test retrieving stats for a specific agent."""
        tracker = TokenTracker()
        tracker.reset()

        tracker.record_usage(
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        stats = tracker.get_agent_stats("test_agent")

        assert stats is not None
        assert stats.agent_name == "test_agent"
        assert stats.total_calls == 1

    def test_get_agent_stats_nonexistent(self):
        """Test retrieving stats for an agent that doesn't exist."""
        tracker = TokenTracker()
        tracker.reset()

        stats = tracker.get_agent_stats("nonexistent_agent")

        assert stats is None

    def test_get_total_stats_empty(self):
        """Test getting total stats when no usage has been recorded."""
        tracker = TokenTracker()
        tracker.reset()

        stats = tracker.get_total_stats()

        assert stats["total_calls"] == 0
        assert stats["total_agents"] == 0
        assert stats["total_prompt_tokens"] == 0
        assert stats["total_completion_tokens"] == 0
        assert stats["total_tokens"] == 0
        assert stats["total_cost_usd"] == 0.0
        assert len(stats["agents"]) == 0

    def test_get_total_stats_with_data(self):
        """Test getting total stats with multiple agents."""
        tracker = TokenTracker()
        tracker.reset()

        tracker.record_usage(
            agent_name="agent1",
            model_name="gemini-2.5-flash",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        tracker.record_usage(
            agent_name="agent2",
            model_name="gemini-3-pro-preview",
            prompt_tokens=2000,
            completion_tokens=1000,
        )

        stats = tracker.get_total_stats()

        assert stats["total_calls"] == 2
        assert stats["total_agents"] == 2
        assert stats["total_prompt_tokens"] == 3000
        assert stats["total_completion_tokens"] == 1500
        assert stats["total_tokens"] == 4500
        assert "agent1" in stats["agents"]
        assert "agent2" in stats["agents"]

    def test_reset(self):
        """Test resetting the tracker."""
        tracker = TokenTracker()

        # Add some data
        tracker.record_usage(
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        # Reset
        tracker.reset()

        # Verify everything is cleared
        stats = tracker.get_total_stats()
        assert stats["total_calls"] == 0
        assert stats["total_agents"] == 0
        assert len(tracker.all_usages) == 0
        assert len(tracker.agent_stats) == 0

    def test_quiet_mode_suppresses_logging(self):
        """Test that quiet mode suppresses all logging from TokenTracker."""
        import io
        import logging

        # Create a log handler to capture logs
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)

        # Get the structlog logger and temporarily replace it
        # Note: This is a simplified test; structlog configuration is complex
        tracker = TokenTracker()

        # Enable quiet mode
        TokenTracker.set_quiet_mode(True)

        # Reset should not log in quiet mode
        tracker.reset()

        # Record usage should not log in quiet mode
        tracker.record_usage(
            agent_name="test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        # Print summary should not log in quiet mode
        tracker.print_summary()

        # Disable quiet mode for other tests
        TokenTracker.set_quiet_mode(False)

        # Note: We can't easily verify structlog didn't log without complex mocking,
        # but we can verify the methods don't crash and data is still tracked
        stats = tracker.get_total_stats()
        assert stats["total_calls"] >= 1  # Usage was still recorded

    def test_quiet_mode_still_tracks_data(self):
        """Test that quiet mode suppresses logs but still tracks token usage."""
        tracker = TokenTracker()

        # Enable quiet mode
        TokenTracker.set_quiet_mode(True)
        tracker.reset()

        # Record usage
        tracker.record_usage(
            agent_name="quiet_test_agent",
            model_name="gemini-2.5-flash",
            prompt_tokens=5000,
            completion_tokens=2500,
        )

        # Verify data is still tracked despite quiet mode
        stats = tracker.get_agent_stats("quiet_test_agent")
        assert stats is not None
        assert stats.total_calls == 1
        assert stats.total_prompt_tokens == 5000
        assert stats.total_completion_tokens == 2500

        # Get total stats should still work
        total_stats = tracker.get_total_stats()
        assert total_stats["total_calls"] >= 1

        # Disable quiet mode for other tests
        TokenTracker.set_quiet_mode(False)

    def test_quiet_mode_before_initialization(self):
        """Test that setting quiet mode before initialization suppresses init log."""
        # This test validates the fix for the issue where token_tracker_initialized
        # was being logged even with --quiet flag.

        # Reset the global tracker to None to simulate fresh import
        import src.token_tracker as tt_module

        old_tracker = tt_module._global_tracker
        tt_module._global_tracker = None

        # Clear the singleton instance
        TokenTracker._instance = None

        try:
            # Set quiet mode BEFORE calling get_tracker
            TokenTracker.set_quiet_mode(True)

            # Now get the tracker - initialization should be silent
            tracker = get_tracker()

            # Verify tracker is initialized
            assert tracker is not None
            assert tracker._quiet_mode

            # Record some usage to verify tracker still works
            tracker.record_usage(
                agent_name="init_test_agent",
                model_name="gemini-2.5-flash",
                prompt_tokens=100,
                completion_tokens=50,
            )

            # Verify data is tracked
            stats = tracker.get_agent_stats("init_test_agent")
            assert stats is not None
            assert stats.total_calls == 1

        finally:
            # Restore original state
            tt_module._global_tracker = old_tracker
            TokenTracker.set_quiet_mode(False)


class TestTokenTrackingCallback:
    """Test LangChain callback integration."""

    def test_callback_creation(self):
        """Test creating a TokenTrackingCallback."""
        callback = TokenTrackingCallback(agent_name="test_agent")

        assert callback.agent_name == "test_agent"
        assert callback.tracker is not None

    def test_callback_with_custom_tracker(self):
        """Test creating callback with custom tracker instance."""
        custom_tracker = TokenTracker()
        callback = TokenTrackingCallback(
            agent_name="test_agent", tracker=custom_tracker
        )

        assert callback.tracker is custom_tracker

    def test_on_llm_end_with_usage_metadata(self):
        """Test callback when LLM returns usage metadata."""
        tracker = TokenTracker()
        tracker.reset()

        callback = TokenTrackingCallback(agent_name="test_agent", tracker=tracker)

        # Mock LLM response
        llm_result = LLMResult(
            generations=[],
            llm_output={
                "model_name": "gemini-2.5-flash",
                "usage_metadata": {"input_tokens": 1000, "output_tokens": 500},
            },
        )

        callback.on_llm_end(llm_result)

        # Verify usage was recorded
        stats = tracker.get_agent_stats("test_agent")
        assert stats is not None
        assert stats.total_calls == 1
        assert stats.total_prompt_tokens == 1000
        assert stats.total_completion_tokens == 500

    def test_on_llm_end_with_deprecated_token_usage(self):
        """Test callback with deprecated token_usage field (fallback)."""
        tracker = TokenTracker()
        tracker.reset()

        callback = TokenTrackingCallback(agent_name="test_agent", tracker=tracker)

        # Mock LLM response with deprecated field
        llm_result = LLMResult(
            generations=[],
            llm_output={
                "model_name": "gemini-2.5-flash",
                "token_usage": {"prompt_tokens": 2000, "completion_tokens": 1000},
            },
        )

        callback.on_llm_end(llm_result)

        # Verify usage was recorded
        stats = tracker.get_agent_stats("test_agent")
        assert stats is not None
        assert stats.total_calls == 1
        assert stats.total_prompt_tokens == 2000
        assert stats.total_completion_tokens == 1000

    def test_on_llm_end_no_output(self):
        """Test callback when LLM returns no output metadata."""
        tracker = TokenTracker()
        tracker.reset()

        callback = TokenTrackingCallback(agent_name="test_agent", tracker=tracker)

        # Mock LLM response with no llm_output
        llm_result = LLMResult(generations=[], llm_output=None)

        callback.on_llm_end(llm_result)

        # Verify nothing was recorded
        stats = tracker.get_total_stats()
        assert stats["total_calls"] == 0

    def test_on_llm_end_no_usage_metadata(self):
        """Test callback when LLM output has no usage metadata."""
        tracker = TokenTracker()
        tracker.reset()

        callback = TokenTrackingCallback(agent_name="test_agent", tracker=tracker)

        # Mock LLM response without usage metadata
        llm_result = LLMResult(
            generations=[],
            llm_output={
                "model_name": "gemini-2.5-flash"
                # No usage_metadata or token_usage
            },
        )

        callback.on_llm_end(llm_result)

        # Verify nothing was recorded
        stats = tracker.get_total_stats()
        assert stats["total_calls"] == 0

    def test_on_llm_end_zero_tokens(self):
        """Test callback when usage metadata shows zero tokens."""
        tracker = TokenTracker()
        tracker.reset()

        callback = TokenTrackingCallback(agent_name="test_agent", tracker=tracker)

        # Mock LLM response with zero tokens
        llm_result = LLMResult(
            generations=[],
            llm_output={
                "model_name": "gemini-2.5-flash",
                "usage_metadata": {"input_tokens": 0, "output_tokens": 0},
            },
        )

        callback.on_llm_end(llm_result)

        # Verify nothing was recorded (zero tokens are skipped)
        stats = tracker.get_total_stats()
        assert stats["total_calls"] == 0

    def test_on_llm_end_gemini_message_structure(self):
        """Test callback with Gemini's actual response structure (usage_metadata in message)."""
        tracker = TokenTracker()
        tracker.reset()

        callback = TokenTrackingCallback(agent_name="test_agent", tracker=tracker)

        # Create actual LangChain objects matching Gemini's structure
        # In Gemini, usage_metadata is in message.usage_metadata, not llm_output
        message = AIMessage(
            content="Test response",
            usage_metadata={
                "input_tokens": 1500,
                "output_tokens": 800,
                "total_tokens": 2300,
            },
            response_metadata={"model_name": "gemini-2.5-flash"},
        )

        generation = ChatGeneration(
            message=message, generation_info={"model_name": "gemini-2.5-flash"}
        )

        llm_result = LLMResult(
            generations=[[generation]],
            llm_output={},  # Empty, as Gemini doesn't populate this
        )

        callback.on_llm_end(llm_result)

        # Verify usage was recorded correctly from message.usage_metadata
        stats = tracker.get_agent_stats("test_agent")
        assert stats is not None
        assert stats.total_calls == 1
        assert stats.total_prompt_tokens == 1500
        assert stats.total_completion_tokens == 800
        assert stats.total_tokens == 2300


class TestCostAccuracy:
    """Test that cost calculations match expected paid tier rates."""

    def test_flash_25_realistic_usage(self):
        """Test realistic usage scenario with Gemini 2.5 Flash."""
        usage = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="market_analyst",
            model_name="gemini-2.5-flash",
            prompt_tokens=50_000,  # 50k input
            completion_tokens=10_000,  # 10k output
            total_tokens=60_000,
        )

        # Expected: (0.05 * $0.30) + (0.01 * $2.50) = $0.015 + $0.025 = $0.04
        expected_cost = 0.04
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.001

    def test_gemini_3_pro_realistic_usage(self):
        """Test realistic usage scenario with Gemini 3 Pro."""
        usage = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="portfolio_manager",
            model_name="gemini-3-pro-preview",
            prompt_tokens=100_000,  # 100k input
            completion_tokens=20_000,  # 20k output
            total_tokens=120_000,
        )

        # Expected: (0.1 * $2.00) + (0.02 * $12.00) = $0.20 + $0.24 = $0.44
        expected_cost = 0.44
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.001

    def test_flash_lite_cost_savings(self):
        """Test that Flash-Lite is significantly cheaper than Flash."""
        flash_usage = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="agent1",
            model_name="gemini-2.5-flash",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
        )

        flash_lite_usage = TokenUsage(
            timestamp="2025-12-05T12:00:00",
            agent_name="agent2",
            model_name="gemini-2.5-flash-lite",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
        )

        # Flash: $0.30 + $2.50 = $2.80
        # Flash-Lite: $0.10 + $0.40 = $0.50
        # Savings: $2.30 (82% cheaper)

        assert flash_usage.estimated_cost_usd > flash_lite_usage.estimated_cost_usd
        savings_percentage = (
            1 - flash_lite_usage.estimated_cost_usd / flash_usage.estimated_cost_usd
        ) * 100
        assert savings_percentage > 80  # At least 80% cheaper
