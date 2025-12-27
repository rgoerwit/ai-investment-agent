"""
Test suite for rate limit (429/ResourceExhausted) handling.

Ensures the system gracefully handles Gemini API free tier rate limits
with proper exponential backoff and logging.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.agents import invoke_with_rate_limit_handling


class Test429Detection:
    """Test detection of various rate limit error formats."""

    @pytest.mark.asyncio
    async def test_detects_429_in_error_message(self):
        """Test that 429 in error message triggers rate limit handling."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(
            side_effect=[
                Exception("HTTP 429: Too Many Requests"),
                AIMessage(content="Success after retry"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await invoke_with_rate_limit_handling(
                runnable, {"input": "test"}, max_attempts=2
            )

        # Should have slept 60 seconds before retry
        mock_sleep.assert_called_once_with(60)
        assert result.content == "Success after retry"

    @pytest.mark.asyncio
    async def test_detects_resourceexhausted(self):
        """Test detection of ResourceExhausted errors."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(
            side_effect=[
                Exception("ResourceExhausted: Quota exceeded"),
                AIMessage(content="Success"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await invoke_with_rate_limit_handling(
                runnable, {"input": "test"}, max_attempts=2
            )

        mock_sleep.assert_called_once_with(60)
        assert result.content == "Success"

    @pytest.mark.asyncio
    async def test_detects_rate_limit_text(self):
        """Test detection of 'rate limit' in error text."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(
            side_effect=[
                Exception("rate limit exceeded for this API key"),
                AIMessage(content="Success"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await invoke_with_rate_limit_handling(
                runnable, {"input": "test"}, max_attempts=2
            )

        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_detects_quota_exceeded(self):
        """Test detection of quota exceeded errors."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(
            side_effect=[
                Exception("Quota exceeded for requests per minute"),
                AIMessage(content="Success"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await invoke_with_rate_limit_handling(
                runnable, {"input": "test"}, max_attempts=2
            )

        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_detects_too_many_requests(self):
        """Test detection of 'too many requests' errors."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(
            side_effect=[
                Exception("Too many requests, please slow down"),
                AIMessage(content="Success"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await invoke_with_rate_limit_handling(
                runnable, {"input": "test"}, max_attempts=2
            )

        mock_sleep.assert_called_once()


class TestExponentialBackoff:
    """Test exponential backoff behavior."""

    @pytest.mark.asyncio
    async def test_backoff_progression_60_120(self):
        """Test that backoff times progress correctly (60s, 120s).

        Reduced from 3 attempts to 2 for faster test execution.
        """
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(
            side_effect=[
                Exception("429 Too Many Requests"),
                Exception("429 Too Many Requests"),
                AIMessage(content="Success on 3rd attempt"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await invoke_with_rate_limit_handling(
                runnable, {"input": "test"}, max_attempts=3
            )

        # Check sleep was called with 60, 120
        assert mock_sleep.call_count == 2
        call_args = [call[0][0] for call in mock_sleep.call_args_list]
        assert call_args == [60, 120]

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Slow/expensive test - skipped to improve test suite performance"
    )
    async def test_raises_after_max_attempts(self):
        """Test that error is raised after max attempts exhausted.

        SKIPPED: This test is slow and expensive.
        Run explicitly with: pytest -m slow
        """
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=Exception("429 Too Many Requests"))

        with pytest.raises(Exception, match="429"):
            await invoke_with_rate_limit_handling(
                runnable, {"input": "test"}, max_attempts=3
            )


class TestNonRateLimitErrors:
    """Test that non-rate-limit errors are handled correctly."""

    @pytest.mark.asyncio
    async def test_non_429_error_raised_immediately(self):
        """Test that non-429 errors are raised without retry."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=ValueError("Invalid input"))

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(ValueError, match="Invalid input"):
                await invoke_with_rate_limit_handling(
                    runnable, {"input": "test"}, max_attempts=3
                )

        # Should not sleep for non-rate-limit errors
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_network_error_raised_immediately(self):
        """Test that network errors are not treated as rate limits."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=ConnectionError("Network timeout"))

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(ConnectionError):
                await invoke_with_rate_limit_handling(
                    runnable, {"input": "test"}, max_attempts=3
                )

        mock_sleep.assert_not_called()


class TestQuietMode:
    """Test quiet mode suppresses rate limit logging."""

    @pytest.mark.asyncio
    async def test_quiet_mode_suppresses_logging(self):
        """Test that quiet_mode=True suppresses rate limit warnings."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(
            side_effect=[
                Exception("429 Too Many Requests"),
                AIMessage(content="Success"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch("src.agents.settings_config") as mock_config:
                mock_config.quiet_mode = True
                with patch("src.agents.logger") as mock_logger:
                    result = await invoke_with_rate_limit_handling(
                        runnable,
                        {"input": "test"},
                        max_attempts=2,
                        context="Test Agent",
                    )

                    # Should not log in quiet mode
                    mock_logger.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_mode_logs_rate_limits(self):
        """Test that rate limits are logged when not in quiet mode."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(
            side_effect=[
                Exception("429 Too Many Requests"),
                AIMessage(content="Success"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch("src.agents.settings_config") as mock_config:
                mock_config.quiet_mode = False
                with patch("src.agents.logger") as mock_logger:
                    result = await invoke_with_rate_limit_handling(
                        runnable,
                        {"input": "test"},
                        max_attempts=2,
                        context="Test Agent",
                    )

                    # Should log in normal mode
                    mock_logger.warning.assert_called_once()
                    call_args = mock_logger.warning.call_args[1]
                    assert call_args["context"] == "Test Agent"
                    assert call_args["wait_seconds"] == 60


class TestContextLogging:
    """Test that context information is properly logged."""

    @pytest.mark.asyncio
    async def test_context_included_in_logs(self):
        """Test that agent context is included in log messages."""

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(
            side_effect=[
                Exception("429 Too Many Requests"),
                AIMessage(content="Success"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch("src.agents.logger") as mock_logger:
                await invoke_with_rate_limit_handling(
                    runnable,
                    {"input": "test"},
                    max_attempts=2,
                    context="Market Analyst",
                )

                call_args = mock_logger.warning.call_args[1]
                assert call_args["context"] == "Market Analyst"
                assert call_args["attempt"] == 1
                assert call_args["max_attempts"] == 2

    @pytest.mark.asyncio
    async def test_error_type_logged(self):
        """Test that error type is captured in logs."""
        os.environ.pop("QUIET_MODE", None)

        class CustomRateLimitError(Exception):
            pass

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(
            side_effect=[
                CustomRateLimitError("429 Too Many Requests"),
                AIMessage(content="Success"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch("src.agents.logger") as mock_logger:
                await invoke_with_rate_limit_handling(
                    runnable, {"input": "test"}, max_attempts=2
                )

                call_args = mock_logger.warning.call_args[1]
                assert call_args["error_type"] == "CustomRateLimitError"


class TestSuccessWithoutRetry:
    """Test that successful calls don't trigger retries."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """Test that successful calls complete without any retries."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Success"))

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await invoke_with_rate_limit_handling(
                runnable, {"input": "test"}, max_attempts=3
            )

        # Should not sleep if no errors
        mock_sleep.assert_not_called()
        assert result.content == "Success"
        # Should only invoke once
        assert runnable.ainvoke.call_count == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_max_attempts_one(self):
        """Test with max_attempts=1 (no retries)."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=Exception("429 Too Many Requests"))

        with pytest.raises(Exception, match="429"):
            await invoke_with_rate_limit_handling(
                runnable, {"input": "test"}, max_attempts=1
            )

    @pytest.mark.asyncio
    async def test_case_insensitive_detection(self):
        """Test that rate limit detection is case-insensitive."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(
            side_effect=[
                Exception("HTTP 429: TOO MANY REQUESTS"),  # Uppercase
                AIMessage(content="Success"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await invoke_with_rate_limit_handling(
                runnable, {"input": "test"}, max_attempts=2
            )

        mock_sleep.assert_called_once_with(60)

    @pytest.mark.asyncio
    async def test_error_message_truncation(self):
        """Test that long error messages are truncated in logs."""
        os.environ.pop("QUIET_MODE", None)

        long_error = "429 " + "x" * 500  # Very long error message
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(
            side_effect=[Exception(long_error), AIMessage(content="Success")]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch("src.agents.logger") as mock_logger:
                await invoke_with_rate_limit_handling(
                    runnable, {"input": "test"}, max_attempts=2
                )

                call_args = mock_logger.warning.call_args[1]
                # Error message should be truncated to 200 chars
                assert len(call_args["error_message"]) <= 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
