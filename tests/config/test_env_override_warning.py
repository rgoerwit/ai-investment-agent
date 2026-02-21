"""
Tests for environment variable override warning system.

Ensures that users are warned when shell environment variables override .env file
settings, particularly for rate limits where mismatches cause severe performance issues.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest


class TestEnvOverrideWarning:
    """Test environment variable override detection and warnings."""

    def test_parse_env_file_basic(self):
        """Test that .env file parsing works correctly."""
        from src.config import _parse_env_file

        # Create mock .env content
        env_content = """
# Comment line
GEMINI_RPM_LIMIT=15    # Free tier
GOOGLE_API_KEY=test_key
# Another comment
FINNHUB_API_KEY=another_key

EMPTY_LINE_ABOVE=yes
"""
        with patch("builtins.open", mock_open(read_data=env_content)):
            with patch.object(Path, "exists", return_value=True):
                result = _parse_env_file()

        # Verify parsing
        assert result["GEMINI_RPM_LIMIT"] == "15"
        assert result["GOOGLE_API_KEY"] == "test_key"
        assert result["FINNHUB_API_KEY"] == "another_key"
        assert result["EMPTY_LINE_ABOVE"] == "yes"

    def test_parse_env_file_inline_comments(self):
        """Test that inline comments are stripped correctly."""
        from src.config import _parse_env_file

        env_content = """
GEMINI_RPM_LIMIT=15    # Free tier (default)
GEMINI_RPM_LIMIT_PAID=360   # Paid tier 1
"""
        with patch("builtins.open", mock_open(read_data=env_content)):
            with patch.object(Path, "exists", return_value=True):
                result = _parse_env_file()

        # Comments should be stripped
        assert result["GEMINI_RPM_LIMIT"] == "15"
        assert result["GEMINI_RPM_LIMIT_PAID"] == "360"

    def test_parse_env_file_quoted_values(self):
        """Test that quoted values are handled correctly."""
        from src.config import _parse_env_file

        env_content = """
API_KEY_SINGLE='single_quoted_value'
API_KEY_DOUBLE="double_quoted_value"
API_KEY_UNQUOTED=unquoted_value
"""
        with patch("builtins.open", mock_open(read_data=env_content)):
            with patch.object(Path, "exists", return_value=True):
                result = _parse_env_file()

        # Quotes should be removed
        assert result["API_KEY_SINGLE"] == "single_quoted_value"
        assert result["API_KEY_DOUBLE"] == "double_quoted_value"
        assert result["API_KEY_UNQUOTED"] == "unquoted_value"

    def test_check_env_overrides_warns_on_higher_rate_limit(self, caplog):
        """Test that warning is emitted when shell has higher rate limit than .env."""
        import logging

        from src.config import _check_env_overrides

        env_content = "GEMINI_RPM_LIMIT=15"

        with patch("builtins.open", mock_open(read_data=env_content)):
            with patch.object(Path, "exists", return_value=True):
                with patch.dict(os.environ, {"GEMINI_RPM_LIMIT": "360"}):
                    with caplog.at_level(logging.WARNING, logger="src.config"):
                        _check_env_overrides()

        # Verify warning was logged
        all_messages = " ".join([record.message for record in caplog.records])
        assert "SHELL ENVIRONMENT OVERRIDE DETECTED" in all_messages
        assert "360" in all_messages
        assert "15" in all_messages

    def test_check_env_overrides_no_warning_when_matching(self, caplog):
        """Test that no warning is emitted when shell matches .env."""
        import logging

        from src.config import _check_env_overrides

        env_content = "GEMINI_RPM_LIMIT=15"

        with patch("builtins.open", mock_open(read_data=env_content)):
            with patch.object(Path, "exists", return_value=True):
                with patch.dict(os.environ, {"GEMINI_RPM_LIMIT": "15"}):
                    with caplog.at_level(logging.WARNING, logger="src.config"):
                        _check_env_overrides()

        # No warning should be emitted
        all_messages = " ".join([record.message for record in caplog.records])
        assert "SHELL ENVIRONMENT OVERRIDE DETECTED" not in all_messages

    def test_check_env_overrides_info_when_lower(self, caplog):
        """Test that info log (not warning) is emitted when shell has lower rate limit."""
        import logging

        from src.config import _check_env_overrides

        env_content = "GEMINI_RPM_LIMIT=360"

        with patch("builtins.open", mock_open(read_data=env_content)):
            with patch.object(Path, "exists", return_value=True):
                with patch.dict(os.environ, {"GEMINI_RPM_LIMIT": "15"}):
                    with caplog.at_level(logging.INFO, logger="src.config"):
                        _check_env_overrides()

        # Info log should be present, not warning
        all_messages = " ".join([record.message for record in caplog.records])
        assert "Shell environment override" in all_messages
        # Should not have the big warning message
        assert "may cause rate limit errors" not in all_messages


class TestRateLimitMismatchScenario:
    """Integration test for the specific rate limit mismatch scenario."""

    def test_rate_limit_mismatch_causes_warning(self, caplog):
        """Regression test: Detect when 360 RPM override conflicts with 15 RPM .env setting."""
        import logging

        from src.config import validate_environment_variables

        # Simulate the exact scenario: .env has 15, shell has 360
        env_content = """
GOOGLE_API_KEY=test_key_123
FINNHUB_API_KEY=test_finnhub
TAVILY_API_KEY=test_tavily
GEMINI_RPM_LIMIT=15
"""
        # Mock config getters (validate_environment_variables now uses config singleton)
        mock_config = MagicMock()
        mock_config.get_google_api_key.return_value = "test_key_123"
        mock_config.get_finnhub_api_key.return_value = "test_finnhub"
        mock_config.get_tavily_api_key.return_value = "test_tavily"
        mock_config.get_eodhd_api_key.return_value = ""
        mock_config.langsmith_project = "test-project"

        with patch("builtins.open", mock_open(read_data=env_content)):
            with patch.object(Path, "exists", return_value=True):
                with patch.dict(
                    os.environ,
                    {
                        "GOOGLE_API_KEY": "test_key_123",
                        "FINNHUB_API_KEY": "test_finnhub",
                        "TAVILY_API_KEY": "test_tavily",
                        "GEMINI_RPM_LIMIT": "360",  # Shell override
                    },
                ):
                    with patch("src.config.config", mock_config):
                        with patch("src.config.configure_langsmith_tracing"):
                            with caplog.at_level(logging.WARNING, logger="src.config"):
                                validate_environment_variables()

        # Verify the specific warning is logged
        all_messages = " ".join(
            [
                record.message
                for record in caplog.records
                if record.levelname == "WARNING"
            ]
        )
        assert "SHELL ENVIRONMENT OVERRIDE DETECTED" in all_messages
        assert "HTTP 429 errors" in all_messages
        assert "unset GEMINI_RPM_LIMIT" in all_messages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
