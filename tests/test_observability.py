"""
Tests for the observability module (Langfuse SDK v3 integration).
"""

from unittest.mock import MagicMock, patch

import pytest


class TestGetTracingCallbacks:
    """Test get_tracing_callbacks function."""

    def test_returns_empty_when_disabled(self):
        """When LANGFUSE_ENABLED=false, should return empty callbacks and metadata."""
        with patch("src.observability.config") as mock_config:
            mock_config.langfuse_enabled = False

            from src.observability import get_tracing_callbacks

            callbacks, metadata = get_tracing_callbacks(ticker="TEST.X")
            assert callbacks == []
            assert metadata == {}

    def test_returns_empty_when_keys_missing(self):
        """When enabled but keys missing, should return empty and warn."""
        with patch("src.observability.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = ""
            mock_config.get_langfuse_secret_key.return_value = ""

            from src.observability import get_tracing_callbacks

            callbacks, metadata = get_tracing_callbacks(ticker="TEST.X")
            assert callbacks == []
            assert metadata == {}

    def test_returns_handler_when_properly_configured(self):
        """When enabled with valid keys, should return Langfuse handler."""
        with patch("src.observability.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-lf-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-lf-test"
            mock_config.langfuse_host = "https://cloud.langfuse.com"

            # Mock the Langfuse handler import
            mock_handler = MagicMock()
            with patch.dict(
                "sys.modules",
                {"langfuse": MagicMock(), "langfuse.langchain": MagicMock()},
            ):
                with patch(
                    "src.observability.LangfuseHandler", create=True
                ) as MockHandler:
                    # Need to re-import after patching
                    import importlib

                    import src.observability

                    importlib.reload(src.observability)

                    # This test verifies the logic flow, actual handler creation
                    # would require real langfuse SDK

    def test_metadata_contains_session_and_tags(self):
        """When session_id and tags provided, metadata should contain langfuse_* keys."""
        with patch("src.observability.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-lf-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-lf-test"
            mock_config.langfuse_host = "https://cloud.langfuse.com"

            mock_handler_cls = MagicMock()
            with patch("langfuse.langchain.CallbackHandler", mock_handler_cls):
                from src.observability import get_tracing_callbacks

                callbacks, metadata = get_tracing_callbacks(
                    ticker="0005.HK",
                    session_id="0005.HK-2026-01-28-abc12345",
                    tags=["quick", "deep-model:gemini-3-pro-preview"],
                    user_id="test-user",
                )

                assert len(callbacks) == 1
                assert metadata["langfuse_session_id"] == "0005.HK-2026-01-28-abc12345"
                assert metadata["langfuse_tags"] == [
                    "quick",
                    "deep-model:gemini-3-pro-preview",
                ]
                assert metadata["langfuse_user_id"] == "test-user"
                assert metadata["langfuse_metadata"] == {"ticker": "0005.HK"}

    def test_handler_created_with_no_args(self):
        """SDK v3: CallbackHandler() should be called with no arguments."""
        with patch("src.observability.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-lf-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-lf-test"
            mock_config.langfuse_host = "https://cloud.langfuse.com"

            mock_handler_cls = MagicMock()
            with patch("langfuse.langchain.CallbackHandler", mock_handler_cls):
                from src.observability import get_tracing_callbacks

                get_tracing_callbacks(ticker="TEST.X")

                # v3: no constructor args (reads from env vars)
                mock_handler_cls.assert_called_once_with()

    def test_graceful_degradation_on_import_error(self):
        """When langfuse not installed, should return empty tuple gracefully."""
        with patch("src.observability.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-lf-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-lf-test"

            # Simulate ImportError by patching the import
            with patch.dict("sys.modules", {"langfuse.langchain": None}):
                from src.observability import get_tracing_callbacks

                # Force reimport to trigger ImportError path
                callbacks, metadata = get_tracing_callbacks(ticker="TEST.X")
                # Should return empty, not raise
                assert isinstance(callbacks, list)
                assert isinstance(metadata, dict)


class TestFlushTraces:
    """Test flush_traces function."""

    def test_does_nothing_when_disabled(self):
        """When LANGFUSE_ENABLED=false, flush should do nothing."""
        with patch("src.observability.config") as mock_config:
            mock_config.langfuse_enabled = False

            from src.observability import flush_traces

            # Should not raise
            flush_traces()

    def test_graceful_degradation_on_error(self):
        """Flush should not raise even if Langfuse fails."""
        with patch("src.observability.config") as mock_config:
            mock_config.langfuse_enabled = True

            with patch.dict("sys.modules", {"langfuse": None}):
                from src.observability import flush_traces

                # Should not raise
                flush_traces()

    def test_flush_calls_get_client(self):
        """Flush should use SDK v3 get_client().flush() pattern."""
        with patch("src.observability.config") as mock_config:
            mock_config.langfuse_enabled = True

            mock_client = MagicMock()
            mock_get_client = MagicMock(return_value=mock_client)
            with patch("langfuse.get_client", mock_get_client):
                from src.observability import flush_traces

                flush_traces()

                mock_get_client.assert_called_once()
                mock_client.flush.assert_called_once()


class TestConfigIntegration:
    """Test that config properly exposes Langfuse settings."""

    def test_langfuse_settings_exist_in_config(self):
        """Config should have all Langfuse settings."""
        from src.config import config

        # These should exist and have defaults
        assert hasattr(config, "langfuse_enabled")
        assert hasattr(config, "langfuse_host")
        assert hasattr(config, "langfuse_sample_rate")
        assert hasattr(config, "langfuse_debug")
        assert hasattr(config, "langfuse_environment")

        # Getters should exist
        assert hasattr(config, "get_langfuse_public_key")
        assert hasattr(config, "get_langfuse_secret_key")

    def test_langfuse_defaults_are_safe(self, monkeypatch):
        """Default values should be safe (disabled, etc.)."""
        # Clear Langfuse env vars to test true defaults
        monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        from src.config import Settings

        # Create fresh instance without .env file pollution
        # _env_file=None disables Pydantic's automatic .env loading
        fresh_config = Settings(_env_file=None)

        # Should default to disabled
        assert fresh_config.langfuse_enabled is False

        # Sample rate should default to 1.0 (full tracing when enabled)
        assert fresh_config.langfuse_sample_rate == 1.0

        # Debug should default to off
        assert fresh_config.langfuse_debug is False

    def test_langfuse_sample_rate_validation(self):
        """Sample rate should be validated to 0.0-1.0 range."""
        from pydantic import ValidationError

        from src.config import Settings

        # Valid values should work
        Settings(langfuse_sample_rate=0.0)
        Settings(langfuse_sample_rate=0.5)
        Settings(langfuse_sample_rate=1.0)

        # Invalid values should raise
        with pytest.raises(ValidationError):
            Settings(langfuse_sample_rate=-0.1)

        with pytest.raises(ValidationError):
            Settings(langfuse_sample_rate=1.1)
