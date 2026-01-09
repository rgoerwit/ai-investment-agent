"""Pytest configuration for Multi-Agent Trading System tests."""

import logging
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Capture real API key if present (for integration tests)
_REAL_GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """
    Set up test environment variables.
    This fixture runs for the entire session and applies default MOCK values.
    Individual tests that need real keys (integration tests) must override this.
    """
    test_env = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "ERROR",
        "ENABLE_MEMORY": "false",
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "GOOGLE_API_KEY": "test-key",  # Default to dummy key
        "TAVILY_API_KEY": "test-key",
        "FINNHUB_API_KEY": "test-key",
    }

    # 1. Patch os.environ
    with patch.dict(os.environ, test_env, clear=False):
        # 2. Update the global config singleton IN-PLACE
        # We import here so the import happens AFTER os.environ is patched
        from src.config import Settings, config

        # Create a new Settings object which reads the NOW-PATCHED os.environ
        # (and ignores any local .env file because it's already loaded)
        new_settings = Settings()

        # Update the existing global singleton's state to match the new settings
        # This fixes the "import time" problem because we modify the object everyone is holding
        config.__dict__.update(new_settings.__dict__)

        yield


@pytest.fixture(autouse=True)
def configure_structlog_for_tests():
    """Configure structlog for test environment."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            # FIX: Removed format_exc_info because ConsoleRenderer handles exceptions prettily
            # structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.root.setLevel(logging.WARNING)
    yield


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    from unittest.mock import AsyncMock, MagicMock

    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(content="BUY"))
    return mock


@pytest.fixture(autouse=True)
def skip_chart_generation_in_tests(monkeypatch):
    """Prevent chart generation from writing to ./images/ during tests.

    QuietModeReporter generates charts by default, writing to ./images/.
    This fixture patches the chart generation methods to no-op, preventing
    leftover files from accumulating in the repo.

    Tests that specifically need chart generation should use tmp_path
    and explicitly set image_dir.
    """
    from src.report_generator import QuietModeReporter

    monkeypatch.setattr(QuietModeReporter, "_generate_chart", lambda self, r: None)
    monkeypatch.setattr(
        QuietModeReporter, "_generate_radar_chart", lambda self, r: None
    )
