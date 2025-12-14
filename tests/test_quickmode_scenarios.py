"""
Tests for the quick_mode and thinking_level logic in src/llms.py.
Ensures that the correct model and thinking level are selected based on:
- The --quick flag (quick_mode)
- Whether QUICK_MODEL and DEEP_MODEL are identical
- The model version (Gemini 3+ vs. older models)
"""

import pytest
from unittest.mock import patch, MagicMock

# Subject Matter Under Test: The LLM creation functions
from src.llms import create_deep_thinking_llm, create_quick_thinking_llm

# Mock data for different model configurations
GEMINI_3_PRO = "gemini-3-pro-preview"
GEMINI_2_FLASH = "gemini-2.0-flash"


@pytest.fixture(autouse=True)
def mock_create_gemini_model():
    """
    Auto-used fixture to mock the core `create_gemini_model` factory.
    This allows us to inspect the arguments it's called with (e.g., thinking_level)
    without actually creating a model instance.
    """
    with patch("src.llms.create_gemini_model") as mock:
        # Return a mock object so the function doesn't return None
        mock.return_value = MagicMock()
        yield mock

@pytest.fixture
def mock_config():
    """Mocks the config object in the llms module."""
    with patch("src.llms.config") as mock_conf:
        # Set default values for timeout/retries to avoid dealing with MagicMocks
        mock_conf.api_timeout = 300
        mock_conf.api_retry_attempts = 10
        yield mock_conf


def test_quick_mode_on_identical_gemini_3_models(mock_create_gemini_model, mock_config):
    """
    SCENARIO: --quick is ON, and QUICK_MODEL == DEEP_MODEL (both are Gemini 3).
    EXPECTATION: All agents should use the same model with thinking_level="low".
    """
    # Arrange: Set models to be identical and Gemini 3
    mock_config.quick_think_llm = GEMINI_3_PRO
    mock_config.deep_think_llm = GEMINI_3_PRO

    # Act: Create both a "deep" and "quick" LLM instance
    # The quick_mode=True simulates the --quick flag being passed to the deep LLM
    create_deep_thinking_llm(quick_mode=True)
    create_quick_thinking_llm()

    # Assert: Check the arguments passed to the underlying model factory
    assert mock_create_gemini_model.call_count == 2
    
    # Both calls should have specified thinking_level="low"
    for call in mock_create_gemini_model.call_args_list:
        assert call.args[0] == GEMINI_3_PRO  # Correctly check positional arg
        assert call.kwargs.get("thinking_level") == "low"


def test_quick_mode_off_identical_gemini_3_models(mock_create_gemini_model, mock_config):
    """
    SCENARIO: --quick is OFF, and QUICK_MODEL == DEEP_MODEL (both are Gemini 3).
    EXPECTATION: 
        - Deep agents use the model with thinking_level="high".
        - Quick agents use the model with thinking_level="low".
    """
    # Arrange: Set models to be identical and Gemini 3
    mock_config.quick_think_llm = GEMINI_3_PRO
    mock_config.deep_think_llm = GEMINI_3_PRO

    # Act: Create LLMs with quick_mode=False
    create_deep_thinking_llm(quick_mode=False)
    create_quick_thinking_llm()

    # Assert: Find the specific calls and check their arguments
    assert mock_create_gemini_model.call_count == 2

    # Find the call for the deep LLM (temperature=0.1) and check it
    deep_call = next(c for c in mock_create_gemini_model.call_args_list if c.args[1] == 0.1)
    assert deep_call.args[0] == GEMINI_3_PRO
    assert deep_call.kwargs.get("thinking_level") == "high"

    # Find the call for the quick LLM (temperature=0.3) and check it
    quick_call = next(c for c in mock_create_gemini_model.call_args_list if c.args[1] == 0.3)
    assert quick_call.args[0] == GEMINI_3_PRO
    assert quick_call.kwargs.get("thinking_level") == "low"


def test_quick_mode_off_different_models(mock_create_gemini_model, mock_config):
    """
    SCENARIO: --quick is OFF, and QUICK_MODEL != DEEP_MODEL.
    EXPECTATION: Each agent uses its respective model, and no thinking_level is applied.
    """
    # Arrange: Set models to be different
    mock_config.quick_think_llm = GEMINI_2_FLASH
    mock_config.deep_think_llm = GEMINI_3_PRO

    # Act: Create LLMs with quick_mode=False
    create_deep_thinking_llm(quick_mode=False)
    create_quick_thinking_llm()

    # Assert:
    assert mock_create_gemini_model.call_count == 2
    
    # Check the deep call
    deep_call = next(c for c in mock_create_gemini_model.call_args_list if c.args[0] == GEMINI_3_PRO)
    assert deep_call.kwargs.get("thinking_level") is None

    # Check the quick call
    quick_call = next(c for c in mock_create_gemini_model.call_args_list if c.args[0] == GEMINI_2_FLASH)
    assert quick_call.kwargs.get("thinking_level") is None


def test_no_thinking_level_for_gemini_2_models(mock_create_gemini_model, mock_config):
    """
    SCENARIO: Models are identical but are Gemini 2 (which doesn't support thinking_level).
    EXPECTATION: No thinking_level should be passed, regardless of quick_mode.
    """
    # Arrange: Set models to be identical and Gemini 2
    mock_config.quick_think_llm = GEMINI_2_FLASH
    mock_config.deep_think_llm = GEMINI_2_FLASH

    # Act: Create LLMs for both quick and deep modes
    create_deep_thinking_llm(quick_mode=True)
    create_deep_thinking_llm(quick_mode=False)
    create_quick_thinking_llm()

    # Assert: No call should have a thinking_level set
    assert mock_create_gemini_model.call_count == 3
    for call in mock_create_gemini_model.call_args_list:
        assert call.args[0] == GEMINI_2_FLASH
        assert call.kwargs.get("thinking_level") is None