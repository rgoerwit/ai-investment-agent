"""
Tests for the simplified thinking_level logic in src/llms.py.
Verifies that thinking_level is set based only on the model version,
not on any comparison between DEEP_MODEL and QUICK_MODEL.
"""
import pytest
from unittest.mock import patch, MagicMock

# The functions to test
from src.llms import create_deep_thinking_llm, create_quick_thinking_llm, _is_gemini_v3_or_greater

# Mock data for model names
GEMINI_3_PRO = "gemini-3-pro-preview"
GEMINI_4_ULTRA = "gemini-4-ultra"
GEMINI_2_FLASH = "gemini-2.0-flash"


@pytest.fixture(autouse=True)
def mock_create_gemini_model():
    """Mocks the core `create_gemini_model` factory to inspect its inputs."""
    with patch("src.llms.create_gemini_model") as mock:
        mock.return_value = MagicMock()
        yield mock

@pytest.fixture
def mock_config():
    """Mocks the config object in the llms module to control model names."""
    with patch("src.llms.config") as mock_conf:
        mock_conf.api_timeout = 300
        mock_conf.api_retry_attempts = 10
        yield mock_conf


@pytest.mark.parametrize("model_name, expected", [
    ("gemini-3-pro", True),
    ("gemini-3.5-pro", True),
    ("gemini-4-ultra", True),
    ("gemini-10-alpha", True),
    ("gemini-2.0-flash", False),
    ("gemini-1.5-pro", False),
    ("not-a-gemini", False),
    ("gemini-pro", False),
])
def test_is_gemini_v3_or_greater_helper(model_name, expected):
    """Tests the version checking helper function directly for robustness."""
    assert _is_gemini_v3_or_greater(model_name) == expected


def test_quick_llm_sets_low_thinking_level_on_gemini_3_plus(mock_create_gemini_model, mock_config):
    """
    SCENARIO: The QUICK_MODEL is a Gemini 3+ model.
    EXPECTATION: thinking_level should be "low".
    """
    # Arrange: Use a Gemini 3+ model for the quick LLM
    mock_config.quick_think_llm = GEMINI_3_PRO
    
    # Act
    create_quick_thinking_llm()
    
    # Assert
    mock_create_gemini_model.assert_called_once()
    call_kwargs = mock_create_gemini_model.call_args.kwargs
    assert call_kwargs.get("thinking_level") == "low"


def test_quick_llm_has_no_thinking_level_on_gemini_2(mock_create_gemini_model, mock_config):
    """
    SCENARIO: The QUICK_MODEL is a Gemini 2 model.
    EXPECTATION: thinking_level should be None.
    """
    # Arrange: Use a Gemini 2 model for the quick LLM
    mock_config.quick_think_llm = GEMINI_2_FLASH
    
    # Act
    create_quick_thinking_llm()
    
    # Assert
    mock_create_gemini_model.assert_called_once()
    call_kwargs = mock_create_gemini_model.call_args.kwargs
    assert call_kwargs.get("thinking_level") is None


def test_deep_llm_sets_high_thinking_level_on_gemini_4(mock_create_gemini_model, mock_config):
    """
    SCENARIO: The DEEP_MODEL is a Gemini 4 model.
    EXPECTATION: thinking_level should be "high".
    """
    # Arrange: Use a Gemini 4 model for the deep LLM
    mock_config.deep_think_llm = GEMINI_4_ULTRA
    
    # Act
    create_deep_thinking_llm()
    
    # Assert
    mock_create_gemini_model.assert_called_once()
    call_kwargs = mock_create_gemini_model.call_args.kwargs
    assert call_kwargs.get("thinking_level") == "high"


def test_deep_llm_has_no_thinking_level_on_gemini_2(mock_create_gemini_model, mock_config):
    """
    SCENARIO: The DEEP_MODEL is a Gemini 2 model.
    EXPECTATION: thinking_level should be None.
    """
    # Arrange: Use a Gemini 2 model for the deep LLM
    mock_config.deep_think_llm = GEMINI_2_FLASH
    
    # Act
    create_deep_thinking_llm()
    
    # Assert
    mock_create_gemini_model.assert_called_once()
    call_kwargs = mock_create_gemini_model.call_args.kwargs
    assert call_kwargs.get("thinking_level") is None