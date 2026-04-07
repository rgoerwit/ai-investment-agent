"""
Tests for the simplified thinking_level logic in src/llms.py.
Verifies that thinking_level is set based only on the model version,
not on any comparison between DEEP_MODEL and QUICK_MODEL.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest


def _is_gemini_v3_or_greater(model_name: str) -> bool:
    from src.llms import _is_gemini_v3_or_greater as impl

    return impl(model_name)


def _create_quick_thinking_llm(*args, **kwargs):
    from src.llms import create_quick_thinking_llm

    return create_quick_thinking_llm(*args, **kwargs)


def _create_deep_thinking_llm(*args, **kwargs):
    from src.llms import create_deep_thinking_llm

    return create_deep_thinking_llm(*args, **kwargs)


def _create_gemini_model(*args, **kwargs):
    from src.llms import create_gemini_model

    return create_gemini_model(*args, **kwargs)


# Mock data for model names
GEMINI_3_PRO = "gemini-3-pro-preview"
GEMINI_4_ULTRA = "gemini-4-ultra"
GEMINI_2_FLASH = "gemini-2.0-flash"
GEMINI_2_5_FLASH = "gemini-2.5-flash"


@pytest.fixture(autouse=True)
def mock_create_gemini_model(request):
    """Mocks the core `create_gemini_model` factory to inspect its inputs."""
    if request.node.name in {
        "test_create_gemini_model_logs_thinking_level_at_debug",
        "test_create_gemini_model_uses_thinking_budget_for_gemini_2_5",
    }:
        yield None
        return

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


@pytest.mark.parametrize(
    "model_name, expected",
    [
        ("gemini-3-pro", True),
        ("gemini-3.5-pro", True),
        ("gemini-4-ultra", True),
        ("gemini-10-alpha", True),
        ("gemini-2.0-flash", False),
        ("gemini-1.5-pro", False),
        ("not-a-gemini", False),
        ("gemini-pro", False),
    ],
)
def test_is_gemini_v3_or_greater_helper(model_name, expected):
    """Tests the version checking helper function directly for robustness."""
    assert _is_gemini_v3_or_greater(model_name) == expected


def test_quick_llm_sets_low_thinking_level_on_gemini_3_plus(
    mock_create_gemini_model, mock_config
):
    """
    SCENARIO: The QUICK_MODEL is a Gemini 3+ model.
    EXPECTATION: thinking_level should be "low".
    """
    # Arrange: Use a Gemini 3+ model for the quick LLM
    mock_config.quick_think_llm = GEMINI_3_PRO

    # Act
    _create_quick_thinking_llm()

    # Assert
    mock_create_gemini_model.assert_called_once()
    call_kwargs = mock_create_gemini_model.call_args.kwargs
    assert call_kwargs.get("thinking_level") == "low"


def test_quick_llm_has_no_thinking_level_on_gemini_2(
    mock_create_gemini_model, mock_config
):
    """
    SCENARIO: The QUICK_MODEL is a Gemini 2 model.
    EXPECTATION: thinking_level should be None.
    """
    # Arrange: Use a Gemini 2 model for the quick LLM
    mock_config.quick_think_llm = GEMINI_2_FLASH

    # Act
    _create_quick_thinking_llm()

    # Assert
    mock_create_gemini_model.assert_called_once()
    call_kwargs = mock_create_gemini_model.call_args.kwargs
    assert call_kwargs.get("thinking_level") is None
    assert call_kwargs.get("max_output_tokens") is None


def test_quick_llm_sets_low_thinking_level_on_gemini_2_5(
    mock_create_gemini_model, mock_config
):
    mock_config.quick_think_llm = GEMINI_2_5_FLASH

    _create_quick_thinking_llm()

    call_kwargs = mock_create_gemini_model.call_args.kwargs
    assert call_kwargs.get("thinking_level") == "low"


def test_deep_llm_sets_high_thinking_level_on_gemini_4(
    mock_create_gemini_model, mock_config
):
    """
    SCENARIO: The DEEP_MODEL is a Gemini 4 model.
    EXPECTATION: thinking_level should be "high".
    """
    # Arrange: Use a Gemini 4 model for the deep LLM
    mock_config.deep_think_llm = GEMINI_4_ULTRA

    # Act
    _create_deep_thinking_llm()

    # Assert
    mock_create_gemini_model.assert_called_once()
    call_kwargs = mock_create_gemini_model.call_args.kwargs
    assert call_kwargs.get("thinking_level") == "high"


def test_deep_llm_has_no_thinking_level_on_gemini_2(
    mock_create_gemini_model, mock_config
):
    """
    SCENARIO: The DEEP_MODEL is a Gemini 2 model.
    EXPECTATION: thinking_level should be None.
    """
    # Arrange: Use a Gemini 2 model for the deep LLM
    mock_config.deep_think_llm = GEMINI_2_FLASH

    # Act
    _create_deep_thinking_llm()

    # Assert
    mock_create_gemini_model.assert_called_once()
    call_kwargs = mock_create_gemini_model.call_args.kwargs
    assert call_kwargs.get("thinking_level") is None


def test_deep_llm_sets_high_thinking_level_on_gemini_2_5(
    mock_create_gemini_model, mock_config
):
    mock_config.deep_think_llm = GEMINI_2_5_FLASH

    _create_deep_thinking_llm()

    call_kwargs = mock_create_gemini_model.call_args.kwargs
    assert call_kwargs.get("thinking_level") == "high"


def test_quick_llm_passes_through_max_output_tokens(
    mock_create_gemini_model, mock_config
):
    mock_config.quick_think_llm = GEMINI_3_PRO

    _create_quick_thinking_llm(max_output_tokens=4096)

    call_kwargs = mock_create_gemini_model.call_args.kwargs
    assert call_kwargs.get("max_output_tokens") == 4096


def test_deep_llm_passes_through_max_output_tokens(
    mock_create_gemini_model, mock_config
):
    mock_config.deep_think_llm = GEMINI_4_ULTRA

    _create_deep_thinking_llm(max_output_tokens=8192)

    call_kwargs = mock_create_gemini_model.call_args.kwargs
    assert call_kwargs.get("max_output_tokens") == 8192


def test_create_gemini_model_logs_thinking_level_at_debug(caplog):
    with patch("src.llms.ChatGoogleGenerativeAI", return_value=MagicMock()):
        caplog.set_level(logging.DEBUG, logger="src.llms")

        _create_gemini_model(
            "gemini-3-pro-preview",
            temperature=0.1,
            timeout=30,
            max_retries=1,
            thinking_level="high",
        )

    info_messages = [
        record.message
        for record in caplog.records
        if record.levelno == logging.INFO and "thinking_level_applied" in record.message
    ]
    debug_records = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG and "thinking_level_applied" in r.message
    ]

    assert info_messages == []
    assert debug_records
    assert "high" in debug_records[0].message
    assert "gemini-3-pro-preview" in debug_records[0].message


def test_create_gemini_model_uses_thinking_budget_for_gemini_2_5():
    with patch("src.llms.ChatGoogleGenerativeAI", return_value=MagicMock()) as mock_llm:
        with patch("src.llms.config") as mock_config:
            mock_config.llm_base_output_tokens = 4096
            mock_config.get_google_api_key.return_value = "test-key"

            _create_gemini_model(
                GEMINI_2_5_FLASH,
                temperature=0.1,
                timeout=30,
                max_retries=1,
                thinking_level="medium",
            )

    call_kwargs = mock_llm.call_args.kwargs
    assert call_kwargs["thinking_budget"] == 4096
    assert "thinking_level" not in call_kwargs
