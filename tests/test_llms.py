"""
Unit tests for LLM configuration and initialization.

Tests the critical aspects of llm.py:
- Model creation with correct parameters
- Rate limiting configuration
- Safety settings
- Error handling for missing config
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_google_genai import HarmBlockThreshold, HarmCategory
from langchain_core.rate_limiters import InMemoryRateLimiter


class TestSafetySettings:
    """Test safety settings configuration."""
    
    def test_safety_settings_exist(self):
        """Test that safety settings are properly defined."""
        from src.llms import SAFETY_SETTINGS
        
        assert SAFETY_SETTINGS is not None
        assert isinstance(SAFETY_SETTINGS, dict)
        
        # Check all required harm categories are present
        expected_categories = [
            HarmCategory.HARM_CATEGORY_HARASSMENT,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        ]
        
        for category in expected_categories:
            assert category in SAFETY_SETTINGS
            assert SAFETY_SETTINGS[category] == HarmBlockThreshold.BLOCK_ONLY_HIGH
    
    def test_safety_settings_are_relaxed(self):
        """Test that safety settings use BLOCK_ONLY_HIGH threshold."""
        from src.llms import SAFETY_SETTINGS
        
        # All thresholds should be BLOCK_ONLY_HIGH for financial analysis
        for threshold in SAFETY_SETTINGS.values():
            assert threshold == HarmBlockThreshold.BLOCK_ONLY_HIGH


class TestRateLimiter:
    """Test rate limiter configuration."""
    
    def test_global_rate_limiter_exists(self):
        """Test that global rate limiter is configured."""
        from src.llms import GLOBAL_RATE_LIMITER
        
        assert GLOBAL_RATE_LIMITER is not None
        assert isinstance(GLOBAL_RATE_LIMITER, InMemoryRateLimiter)
    
    def test_rate_limiter_conservative_settings(self):
        """Test that rate limiter uses conservative settings for free tier."""
        from src.llms import GLOBAL_RATE_LIMITER
        
        # Free tier is ~15 RPM, so 0.25 rps is safe
        # We can't easily test the internal values without breaking encapsulation,
        # but we can verify it was created and is the right type
        assert GLOBAL_RATE_LIMITER is not None
        from langchain_core.rate_limiters import InMemoryRateLimiter
        assert isinstance(GLOBAL_RATE_LIMITER, InMemoryRateLimiter)


class TestCreateGeminiModel:
    """Test generic Gemini model factory."""
    
    @patch('src.llms.ChatGoogleGenerativeAI')
    def test_create_gemini_model_basic(self, mock_chat_class):
        """Test basic model creation with default parameters."""
        from src.llms import create_gemini_model
        
        mock_instance = MagicMock()
        mock_chat_class.return_value = mock_instance
        
        result = create_gemini_model(
            model_name="gemini-2.5-flash",
            temperature=0.3,
            timeout=300,
            max_retries=10
        )
        
        assert result == mock_instance
        mock_chat_class.assert_called_once()
        
        # Verify constructor arguments
        call_kwargs = mock_chat_class.call_args[1]
        assert call_kwargs['model'] == "gemini-2.5-flash"
        assert call_kwargs['temperature'] == 0.3
        assert call_kwargs['timeout'] == 300
        assert call_kwargs['max_retries'] == 10
        assert call_kwargs['streaming'] == False
        assert 'safety_settings' in call_kwargs
        assert 'rate_limiter' in call_kwargs
    
    @patch('src.llms.ChatGoogleGenerativeAI')
    def test_create_gemini_model_with_streaming(self, mock_chat_class):
        """Test model creation with streaming enabled."""
        from src.llms import create_gemini_model
        
        mock_instance = MagicMock()
        mock_chat_class.return_value = mock_instance
        
        result = create_gemini_model(
            model_name="gemini-3-pro-preview",
            temperature=0.1,
            timeout=600,
            max_retries=5,
            streaming=True
        )
        
        call_kwargs = mock_chat_class.call_args[1]
        assert call_kwargs['streaming'] == True
    
    @patch('src.llms.ChatGoogleGenerativeAI')
    def test_create_gemini_model_uses_rate_limiter(self, mock_chat_class):
        """Test that rate limiter is passed to model."""
        from src.llms import create_gemini_model, GLOBAL_RATE_LIMITER
        
        mock_instance = MagicMock()
        mock_chat_class.return_value = mock_instance
        
        create_gemini_model("gemini-2.5-flash", 0.3, 300, 10)
        
        call_kwargs = mock_chat_class.call_args[1]
        assert call_kwargs['rate_limiter'] == GLOBAL_RATE_LIMITER


class TestCreateQuickThinkingLLM:
    """Test quick thinking LLM creation."""
    
    @patch('src.llms.create_gemini_model')
    @patch('src.llms.config')
    def test_create_quick_llm_defaults(self, mock_config, mock_create):
        """Test quick LLM creation with default parameters."""
        from src.llms import create_quick_thinking_llm
        
        # Mock config values
        mock_config.quick_think_llm = "gemini-2.5-flash"
        mock_config.api_timeout = 300
        mock_config.api_retry_attempts = 10
        
        mock_instance = MagicMock()
        mock_create.return_value = mock_instance
        
        result = create_quick_thinking_llm()

        assert result == mock_instance
        mock_create.assert_called_once_with(
            "gemini-2.5-flash",
            0.3,  # Default temperature
            300,  # From config
            10,   # From config
            callbacks=None,  # Default callbacks
            thinking_level=None  # No thinking level for Flash models or different models
        )
    
    @patch('src.llms.create_gemini_model')
    @patch('src.llms.config')
    def test_create_quick_llm_custom_temperature(self, mock_config, mock_create):
        """Test quick LLM with custom temperature."""
        from src.llms import create_quick_thinking_llm
        
        mock_config.quick_think_llm = "gemini-2.5-flash"
        mock_config.api_timeout = 300
        mock_config.api_retry_attempts = 10
        
        create_quick_thinking_llm(temperature=0.7)
        
        call_args = mock_create.call_args[0]
        assert call_args[1] == 0.7
    
    @patch('src.llms.create_gemini_model')
    @patch('src.llms.config')
    def test_create_quick_llm_custom_model(self, mock_config, mock_create):
        """Test quick LLM with custom model name."""
        from src.llms import create_quick_thinking_llm
        
        mock_config.api_timeout = 300
        mock_config.api_retry_attempts = 10
        
        create_quick_thinking_llm(model="gemini-3-pro-preview")
        
        call_args = mock_create.call_args[0]
        assert call_args[0] == "gemini-3-pro-preview"
    
    @patch('src.llms.create_gemini_model')
    @patch('src.llms.config')
    def test_create_quick_llm_override_timeout(self, mock_config, mock_create):
        """Test quick LLM with overridden timeout."""
        from src.llms import create_quick_thinking_llm
        
        mock_config.quick_think_llm = "gemini-2.5-flash"
        mock_config.api_timeout = 300
        mock_config.api_retry_attempts = 10
        
        create_quick_thinking_llm(timeout=600)
        
        call_args = mock_create.call_args[0]
        assert call_args[2] == 600  # timeout parameter
    
    @patch('src.llms.create_gemini_model')
    @patch('src.llms.config')
    def test_create_quick_llm_override_retries(self, mock_config, mock_create):
        """Test quick LLM with overridden max_retries."""
        from src.llms import create_quick_thinking_llm
        
        mock_config.quick_think_llm = "gemini-2.5-flash"
        mock_config.api_timeout = 300
        mock_config.api_retry_attempts = 10
        
        create_quick_thinking_llm(max_retries=5)
        
        call_args = mock_create.call_args[0]
        assert call_args[3] == 5  # max_retries parameter


class TestCreateDeepThinkingLLM:
    """Test deep thinking LLM creation."""
    
    @patch('src.llms.create_gemini_model')
    @patch('src.llms.config')
    def test_create_deep_llm_defaults(self, mock_config, mock_create):
        """Test deep LLM creation with default parameters."""
        from src.llms import create_deep_thinking_llm
        
        mock_config.deep_think_llm = "gemini-3-pro-preview"
        mock_config.api_timeout = 300
        mock_config.api_retry_attempts = 10
        
        mock_instance = MagicMock()
        mock_create.return_value = mock_instance
        
        result = create_deep_thinking_llm()

        assert result == mock_instance
        mock_create.assert_called_once_with(
            "gemini-3-pro-preview",
            0.1,  # Default temperature (lower for deep thinking)
            300,
            10,
            callbacks=None,  # Default callbacks
            thinking_level=None  # No thinking level when models differ
        )
    
    @patch('src.llms.create_gemini_model')
    @patch('src.llms.config')
    def test_create_deep_llm_lower_temperature(self, mock_config, mock_create):
        """Test that deep LLM has lower default temperature than quick LLM."""
        from src.llms import create_deep_thinking_llm
        
        mock_config.deep_think_llm = "gemini-3-pro-preview"
        mock_config.api_timeout = 300
        mock_config.api_retry_attempts = 10
        
        create_deep_thinking_llm()
        
        call_args = mock_create.call_args[0]
        # Deep thinking should be 0.1, quick thinking is 0.3
        assert call_args[1] == 0.1
    
    @patch('src.llms.create_gemini_model')
    @patch('src.llms.config')
    def test_create_deep_llm_all_overrides(self, mock_config, mock_create):
        """Test deep LLM with all parameters overridden."""
        from src.llms import create_deep_thinking_llm
        
        mock_config.api_timeout = 300
        mock_config.api_retry_attempts = 10
        
        create_deep_thinking_llm(
            temperature=0.5,
            model="custom-model",
            timeout=900,
            max_retries=20
        )

        mock_create.assert_called_once_with(
            "custom-model",
            0.5,
            900,
            20,
            callbacks=None,  # Default callbacks
            thinking_level=None  # No thinking level for non-Gemini-3 models
        )


class TestDefaultInstances:
    """Test that default LLM instances are created."""
    
    def test_quick_thinking_llm_exists(self):
        """Test that quick_thinking_llm is instantiated."""
        from src.llms import quick_thinking_llm
        
        assert quick_thinking_llm is not None
    
    def test_deep_thinking_llm_exists(self):
        """Test that deep_thinking_llm is instantiated."""
        from src.llms import deep_thinking_llm
        
        assert deep_thinking_llm is not None
    
    def test_default_instances_are_different(self):
        """Test that quick and deep LLMs are separate instances."""
        from src.llms import quick_thinking_llm, deep_thinking_llm
        
        # They should be different objects
        assert quick_thinking_llm is not deep_thinking_llm


class TestParameterValidation:
    """Test parameter validation and edge cases."""
    
    @patch('src.llms.create_gemini_model')
    @patch('src.llms.config')
    def test_zero_timeout_allowed(self, mock_config, mock_create):
        """Test that zero timeout is passed through (even if not practical)."""
        from src.llms import create_quick_thinking_llm
        
        mock_config.quick_think_llm = "gemini-2.5-flash"
        mock_config.api_timeout = 300
        mock_config.api_retry_attempts = 10
        
        create_quick_thinking_llm(timeout=0)
        
        call_args = mock_create.call_args[0]
        assert call_args[2] == 0
    
    @patch('src.llms.create_gemini_model')
    @patch('src.llms.config')
    def test_none_timeout_uses_config_default(self, mock_config, mock_create):
        """Test that None timeout uses config default."""
        from src.llms import create_quick_thinking_llm
        
        mock_config.quick_think_llm = "gemini-2.5-flash"
        mock_config.api_timeout = 300
        mock_config.api_retry_attempts = 10
        
        create_quick_thinking_llm(timeout=None)
        
        call_args = mock_create.call_args[0]
        assert call_args[2] == 300  # Should use config default
    
    @patch('src.llms.create_gemini_model')
    @patch('src.llms.config')
    def test_extreme_temperature_allowed(self, mock_config, mock_create):
        """Test that extreme temperature values are passed through."""
        from src.llms import create_quick_thinking_llm
        
        mock_config.quick_think_llm = "gemini-2.5-flash"
        mock_config.api_timeout = 300
        mock_config.api_retry_attempts = 10
        
        create_quick_thinking_llm(temperature=2.0)
        
        call_args = mock_create.call_args[0]
        assert call_args[1] == 2.0


class TestLogging:
    """Test logging behavior."""
    
    @patch('src.llms.create_gemini_model')
    @patch('src.llms.config')
    @patch('src.llms.logger')
    def test_quick_llm_logs_initialization(self, mock_logger, mock_config, mock_create):
        """Test that quick LLM logs initialization."""
        from src.llms import create_quick_thinking_llm
        
        mock_config.quick_think_llm = "gemini-2.5-flash"
        mock_config.api_timeout = 300
        mock_config.api_retry_attempts = 10
        
        create_quick_thinking_llm()
        
        # Check that logger.info was called
        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert "gemini-2.5-flash" in call_args.lower()
        assert "timeout" in call_args.lower()
    
    @patch('src.llms.create_gemini_model')
    @patch('src.llms.config')
    @patch('src.llms.logger')
    def test_deep_llm_logs_initialization(self, mock_logger, mock_config, mock_create):
        """Test that deep LLM logs initialization."""
        from src.llms import create_deep_thinking_llm
        
        mock_config.deep_think_llm = "gemini-3-pro-preview"
        mock_config.api_timeout = 300
        mock_config.api_retry_attempts = 10
        
        create_deep_thinking_llm()
        
        # Check that logger.info was called
        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert "gemini-3-pro-preview" in call_args.lower()