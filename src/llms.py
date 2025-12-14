"""
LLM configuration and initialization module.
Updated for Google Gemini 3 with Safety Settings and Rate Limiting.
Includes token tracking for cost monitoring.
UPDATED: Configurable rate limits via GEMINI_RPM_LIMIT environment variable.
UPDATED: Added OpenAI consultant LLM for cross-validation (Dec 2025).
"""

import logging
import os
from typing import Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.callbacks import BaseCallbackHandler
from src.config import config

logger = logging.getLogger(__name__)

# Relax safety settings slightly for financial/market analysis context
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

# Configurable rate limiter based on Gemini API tier
# Tier detection via GEMINI_RPM_LIMIT environment variable:
#   - Free tier: 15 RPM (default)
#   - Paid tier 1: 360 RPM (set GEMINI_RPM_LIMIT=360)
#   - Paid tier 2: 1000 RPM (set GEMINI_RPM_LIMIT=1000)
#
# Rate limiter settings are calculated to be conservative:
# - RPS = RPM / 60 (convert to requests per second)
# - Reduce by 20% for safety margin to avoid hitting limits
# - max_bucket_size allows brief bursts without throttling

def _create_rate_limiter_from_rpm(rpm: int) -> InMemoryRateLimiter:
    """
    Create a rate limiter from RPM (requests per minute) setting.

    Args:
        rpm: Target requests per minute (e.g., 15 for free tier, 360 for paid)

    Returns:
        Configured InMemoryRateLimiter
    """
    # Convert RPM to RPS with 20% safety margin
    safety_factor = 0.8  # Use 80% of limit to avoid edge cases
    rps = (rpm / 60.0) * safety_factor

    # Bucket size: allow bursts up to 10% of RPM for parallel agent execution
    max_bucket = max(5, int(rpm * 0.1))

    logger.info(
        f"Rate limiter configured: {rpm} RPM → {rps:.2f} RPS "
        f"(80% of limit, bucket size: {max_bucket})"
    )

    return InMemoryRateLimiter(
        requests_per_second=rps,
        check_every_n_seconds=0.1,
        max_bucket_size=max_bucket
    )

# Initialize global rate limiter from config
GLOBAL_RATE_LIMITER = _create_rate_limiter_from_rpm(config.gemini_rpm_limit)

def create_gemini_model(
    model_name: str,
    temperature: float,
    timeout: int,
    max_retries: int,
    streaming: bool = False,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    thinking_level: Optional[str] = None
) -> BaseChatModel:
    """
    Generic factory for Gemini models with optional callbacks and thinking level.

    Args:
        model_name: Gemini model identifier
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        max_retries: Max retry attempts
        streaming: Enable streaming responses
        callbacks: Optional callback handlers
        thinking_level: Optional thinking level ("low" or "high") for Gemini 3+ models

    Returns:
        Configured ChatGoogleGenerativeAI instance

    Note:
        thinking_level is only applied if the model supports it (Gemini 3+ models).
        If provided for unsupported models, it will be silently ignored by the API.
    """

    # Build kwargs for ChatGoogleGenerativeAI
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
        "safety_settings": SAFETY_SETTINGS,
        "streaming": streaming,
        "rate_limiter": GLOBAL_RATE_LIMITER,
        "convert_system_message_to_human": False,
        "max_output_tokens": 32768,
        "callbacks": callbacks or []
    }

    # Apply thinking_level if provided and model supports it
    # Only Gemini 3+ models support thinking_level parameter
    if thinking_level and model_name.startswith("gemini-3"):
        kwargs["thinking_level"] = thinking_level
        logger.info(f"Applying thinking_level={thinking_level} to {model_name}")

    llm = ChatGoogleGenerativeAI(**kwargs)
    return llm

def create_quick_thinking_llm(
    temperature: float = 0.3,
    model: Optional[str] = None,
    timeout: int = None, # Allow override or use config default
    max_retries: int = None, # Allow override or use config default
    callbacks: Optional[List[BaseCallbackHandler]] = None
) -> BaseChatModel:
    """
    Create a quick thinking LLM.

    If QUICK_MODEL == DEEP_MODEL and the model supports thinking_level (Gemini 3+),
    automatically applies thinking_level="low" for faster responses.

    Returns:
        Configured ChatGoogleGenerativeAI instance
    """
    model_name = model or config.quick_think_llm
    # Use config defaults if not provided
    final_timeout = timeout if timeout is not None else config.api_timeout
    final_retries = max_retries if max_retries is not None else config.api_retry_attempts

    # Determine if we should apply thinking_level
    # Only apply if: (1) same model as deep, (2) model supports thinking_level
    thinking_level = None
    if config.quick_think_llm == config.deep_think_llm and model_name.startswith("gemini-3"):
        thinking_level = "low"
        logger.info(f"Quick LLM using same model as Deep LLM ({model_name}) - applying thinking_level=low")

    logger.info(f"Initializing Quick LLM: {model_name} (timeout={final_timeout}, retries={final_retries})")
    return create_gemini_model(
        model_name, temperature, final_timeout, final_retries,
        callbacks=callbacks, thinking_level=thinking_level
    )

def create_deep_thinking_llm(
    temperature: float = 0.1,
    model: Optional[str] = None,
    timeout: int = None, # Allow override or use config default
    max_retries: int = None, # Allow override or use config default
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    quick_mode: bool = False
) -> BaseChatModel:
    """
    Create a deep thinking LLM.

    If QUICK_MODEL == DEEP_MODEL and the model supports thinking_level (Gemini 3+),
    automatically applies thinking_level based on quick_mode:
    - quick_mode=False → thinking_level="high" (deep reasoning)
    - quick_mode=True → thinking_level="low" (faster, still same model)

    Args:
        temperature: Sampling temperature (default 0.1 for deterministic)
        model: Model override (defaults to config.deep_think_llm)
        timeout: Request timeout override
        max_retries: Retry attempts override
        callbacks: Optional callback handlers
        quick_mode: If True, use low thinking level for speed

    Returns:
        Configured ChatGoogleGenerativeAI instance
    """
    model_name = model or config.deep_think_llm
    # Use config defaults if not provided
    final_timeout = timeout if timeout is not None else config.api_timeout
    final_retries = max_retries if max_retries is not None else config.api_retry_attempts

    # Determine if we should apply thinking_level
    # Only apply if: (1) same model as quick, (2) model supports thinking_level
    thinking_level = None
    if config.quick_think_llm == config.deep_think_llm and model_name.startswith("gemini-3"):
        thinking_level = "low" if quick_mode else "high"
        logger.info(
            f"Deep LLM using same model as Quick LLM ({model_name}) - "
            f"applying thinking_level={thinking_level} (quick_mode={quick_mode})"
        )

    logger.info(f"Initializing Deep LLM: {model_name} (timeout={final_timeout}, retries={final_retries})")
    return create_gemini_model(
        model_name, temperature, final_timeout, final_retries,
        callbacks=callbacks, thinking_level=thinking_level
    )

# Initialize default instances
quick_thinking_llm = create_quick_thinking_llm()
deep_thinking_llm = create_deep_thinking_llm()


def create_consultant_llm(
    temperature: float = 0.3,
    model: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 3,
    quick_mode: bool = False,
    callbacks: Optional[List[BaseCallbackHandler]] = None
) -> BaseChatModel:
    """
    Create an OpenAI consultant LLM for cross-validation.

    Uses OpenAI (ChatGPT) instead of Gemini to provide independent perspective
    on Gemini's analysis outputs. This helps detect biases and groupthink.

    Args:
        temperature: Sampling temperature (default 0.3 for balanced creativity)
        model: Model name (overrides env vars if provided)
        timeout: Request timeout in seconds
        max_retries: Max retry attempts for failed requests
        quick_mode: If True, use CONSULTANT_QUICK_MODEL env var (default False)
        callbacks: Optional callback handlers for token tracking

    Returns:
        Configured ChatOpenAI instance

    Raises:
        ValueError: If OPENAI_API_KEY not found in environment
        ImportError: If langchain-openai package not installed

    Notes:
        - Requires OPENAI_API_KEY environment variable
        - Normal mode: Uses CONSULTANT_MODEL env var (defaults to gpt-4o)
        - Quick mode: Uses CONSULTANT_QUICK_MODEL env var (defaults to gpt-4o-mini)
        - Optional ENABLE_CONSULTANT env var (defaults to true)
        - gpt-4o is recommended as of Dec 2025 (GPT-4 Omni)
        - ChatGPT 5.2 not yet available via API as of Dec 2025

    Example:
        >>> consultant_llm = create_consultant_llm()
        >>> result = consultant_llm.invoke("Review this analysis...")
        >>> quick_llm = create_consultant_llm(quick_mode=True)
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai package not found. Install with: "
            "pip install langchain-openai>=0.3.0"
        )

    # Check if consultant is enabled
    enable_consultant = os.environ.get("ENABLE_CONSULTANT", "true").lower()
    if enable_consultant == "false":
        raise ValueError(
            "Consultant LLM is disabled. Set ENABLE_CONSULTANT=true to enable."
        )

    # Get OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "The consultant node requires an OpenAI API key for cross-validation. "
            "Add OPENAI_API_KEY to your .env file or set ENABLE_CONSULTANT=false."
        )

    # Get model name from env or use default
    # Note: As of Dec 2025, gpt-4o (GPT-4 Omni) is the latest production model
    # ChatGPT 5.2 is not yet available via API
    if model:
        # Explicit model override
        model_name = model
    elif quick_mode:
        # Quick mode: use faster/cheaper model (defaults to gpt-4o-mini)
        model_name = os.environ.get("CONSULTANT_QUICK_MODEL", "gpt-4o-mini")
    else:
        # Normal mode: use full model (defaults to gpt-4o)
        model_name = os.environ.get("CONSULTANT_MODEL", "gpt-4o")

    logger.info(
        f"Initializing Consultant LLM (OpenAI): {model_name} "
        f"(timeout={timeout}s, retries={max_retries})"
    )

    # Create ChatOpenAI instance with similar config to Gemini models
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        openai_api_key=api_key,
        callbacks=callbacks or [],
        # Match Gemini's max output for consistency
        max_tokens=4096,  # OpenAI default, sufficient for consultant reports
        # Enable streaming for better UX (optional)
        streaming=False
    )

    return llm


# Initialize consultant LLM (lazy initialization to handle missing API key gracefully)
_consultant_llm_instance = None


def get_consultant_llm(
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    quick_mode: bool = False
) -> Optional[BaseChatModel]:
    """
    Get or create the consultant LLM instance.

    Uses lazy initialization to gracefully handle missing OPENAI_API_KEY.
    If consultant is disabled or API key is missing, returns None.

    Args:
        callbacks: Optional callback handlers for token tracking
        quick_mode: If True, use CONSULTANT_QUICK_MODEL (gpt-4o-mini by default)

    Returns:
        ChatOpenAI instance or None if consultant disabled/unavailable

    Note:
        Caching is NOT affected by quick_mode - the instance is created once
        with the mode that was first requested. This matches Gemini behavior
        where models are configured at graph build time, not per-run.
    """
    global _consultant_llm_instance

    # Check if consultant is enabled
    enable_consultant = os.environ.get("ENABLE_CONSULTANT", "true").lower()
    if enable_consultant == "false":
        logger.info("Consultant LLM disabled via ENABLE_CONSULTANT=false")
        return None

    # Check if API key exists
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY not found - consultant node will be skipped. "
            "To enable consultant cross-validation, add OPENAI_API_KEY to .env"
        )
        return None

    # Lazy initialization
    if _consultant_llm_instance is None:
        try:
            _consultant_llm_instance = create_consultant_llm(
                callbacks=callbacks,
                quick_mode=quick_mode
            )
        except Exception as e:
            logger.error(f"Failed to initialize consultant LLM: {str(e)}")
            return None

    return _consultant_llm_instance