"""
LLM configuration and initialization module.
Updated for Google Gemini 3 with Safety Settings and Rate Limiting.
Includes token tracking for cost monitoring.
UPDATED: Configurable rate limits via GEMINI_RPM_LIMIT environment variable.
UPDATED: Added OpenAI consultant LLM for cross-validation (Dec 2025).
"""

import logging
import os
import re
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

def _is_gemini_v3_or_greater(model_name: str) -> bool:
    """
    Checks if a Gemini model name is version 3.0 or greater.
    This is used to determine if the 'thinking_level' parameter is supported.
    """
    if not model_name.startswith("gemini-"):
        return False
    
    match = re.search(r"gemini-([0-9.]+)", model_name)
    if not match:
        return False
    
    version_str = match.group(1)
    try:
        major_version = int(version_str.split('.')[0])
        return major_version >= 3
    except (ValueError, IndexError):
        return False

# ... (rest of the file is the same until create_gemini_model)

def _create_rate_limiter_from_rpm(rpm: int) -> InMemoryRateLimiter:
    """
    Create a rate limiter from RPM (requests per minute) setting.
    """
    safety_factor = 0.8
    rps = (rpm / 60.0) * safety_factor
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
    Generic factory for Gemini models.
    """
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

    if thinking_level and _is_gemini_v3_or_greater(model_name):
        kwargs["thinking_level"] = thinking_level
        logger.info(f"Applying thinking_level={thinking_level} to {model_name}")

    return ChatGoogleGenerativeAI(**kwargs)

def create_quick_thinking_llm(
    temperature: float = 0.3,
    model: Optional[str] = None,
    timeout: int = None,
    max_retries: int = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None
) -> BaseChatModel:
    """
    Create a quick thinking LLM.
    If the QUICK_MODEL is Gemini 3+, this will set thinking_level="low".
    """
    model_name = model or config.quick_think_llm
    final_timeout = timeout if timeout is not None else config.api_timeout
    final_retries = max_retries if max_retries is not None else config.api_retry_attempts

    thinking_level = None
    if _is_gemini_v3_or_greater(model_name):
        thinking_level = "low"
        logger.info(
            f"Quick LLM ({model_name}) is Gemini 3+ - applying thinking_level=low"
        )

    logger.info(f"Initializing Quick LLM: {model_name} (timeout={final_timeout}, retries={final_retries})")
    return create_gemini_model(
        model_name, temperature, final_timeout, final_retries,
        callbacks=callbacks, thinking_level=thinking_level
    )

def create_deep_thinking_llm(
    temperature: float = 0.1,
    model: Optional[str] = None,
    timeout: int = None,
    max_retries: int = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None
) -> BaseChatModel:
    """
    Create a deep thinking LLM.
    If the DEEP_MODEL is Gemini 3+, this will set thinking_level="high".
    """
    model_name = model or config.deep_think_llm
    final_timeout = timeout if timeout is not None else config.api_timeout
    final_retries = max_retries if max_retries is not None else config.api_retry_attempts

    thinking_level = None
    if _is_gemini_v3_or_greater(model_name):
        thinking_level = "high"
        logger.info(
            f"Deep LLM ({model_name}) is Gemini 3+ - applying thinking_level=high"
        )

    logger.info(f"Initializing Deep LLM: {model_name} (timeout={final_timeout}, retries={final_retries})")
    return create_gemini_model(
        model_name, temperature, final_timeout, final_retries,
        callbacks=callbacks, thinking_level=thinking_level
    )

# Initialize default instances
quick_thinking_llm = create_quick_thinking_llm()
deep_thinking_llm = create_deep_thinking_llm()

# ... (rest of the file is the same)
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
