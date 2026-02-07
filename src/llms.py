"""
LLM configuration and initialization module.
Updated for Google Gemini 3 with Safety Settings and Rate Limiting.
Includes token tracking for cost monitoring.
UPDATED: Configurable rate limits via GEMINI_RPM_LIMIT environment variable.
UPDATED: Added OpenAI consultant LLM for cross-validation (Dec 2025).
"""

import logging
import re

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

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
    Checks if a Gemini model supports 'thinking_level' parameter.

    Includes:
    - Gemini 3.0+ models (e.g., gemini-3-pro-preview)
    - Any model with 'thinking' in the name (e.g., gemini-2.0-flash-thinking-exp)
    """
    if not model_name.startswith("gemini-"):
        return False

    # Explicit support for "thinking" models regardless of version number
    if "thinking" in model_name.lower():
        return True

    match = re.search(r"gemini-([0-9.]+)", model_name)
    if not match:
        return False

    version_str = match.group(1)
    try:
        major_version = int(version_str.split(".")[0])
        return major_version >= 3
    except (ValueError, IndexError):
        return False


def is_gemini_v3_or_greater(model_name: str) -> bool:
    """
    Public wrapper to check if a Gemini model is version 3.0 or greater.

    Used by agents to determine if retry with high thinking_level is beneficial.
    Only Gemini 3+ models support the thinking_level parameter.

    Args:
        model_name: The model name string (e.g., "gemini-3-pro-preview")

    Returns:
        True if model is Gemini 3.0 or greater, False otherwise
    """
    return _is_gemini_v3_or_greater(model_name)


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
        requests_per_second=rps, check_every_n_seconds=0.1, max_bucket_size=max_bucket
    )


GLOBAL_RATE_LIMITER = _create_rate_limiter_from_rpm(config.gemini_rpm_limit)

# Track LLM instances for cleanup
_llm_instances: dict = {}
_llm_instance_counter: int = 0


def get_all_llm_instances() -> dict:
    """
    Get all tracked LLM instances for cleanup.

    Returns:
        Dict mapping instance names to LLM objects
    """
    return _llm_instances.copy()


def create_gemini_model(
    model_name: str,
    temperature: float,
    timeout: int,
    max_retries: int,
    streaming: bool = False,
    callbacks: list[BaseCallbackHandler] | None = None,
    thinking_level: str | None = None,
) -> BaseChatModel:
    """
    Generic factory for Gemini models.
    All created instances are tracked for proper cleanup at shutdown.

    Note: API key is explicitly passed from config to avoid dependency on
    os.environ being populated by load_dotenv() (Pydantic Settings handles
    .env loading for our config, but third-party libs like LangChain still
    expect explicit api_key or os.environ values).
    """
    global _llm_instance_counter

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
        "callbacks": callbacks or [],
        "api_key": config.get_google_api_key(),  # Explicit API key from config
    }

    if thinking_level and _is_gemini_v3_or_greater(model_name):
        kwargs["thinking_level"] = thinking_level
        logger.info(f"Applying thinking_level={thinking_level} to {model_name}")

    llm = ChatGoogleGenerativeAI(**kwargs)

    # Track instance for cleanup
    _llm_instance_counter += 1
    instance_name = f"gemini_{model_name}_{_llm_instance_counter}"
    _llm_instances[instance_name] = llm

    return llm


def create_quick_thinking_llm(
    temperature: float = 0.3,
    model: str | None = None,
    timeout: int = None,
    max_retries: int = None,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> BaseChatModel:
    """
    Create a quick thinking LLM.
    If the QUICK_MODEL is Gemini 3+, this will set thinking_level="low".
    """
    model_name = model or config.quick_think_llm
    final_timeout = timeout if timeout is not None else config.api_timeout
    final_retries = (
        max_retries if max_retries is not None else config.api_retry_attempts
    )

    thinking_level = None
    if _is_gemini_v3_or_greater(model_name):
        thinking_level = "low"
        logger.info(
            f"Quick LLM ({model_name}) is Gemini 3+ - applying thinking_level=low"
        )
    elif model_name.startswith("gemini-"):
        # Gemini model but NOT 3+ (likely 2.x)
        logger.warning(
            f"QUICK_MODEL is {model_name} (Gemini 2.x) - tool calling bugs "
            f"may occur with some LangGraph versions. Recommend using "
            f"Gemini 3+ (e.g., gemini-3-pro-preview) for QUICK_MODEL. "
            f"Gemini 3+ models use thinking_level='low' for data gathering."
        )

    logger.info(
        f"Initializing Quick LLM: {model_name} "
        f"(timeout={final_timeout}, retries={final_retries})"
    )
    return create_gemini_model(
        model_name,
        temperature,
        final_timeout,
        final_retries,
        callbacks=callbacks,
        thinking_level=thinking_level,
    )


def create_deep_thinking_llm(
    temperature: float = 0.1,
    model: str | None = None,
    timeout: int = None,
    max_retries: int = None,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> BaseChatModel:
    """
    Create a deep thinking LLM.
    If the DEEP_MODEL is Gemini 3+, this will set thinking_level="high".
    """
    model_name = model or config.deep_think_llm
    final_timeout = timeout if timeout is not None else config.api_timeout
    final_retries = (
        max_retries if max_retries is not None else config.api_retry_attempts
    )

    thinking_level = None
    if _is_gemini_v3_or_greater(model_name):
        thinking_level = "high"
        logger.info(
            f"Deep LLM ({model_name}) is Gemini 3+ - applying thinking_level=high"
        )

    logger.info(
        f"Initializing Deep LLM: {model_name} "
        f"(timeout={final_timeout}, retries={final_retries})"
    )
    return create_gemini_model(
        model_name,
        temperature,
        final_timeout,
        final_retries,
        callbacks=callbacks,
        thinking_level=thinking_level,
    )


# Initialize default instances (automatically tracked by create_gemini_model)
quick_thinking_llm = create_quick_thinking_llm()
deep_thinking_llm = create_deep_thinking_llm()


# ... (rest of the file is the same)
def create_consultant_llm(
    temperature: float = 0.3,
    model: str | None = None,
    timeout: int = 120,
    max_retries: int = 3,
    quick_mode: bool = False,
    callbacks: list[BaseCallbackHandler] | None = None,
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
    except ImportError as e:
        raise ImportError(
            "langchain-openai package not found. Install with: "
            "pip install langchain-openai>=0.3.0"
        ) from e

    # Check if consultant is enabled (via config, not os.environ)
    if not config.enable_consultant:
        raise ValueError(
            "Consultant LLM is disabled. Set ENABLE_CONSULTANT=true to enable."
        )

    # Get OpenAI API key via config (SecretStr protected)
    api_key = config.get_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "The consultant node requires an OpenAI API key for cross-validation. "
            "Add OPENAI_API_KEY to your .env file or set ENABLE_CONSULTANT=false."
        )

    # Get model name from config (not os.environ)
    # Note: As of Dec 2025, gpt-4o (GPT-4 Omni) is the latest production model
    # ChatGPT 5.2 is not yet available via API
    if model:
        # Explicit model override
        model_name = model
    elif quick_mode:
        # Quick mode: use faster/cheaper model (defaults to gpt-4o-mini)
        model_name = config.consultant_quick_model
    else:
        # Normal mode: use full model (defaults to gpt-4o)
        model_name = config.consultant_model

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
        # Increased from 4096 - consultant reviews need room for tiered flags + forensic analysis
        max_tokens=16384,
        # Enable streaming for better UX (optional)
        streaming=False,
    )

    return llm


def create_auditor_llm(
    callbacks: list[BaseCallbackHandler] | None = None,
) -> BaseChatModel | None:
    """
    Create Auditor LLM with fallback logic.
    Returns None if ENABLE_CONSULTANT is false.

    Logic:
    1. If ENABLE_CONSULTANT is False -> None
    2. If AUDITOR_MODEL is set -> Use it
    3. If CONSULTANT_MODEL is set -> Use it (Fallback)
    4. Default -> gpt-4o
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        logger.warning(f"langchain-openai not found: {e}")
        return None

    if not config.enable_consultant:
        return None

    # Get OpenAI API key via config
    api_key = config.get_openai_api_key()
    if not api_key:
        logger.warning("OPENAI_API_KEY not found - Auditor will be disabled.")
        return None

    # Determine model: Specific -> Consultant -> Default
    model_name = config.auditor_model or config.consultant_model or "gpt-4o"

    logger.info(f"Initializing Auditor LLM: {model_name}")

    return ChatOpenAI(
        model=model_name,
        temperature=0,  # Forensic work requires precision
        timeout=120,
        max_retries=3,
        openai_api_key=api_key,
        callbacks=callbacks or [],
        max_tokens=16384,  # Increased from 4096 - auditor needs room for forensic analysis
        streaming=False,
    )


def create_writer_llm(
    temperature: float = 0.7,
    timeout: int | None = None,
    max_retries: int = 3,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> BaseChatModel:
    """
    Create the LLM for article writing.

    Prefers Claude (Anthropic) when CLAUDE_KEY is configured.
    Falls back gracefully to Gemini deep thinking LLM when not.

    Args:
        temperature: Sampling temperature. NOTE: Overridden to 1.0
                     when Claude adaptive thinking is active (API constraint).
        timeout: Request timeout in seconds (default from config)
        max_retries: Max retry attempts
        callbacks: Optional callback handlers

    Returns:
        ChatAnthropic or ChatGoogleGenerativeAI instance
    """
    api_key = config.get_claude_api_key()

    if not api_key:
        logger.warning(
            "CLAUDE_KEY not found — falling back to Gemini for article writer. "
            "To use Claude, add CLAUDE_KEY=sk-ant-... to your .env file."
        )
        return create_deep_thinking_llm(
            temperature=temperature,
            callbacks=callbacks,
        )

    # --- Claude path ---
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        logger.warning(
            "langchain-anthropic not installed — falling back to Gemini. "
            "Install with: poetry add langchain-anthropic"
        )
        return create_deep_thinking_llm(
            temperature=temperature,
            callbacks=callbacks,
        )

    model_name = config.writer_model
    final_timeout = float(timeout if timeout is not None else config.api_timeout)

    # Build kwargs — base configuration
    kwargs: dict = {
        "model": model_name,
        "max_tokens": 16384,
        "max_retries": max_retries,
        "timeout": final_timeout,
        "callbacks": callbacks or [],
        "anthropic_api_key": api_key,
    }

    # Thinking configuration — model-dependent
    if "opus-4-6" in model_name:
        # Opus 4.6: adaptive thinking (Claude decides when/how much to think)
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["model_kwargs"] = {"effort": "high"}
        # CRITICAL: Anthropic returns 400 if temperature != 1.0 with thinking.
        # Omit temperature entirely — SDK defaults to 1.0.
        logger.info(
            f"Writer LLM ({model_name}): adaptive thinking, effort=high, "
            f"temperature=1.0 (Anthropic constraint, requested {temperature} ignored)"
        )
    elif "sonnet" in model_name or "opus" in model_name:
        # Other Claude 4.x models: manual extended thinking
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8192}
        # Same temperature constraint applies
        logger.info(f"Writer LLM ({model_name}): extended thinking, budget=8192 tokens")
    else:
        # Haiku or unknown models: no thinking, use requested temperature
        kwargs["temperature"] = temperature
        logger.info(
            f"Writer LLM ({model_name}): no thinking, temperature={temperature}"
        )

    llm = ChatAnthropic(**kwargs)

    # Track instance for cleanup (consistent with Gemini tracking)
    global _llm_instance_counter
    _llm_instance_counter += 1
    instance_name = f"claude_{model_name}_{_llm_instance_counter}"
    _llm_instances[instance_name] = llm

    logger.info(
        f"Initialized Writer LLM (Claude): {model_name} "
        f"(timeout={final_timeout}s, max_tokens=16384)"
    )

    return llm


def create_editor_llm(
    callbacks: list[BaseCallbackHandler] | None = None,
) -> BaseChatModel | None:
    """
    Create Editor-in-Chief LLM for article revision and fact-checking.

    Returns None if ENABLE_CONSULTANT is false or OPENAI_API_KEY missing.

    Fallback chain: EDITOR_MODEL -> CONSULTANT_MODEL -> "gpt-4o"

    Args:
        callbacks: Optional callback handlers for token tracking

    Returns:
        ChatOpenAI instance or None if editor unavailable
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        logger.warning(f"langchain-openai not found: {e}")
        return None

    if not config.enable_consultant:
        logger.info("Editor LLM disabled (ENABLE_CONSULTANT=false)")
        return None

    api_key = config.get_openai_api_key()
    if not api_key:
        logger.warning("OPENAI_API_KEY not found - Editor will be disabled.")
        return None

    # Fallback chain: EDITOR_MODEL -> CONSULTANT_MODEL -> gpt-4o
    model_name = config.editor_model or config.consultant_model or "gpt-4o"

    logger.info(f"Initializing Editor LLM: {model_name}")

    return ChatOpenAI(
        model=model_name,
        temperature=0.3,  # Slightly creative for style suggestions
        timeout=120,
        max_retries=3,
        openai_api_key=api_key,
        callbacks=callbacks or [],
        max_tokens=8192,  # Editor feedback is concise JSON
        streaming=False,
    )


# Initialize consultant LLM (lazy initialization to handle missing API key gracefully)
_consultant_llm_instance = None


def get_consultant_llm(
    callbacks: list[BaseCallbackHandler] | None = None, quick_mode: bool = False
) -> BaseChatModel | None:
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

    # Skip consultant in quick mode for performance
    if quick_mode:
        logger.info("Consultant LLM skipped in quick mode for performance")
        return None

    # Check if consultant is enabled (via config, not os.environ)
    if not config.enable_consultant:
        logger.info("Consultant LLM disabled via ENABLE_CONSULTANT=false")
        return None

    # Check if API key exists (via config with SecretStr protection)
    if not config.get_openai_api_key():
        logger.warning(
            "OPENAI_API_KEY not found - consultant node will be skipped. "
            "To enable consultant cross-validation, add OPENAI_API_KEY to .env"
        )
        return None

    # Lazy initialization
    if _consultant_llm_instance is None:
        try:
            _consultant_llm_instance = create_consultant_llm(
                callbacks=callbacks, quick_mode=quick_mode
            )
        except Exception as e:
            logger.error(f"Failed to initialize consultant LLM: {str(e)}")
            return None

    return _consultant_llm_instance
