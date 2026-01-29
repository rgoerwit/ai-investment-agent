"""
Observability module for LLM tracing.

Provides unified callback handlers for tracing LangGraph executions.
Currently supports Langfuse (open-source) with graceful degradation.

Updated for Langfuse Python SDK v3 (Jan 2026):
- CallbackHandler() takes no constructor arguments
- Session ID, tags, and user ID are passed via LangChain config metadata
  using langfuse_session_id, langfuse_tags, langfuse_user_id keys
- SDK reads credentials from LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
  LANGFUSE_BASE_URL environment variables (exported by config.py)
- Flush via get_client().flush()

Usage:
    from src.observability import get_tracing_callbacks

    callbacks, trace_metadata = get_tracing_callbacks(
        ticker="0005.HK",
        session_id="0005.HK-2026-01-28-a3f7b2c1",
        tags=["quick", "deep-model:gemini-3-pro-preview"],
    )
    result = await graph.ainvoke(
        state,
        config={"callbacks": callbacks, "metadata": trace_metadata},
    )

Note:
    Callbacks are injected at graph.invoke() level (not LLM level) to produce
    unified traces showing the full multi-agent execution flow.
"""

from typing import Any

import structlog
from langchain_core.callbacks import BaseCallbackHandler

from src.config import config

logger = structlog.get_logger(__name__)


def get_tracing_callbacks(
    ticker: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
) -> tuple[list[BaseCallbackHandler], dict[str, Any]]:
    """
    Get configured tracing callbacks and metadata for graph invocation.

    Returns a tuple of (callbacks, metadata) based on enabled observability
    providers. Pass callbacks to config["callbacks"] and metadata to
    config["metadata"] in graph.ainvoke().

    Langfuse SDK v3 notes:
    - CallbackHandler() reads credentials from environment variables
      (exported by config.py's setup_environment)
    - Trace attributes (session_id, tags, user_id) are passed via
      LangChain's config metadata with langfuse_* prefixed keys

    Args:
        ticker: Stock ticker being analyzed (used as trace metadata)
        session_id: Optional session ID for grouping traces
        user_id: Optional user ID for filtering
        tags: Optional tags for categorization

    Returns:
        Tuple of (callback handlers list, metadata dict).
        Both are empty if no observability is enabled.
    """
    callbacks: list[BaseCallbackHandler] = []
    metadata: dict[str, Any] = {}

    if not config.langfuse_enabled:
        return callbacks, metadata

    # Check for required keys (exported to os.environ by config.py)
    public_key = config.get_langfuse_public_key()
    secret_key = config.get_langfuse_secret_key()

    if not public_key or not secret_key:
        logger.warning(
            "langfuse_missing_keys",
            hint="Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env",
        )
        return callbacks, metadata

    try:
        # Import here to allow graceful degradation if langfuse not installed
        from langfuse.langchain import CallbackHandler as LangfuseHandler

        # SDK v3: no constructor args. Reads LANGFUSE_PUBLIC_KEY,
        # LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL from os.environ
        # (exported by config.py setup_environment).
        handler = LangfuseHandler()
        callbacks.append(handler)

        # SDK v3: trace attributes go in LangChain config metadata
        # with langfuse_* prefixed keys
        if session_id:
            metadata["langfuse_session_id"] = session_id
        if user_id:
            metadata["langfuse_user_id"] = user_id
        if tags:
            metadata["langfuse_tags"] = tags
        if ticker:
            metadata["langfuse_metadata"] = {"ticker": ticker}

        logger.info(
            "langfuse_callback_created",
            base_url=config.langfuse_host,
            session_id=session_id,
            ticker=ticker,
        )

    except ImportError:
        logger.warning(
            "langfuse_not_installed",
            hint="Run 'poetry add langfuse' to enable Langfuse tracing",
        )
    except Exception as e:
        # Don't let observability failures break the analysis
        logger.error(
            "langfuse_init_failed",
            error=str(e),
            hint="Check LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL",
        )

    return callbacks, metadata


def flush_traces() -> None:
    """
    Flush any pending traces to the observability backend.

    Call this at the end of an analysis to ensure all traces are sent
    before the process exits. Uses the SDK v3 singleton client.
    """
    if not config.langfuse_enabled:
        return

    try:
        from langfuse import get_client

        get_client().flush()
        logger.debug("langfuse_traces_flushed")
    except Exception as e:
        logger.warning("langfuse_flush_failed", error=str(e))
