"""
Tavily utility functions shared across modules.

This module exists to avoid circular imports between toolkit.py and enhanced_sentiment_toolkit.py.
"""

import asyncio
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Tavily timeout configuration (seconds)
# Web searches can hang indefinitely; this prevents blocking the entire agent
TAVILY_TIMEOUT_SECONDS = 30

# Will be set by toolkit.py after tavily_tool is initialized
_tavily_tool = None


def set_tavily_tool(tool: Any) -> None:
    """Set the tavily tool instance (called by toolkit.py during initialization)."""
    global _tavily_tool
    _tavily_tool = tool


async def tavily_search_with_timeout(
    query: dict[str, str], timeout: float = TAVILY_TIMEOUT_SECONDS
) -> Any:
    """
    Execute Tavily search with timeout protection.

    Args:
        query: Query dict for tavily_tool.ainvoke (e.g., {"query": "search terms"})
        timeout: Maximum seconds to wait (default: TAVILY_TIMEOUT_SECONDS)

    Returns:
        Tavily search results, or None if timeout/error occurs
    """
    if not _tavily_tool:
        return None
    try:
        return await asyncio.wait_for(_tavily_tool.ainvoke(query), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "tavily_search_timeout",
            query=query.get("query", "")[:100],
            timeout_seconds=timeout,
        )
        return None
    except Exception as e:
        logger.warning(
            "tavily_search_error",
            query=query.get("query", "")[:100],
            error=str(e),
        )
        return None
