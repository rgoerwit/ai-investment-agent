"""
Tavily utility functions shared across modules.

This module exists to avoid circular imports between toolkit.py and enhanced_sentiment_toolkit.py.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Literal

import structlog

from src.config import config
from src.runtime_services import get_current_inspection_service
from src.tooling.inspector import InspectionEnvelope, SourceKind

logger = structlog.get_logger(__name__)

# Tavily timeout configuration (seconds)
# Web searches can hang indefinitely; this prevents blocking the entire agent
TAVILY_TIMEOUT_SECONDS = 30
TavilyProfile = Literal["news_basic", "finance_deep"]

# Will be set by toolkit.py after tavily_tool is initialized
_tavily_tool = None
_profile_tools: dict[tuple[str, bool], Any] = {}


def set_tavily_tool(tool: Any) -> None:
    """Set the tavily tool instance (called by toolkit.py during initialization)."""
    global _tavily_tool
    _tavily_tool = tool


def _build_tavily_tool(
    profile: TavilyProfile, *, include_answer: bool = False
) -> Any | None:
    """Build and cache a Tavily tool with a constrained profile."""
    cache_key = (profile, include_answer)
    if cache_key in _profile_tools:
        return _profile_tools[cache_key]

    api_key = config.get_tavily_api_key()
    if not api_key:
        return None

    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        logger.warning(
            "tavily_not_installed",
            hint="Run 'poetry add langchain-tavily' to enable Tavily search",
        )
        return None

    kwargs: dict[str, Any] = {
        "tavily_api_key": api_key,
        "max_results": 3,
        "include_raw_content": False,
        "include_answer": include_answer,
    }
    if profile == "finance_deep":
        kwargs.update({"topic": "finance", "search_depth": "advanced"})
    else:
        kwargs.update({"topic": "news", "search_depth": "basic"})

    tool = TavilySearch(**kwargs)
    _profile_tools[cache_key] = tool
    return tool


async def _inspect_tavily_result(raw: Any, query_text: str) -> Any:
    if raw is None:
        return None

    content_text = raw if isinstance(raw, str) else str(raw)
    envelope = InspectionEnvelope(
        content_text=content_text,
        raw_content=raw,
        source_kind=SourceKind.web_search,
        source_name="tavily",
        metadata={"query": query_text[:200]},
    )
    return await get_current_inspection_service().check(envelope)


async def tavily_search_with_timeout(
    query: dict[str, str], timeout: float = TAVILY_TIMEOUT_SECONDS
) -> Any:
    """
    Execute Tavily search with timeout protection.

    All results pass through INSPECTION_SERVICE before being returned to callers.

    Args:
        query: Query dict for tavily_tool.ainvoke (e.g., {"query": "search terms"})
        timeout: Maximum seconds to wait (default: TAVILY_TIMEOUT_SECONDS)

    Returns:
        Tavily search results (inspected), or None if timeout/error occurs
    """
    if not _tavily_tool:
        return None
    try:
        raw = await asyncio.wait_for(_tavily_tool.ainvoke(query), timeout=timeout)
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

    # Preserve the original payload shape on allow/fail-open paths so callers
    # that merge structured Tavily results do not regress.
    return await _inspect_tavily_result(raw, query.get("query", ""))


async def search_tavily_inspected(
    query: str,
    *,
    profile: TavilyProfile,
    timeout: float = TAVILY_TIMEOUT_SECONDS,
    include_answer: bool = False,
) -> Any:
    """Run a profile-scoped Tavily search behind inspection boundaries."""
    tool = _build_tavily_tool(profile, include_answer=include_answer)
    if tool is None:
        return None

    try:
        raw = await asyncio.wait_for(tool.ainvoke({"query": query}), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "tavily_search_timeout",
            query=query[:100],
            timeout_seconds=timeout,
            profile=profile,
        )
        return None
    except Exception as exc:
        logger.warning(
            "tavily_search_error",
            query=query[:100],
            error=str(exc),
            profile=profile,
        )
        return None

    return await _inspect_tavily_result(raw, query)


def search_tavily_sync_inspected(
    query: str,
    *,
    profile: TavilyProfile,
    timeout: float = TAVILY_TIMEOUT_SECONDS,
    include_answer: bool = False,
) -> Any:
    """Synchronous Tavily entrypoint for non-async call sites."""
    tool = _build_tavily_tool(profile, include_answer=include_answer)
    if tool is None:
        return None

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tool.invoke, {"query": query})
        try:
            raw = future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.warning(
                "tavily_search_timeout",
                query=query[:100],
                timeout_seconds=timeout,
                profile=profile,
            )
            return None
        except Exception as exc:
            logger.warning(
                "tavily_search_error",
                query=query[:100],
                error=str(exc),
                profile=profile,
            )
            return None

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, _inspect_tavily_result(raw, query)).result()
