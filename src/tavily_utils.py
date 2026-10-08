"""
Tavily utility functions shared across modules.

This module exists to avoid circular imports between toolkit.py and enhanced_sentiment_toolkit.py.
"""

import asyncio
import time
from typing import Any, Literal

import structlog

from src.blocking_io import (
    BlockingCallPolicy,
    BlockingCallSaturatedError,
    BlockingCallTimeoutError,
    InFlightLimit,
    run_blocking_call_sync,
)
from src.config import config
from src.error_safety import summarize_exception
from src.runtime_services import get_current_inspection_service
from src.tooling.inspector import InspectionEnvelope, SourceKind

logger = structlog.get_logger(__name__)

# Tavily timeout configuration (seconds)
# Web searches can hang indefinitely; this prevents blocking the entire agent
TAVILY_TIMEOUT_SECONDS = 30
TavilyProfile = Literal["news_basic", "finance_deep"]

# Sync entrypoint only (search_tavily_sync_inspected). Its callers — the
# article writer's fact-check lookup and the portfolio manager's macro-news
# lookup — run one search at a time, so two slots leave headroom while capping
# how many stalled workers can be parked at once.
_TAVILY_SYNC_IN_FLIGHT = InFlightLimit(2)
_TAVILY_SYNC_POLICY = BlockingCallPolicy(
    "tavily.sync_search", TAVILY_TIMEOUT_SECONDS, in_flight=_TAVILY_SYNC_IN_FLIGHT
)

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
    from src.async_utils import run_with_hard_timeout

    try:
        raw = await run_with_hard_timeout(
            _tavily_tool.ainvoke(query),
            timeout=timeout,
            label=f"tavily:{query.get('query', '')[:60]}",
        )
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
            **summarize_exception(e, operation="tavily_search_error"),
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

    from src.async_utils import run_with_hard_timeout

    try:
        raw = await run_with_hard_timeout(
            tool.ainvoke({"query": query}),
            timeout=timeout,
            label=f"tavily:{profile}:{query[:60]}",
        )
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
            **summarize_exception(exc, operation="tavily_search_error"),
            profile=profile,
        )
        return None

    return await _inspect_tavily_result(raw, query)


async def extract_tavily_inspected(
    urls: list[str],
    *,
    query: str,
    timeout: float = TAVILY_TIMEOUT_SECONDS,
) -> Any:
    """Extract query-relevant chunks from a bounded list of discovered URLs."""
    api_key = config.get_tavily_api_key()
    if not api_key or not urls:
        return None
    try:
        from langchain_tavily import TavilyExtract
        from langchain_tavily._utilities import TavilyExtractAPIWrapper
        from pydantic import SecretStr
    except ImportError:
        logger.warning(
            "tavily_not_installed",
            hint="Run 'poetry add langchain-tavily' to enable Tavily extraction",
        )
        return None

    tool = TavilyExtract(
        apiwrapper=TavilyExtractAPIWrapper(tavily_api_key=SecretStr(api_key)),
        extract_depth="advanced",
        query=query,
        chunks_per_source=3,
        include_images=False,
        format="text",
        include_usage=False,
    )
    from src.async_utils import run_with_hard_timeout

    try:
        raw = await run_with_hard_timeout(
            tool.ainvoke({"urls": urls[:3]}),
            timeout=timeout,
            label=f"tavily:extract:{query[:60]}",
        )
    except asyncio.TimeoutError:
        logger.warning(
            "tavily_extract_timeout",
            url_count=len(urls[:3]),
            timeout_seconds=timeout,
        )
        return None
    except Exception as exc:
        logger.warning(
            "tavily_extract_error",
            url_count=len(urls[:3]),
            **summarize_exception(exc, operation="tavily_extract_error"),
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
    """Synchronous Tavily entrypoint for non-async call sites.

    *timeout* bounds the caller's whole wait — search and inspection together —
    because both run in one worker. The worker is a daemon thread drawn from
    ``_TAVILY_SYNC_POLICY``'s in-flight limit: a stalled search is orphaned
    (``langchain_tavily``'s sync client has no transport timeout to end it),
    but repeated stalls cannot accumulate past the limit, and once saturated
    further calls return ``None`` immediately rather than waiting out *timeout*.

    The worker checks the same absolute deadline: a search that completes
    after it is discarded before inspection, and inspection gets only the
    remaining allowance (an inspection timeout returns ``None``, never the
    uninspected payload).

    Search failures and timeouts return ``None``; an inspection failure
    propagates, as before.
    """
    tool = _build_tavily_tool(profile, include_answer=include_answer)
    if tool is None:
        return None

    # The worker shares the caller's absolute deadline. Without it, a search
    # that finishes after the caller gave up would still start inspection —
    # possibly a paid LLM call — for a result nobody will read, and hold its
    # slot for as long as that takes.
    deadline = time.monotonic() + timeout

    def _search_then_inspect() -> Any:
        try:
            raw = tool.invoke({"query": query})
        except Exception as exc:
            logger.warning(
                "tavily_search_error",
                query=query[:100],
                **summarize_exception(exc, operation="tavily_search_error"),
                profile=profile,
            )
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "tavily_search_late_result_discarded",
                query=query[:100],
                timeout_seconds=timeout,
                profile=profile,
            )
            return None
        # Own event loop: this runs on a worker thread, so it is safe even
        # when the sync caller sits inside a running loop. wait_for is safe
        # here: inspection is pure async code that honours cancellation.
        try:
            return asyncio.run(
                asyncio.wait_for(_inspect_tavily_result(raw, query), remaining)
            )
        except TimeoutError:
            # Fail closed: an uninspected payload is never returned.
            logger.warning(
                "tavily_inspection_timeout",
                query=query[:100],
                timeout_seconds=timeout,
                profile=profile,
            )
            return None

    policy = _TAVILY_SYNC_POLICY.with_timeout(timeout).with_label(
        f"{_TAVILY_SYNC_POLICY.label}:{profile}:{query[:60]}"
    )
    try:
        return run_blocking_call_sync(policy, _search_then_inspect)
    except BlockingCallSaturatedError:
        logger.warning(
            "tavily_search_saturated",
            query=query[:100],
            max_in_flight=_TAVILY_SYNC_IN_FLIGHT.max_in_flight,
            profile=profile,
        )
        return None
    except BlockingCallTimeoutError:
        logger.warning(
            "tavily_search_timeout",
            query=query[:100],
            timeout_seconds=timeout,
            profile=profile,
        )
        return None
