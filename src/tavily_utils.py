"""
Tavily utility functions shared across modules.

This module exists to avoid circular imports between toolkit.py and enhanced_sentiment_toolkit.py.
"""

import asyncio
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
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import require_allowance
from src.runtime_services import get_current_inspection_service
from src.tooling.inspector import InspectionEnvelope, SourceKind

logger = structlog.get_logger(__name__)

# Web searches can hang indefinitely, so every entrypoint is bounded. The bound
# is a setting read at the call (TAVILY_SEARCH_TIMEOUT_SECONDS, or
# TAVILY_EXTRACT_TIMEOUT_SECONDS for extraction) unless the caller passes its
# own, and either way it shrinks to a caller's enclosing deadline.
TavilyProfile = Literal["news_basic", "finance_deep"]

# Sync entrypoint only (search_tavily_sync_inspected). Its callers — the
# article writer's fact-check lookup and the portfolio manager's macro-news
# lookup — run one search at a time, so two slots leave headroom while capping
# how many stalled workers can be parked at once. The 30 s here is only the
# shipped default of tavily_search_timeout_seconds (pinned by a parity test);
# each call applies the resolved bound with with_timeout().
_TAVILY_SYNC_IN_FLIGHT = InFlightLimit(2)
_TAVILY_SYNC_POLICY = BlockingCallPolicy(
    "tavily.sync_search", 30.0, in_flight=_TAVILY_SYNC_IN_FLIGHT
)


def _search_timeout(timeout: float | None) -> float:
    """The caller's own bound when it passes one, else the setting, read now.

    Each entrypoint then shrinks it with ``require_allowance`` *before* the
    request is created; a spent caller deadline raises TimeoutError there,
    which the entrypoint handles exactly like its own timeout.
    """
    return config.tavily_search_timeout_seconds if timeout is None else timeout


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


async def _inspect_before_deadline(
    raw: Any, query_text: str, *, deadline: float, log_fields: dict[str, Any]
) -> Any:
    """Inspect *raw* within what is left of *deadline*, or return ``None``.

    The entrypoint's bound covers search *and* inspection, on both the async
    and the sync path. A result that arrives with nothing left is discarded
    before inspection starts (which may be a paid LLM judge call), and an
    inspection that overruns returns ``None``, never the uninspected payload
    (``InspectionService.check_within``: deadline-only, so a judge that
    ignores cancellation cannot hold the caller). An enforced
    ``inspection_failed`` (a fail-closed backend error) returns ``None`` too:
    one meaning for "could not inspect".
    """
    if raw is None:
        return None
    remaining = deadline - llm_timeouts.now()
    if remaining <= 0:
        logger.warning("tavily_search_late_result_discarded", **log_fields)
        return None
    outcome = await get_current_inspection_service().check_within(
        InspectionEnvelope(
            content_text=raw if isinstance(raw, str) else str(raw),
            raw_content=raw,
            source_kind=SourceKind.web_search,
            source_name="tavily",
            metadata={"query": query_text[:200]},
        ),
        seconds=remaining,
        label=f"tavily:inspect:{query_text[:60]}",
    )
    if outcome is None:
        logger.warning("tavily_inspection_timeout", **log_fields)
        return None
    decision, inspected = outcome
    if decision.enforced == "inspection_failed":
        # Could not inspect: the same fail-closed result as a timeout.
        logger.warning("tavily_inspection_failed", **log_fields)
        return None
    return inspected


async def tavily_search_with_timeout(
    query: dict[str, str], timeout: float | None = None
) -> Any:
    """
    Execute Tavily search with timeout protection.

    All results pass through INSPECTION_SERVICE before being returned to callers.

    Args:
        query: Query dict for tavily_tool.ainvoke (e.g., {"query": "search terms"})
        timeout: Maximum seconds to wait for search and inspection together
            (default: TAVILY_SEARCH_TIMEOUT_SECONDS), shrunk to a caller's
            enclosing deadline

    Returns:
        Tavily search results (inspected), or None if timeout/error occurs
    """
    if not _tavily_tool:
        return None
    from src.async_utils import run_with_hard_timeout

    timeout = _search_timeout(timeout)
    try:
        # Resolved before ainvoke() creates the request coroutine, so a spent
        # caller deadline leaves nothing un-awaited.
        timeout = require_allowance(timeout, label="tavily.search")
        # One deadline for search and inspection, as on the sync path.
        deadline = llm_timeouts.now() + timeout
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
    return await _inspect_before_deadline(
        raw,
        query.get("query", ""),
        deadline=deadline,
        log_fields={
            "query": query.get("query", "")[:100],
            "timeout_seconds": timeout,
        },
    )


async def search_tavily_inspected(
    query: str,
    *,
    profile: TavilyProfile,
    timeout: float | None = None,
    include_answer: bool = False,
) -> Any:
    """Run a profile-scoped Tavily search behind inspection boundaries.

    *timeout* defaults to ``TAVILY_SEARCH_TIMEOUT_SECONDS`` and is shrunk to a
    caller's enclosing deadline. It covers search and inspection together.
    """
    tool = _build_tavily_tool(profile, include_answer=include_answer)
    if tool is None:
        return None

    from src.async_utils import run_with_hard_timeout

    timeout = _search_timeout(timeout)
    try:
        timeout = require_allowance(timeout, label=f"tavily.search:{profile}")
        deadline = llm_timeouts.now() + timeout
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

    return await _inspect_before_deadline(
        raw,
        query,
        deadline=deadline,
        log_fields={
            "query": query[:100],
            "timeout_seconds": timeout,
            "profile": profile,
        },
    )


async def extract_tavily_inspected(
    urls: list[str],
    *,
    query: str,
    timeout: float | None = None,
) -> Any:
    """Extract query-relevant chunks from a bounded list of discovered URLs.

    *timeout* defaults to ``TAVILY_EXTRACT_TIMEOUT_SECONDS`` and is shrunk to a
    caller's enclosing deadline. It covers extraction and inspection together.
    """
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

    if timeout is None:
        timeout = config.tavily_extract_timeout_seconds
    try:
        timeout = require_allowance(timeout, label="tavily.extract")
        deadline = llm_timeouts.now() + timeout
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

    return await _inspect_before_deadline(
        raw,
        query,
        deadline=deadline,
        log_fields={
            "query": query[:100],
            "url_count": len(urls[:3]),
            "timeout_seconds": timeout,
        },
    )


def search_tavily_sync_inspected(
    query: str,
    *,
    profile: TavilyProfile,
    timeout: float | None = None,
    include_answer: bool = False,
) -> Any:
    """Synchronous Tavily entrypoint for non-async call sites.

    *timeout* (default ``TAVILY_SEARCH_TIMEOUT_SECONDS``, shrunk to a caller's
    enclosing deadline) bounds the caller's whole wait — search and inspection
    together — because both run in one worker. A caller deadline already spent
    returns ``None`` without starting a worker. The worker is a daemon thread drawn from
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

    timeout = _search_timeout(timeout)
    try:
        timeout = require_allowance(timeout, label=_TAVILY_SYNC_POLICY.label)
    except TimeoutError:
        logger.warning(
            "tavily_search_timeout",
            query=query[:100],
            timeout_seconds=timeout,
            profile=profile,
        )
        return None

    # The worker shares the caller's absolute deadline, taken from the clamped
    # bound. Without it, a search that finishes after the caller gave up would
    # still start inspection — possibly a paid LLM call — for a result nobody
    # will read, and hold its slot for as long as that takes.
    deadline = llm_timeouts.now() + timeout

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
        # Own event loop: this runs on a worker thread, so it is safe even
        # when the sync caller sits inside a running loop.
        return asyncio.run(
            _inspect_before_deadline(
                raw,
                query,
                deadline=deadline,
                log_fields={
                    "query": query[:100],
                    "timeout_seconds": timeout,
                    "profile": profile,
                },
            )
        )

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
