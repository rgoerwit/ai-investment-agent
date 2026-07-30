"""Shared helpers for tool modules."""

import asyncio
import concurrent.futures
import math  # noqa: E402  (kept after concurrent.futures for grouping)
import re
import threading
from collections.abc import Sequence
from typing import Any

import structlog

from src.config import config
from src.error_safety import summarize_exception

logger = structlog.get_logger(__name__)

DDG_SEARCH_TIMEOUT_SECONDS = 8.0

# DDG concurrency safety
# ----------------------
# Background: ``ddgs`` is lazy-loaded on first call, and its HTTP-client
# ``__init__`` calls ``logging.getLogger`` while doing internal setup that
# holds the GIL. Production hang on 2026-05-09 (0883.HK) showed multiple
# asyncio worker threads entering DDG init concurrently — one held the GIL
# inside DDG init while peers waited on the import lock + ``logging._lock``.
# The asyncio loop in the main thread was starved; ``run_with_hard_timeout``
# deadlines never fired. py-spy confirmed three threads simultaneously
# stuck in ``ddgs/http_client.py:__init__`` and ``ddgs/__init__.py:_load_real``.
#
# The fix has two parts. (1) Constructor re-entry is serialized by a
# process-wide ``threading.Lock`` (``_DDG_INIT_LOCK``) held ONLY around the
# ``DDGS(...)`` constructor — a sub-millisecond critical section — so no two
# threads can be inside DDG's lazy HTTP-client init at once. (2) The dedicated
# executor uses a small worker pool (not a single worker): a DDG call that
# hangs on an uncancellable socket read orphans just one worker thread
# (``run_with_hard_timeout`` bounds the caller at DDG_SEARCH_TIMEOUT_SECONDS
# but cannot reclaim the OS thread), leaving the other workers free. A single
# worker would be permanently saturated by the first hang, silently disabling
# DDG fallback for the rest of the process. Network I/O (``.text()``) runs
# outside the init lock, so searches proceed in parallel.
#
# We also pre-warm DDG once on the main thread at module load. Any
# first-time module-state init then happens single-threaded, before any
# event loop or worker thread exists. Failure of the warm-up is non-fatal.
try:
    import ddgs as _ddgs_module  # noqa: F401  (force eager import)
    from ddgs import DDGS as _DDGS_warm

    try:
        _DDGS_warm(timeout=1)
    except Exception as _warm_exc:  # noqa: BLE001
        logger.debug("ddgs_warm_skip", error=str(_warm_exc))
    del _DDGS_warm

    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.debug("ddgs_not_installed_at_startup")

_DDG_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
# Held ONLY around the DDGS(...) constructor so concurrent workers can't
# re-enter DDG's lazy HTTP-client init; the network call runs outside it.
_DDG_INIT_LOCK = threading.Lock()
# >1 so one hung (uncancellable) socket read orphans a single worker instead of
# saturating the pool and silently disabling DDG fallback for the rest of the run.
_DDG_EXECUTOR_MAX_WORKERS = 4


def _get_ddg_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the dedicated DDG worker pool.

    Lazy so that test harnesses which don't touch DDG don't pay the cost of
    threads, and so a worker process that forks before DDG is used gets a
    fresh executor in the child rather than inheriting a half-initialized one
    from the parent. Constructor safety comes from ``_DDG_INIT_LOCK``, not from
    restricting the pool to a single worker.
    """
    global _DDG_EXECUTOR
    if _DDG_EXECUTOR is None:
        _DDG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=_DDG_EXECUTOR_MAX_WORKERS, thread_name_prefix="ddg-search"
        )
    return _DDG_EXECUTOR


TAVILY_AVAILABLE = False
tavily_tool = None
_tavily_api_key = config.get_tavily_api_key()
if _tavily_api_key:
    try:
        from langchain_tavily import TavilySearch

        tavily_tool = TavilySearch(max_results=5, tavily_api_key=_tavily_api_key)
        TAVILY_AVAILABLE = True
    except ImportError:
        logger.warning(
            "tavily_not_installed",
            hint="Run 'poetry add langchain-tavily' to enable Tavily search",
        )
else:
    logger.warning("tavily_api_key_not_set_tavily_tools_disabled")

from src.tavily_utils import set_tavily_tool, tavily_search_with_timeout

if tavily_tool:
    set_tavily_tool(tavily_tool)

_tavily_search_with_timeout = tavily_search_with_timeout

_TAVILY_XML_HEADER = '<search_results source="tavily" data_type="external_web_content">'
_TAVILY_XML_FOOTER = "</search_results>"
_TAVILY_TRUNCATION_NOTE = (
    "[...truncated]\n"
    "[summary policy: result metadata and query-relevant excerpts preserved]"
)


def _sanitize_for_xml_wrapper(text: str) -> str:
    """Remove sequences that could break out of our XML wrapper."""
    return text.replace("</search_results>", "[removed]")


def _bounded_field(text: str, max_chars: int) -> str:
    """Bound metadata only when an unusually small envelope makes it necessary."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _query_terms(query: str | None) -> list[str]:
    """Return useful search terms, longest first, for excerpt selection."""
    if not query:
        return []
    terms = {
        token.strip(".,:;!?()[]{}\"'")
        for token in re.split(r"\s+", query)
        if len(token.strip(".,:;!?()[]{}\"'")) >= 2
    }
    return sorted(terms, key=len, reverse=True)


def _query_centered_excerpt(
    content: str,
    *,
    max_chars: int,
    query: str | None,
    priority_terms: Sequence[str] = (),
) -> str:
    """Return a bounded excerpt centered on the first useful query-term match."""
    if max_chars <= 0:
        return ""
    if len(content) <= max_chars:
        return content

    folded = content.casefold()
    match_start: int | None = None
    match_length = 0
    query_terms = _query_terms(query)
    preferred_terms = [
        term.strip()
        for term in priority_terms
        if isinstance(term, str) and len(term.strip()) >= 2
    ]
    terms = preferred_terms + [
        term
        for term in query_terms
        if term.casefold()
        not in {preferred.casefold() for preferred in preferred_terms}
    ]
    for term in terms:
        index = folded.find(term.casefold())
        if index >= 0 and (match_start is None or len(term) > match_length):
            match_start = index
            match_length = len(term)

    if match_start is None:
        return content[: max(0, max_chars - 3)] + "..."

    marker_chars = 6
    window_chars = max(0, max_chars - marker_chars)
    start = max(0, match_start - window_chars // 2)
    end = min(len(content), start + window_chars)
    start = max(0, end - window_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    return prefix + content[start:end] + suffix


def _normalize_search_items(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict) and isinstance(result.get("results"), list):
        result = result["results"]
    if not isinstance(result, list):
        return [{"raw": _sanitize_for_xml_wrapper(str(result))}]

    normalized: list[dict[str, Any]] = []
    for item in result:
        if not isinstance(item, dict):
            normalized.append({"raw": _sanitize_for_xml_wrapper(str(item))})
            continue
        normalized.append(
            {
                "title": _sanitize_for_xml_wrapper(str(item.get("title", "No Title"))),
                "content": _sanitize_for_xml_wrapper(
                    str(item.get("content", "No Content"))
                ),
                "url": _sanitize_for_xml_wrapper(str(item.get("url", "No URL"))),
                "score": item.get("score"),
                "published": _sanitize_for_xml_wrapper(
                    str(item.get("published_date", ""))
                ),
                "source": _sanitize_for_xml_wrapper(str(item.get("_source", ""))),
            }
        )
    return normalized


def _render_search_item(
    item: dict[str, Any],
    *,
    summary_budget: int | None,
    query: str | None,
    priority_terms: Sequence[str] = (),
    title_limit: int = 240,
    url_limit: int = 500,
) -> str:
    raw = item.get("raw")
    if raw is not None:
        raw_text = str(raw)
        excerpt = (
            raw_text
            if summary_budget is None
            else _query_centered_excerpt(
                raw_text,
                max_chars=summary_budget,
                query=query,
                priority_terms=priority_terms,
            )
        )
        truncated_attr = (
            f' truncated="true" original_chars="{len(raw_text)}"'
            if len(excerpt) < len(raw_text)
            else ""
        )
        return f"<result{truncated_attr}><raw>{excerpt}</raw></result>"

    title = _bounded_field(str(item.get("title", "No Title")), title_limit)
    url = _bounded_field(str(item.get("url", "No URL")), url_limit)
    content = str(item.get("content", "No Content"))
    excerpt = (
        content
        if summary_budget is None
        else _query_centered_excerpt(
            content,
            max_chars=summary_budget,
            query=query,
            priority_terms=priority_terms,
        )
    )
    score = item.get("score")
    published = str(item.get("published", ""))
    source = str(item.get("source", ""))
    relevance_attr = f' relevance="{score:.2f}"' if score is not None else ""
    published_attr = f' published="{published}"' if published else ""
    source_attr = f' provider="{source}"' if source else ""
    truncated_attr = (
        f' truncated="true" original_chars="{len(content)}"'
        if len(excerpt) < len(content)
        else ""
    )
    return (
        f"<result{relevance_attr}{published_attr}{source_attr}{truncated_attr}>\n"
        f"<title>{title}</title>\n"
        f"<url>{url}</url>\n"
        f"<summary>{excerpt}</summary>\n"
        f"</result>"
    )


def _format_and_truncate_tavily_result(
    result: Any,
    max_chars: int | None = None,
    *,
    query: str | None = None,
    priority_terms: Sequence[str] = (),
) -> str:
    """Format search results without letting one long document erase others.

    The global envelope remains bounded, but normal result metadata is reserved
    before summary space is allocated fairly. Oversized summaries become
    query-centered excerpts instead of causing the whole result to disappear.
    """
    if max_chars is None:
        from src.config import config as runtime_config

        max_chars = runtime_config.tavily_max_chars
    max_chars = max(160, max_chars)
    items = _normalize_search_items(result)
    full_items = [
        _render_search_item(
            item,
            summary_budget=None,
            query=query,
            priority_terms=priority_terms,
        )
        for item in items
    ]
    wrapped = (
        f"{_TAVILY_XML_HEADER}\n" + "\n".join(full_items) + f"\n{_TAVILY_XML_FOOTER}"
    )
    if len(wrapped) <= max_chars:
        return wrapped

    footer = f"\n{_TAVILY_TRUNCATION_NOTE}\n{_TAVILY_XML_FOOTER}"
    body_budget = max_chars - len(_TAVILY_XML_HEADER) - len(footer) - 1
    item_count = max(1, len(items))

    # At the normal 7,000-character limit this preserves complete titles and
    # URLs. Tiny test/operator envelopes degrade metadata uniformly rather than
    # dropping later candidates altogether.
    metadata_probe = [
        _render_search_item(
            item,
            summary_budget=0,
            query=query,
            priority_terms=priority_terms,
        )
        for item in items
    ]
    metadata_chars = len("\n".join(metadata_probe))
    title_limit = 240
    url_limit = 500
    if metadata_chars > body_budget:
        per_item = max(64, body_budget // item_count)
        title_limit = max(16, min(120, per_item // 3))
        url_limit = max(24, min(240, per_item // 2))
        metadata_probe = [
            _render_search_item(
                item,
                summary_budget=0,
                query=query,
                priority_terms=priority_terms,
                title_limit=title_limit,
                url_limit=url_limit,
            )
            for item in items
        ]
        metadata_chars = len("\n".join(metadata_probe))

    summary_total = max(0, body_budget - metadata_chars)
    summary_budget = summary_total // item_count
    rendered_items = [
        _render_search_item(
            item,
            summary_budget=summary_budget,
            query=query,
            priority_terms=priority_terms,
            title_limit=title_limit,
            url_limit=url_limit,
        )
        for item in items
    ]
    bounded = f"{_TAVILY_XML_HEADER}\n" + "\n".join(rendered_items) + footer

    # Attribute overhead can vary slightly once truncation telemetry appears.
    # Reduce all excerpts evenly until the hard envelope is met.
    while len(bounded) > max_chars and summary_budget > 0:
        excess_per_item = math.ceil((len(bounded) - max_chars) / item_count)
        summary_budget = max(0, summary_budget - max(1, excess_per_item))
        rendered_items = [
            _render_search_item(
                item,
                summary_budget=summary_budget,
                query=query,
                priority_terms=priority_terms,
                title_limit=title_limit,
                url_limit=url_limit,
            )
            for item in items
        ]
        bounded = f"{_TAVILY_XML_HEADER}\n" + "\n".join(rendered_items) + footer

    return bounded


async def fetch_with_timeout(coroutine, timeout_seconds=10, error_msg="Timeout"):
    from src.async_utils import run_with_hard_timeout

    try:
        return await run_with_hard_timeout(
            coroutine,
            timeout=timeout_seconds,
            label=f"shared.fetch_with_timeout:{error_msg}",
        )
    except asyncio.TimeoutError:
        logger.warning("yfinance_timeout", context=error_msg)
        return None
    except Exception as exc:
        logger.warning(
            "yfinance_fetch_failed",
            context=error_msg,
            **summarize_exception(exc, operation="yfinance_fetch"),
        )
        return None


async def extract_company_name_async(ticker_or_obj) -> str:
    """Resolve the search-friendly company name through the shared resolver.

    Tool search queries intentionally use the normalized name with legal suffixes
    stripped. Runtime state/output should use CompanyNameResult.canonical_name.
    """
    if not isinstance(ticker_or_obj, str):
        info = getattr(ticker_or_obj, "info", None)
        if isinstance(info, dict):
            from src.ticker_utils import (
                _is_valid_company_name,
                normalize_company_name,
            )

            candidate = info.get("longName") or info.get("shortName")
            obj_ticker = getattr(ticker_or_obj, "ticker", "") or ""
            if (
                isinstance(candidate, str)
                and candidate.strip()
                and _is_valid_company_name(candidate.strip(), obj_ticker)
            ):
                return normalize_company_name(candidate.strip())
            # Object-path candidate rejected (CSV identifier blob, ticker echo,
            # empty); fall through to the string resolver below.

    ticker_str = (
        ticker_or_obj
        if isinstance(ticker_or_obj, str)
        else getattr(ticker_or_obj, "ticker", str(ticker_or_obj))
    )
    try:
        from src.ticker_utils import resolve_company_name

        result = await resolve_company_name(ticker_str)
        return result.name if result.is_resolved else ticker_str
    except Exception:
        return ticker_str


def _safe_float(value: Any) -> float | None:
    """Safely convert value to float, handling None, strings, NaN, and Inf."""
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "")
        converted = float(value)
        if math.isnan(converted) or math.isinf(converted):
            return None
        return converted
    except (ValueError, TypeError):
        return None


def _format_val(value: Any, fmt: str = "{:.2f}", default: str = "N/A") -> str:
    """Format a value safely, returning default if invalid."""
    val = _safe_float(value)
    if val is None:
        return default
    return fmt.format(val)


def _sanitize_for_json(data: dict) -> dict:
    """Sanitize data for JSON encoding."""
    sanitized: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            sanitized[key] = _sanitize_for_json(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_for_json(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, float):
            if math.isinf(value) or math.isnan(value):
                sanitized[key] = None
            elif key == "currentPrice" and value < 0:
                sanitized[key] = None
            else:
                sanitized[key] = value
        elif isinstance(value, str) and key not in {
            "_data_source",
            "currency",
            "symbol",
        }:
            try:
                sanitized[key] = float(value)
            except (ValueError, TypeError):
                sanitized[key] = value
        else:
            sanitized[key] = value
    return sanitized


async def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo fallback search. Returns list of {title, href, body}.

    Runs on the dedicated DDG worker pool. The ``DDGS(...)`` constructor is
    serialized by ``_DDG_INIT_LOCK`` to prevent the import-lock / logging-lock
    re-entry deadlock observed in production; the network call runs outside the
    lock so concurrent searches proceed in parallel, and one hung read orphans
    only a single worker. See module-level note for full background.
    """
    from src.async_utils import run_with_hard_timeout

    if not DDGS_AVAILABLE:
        return []

    try:
        from ddgs import DDGS

        def _sync_search():
            with _DDG_INIT_LOCK:
                client = DDGS(timeout=5)
            # Materialize inside the worker thread (defensive: .text() returns a
            # list today, but pin it so no lazy iterator escapes the pool).
            return list(client.text(query, max_results=max_results))

        loop = asyncio.get_running_loop()
        results = await run_with_hard_timeout(
            loop.run_in_executor(_get_ddg_executor(), _sync_search),
            timeout=DDG_SEARCH_TIMEOUT_SECONDS,
            label=f"ddg:{query[:60]}",
        )
        return results if results else []
    except asyncio.TimeoutError:
        logger.debug(
            "ddg_search_timeout",
            query=query[:100],
            timeout_seconds=DDG_SEARCH_TIMEOUT_SECONDS,
        )
        return []
    except Exception as exc:
        logger.debug("ddg_search_error", error=str(exc))
        return []


def _merge_search_results(tavily_results, ddg_results) -> list[dict]:
    """Merge Tavily and DDG results, deduplicating by URL."""
    merged = []
    seen_urls = set()

    if isinstance(tavily_results, list):
        for item in tavily_results:
            if isinstance(item, dict):
                item = {**item, "_source": item.get("_source", "tavily")}
                url = item.get("url", "")
                if url:
                    seen_urls.add(url.rstrip("/"))
                merged.append(item)
    elif isinstance(tavily_results, dict) and "results" in tavily_results:
        for item in tavily_results.get("results", []):
            if isinstance(item, dict):
                item = {**item, "_source": item.get("_source", "tavily")}
                url = item.get("url", "")
                if url:
                    seen_urls.add(url.rstrip("/"))
                merged.append(item)

    if isinstance(ddg_results, list):
        for item in ddg_results:
            if isinstance(item, dict):
                url = item.get("href", item.get("url", ""))
                if url and url.rstrip("/") not in seen_urls:
                    seen_urls.add(url.rstrip("/"))
                    merged.append(
                        {
                            "title": item.get("title", ""),
                            "url": url,
                            "content": item.get("body", item.get("content", "")),
                            "_source": "duckduckgo",
                        }
                    )

    return merged
