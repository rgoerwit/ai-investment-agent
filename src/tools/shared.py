"""Shared helpers for tool modules."""

import asyncio
import math
from typing import Any

import structlog

from src.config import config

logger = structlog.get_logger(__name__)

DDG_SEARCH_TIMEOUT_SECONDS = 8.0

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
    logger.warning("TAVILY_API_KEY not set. Tavily tools disabled.")

from src.tavily_utils import set_tavily_tool, tavily_search_with_timeout

if tavily_tool:
    set_tavily_tool(tavily_tool)

_tavily_search_with_timeout = tavily_search_with_timeout

_TAVILY_XML_HEADER = '<search_results source="tavily" data_type="external_web_content">'
_TAVILY_XML_FOOTER = "</search_results>"
_TAVILY_TRUNCATION_SUFFIX = "\n[...truncated]\n</search_results>"
_TAVILY_TRUNCATION_RESERVE = len(_TAVILY_TRUNCATION_SUFFIX) + 10


def _sanitize_for_xml_wrapper(text: str) -> str:
    """Remove sequences that could break out of our XML wrapper."""
    return text.replace("</search_results>", "[removed]")


def _format_and_truncate_tavily_result(
    result: Any, max_chars: int | None = None
) -> str:
    """Format and truncate Tavily search result with security boundaries."""
    if max_chars is None:
        from src.config import config as runtime_config

        max_chars = runtime_config.tavily_max_chars

    formatted_str = ""

    if isinstance(result, list):
        formatted_items = []
        for item in result:
            if isinstance(item, dict):
                title = _sanitize_for_xml_wrapper(str(item.get("title", "No Title")))
                content = _sanitize_for_xml_wrapper(
                    str(item.get("content", "No Content"))
                )
                url = _sanitize_for_xml_wrapper(str(item.get("url", "No URL")))
                score = item.get("score")
                published = item.get("published_date")
                relevance_attr = (
                    f' relevance="{score:.2f}"' if score is not None else ""
                )
                published_attr = f' published="{published}"' if published else ""
                formatted_items.append(
                    f"<result{relevance_attr}{published_attr}>\n"
                    f"<title>{title}</title>\n"
                    f"<url>{url}</url>\n"
                    f"<summary>{content}</summary>\n"
                    f"</result>"
                )
            else:
                sanitized = _sanitize_for_xml_wrapper(str(item))
                formatted_items.append(f"<result><raw>{sanitized}</raw></result>")
        formatted_str = "\n".join(formatted_items)
    elif (
        isinstance(result, dict)
        and "results" in result
        and isinstance(result["results"], list)
    ):
        return _format_and_truncate_tavily_result(result["results"], max_chars)
    else:
        sanitized = _sanitize_for_xml_wrapper(str(result))
        formatted_str = f"<result><raw>{sanitized}</raw></result>"

    wrapped = f"{_TAVILY_XML_HEADER}\n{formatted_str}\n{_TAVILY_XML_FOOTER}"

    if len(wrapped) > max_chars:
        search_limit = max_chars - _TAVILY_TRUNCATION_RESERVE
        last_complete_result = wrapped.rfind("</result>", 0, search_limit)
        if last_complete_result > 0:
            cut_point = last_complete_result + len("</result>")
            return wrapped[:cut_point] + _TAVILY_TRUNCATION_SUFFIX
        return wrapped[:search_limit] + "\n[...truncated mid-result]\n</search_results>"

    return wrapped


async def fetch_with_timeout(coroutine, timeout_seconds=10, error_msg="Timeout"):
    try:
        return await asyncio.wait_for(coroutine, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"YFINANCE TIMEOUT: {error_msg}")
        return None
    except Exception as exc:
        logger.warning(f"YFINANCE ERROR: {error_msg} - {str(exc)}")
        return None


async def extract_company_name_async(ticker_or_obj) -> str:
    """Resolve company name through the shared multi-source resolver."""
    if not isinstance(ticker_or_obj, str):
        info = getattr(ticker_or_obj, "info", None)
        if isinstance(info, dict):
            from src.ticker_utils import normalize_company_name

            candidate = info.get("longName") or info.get("shortName")
            if isinstance(candidate, str) and candidate.strip():
                return normalize_company_name(candidate.strip())

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
    sanitized = {}
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
    """DuckDuckGo fallback search. Returns list of {title, href, body}."""
    try:
        from ddgs import DDGS

        def _sync_search():
            return DDGS(timeout=5).text(query, max_results=max_results)

        results = await asyncio.wait_for(
            asyncio.to_thread(_sync_search),
            timeout=DDG_SEARCH_TIMEOUT_SECONDS,
        )
        return results if results else []
    except ImportError:
        logger.debug("ddgs_not_installed")
        return []
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
                url = item.get("url", "")
                if url:
                    seen_urls.add(url.rstrip("/"))
                merged.append(item)
    elif isinstance(tavily_results, dict) and "results" in tavily_results:
        for item in tavily_results.get("results", []):
            if isinstance(item, dict):
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
                        }
                    )

    return merged
