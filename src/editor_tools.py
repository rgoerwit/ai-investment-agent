"""
Tools for the Editor-in-Chief agent.

Design: "Dumb Tool" pattern - tools do pure I/O only.
The Agent (Editor) handles all reasoning/verification logic.
"""

from urllib.parse import urljoin

import httpx
import structlog
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from src.error_safety import (
    format_error_message,
    redact_sensitive_text,
    summarize_exception,
)
from src.runtime_diagnostics import classify_failure
from src.runtime_services import get_current_inspection_service
from src.tooling.inspector import InspectionEnvelope, SourceKind
from src.tooling.tool_argument_policy import _is_reasonable_reference_url

logger = structlog.get_logger(__name__)

# Maximum characters to fetch from each reference URL
MAX_REFERENCE_CHARS = 5000

# Request timeout in seconds
REQUEST_TIMEOUT = 10.0
MAX_REFERENCE_REDIRECTS = 3

# User agent for requests
USER_AGENT = "Mozilla/5.0 (compatible; InvestorAgent/1.0; +https://github.com/rgoerwit/ai-investment-agent)"

_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


def _safe_url_preview(url: str) -> str:
    return redact_sensitive_text(url, max_chars=80)


def _safe_query_preview(query: str) -> str:
    return redact_sensitive_text(query, max_chars=80)


def _resolve_redirect_target(current_url: str, location: str) -> str:
    return urljoin(current_url, location)


async def _fetch_reference_response(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
) -> httpx.Response:
    current_url = url

    for _ in range(MAX_REFERENCE_REDIRECTS + 1):
        if not _is_reasonable_reference_url(current_url):
            raise ValueError(f"redirect target rejected by policy: {current_url}")

        response = await client.get(
            current_url,
            headers=headers,
            follow_redirects=False,
        )

        if response.status_code not in _REDIRECT_STATUS_CODES:
            return response

        location = response.headers.get("location", "").strip()
        if not location:
            return response

        current_url = _resolve_redirect_target(current_url, location)

    raise ValueError("too many redirects")


@tool("fetch_reference_content")
async def fetch_reference_content(url: str) -> str:
    """
    Fetch the text content of a URL to verify a citation.

    Use this tool to retrieve source material for fact-checking claims
    in the article. The Editor will compare the returned text against
    the claim to determine if it's supported.

    Args:
        url: The reference URL from the article's References section

    Returns:
        Text content from the URL (truncated to 5000 chars), or error message
    """
    if not _is_reasonable_reference_url(url):
        return f"INVALID_URL: '{url}' is not a valid HTTP/HTTPS URL"

    try:
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await _fetch_reference_response(client, url, headers)
            resp.raise_for_status()

        # Parse HTML and extract text
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove non-content elements
        for tag in soup(
            ["script", "style", "nav", "footer", "header", "aside", "form"]
        ):
            tag.decompose()

        # Extract text with whitespace normalization
        text = soup.get_text(separator=" ", strip=True)

        # Normalize multiple spaces
        text = " ".join(text.split())

        if len(text) < 100:
            logger.warning(
                "Reference content too short",
                url=_safe_url_preview(url),
                chars=len(text),
            )
            return f"INSUFFICIENT_CONTENT: URL returned only {len(text)} chars of text"

        # Truncate to limit
        if len(text) > MAX_REFERENCE_CHARS:
            text = text[:MAX_REFERENCE_CHARS] + "...[truncated]"

        logger.info(
            "Fetched reference content",
            url=_safe_url_preview(url),
            chars=len(text),
        )

        envelope = InspectionEnvelope(
            content_text=text,
            source_kind=SourceKind.web_fetch,
            source_name="httpx",
            source_uri=url,
            metadata={"url": url[:200]},
        )
        return await get_current_inspection_service().check(envelope)

    except httpx.TimeoutException:
        logger.warning(
            "reference_fetch_timeout",
            url=_safe_url_preview(url),
            failure_kind="timeout",
        )
        return "FETCH_FAILED: Request timed out after 10 seconds"

    except ValueError as exc:
        summary = summarize_exception(
            exc,
            operation="fetch_reference_content",
            provider="unknown",
        )
        logger.warning(
            "reference_fetch_rejected",
            url=_safe_url_preview(url),
            **summary,
        )
        if "redirect" in str(exc).lower():
            return f"FETCH_FAILED: redirect blocked ({summary['error_type']})"
        return format_error_message(
            operation="validating fetch_reference_content URL",
            error_type=summary["error_type"],
            message_preview=summary["message_preview"],
        ).replace("Error in validating fetch_reference_content URL", "INVALID_URL")

    except httpx.HTTPStatusError as e:
        logger.warning(
            "reference_fetch_http_error",
            url=_safe_url_preview(url),
            status=e.response.status_code,
            response_preview=redact_sensitive_text(
                e.response.text or "",
                max_chars=64,
            ),
        )
        return f"FETCH_FAILED: HTTP {e.response.status_code}"

    except httpx.RequestError as e:
        details = classify_failure(e, provider="unknown")
        logger.warning(
            "reference_fetch_request_error",
            url=_safe_url_preview(url),
            failure_kind=details.kind,
            retryable=details.retryable,
            error_type=details.error_type,
            error_message=details.message,
        )
        return f"FETCH_FAILED: {type(e).__name__}"

    except Exception as e:
        summary = summarize_exception(
            e,
            operation="fetch_reference_content",
            provider="unknown",
        )
        logger.error(
            "reference_fetch_unexpected_error",
            url=_safe_url_preview(url),
            **summary,
        )
        return "FETCH_FAILED: " + format_error_message(
            operation="fetch_reference_content",
            error_type=summary["error_type"],
            message_preview=summary["message_preview"],
        )


MAX_CLAIM_SEARCH_CHARS = 3000


@tool("search_claim")
async def search_claim(query: str) -> str:
    """
    Search the web to verify a specific factual claim in the article.

    Use this to fact-check event dates, product launches, company history,
    or any narrative claim that cannot be verified from the DATA_BLOCK alone.
    Do NOT use for generic financial metrics (those come from DATA_BLOCK).

    Args:
        query: A specific, targeted search query (e.g., "Tsuburaya Fields
               Ultraman Card Game launch date")

    Returns:
        Search results summary (truncated to 3000 chars), or error message
    """
    if not query or len(query.strip()) < 5:
        return "INVALID_QUERY: Query too short — be specific about the claim to verify"

    try:
        from src.tavily_utils import tavily_search_with_timeout

        result = await tavily_search_with_timeout({"query": query.strip()})
        if not result:
            return "SEARCH_UNAVAILABLE: Tavily search returned no results (API may be unavailable)"

        if isinstance(result, str) and result.startswith("TOOL_BLOCKED:"):
            return result

        text = str(result)
        if len(text) > MAX_CLAIM_SEARCH_CHARS:
            text = text[:MAX_CLAIM_SEARCH_CHARS] + "...[truncated]"
        logger.info(
            "claim_search_complete",
            query=_safe_query_preview(query),
            result_chars=len(text),
        )
        return text

    except Exception as e:
        summary = summarize_exception(
            e,
            operation="search_claim",
            provider="unknown",
        )
        logger.warning(
            "claim_search_failed",
            query=_safe_query_preview(query),
            **summary,
        )
        return "SEARCH_FAILED: " + format_error_message(
            operation="search_claim",
            error_type=summary["error_type"],
            message_preview=summary["message_preview"],
        )


def get_editor_tools() -> list:
    """
    Get the list of tools available to the Editor-in-Chief.

    Returns:
        List of tool functions for binding to the editor LLM
    """
    return [fetch_reference_content, search_claim]


# NOTE: BeautifulSoup stripping is sufficient for V1. If content extraction
# proves unreliable (too much nav/boilerplate), consider upgrading to
# `trafilatura` or `readability-lxml` for smarter main-content detection.
