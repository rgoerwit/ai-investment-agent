"""
Tools for the Editor-in-Chief agent.

Design: "Dumb Tool" pattern - tools do pure I/O only.
The Agent (Editor) handles all reasoning/verification logic.
"""

import httpx
import structlog
from bs4 import BeautifulSoup
from langchain_core.tools import tool

logger = structlog.get_logger(__name__)

# Maximum characters to fetch from each reference URL
MAX_REFERENCE_CHARS = 5000

# Request timeout in seconds
REQUEST_TIMEOUT = 10.0

# User agent for requests
USER_AGENT = "Mozilla/5.0 (compatible; InvestorAgent/1.0; +https://github.com/rgoerwit/ai-investment-agent)"


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
    if not url or not url.startswith(("http://", "https://")):
        return f"INVALID_URL: '{url}' is not a valid HTTP/HTTPS URL"

    try:
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(url, headers=headers, follow_redirects=True)
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
                url=url[:80],
                chars=len(text),
            )
            return f"INSUFFICIENT_CONTENT: URL returned only {len(text)} chars of text"

        # Truncate to limit
        if len(text) > MAX_REFERENCE_CHARS:
            text = text[:MAX_REFERENCE_CHARS] + "...[truncated]"

        logger.info(
            "Fetched reference content",
            url=url[:80],
            chars=len(text),
        )

        return text

    except httpx.TimeoutException:
        logger.warning("Reference fetch timeout", url=url[:80])
        return "FETCH_FAILED: Request timed out after 10 seconds"

    except httpx.HTTPStatusError as e:
        logger.warning(
            "Reference fetch HTTP error",
            url=url[:80],
            status=e.response.status_code,
        )
        return f"FETCH_FAILED: HTTP {e.response.status_code}"

    except httpx.RequestError as e:
        logger.warning("Reference fetch request error", url=url[:80], error=str(e))
        return f"FETCH_FAILED: {type(e).__name__}"

    except Exception as e:
        logger.error("Unexpected error fetching reference", url=url[:80], error=str(e))
        return f"FETCH_FAILED: {str(e)}"


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

        # Parse results — tavily_search_with_timeout returns raw tool output
        text = str(result)
        if len(text) > MAX_CLAIM_SEARCH_CHARS:
            text = text[:MAX_CLAIM_SEARCH_CHARS] + "...[truncated]"

        logger.info("claim_search_complete", query=query[:80], result_chars=len(text))
        return text

    except Exception as e:
        logger.warning("claim_search_failed", query=query[:80], error=str(e))
        return f"SEARCH_FAILED: {str(e)}"


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
