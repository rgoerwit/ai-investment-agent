"""Pre-call argument validation for high-risk outbound tools.

A separate ``ToolHook`` from ``ContentInspectionHook`` — that hook is an
after-output adapter; argument policy is a distinct concern.

Scope: editor freeform egress tools only (``fetch_reference_content``,
``search_claim``).  These are the highest-risk outbound surfaces because
an injected LLM can be tricked into fetching attacker-controlled URLs
or crafting exfiltration queries.
"""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

import structlog

from src.tooling.runtime import ToolCallBlocked, ToolInvocation, ToolResult

logger = structlog.get_logger(__name__)

# Maximum URL query-string length before flagging.
_MAX_QUERY_STRING_LENGTH = 500

# Maximum search query text length.
_MAX_SEARCH_QUERY_LENGTH = 500

# Allowed URL schemes.
_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Suspicious patterns in URLs (data exfiltration signals).
_SUSPICIOUS_URL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"webhook\.site", re.I),
    re.compile(r"requestbin\.com", re.I),
    re.compile(r"pipedream\.net", re.I),
    re.compile(r"ngrok\.io", re.I),
    re.compile(r"burpcollaborator\.net", re.I),
]


def _is_reasonable_reference_url(url: str) -> bool:
    """Return True if the URL looks like a legitimate reference."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False

    if not parsed.hostname:
        return False

    # Flag excessively long query strings (exfiltration signal).
    if parsed.query and len(parsed.query) > _MAX_QUERY_STRING_LENGTH:
        return False

    # Flag known exfiltration endpoints.
    for pattern in _SUSPICIOUS_URL_PATTERNS:
        if pattern.search(url):
            return False

    return True


def _looks_like_pasted_payload(query: str) -> bool:
    """Return True if the search query looks like pasted content rather than a query."""
    # Multiple newlines suggest pasted block content.
    if query.count("\n") > 3:
        return True
    # High ratio of special characters.
    special = sum(1 for c in query if c in "{}[]<>/\\|=")
    if len(query) > 50 and special / len(query) > 0.15:
        return True
    return False


class ToolArgumentPolicyHook:
    """Pre-call argument validation for high-risk outbound tools.

    Implements ``ToolHook`` protocol.
    """

    def __init__(self, mode: Literal["warn", "block"] = "warn") -> None:
        self._mode = mode

    async def before(self, call: ToolInvocation) -> ToolInvocation:
        if call.source != "editor":
            return call

        if call.name == "fetch_reference_content":
            url = str(call.args.get("url", ""))
            if not _is_reasonable_reference_url(url):
                if self._mode == "block":
                    raise ToolCallBlocked("editor reference URL rejected by policy")
                logger.warning(
                    "outbound_url_suspicious",
                    url=url[:120],
                    tool=call.name,
                    agent_key=call.agent_key,
                )

        if call.name == "search_claim":
            query = str(call.args.get("query", ""))
            if len(query) > _MAX_SEARCH_QUERY_LENGTH or _looks_like_pasted_payload(
                query
            ):
                if self._mode == "block":
                    raise ToolCallBlocked("editor search query rejected by policy")
                logger.warning(
                    "outbound_query_suspicious",
                    query=query[:80],
                    tool=call.name,
                    agent_key=call.agent_key,
                )

        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        return result  # pass-through
