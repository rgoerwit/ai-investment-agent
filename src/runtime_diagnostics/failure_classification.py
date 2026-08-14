from __future__ import annotations

import re
import socket
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from src.error_safety import redact_sensitive_text

ProviderName = Literal[
    "google",
    "openai",
    "anthropic",
    "deepseek",
    "zai",
    "moonshot",
    "unknown",
]
_PROVIDER_NAMES: frozenset[str] = frozenset(
    {"google", "openai", "anthropic", "deepseek", "zai", "moonshot", "unknown"}
)
FailureKind = Literal[
    "dns_resolution",
    "connect_error",
    "timeout",
    "auth_error",
    "rate_limit",
    "quota_error",
    "server_error",
    "model_not_found",
    "bad_request",
    "application_error",
    "data_unavailable",
    "provider_safety_block",
    "provider_partial_response",
    "unknown_provider_error",
]
ArtifactErrorKind = FailureKind | Literal["application_error"]


@dataclass(frozen=True)
class FailureDetails:
    kind: FailureKind
    provider: ProviderName
    host: str | None
    error_type: str
    root_cause_type: str
    retryable: bool
    message: str
    # Host of the endpoint the call was sent to, when the caller supplies it.
    # This is what identifies the real vendor behind an OpenAI-compatible seat,
    # without a vendor lookup table that would need editing per new provider.
    endpoint_host: str | None = None


_HOST_PATTERN = re.compile(r"(?:host|https?://)([A-Za-z0-9.-]+\.[A-Za-z]{2,})")

# A status code must be a standalone number *introduced as one*. Two distinct
# failure modes, both live before this pattern existed:
#   - substring matching found "403" inside "executive order 14032" — prompt text
#     a content filter happily echoes back in its rejection body;
#   - a three-digit number anywhere at all was read as a status, so an exception
#     carrying financial prose ("revenue was 500 million", "debt was 403 million")
#     classified as server_error / auth_error, changing retry and breaker
#     behaviour. In an equity-analysis system those numbers are everywhere.
# Requiring a code/status/http/error introducer (or line start) loses **zero**
# codes across the 685 error messages persisted in results/, while rejecting all
# of the prose cases. Widen only against re-measured corpus evidence.
# Three introducers, each deliberately narrow:
#   1. a keyword — ``Error code: 400``, ``'status': 503``, ``HTTP 504``;
#   2. line start — a message that opens with the bare code;
#   3. an exception class-name prefix at line start — ``ServiceUnavailable: 503``.
# Arm 3 exists because ``combined`` prefixes the class name, so a bare-status
# message is never at position 0, and Google's ``ResourceExhausted`` /
# ``ServiceUnavailable`` contain no keyword. Anchoring it to ``^identifier:``
# is what keeps it from degenerating into "any colon": a generic ``[:\-=]`` arm
# reads ``{"revenue": 500}`` as a server error and ``metric=429`` as a rate
# limit, and this system's exception text is full of such structured magnitudes.
_STATUS_CODE_PATTERN = re.compile(
    r"(?:(?:code|status|http|error)\b[^\w\n]{0,4}|^[^\W\d][\w.]*:[ \t]*|^)"
    r"(?<!\d)(\d{3})(?!\d)",
    re.IGNORECASE | re.MULTILINE,
)

# Refusal detection matches *token proximity*, not sentences. Vendor wording is
# the least stable part of a safety layer — these checks are typically bolted on
# and reworded — while the shape of the signal is invariant: a word naming the
# material sits next to a word naming the rejection ("content_filter",
# "PROHIBITED_CONTENT", "Content Exists Risk", "data_inspection_failed",
# "content management policy"). Matching the pair adapts to rewordings and to
# vendors we have never seen; matching a sentence would not.

# Rejection verbs — unambiguous in any context.
_REJECTION_VERBS = (
    "filter",  # filtered / filtering
    "block",  # blocked / blocking
    "reject",  # rejected / rejection
    "violat",  # violation / violates
    "refus",  # refusal / refused
    "prohibit",
    "disallow",
    "censor",
    "withheld",
)
# Weaker signals, admissible only next to "content" — this system analyses
# equities, so "safety risk" and "safety policy" are ordinary subject matter and
# must never read as a provider refusal.
_REJECTION_QUALIFIERS = (
    "unsafe",
    "sensitive",
    "harmful",
    "flagged",
    "policy",
    "risk",
)
# Which rejection tokens each material token may pair with. Asymmetric on
# purpose — see the qualifier comment above.
_MATERIAL_PAIRINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("content", _REJECTION_VERBS + _REJECTION_QUALIFIERS),
    ("safety", _REJECTION_VERBS),
    # Alibaba/Qwen's `data_inspection_failed`. "inspection" cannot stand alone:
    # this repo has its own InspectionService / ContentInspectionHook, ~245
    # mentions, so a bare match would misread our own errors as provider
    # refusals. The "data" qualifier is what makes it the vendor's code.
    ("inspection", ("data",)),
)
# How far apart the two may sit. Three spans "content management policy" and
# "Content Exists Risk" without letting coincidental words pair up across a
# sentence. Deliberately excludes generic outcome words like "failed": an
# unrelated "…_content failed" must not read as a refusal.
_PROXIMITY_WINDOW = 3

# Tokens distinctive enough to stand alone — vocabulary that appears only when a
# moderation layer is speaking, and that this repo does not use for anything
# else (verified by grep before adding: "inspection" failed that test and is a
# pairing instead, "guardrail" appears twice in unrelated prose so it is out).
# "recitation" is Gemini's finish_reason.
_STANDALONE_TOKENS = (
    "moderation",
    "recitation",
)

# Our own error marker, and a vendor policy code whose message may be
# Chinese-only (Z.AI/GLM 1301). The code stays a conjunction so a stray 1301 —
# an order id, a byte count — cannot classify a call as refused.
_OWN_REFUSAL_MARKER = "provider safety block"
_POLICY_CODE_PATTERN = re.compile(r"(?<!\d)1301(?!\d)")
_POLICY_CONTEXT_PATTERN = re.compile(
    r"policy|content|sensitive|unsafe|moderation|不安全|敏感", re.IGNORECASE
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")
_SEPARATORS = re.compile(r"[^0-9a-z一-鿿]+")


def _normalize_tokens(text: str) -> list[str]:
    """Split on punctuation and camelCase so spelling variants tokenize alike.

    ``content_filter``, ``contentFilter`` and ``content filter`` all become
    ``["content", "filter"]``.
    """
    folded = _SEPARATORS.sub(" ", _CAMEL_BOUNDARY.sub(" ", text).lower())
    return folded.split()


def _has_adjacent_pair(tokens: list[str]) -> bool:
    for material, rejections in _MATERIAL_PAIRINGS:
        material_at = [i for i, tok in enumerate(tokens) if tok.startswith(material)]
        if not material_at:
            continue
        rejection_at = [i for i, tok in enumerate(tokens) if tok.startswith(rejections)]
        if any(
            abs(m - r) <= _PROXIMITY_WINDOW for m in material_at for r in rejection_at
        ):
            return True
    return False


def is_provider_content_block(text: str) -> bool:
    """Return True when a provider rejected the call on content-policy grounds.

    The single definition of "the provider refused this". Both ``classify_failure``
    and the APAC specialist's policy-block retry consume it, so the two cannot
    drift apart — they previously recognized disjoint vocabularies, and the only
    refusal actually observed on the APAC seat matched the node-local detector
    but not the global one.
    """
    tokens = _normalize_tokens(text)
    if _OWN_REFUSAL_MARKER in " ".join(tokens):
        return True
    if any(token.startswith(_STANDALONE_TOKENS) for token in tokens):
        return True
    if _has_adjacent_pair(tokens):
        return True
    return bool(
        _POLICY_CODE_PATTERN.search(text) and _POLICY_CONTEXT_PATTERN.search(text)
    )


def _root_cause(exc: BaseException) -> BaseException:
    current = exc
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        next_exc = current.__cause__ or current.__context__
        if next_exc is None:
            break
        current = next_exc
    return current


def _url_host(url: str | None) -> str | None:
    """Host portion of a base URL, or None. No vendor table, no allowlist."""
    if not url:
        return None
    host = urlsplit(url if "//" in url else f"//{url}").hostname
    return host or None


def _extract_host(message: str) -> str | None:
    match = _HOST_PATTERN.search(message)
    if not match:
        return None
    host = match.group(1)
    return host.rstrip(":/")


def infer_provider(
    model_name: str | None = None, class_name: str | None = None
) -> ProviderName:
    """Infer provider identity for legacy or externally constructed clients.

    Registry-built clients use :func:`get_runtime_provider`, whose stamped
    identity takes precedence over this model/class-name fallback. This keeps an
    OpenAI-compatible SDK transport from inheriting OpenAI-only runtime policy.
    """
    haystack = " ".join(part for part in (model_name, class_name) if part).lower()
    if "gemini" in haystack or "google" in haystack:
        return "google"
    if "deepseek" in haystack:
        return "deepseek"
    if "kimi" in haystack or "moonshot" in haystack:
        return "moonshot"
    if "glm" in haystack or "zai" in haystack or "z.ai" in haystack:
        return "zai"
    if "gpt" in haystack or "openai" in haystack:
        return "openai"
    if "claude" in haystack or "anthropic" in haystack:
        return "anthropic"
    return "unknown"


def get_runtime_provider(runnable: Any) -> ProviderName:
    """Return the bound provider policy identity, falling back for legacy clients."""

    stamped = getattr(runnable, "_llm_runtime_provider", None)
    if isinstance(stamped, str) and stamped in _PROVIDER_NAMES:
        return cast(ProviderName, stamped)
    return infer_provider(
        model_name=get_model_name(runnable),
        class_name=get_class_name(runnable),
    )


def get_base_url(runnable: Any) -> str | None:
    """Best-effort endpoint URL for a chat model.

    Returns the **full URL**, which may carry a path, query string, or embedded
    credentials (``https://user:key@host/v1``). Never log this directly — log
    ``get_endpoint_host(runnable)``, or pass it to ``classify_failure(base_url=)``
    which stores only the parsed host.
    """
    for attr in ("openai_api_base", "base_url", "api_base"):
        value = getattr(runnable, attr, None)
        if value:
            return str(value)
    client = getattr(runnable, "root_client", None)
    value = getattr(client, "base_url", None)
    return str(value) if value else None


def get_endpoint_host(runnable: Any) -> str | None:
    """Log-safe endpoint host for a chat model — the single definition.

    ``FailureDetails.endpoint_host`` and every log site must agree, and neither
    may emit the credential- or path-bearing full URL.
    """
    return _url_host(get_base_url(runnable))


def get_model_name(runnable: Any) -> str | None:
    for attr in ("model_name", "model", "_default_model"):
        value = getattr(runnable, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def get_class_name(runnable: Any) -> str:
    return type(runnable).__name__


def classify_failure(
    exc: BaseException,
    *,
    provider: str | None = None,
    model_name: str | None = None,
    class_name: str | None = None,
    base_url: str | None = None,
) -> FailureDetails:
    root = _root_cause(exc)
    message = str(exc)
    root_message = str(root)
    combined = f"{type(exc).__name__}: {message}\n{type(root).__name__}: {root_message}".lower()
    statuses = set(_STATUS_CODE_PATTERN.findall(combined))

    derived_provider = infer_provider(
        model_name=model_name,
        class_name=class_name or type(exc).__module__,
    )
    final_provider = provider or derived_provider
    normalized_provider: ProviderName = (
        cast(ProviderName, final_provider)
        if final_provider in _PROVIDER_NAMES
        else "unknown"
    )
    host = _extract_host(message) or _extract_host(root_message)

    if isinstance(root, socket.gaierror) or any(
        marker in combined
        for marker in (
            "nodename nor servname provided",
            "temporary failure in name resolution",
            "name or service not known",
            "failed to resolve host",
            "could not resolve host",
        )
    ):
        kind: FailureKind = "dns_resolution"
        retryable = True
    elif any(
        marker in combined
        for marker in ("timed out", "timeout", "readtimeout", "connecttimeout")
    ):
        kind = "timeout"
        retryable = True
    elif type(root).__name__ in {"YFPricesMissingError", "YFTzMissingError"} or any(
        marker in combined
        for marker in (
            "possibly delisted",
            "no price data found",
            "no timezone found",
            # yfinance/urllib phrasing for a dead symbol; provider 404s use
            # different wording ("model not found", "is not found for api").
            "http error 404",
            "quote not found for symbol",
        )
    ):
        # Expected data absence (delisted/migrated tickers), not a system fault.
        kind = "data_unavailable"
        retryable = False
    elif isinstance(
        root,
        TypeError | AttributeError | ImportError | NotImplementedError | AssertionError,
    ):
        kind = "application_error"
        retryable = False
    elif "429" in statuses or any(
        marker in combined
        for marker in ("rate limit", "too many requests", "ratelimit")
    ):
        kind = "rate_limit"
        retryable = True
    elif "quota" in combined or "resourceexhausted" in combined:
        kind = "quota_error"
        retryable = True
    elif is_provider_content_block(combined):
        # Placed before auth/bad_request/server_error. A content-policy rejection
        # carries whatever status the vendor chose (400, 403, occasionally 500)
        # plus, frequently, an echo of the offending prompt span — so any of those
        # branches would swallow it. The distinction is load-bearing rather than
        # cosmetic: provider_safety_block is excluded from the circuit breakers
        # (a content block is not evidence the provider is unhealthy), so a
        # misfiled refusal fast-fails unrelated sibling agents.
        kind = "provider_safety_block"
        retryable = False
    elif statuses & {"401", "403"} or any(
        marker in combined
        for marker in (
            "unauthorized",
            "forbidden",
            "invalid api key",
            "authentication",
        )
    ):
        kind = "auth_error"
        retryable = False
    elif "provider_partial_response" in combined:
        kind = "provider_partial_response"
        retryable = True
    elif statuses & {"500", "502", "503", "504", "520", "521", "522", "523", "524"} or (
        # "internalservererror" catches the OpenAI SDK class name (raised for
        # any >=500, including Cloudflare 52x bodies that name no 50x code the
        # status match would find); 520-524 are Cloudflare origin errors.
        "internal server error" in combined or "internalservererror" in combined
    ):
        kind = "server_error"
        retryable = True
    elif any(
        marker in combined
        for marker in (
            "model not found",
            "is not found for api",
            "no such model",
        )
    ):
        kind = "model_not_found"
        retryable = False
    elif any(
        marker in combined
        for marker in (
            "clientpayloaderror",
            "transferencodingerror",
            "not enough data to satisfy transfer length header",
            "response payload is not completed",
        )
    ):
        kind = "connect_error"
        retryable = True
    elif "400" in statuses or any(
        marker in combined for marker in ("bad request", "invalid_request_error")
    ):
        kind = "bad_request"
        retryable = False
    elif any(
        marker in combined
        for marker in (
            "connection error",
            "connecterror",
            "cannot connect to host",
            "connection reset",
            "connection aborted",
            "remotedisconnected",
            "ssl",
            "certificate",
            "handshake",
            "eof occurred in violation",
            "broken pipe",
            "proxy",
        )
    ):
        kind = "connect_error"
        retryable = True
    else:
        kind = "unknown_provider_error"
        retryable = False

    return FailureDetails(
        kind=kind,
        provider=normalized_provider,
        host=host,
        error_type=type(exc).__name__,
        root_cause_type=type(root).__name__,
        retryable=retryable,
        message=redact_sensitive_text(message, max_chars=200),
        endpoint_host=_url_host(base_url),
    )
