from __future__ import annotations

import re
import socket
from dataclasses import dataclass
from typing import Any, Literal, cast

from src.error_safety import redact_sensitive_text

ProviderName = Literal["google", "openai", "anthropic", "deepseek", "unknown"]
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


_HOST_PATTERN = re.compile(r"(?:host|https?://)([A-Za-z0-9.-]+\.[A-Za-z]{2,})")


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


def _extract_host(message: str) -> str | None:
    match = _HOST_PATTERN.search(message)
    if not match:
        return None
    host = match.group(1)
    return host.rstrip(":/")


def infer_provider(
    model_name: str | None = None, class_name: str | None = None
) -> ProviderName:
    haystack = " ".join(part for part in (model_name, class_name) if part).lower()
    if "gemini" in haystack or "google" in haystack:
        return "google"
    if "deepseek" in haystack:
        return "deepseek"
    if "gpt" in haystack or "openai" in haystack:
        return "openai"
    if "claude" in haystack or "anthropic" in haystack:
        return "anthropic"
    return "unknown"


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
) -> FailureDetails:
    root = _root_cause(exc)
    message = str(exc)
    root_message = str(root)
    combined = f"{type(exc).__name__}: {message}\n{type(root).__name__}: {root_message}".lower()

    derived_provider = infer_provider(
        model_name=model_name,
        class_name=class_name or type(exc).__module__,
    )
    final_provider = provider or derived_provider
    normalized_provider: ProviderName = (
        cast(ProviderName, final_provider)
        if final_provider in {"google", "openai", "anthropic", "deepseek", "unknown"}
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
    elif any(
        marker in combined
        for marker in ("429", "rate limit", "too many requests", "ratelimit")
    ):
        kind = "rate_limit"
        retryable = True
    elif "quota" in combined or "resourceexhausted" in combined:
        kind = "quota_error"
        retryable = True
    elif any(
        marker in combined
        for marker in (
            "401",
            "403",
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
    elif any(
        marker in combined
        # "internalservererror" catches the OpenAI SDK class name (raised for
        # any >=500, including Cloudflare 52x bodies that name no 50x code the
        # spaced marker would match); 520-524 are Cloudflare origin errors.
        for marker in (
            "500",
            "502",
            "503",
            "504",
            "520",
            "521",
            "522",
            "523",
            "524",
            "internal server error",
            "internalservererror",
        )
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
    elif any(
        marker in combined for marker in ("400", "bad request", "invalid_request_error")
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
    )
