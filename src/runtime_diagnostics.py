from __future__ import annotations

import re
import socket
from dataclasses import asdict, dataclass
from typing import Any, Literal

from src.data_block_utils import has_parseable_data_block

ProviderName = Literal["google", "openai", "anthropic", "unknown"]
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
    "unknown_provider_error",
]
ArtifactErrorKind = FailureKind | Literal["application_error"]

FUNDAMENTALS_SYNC_FIELDS = frozenset(
    {"raw_fundamentals_data", "foreign_language_report", "legal_report"}
)
SYNC_CHECK_FIELDS = frozenset(
    {
        "market_report",
        "sentiment_report",
        "news_report",
        "value_trap_report",
        "auditor_report",
    }
)
REQUIRED_PUBLISHABLE_ARTIFACTS = frozenset(
    {
        "market_report",
        "sentiment_report",
        "news_report",
        "value_trap_report",
        "fundamentals_report",
        "final_trade_decision",
    }
)
QUICK_REQUIRED_PUBLISHABLE_ARTIFACTS = REQUIRED_PUBLISHABLE_ARTIFACTS - frozenset(
    {"value_trap_report"}
)
OPTIONAL_PUBLISHABLE_ARTIFACTS = frozenset(
    {"auditor_report", "consultant_review", "valuation_params"}
)
QUICK_OPTIONAL_PUBLISHABLE_ARTIFACTS = OPTIONAL_PUBLISHABLE_ARTIFACTS | frozenset(
    {"value_trap_report"}
)


@dataclass(frozen=True)
class FailureDetails:
    kind: FailureKind
    provider: ProviderName
    host: str | None
    error_type: str
    root_cause_type: str
    retryable: bool
    message: str


@dataclass(frozen=True)
class ArtifactStatus:
    complete: bool
    ok: bool
    content: str | None
    error_kind: ArtifactErrorKind | None = None
    provider: str | None = None
    message: str | None = None
    retryable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    elif any(
        marker in combined
        for marker in ("500", "502", "503", "504", "internal server error")
    ):
        kind = "server_error"
        retryable = True
    elif any(
        marker in combined
        for marker in (
            "404",
            "not found",
            "model not found",
            "is not found for api",
            "no such model",
        )
    ):
        kind = "model_not_found"
        retryable = False
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
        provider=final_provider
        if final_provider in {"google", "openai", "anthropic"}
        else "unknown",
        host=host,
        error_type=type(exc).__name__,
        root_cause_type=type(root).__name__,
        retryable=retryable,
        message=message[:200],
    )


def success_artifact(
    field: str, content: str, *, provider: str | None = None
) -> dict[str, Any]:
    return {
        field: content,
        "artifact_statuses": {
            field: ArtifactStatus(
                complete=True,
                ok=True,
                content=content,
                provider=provider,
            ).as_dict()
        },
    }


def failure_artifact(
    field: str,
    exc: BaseException | str,
    *,
    provider: str | None = None,
    fallback_content: str = "",
    error_kind: ArtifactErrorKind | None = None,
) -> dict[str, Any]:
    if isinstance(exc, BaseException):
        details = classify_failure(exc, provider=provider)
        status = ArtifactStatus(
            complete=True,
            ok=False,
            content=fallback_content or None,
            error_kind=details.kind,
            provider=details.provider,
            message=details.message,
            retryable=details.retryable,
        )
    else:
        status = ArtifactStatus(
            complete=True,
            ok=False,
            content=fallback_content or None,
            error_kind=error_kind or "application_error",
            provider=provider,
            message=str(exc)[:400],
            retryable=False,
        )

    return {
        field: fallback_content,
        "artifact_statuses": {field: status.as_dict()},
    }


def get_artifact_status(state: dict[str, Any], field: str) -> ArtifactStatus:
    statuses = state.get("artifact_statuses", {}) or {}
    raw = statuses.get(field)
    if isinstance(raw, dict):
        return ArtifactStatus(
            complete=bool(raw.get("complete", True)),
            ok=bool(raw.get("ok")),
            content=raw.get("content"),
            error_kind=raw.get("error_kind"),
            provider=raw.get("provider"),
            message=raw.get("message"),
            retryable=bool(raw.get("retryable")),
        )

    content = state.get(field)
    normalized = content if isinstance(content, str) else None
    complete = bool(normalized)
    return ArtifactStatus(
        complete=complete,
        ok=bool(normalized),
        content=normalized,
    )


def is_artifact_complete(state: dict[str, Any], field: str) -> bool:
    return get_artifact_status(state, field).complete


def is_artifact_valid(state: dict[str, Any], field: str) -> bool:
    return get_artifact_status(state, field).ok


def get_valid_artifact_content(
    state: dict[str, Any], field: str, default: str = ""
) -> str:
    status = get_artifact_status(state, field)
    if not status.ok or not status.content:
        return default
    return status.content


def _is_quick_mode_result(result: dict[str, Any]) -> bool:
    run_summary = result.get("run_summary")
    if isinstance(run_summary, dict):
        return bool(run_summary.get("quick_mode", False))

    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        return bool(metadata.get("quick_mode", False))

    return bool(result.get("quick_mode", False))


def get_required_publishable_artifacts(result: dict[str, Any]) -> frozenset[str]:
    return (
        QUICK_REQUIRED_PUBLISHABLE_ARTIFACTS
        if _is_quick_mode_result(result)
        else REQUIRED_PUBLISHABLE_ARTIFACTS
    )


def get_optional_publishable_artifacts(result: dict[str, Any]) -> frozenset[str]:
    return (
        QUICK_OPTIONAL_PUBLISHABLE_ARTIFACTS
        if _is_quick_mode_result(result)
        else OPTIONAL_PUBLISHABLE_ARTIFACTS
    )


def build_analysis_validity(result: dict[str, Any]) -> dict[str, Any]:
    fundamentals = get_valid_artifact_content(result, "fundamentals_report")
    pm_decision = get_valid_artifact_content(result, "final_trade_decision")
    data_block_present = has_parseable_data_block(fundamentals)
    required_failures: dict[str, Any] = {}
    optional_failures: dict[str, Any] = {}
    required_artifacts = get_required_publishable_artifacts(result)
    optional_artifacts = get_optional_publishable_artifacts(result)

    for field in required_artifacts:
        status = get_artifact_status(result, field)
        if not status.ok:
            required_failures[field] = status.as_dict()

    for field in optional_artifacts:
        status = get_artifact_status(result, field)
        if status.complete and not status.ok:
            optional_failures[field] = status.as_dict()

    pre_screening = result.get("pre_screening_result")
    has_valid_pre_screening = pre_screening in {"PASS", "REJECT"}
    if not has_valid_pre_screening:
        required_failures["pre_screening_result"] = {
            "complete": bool(pre_screening),
            "ok": False,
            "content": pre_screening if isinstance(pre_screening, str) else None,
            "error_kind": "application_error",
            "provider": "unknown",
            "message": "Pre-screening result missing or invalid",
            "retryable": False,
        }

    publishable = bool(
        pm_decision
        and fundamentals
        and data_block_present
        and has_valid_pre_screening
        and not required_failures
    )
    return {
        "publishable": publishable,
        "has_valid_pm_decision": bool(pm_decision),
        "has_valid_fundamentals": bool(fundamentals),
        "has_data_block": data_block_present,
        "has_valid_pre_screening": has_valid_pre_screening,
        "required_artifacts": sorted(required_artifacts),
        "optional_artifacts": sorted(optional_artifacts),
        "required_failures": required_failures,
        "optional_failures": optional_failures,
        # Backward-compatible alias for older callers/tests.
        "fatal_failures": required_failures,
    }


def is_publishable_analysis(result: dict[str, Any]) -> bool:
    validity = result.get("analysis_validity")
    if isinstance(validity, dict) and "publishable" in validity:
        return bool(validity["publishable"])
    return bool(build_analysis_validity(result)["publishable"])
