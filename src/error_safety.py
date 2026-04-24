from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_DEFAULT_PREVIEW_CHARS = 32
_MAX_LIST_ITEMS = 10
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)(api[_-]?key|api[_-]?token|access[_-]?token|refresh[_-]?token|token|secret|password|authorization|bearer|cookie|client[_-]?secret)"
)
_INLINE_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|api[_-]?token|access[_-]?token|refresh[_-]?token|token|secret|password|authorization|cookie|client[_-]?secret)\b\s*[:=]\s*([^\s,;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._\-+/=]+)")
_KNOWN_SECRET_VALUE_PATTERN = re.compile(
    r"\b(?:AIza[0-9A-Za-z\-_]{20,}|sk-[A-Za-z0-9]{16,}|pk-[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,}|ya29\.[A-Za-z0-9._\-]+)\b"
)
_HIGH_ENTROPY_VALUE_PATTERN = re.compile(
    r"\b(?=[A-Za-z0-9._\-/+=]{24,}\b)(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9._\-/+=]+\b"
)
_PATH_HINT_PATTERN = re.compile(
    r"[/\\]|\.env\b|\.json\b|\.sqlite\b|\.md\b", re.IGNORECASE
)


def is_sensitive_key(key: str) -> bool:
    return bool(_SENSITIVE_KEY_PATTERN.search(key))


def _truncate(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _redact_url_query_values(text: str) -> str:
    if "://" not in text:
        return text
    try:
        parsed = urlsplit(text)
        if not parsed.query:
            return text
        redacted_query = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            if is_sensitive_key(key):
                redacted_query.append((key, "[REDACTED]"))
            else:
                redacted_query.append((key, value))
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(redacted_query, doseq=True),
                parsed.fragment,
            )
        )
    except Exception:
        return text


def redact_sensitive_text(text: str, *, max_chars: int = _DEFAULT_PREVIEW_CHARS) -> str:
    if not text:
        return ""

    redacted = _redact_url_query_values(str(text))
    redacted = _INLINE_SECRET_PATTERN.sub(r"\1=[REDACTED]", redacted)
    redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", redacted)
    redacted = _KNOWN_SECRET_VALUE_PATTERN.sub("[REDACTED]", redacted)
    redacted = _HIGH_ENTROPY_VALUE_PATTERN.sub("[REDACTED]", redacted)
    return _truncate(redacted, max_chars)


def _sanitize_scalar(value: Any, *, max_chars: int) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value
    if isinstance(value, Path):
        return "[path]"

    text = str(value)
    if _PATH_HINT_PATTERN.search(text):
        return "[path]"
    return redact_sensitive_text(text, max_chars=max_chars)


def safe_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    allowlist: Iterable[str] | None = None,
    max_chars: int = _DEFAULT_PREVIEW_CHARS,
) -> dict[str, Any]:
    if not metadata:
        return {}

    allowed = set(allowlist or ())
    use_allowlist = bool(allowed)
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if use_allowlist and key not in allowed:
            continue
        if is_sensitive_key(key) or value is None:
            continue
        if isinstance(value, Mapping):
            nested = safe_metadata(value, max_chars=max_chars)
            if nested:
                sanitized[key] = nested
            continue
        if isinstance(value, list | tuple | set):
            items: list[Any] = []
            for item in list(value)[:_MAX_LIST_ITEMS]:
                if isinstance(item, Mapping):
                    nested = safe_metadata(item, max_chars=max_chars)
                    if nested:
                        items.append(nested)
                else:
                    items.append(_sanitize_scalar(item, max_chars=max_chars))
            if items:
                sanitized[key] = items
            continue
        sanitized[key] = _sanitize_scalar(value, max_chars=max_chars)
    return sanitized


def safe_trace_input(
    payload: Mapping[str, Any] | None,
    *,
    allowlist: Iterable[str] | None = None,
    max_chars: int = _DEFAULT_PREVIEW_CHARS,
) -> dict[str, Any]:
    return safe_metadata(payload, allowlist=allowlist, max_chars=max_chars)


def format_error_message(
    *,
    operation: str,
    error_type: str,
    message_preview: str | None = None,
) -> str:
    message = f"Error in {operation}: {error_type}"
    if message_preview:
        message += f" (preview: {message_preview})"
    return message


def summarize_exception(
    exc: BaseException,
    *,
    operation: str,
    provider: str | None = None,
    preview_chars: int = _DEFAULT_PREVIEW_CHARS,
) -> dict[str, Any]:
    from src.runtime_diagnostics import classify_failure

    details = classify_failure(exc, provider=provider)
    preview = redact_sensitive_text(details.message, max_chars=preview_chars)
    summary = {
        "operation": operation,
        "error_type": details.error_type,
        "root_cause_type": details.root_cause_type,
        "failure_kind": details.kind,
        "retryable": details.retryable,
        "host": details.host,
        "message_preview": preview or None,
    }
    return summary


def safe_error_payload(
    exc: BaseException,
    *,
    operation: str,
    provider: str | None = None,
    extra: Mapping[str, Any] | None = None,
    preview_chars: int = _DEFAULT_PREVIEW_CHARS,
) -> dict[str, Any]:
    summary = summarize_exception(
        exc,
        operation=operation,
        provider=provider,
        preview_chars=preview_chars,
    )
    payload: dict[str, Any] = {
        "error": format_error_message(
            operation=operation,
            error_type=summary["error_type"],
            message_preview=summary["message_preview"],
        ),
        "error_type": summary["error_type"],
        "failure_kind": summary["failure_kind"],
        "retryable": summary["retryable"],
    }
    if summary["message_preview"]:
        payload["message_preview"] = summary["message_preview"]
    if summary["host"]:
        payload["host"] = summary["host"]
    if extra:
        payload.update(extra)
    return payload
