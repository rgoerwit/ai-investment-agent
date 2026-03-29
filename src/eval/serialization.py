from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)
_SERIALIZATION_FALLBACK_MARKER = "__serialization_fallback__"
_LOGGED_SERIALIZATION_FALLBACK_TYPES: set[str] = set()


def _normalize_message(value: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = {"type": type(value).__name__}
    content = getattr(value, "content", None)
    if content is not None:
        normalized["content"] = normalize_for_json(content)
    name = getattr(value, "name", None)
    if name:
        normalized["name"] = name
    tool_calls = getattr(value, "tool_calls", None)
    if tool_calls:
        normalized["tool_calls"] = normalize_for_json(tool_calls)
    additional_kwargs = getattr(value, "additional_kwargs", None)
    if additional_kwargs:
        normalized["additional_kwargs"] = normalize_for_json(additional_kwargs)
    response_metadata = getattr(value, "response_metadata", None)
    if response_metadata:
        normalized["response_metadata"] = normalize_for_json(response_metadata)
    return normalized


def normalize_for_json(value: Any) -> Any:
    """Convert runtime values into JSON-safe structures."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(normalize_for_json(v) for v in value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [normalize_for_json(v) for v in value]
    if hasattr(value, "content") and type(value).__name__.endswith("Message"):
        return _normalize_message(value)
    if hasattr(value, "__dict__"):
        return {
            "__class__": type(value).__name__,
            **normalize_for_json(vars(value)),
        }
    fallback_type = type(value).__name__
    if fallback_type not in _LOGGED_SERIALIZATION_FALLBACK_TYPES:
        logger.warning("serialization_fallback", value_type=fallback_type)
        _LOGGED_SERIALIZATION_FALLBACK_TYPES.add(fallback_type)
    return {
        _SERIALIZATION_FALLBACK_MARKER: True,
        "type": fallback_type,
        "repr": repr(value),
    }


def contains_serialization_fallback(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get(_SERIALIZATION_FALLBACK_MARKER) is True:
            return True
        return any(contains_serialization_fallback(v) for v in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(contains_serialization_fallback(v) for v in value)
    return False
