"""Normalize provider response metadata before generic runtime decisions."""

from dataclasses import dataclass
from typing import Any

from src.agents.message_utils import extract_string_content


@dataclass(frozen=True)
class NormalizedUsage:
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class NormalizedResponse:
    text: str
    finish_reason: str | None
    truncated: bool
    refusal_kind: str | None
    tool_calls: tuple[Any, ...]
    usage: NormalizedUsage | None
    provider: str | None
    model: str | None


def normalize_response(response: Any) -> NormalizedResponse:
    metadata = getattr(response, "response_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    status = metadata.get("status")
    details = metadata.get("incomplete_details")
    incomplete_reason = details.get("reason") if isinstance(details, dict) else None
    finish = (
        metadata.get("finish_reason")
        or metadata.get("stop_reason")
        or metadata.get("done_reason")
        or (incomplete_reason if status == "incomplete" else status)
    )
    finish_text = str(finish) if finish is not None else None
    truncated = finish_text in {
        "length",
        "max_tokens",
        "MAX_TOKENS",
        "max_output_tokens",
    }
    refusal = None
    additional = getattr(response, "additional_kwargs", None)
    if isinstance(additional, dict) and additional.get("refusal"):
        refusal = "refusal"
    if finish_text in {"content_filter", "SAFETY", "refusal"}:
        refusal = finish_text.lower()
    usage_data = getattr(response, "usage_metadata", None)
    usage = None
    if isinstance(usage_data, dict):
        usage = NormalizedUsage(
            input_tokens=_optional_int(usage_data.get("input_tokens")),
            output_tokens=_optional_int(usage_data.get("output_tokens")),
        )
    tool_calls = getattr(response, "tool_calls", None)
    return NormalizedResponse(
        text=extract_string_content(getattr(response, "content", "")),
        finish_reason=finish_text,
        truncated=truncated,
        refusal_kind=refusal,
        tool_calls=tuple(tool_calls) if isinstance(tool_calls, list) else (),
        usage=usage,
        provider=_optional_str(metadata.get("provider")),
        model=_optional_str(metadata.get("model_name") or metadata.get("model")),
    )


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None
