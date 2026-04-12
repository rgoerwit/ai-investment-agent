from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class TokenUsageBreakdown:
    input_tokens: int | None
    total_output_tokens: int | None
    thinking_tokens: int | None
    visible_output_tokens: int | None
    total_tokens: int | None


def _is_mockish(value: Any) -> bool:
    return type(value).__module__.startswith("unittest.mock")


def _coerce_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _with_visible_output(
    *,
    input_tokens: int | None,
    total_output_tokens: int | None,
    thinking_tokens: int | None,
    total_tokens: int | None,
) -> TokenUsageBreakdown:
    visible_output_tokens = None
    if total_output_tokens is not None and thinking_tokens is not None:
        visible_output_tokens = max(total_output_tokens - thinking_tokens, 0)
    return TokenUsageBreakdown(
        input_tokens=input_tokens,
        total_output_tokens=total_output_tokens,
        thinking_tokens=thinking_tokens,
        visible_output_tokens=visible_output_tokens,
        total_tokens=total_tokens,
    )


def _metadata_get(container: Any, key: str) -> Any:
    if isinstance(container, dict):
        return container.get(key)
    if container is None or callable(container) or _is_mockish(container):
        return None
    value = getattr(container, key, None)
    if callable(value) or _is_mockish(value):
        return None
    return value


def _extract_from_usage_container(usage: Any) -> TokenUsageBreakdown | None:
    if usage is None or callable(usage) or _is_mockish(usage):
        return None

    output_details = _metadata_get(usage, "output_token_details") or {}
    input_tokens = _coerce_optional_int(
        _metadata_get(usage, "input_tokens") or _metadata_get(usage, "prompt_tokens")
    )
    total_output_tokens = _coerce_optional_int(
        _metadata_get(usage, "output_tokens")
        or _metadata_get(usage, "completion_tokens")
        or _metadata_get(usage, "total_output_tokens")
    )
    thinking_tokens = _coerce_optional_int(_metadata_get(output_details, "reasoning"))
    total_tokens = _coerce_optional_int(_metadata_get(usage, "total_tokens"))
    if (
        input_tokens is None
        and total_output_tokens is None
        and thinking_tokens is None
        and total_tokens is None
    ):
        return None
    if (
        total_tokens is None
        and input_tokens is not None
        and total_output_tokens is not None
    ):
        total_tokens = input_tokens + total_output_tokens
    return _with_visible_output(
        input_tokens=input_tokens,
        total_output_tokens=total_output_tokens,
        thinking_tokens=thinking_tokens,
        total_tokens=total_tokens,
    )


def _extract_from_token_usage_container(token_usage: Any) -> TokenUsageBreakdown | None:
    if token_usage is None or callable(token_usage) or _is_mockish(token_usage):
        return None

    completion_details = _metadata_get(token_usage, "completion_tokens_details") or {}
    input_tokens = _coerce_optional_int(
        _metadata_get(token_usage, "input_tokens")
        or _metadata_get(token_usage, "prompt_tokens")
    )
    total_output_tokens = _coerce_optional_int(
        _metadata_get(token_usage, "output_tokens")
        or _metadata_get(token_usage, "completion_tokens")
    )
    thinking_tokens = _coerce_optional_int(
        _metadata_get(completion_details, "reasoning_tokens")
    )
    total_tokens = _coerce_optional_int(_metadata_get(token_usage, "total_tokens"))
    if (
        input_tokens is None
        and total_output_tokens is None
        and thinking_tokens is None
        and total_tokens is None
    ):
        return None
    if (
        total_tokens is None
        and input_tokens is not None
        and total_output_tokens is not None
    ):
        total_tokens = input_tokens + total_output_tokens
    return _with_visible_output(
        input_tokens=input_tokens,
        total_output_tokens=total_output_tokens,
        thinking_tokens=thinking_tokens,
        total_tokens=total_tokens,
    )


def _extract_direct_metadata(value: Any) -> TokenUsageBreakdown | None:
    usage_metadata = getattr(value, "usage_metadata", None)
    usage = _extract_from_usage_container(usage_metadata)
    if usage is not None:
        return usage

    response_metadata = getattr(value, "response_metadata", None)
    token_usage = _metadata_get(response_metadata, "token_usage")
    parsed_token_usage = _extract_from_token_usage_container(token_usage)
    if parsed_token_usage is not None:
        return parsed_token_usage

    response_usage = _metadata_get(response_metadata, "usage")
    parsed_usage = _extract_from_usage_container(response_usage)
    if parsed_usage is not None:
        return parsed_usage

    return None


def _extract_from_llm_result(value: Any) -> TokenUsageBreakdown | None:
    generations = getattr(value, "generations", None)
    if (
        isinstance(generations, Sequence)
        and not isinstance(generations, str | bytes)
        and generations
    ):
        first_generation_list = generations[0]
        if isinstance(first_generation_list, Sequence) and first_generation_list:
            first_generation = first_generation_list[0]
            message = getattr(first_generation, "message", None)
            if message is not None:
                direct = _extract_direct_metadata(message)
                if direct is not None:
                    return direct

    llm_output = getattr(value, "llm_output", None)
    usage_metadata = _metadata_get(llm_output, "usage_metadata")
    usage = _extract_from_usage_container(usage_metadata)
    if usage is not None:
        return usage

    token_usage = _metadata_get(llm_output, "token_usage")
    parsed_token_usage = _extract_from_token_usage_container(token_usage)
    if parsed_token_usage is not None:
        return parsed_token_usage

    return None


def extract_token_usage_breakdown(value: Any) -> TokenUsageBreakdown:
    if value is None:
        return TokenUsageBreakdown(None, None, None, None, None)

    direct = _extract_direct_metadata(value)
    if direct is not None:
        return direct

    llm_result = _extract_from_llm_result(value)
    if llm_result is not None:
        return llm_result

    return TokenUsageBreakdown(None, None, None, None, None)
