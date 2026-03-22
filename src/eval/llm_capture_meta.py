from __future__ import annotations

from typing import Any


def normalize_reasoning_level(runnable: Any, model_name: str | None) -> str | None:
    """Normalize provider-specific reasoning configuration into coarse buckets."""
    model_kwargs = getattr(runnable, "model_kwargs", None) or {}

    raw_level = getattr(runnable, "thinking_level", None) or model_kwargs.get(
        "thinking_level"
    )
    if raw_level:
        value = str(raw_level).lower()
        return value if value in {"low", "medium", "high"} else "adaptive"

    raw_effort = getattr(runnable, "reasoning_effort", None) or model_kwargs.get(
        "reasoning_effort"
    )
    if raw_effort:
        value = str(raw_effort).lower()
        return value if value in {"low", "medium", "high"} else "adaptive"

    raw_thinking = getattr(runnable, "thinking", None) or model_kwargs.get("thinking")
    if raw_thinking:
        if isinstance(raw_thinking, dict) and raw_thinking.get("type") == "adaptive":
            return "adaptive"
        return "high"

    if model_name and "thinking" in model_name.lower():
        return "adaptive"

    return None


def extract_vendor_reasoning_config(
    runnable: Any, provider: str | None
) -> dict[str, Any] | None:
    """Capture the raw provider-specific reasoning/thinking config for provenance."""
    model_kwargs = getattr(runnable, "model_kwargs", None) or {}

    thinking_level = getattr(runnable, "thinking_level", None) or model_kwargs.get(
        "thinking_level"
    )
    if thinking_level is not None:
        return {
            "provider": provider,
            "name": "thinking_level",
            "value": thinking_level,
        }

    reasoning_effort = getattr(runnable, "reasoning_effort", None) or model_kwargs.get(
        "reasoning_effort"
    )
    if reasoning_effort is not None:
        return {
            "provider": provider,
            "name": "reasoning_effort",
            "value": reasoning_effort,
        }

    thinking = getattr(runnable, "thinking", None) or model_kwargs.get("thinking")
    if thinking is not None:
        return {
            "provider": provider,
            "name": "thinking",
            "value": thinking,
        }

    return None


def extract_token_usage(result: Any) -> dict[str, int | None]:
    """Best-effort token usage extraction across LangChain provider wrappers."""
    usage = getattr(result, "usage_metadata", None)
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        output_details = usage.get("output_token_details") or {}
        thinking_tokens = output_details.get("reasoning")
        return {
            "input_tokens": _to_int(input_tokens),
            "output_tokens": _to_int(output_tokens),
            "thinking_tokens": _to_int(thinking_tokens),
            "total_tokens": _to_int(total_tokens),
        }

    response_metadata = getattr(result, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            input_tokens = token_usage.get("input_tokens") or token_usage.get(
                "prompt_tokens"
            )
            output_tokens = token_usage.get("output_tokens") or token_usage.get(
                "completion_tokens"
            )
            completion_details = token_usage.get("completion_tokens_details") or {}
            thinking_tokens = completion_details.get("reasoning_tokens")
            return {
                "input_tokens": _to_int(input_tokens),
                "output_tokens": _to_int(output_tokens),
                "thinking_tokens": _to_int(thinking_tokens),
                "total_tokens": _to_int(token_usage.get("total_tokens")),
            }

        usage = response_metadata.get("usage")
        if isinstance(usage, dict):
            return {
                "input_tokens": _to_int(usage.get("input_tokens")),
                "output_tokens": _to_int(usage.get("output_tokens")),
                "thinking_tokens": None,
                "total_tokens": _to_int(usage.get("input_tokens"))
                + _to_int(usage.get("output_tokens"))
                if _to_int(usage.get("input_tokens")) is not None
                and _to_int(usage.get("output_tokens")) is not None
                else None,
            }

    return {
        "input_tokens": None,
        "output_tokens": None,
        "thinking_tokens": None,
        "total_tokens": None,
    }


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
