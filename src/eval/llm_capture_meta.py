from __future__ import annotations

from typing import Any

from src.llm_usage import extract_token_usage_breakdown


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
    usage = extract_token_usage_breakdown(result)
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.total_output_tokens,
        "thinking_tokens": usage.thinking_tokens,
        "total_tokens": usage.total_tokens,
    }
