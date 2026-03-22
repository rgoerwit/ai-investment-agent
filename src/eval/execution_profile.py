from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def summarize_agent_llm_profile(
    llm_ids: list[int],
    llm_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the execution profile for one agent from linked LLM call rows."""
    rows = [llm_index[llm_id] for llm_id in llm_ids if llm_id in llm_index]
    success_rows = [row for row in rows if row.get("status") == "success"]

    primary = success_rows[-1] if success_rows else None
    last = rows[-1] if rows else None

    models = _ordered_unique(
        (row.get("response_model") or row.get("model")) for row in rows
    )
    providers = _ordered_unique(row.get("provider") for row in rows)
    reasoning = _ordered_unique(row.get("reasoning_level") for row in rows)

    return {
        "effective_models": models,
        "effective_providers": providers,
        "effective_reasoning_levels": reasoning,
        "primary_model": None
        if primary is None
        else (primary.get("response_model") or primary.get("model")),
        "primary_provider": None if primary is None else primary.get("provider"),
        "primary_reasoning_level": None
        if primary is None
        else primary.get("reasoning_level"),
        "primary_status": None if primary is None else "success",
        "last_attempt_model": None
        if last is None
        else (last.get("response_model") or last.get("model")),
        "last_attempt_provider": None if last is None else last.get("provider"),
        "last_attempt_reasoning_level": None
        if last is None
        else last.get("reasoning_level"),
        "last_attempt_status": None if last is None else last.get("status"),
        "llm_call_count": len(rows),
    }
