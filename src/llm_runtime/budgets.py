"""Provider-neutral generation-budget derivation and construction metadata."""

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel

from src.llm_budgets import GenerationBudget, get_generation_budget


def resolve_generation_budget(
    settings: Any,
    *,
    intent_tokens: int,
    reasoning_value: str | None,
) -> GenerationBudget:
    """Keep visible-output intent separate from a provider's reasoning reserve."""

    reserve_class: Literal["default", "deep"] = (
        "deep" if reasoning_value in {"high", "xhigh", "max"} else "default"
    )
    return get_generation_budget(
        intent_tokens=intent_tokens,
        reserve_class=reserve_class,
        reserve_enabled=reasoning_value is not None,
        default_reserve_tokens=int(settings.llm_default_reasoning_reserve_tokens),
        deep_reserve_tokens=int(settings.llm_deep_reasoning_reserve_tokens),
    )


def stamp_budget_metadata(
    llm: BaseChatModel,
    *,
    callbacks: Sequence[BaseCallbackHandler],
    budget: GenerationBudget,
    intent_attr: str,
    api_attr: str,
) -> None:
    """Expose application-owned caps to accounting and parity contracts."""

    setattr(llm, intent_attr, budget.intent_tokens)
    setattr(llm, api_attr, budget.api_cap_tokens)
    llm._configured_reasoning_reserve_tokens = budget.reserve_tokens  # type: ignore[attr-defined]
    for callback in callbacks:
        if hasattr(callback, "output_token_cap"):
            callback.output_token_cap = budget.intent_tokens  # type: ignore[attr-defined]
            callback.api_output_token_cap = budget.api_cap_tokens  # type: ignore[attr-defined]
            callback.reasoning_reserve_tokens = budget.reserve_tokens  # type: ignore[attr-defined]
