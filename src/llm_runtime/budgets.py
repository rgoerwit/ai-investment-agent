"""Provider-neutral generation-budget derivation and construction metadata."""

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel

from src.llm_budgets import GenerationBudget, get_generation_budget
from src.llm_runtime.seats import ModelIntent

_DEEP_RESERVE_INTENTS = frozenset(
    {ModelIntent.REASONING, ModelIntent.CRITICAL, ModelIntent.ESCALATION}
)
_DEEP_REASONING_VALUES = frozenset({"high", "xhigh", "max"})


def reserve_class_for_request(
    intent: ModelIntent | None,
    reasoning_value: str | None,
) -> Literal["default", "deep"]:
    """Choose one provider-neutral reasoning-reserve class for a request.

    Intent is authoritative for seats whose contract is inherently reasoning-heavy.
    The resolved reasoning value remains a fallback for legacy and direct factories
    that do not carry a canonical seat intent.
    """

    if intent in _DEEP_RESERVE_INTENTS or reasoning_value in _DEEP_REASONING_VALUES:
        return "deep"
    return "default"


def resolve_generation_budget(
    settings: Any,
    *,
    intent_tokens: int,
    reasoning_value: str | None,
    intent: ModelIntent | None = None,
) -> GenerationBudget:
    """Keep visible-output intent separate from a provider's reasoning reserve."""

    return get_generation_budget(
        intent_tokens=intent_tokens,
        reserve_class=reserve_class_for_request(intent, reasoning_value),
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
