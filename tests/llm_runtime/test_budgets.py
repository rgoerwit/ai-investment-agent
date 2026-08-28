from src.config import Settings
from src.llm_runtime.budgets import (
    reserve_class_for_request,
    resolve_generation_budget,
)
from src.llm_runtime.seats import ModelIntent


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_default_reasoning_reserve_tokens=2_048,
        llm_deep_reasoning_reserve_tokens=8_192,
    )


def test_critical_medium_request_uses_deep_reserve() -> None:
    budget = resolve_generation_budget(
        _settings(),
        intent_tokens=10_923,
        reasoning_value="medium",
        intent=ModelIntent.CRITICAL,
    )

    assert budget.intent_tokens == 10_923
    assert budget.reserve_tokens == 8_192
    assert budget.api_cap_tokens == 19_115


def test_fast_medium_request_keeps_default_reserve() -> None:
    budget = resolve_generation_budget(
        _settings(),
        intent_tokens=4_096,
        reasoning_value="medium",
        intent=ModelIntent.FAST,
    )

    assert budget.reserve_tokens == 2_048
    assert budget.api_cap_tokens == 6_144


def test_reasoning_disabled_request_has_no_reserve_even_for_critical_intent() -> None:
    budget = resolve_generation_budget(
        _settings(),
        intent_tokens=4_096,
        reasoning_value=None,
        intent=ModelIntent.CRITICAL,
    )

    assert reserve_class_for_request(ModelIntent.CRITICAL, None) == "deep"
    assert budget.reserve_tokens == 0
    assert budget.api_cap_tokens == 4_096


def test_legacy_high_effort_still_uses_deep_reserve_without_intent() -> None:
    assert reserve_class_for_request(None, "high") == "deep"
