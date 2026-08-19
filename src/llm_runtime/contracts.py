"""Stable construction semantics owned by the application.

These contracts deliberately omit arbitrary LangChain internals. They capture the
derived values whose accidental loss changes output quality, transport behavior,
or accounting without necessarily preventing a client from being constructed.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConstructionContract:
    """Normalized, provider-independent description of a constructed chat model."""

    seat_id: str
    model: str
    transport_class: str
    intent_output_cap_tokens: int | None
    api_output_cap_tokens: int | None
    configured_reasoning_reserve_tokens: int
    reasoning_intent: str | None
    service_tier: str | None
    use_responses_api: bool
    extra_body: Mapping[str, object] | None
    timeout_seconds: float | None
    max_retries: int | None
    limiter_key: tuple[str, str | None] | None
    callback_agent: str


def _first_attr(value: Any, *names: str) -> Any:
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def capture_construction_contract(
    llm: Any,
    *,
    seat_id: str,
    callback_agent: str,
    reasoning_intent: str | None = None,
    limiter_key: tuple[str, str | None] | None = None,
) -> ConstructionContract:
    """Capture curated construction semantics without serializing SDK internals."""

    model = _first_attr(llm, "model_name", "model")
    intent_cap = _first_attr(
        llm, "_configured_max_output_tokens", "_configured_max_completion_tokens"
    )
    api_cap = _first_attr(
        llm, "_configured_api_output_tokens", "_configured_api_completion_tokens"
    )
    timeout = _first_attr(llm, "timeout", "request_timeout")
    max_retries = getattr(llm, "max_retries", None)
    extra_body = getattr(llm, "extra_body", None)
    if reasoning_intent is None:
        reasoning_intent = _first_attr(llm, "reasoning_effort", "thinking_level")
    return ConstructionContract(
        seat_id=seat_id,
        model=str(model or ""),
        transport_class=f"{type(llm).__module__}.{type(llm).__qualname__}",
        intent_output_cap_tokens=int(intent_cap) if intent_cap is not None else None,
        api_output_cap_tokens=int(api_cap) if api_cap is not None else None,
        configured_reasoning_reserve_tokens=int(
            getattr(llm, "_configured_reasoning_reserve_tokens", 0)
        ),
        reasoning_intent=reasoning_intent,
        service_tier=getattr(llm, "service_tier", None),
        use_responses_api=bool(getattr(llm, "use_responses_api", False)),
        extra_body=dict(extra_body) if isinstance(extra_body, Mapping) else None,
        timeout_seconds=float(timeout) if timeout is not None else None,
        max_retries=int(max_retries) if max_retries is not None else None,
        limiter_key=limiter_key,
        callback_agent=callback_agent,
    )
