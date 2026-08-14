"""Provider-selected message preparation with seat isolation preserved."""

from typing import Any

from langchain_core.messages import BaseMessage

from src.agents.message_utils import (
    filter_messages_by_agent,
    filter_messages_for_gemini,
)


def adapter_kind_for_model(llm: Any) -> str:
    stamped = getattr(llm, "_llm_adapter_kind", None)
    if isinstance(stamped, str):
        return stamped
    qualified = f"{type(llm).__module__}.{type(llm).__name__}".lower()
    if "google" in qualified or "gemini" in qualified:
        return "google_native"
    if "anthropic" in qualified or "claude" in qualified:
        return "anthropic_native"
    if "openai" in qualified:
        return "openai_native"
    return "unknown"


def prepare_messages_for_model(
    llm: Any, messages: list[BaseMessage], *, agent_key: str
) -> list[BaseMessage]:
    """Apply agent isolation to all providers and transport cleanup only to Google."""

    if adapter_kind_for_model(llm) == "google_native":
        return filter_messages_for_gemini(messages, agent_key=agent_key)
    return filter_messages_by_agent(messages, agent_key)
