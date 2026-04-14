from __future__ import annotations

from typing import Any

import structlog
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = structlog.get_logger(__name__)


def filter_messages_by_agent(
    messages: list[BaseMessage], agent_key: str
) -> list[BaseMessage]:
    """
    Filter messages to only include this agent's conversation history.
    """
    if not messages:
        return []

    tool_msg_agents = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tag = (
                msg.additional_kwargs.get("agent_key")
                if msg.additional_kwargs
                else None
            )
            tool_msg_agents.append(tag)
    logger.debug(
        "filter_messages_tool_tags",
        agent_key=agent_key,
        total_tool_messages=len(tool_msg_agents),
        tool_message_tags=tool_msg_agents,
    )

    filtered = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered.append(msg)
        elif isinstance(msg, AIMessage):
            if getattr(msg, "name", None) == agent_key:
                filtered.append(msg)
        elif isinstance(msg, ToolMessage):
            msg_agent = (
                msg.additional_kwargs.get("agent_key")
                if msg.additional_kwargs
                else None
            )
            if msg_agent == agent_key:
                filtered.append(msg)

    return filtered


def filter_messages_for_gemini(
    messages: list[BaseMessage], agent_key: str | None = None
) -> list[BaseMessage]:
    """
    Filter and format messages for Gemini API compatibility.
    """
    if agent_key:
        messages = filter_messages_by_agent(messages, agent_key)

    if not messages:
        return []

    filtered = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        is_consecutive_human = (
            filtered
            and isinstance(msg, HumanMessage)
            and isinstance(filtered[-1], HumanMessage)
        )
        if is_consecutive_human:
            last_msg = filtered.pop()
            new_content = f"{last_msg.content}\n\n{msg.content}"
            filtered.append(HumanMessage(content=new_content))
        else:
            filtered.append(msg)
    return filtered


def extract_string_content(content: Any) -> str:
    """
    Safely extract string content from LLM response.content.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return extract_string_content(content["content"])
        if "parts" in content:
            parts = content["parts"]
            if isinstance(parts, list):
                text_parts = [extract_string_content(p) for p in parts]
                return "\n".join(filter(None, text_parts))
        # Typed non-text blocks (e.g. OpenAI reasoning summary) — no textual payload
        if content.get("type") in ("reasoning",):
            return ""
        logger.debug("response_content_is_dict", keys=list(content.keys()))
        return str(content)

    if isinstance(content, list):
        if len(content) == 0:
            return ""
        if len(content) == 1:
            return extract_string_content(content[0])
        text_parts = [extract_string_content(item) for item in content]
        return "\n".join(filter(None, text_parts))

    return str(content) if content is not None else ""
