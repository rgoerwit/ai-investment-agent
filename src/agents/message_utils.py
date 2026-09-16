from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.text_patterns import URL_RE
from src.tooling.evidence_recorder import (
    EvidenceAuthority,
    EvidenceRecord,
    EvidenceStatus,
    classify_evidence_value,
    normalize_http_url,
    resolve_evidence_authority,
)

logger = structlog.get_logger(__name__)


class ToolHistoryIntegrityError(ValueError):
    """A model transcript contains an invalid tool-call/result relationship."""

    def __init__(
        self,
        agent_key: str,
        *,
        orphaned_outputs: int = 0,
        missing_outputs: int = 0,
        duplicate_calls: int = 0,
        duplicate_outputs: int = 0,
        ownership_mismatches: int = 0,
    ) -> None:
        self.agent_key = agent_key
        self.orphaned_outputs = orphaned_outputs
        self.missing_outputs = missing_outputs
        self.duplicate_calls = duplicate_calls
        self.duplicate_outputs = duplicate_outputs
        self.ownership_mismatches = ownership_mismatches
        parts = []
        if orphaned_outputs:
            parts.append(f"orphaned tool outputs={orphaned_outputs}")
        if missing_outputs:
            parts.append(f"tool calls missing outputs={missing_outputs}")
        if duplicate_calls:
            parts.append(f"duplicate tool call ids={duplicate_calls}")
        if duplicate_outputs:
            parts.append(f"duplicate tool outputs={duplicate_outputs}")
        if ownership_mismatches:
            parts.append(f"agent ownership mismatches={ownership_mismatches}")
        details = ", ".join(parts) or "invalid tool history"
        super().__init__(
            f"{agent_key} tool history is incomplete or inconsistent: {details}. "
            "The branch was stopped before another LLM request."
        )


def tool_call_ids(message: BaseMessage) -> tuple[str, ...]:
    """Return normalized tool-call identifiers carried by one assistant message."""

    if not isinstance(message, AIMessage):
        return ()
    tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(tool_calls, list):
        return ()
    ids: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        call_id = tool_call.get("id")
        if isinstance(call_id, str) and call_id:
            ids.append(call_id)
    return tuple(ids)


def message_agent_key(message: BaseMessage) -> str | None:
    """Return the canonical owner tag used to isolate parallel transcripts."""

    if isinstance(message, AIMessage):
        name = getattr(message, "name", None)
        return name if isinstance(name, str) and name else None
    if isinstance(message, ToolMessage):
        value = (message.additional_kwargs or {}).get("agent_key")
        return value if isinstance(value, str) and value else None
    return None


def validate_tool_history(messages: Sequence[BaseMessage], *, agent_key: str) -> None:
    """Reject malformed tool transcripts before any provider request is made.

    Providers differ in how strictly they validate tool history. Application
    correctness cannot depend on that tolerance: every tool output must follow
    exactly one matching assistant call owned by the same agent, and every call
    sent back to a model must have exactly one result.
    """

    calls: dict[str, tuple[int, str | None]] = {}
    outputs: dict[str, list[tuple[int, str | None]]] = {}
    duplicate_calls = 0
    for index, message in enumerate(messages):
        owner = message_agent_key(message)
        for call_id in tool_call_ids(message):
            if call_id in calls:
                duplicate_calls += 1
            calls[call_id] = (index, owner)
        if isinstance(message, ToolMessage):
            call_id = getattr(message, "tool_call_id", "") or ""
            outputs.setdefault(call_id, []).append((index, owner))

    orphaned = 0
    duplicates = 0
    ownership_mismatches = 0
    for call_id, result_entries in outputs.items():
        call_entry = calls.get(call_id)
        if call_entry is None:
            orphaned += len(result_entries)
            continue
        call_index, call_owner = call_entry
        if len(result_entries) > 1:
            duplicates += len(result_entries) - 1
        for result_index, result_owner in result_entries:
            if result_index <= call_index:
                orphaned += 1
            if call_owner and result_owner and call_owner != result_owner:
                ownership_mismatches += 1

    missing = sum(1 for call_id in calls if len(outputs.get(call_id, ())) == 0)
    if orphaned or missing or duplicate_calls or duplicates or ownership_mismatches:
        raise ToolHistoryIntegrityError(
            agent_key,
            orphaned_outputs=orphaned,
            missing_outputs=missing,
            duplicate_calls=duplicate_calls,
            duplicate_outputs=duplicates,
            ownership_mismatches=ownership_mismatches,
        )


@dataclass(frozen=True, slots=True)
class ToolEvidenceRecord:
    """Agent-facing evidence view with compatibility for legacy tuple consumers."""

    tool_name: str | None
    content: str
    urls: set[str]
    evidence_status: EvidenceStatus
    authority: EvidenceAuthority

    def __iter__(self):
        yield self.tool_name
        yield self.content
        yield self.urls

    def __getitem__(self, index: int):
        return (self.tool_name, self.content, self.urls)[index]


_URL_RE = URL_RE


def make_tool_evidence_record(
    *,
    tool_name: str | None,
    content: str,
    urls: Iterable[str],
    evidence_status: EvidenceStatus | None = None,
) -> ToolEvidenceRecord:
    """Build the canonical agent-facing view of inspected tool evidence."""
    normalized_urls = {
        normalized
        for url in urls
        if (normalized := normalize_http_url(url)) is not None
    }
    _, classified_status, _, bounded = classify_evidence_value(
        tool_name or "",
        content,
    )
    status = evidence_status or classified_status
    return ToolEvidenceRecord(
        tool_name=tool_name,
        content=bounded,
        urls=normalized_urls,
        evidence_status=status,
        authority=resolve_evidence_authority(
            tool_name=tool_name or "",
            evidence_status=status,
            urls=tuple(sorted(normalized_urls)),
        ),
    )


def evidence_record_to_tool_evidence(record: EvidenceRecord) -> ToolEvidenceRecord:
    """Convert one run-scoped ledger record without weakening its status."""
    return make_tool_evidence_record(
        tool_name=record.tool_name,
        content=record.content,
        urls=record.urls,
        evidence_status=record.evidence_status,
    )


def tool_evidence_records(
    messages: Sequence[BaseMessage],
) -> list[ToolEvidenceRecord]:
    """Extract tool name, text, and normalized URLs from ToolMessages."""

    records: list[ToolEvidenceRecord] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = extract_string_content(message.content)
        records.append(
            make_tool_evidence_record(
                tool_name=message.name,
                content=content,
                urls=(match.group(0) for match in _URL_RE.finditer(content)),
            )
        )
    return records


def tool_evidence_urls(messages: Sequence[BaseMessage]) -> set[str]:
    """Return normalized URLs that actually occurred in tool output."""

    return {
        url for _name, _content, urls in tool_evidence_records(messages) for url in urls
    }


def filter_messages_by_agent(
    messages: list[BaseMessage], agent_key: str
) -> list[BaseMessage]:
    """
    Filter messages to only include this agent's conversation history.
    """
    if not messages:
        return []

    tool_msg_agents: list[Any] = []
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

    filtered: list[BaseMessage] = []
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

    filtered: list[BaseMessage] = []
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
        # Typed non-text blocks — no textual payload. Stringifying a
        # function_call/tool_use dict here once persisted a raw tool-call as a
        # consultant review (3679.T 2026-07-11): the non-empty garbage
        # suppressed the loop's empty-content forced-synthesis fallback.
        if content.get("type") in (
            "reasoning",
            "thinking",
            "redacted_thinking",
            "function_call",
            "tool_use",
        ):
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


def latest_agent_text(messages: Sequence[BaseMessage], agent_key: str) -> str:
    """Return the latest non-tool-call response emitted by a named agent."""
    for message in reversed(messages):
        if (
            isinstance(message, AIMessage)
            and getattr(message, "name", None) == agent_key
            and not getattr(message, "tool_calls", None)
        ):
            content = extract_string_content(message.content)
            if content.strip():
                return content
    return ""
