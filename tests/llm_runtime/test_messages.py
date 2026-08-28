from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.message_utils import ToolHistoryIntegrityError
from src.llm_runtime.messages import prepare_messages_for_model


def test_non_google_keeps_tool_message_and_call_id() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "x", "args": {}, "id": "call-1"}],
            name="agent",
        ),
        ToolMessage(
            content="ok",
            tool_call_id="call-1",
            additional_kwargs={"agent_key": "agent"},
        ),
    ]
    prepared = prepare_messages_for_model(
        SimpleNamespace(_llm_adapter_kind="openai_native"), messages, agent_key="agent"
    )
    assert prepared == messages
    assert prepared[1].tool_call_id == "call-1"


def test_google_cleanup_merges_consecutive_human_messages() -> None:
    messages = [HumanMessage(content="one"), HumanMessage(content="two")]
    prepared = prepare_messages_for_model(
        SimpleNamespace(_llm_adapter_kind="google_native"), messages, agent_key="agent"
    )
    assert len(prepared) == 1
    assert prepared[0].content == "one\n\ntwo"


@pytest.mark.parametrize("adapter_kind", ["openai_native", "google_native"])
def test_every_provider_rejects_orphaned_tool_output_before_invocation(
    adapter_kind: str,
) -> None:
    messages = [
        HumanMessage(content="analyze"),
        ToolMessage(
            content="orphaned",
            tool_call_id="call-1",
            additional_kwargs={"agent_key": "agent"},
        ),
    ]

    with pytest.raises(ToolHistoryIntegrityError, match="orphaned tool outputs=1"):
        prepare_messages_for_model(
            SimpleNamespace(_llm_adapter_kind=adapter_kind),
            messages,
            agent_key="agent",
        )


def test_rejects_missing_and_duplicate_tool_outputs() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "x", "args": {}, "id": "call-1"},
                {"name": "y", "args": {}, "id": "call-2"},
            ],
            name="agent",
        ),
        ToolMessage(
            content="one",
            tool_call_id="call-1",
            additional_kwargs={"agent_key": "agent"},
        ),
        ToolMessage(
            content="duplicate",
            tool_call_id="call-1",
            additional_kwargs={"agent_key": "agent"},
        ),
    ]

    with pytest.raises(ToolHistoryIntegrityError) as exc_info:
        prepare_messages_for_model(
            SimpleNamespace(_llm_adapter_kind="openai_native"),
            messages,
            agent_key="agent",
        )

    assert exc_info.value.missing_outputs == 1
    assert exc_info.value.duplicate_outputs == 1
    assert exc_info.value.ownership_mismatches == 0


def test_rejects_reused_tool_call_identifier() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "x", "args": {}, "id": "call-1"}],
            name="agent",
        ),
        ToolMessage(
            content="one",
            tool_call_id="call-1",
            additional_kwargs={"agent_key": "agent"},
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "y", "args": {}, "id": "call-1"}],
            name="agent",
        ),
        ToolMessage(
            content="two",
            tool_call_id="call-1",
            additional_kwargs={"agent_key": "agent"},
        ),
    ]

    with pytest.raises(ToolHistoryIntegrityError) as exc_info:
        prepare_messages_for_model(
            SimpleNamespace(_llm_adapter_kind="openai_native"),
            messages,
            agent_key="agent",
        )

    assert exc_info.value.duplicate_calls == 1


def test_validated_history_serializes_as_strict_responses_api_pair() -> None:
    """Lock the application invariant to the transport shape OpenAI enforces."""
    from langchain_openai import ChatOpenAI

    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "x", "args": {}, "id": "call-1"}],
            name="agent",
        ),
        ToolMessage(
            content="ok",
            tool_call_id="call-1",
            additional_kwargs={"agent_key": "agent"},
        ),
    ]
    model = ChatOpenAI(
        model="gpt-5.4-mini",
        api_key="test-key",
        use_responses_api=True,
        output_version="responses/v1",
    )

    prepared = prepare_messages_for_model(model, messages, agent_key="agent")
    payload = model._get_request_payload(prepared)

    assert payload["input"] == [
        {
            "type": "function_call",
            "name": "x",
            "arguments": "{}",
            "call_id": "call-1",
        },
        {
            "type": "function_call_output",
            "output": "ok",
            "call_id": "call-1",
        },
    ]
