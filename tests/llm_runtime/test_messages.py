from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
