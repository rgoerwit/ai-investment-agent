from langchain_core.messages import AIMessage

from src.llm_runtime.responses import normalize_response


def test_openai_incomplete_response_normalizes_as_truncated() -> None:
    response = AIMessage(
        content="partial",
        response_metadata={
            "status": "incomplete",
            "incomplete_details": {"reason": "max_tokens"},
            "model": "gpt-5.4",
        },
        usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    )
    normalized = normalize_response(response)
    assert normalized.truncated is True
    assert normalized.finish_reason == "max_tokens"
    assert normalized.usage is not None and normalized.usage.output_tokens == 20


def test_tool_call_with_empty_text_is_valid_and_preserved() -> None:
    response = AIMessage(
        content="", tool_calls=[{"name": "lookup", "args": {}, "id": "call-1"}]
    )
    normalized = normalize_response(response)
    assert normalized.text == ""
    assert normalized.tool_calls[0]["id"] == "call-1"
    assert normalized.refusal_kind is None


def test_provider_refusal_is_distinct_from_truncation() -> None:
    normalized = normalize_response(
        AIMessage(content="", response_metadata={"finish_reason": "content_filter"})
    )
    assert normalized.refusal_kind == "content_filter"
    assert normalized.truncated is False
