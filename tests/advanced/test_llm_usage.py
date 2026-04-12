from types import SimpleNamespace
from unittest.mock import Mock

from src.llm_usage import extract_token_usage_breakdown


def test_extract_token_usage_breakdown_from_gemini_usage_metadata():
    response = Mock()
    response.usage_metadata = {
        "input_tokens": 100,
        "output_tokens": 900,
        "total_tokens": 1000,
        "output_token_details": {"reasoning": 800},
    }
    response.response_metadata = {}

    usage = extract_token_usage_breakdown(response)

    assert usage.input_tokens == 100
    assert usage.total_output_tokens == 900
    assert usage.thinking_tokens == 800
    assert usage.visible_output_tokens == 100
    assert usage.total_tokens == 1000


def test_extract_token_usage_breakdown_from_openai_token_usage():
    response = Mock()
    response.usage_metadata = None
    response.response_metadata = {
        "token_usage": {
            "prompt_tokens": 120,
            "completion_tokens": 880,
            "total_tokens": 1000,
            "completion_tokens_details": {"reasoning_tokens": 700},
        }
    }

    usage = extract_token_usage_breakdown(response)

    assert usage.input_tokens == 120
    assert usage.total_output_tokens == 880
    assert usage.thinking_tokens == 700
    assert usage.visible_output_tokens == 180
    assert usage.total_tokens == 1000


def test_extract_token_usage_breakdown_from_openai_object_metadata():
    response = Mock()
    response.usage_metadata = None
    response.response_metadata = SimpleNamespace(
        token_usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=880,
            total_tokens=1000,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=700),
        )
    )

    usage = extract_token_usage_breakdown(response)

    assert usage.input_tokens == 120
    assert usage.total_output_tokens == 880
    assert usage.thinking_tokens == 700
    assert usage.visible_output_tokens == 180
    assert usage.total_tokens == 1000


def test_extract_token_usage_breakdown_without_reasoning_detail():
    response = Mock()
    response.usage_metadata = {"input_tokens": 100, "output_tokens": 200}
    response.response_metadata = {}

    usage = extract_token_usage_breakdown(response)

    assert usage.input_tokens == 100
    assert usage.total_output_tokens == 200
    assert usage.thinking_tokens is None
    assert usage.visible_output_tokens is None
    assert usage.total_tokens == 300


def test_extract_token_usage_breakdown_handles_malformed_metadata():
    response = Mock()
    response.usage_metadata = {"input_tokens": "abc", "output_tokens": None}
    response.response_metadata = {"token_usage": "bad"}

    usage = extract_token_usage_breakdown(response)

    assert usage.input_tokens is None
    assert usage.total_output_tokens is None
    assert usage.thinking_tokens is None
    assert usage.visible_output_tokens is None
    assert usage.total_tokens is None
