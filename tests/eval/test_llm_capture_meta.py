from unittest.mock import Mock

from src.eval.llm_capture_meta import extract_token_usage


def test_extract_token_usage_delegates_to_shared_parser_for_gemini():
    response = Mock()
    response.usage_metadata = {
        "input_tokens": 120,
        "output_tokens": 900,
        "output_token_details": {"reasoning": 700},
        "total_tokens": 1020,
    }
    response.response_metadata = {}

    usage = extract_token_usage(response)

    assert usage == {
        "input_tokens": 120,
        "output_tokens": 900,
        "thinking_tokens": 700,
        "total_tokens": 1020,
    }


def test_extract_token_usage_delegates_to_shared_parser_for_openai():
    response = Mock()
    response.usage_metadata = None
    response.response_metadata = {
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 400,
            "completion_tokens_details": {"reasoning_tokens": 250},
            "total_tokens": 500,
        }
    }

    usage = extract_token_usage(response)

    assert usage == {
        "input_tokens": 100,
        "output_tokens": 400,
        "thinking_tokens": 250,
        "total_tokens": 500,
    }
