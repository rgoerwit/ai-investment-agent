from types import SimpleNamespace
from unittest.mock import Mock

from src.eval.llm_capture_meta import (
    extract_token_usage,
    extract_vendor_reasoning_config,
    gemini_thinking_level,
    normalize_reasoning_level,
)


def test_gemini_thinking_level_reads_legacy_attribute():
    runnable = SimpleNamespace(thinking_level="high", model_kwargs={})

    assert gemini_thinking_level(runnable) == "high"


def test_gemini_thinking_level_reads_new_reasoning_effort_attribute():
    runnable = SimpleNamespace(reasoning_effort="medium", model_kwargs={})

    assert gemini_thinking_level(runnable) == "medium"


def test_gemini_thinking_level_preserves_model_kwargs_fallback():
    runnable = SimpleNamespace(model_kwargs={"reasoning_effort": "low"})

    assert gemini_thinking_level(runnable) == "low"


def test_gemini_reasoning_provenance_keeps_canonical_thinking_level_name():
    runnable = SimpleNamespace(reasoning_effort="high", model_kwargs={})

    assert extract_vendor_reasoning_config(runnable, "google") == {
        "provider": "google",
        "name": "thinking_level",
        "value": "high",
    }


def test_non_gemini_reasoning_provenance_keeps_reasoning_effort_name():
    runnable = SimpleNamespace(reasoning_effort="high", model_kwargs={})

    assert extract_vendor_reasoning_config(runnable, "openai") == {
        "provider": "openai",
        "name": "reasoning_effort",
        "value": "high",
    }


def test_normalize_reasoning_level_supports_new_gemini_attribute():
    runnable = SimpleNamespace(reasoning_effort="medium", model_kwargs={})

    assert normalize_reasoning_level(runnable, "gemini-3.5-flash") == "medium"


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
