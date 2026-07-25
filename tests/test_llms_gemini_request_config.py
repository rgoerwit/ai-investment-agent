"""Gemini request-configuration compatibility tests.

The installed LangChain adapter supplies ``candidate_count=1`` and a default
temperature even when this repository does not. These tests inspect the final
request config so dependency defaults cannot silently reintroduce fields Google
says to omit from Gemini 3.x requests.
"""

from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

import src.llms as llms_mod


def _request_payload(
    model: str,
    *,
    generation_config: dict[str, Any] | None = None,
    **invoke_kwargs: Any,
) -> tuple[dict[str, Any], Any]:
    llm = llms_mod._TieredChatGoogleGenerativeAI(
        model=model,
        api_key="test-key",
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        candidate_count=2,
        max_output_tokens=1024,
        thinking_level="low",
        service_tier="flex",
    )
    request = llm._prepare_request(
        [HumanMessage(content="hi")],
        generation_config=generation_config,
        **invoke_kwargs,
    )
    config = request["config"]
    return config.model_dump(exclude_none=True), config


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-2.5-flash", frozenset()),
        (
            "gemini-3-flash-preview",
            frozenset({"candidate_count", "temperature", "top_p", "top_k"}),
        ),
        (
            "gemini-3.1-pro-preview",
            frozenset({"candidate_count", "temperature", "top_p", "top_k"}),
        ),
        (
            "gemini-3.5-flash",
            frozenset({"candidate_count", "temperature", "top_p", "top_k"}),
        ),
        (
            "gemini-3.5-flash-lite",
            frozenset({"candidate_count", "temperature", "top_p", "top_k"}),
        ),
        (
            "gemini-3.6-flash",
            frozenset({"candidate_count", "temperature", "top_p", "top_k"}),
        ),
        (
            "models/gemini-3.6-flash-002",
            frozenset({"candidate_count", "temperature", "top_p", "top_k"}),
        ),
        (
            "gemini-4-flash",
            frozenset({"candidate_count", "temperature", "top_p", "top_k"}),
        ),
        ("gpt-5.6-terra", frozenset()),
        ("", frozenset()),
        ("garbage", frozenset()),
    ],
)
def test_generation_field_policy(model, expected):
    assert llms_mod._gemini_generation_fields_to_omit(model) == expected


@pytest.mark.parametrize(
    ("model", "omitted_fields"),
    [
        (
            "gemini-3.1-pro-preview",
            {"candidate_count", "temperature", "top_p", "top_k"},
        ),
        (
            "gemini-3.5-flash-lite",
            {"candidate_count", "temperature", "top_p", "top_k"},
        ),
        (
            "gemini-3.6-flash",
            {"candidate_count", "temperature", "top_p", "top_k"},
        ),
    ],
)
def test_final_request_omits_model_policy_fields(model, omitted_fields):
    payload, config = _request_payload(model)

    assert omitted_fields.isdisjoint(payload)
    assert payload["max_output_tokens"] == 1024
    assert config.thinking_config.thinking_level.value == "LOW"
    assert config.service_tier.value == "flex"


def test_legacy_gemini_request_retains_supported_generation_fields():
    payload, _ = _request_payload("gemini-2.5-flash")

    assert payload["candidate_count"] == 2
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 40


def test_gemini_3_before_sampling_deprecation_uses_model_sampling_defaults():
    payload, _ = _request_payload("gemini-3.5-flash")

    assert {
        "candidate_count",
        "temperature",
        "top_p",
        "top_k",
    }.isdisjoint(payload)


def test_invoke_generation_config_cannot_reintroduce_omitted_fields():
    payload, _ = _request_payload(
        "gemini-3.6-flash",
        generation_config={
            "candidate_count": 4,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 80,
            "max_output_tokens": 2048,
        },
    )

    assert {
        "candidate_count",
        "temperature",
        "top_p",
        "top_k",
    }.isdisjoint(payload)
    assert payload["max_output_tokens"] == 2048


def test_invoke_kwargs_cannot_reintroduce_omitted_fields_or_api_aliases():
    payload, _ = _request_payload(
        "gemini-3.6-flash",
        candidate_count=4,
        temperature=0.8,
        top_p=0.95,
        top_k=80,
        candidateCount=5,
        topP=0.96,
        topK=90,
    )

    assert {
        "candidate_count",
        "temperature",
        "top_p",
        "top_k",
    }.isdisjoint(payload)


def test_invalid_omitted_override_is_ignored_for_fixed_sampling_model():
    payload, _ = _request_payload(
        "gemini-3.6-flash",
        generation_config={"top_p": "not-a-number"},
    )

    assert "top_p" not in payload


def test_invalid_supported_generation_field_still_raises():
    with pytest.raises(ValidationError):
        _request_payload(
            "gemini-3.6-flash",
            generation_config={"max_output_tokens": "not-an-integer"},
        )
