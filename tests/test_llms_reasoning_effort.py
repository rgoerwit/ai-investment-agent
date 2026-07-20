"""Tests for provider-model reasoning capability selection."""

import pytest

from src.llms import _openai_gpt5_reasoning_effort


@pytest.mark.parametrize(
    "model",
    ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.4-mini"],
)
def test_current_gpt5_models_use_low_effort_in_quick_mode(model):
    assert _openai_gpt5_reasoning_effort(model, quick_mode=True) == "low"


@pytest.mark.parametrize(
    "model",
    ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.4-mini"],
)
def test_current_gpt5_models_use_medium_effort_in_normal_mode(model):
    assert _openai_gpt5_reasoning_effort(model, quick_mode=False) == "medium"


@pytest.mark.parametrize("model", ["gpt-5.6-sol-pro", "gpt-5.4-pro", "gpt-4o"])
def test_pro_and_non_gpt5_models_do_not_get_gpt5_effort(model):
    assert _openai_gpt5_reasoning_effort(model, quick_mode=True) is None
