"""Regression guards for the article-writer model wiring.

Invariants (verified during the July 2026 3393.T review, previously untested):
- the Gemini fallback writer tracks the configured DEEP_MODEL — it is never a
  hardcoded model name (the ``gemini-3.5-flash`` seen in fallback runs is just
  the operator's ``.env`` DEEP_MODEL value);
- neither writer path changes under ``--quick``: quick mode affects analysis
  agents and clamps, not article voice.

Chain invariants (July 2026 writer-fallback-chain feature):
- tier order is EDITOR_MODEL (OpenAI) → Gemini floor, the floor always present;
- the OpenAI tier is key- AND switch-aware (``ENABLE_CONSULTANT`` is the
  OpenAI master switch);
- tiers are lazy — building the chain constructs no LLM;
- the OpenAI tier uses the writer's 16384 long-form budget, not the editor's
  8192, and low reasoning effort.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import src.llms as llms_mod
from src.config import config
from src.llms import (
    create_writer_fallback_llm,
    create_writer_llm,
    create_writer_openai_fallback_llm,
    writer_fallback_chain,
)
from src.runtime_config import RuntimeConfig, build_runtime_config, use_runtime_config


def _openai_available(available: bool = True):
    """Patch the three inputs of _writer_openai_tier_available coherently."""
    return patch("src.llms._writer_openai_tier_available", return_value=available)


def _runtime(**overrides) -> RuntimeConfig:
    return RuntimeConfig.from_config(config).with_overrides(**overrides)


def test_writer_fallback_tracks_deep_model():
    runtime = _runtime(deep_think_llm="sentinel-deep-model")

    with (
        use_runtime_config(runtime),
        patch("src.llms.create_gemini_model") as create_mock,
    ):
        create_writer_fallback_llm()

    model_name = create_mock.call_args.args[0]
    assert model_name == "sentinel-deep-model"
    assert model_name != "gemini-3.5-flash"


def test_writer_fallback_quick_mode_invariant():
    runtime = _runtime(
        deep_think_llm="sentinel-deep-model",
        quick_think_llm="sentinel-quick-model",
        quick_mode_active=True,
    )

    with (
        use_runtime_config(runtime),
        patch("src.llms.create_gemini_model") as create_mock,
    ):
        create_writer_fallback_llm()

    assert create_mock.call_args.args[0] == "sentinel-deep-model"


def test_build_runtime_config_quick_does_not_override_deep_model():
    args = SimpleNamespace(quick=True)

    runtime = build_runtime_config(args, config)

    assert runtime.quick_mode_active is True
    assert runtime.deep_think_llm == config.deep_think_llm


def test_claude_writer_model_quick_mode_invariant():
    runtime = _runtime(quick_mode_active=True, quick_think_llm="sentinel-quick-model")
    mock_config = MagicMock()
    mock_config.get_claude_api_key.return_value = "fake-key"
    mock_config.writer_model = "claude-sentinel-writer"
    mock_config.api_timeout = 300

    with use_runtime_config(runtime), patch("src.llms.config", mock_config):
        llm = create_writer_llm()

    assert llm.model == "claude-sentinel-writer"


class TestWriterOpenAITierAvailability:
    """_writer_openai_tier_available: lib + key + ENABLE_CONSULTANT, all required."""

    def _availability(self, *, lib: bool, consultant: bool, key: str) -> bool:
        mock_config = MagicMock()
        mock_config.enable_consultant = consultant
        mock_config.get_openai_api_key.return_value = key
        with (
            patch("src.llms._langchain_openai_available", return_value=lib),
            patch("src.llms.config", mock_config),
        ):
            return llms_mod._writer_openai_tier_available()

    def test_available_when_all_three_present(self):
        assert self._availability(lib=True, consultant=True, key="sk-test") is True

    def test_unavailable_without_key(self):
        assert self._availability(lib=True, consultant=True, key="") is False

    def test_unavailable_when_consultant_disabled(self):
        """ENABLE_CONSULTANT is the OpenAI master switch — key alone is not enough."""
        assert self._availability(lib=True, consultant=False, key="sk-test") is False

    def test_unavailable_without_lib(self):
        assert self._availability(lib=False, consultant=True, key="sk-test") is False


class TestWriterFallbackChain:
    def test_chain_prefers_editor_tier_when_openai_usable(self):
        with _openai_available(True):
            labels = [tier.label for tier in writer_fallback_chain()]
        assert labels == ["editor_model", "gemini_last_resort"]

    def test_chain_is_gemini_only_when_openai_unusable(self):
        with _openai_available(False):
            labels = [tier.label for tier in writer_fallback_chain()]
        assert labels == ["gemini_last_resort"]

    def test_gemini_floor_always_present(self):
        for available in (True, False):
            with _openai_available(available):
                chain = writer_fallback_chain()
            assert chain, "chain must never be empty"
            assert chain[-1].label == "gemini_last_resort"

    def test_chain_is_lazy(self):
        """Building the chain constructs no LLM — tiers build on attempt only."""
        with (
            _openai_available(True),
            patch("src.llms.create_writer_openai_fallback_llm") as mock_openai,
            patch("src.llms.create_writer_fallback_llm") as mock_gemini,
        ):
            chain = writer_fallback_chain()
            mock_openai.assert_not_called()
            mock_gemini.assert_not_called()

            chain[0].build()
            mock_openai.assert_called_once()
            mock_gemini.assert_not_called()

            chain[1].build()
            mock_gemini.assert_called_once()

    def test_gemini_tier_receives_writer_temperature(self):
        with (
            _openai_available(False),
            patch("src.llms.create_writer_fallback_llm") as mock_gemini,
        ):
            writer_fallback_chain(temperature=0.7)[0].build()
        assert mock_gemini.call_args.kwargs["temperature"] == 0.7

    def test_openai_tier_build_raises_if_availability_lost(self):
        """Race guard: availability at chain build, gone at attempt → clean raise
        (the runtime loop treats it like any tier failure and advances)."""
        with _openai_available(True):
            chain = writer_fallback_chain()
        with (
            _openai_available(False),
            pytest.raises(RuntimeError, match="unavailable"),
        ):
            chain[0].build()

    def test_chain_shape_quick_mode_invariant(self):
        runtime = _runtime(quick_mode_active=True)
        with _openai_available(True), use_runtime_config(runtime):
            labels = [tier.label for tier in writer_fallback_chain()]
        assert labels == ["editor_model", "gemini_last_resort"]


class TestWriterOpenAIFallbackFactory:
    def _build(self, editor_model: str, consultant_model: str = "gpt-5.4"):
        mock_config = MagicMock()
        mock_config.enable_consultant = True
        mock_config.get_openai_api_key.return_value = "sk-test"
        mock_config.editor_model = editor_model
        mock_config.consultant_model = consultant_model
        captured: dict = {}

        def _capture(kwargs):
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("src.llms._langchain_openai_available", return_value=True),
            patch("src.llms.config", mock_config),
            patch("src.llms._construct_chat_openai", side_effect=_capture),
        ):
            llm = create_writer_openai_fallback_llm()
        return llm, captured

    def test_uses_editor_model_with_writer_budget(self):
        llm, kwargs = self._build("gpt-5.4")
        assert llm is not None
        assert kwargs["model"] == "gpt-5.4"
        # Writer's long-form intent (16384) + reserve — never the editor's 8192.
        assert kwargs["max_completion_tokens"] >= 16384
        assert kwargs["use_responses_api"] is True

    def test_gpt5_nonpro_gets_low_reasoning_effort(self):
        """Prose needs output budget, not reasoning depth (editor uses medium)."""
        _llm, kwargs = self._build("gpt-5.4")
        assert kwargs["reasoning_effort"] == "low"

    def test_editor_model_fallback_chain_to_consultant_model(self):
        _llm, kwargs = self._build("", consultant_model="gpt-5.4-mini")
        assert kwargs["model"] == "gpt-5.4-mini"

    def test_returns_none_when_unavailable(self):
        mock_config = MagicMock()
        mock_config.enable_consultant = False
        mock_config.get_openai_api_key.return_value = "sk-test"
        with (
            patch("src.llms._langchain_openai_available", return_value=True),
            patch("src.llms.config", mock_config),
        ):
            assert create_writer_openai_fallback_llm() is None
