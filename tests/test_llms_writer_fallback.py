"""Regression guards for the article-writer model wiring.

Invariants (verified during the July 2026 3393.T review, previously untested):
- the Gemini fallback writer tracks the configured DEEP_MODEL — it is never a
  hardcoded model name (the ``gemini-3.5-flash`` seen in fallback runs is just
  the operator's ``.env`` DEEP_MODEL value);
- neither writer path changes under ``--quick``: quick mode affects analysis
  agents and clamps, not article voice.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config import config
from src.llms import create_writer_fallback_llm, create_writer_llm
from src.runtime_config import RuntimeConfig, build_runtime_config, use_runtime_config


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
