"""Tests for the writer-shaped Gemini fallback and the truncation guard.

Covers the June 2026 1928.T failure mode: when the Claude article writer is
unavailable, the fallback must NOT inherit the analyst/PM deep-reasoning profile
(``thinking_level="high"`` + deep reserve), because for Gemini 3+ the hidden
reasoning shares the completion-token pool and starves the article mid-sentence.
The fallback must use low/no thinking with an explicit visible-output budget,
and any residual ``MAX_TOKENS`` truncation must be logged loudly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import src.llms as llms


def _fake_runtime(model: str) -> SimpleNamespace:
    return SimpleNamespace(deep_think_llm=model, api_retry_attempts=2)


class TestWriterFallbackFactory:
    @patch("src.llms.create_gemini_model")
    def test_v3_model_uses_low_thinking_and_16k_budget(self, mock_create_gemini):
        with patch(
            "src.llms.get_runtime_config",
            return_value=_fake_runtime("gemini-3.5-flash"),
        ):
            llms.create_writer_fallback_llm()

        args, kwargs = mock_create_gemini.call_args
        assert args[0] == "gemini-3.5-flash"
        assert kwargs["thinking_level"] == "low"
        assert kwargs["max_output_tokens"] == 16384
        assert kwargs["reserve_class"] == "default"

    @patch("src.llms.create_gemini_model")
    def test_non_v3_model_disables_thinking_but_keeps_budget(self, mock_create_gemini):
        with patch(
            "src.llms.get_runtime_config",
            return_value=_fake_runtime("gemini-2.0-flash"),
        ):
            llms.create_writer_fallback_llm()

        args, kwargs = mock_create_gemini.call_args
        assert kwargs["thinking_level"] is None
        assert kwargs["max_output_tokens"] == 16384
        assert kwargs["reserve_class"] == "default"

    def test_budget_protects_visible_output(self):
        """The 16384 visible intent is never cannibalized by the reserve."""
        from src.llm_budgets import get_generation_budget

        # v3 + low thinking ⇒ reserve_enabled=True, reserve_class="default".
        budget = get_generation_budget(
            intent_tokens=16384,
            reserve_class="default",
            reserve_enabled=True,
            default_reserve_tokens=2048,
            deep_reserve_tokens=8192,
        )
        assert budget.intent_tokens == 16384
        assert budget.reserve_tokens == 2048
        assert budget.api_cap_tokens == 18432

    @patch("src.llms.create_writer_fallback_llm")
    def test_create_writer_llm_uses_fallback_when_no_claude_key(self, mock_fallback):
        sentinel = MagicMock()
        mock_fallback.return_value = sentinel
        # config is a pydantic Settings singleton — patch the method on the
        # class, not the instance (pydantic blocks instance-attr assignment).
        with patch.object(type(llms.config), "get_claude_api_key", return_value=""):
            result = llms.create_writer_llm()
        mock_fallback.assert_called_once()
        assert result is sentinel


class _RecordingLogger:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def debug(self, event, **kwargs):
        self.events.append((event, kwargs))

    def info(self, event, **kwargs):
        self.events.append((event, kwargs))

    def warning(self, event, **kwargs):
        self.events.append((event, kwargs))

    def error(self, event, **kwargs):
        self.events.append((event, kwargs))


def _make_writer(mock_create_writer, response):
    from src.article_writer import ArticleWriter

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = response
    mock_create_writer.return_value = mock_llm
    return ArticleWriter(
        samples_dir=Path("writing_samples")
        if Path("writing_samples").exists()
        else None,
    )


class TestTruncationGuard:
    @patch("src.article_writer.get_model_name", return_value="gemini-3.5-flash")
    @patch("src.article_writer.create_writer_llm")
    def test_max_tokens_finish_reason_emits_warning(
        self, mock_create_writer, _mock_model_name, monkeypatch
    ):
        import src.article_writer as aw

        response = MagicMock()
        response.content = "# Title\n\nBody text that was cut off"
        response.response_metadata = {"finish_reason": "MAX_TOKENS"}
        writer = _make_writer(mock_create_writer, response)

        recorder = _RecordingLogger()
        monkeypatch.setattr(aw, "logger", recorder)
        writer._invoke_writer([MagicMock()])

        truncations = [
            kwargs
            for event, kwargs in recorder.events
            if event == "writer_output_truncated"
        ]
        assert len(truncations) == 1
        assert truncations[0]["finish_reason"] == "MAX_TOKENS"

    @patch("src.article_writer.create_writer_llm")
    def test_stop_finish_reason_does_not_warn(self, mock_create_writer, monkeypatch):
        import src.article_writer as aw

        response = MagicMock()
        response.content = "# Title\n\nComplete body."
        response.response_metadata = {"finish_reason": "STOP"}
        writer = _make_writer(mock_create_writer, response)

        recorder = _RecordingLogger()
        monkeypatch.setattr(aw, "logger", recorder)
        writer._invoke_writer([MagicMock()])

        assert not [e for e, _ in recorder.events if e == "writer_output_truncated"]

    @patch("src.article_writer.create_writer_llm")
    def test_missing_response_metadata_is_safe(self, mock_create_writer, monkeypatch):
        import src.article_writer as aw

        # A plain string-content response with no response_metadata attribute.
        response = SimpleNamespace(content="# Title\n\nComplete body.")
        writer = _make_writer(mock_create_writer, response)

        recorder = _RecordingLogger()
        monkeypatch.setattr(aw, "logger", recorder)
        article = writer._invoke_writer([MagicMock()])

        assert "Complete body." in article
        assert not [e for e, _ in recorder.events if e == "writer_output_truncated"]
