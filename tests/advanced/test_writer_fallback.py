"""Tests for the writer-shaped Gemini fallback and the truncation guard.

Covers the June 2026 1928.T failure mode: when the Claude article writer is
unavailable, the fallback must NOT inherit the analyst/PM deep-reasoning profile
(``thinking_level="high"`` + deep reserve), because for Gemini 3+ the hidden
reasoning shares the completion-token pool and starves the article mid-sentence.
The fallback must use low/no thinking with an explicit visible-output budget,
and any residual ``MAX_TOKENS`` truncation must be logged loudly.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import src.llms as llms

_REPO_ROOT = Path(__file__).resolve().parents[2]


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
    def test_create_writer_llm_no_claude_key_no_openai_uses_gemini(self, mock_fallback):
        """No Claude key AND no usable OpenAI tier → Gemini floor (old behavior)."""
        sentinel = MagicMock()
        mock_fallback.return_value = sentinel
        # config is a pydantic Settings singleton — patch the method on the
        # class, not the instance (pydantic blocks instance-attr assignment).
        with (
            patch.object(type(llms.config), "get_claude_api_key", return_value=""),
            patch("src.llms._writer_openai_tier_available", return_value=False),
        ):
            result = llms.create_writer_llm()
        mock_fallback.assert_called_once()
        assert result is sentinel

    @patch("src.llms.create_writer_fallback_llm")
    @patch("src.llms.create_writer_openai_fallback_llm")
    def test_create_writer_llm_no_claude_key_prefers_openai_tier(
        self, mock_openai_fallback, mock_gemini_fallback
    ):
        """No Claude key with OpenAI usable → EDITOR_MODEL tier, Gemini untouched."""
        sentinel = MagicMock()
        mock_openai_fallback.return_value = sentinel
        with (
            patch.object(type(llms.config), "get_claude_api_key", return_value=""),
            patch("src.llms._writer_openai_tier_available", return_value=True),
        ):
            result = llms.create_writer_llm()
        assert result is sentinel
        mock_openai_fallback.assert_called_once()
        mock_gemini_fallback.assert_not_called()

    def test_create_writer_llm_no_key_routes_through_shared_chain(self):
        """Parity guard: the no-key path consumes writer_fallback_chain — the
        same resolver the runtime-error path iterates — so the two preference
        orders cannot diverge."""
        sentinel = MagicMock()
        chain = [llms.WriterTier("editor_model", lambda: sentinel)]
        with (
            patch.object(type(llms.config), "get_claude_api_key", return_value=""),
            patch("src.llms.writer_fallback_chain", return_value=chain) as mock_chain,
        ):
            result = llms.create_writer_llm()
        assert result is sentinel
        mock_chain.assert_called_once()


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


class TestHonestFallbackEventNames:
    """Fallback-path log event *names* must be family-neutral.

    Invariant: an event name must never bake in a destination model family the
    code didn't necessarily construct (the
    ``claude_writer_failed_falling_back_to_gemini`` bug class). Accurate
    source-family names (``writer_no_claude_key``,
    ``claude_writer_primary_failed``) are allowed — only fallback-shaped names
    are constrained; model/provider belong in structured fields.
    """

    _FAMILY = re.compile(r"gemini|gpt|claude|openai|anthropic", re.IGNORECASE)
    _FALLBACK = re.compile(r"fall.?back", re.IGNORECASE)
    _SCANNED_FILES = ("src/article_writer.py", "src/llms.py")

    @staticmethod
    def _logger_event_names(path: Path) -> list[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"debug", "info", "warning", "error", "critical"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names.append(node.args[0].value)
        return names

    def test_fallback_event_names_are_family_neutral(self):
        offenders = []
        for rel in self._SCANNED_FILES:
            for event in self._logger_event_names(_REPO_ROOT / rel):
                if self._FALLBACK.search(event) and self._FAMILY.search(event):
                    offenders.append((rel, event))
        assert (
            offenders == []
        ), f"Fallback-path event names must not hardcode a model family: {offenders}"

    def test_old_hardcoded_gemini_fallback_event_gone(self):
        old_event = "claude_writer_failed_falling_back_to_gemini"
        hits = []
        for top in ("src", "scripts"):
            for py_file in (_REPO_ROOT / top).rglob("*.py"):
                if old_event in py_file.read_text(encoding="utf-8"):
                    hits.append(str(py_file.relative_to(_REPO_ROOT)))
        assert hits == []


class TestTruncationGuard:
    @patch("src.article_writer.get_model_name", return_value="gemini-3.5-flash")
    @patch("src.article_writer.create_writer_llm")
    def test_max_tokens_finish_reason_raises(
        self, mock_create_writer, _mock_model_name, monkeypatch
    ):
        """Truncated output is a hard error — never return a clipped draft."""
        import src.article_writer as aw

        response = MagicMock()
        response.content = "# Title\n\nBody text that was cut off"
        response.response_metadata = {"finish_reason": "MAX_TOKENS"}
        writer = _make_writer(mock_create_writer, response)

        recorder = _RecordingLogger()
        monkeypatch.setattr(aw, "logger", recorder)
        with pytest.raises(RuntimeError, match="output token limit"):
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
