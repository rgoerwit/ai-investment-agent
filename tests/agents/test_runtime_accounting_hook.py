"""Tests for the _accounting_hook context manager in src/agents/runtime.py.

Stage 5 of the logging-audit follow-up. Verifies:
- Exception inside the with-block is swallowed (no propagation).
- BaseException (e.g., KeyboardInterrupt) is NOT swallowed.
- The log event uses summarize_exception output (sanitized), not raw str(exc).

These tests monkeypatch ``src.agents.runtime.logger`` directly rather than using
structlog.testing.capture_logs() — the latter is sensitive to global structlog
configuration set by other tests in the suite and gives flaky captures here.
"""

from __future__ import annotations

import pytest

from src.agents import runtime as runtime_module
from src.agents.runtime import _accounting_hook


class _RecordingLogger:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def debug(self, event, **kwargs):
        self.events.append((event, kwargs))

    # No-ops for completeness if other levels get called.
    def info(self, event, **kwargs):
        self.events.append((event, kwargs))

    def warning(self, event, **kwargs):
        self.events.append((event, kwargs))

    def error(self, event, **kwargs):
        self.events.append((event, kwargs))


def test_accounting_hook_swallows_exception(monkeypatch) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(runtime_module, "logger", recorder)
    with _accounting_hook("test_label"):
        raise RuntimeError("provider error: https://secret/path?token=abc")
    failures = [
        kwargs for event, kwargs in recorder.events if event == "accounting_hook_failed"
    ]
    assert len(failures) == 1
    record = failures[0]
    assert record["hook"] == "test_label"
    # summarize_exception fields are present (sanitized, not raw str(exc))
    assert "error_type" in record
    assert "failure_kind" in record
    assert record["operation"] == "accounting:test_label"
    assert "error" not in record  # raw str(exc) MUST NOT appear


def test_accounting_hook_does_not_swallow_keyboard_interrupt() -> None:
    with pytest.raises(KeyboardInterrupt):
        with _accounting_hook("test_label"):
            raise KeyboardInterrupt()


def test_accounting_hook_noop_when_block_succeeds(monkeypatch) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(runtime_module, "logger", recorder)
    with _accounting_hook("test_label"):
        pass
    assert recorder.events == []
