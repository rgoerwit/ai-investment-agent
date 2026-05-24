"""Tests for the shutdown hard-timeout path in `src.main`.

Background: an overnight macOS DNS outage left a "completed" analysis sitting
in `cleanup_async_resources()` for minutes because `httpx.AsyncClient.aclose()`
blocked on dead sockets. The finally block now wraps cleanup in
`run_with_hard_timeout` and calls `os._exit()` if cleanup exceeds the cap, so
the pipeline watchdog isn't needed to recover the process.

Three tests:
- happy: cleanup completes quickly → normal exit (no os._exit).
- edge: cleanup takes < timeout → still normal exit.
- error: cleanup hangs → warning emitted, os._exit called with code 0.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


async def _async_none():
    return None


async def _async_result(value):
    return value


def _patch_main_success_path(monkeypatch, args):
    """Wire up the same minimal stubs as test_main_cli's success path."""
    from src.cli import OutputTargets

    monkeypatch.setattr("src.main.cli.parse_arguments", lambda: args)
    monkeypatch.setattr("src.main.cli._validate_cli_args", lambda passed_args: None)
    monkeypatch.setattr(
        "src.main.cli._resolve_output_targets",
        lambda passed_args: OutputTargets(None, Path("images"), True),
    )
    monkeypatch.setattr(
        "src.main._setup_runtime", lambda passed_args, targets: ({}, object())
    )
    monkeypatch.setattr(
        "src.main._maybe_run_ticker_retrospective",
        lambda passed_args: _async_none(),
    )
    monkeypatch.setattr(
        "src.main.output._emit_start_banner",
        lambda passed_args, targets, **kwargs: "banner",
    )
    monkeypatch.setattr(
        "src.main._execute_analysis",
        lambda passed_args, targets, **kwargs: _async_result(
            {"analysis_validity": {"publishable": True}}
        ),
    )
    monkeypatch.setattr(
        "src.main._attach_run_summary",
        lambda result, passed_args, preflight: None,
    )
    monkeypatch.setattr(
        "src.main.output._render_primary_output",
        lambda result, passed_args, targets, banner, **kwargs: (None, None, None),
    )
    monkeypatch.setattr(
        "src.main.persistence._persist_analysis_outputs",
        lambda result, passed_args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.main.persistence._maybe_save_rejection_record",
        lambda result, passed_args, **kwargs: _async_none(),
    )
    monkeypatch.setattr(
        "src.main.output._maybe_generate_article",
        lambda result,
        passed_args,
        targets,
        company_name,
        report,
        reporter,
        **kwargs: _async_result(False),
    )
    monkeypatch.setattr(
        "src.main._log_final_summary",
        lambda result, passed_args, article_generated: None,
    )


def _basic_args():
    return SimpleNamespace(
        retrospective_only=False,
        ticker="6083.T",
        quick=True,
        strict=False,
        article=False,
        quiet=False,
        brief=False,
        svg=False,
        transparent=False,
        imagedir=None,
    )


class TestShutdownTimeout:
    def test_happy_path_cleanup_finishes_promptly(self, monkeypatch):
        """Cleanup returns under the cap — no os._exit, normal return."""
        from src.main import main

        _patch_main_success_path(monkeypatch, _basic_args())
        monkeypatch.setattr(
            "src.cleanup.cleanup_async_resources", lambda: _async_none()
        )

        with patch("src.main.os._exit") as mock_exit:
            rc = asyncio.run(main())

        assert rc == 0
        mock_exit.assert_not_called()

    def test_edge_cleanup_finishes_just_under_cap(self, monkeypatch):
        """Cleanup that sleeps briefly (under the cap) does not force-exit."""
        from src.main import config, main

        # Pin the cap high enough that a 50ms sleep is comfortably under.
        monkeypatch.setattr(config, "shutdown_hard_timeout_seconds", 5.0)

        async def slow_but_ok():
            await asyncio.sleep(0.05)

        _patch_main_success_path(monkeypatch, _basic_args())
        monkeypatch.setattr("src.cleanup.cleanup_async_resources", slow_but_ok)

        with patch("src.main.os._exit") as mock_exit:
            rc = asyncio.run(main())

        assert rc == 0
        mock_exit.assert_not_called()

    def test_error_cleanup_hangs_forces_exit(self, monkeypatch):
        """Cleanup that never returns triggers os._exit(0)."""
        from src.main import config, main

        # Keep the cap small so the test runs fast.
        monkeypatch.setattr(config, "shutdown_hard_timeout_seconds", 0.2)

        async def never_returns():
            await asyncio.sleep(60)  # would block forever in a real run

        _patch_main_success_path(monkeypatch, _basic_args())
        monkeypatch.setattr("src.cleanup.cleanup_async_resources", never_returns)

        warning_events: list[tuple[str, dict]] = []

        def fake_warning(event, **kwargs):
            warning_events.append((event, kwargs))

        monkeypatch.setattr("src.main.logger.warning", fake_warning)

        with patch("src.main.os._exit") as mock_exit:
            rc = asyncio.run(main())

        # Body finished cleanly. The warning fired and os._exit was called.
        assert rc == 0
        mock_exit.assert_called_once_with(0)
        assert any(
            event == "shutdown_cleanup_hard_timeout" for event, _ in warning_events
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
