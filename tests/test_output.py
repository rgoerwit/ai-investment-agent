"""Focused tests for extracted output helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_load_company_name_for_output_retries_normalized_alias():
    from src.output import _load_company_name_for_output

    requested_symbols = []

    class _Ticker:
        def __init__(self, symbol):
            self.info = {"longName": "Truecaller AB"} if symbol == "TRUE-B.ST" else {}

    class _Future:
        def __init__(self, fn):
            self._fn = fn

        def result(self, timeout=None):
            return self._fn()

    class _Executor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn):
            return _Future(fn)

    fake_yfinance = MagicMock()

    def _ticker_factory(symbol):
        requested_symbols.append(symbol)
        return _Ticker(symbol)

    fake_yfinance.Ticker.side_effect = _ticker_factory

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yfinance)
        assert (
            _load_company_name_for_output(
                "TRUE.B.ST",
                thread_pool_executor_cls=lambda max_workers=1: _Executor(),
            )
            == "Truecaller AB"
        )

    assert requested_symbols == ["TRUE.B.ST", "TRUE-B.ST"]


def test_render_primary_output_writes_report_without_banner(tmp_path):
    from src.cli import OutputTargets
    from src.output import _render_primary_output

    output_file = tmp_path / "report.md"
    args = SimpleNamespace(
        ticker="TST",
        brief=False,
        quiet=False,
        quick=False,
        svg=False,
        transparent=False,
    )

    class StubReporter:
        def __init__(self, *args, **kwargs):
            pass

        def generate_report(self, result, brief_mode=False):
            return "# TST Report\n\nBody"

    _render_primary_output(
        {"final_trade_decision": "BUY"},
        args,
        OutputTargets(
            output_file=output_file, image_dir=tmp_path / "images", skip_charts=True
        ),
        welcome_banner="# Banner",
        reporter_cls=StubReporter,
        company_name_loader=lambda ticker: "Test Corp",
        cost_suffix_fn=lambda: "",
    )

    assert output_file.read_text() == "# TST Report\n\nBody"


@pytest.mark.asyncio
async def test_maybe_generate_article_skips_invalid_analysis():
    from src.cli import OutputTargets
    from src.output import _maybe_generate_article

    handle_article_generation = AsyncMock()
    console = MagicMock()
    logger = MagicMock()
    args = SimpleNamespace(
        ticker="7203.T",
        article=True,
        quiet=False,
        brief=False,
        quick=False,
        svg=False,
        transparent=False,
        imagedir=None,
    )

    generated = await _maybe_generate_article(
        {"analysis_validity": {"publishable": False}},
        args,
        OutputTargets(output_file=None, image_dir=Path("images"), skip_charts=True),
        company_name=None,
        report=None,
        reporter=None,
        logger_obj=logger,
        console_obj=console,
        handle_article_generation_fn=handle_article_generation,
        publishable_analysis_fn=lambda result: False,
    )

    assert generated is False
    handle_article_generation.assert_not_called()
    console.print.assert_called_once()


def test_report_analysis_failure_quiet_mode(capsys):
    from src.output import _report_analysis_failure

    _report_analysis_failure(SimpleNamespace(quiet=True, brief=False))

    assert "# Analysis Failed" in capsys.readouterr().out
