"""Focused tests for the extracted CLI helpers."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest


def test_parse_arguments_allows_cleanup_without_ticker(monkeypatch):
    from src.cli import parse_arguments

    monkeypatch.setattr(sys, "argv", ["prog", "--capture-baseline-cleanup"])

    args = parse_arguments()

    assert args.capture_baseline_cleanup is True
    assert args.ticker is None


def test_parse_arguments_debug_implies_verbose(monkeypatch):
    from src.cli import parse_arguments

    monkeypatch.setattr(sys, "argv", ["prog", "--ticker", "6083.T", "--debug"])

    args = parse_arguments()

    assert args.debug is True
    assert args.verbose is True


def test_resolve_output_paths_uses_output_sibling_images():
    from src.cli import resolve_output_paths

    output_file, image_dir = resolve_output_paths(
        Namespace(output="results/report.md", imagedir=None)
    )

    assert output_file == Path("results/report.md")
    assert image_dir == Path("results/images")


def test_validate_cli_args_rejects_quick_with_chart_flags(capsys):
    from src.cli import _validate_cli_args

    with pytest.raises(SystemExit) as exc_info:
        _validate_cli_args(
            Namespace(quick=True, transparent=True, svg=False),
        )

    assert exc_info.value.code == 2
    assert "--transparent has no effect with --quick" in capsys.readouterr().err


def test_resolve_article_path_relative_to_output_dir():
    from src.cli import resolve_article_path

    args = Namespace(article="article.md", output="results/report.md")

    assert resolve_article_path(args, "0005.HK") == Path("results/article.md")


def test_resolve_article_path_default_uses_results_dir(monkeypatch, tmp_path):
    from src.cli import resolve_article_path

    monkeypatch.setattr("src.cli.config.results_dir", tmp_path)

    args = Namespace(article=True, output=None)

    assert resolve_article_path(args, "0005.HK") == tmp_path / "0005_HK_article.md"
