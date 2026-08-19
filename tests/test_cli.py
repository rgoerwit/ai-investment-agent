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


@pytest.mark.parametrize("flag", ["quick_model", "deep_model"])
def test_model_flags_are_accepted_under_both_binding_schemas(flag):
    """Neither schema rejects the flags; the resolver decides what they mean."""
    from src.cli import _validate_cli_args

    values = {
        "quick": False,
        "transparent": False,
        "svg": False,
        "quick_model": None,
        "deep_model": None,
    }
    values[flag] = "gemini-3.6-flash"

    for provider in ("google", None):
        _validate_cli_args(
            Namespace(**values), settings=Namespace(llm_base_provider=provider)
        )


def test_model_flag_help_names_the_intent_each_one_drives():
    """The mapping is non-obvious, so it has to be in --help, not just docs."""
    from src.cli import build_arg_parser

    actions = {
        action.dest: action.help
        for action in build_arg_parser()._actions
        if action.dest in {"quick_model", "deep_model"}
    }

    assert "fast" in actions["quick_model"]
    assert "reasoning" in actions["deep_model"]
    # The APEX carve-out is the surprising part: --deep-model does NOT move the
    # two gate-critical seats, because they are 'critical', not 'reasoning'.
    assert "critical" in actions["deep_model"]
    assert "LLM_SEAT_MODEL_OVERRIDES" in actions["deep_model"]


def test_explicit_article_path_is_cwd_relative_even_with_output():
    from src.cli import resolve_article_path

    args = Namespace(article="scratch/article.md", output="scratch/report.md")

    assert resolve_article_path(args, "4776.T") == Path("scratch/article.md")


def test_bare_article_derives_from_output():
    from src.cli import resolve_article_path

    args = Namespace(article=True, output="scratch/report.md")

    assert resolve_article_path(args, "4776.T") == Path("scratch/report_article.md")


def test_explicit_article_without_suffix_adds_md():
    from src.cli import resolve_article_path

    args = Namespace(article="scratch/article", output="scratch/report.md")

    assert resolve_article_path(args, "4776.T") == Path("scratch/article.md")


def test_resolve_article_path_default_uses_results_dir(monkeypatch, tmp_path):
    from src.cli import resolve_article_path

    monkeypatch.setattr("src.cli.config.results_dir", tmp_path)

    args = Namespace(article=True, output=None)

    assert resolve_article_path(args, "0005.HK") == tmp_path / "0005_HK_article.md"
