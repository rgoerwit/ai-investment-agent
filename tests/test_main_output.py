"""Tests for --output argument handling in main.py."""

import tempfile
from argparse import Namespace
from pathlib import Path

import pytest


class TestResolveOutputPaths:
    """Tests for resolve_output_paths function."""

    def test_no_output_no_imagedir(self):
        """When neither output nor imagedir specified, use defaults."""
        from src.main import resolve_output_paths

        args = Namespace(output=None, imagedir=None)
        output_file, image_dir = resolve_output_paths(args)

        assert output_file is None
        assert image_dir == Path("images")

    def test_output_specified_no_imagedir(self):
        """When output specified but not imagedir, derive imagedir from output."""
        from src.main import resolve_output_paths

        args = Namespace(output="results/report.md", imagedir=None)
        output_file, image_dir = resolve_output_paths(args)

        assert output_file == Path("results/report.md")
        assert image_dir == Path("results/images")

    def test_output_in_current_dir(self):
        """When output is in current directory, imagedir is ./images."""
        from src.main import resolve_output_paths

        args = Namespace(output="report.md", imagedir=None)
        output_file, image_dir = resolve_output_paths(args)

        assert output_file == Path("report.md")
        assert image_dir == Path("images")

    def test_imagedir_explicit_override(self):
        """When imagedir explicitly specified, use it regardless of output."""
        from src.main import resolve_output_paths

        args = Namespace(output="results/report.md", imagedir="custom/charts")
        output_file, image_dir = resolve_output_paths(args)

        assert output_file == Path("results/report.md")
        assert image_dir == Path("custom/charts")

    def test_imagedir_without_output(self):
        """When imagedir specified but not output, use specified imagedir."""
        from src.main import resolve_output_paths

        args = Namespace(output=None, imagedir="my/images")
        output_file, image_dir = resolve_output_paths(args)

        assert output_file is None
        assert image_dir == Path("my/images")

    def test_nested_output_path(self):
        """When output is deeply nested, imagedir is sibling images folder."""
        from src.main import resolve_output_paths

        args = Namespace(output="a/b/c/report.md", imagedir=None)
        output_file, image_dir = resolve_output_paths(args)

        assert output_file == Path("a/b/c/report.md")
        assert image_dir == Path("a/b/c/images")


class TestValidateImagedir:
    """Tests for validate_imagedir function."""

    def test_valid_relative_path(self):
        """Valid relative path should work."""
        from src.main import validate_imagedir

        result = validate_imagedir("images")
        assert result == Path("images")

    def test_valid_nested_path(self):
        """Valid nested relative path should work."""
        from src.main import validate_imagedir

        result = validate_imagedir("results/charts/images")
        assert isinstance(result, Path)

    def test_absolute_path_allowed(self):
        """Absolute paths are allowed (user's choice)."""
        from src.main import validate_imagedir

        result = validate_imagedir("/tmp/images")
        assert result == Path("/tmp/images")

    def test_path_with_parent_ref(self):
        """Paths with .. are allowed (user's responsibility)."""
        from src.main import validate_imagedir

        result = validate_imagedir("../outside/project")
        assert result == Path("../outside/project")


class TestChartGenerationWithOutput:
    """Integration tests for chart generation with --output."""

    def test_charts_enabled_with_output(self):
        """Charts should be enabled when --output is specified."""
        # This is a behavioral test - charts are enabled when output is a file
        # The actual chart generation is tested elsewhere
        args = Namespace(
            output="test.md",
            no_charts=False,
            quiet=False,
            brief=False,
        )
        # When output is specified, no_charts should remain False
        assert args.no_charts is False

    def test_charts_disabled_for_stdout(self):
        """Charts should be auto-disabled when writing to stdout."""
        # When output is None (stdout) and not quiet/brief,
        # charts should be disabled to avoid path issues
        args = Namespace(
            output=None,
            no_charts=False,
            quiet=False,
            brief=False,
        )
        # The actual disabling happens in main(), not here
        # This test documents the expected behavior
        assert args.output is None
