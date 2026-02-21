from argparse import Namespace
from pathlib import Path

import pytest

from src.main import resolve_output_paths


class TestCliPathResolution:
    """Test the logic for determining output and image directories."""

    def test_default_stdout(self):
        """Test default behavior (no output file, no image dir)."""
        args = Namespace(output=None, imagedir=None)
        output_file, image_dir = resolve_output_paths(args)

        assert output_file is None
        assert image_dir == Path("images")

    def test_explicit_output_only(self):
        """Test setting --output defaults image dir to subdirectory."""
        args = Namespace(output="results/report.md", imagedir=None)
        output_file, image_dir = resolve_output_paths(args)

        assert output_file == Path("results/report.md")
        assert image_dir == Path("results/images")

    def test_explicit_output_and_imagedir(self):
        """Test setting both --output and --imagedir."""
        args = Namespace(output="results/report.md", imagedir="custom/charts")
        output_file, image_dir = resolve_output_paths(args)

        assert output_file == Path("results/report.md")
        assert image_dir == Path("custom/charts")

    def test_imagedir_only(self):
        """Test setting --imagedir without --output (stdout mode)."""
        args = Namespace(output=None, imagedir="custom/charts")
        output_file, image_dir = resolve_output_paths(args)

        assert output_file is None
        assert image_dir == Path("custom/charts")

    def test_output_in_current_dir(self):
        """Test output file in current directory."""
        args = Namespace(output="report.md", imagedir=None)
        output_file, image_dir = resolve_output_paths(args)

        assert output_file == Path("report.md")
        # Should default to ./images, not just "images" (pathlib handles this, but worth checking equality)
        assert image_dir.name == "images"
        assert image_dir.parent == Path("report.md").parent
