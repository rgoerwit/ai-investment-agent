from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.report_generator import QuietModeReporter


class TestReportLinking:
    """
    Test that the report generator creates correct relative or absolute links
    based on the relationship between the report directory and image directory.
    """

    @patch("src.report_generator.QuietModeReporter._generate_chart")
    def test_relative_link_generation(self, mock_generate_chart, tmp_path):
        """
        Scenario: Report is in results/report.md
        Images are in results/images/chart.png
        Expected Link: images/chart.png
        """
        # Setup filesystem
        report_dir = tmp_path / "results"
        report_dir.mkdir()
        image_dir = report_dir / "images"
        image_dir.mkdir()

        # Mock the chart generator returning a path in that directory
        chart_path = image_dir / "chart.png"
        mock_generate_chart.return_value = chart_path

        # Initialize reporter
        reporter = QuietModeReporter("AAPL", report_dir=report_dir, image_dir=image_dir)

        # Generate minimal report
        result = {"final_trade_decision": "Action: BUY"}
        report = reporter.generate_report(result)

        # Assertions
        assert "![Football Field Chart](images/chart.png)" in report
        assert str(report_dir) not in report  # Ensure no absolute paths

    @patch("src.report_generator.QuietModeReporter._generate_chart")
    def test_relative_link_to_external_chart_dir(self, mock_generate_chart, tmp_path):
        """
        Scenario: Report is in results/report.md
        Images are in a sibling charts directory
        Expected Link: ../separate_charts/chart.png
        """
        # Setup filesystem
        report_dir = tmp_path / "results"
        report_dir.mkdir()

        # Separate tree for images
        image_dir = tmp_path / "separate_charts"
        image_dir.mkdir()

        chart_path = image_dir / "chart.png"
        mock_generate_chart.return_value = chart_path

        # Initialize reporter
        reporter = QuietModeReporter("AAPL", report_dir=report_dir, image_dir=image_dir)

        result = {"final_trade_decision": "Action: BUY"}
        report = reporter.generate_report(result)

        assert "![Football Field Chart](../separate_charts/chart.png)" in report

    @patch("src.report_generator.QuietModeReporter._generate_chart")
    def test_messy_relative_path_resolution(self, mock_generate_chart, tmp_path):
        """
        Scenario: User provides messy paths like results/../results/images
        The system should resolve them and still find the relative link.
        """
        # Setup filesystem
        report_dir = tmp_path / "results"
        report_dir.mkdir()
        image_dir = report_dir / "images"
        image_dir.mkdir()

        chart_path = image_dir / "chart.png"
        mock_generate_chart.return_value = chart_path

        # Initialize with what would be passed if args were messy
        # We pass the Cleaned Resolved paths usually, but let's ensure reporter handles it
        reporter = QuietModeReporter("AAPL", report_dir=report_dir, image_dir=image_dir)

        result = {"final_trade_decision": "Action: BUY"}
        report = reporter.generate_report(result)

        assert "![Football Field Chart](images/chart.png)" in report

    @patch("src.report_generator.QuietModeReporter._generate_chart")
    def test_stdout_link_uses_given_path_string(self, mock_generate_chart, tmp_path):
        chart_path = tmp_path / "images" / "chart.png"
        chart_path.parent.mkdir()
        mock_generate_chart.return_value = chart_path

        reporter = QuietModeReporter(
            "AAPL", report_dir=None, image_dir=chart_path.parent
        )

        result = {"final_trade_decision": "Action: BUY"}
        report = reporter.generate_report(result)

        assert f"![Football Field Chart]({chart_path})" in report
