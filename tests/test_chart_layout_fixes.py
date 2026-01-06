import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

# Set backend to Agg before importing generators to prevent GUI requirement
matplotlib.use("Agg")

from src.charts.base import ChartConfig, ChartFormat, FootballFieldData, RadarChartData
from src.charts.generators.football_field import generate_football_field
from src.charts.generators.radar_chart import generate_radar_chart


class TestChartLayoutFixes(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for output
        self.test_dir = tempfile.mkdtemp()
        self.config = ChartConfig(
            output_dir=Path(self.test_dir),
            format=ChartFormat.PNG,
            dpi=100,
            width_inches=10,
            height_inches=6,
            transparent=False,
        )

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    @patch("src.charts.generators.radar_chart.plt")
    def test_radar_chart_footnote_spacing(self, mock_plt):
        """Test that radar chart reserves space at bottom for footnotes."""
        # Setup mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        data = RadarChartData(
            ticker="TEST",
            trade_date="2026-01-01",
            health_score=50,
            growth_score=50,
            valuation_score=50,
            undiscovered_score=50,
            regulatory_score=50,
            jurisdiction_score=50,
            footnote="Long footnote that needs space",
        )

        generate_radar_chart(data, self.config)

        # Verify tight_layout was called with the specific rect parameter
        # to reserve bottom 8% (0.08) for footnotes
        mock_fig.tight_layout.assert_called_with(rect=[0, 0.08, 1, 0.92])

    @patch("src.charts.generators.football_field.plt")
    def test_football_field_warning_box_spacing(self, mock_plt):
        """Test that football field chart adjusts Y-limits for warning box."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Case 1: With warnings
        data_with_warnings = FootballFieldData(
            ticker="TEST",
            trade_date="2026-01-01",
            current_price=100,
            fifty_two_week_low=80,
            fifty_two_week_high=120,
            quality_warnings=["Warning 1"],
            footnote="Footnote",
        )

        generate_football_field(data_with_warnings, self.config)

        # Expect 1 bar (52-week range) + 0 extra bars (no targets provided).
        # Base top limit = 1 bar - 0.5 = 0.5.
        # With warning, we add 1.0 -> 1.5.
        # So ylim should be set to (-1.0, 1.5)

        # Inspect call args to set_ylim
        args, _ = mock_ax.set_ylim.call_args
        self.assertEqual(args[0], -1.0)
        self.assertEqual(args[1], 2.5)  # 1.5 (base) + 1.0 (padding)

        # Verify text box position (approximate check of call args)
        # We look for the call that adds the warning text
        warning_text_calls = [
            call
            for call in mock_ax.text.call_args_list
            if "DATA QUALITY WARNING" in str(call)
        ]
        self.assertTrue(len(warning_text_calls) > 0)
        # Check coordinates of the warning box (Top Right: 0.98, 0.97)
        args, kwargs = warning_text_calls[0]
        self.assertAlmostEqual(args[0], 0.98)
        self.assertAlmostEqual(args[1], 0.97)
        self.assertEqual(kwargs.get("va"), "top")

    @patch("src.charts.generators.football_field.plt")
    def test_football_field_no_warning_spacing(self, mock_plt):
        """Test that football field chart uses standard limits without warnings."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        data_no_warnings = FootballFieldData(
            ticker="TEST",
            trade_date="2026-01-01",
            current_price=100,
            fifty_two_week_low=80,
            fifty_two_week_high=120,
            quality_warnings=None,
        )

        generate_football_field(data_no_warnings, self.config)

        # 1 bar total (52 week range). Top limit = 1 - 0.5 = 0.5.
        args, _ = mock_ax.set_ylim.call_args
        self.assertEqual(args[1], 0.5)

    def test_file_cleanup_integration(self):
        """Integration test to verify file generation and confirm cleanup capability."""
        # Use real generation with Agg backend (configured at top)

        data = RadarChartData(
            ticker="REAL_TEST",
            trade_date="2026-01-01",
            health_score=50,
            growth_score=50,
            valuation_score=50,
            undiscovered_score=50,
            regulatory_score=50,
            jurisdiction_score=50,
        )

        output_path = generate_radar_chart(data, self.config)

        # Verify file exists
        self.assertTrue(output_path.exists())
        self.assertTrue(output_path.is_file())

        # Cleanup is handled by tearDown (rmtree of self.test_dir),
        # but let's verify individual deletion logic here as requested by user.
        output_path.unlink()
        self.assertFalse(output_path.exists())
