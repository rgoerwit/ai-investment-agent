"""Tests for Radar Chart generator."""

import tempfile
from pathlib import Path

import pytest

from src.charts.base import ChartConfig, ChartFormat, RadarChartData
from src.charts.generators.radar_chart import generate_radar_chart


class TestGenerateRadarChart:
    """Tests for generate_radar_chart function."""

    @pytest.fixture
    def base_data(self):
        """Standard test data for radar chart."""
        return RadarChartData(
            ticker="TEST",
            trade_date="2025-01-01",
            health_score=80.0,
            growth_score=60.0,
            valuation_score=70.0,
            undiscovered_score=90.0,
            safety_score=85.0,
            pe_ratio=15.5,
            analyst_count=3,
        )

    def test_generate_png_chart(self, base_data):
        """Test generating a PNG chart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir), format=ChartFormat.PNG)
            result = generate_radar_chart(base_data, config)

            assert result is not None
            assert result.exists()
            assert result.suffix == ".png"
            assert "TEST" in result.name
            assert "2025-01-01" in result.name

    def test_generate_svg_chart(self, base_data):
        """Test generating an SVG chart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir), format=ChartFormat.SVG)
            result = generate_radar_chart(base_data, config)

            assert result is not None
            assert result.exists()
            assert result.suffix == ".svg"

    def test_generate_transparent_chart(self, base_data):
        """Test generating a chart with transparent background."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir), transparent=True)
            result = generate_radar_chart(base_data, config)

            assert result is not None
            assert result.exists()
            # Note: We can't easily verify alpha channel content without PIL/numpy image analysis,
            # but ensuring it runs without error is the primary check here.

    def test_generate_creates_output_directory(self, base_data):
        """Test that generate creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "radar"
            config = ChartConfig(output_dir=nested_dir)
            result = generate_radar_chart(base_data, config)

            assert result is not None
            assert nested_dir.exists()
            assert result.exists()

    def test_generate_handles_special_ticker_characters(self, base_data):
        """Test that generate handles tickers with special characters."""
        data = RadarChartData(
            ticker="0005.HK",
            trade_date="2025-01-01",
            health_score=80.0,
            growth_score=60.0,
            valuation_score=70.0,
            undiscovered_score=90.0,
            safety_score=85.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_radar_chart(data, config)

            assert result is not None
            assert result.exists()
            # Dots should be replaced with underscores
            assert "0005_HK" in result.name
