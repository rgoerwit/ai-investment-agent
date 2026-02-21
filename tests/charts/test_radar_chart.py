"""Tests for Radar Chart generator."""

import tempfile
from pathlib import Path

import pytest

from src.charts.base import ChartConfig, ChartFormat, RadarChartData
from src.charts.generators.radar_chart import generate_radar_chart


class TestRadarChartData:
    """Tests for RadarChartData dataclass."""

    def test_all_required_fields(self):
        """Test that all required fields are present."""
        data = RadarChartData(
            ticker="TEST",
            trade_date="2025-01-01",
            health_score=80.0,
            growth_score=60.0,
            valuation_score=70.0,
            undiscovered_score=90.0,
            regulatory_score=85.0,
            jurisdiction_score=75.0,
        )
        assert data.ticker == "TEST"
        assert data.health_score == 80.0
        assert data.regulatory_score == 85.0
        assert data.jurisdiction_score == 75.0

    def test_optional_fields_default_to_none(self):
        """Test that optional fields default to None."""
        data = RadarChartData(
            ticker="TEST",
            trade_date="2025-01-01",
            health_score=80.0,
            growth_score=60.0,
            valuation_score=70.0,
            undiscovered_score=90.0,
            regulatory_score=85.0,
            jurisdiction_score=75.0,
        )
        assert data.pe_ratio is None
        assert data.de_ratio is None
        assert data.roa is None


class TestGenerateRadarChart:
    """Tests for generate_radar_chart function."""

    @pytest.fixture
    def base_data(self):
        """Standard test data for radar chart (6 axes)."""
        return RadarChartData(
            ticker="TEST",
            trade_date="2025-01-01",
            health_score=80.0,
            growth_score=60.0,
            valuation_score=70.0,
            undiscovered_score=90.0,
            regulatory_score=85.0,
            jurisdiction_score=75.0,
            pe_ratio=15.5,
            de_ratio=0.5,
            roa=12.0,
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

    def test_generate_handles_special_ticker_characters(self):
        """Test that generate handles tickers with special characters."""
        data = RadarChartData(
            ticker="0005.HK",
            trade_date="2025-01-01",
            health_score=80.0,
            growth_score=60.0,
            valuation_score=70.0,
            undiscovered_score=90.0,
            regulatory_score=85.0,
            jurisdiction_score=75.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_radar_chart(data, config)

            assert result is not None
            assert result.exists()
            # Dots should be replaced with underscores
            assert "0005_HK" in result.name


class TestRadarChartEdgeCases:
    """Tests for edge cases in radar chart generation."""

    def test_all_scores_at_100(self):
        """Test chart with all scores at maximum."""
        data = RadarChartData(
            ticker="PERFECT",
            trade_date="2025-01-01",
            health_score=100.0,
            growth_score=100.0,
            valuation_score=100.0,
            undiscovered_score=100.0,
            regulatory_score=100.0,
            jurisdiction_score=100.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_radar_chart(data, config)

            assert result is not None
            assert result.exists()

    def test_all_scores_at_0(self):
        """Test chart with all scores at minimum."""
        data = RadarChartData(
            ticker="WORST",
            trade_date="2025-01-01",
            health_score=0.0,
            growth_score=0.0,
            valuation_score=0.0,
            undiscovered_score=0.0,
            regulatory_score=0.0,
            jurisdiction_score=0.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_radar_chart(data, config)

            assert result is not None
            assert result.exists()

    def test_mixed_extreme_scores(self):
        """Test chart with mix of very high and very low scores."""
        data = RadarChartData(
            ticker="MIXED",
            trade_date="2025-01-01",
            health_score=95.0,
            growth_score=5.0,
            valuation_score=100.0,
            undiscovered_score=0.0,
            regulatory_score=50.0,
            jurisdiction_score=25.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_radar_chart(data, config)

            assert result is not None
            assert result.exists()

    def test_scores_outside_range_clamped(self):
        """Test that scores outside 0-100 are clamped."""
        data = RadarChartData(
            ticker="CLAMP",
            trade_date="2025-01-01",
            health_score=150.0,  # Above 100
            growth_score=-20.0,  # Below 0
            valuation_score=100.0,
            undiscovered_score=50.0,
            regulatory_score=80.0,
            jurisdiction_score=60.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_radar_chart(data, config)

            # Should generate without error (clamping happens internally)
            assert result is not None
            assert result.exists()

    def test_all_scores_at_50_neutral(self):
        """Test chart with all neutral (50%) scores."""
        data = RadarChartData(
            ticker="NEUTRAL",
            trade_date="2025-01-01",
            health_score=50.0,
            growth_score=50.0,
            valuation_score=50.0,
            undiscovered_score=50.0,
            regulatory_score=50.0,
            jurisdiction_score=50.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_radar_chart(data, config)

            assert result is not None
            assert result.exists()

    def test_very_low_values_labels_visible(self):
        """Test that labels are visible even with very low values."""
        data = RadarChartData(
            ticker="LOW",
            trade_date="2025-01-01",
            health_score=5.0,
            growth_score=10.0,
            valuation_score=3.0,
            undiscovered_score=8.0,
            regulatory_score=2.0,
            jurisdiction_score=1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_radar_chart(data, config)

            # Chart should generate - labels should be pushed out for visibility
            assert result is not None
            assert result.exists()
