"""Tests for Football Field chart generator."""

import tempfile
from pathlib import Path

import pytest

from src.charts.base import ChartConfig, ChartFormat, FootballFieldData
from src.charts.generators.football_field import generate_football_field


class TestFootballFieldData:
    """Tests for FootballFieldData dataclass."""

    def test_has_minimum_data_true(self):
        """Test has_minimum_data returns True when required fields present."""
        data = FootballFieldData(
            ticker="AAPL",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
        )
        assert data.has_minimum_data() is True

    def test_has_minimum_data_false_missing_price(self):
        """Test has_minimum_data returns False when current_price missing."""
        data = FootballFieldData(
            ticker="AAPL",
            trade_date="2025-01-01",
            current_price=0,  # Invalid
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
        )
        assert data.has_minimum_data() is False

    def test_has_minimum_data_false_missing_high(self):
        """Test has_minimum_data returns False when fifty_two_week_high missing."""
        data = FootballFieldData(
            ticker="AAPL",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=0,  # Invalid
            fifty_two_week_low=120.00,
        )
        assert data.has_minimum_data() is False

    def test_has_external_targets_true(self):
        """Test has_external_targets when targets available."""
        data = FootballFieldData(
            ticker="AAPL",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
            external_target_high=190.00,
            external_target_low=160.00,
        )
        assert data.has_external_targets() is True

    def test_has_external_targets_false(self):
        """Test has_external_targets when targets missing."""
        data = FootballFieldData(
            ticker="AAPL",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
        )
        assert data.has_external_targets() is False

    def test_has_our_targets_true(self):
        """Test has_our_targets when targets available."""
        data = FootballFieldData(
            ticker="AAPL",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
            our_target_high=175.00,
            our_target_low=155.00,
        )
        assert data.has_our_targets() is True

    def test_has_our_targets_false(self):
        """Test has_our_targets when targets missing."""
        data = FootballFieldData(
            ticker="AAPL",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
        )
        assert data.has_our_targets() is False


class TestChartConfig:
    """Tests for ChartConfig dataclass."""

    def test_default_values(self):
        """Test ChartConfig has sensible defaults."""
        config = ChartConfig()
        assert config.format == ChartFormat.PNG
        assert config.transparent is False
        assert config.dpi == 300
        assert config.width_inches == 6.0
        assert config.height_inches == 4.0

    def test_string_path_conversion(self):
        """Test ChartConfig converts string paths to Path objects."""
        config = ChartConfig(output_dir="./test_images")
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("./test_images")


class TestGenerateFootballField:
    """Tests for generate_football_field function."""

    def test_generate_png_chart(self):
        """Test generating a PNG chart with minimum data."""
        data = FootballFieldData(
            ticker="TEST",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir), format=ChartFormat.PNG)
            result = generate_football_field(data, config)

            assert result is not None
            assert result.exists()
            assert result.suffix == ".png"
            assert "TEST" in result.name
            assert "2025-01-01" in result.name

    def test_generate_svg_chart(self):
        """Test generating an SVG chart."""
        data = FootballFieldData(
            ticker="TEST",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir), format=ChartFormat.SVG)
            result = generate_football_field(data, config)

            assert result is not None
            assert result.exists()
            assert result.suffix == ".svg"

    def test_generate_transparent_chart(self):
        """Test generating a chart with transparent background."""
        data = FootballFieldData(
            ticker="TEST",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir), transparent=True)
            result = generate_football_field(data, config)

            assert result is not None
            assert result.exists()

    def test_generate_transparent_svg_chart(self):
        """Test generating an SVG chart with transparent background."""
        data = FootballFieldData(
            ticker="TEST",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(
                output_dir=Path(tmpdir), format=ChartFormat.SVG, transparent=True
            )
            result = generate_football_field(data, config)

            assert result is not None
            assert result.exists()
            assert result.suffix == ".svg"

    def test_generate_chart_with_all_data(self):
        """Test generating a chart with all optional data."""
        data = FootballFieldData(
            ticker="AAPL",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
            moving_avg_50=148.00,
            moving_avg_200=142.00,
            external_target_high=190.00,
            external_target_low=160.00,
            external_target_mean=175.00,
            our_target_high=175.00,
            our_target_low=155.00,
            target_methodology="P/E normalization",
            target_confidence="HIGH",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_football_field(data, config)

            assert result is not None
            assert result.exists()

    def test_generate_returns_none_insufficient_data(self):
        """Test that generate returns None with insufficient data."""
        data = FootballFieldData(
            ticker="TEST",
            trade_date="2025-01-01",
            current_price=0,  # Invalid
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_football_field(data, config)

            assert result is None

    def test_generate_creates_output_directory(self):
        """Test that generate creates output directory if it doesn't exist."""
        data = FootballFieldData(
            ticker="TEST",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "images"
            config = ChartConfig(output_dir=nested_dir)
            result = generate_football_field(data, config)

            assert result is not None
            assert nested_dir.exists()
            assert result.exists()

    def test_generate_handles_special_ticker_characters(self):
        """Test that generate handles tickers with special characters."""
        data = FootballFieldData(
            ticker="0005.HK",
            trade_date="2025-01-01",
            current_price=55.00,
            fifty_two_week_high=65.00,
            fifty_two_week_low=45.00,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_football_field(data, config)

            assert result is not None
            assert result.exists()
            # Dots should be replaced with underscores
            assert "0005_HK" in result.name

    def test_generate_filters_unreasonable_targets(self):
        """Test that obviously wrong LLM-calculated targets are filtered out.

        This protects against LLM arithmetic hallucinations that would
        distort the chart scale (e.g., $5000 target for a $50 stock).
        """
        data = FootballFieldData(
            ticker="TEST",
            trade_date="2025-01-01",
            current_price=50.00,
            fifty_two_week_high=60.00,
            fifty_two_week_low=40.00,
            # Reasonable external targets (within bounds)
            external_target_high=70.00,
            external_target_low=55.00,
            # Unreasonable "our targets" - LLM math error (10x current price!)
            our_target_high=500.00,
            our_target_low=400.00,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_football_field(data, config)

            # Chart should still generate (52-week and external are valid)
            assert result is not None
            assert result.exists()

    def test_generate_accepts_reasonable_targets(self):
        """Test that reasonable targets within bounds are included."""
        data = FootballFieldData(
            ticker="TEST",
            trade_date="2025-01-01",
            current_price=100.00,
            fifty_two_week_high=120.00,
            fifty_two_week_low=80.00,
            # Reasonable targets (+50% and +20% from current)
            our_target_high=150.00,
            our_target_low=120.00,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_football_field(data, config)

            assert result is not None
            assert result.exists()

    def test_generate_chart_tight_price_range(self):
        """Test chart generation when 52-week range is very tight (< 2%).

        This validates the minimum padding floor works correctly for
        stocks that have barely moved all year.
        """
        data = FootballFieldData(
            ticker="STABLE",
            trade_date="2025-01-01",
            current_price=100.00,
            fifty_two_week_high=101.00,  # Only 2% total range
            fifty_two_week_low=99.00,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_football_field(data, config)

            assert result is not None
            assert result.exists()
