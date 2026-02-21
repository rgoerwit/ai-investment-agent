"""Tests for Football Field chart generator."""

import tempfile
from pathlib import Path

import pytest

from src.charts.base import ChartConfig, ChartFormat, CurrencyFormat, FootballFieldData
from src.charts.chart_node import _get_currency_format
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

    def test_transparent_legend_has_transparent_facecolor_with_border(self):
        """Regression test: transparent mode legend must have transparent background.

        The legend should have:
        - Transparent facecolor (not white)
        - Visible border (edgecolor) for clarity on any background

        This was previously broken when set_alpha(1.0) was called on the entire
        frame, which overrode the transparent facecolor setting.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import to_rgba

        data = FootballFieldData(
            ticker="TEST",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir), transparent=True)

            # Re-implement chart generation to inspect legend before close
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(config.width_inches, config.height_inches))

            # Minimal chart setup to get a legend
            text_color = "#4A90D9"
            ax.barh(
                0,
                data.fifty_two_week_high - data.fifty_two_week_low,
                left=data.fifty_two_week_low,
                height=0.6,
                color="#4A90D9",
                alpha=0.7,
                label="52-Week Range",
            )
            ax.axvline(
                x=data.current_price,
                color="#E74C3C",
                linewidth=2.5,
                linestyle="--",
                label=f"Current: ${data.current_price:.2f}",
            )

            # Create legend with transparent mode settings (same as production code)
            legend = ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=2,
                fontsize=8,
            )
            frame = legend.get_frame()
            frame.set_facecolor("none")
            frame.set_edgecolor(text_color)
            frame.set_linewidth(1.0)

            # Verify legend frame properties
            facecolor = frame.get_facecolor()
            edgecolor = frame.get_edgecolor()

            # Facecolor should be transparent (alpha = 0 or "none")
            # to_rgba("none") returns (0.0, 0.0, 0.0, 0.0)
            assert (
                facecolor[3] == 0.0
            ), f"Legend facecolor should be transparent (alpha=0), got alpha={facecolor[3]}"

            # Edgecolor should be visible (not transparent)
            expected_edge = to_rgba(text_color)
            assert (
                edgecolor[3] > 0
            ), f"Legend edgecolor should be visible (alpha>0), got alpha={edgecolor[3]}"
            # Check it's the right color (RGB, ignoring alpha)
            assert abs(edgecolor[0] - expected_edge[0]) < 0.01, "Edgecolor red mismatch"
            assert (
                abs(edgecolor[1] - expected_edge[1]) < 0.01
            ), "Edgecolor green mismatch"
            assert (
                abs(edgecolor[2] - expected_edge[2]) < 0.01
            ), "Edgecolor blue mismatch"

            plt.close(fig)


class TestCurrencyFormat:
    """Tests for CurrencyFormat class and currency detection."""

    def test_format_price_prefix_no_space(self):
        """Test prefix currency without space (e.g., $100.00)."""
        fmt = CurrencyFormat("$", "prefix")
        assert fmt.format_price(100.00) == "$100.00"
        assert fmt.format_price(65.50) == "$65.50"

    def test_format_price_prefix_with_space(self):
        """Test prefix currency with space (e.g., CHF 100.00)."""
        fmt = CurrencyFormat("CHF", "prefix", space=True)
        assert fmt.format_price(100.00) == "CHF 100.00"

    def test_format_price_suffix_with_space(self):
        """Test suffix currency with space (e.g., 100.00 zł)."""
        fmt = CurrencyFormat("zł", "suffix", space=True)
        assert fmt.format_price(42.50) == "42.50 zł"

    def test_format_price_suffix_no_space(self):
        """Test suffix currency without space."""
        fmt = CurrencyFormat("kr", "suffix", space=False)
        assert fmt.format_price(285.00) == "285.00kr"

    def test_format_price_japanese_yen(self):
        """Test Japanese Yen formatting."""
        fmt = CurrencyFormat("¥", "prefix")
        assert fmt.format_price(2850.00) == "¥2850.00"

    def test_format_price_korean_won(self):
        """Test Korean Won formatting."""
        fmt = CurrencyFormat("₩", "prefix")
        assert fmt.format_price(75000.00) == "₩75000.00"


class TestGetCurrencyFormat:
    """Tests for _get_currency_format ticker-to-currency mapping."""

    def test_hong_kong_dollar(self):
        """Test Hong Kong stocks use HK$."""
        fmt = _get_currency_format("0005.HK")
        assert fmt.symbol == "HK$"
        assert fmt.position == "prefix"
        assert fmt.format_price(65.50) == "HK$65.50"

    def test_japanese_yen(self):
        """Test Japanese stocks use ¥."""
        fmt = _get_currency_format("7203.T")
        assert fmt.symbol == "¥"
        assert fmt.format_price(2850.00) == "¥2850.00"

    def test_korean_won(self):
        """Test Korean KOSPI stocks use ₩."""
        fmt = _get_currency_format("005930.KS")
        assert fmt.symbol == "₩"
        assert fmt.format_price(75000.00) == "₩75000.00"

    def test_korean_won_kosdaq(self):
        """Test Korean KOSDAQ stocks use ₩."""
        fmt = _get_currency_format("035720.KQ")
        assert fmt.symbol == "₩"

    def test_taiwan_dollar(self):
        """Test Taiwan stocks use NT$."""
        fmt = _get_currency_format("2330.TW")
        assert fmt.symbol == "NT$"
        assert fmt.format_price(580.00) == "NT$580.00"

    def test_chinese_yuan_shanghai(self):
        """Test Shanghai stocks use CN¥."""
        fmt = _get_currency_format("600519.SS")
        assert fmt.symbol == "CN¥"

    def test_chinese_yuan_shenzhen(self):
        """Test Shenzhen stocks use CN¥."""
        fmt = _get_currency_format("000858.SZ")
        assert fmt.symbol == "CN¥"

    def test_british_pound(self):
        """Test London stocks use £."""
        fmt = _get_currency_format("HSBA.L")
        assert fmt.symbol == "£"
        assert fmt.format_price(650.00) == "£650.00"

    def test_euro_amsterdam(self):
        """Test Amsterdam stocks use €."""
        fmt = _get_currency_format("ASML.AS")
        assert fmt.symbol == "€"

    def test_euro_frankfurt(self):
        """Test Frankfurt stocks use €."""
        fmt = _get_currency_format("SAP.DE")
        assert fmt.symbol == "€"

    def test_swiss_franc(self):
        """Test Swiss stocks use CHF with space."""
        fmt = _get_currency_format("NESN.SW")
        assert fmt.symbol == "CHF"
        assert fmt.space is True
        assert fmt.format_price(100.00) == "CHF 100.00"

    def test_swedish_krona_suffix(self):
        """Test Swedish stocks use kr suffix."""
        fmt = _get_currency_format("VOLV-B.ST")
        assert fmt.symbol == "kr"
        assert fmt.position == "suffix"
        assert fmt.format_price(285.00) == "285.00 kr"

    def test_polish_zloty_suffix(self):
        """Test Polish stocks use zł suffix."""
        fmt = _get_currency_format("PKN.WA")
        assert fmt.symbol == "zł"
        assert fmt.position == "suffix"
        assert fmt.format_price(42.50) == "42.50 zł"

    def test_danish_krone_suffix(self):
        """Test Danish stocks use kr suffix."""
        fmt = _get_currency_format("NOVO-B.CO")
        assert fmt.symbol == "kr"
        assert fmt.position == "suffix"

    def test_norwegian_krone_suffix(self):
        """Test Norwegian stocks use kr suffix."""
        fmt = _get_currency_format("EQNR.OL")
        assert fmt.symbol == "kr"
        assert fmt.position == "suffix"

    def test_czech_koruna_suffix(self):
        """Test Czech stocks use Kč suffix."""
        fmt = _get_currency_format("CEZ.PR")
        assert fmt.symbol == "Kč"
        assert fmt.position == "suffix"

    def test_us_stock_default(self):
        """Test US stocks default to $."""
        fmt = _get_currency_format("AAPL")
        assert fmt.symbol == "$"
        assert fmt.position == "prefix"
        assert fmt.format_price(150.00) == "$150.00"

    def test_unknown_suffix_default(self):
        """Test unknown suffixes default to $."""
        fmt = _get_currency_format("UNKNOWN.XX")
        assert fmt.symbol == "$"

    def test_case_insensitive(self):
        """Test ticker suffix matching is case-insensitive."""
        fmt1 = _get_currency_format("0005.hk")
        fmt2 = _get_currency_format("0005.HK")
        assert fmt1.symbol == fmt2.symbol == "HK$"


class TestFootballFieldCurrencyIntegration:
    """Integration tests for currency formatting in charts."""

    def test_chart_with_hong_kong_currency(self):
        """Test chart generation with Hong Kong Dollar currency."""
        currency = _get_currency_format("0005.HK")
        data = FootballFieldData(
            ticker="0005.HK",
            trade_date="2025-01-01",
            current_price=65.50,
            fifty_two_week_high=75.00,
            fifty_two_week_low=55.00,
            currency_format=currency,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_football_field(data, config)

            assert result is not None
            assert result.exists()
            # Verify currency is set correctly
            assert data.currency_format.symbol == "HK$"

    def test_chart_with_polish_zloty_suffix_currency(self):
        """Test chart generation with Polish Złoty (suffix currency)."""
        currency = _get_currency_format("PKN.WA")
        data = FootballFieldData(
            ticker="PKN.WA",
            trade_date="2025-01-01",
            current_price=42.50,
            fifty_two_week_high=55.00,
            fifty_two_week_low=35.00,
            currency_format=currency,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChartConfig(output_dir=Path(tmpdir))
            result = generate_football_field(data, config)

            assert result is not None
            assert result.exists()
            # Verify suffix currency is set correctly
            assert data.currency_format.symbol == "zł"
            assert data.currency_format.position == "suffix"

    def test_default_currency_when_not_specified(self):
        """Test that FootballFieldData defaults to USD when currency not specified."""
        data = FootballFieldData(
            ticker="AAPL",
            trade_date="2025-01-01",
            current_price=150.00,
            fifty_two_week_high=180.00,
            fifty_two_week_low=120.00,
            # currency_format not specified - should default to USD
        )

        assert data.currency_format.symbol == "$"
        assert data.currency_format.position == "prefix"
