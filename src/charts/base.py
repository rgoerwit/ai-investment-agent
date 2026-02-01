"""
Base classes and data structures for chart generation.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ChartFormat(Enum):
    """Supported chart output formats."""

    PNG = "png"
    SVG = "svg"


@dataclass
class CurrencyFormat:
    """Currency display format for international stocks.

    Handles both prefix currencies (e.g., $100, £100, ¥100) and
    suffix currencies (e.g., 100 zł, 100 kr, 100 Kč).
    """

    symbol: str
    position: str = "prefix"  # "prefix" or "suffix"
    space: bool = False  # Whether to add space between symbol and number

    def format_price(self, value: float) -> str:
        """Format a price value with correct currency placement.

        Args:
            value: The numeric price value

        Returns:
            Formatted string with currency symbol in correct position

        Examples:
            >>> CurrencyFormat("$", "prefix").format_price(65.50)
            '$65.50'
            >>> CurrencyFormat("zł", "suffix", space=True).format_price(42.50)
            '42.50 zł'
        """
        sep = " " if self.space else ""
        if self.position == "prefix":
            return f"{self.symbol}{sep}{value:.2f}"
        else:  # suffix
            return f"{value:.2f}{sep}{self.symbol}"


# Default USD format
DEFAULT_CURRENCY = CurrencyFormat("$", "prefix")


@dataclass
class ChartConfig:
    """Configuration for chart generation."""

    output_dir: Path = field(default_factory=lambda: Path("images"))
    format: ChartFormat = ChartFormat.PNG
    transparent: bool = False  # White grid by default
    dpi: int = 300
    width_inches: float = 6.0
    height_inches: float = 4.0
    filename_stem: str | None = None  # If provided, use as base for image filename

    def __post_init__(self):
        """Ensure output_dir is a Path object."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class FootballFieldData:
    """Data required for football field valuation chart.

    Combines raw facts (from DATA_BLOCK) with calculated targets
    (from Research Manager VALUATION_TARGETS).
    """

    # Identity
    ticker: str
    trade_date: str

    # Raw facts (from DATA_BLOCK - required for chart)
    current_price: float
    fifty_two_week_high: float
    fifty_two_week_low: float

    # Currency formatting (derived from ticker exchange suffix)
    currency_format: CurrencyFormat = field(default_factory=lambda: DEFAULT_CURRENCY)

    # Raw facts (from DATA_BLOCK - optional)
    moving_avg_50: float | None = None
    moving_avg_200: float | None = None

    # External analyst consensus (from DATA_BLOCK, yfinance source)
    external_target_high: float | None = None
    external_target_low: float | None = None
    external_target_mean: float | None = None

    # Our calculated targets (from Research Manager VALUATION_TARGETS)
    our_target_low: float | None = None
    our_target_high: float | None = None
    target_methodology: str | None = None
    target_confidence: str | None = None

    # Data quality warnings to display on chart
    quality_warnings: list[str] | None = None

    # Methodology footnote
    footnote: str | None = None

    def has_minimum_data(self) -> bool:
        """Check if we have enough data to generate chart.

        Minimum requirements:
        - Current price > 0
        - 52-week high > 0
        - 52-week low > 0
        """
        return all(
            [
                self.current_price is not None and self.current_price > 0,
                self.fifty_two_week_high is not None and self.fifty_two_week_high > 0,
                self.fifty_two_week_low is not None and self.fifty_two_week_low > 0,
            ]
        )

    def has_external_targets(self) -> bool:
        """Check if external analyst targets are available."""
        return (
            self.external_target_low is not None
            and self.external_target_high is not None
            and self.external_target_low > 0
            and self.external_target_high > 0
        )

    def has_our_targets(self) -> bool:
        """Check if our calculated targets are available."""
        return (
            self.our_target_low is not None
            and self.our_target_high is not None
            and self.our_target_low > 0
            and self.our_target_high > 0
        )


@dataclass
class RadarChartData:
    """Data required for thesis alignment radar chart.

    Scores are normalized to 0-100 where 100 is best.
    Uses 6 axes for comprehensive thesis alignment visualization:
    - Health: Financial health composite (D/E, ROA influence)
    - Growth: Growth transition score
    - Valuation: P/E and PEG-based value assessment
    - Undiscovered: Low analyst coverage = higher score
    - Regulatory: PFIC, VIE, CMIC, ADR risk factors
    - Jurisdiction: Country/exchange stability
    """

    ticker: str
    trade_date: str

    # Core Dimensions (0-100) - 6 axes
    health_score: float
    growth_score: float
    valuation_score: float
    undiscovered_score: float
    regulatory_score: float  # Split from old "safety" - PFIC/VIE/CMIC/ADR risks
    jurisdiction_score: float  # Split from old "safety" - country/exchange risk

    # Uncertainty markers (asterisk displayed on affected axes)
    axis_warnings: dict[str, bool] = field(default_factory=dict)
    # Keys: "health", "growth", "valuation", "regulatory", "undiscovered", "jurisdiction"

    # Methodology/caveat footnote (always shown if set)
    footnote: str | None = None

    # Raw values for labels and tooltips
    pe_ratio: float | None = None
    peg_ratio: float | None = None
    de_ratio: float | None = None
    roa: float | None = None
    analyst_count: int | None = None
    risk_tally: float | None = None
