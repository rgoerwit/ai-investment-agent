"""
Radar chart generator for Thesis Alignment.

Generates a 5-axis radar chart showing:
- Financial Health
- Growth Transition
- Valuation Alignment
- Undiscovered Status
- Safety/Risk Profile
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import structlog

from src.charts.base import ChartConfig, ChartFormat, RadarChartData

logger = structlog.get_logger(__name__)


def generate_radar_chart(
    data: RadarChartData,
    config: ChartConfig | None = None,
) -> Path | None:
    """Generate thesis alignment radar chart.

    Args:
        data: RadarChartData with normalized scores (0-100)
        config: ChartConfig for styling options

    Returns:
        Path to generated image file, or None if generation failed
    """
    config = config or ChartConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Dimensions
    categories = ["Health", "Growth", "Valuation", "Undiscovered", "Safety"]
    values = [
        data.health_score,
        data.growth_score,
        data.valuation_score,
        data.undiscovered_score,
        data.safety_score,
    ]

    # Close the loop for radar chart
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # Create plot
    fig, ax = plt.subplots(
        figsize=(config.width_inches, config.height_inches),
        subplot_kw=({"polar": True}),
    )

    # Handle transparency
    if config.transparent:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))
    else:
        fig.patch.set_facecolor("white")
        ax.patch.set_facecolor("white")

    # Set colors based on transparency to ensure visibility on both dark/light themes
    if config.transparent:
        # Mid-tone colors that avoid extremes
        axis_label_color = "#4A90D9"  # Mid Blue
        data_label_color = "#00B4D8"  # Mid Cyan
        tick_label_color = "#7F7F7F"  # Mid Grey
        title_color = "#4A90D9"  # Mid Blue
    else:
        axis_label_color = "grey"
        data_label_color = "#0077B6"  # Darker Blue
        tick_label_color = "grey"
        title_color = "black"

    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], categories, color=axis_label_color, size=10)

    # Push the axis labels (categories) further out to avoid overlapping with data labels
    # pad=30 moves the text away from the outer circle
    ax.tick_params(axis="x", pad=30)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75], ["25", "50", "75"], color=tick_label_color, size=7)
    plt.ylim(0, 100)

    # Plot data - Using colorblind-safe Cyan (matches PASS color in CLI)
    chart_color = "#00B4D8"  # Cyan/Blue

    ax.plot(angles, values, linewidth=2, linestyle="solid", color=chart_color)

    # Fill area
    ax.fill(angles, values, chart_color, alpha=0.25)

    # Add specific data labels
    # Calculate label positions
    for angle, value, _label in zip(angles[:-1], values[:-1], categories, strict=True):
        # Add value annotation
        # If value is very low, push it out a bit more to be visible
        # If value is high, +15 keeps it distinct from the 100-line
        offset = 15

        ax.text(
            angle,
            value + offset,
            f"{value:.0f}%",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=data_label_color,
        )

    # Title
    # Shortened title to avoid overlap with top dimensions (Growth/Valuation)
    title_text = f"{data.ticker} Thesis Alignment ({data.trade_date})"

    # Use suptitle (figure-level) instead of axes-level title to keep it fixed at the top
    # This prevents overlap with the "Growth" label which pushes out due to padding
    plt.suptitle(title_text, size=10, y=0.98, fontweight="bold", color=title_color)

    # Reserve top space for title (0.92 top limit)
    # This forces the chart and its padded labels to fit in the bottom 92%
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Generate filename
    safe_ticker = data.ticker.replace(".", "_").replace("/", "_")
    filename = f"{safe_ticker}_{data.trade_date}_radar"

    # Save
    if config.format == ChartFormat.SVG:
        output_path = config.output_dir / f"{filename}.svg"
        plt.savefig(
            output_path,
            format="svg",
            dpi=config.dpi,
            transparent=config.transparent,
            bbox_inches="tight",
        )
    else:
        output_path = config.output_dir / f"{filename}.png"
        plt.savefig(
            output_path,
            format="png",
            dpi=config.dpi,
            transparent=config.transparent,
            bbox_inches="tight",
        )

    plt.close(fig)

    logger.info(
        "Generated radar chart",
        ticker=data.ticker,
        output_path=str(output_path),
    )

    return output_path
