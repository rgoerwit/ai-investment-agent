"""
Football Field valuation chart generator using Seaborn/Matplotlib.

Generates horizontal bar charts showing valuation ranges:
- 52-Week Range (always present)
- External Analyst Consensus (if available)
- Our Target Range (if available)
- Current price marked with vertical line
- Moving averages as reference lines (if available)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import structlog

from src.charts.base import ChartConfig, ChartFormat, FootballFieldData

logger = structlog.get_logger(__name__)


def _is_target_reasonable(
    target: float | None, current_price: float, max_deviation: float = 1.5
) -> bool:
    """Check if a price target is within reasonable bounds.

    LLMs can hallucinate arithmetic results. This filters out obviously wrong
    targets that would distort the chart scale.

    Args:
        target: The target price to validate
        current_price: Current stock price for reference
        max_deviation: Maximum allowed deviation as fraction (1.5 = 150%)
                      Set higher to accommodate volatile "undiscovered" stocks

    Returns:
        True if target is reasonable, False if it's an outlier
    """
    if target is None or target <= 0:
        return False

    # Check if target is within reasonable range of current price
    # Allow targets between 50% below and 300% above current price
    # (generous bounds for volatile small-caps, still catches 10x errors)
    lower_bound = current_price * (1 - max_deviation / 2)  # 25% below at 1.5
    upper_bound = current_price * (1 + max_deviation * 2)  # 400% above at 1.5

    if target < lower_bound or target > upper_bound:
        logger.warning(
            "Target price outside reasonable bounds - likely LLM calculation error",
            target=target,
            current_price=current_price,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        return False

    return True


def _overlay_scenarios(
    ax,
    data: FootballFieldData,
    fmt,
    text_color: str,
) -> float | None:
    """Plot bear/base/bull IV markers + weighted-IV diamond above the existing bars.

    Returns the y-position used for the markers so the caller can extend the
    axis upper limit, or ``None`` when no scenarios are attached (legacy
    single-range view).

    The scenarios object is treated as a duck-typed
    `src.charts.extractors.valuation.ValuationScenarios` — kept as
    ``object | None`` on `FootballFieldData` to avoid a circular import.
    """
    scenarios = data.scenarios
    if scenarios is None:
        return None

    try:
        bear_iv = float(scenarios.bear_iv)
        base_iv = float(scenarios.base_iv)
        bull_iv = float(scenarios.bull_iv)
        weighted_iv = float(scenarios.weighted_iv)
        bear_prob = float(scenarios.bear.probability)
        base_prob = float(scenarios.base.probability)
        bull_prob = float(scenarios.bull.probability)
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning(
            "Scenario overlay skipped: malformed scenarios object",
            error=str(exc),
        )
        return None

    # Reject scenarios with non-positive or non-finite values defensively.
    values = (bear_iv, base_iv, bull_iv, weighted_iv)
    if not all(isinstance(v, float) and v > 0 for v in values):
        logger.warning(
            "Scenario overlay skipped: non-positive IV detected", values=values
        )
        return None

    # Place the marker row just above whatever the topmost bar/warning row is.
    # Caller already extends `top_limit` to accommodate this.
    scenario_y = len(_collect_visible_rows(data)) + 0.1

    # Plot bear / base / bull dots and weighted diamond at scenario_y.
    ax.scatter(
        [bear_iv],
        [scenario_y],
        marker="o",
        color="#E74C3C",  # red
        edgecolors="black",
        s=120,
        zorder=15,
        label=f"Bear IV {fmt(bear_iv)} ({bear_prob:.0f}%)",
    )
    ax.scatter(
        [base_iv],
        [scenario_y],
        marker="o",
        color="#4A90D9",  # blue
        edgecolors="black",
        s=120,
        zorder=15,
        label=f"Base IV {fmt(base_iv)} ({base_prob:.0f}%)",
    )
    ax.scatter(
        [bull_iv],
        [scenario_y],
        marker="o",
        color="#2ECC71",  # green
        edgecolors="black",
        s=120,
        zorder=15,
        label=f"Bull IV {fmt(bull_iv)} ({bull_prob:.0f}%)",
    )
    ax.scatter(
        [weighted_iv],
        [scenario_y],
        marker="D",
        color="black",
        edgecolors="white",
        s=110,
        zorder=16,
        label=f"Weighted IV {fmt(weighted_iv)}",
    )

    # Subtle row label on the left so readers can tell what the markers mean
    # even before scanning the legend.
    ax.text(
        ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else min(values) * 0.9,
        scenario_y,
        "Scenarios",
        ha="right",
        va="center",
        fontsize=8,
        color=text_color,
        style="italic",
    )

    return scenario_y


def _collect_visible_rows(data: FootballFieldData) -> list[str]:
    """Approximate the number of bar rows the renderer will draw.

    Matches the order in `generate_football_field`. Kept in sync with the
    bar-construction block above; only used by the scenario overlay to pick a
    y-position above the topmost bar.
    """
    rows = ["52w"]
    if data.has_external_targets():
        if _is_target_reasonable(
            data.external_target_low, data.current_price
        ) and _is_target_reasonable(data.external_target_high, data.current_price):
            rows.append("ext")
    if data.has_our_targets():
        if _is_target_reasonable(
            data.our_target_low, data.current_price
        ) and _is_target_reasonable(data.our_target_high, data.current_price):
            rows.append("our")
    elif data.quality_warnings:
        rows.append("suspended")
    return rows


def generate_football_field(
    data: FootballFieldData,
    config: ChartConfig | None = None,
) -> Path | None:
    """Generate football field valuation chart.

    Creates a horizontal bar chart showing valuation ranges for the stock.
    Following Cleveland's graphical perception theory (bars enable accurate
    length judgment) and Tufte's data-ink principles.

    Args:
        data: FootballFieldData with price and target information
        config: ChartConfig for styling options (format, transparency, etc.)

    Returns:
        Path to generated image file, or None if insufficient data
    """
    if not data.has_minimum_data():
        logger.warning(
            "Insufficient data for football field chart",
            ticker=data.ticker,
            current_price=data.current_price,
            high=data.fifty_two_week_high,
            low=data.fifty_two_week_low,
        )
        return None

    config = config or ChartConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Set Seaborn style with white grid
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(config.width_inches, config.height_inches))

    # Set colors based on transparency to ensure visibility on both dark/light themes
    if config.transparent:
        text_color = "#4A90D9"  # Mid Blue
        tick_color = "#7F7F7F"  # Mid Grey
        title_color = "#4A90D9"  # Mid Blue
    else:
        text_color = "black"
        tick_color = "black"
        title_color = "black"

    # Handle transparency
    if config.transparent:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    else:
        fig.patch.set_facecolor("white")
        ax.patch.set_facecolor("white")

    # Build bars (bottom to top)
    bars = []
    colors = []
    labels = []

    # 52-Week Range (always present)
    bars.append(
        (data.fifty_two_week_low, data.fifty_two_week_high - data.fifty_two_week_low)
    )
    colors.append("#4A90D9")  # Blue
    labels.append("52-Week Range")

    # External Analyst Range (if available and reasonable)
    if data.has_external_targets():
        ext_low_ok = _is_target_reasonable(data.external_target_low, data.current_price)
        ext_high_ok = _is_target_reasonable(
            data.external_target_high, data.current_price
        )
        if ext_low_ok and ext_high_ok:
            bars.append(
                (
                    data.external_target_low,
                    data.external_target_high - data.external_target_low,
                )
            )
            colors.append("#7B68EE")  # Purple
            labels.append("Analyst Consensus")

    # Our Target Range (if available and reasonable - LLM math can hallucinate)
    if data.has_our_targets():
        our_low_ok = _is_target_reasonable(data.our_target_low, data.current_price)
        our_high_ok = _is_target_reasonable(data.our_target_high, data.current_price)
        if our_low_ok and our_high_ok:
            bars.append(
                (data.our_target_low, data.our_target_high - data.our_target_low)
            )
            colors.append("#2ECC71")  # Green
            label = "Our Target"
            if data.target_confidence:
                label += f" ({data.target_confidence})"
            labels.append(label)
    elif data.quality_warnings:
        # If targets are missing due to quality warnings (skipped valuation), show placeholder
        # Use a dummy range centered on current price to place the label
        bars.append((data.current_price * 0.95, data.current_price * 0.10))
        colors.append("#95A5A6")  # Grey
        labels.append("Valuation Suspended (See Discussion)")

    # Draw horizontal bars
    y_positions = list(range(len(bars)))
    fmt = data.currency_format.format_price  # Currency formatter shorthand
    for i, ((left, width), color, label) in enumerate(
        zip(bars, colors, labels, strict=True)
    ):
        ax.barh(i, width, left=left, height=0.6, color=color, alpha=0.7, label=label)
        # Add range labels at ends of bars
        label_y = i + 0.05
        ax.text(
            left - 0.02 * (data.fifty_two_week_high - data.fifty_two_week_low),
            label_y,
            fmt(left),
            ha="right",
            va="bottom",
            fontsize=8,
            color=text_color,
        )
        ax.text(
            left + width + 0.02 * (data.fifty_two_week_high - data.fifty_two_week_low),
            label_y,
            fmt(left + width),
            ha="left",
            va="bottom",
            fontsize=8,
            color=text_color,
        )

    # Current price line (prominent) - contained within plot area
    ax.axvline(
        x=data.current_price,
        color="#E74C3C",
        linewidth=2.5,
        linestyle="--",
        label=f"Current: {fmt(data.current_price)}",
        ymin=0,
        ymax=0.95,  # Stop before title
        zorder=10,
    )

    # Moving averages (if available) - subtle reference lines
    # Lines limited to bar area, labels placed below x-axis to avoid title clutter
    if data.moving_avg_50 is not None and data.moving_avg_50 > 0:
        ax.axvline(
            x=data.moving_avg_50,
            color="#F39C12",
            linewidth=1,
            linestyle=":",
            alpha=0.7,
            ymin=0,
            ymax=0.90,  # Shorter than current price line
        )
        ax.text(
            data.moving_avg_50,
            -0.52,  # Below the bottom bar (in data coordinates)
            "50MA",
            fontsize=7,
            ha="center",
            va="top",
            color="#F39C12",
            fontweight="bold",
        )

    if data.moving_avg_200 is not None and data.moving_avg_200 > 0:
        ax.axvline(
            x=data.moving_avg_200,
            color="#9B59B6",
            linewidth=1,
            linestyle=":",
            alpha=0.7,
            ymin=0,
            ymax=0.90,  # Shorter than current price line
        )
        ax.text(
            data.moving_avg_200,
            -0.72,  # Below 50MA label (in data coordinates)
            "200MA",
            fontsize=7,
            ha="center",
            va="top",
            color="#9B59B6",
            fontweight="bold",
        )

    # Warning Box (if warnings exist)
    if data.quality_warnings:
        warning_text = "[!] DATA QUALITY WARNING\n" + "\n".join(data.quality_warnings)
        ax.text(
            0.98,
            0.97,  # Top right
            warning_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#fff3cd",
                "edgecolor": "#856404",
                "alpha": 0.9,
            },
            fontsize=6,
            color="#856404",
            zorder=20,  # Ensure on top
        )

    # Scenario valuation overlay (bear/base/bull markers + weighted diamond).
    # Renders only when `data.scenarios` is populated; degrades silently to
    # the legacy bar-only view otherwise.
    scenario_y = _overlay_scenarios(ax, data, fmt, text_color)

    # Methodology Footnote
    if data.footnote:
        fig.text(
            0.5,
            0.01,
            data.footnote,
            ha="center",
            fontsize=7,
            color=text_color,
            style="italic",
        )

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, color=text_color)
    ax.set_xlabel("Price", color=text_color)
    ax.set_title(
        f"{data.ticker} Valuation Range ({data.trade_date})", color=title_color
    )

    # Set tick label colors
    ax.tick_params(axis="both", colors=tick_color)

    # Place legend below chart to avoid any overlap with data
    if config.transparent:
        # Transparent mode: no background fill, but add border for clarity
        legend = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,  # Two columns for compact horizontal layout
            fontsize=8,
        )
        # Set facecolor and edgecolor separately on the frame
        # (passing facecolor="none" to legend() can be overridden by set_alpha)
        frame = legend.get_frame()
        frame.set_facecolor("none")  # Transparent background
        frame.set_edgecolor(text_color)  # Border for visibility
        frame.set_linewidth(1.0)
        plt.setp(legend.get_texts(), color=text_color)
    else:
        legend = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,  # Two columns for compact horizontal layout
            fontsize=8,
            framealpha=0.9,
        )

    # Determine x-axis range with padding
    all_values = [data.fifty_two_week_low, data.fifty_two_week_high, data.current_price]
    if data.external_target_low:
        all_values.append(data.external_target_low)
    if data.external_target_high:
        all_values.append(data.external_target_high)
    if data.our_target_low:
        all_values.append(data.our_target_low)
    if data.our_target_high:
        all_values.append(data.our_target_high)
    if data.moving_avg_50:
        all_values.append(data.moving_avg_50)
    if data.moving_avg_200:
        all_values.append(data.moving_avg_200)
    if data.scenarios is not None:
        for attr in ("bear_iv", "base_iv", "bull_iv", "weighted_iv"):
            value = getattr(data.scenarios, attr, None)
            if isinstance(value, int | float) and value > 0:
                all_values.append(float(value))

    min_val, max_val = min(all_values), max(all_values)
    padding = (max_val - min_val) * 0.25  # 25% padding for labels
    # Ensure minimum padding for tight ranges (at least 5% of current price)
    min_padding = data.current_price * 0.05
    padding = max(padding, min_padding)
    # Clamp lower bound to 0 to avoid negative prices on chart (penny stocks edge case)
    ax.set_xlim(max(0, min_val - padding), max_val + padding)

    # Set y-axis limits with padding below for MA labels (two rows)
    # If warnings exist, add extra space at top for the warning box
    top_limit = len(bars) - 0.5
    if data.quality_warnings:
        top_limit += 1.0
    if scenario_y is not None:
        # Make room for the scenario markers row and its label.
        top_limit = max(top_limit, scenario_y + 0.8)

    ax.set_ylim(-1.0, top_limit)

    # Use OO API for thread-safety (avoid plt global state)
    fig.tight_layout()

    # Generate filename - use config.filename_stem if provided, else ticker_date
    if config.filename_stem:
        filename = f"{config.filename_stem}_football_field"
    else:
        safe_ticker = data.ticker.replace(".", "_").replace("/", "_")
        filename = f"{safe_ticker}_{data.trade_date}_football_field"

    # Save in requested format (use fig.savefig for OO API)
    if config.format == ChartFormat.SVG:
        output_path = config.output_dir / f"{filename}.svg"
        fig.savefig(
            output_path,
            format="svg",
            dpi=config.dpi,
            transparent=config.transparent,
            bbox_inches="tight",
        )
    else:
        output_path = config.output_dir / f"{filename}.png"
        fig.savefig(
            output_path,
            format="png",
            dpi=config.dpi,
            transparent=config.transparent,
            bbox_inches="tight",
        )

    plt.close(fig)

    logger.info(
        "Generated football field chart",
        ticker=data.ticker,
        output_path=str(output_path),
        format=config.format.value,
    )

    return output_path
