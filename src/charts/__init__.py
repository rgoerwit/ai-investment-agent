"""
Charts module for generating valuation visualizations.

This module provides chart generation capabilities for investment reports,
including Football Field valuation charts and Thesis Alignment radar charts.

Post-PM Chart Generation:
Charts are generated AFTER the Portfolio Manager verdict to ensure visuals
align with the final investment decision. The ChartGenerator node uses
PM_BLOCK data when available, falling back to DATA_BLOCK for raw metrics.
"""

from src.charts.base import ChartConfig, ChartFormat, FootballFieldData, RadarChartData
from src.charts.chart_node import create_chart_generator_node

__all__ = [
    "ChartConfig",
    "ChartFormat",
    "FootballFieldData",
    "RadarChartData",
    "create_chart_generator_node",
]
