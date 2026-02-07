"""
Tests for consultant verification tools.

Tests both spot_check_metric (yfinance) and spot_check_metric_alt (FMP) tools.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consultant_tools import (
    FMP_FIELD_MAP,
    spot_check_metric,
    spot_check_metric_alt,
)


class TestSpotCheckMetric:
    """Tests for yfinance-based spot check tool."""

    @pytest.mark.asyncio
    async def test_unknown_metric_returns_error(self):
        """Unknown metric returns error with allowed list."""
        result = json.loads(
            await spot_check_metric.ainvoke(
                {"ticker": "7203.T", "metric": "bogusMetric"}
            )
        )
        assert "error" in result
        assert "allowed" in result

    @pytest.mark.asyncio
    async def test_valid_metric_returns_result(self):
        """Valid metric returns structured result."""
        mock_info = {"trailingPE": 12.5}
        with patch("src.consultant_tools.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.info = mock_info
            result = json.loads(
                await spot_check_metric.ainvoke(
                    {"ticker": "7203.T", "metric": "trailingPE"}
                )
            )
        assert result["ticker"] == "7203.T"
        assert result["metric"] == "trailingPE"
        assert result["source"] == "yfinance_direct"


class TestSpotCheckMetricAlt:
    """Tests for FMP-based alt-source spot check tool."""

    @pytest.mark.asyncio
    async def test_unknown_metric_returns_error(self):
        """Unknown metric returns error with available metrics list."""
        result = json.loads(
            await spot_check_metric_alt.ainvoke(
                {"ticker": "7203.T", "metric": "bogusMetric"}
            )
        )
        assert "error" in result
        assert "available_metrics" in result

    @pytest.mark.asyncio
    async def test_fmp_unavailable_returns_message(self):
        """FMP API key missing returns clear unavailable message."""
        mock_fmp = MagicMock()
        mock_fmp.is_available.return_value = False

        with patch("src.data.fmp_fetcher.get_fmp_fetcher", return_value=mock_fmp):
            result = json.loads(
                await spot_check_metric_alt.ainvoke(
                    {"ticker": "7203.T", "metric": "operatingCashflow"}
                )
            )
        assert "error" in result
        assert "unavailable" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fmp_returns_valid_data(self):
        """Mock FMP response returns correct format."""
        mock_fmp = MagicMock()
        mock_fmp.is_available.return_value = True
        mock_fmp._get = AsyncMock(
            return_value=[{"operatingCashFlow": 7_800_000_000, "period": "FY"}]
        )
        mock_fmp.__aenter__ = AsyncMock(return_value=mock_fmp)
        mock_fmp.__aexit__ = AsyncMock(return_value=None)

        with patch("src.data.fmp_fetcher.get_fmp_fetcher", return_value=mock_fmp):
            result = json.loads(
                await spot_check_metric_alt.ainvoke(
                    {"ticker": "2767.T", "metric": "operatingCashflow"}
                )
            )

        assert result["ticker"] == "2767.T"
        assert result["metric"] == "operatingCashflow"
        assert result["value"] == 7_800_000_000
        assert result["source"] == "fmp_direct"
        assert result["fmp_field"] == "operatingCashFlow"

    @pytest.mark.asyncio
    async def test_fmp_no_data_returns_null_value(self):
        """FMP returns empty data → value is None."""
        mock_fmp = MagicMock()
        mock_fmp.is_available.return_value = True
        mock_fmp._get = AsyncMock(return_value=[])
        mock_fmp.__aenter__ = AsyncMock(return_value=mock_fmp)
        mock_fmp.__aexit__ = AsyncMock(return_value=None)

        with patch("src.data.fmp_fetcher.get_fmp_fetcher", return_value=mock_fmp):
            result = json.loads(
                await spot_check_metric_alt.ainvoke(
                    {"ticker": "UNKNOWN.T", "metric": "netIncomeToCommon"}
                )
            )

        assert result["value"] is None
        assert "No data" in result.get("note", "")

    def test_fmp_field_map_covers_critical_metrics(self):
        """FMP field map includes the most error-prone metrics."""
        critical = ["operatingCashflow", "freeCashflow", "netIncomeToCommon"]
        for metric in critical:
            assert metric in FMP_FIELD_MAP, f"{metric} missing from FMP_FIELD_MAP"

    def test_fmp_field_map_has_valid_tuples(self):
        """All FMP field map entries are (endpoint, field) tuples."""
        for metric, mapping in FMP_FIELD_MAP.items():
            assert isinstance(mapping, tuple), f"{metric} mapping is not a tuple"
            assert len(mapping) == 2, f"{metric} mapping should have 2 elements"
            endpoint, field = mapping
            assert isinstance(endpoint, str), f"{metric} endpoint should be str"
            assert isinstance(field, str), f"{metric} field should be str"
