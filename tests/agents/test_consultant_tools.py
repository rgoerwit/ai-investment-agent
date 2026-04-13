"""
Tests for consultant verification tools.

Tests both spot_check_metric (yfinance) and spot_check_metric_alt (FMP) tools.
"""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consultant_tools import (
    FMP_FIELD_MAP,
    spot_check_metric,
    spot_check_metric_alt,
)
from src.data.fmp_fetcher import FMPSubscriptionUnavailableError


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

    @pytest.mark.asyncio
    async def test_timeout_returns_error_payload(self):
        """Stalled yfinance access should return structured timeout JSON."""

        async def slow_to_thread(_func):
            await asyncio.sleep(1)

        with patch("src.consultant_tools.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.info = {"trailingPE": 12.5}
            with patch("src.consultant_tools.SPOT_CHECK_TIMEOUT_SECONDS", 0.01):
                with patch(
                    "src.consultant_tools.asyncio.to_thread",
                    side_effect=slow_to_thread,
                ):
                    result = json.loads(
                        await spot_check_metric.ainvoke(
                            {"ticker": "7203.T", "metric": "trailingPE"}
                        )
                    )

        assert result["ticker"] == "7203.T"
        assert result["metric"] == "trailingPE"
        assert "timed out" in result["error"].lower()


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
        mock_fmp.api_key = None

        with patch("src.data.fmp_fetcher.get_fmp_fetcher", return_value=mock_fmp):
            result = json.loads(
                await spot_check_metric_alt.ainvoke(
                    {"ticker": "7203.T", "metric": "operatingCashflow"}
                )
            )
        assert "error" in result
        assert "unavailable" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fmp_cooldown_returns_rate_limit_reason(self):
        """Cooldown state should not be mislabeled as missing API key."""
        mock_fmp = MagicMock()
        mock_fmp.is_available.return_value = False
        mock_fmp.api_key = "configured-key"
        mock_fmp._cooldown_until = datetime.now() + timedelta(minutes=5)

        with patch("src.data.fmp_fetcher.get_fmp_fetcher", return_value=mock_fmp):
            result = json.loads(
                await spot_check_metric_alt.ainvoke(
                    {"ticker": "7203.T", "metric": "operatingCashflow"}
                )
            )

        assert result["provider"] == "fmp"
        assert result["failure_kind"] == "rate_limit"
        assert result["retryable"] is True
        assert "cooldown" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fmp_returns_valid_data(self):
        """Mock FMP response returns correct format (no async context manager)."""
        mock_fmp = MagicMock()
        mock_fmp.is_available.return_value = True
        mock_fmp._get = AsyncMock(
            return_value=[{"operatingCashFlow": 7_800_000_000, "period": "FY"}]
        )

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

        with patch("src.data.fmp_fetcher.get_fmp_fetcher", return_value=mock_fmp):
            result = json.loads(
                await spot_check_metric_alt.ainvoke(
                    {"ticker": "UNKNOWN.T", "metric": "netIncomeToCommon"}
                )
            )

        assert result["value"] is None
        assert "No data" in result.get("note", "")

    @pytest.mark.asyncio
    async def test_fmp_no_async_context_manager_needed(self):
        """FMPFetcher without __aenter__/__aexit__ must not crash."""
        mock_fmp = MagicMock(spec=["is_available", "_get"])
        mock_fmp.is_available.return_value = True
        mock_fmp._get = AsyncMock(
            return_value=[{"operatingCashFlow": 5_000_000, "period": "FY"}]
        )

        with patch("src.data.fmp_fetcher.get_fmp_fetcher", return_value=mock_fmp):
            result = json.loads(
                await spot_check_metric_alt.ainvoke(
                    {"ticker": "EVO.ST", "metric": "operatingCashflow"}
                )
            )

        assert result["value"] == 5_000_000
        assert result["source"] == "fmp_direct"

    @pytest.mark.asyncio
    async def test_fmp_invalid_key_returns_structured_failure(self):
        """Configuration failures should be explicit and machine-readable."""
        mock_fmp = MagicMock()
        mock_fmp.is_available.return_value = True
        mock_fmp._get = AsyncMock(side_effect=ValueError("invalid or expired"))

        with patch("src.data.fmp_fetcher.get_fmp_fetcher", return_value=mock_fmp):
            result = json.loads(
                await spot_check_metric_alt.ainvoke(
                    {"ticker": "1308.HK", "metric": "operatingCashflow"}
                )
            )

        assert result["provider"] == "fmp"
        assert result["failure_kind"] == "auth_error"
        assert result["retryable"] is False

    @pytest.mark.asyncio
    async def test_fmp_subscription_failure_returns_non_retryable_auth_error(
        self, caplog
    ):
        """Subscription/paywall failures should not look retryable and must not warn."""
        import logging

        mock_fmp = MagicMock()
        mock_fmp.is_available.return_value = True
        mock_fmp._get = AsyncMock(
            side_effect=FMPSubscriptionUnavailableError(
                "current FMP plan does not cover this ticker or endpoint"
            )
        )

        with patch("src.data.fmp_fetcher.get_fmp_fetcher", return_value=mock_fmp):
            with caplog.at_level(logging.WARNING, logger="src.consultant_tools"):
                result = json.loads(
                    await spot_check_metric_alt.ainvoke(
                        {"ticker": "AGS.SI", "metric": "operatingCashflow"}
                    )
                )

        assert result["provider"] == "fmp"
        assert result["failure_kind"] == "auth_error"
        assert result["retryable"] is False
        assert "current fmp plan does not cover" in result["suggestion"].lower()

        # Subscription limits are operator-known — must not surface as warnings.
        warning_events = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING
            and "spot_check_alt_subscription_unavailable" in r.message
        ]
        assert (
            warning_events == []
        ), "FMPSubscriptionUnavailableError should log at debug, not warning"

    @pytest.mark.asyncio
    async def test_fmp_generic_failure_returns_endpoint_details(self):
        """Unexpected failures should still identify the endpoint and retryability."""
        mock_fmp = MagicMock()
        mock_fmp.is_available.return_value = True
        mock_fmp._get = AsyncMock(side_effect=RuntimeError("429 Too Many Requests"))

        with patch("src.data.fmp_fetcher.get_fmp_fetcher", return_value=mock_fmp):
            result = json.loads(
                await spot_check_metric_alt.ainvoke(
                    {"ticker": "1308.HK", "metric": "operatingCashflow"}
                )
            )

        assert result["provider"] == "fmp"
        assert result["failure_kind"] == "rate_limit"
        assert result["retryable"] is True
        assert result["fmp_endpoint"] == "cash-flow-statement"

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
