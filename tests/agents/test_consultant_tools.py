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
    get_consultant_tools,
    spot_check_metric,
    spot_check_metric_alt,
    spot_check_metric_mcp_fmp,
)
from src.data.fmp_fetcher import FMPSubscriptionUnavailableError
from src.runtime_services import RuntimeServices, use_runtime_services
from src.tooling.inspection_service import InspectionService
from src.tooling.runtime import ToolExecutionService, ToolResult


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


class TestMCPConsultantTools:
    @pytest.mark.asyncio
    async def test_fmp_mcp_wrapper_returns_value_from_nested_payload(self):
        mock_runtime = MagicMock()
        mock_runtime.execute_raw = AsyncMock(
            return_value={
                "structured_content": {"data": [{"priceEarningsRatio": 14.2}]},
                "payload_profile": "structured_financial",
            }
        )
        services = RuntimeServices(
            tool_service=ToolExecutionService(),
            inspection_service=InspectionService(),
            mcp_runtime=mock_runtime,
        )

        with use_runtime_services(services):
            result = json.loads(
                await spot_check_metric_mcp_fmp.ainvoke(
                    {"ticker": "7203.T", "metric": "trailingPE"}
                )
            )

        assert result["value"] == 14.2
        assert result["provider"] == "fmp"
        assert result["source"] == "fmp_mcp"
        assert result["mcp_tool"] == "statements"
        assert result["mcp_endpoint"] == "metrics-ratios-ttm"
        # MCP call must flow through the canonical mcp__server__tool name with
        # the dispatcher endpoint argument bundled into the args dict.
        mock_runtime.execute_raw.assert_called_once_with(
            "fmp_remote",
            "statements",
            {"symbol": "7203.T", "endpoint": "metrics-ratios-ttm"},
            scope="consultant",
            agent_key="consultant",
        )

    @pytest.mark.asyncio
    async def test_fmp_mcp_wrapper_surfaces_vendor_tool_error(self):
        """When the vendor returns is_error=true (e.g. unknown endpoint), the
        wrapper must surface the text_content as a structured tool_error rather
        than fall through to opaque 'unexpected_mcp_payload_shape'."""
        mock_runtime = MagicMock()
        mock_runtime.execute_raw = AsyncMock(
            return_value={
                "server": "fmp_remote",
                "tool": "statements",
                "is_error": True,
                "payload_profile": "free_text",
                "text_content": "MCP error -32602: endpoint not allowed on free tier",
            }
        )
        services = RuntimeServices(
            tool_service=ToolExecutionService(),
            inspection_service=InspectionService(),
            mcp_runtime=mock_runtime,
        )

        with use_runtime_services(services):
            result = json.loads(
                await spot_check_metric_mcp_fmp.ainvoke(
                    {"ticker": "7203.T", "metric": "trailingPE"}
                )
            )

        assert result["failure_kind"] == "tool_error"
        assert "free tier" in result["error"]
        assert result["source"] == "fmp_mcp"

    @pytest.mark.asyncio
    async def test_fmp_mcp_wrapper_degrades_when_runtime_missing(self):
        with patch(
            "src.consultant_tools.get_current_runtime_services", return_value=None
        ):
            result = json.loads(
                await spot_check_metric_mcp_fmp.ainvoke(
                    {"ticker": "7203.T", "metric": "trailingPE"}
                )
            )

        assert result["failure_kind"] == "config_error"
        assert result["source"] == "fmp_mcp"

    @pytest.mark.asyncio
    async def test_fmp_mcp_wrapper_labels_budget_block_correctly(self):
        with patch(
            "src.consultant_tools._execute_mcp_via_tool_service",
            AsyncMock(
                return_value=ToolResult(
                    value="TOOL_BLOCKED: MCP budget exhausted for fmp_remote",
                    blocked=True,
                    findings=["MCP budget exhausted for fmp_remote"],
                )
            ),
        ):
            services = RuntimeServices(
                tool_service=ToolExecutionService(),
                inspection_service=InspectionService(),
                mcp_runtime=MagicMock(),
            )
            with use_runtime_services(services):
                result = json.loads(
                    await spot_check_metric_mcp_fmp.ainvoke(
                        {"ticker": "7203.T", "metric": "trailingPE"}
                    )
                )

        assert result["failure_kind"] == "budget"
        assert "budget exhausted" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fmp_mcp_wrapper_preserves_sanitized_text_payload(self):
        with patch(
            "src.consultant_tools._execute_mcp_via_tool_service",
            AsyncMock(
                return_value=ToolResult(value="sanitized payload", blocked=False)
            ),
        ):
            services = RuntimeServices(
                tool_service=ToolExecutionService(),
                inspection_service=InspectionService(),
                mcp_runtime=MagicMock(),
            )
            with use_runtime_services(services):
                result = json.loads(
                    await spot_check_metric_mcp_fmp.ainvoke(
                        {"ticker": "7203.T", "metric": "trailingPE"}
                    )
                )

        assert result["text_payload"] == "sanitized payload"
        assert result["note"] == "mcp_payload_sanitized_or_textual"

    def test_get_consultant_tools_hides_mcp_tools_when_disabled(self):
        mock_services = MagicMock(mcp_runtime=object())
        with patch(
            "src.consultant_tools.get_current_runtime_services",
            return_value=mock_services,
        ):
            with patch("src.consultant_tools.config.consultant_mcp_enabled", False):
                tools = get_consultant_tools()

        tool_names = {tool.name for tool in tools}
        assert "spot_check_metric_mcp_fmp" not in tool_names

    def test_get_consultant_tools_hides_mcp_tools_when_servers_are_unavailable(self):
        mock_runtime = MagicMock()
        mock_runtime.is_tool_available.return_value = False
        mock_services = MagicMock(mcp_runtime=mock_runtime)

        with patch(
            "src.consultant_tools.get_current_runtime_services",
            return_value=mock_services,
        ):
            with patch("src.consultant_tools.config.consultant_mcp_enabled", True):
                tools = get_consultant_tools()

        tool_names = {tool.name for tool in tools}
        assert "spot_check_metric_mcp_fmp" not in tool_names

    def test_get_consultant_tools_hides_mcp_tools_without_runtime_capability_check(
        self,
    ):
        mock_services = MagicMock(mcp_runtime=object())

        with patch(
            "src.consultant_tools.get_current_runtime_services",
            return_value=mock_services,
        ):
            with patch("src.consultant_tools.config.consultant_mcp_enabled", True):
                tools = get_consultant_tools()

        tool_names = {tool.name for tool in tools}
        assert "spot_check_metric_mcp_fmp" not in tool_names
