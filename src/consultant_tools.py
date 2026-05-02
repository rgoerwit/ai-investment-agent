"""
Lightweight verification tools for the External Consultant agent.

Design: "Dumb Tool" pattern - tools do pure I/O only.
The Consultant handles all reasoning about whether data conflicts matter.

These tools give the consultant independent access to raw market data,
breaking the circular dependency where all agents rely on the same
Fundamentals Analyst interpretation.

Two spot-check tools:
- spot_check_metric: Fetches from yfinance (same source as DATA_BLOCK pipeline)
- spot_check_metric_alt: Fetches from FMP (independent source for cross-validation)
"""

import asyncio
import json
from typing import Annotated, Any

import structlog
import yfinance as yf
from langchain_core.tools import tool

from src.config import config
from src.error_safety import safe_error_payload, summarize_exception
from src.mcp.errors import MCPCallError, make_mcp_tool_name
from src.runtime_diagnostics import classify_failure
from src.runtime_services import (
    get_current_runtime_services,
    get_current_tool_service,
)
from src.tooling.runtime import ToolInvocation, ToolResult

logger = structlog.get_logger(__name__)

SPOT_CHECK_TIMEOUT_SECONDS = 8.0

# Fields the consultant is allowed to spot-check (decision-critical metrics only)
ALLOWED_FIELDS = frozenset(
    {
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "debtToEquity",
        "returnOnEquity",
        "returnOnAssets",
        "operatingMargins",
        "freeCashflow",
        "operatingCashflow",
        "totalRevenue",
        "netIncomeToCommon",
        "currentPrice",
        "marketCap",
        "dividendYield",
        "payoutRatio",
        "currentRatio",
        "earningsGrowth",
        "revenueGrowth",
    }
)

# Map yfinance field names → FMP API endpoints/fields
FMP_FIELD_MAP: dict[str, tuple[str, str]] = {
    "operatingCashflow": ("cash-flow-statement", "operatingCashFlow"),
    "freeCashflow": ("cash-flow-statement", "freeCashFlow"),
    "netIncomeToCommon": ("income-statement", "netIncome"),
    "totalRevenue": ("income-statement", "revenue"),
    "returnOnEquity": ("ratios", "returnOnEquity"),
    "returnOnAssets": ("ratios", "returnOnAssets"),
    "debtToEquity": ("ratios", "debtEquityRatio"),
    "trailingPE": ("ratios", "priceEarningsRatio"),
    "priceToBook": ("ratios", "priceToBookRatio"),
    "payoutRatio": ("ratios", "payoutRatio"),
    "currentRatio": ("ratios", "currentRatio"),
    "dividendYield": ("ratios", "dividendYield"),
}

FMP_MCP_REQUIRED_TOOLS = frozenset({"statements", "quote"})

# Map yfinance-named metrics → (FMP MCP tool, endpoint enum value).
# FMP MCP uses a dispatcher pattern: each tool takes an ``endpoint`` argument
# that selects the actual operation. Verified empirically via tools/list.
# TTM endpoints chosen for ratios/income/cash-flow because the consultant is
# cross-validating the *current* state, not historical snapshots.
_FMP_METRIC_DISPATCH: dict[str, tuple[str, str]] = {
    "trailingPE": ("statements", "metrics-ratios-ttm"),
    "forwardPE": ("statements", "metrics-ratios-ttm"),
    "priceToBook": ("statements", "metrics-ratios-ttm"),
    "debtToEquity": ("statements", "metrics-ratios-ttm"),
    "returnOnEquity": ("statements", "metrics-ratios-ttm"),
    "returnOnAssets": ("statements", "metrics-ratios-ttm"),
    "operatingMargins": ("statements", "metrics-ratios-ttm"),
    "dividendYield": ("statements", "metrics-ratios-ttm"),
    "payoutRatio": ("statements", "metrics-ratios-ttm"),
    "currentRatio": ("statements", "metrics-ratios-ttm"),
    "freeCashflow": ("statements", "cashflow-statements-ttm"),
    "operatingCashflow": ("statements", "cashflow-statements-ttm"),
    "totalRevenue": ("statements", "income-statements-ttm"),
    "netIncomeToCommon": ("statements", "income-statements-ttm"),
    "currentPrice": ("quote", "quote"),
    "marketCap": ("quote", "quote"),
}


def _fmp_mcp_field_for(metric: str) -> str:
    """Return the response-field name to extract from the FMP MCP payload."""
    if metric in FMP_FIELD_MAP:
        return FMP_FIELD_MAP[metric][1]
    return {"currentPrice": "price", "marketCap": "marketCap"}[metric]


def _build_fmp_access_failure(
    *,
    ticker: str,
    metric: str,
    error: str,
    suggestion: str,
    retryable: bool,
    cooldown_until: str | None = None,
) -> str:
    payload = {
        "error": error,
        "suggestion": suggestion,
        "ticker": ticker,
        "metric": metric,
        "provider": "fmp",
        "failure_kind": "auth_error" if not retryable else "rate_limit",
        "retryable": retryable,
    }
    if cooldown_until is not None:
        payload["cooldown_until"] = cooldown_until
    return json.dumps(payload)


def _build_mcp_access_failure(
    *,
    ticker: str,
    key: str,
    lookup: str,
    provider: str,
    error: str,
    failure_kind: str,
    retryable: bool,
    source: str,
) -> str:
    return json.dumps(
        {
            "error": error,
            "ticker": ticker,
            key: lookup,
            "provider": provider,
            "failure_kind": failure_kind,
            "retryable": retryable,
            "source": source,
        }
    )


async def _execute_mcp_via_tool_service(
    services: Any,
    *,
    server_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    agent_key: str,
    scope: str,
) -> ToolResult:
    """Route an MCP call through the active ToolExecutionService.

    Returns the full ``ToolResult`` from the shared tool plane so caller-facing
    wrappers can distinguish hook-level blocks and preserve structured findings
    without bypassing the normal audit/inspection/budget chokepoint.
    """
    invocation = ToolInvocation(
        name=make_mcp_tool_name(server_id, tool_name),
        args=arguments,
        source="consultant",
        agent_key=agent_key,
    )

    async def runner(args: dict[str, Any]) -> Any:
        return await services.mcp_runtime.execute_raw(
            server_id,
            tool_name,
            args,
            scope=scope,
            agent_key=agent_key,
        )

    return await get_current_tool_service().execute(invocation, runner=runner)


def _mcp_wrapper_available(
    runtime: Any,
    *,
    server_id: str,
    required_tools: frozenset[str],
    scope: str = "consultant",
) -> bool:
    """Return whether a narrow MCP wrapper can be exposed safely."""
    checker = getattr(runtime, "is_tool_available", None)
    if checker is None:
        return False
    return all(
        checker(server_id, tool_name, scope=scope) for tool_name in required_tools
    )


def _classify_mcp_blocked_result(result: ToolResult) -> tuple[str, str]:
    """Map a blocked hook result onto a stable wrapper-facing failure shape."""
    message = result.value if isinstance(result.value, str) else "TOOL_BLOCKED"
    normalized = message.lower()
    if "budget exhausted" in normalized:
        return "budget", message
    return "inspection_blocked", message


def _build_mcp_text_payload(
    *,
    ticker: str,
    key: str,
    lookup: str,
    provider: str,
    source: str,
    text_payload: str,
) -> str:
    return json.dumps(
        {
            "ticker": ticker,
            key: lookup,
            "provider": provider,
            "source": source,
            "text_payload": text_payload,
            "note": "mcp_payload_sanitized_or_textual",
        }
    )


def _extract_candidate_payloads(result: dict[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    for key in ("structured_content", "parsed_text_json"):
        value = result.get(key)
        if value is not None:
            candidates.append(value)
    text_value = result.get("text_content")
    if text_value:
        candidates.append(text_value)
    return candidates


def _find_nested_value(payload: Any, field_name: str) -> Any | None:
    if isinstance(payload, dict):
        if field_name in payload:
            return payload[field_name]
        data = payload.get("data")
        if data is not None:
            found = _find_nested_value(data, field_name)
            if found is not None:
                return found
        for value in payload.values():
            found = _find_nested_value(value, field_name)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_nested_value(item, field_name)
            if found is not None:
                return found
    return None


def _find_period(payload: Any) -> str | None:
    if isinstance(payload, dict):
        period = payload.get("period")
        if isinstance(period, str):
            return period
        data = payload.get("data")
        if data is not None:
            nested = _find_period(data)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _find_period(item)
            if nested is not None:
                return nested
    return None


@tool("spot_check_metric")
async def spot_check_metric(
    ticker: Annotated[str, "Stock ticker (e.g., 7203.T, 0005.HK)"],
    metric: Annotated[str, "Metric name to verify (e.g., trailingPE, debtToEquity)"],
) -> str:
    """
    Fetch a single financial metric directly from yfinance to verify
    a claim in the analyst reports. Use sparingly — only when you suspect
    a specific number is wrong or when DATA_BLOCK and narrative disagree.

    NOTE: This uses the SAME data source (yfinance) as the main pipeline.
    For independent cross-validation, use spot_check_metric_alt (FMP source).

    Returns: JSON with {ticker, metric, value, source} or error.
    """
    if metric not in ALLOWED_FIELDS:
        return json.dumps(
            {
                "error": f"Unknown metric '{metric}'",
                "allowed": sorted(ALLOWED_FIELDS),
            }
        )

    try:
        stock = yf.Ticker(ticker)
        info = await asyncio.wait_for(
            asyncio.to_thread(lambda: stock.info),
            timeout=SPOT_CHECK_TIMEOUT_SECONDS,
        )
        value = info.get(metric)

        return json.dumps(
            {
                "ticker": ticker,
                "metric": metric,
                "value": value,
                "source": "yfinance_direct",
            }
        )
    except asyncio.TimeoutError:
        logger.warning(
            "spot_check_timeout",
            ticker=ticker,
            metric=metric,
            timeout_seconds=SPOT_CHECK_TIMEOUT_SECONDS,
        )
        return json.dumps(
            {
                "error": "Timed out loading yfinance info",
                "ticker": ticker,
                "metric": metric,
            }
        )
    except Exception as e:
        summary = summarize_exception(
            e,
            operation="spot_check_metric",
            provider="unknown",
        )
        logger.warning("spot_check_failed", ticker=ticker, metric=metric, **summary)
        return json.dumps(
            safe_error_payload(
                e,
                operation="spot_check_metric",
                provider="unknown",
                extra={"ticker": ticker, "metric": metric},
            )
        )


@tool("spot_check_metric_alt")
async def spot_check_metric_alt(
    ticker: Annotated[str, "Stock ticker (e.g., 7203.T, 0005.HK)"],
    metric: Annotated[
        str, "Metric name to verify (e.g., operatingCashflow, netIncomeToCommon)"
    ],
) -> str:
    """
    Fetch a single financial metric from Financial Modeling Prep (FMP) as an
    INDEPENDENT alternative source. Use this to cross-validate suspicious values
    from the DATA_BLOCK, which uses yfinance as its primary source.

    Priority metrics for cross-validation: operatingCashflow, freeCashflow,
    netIncomeToCommon (most prone to data source errors for ex-US stocks).

    Returns: JSON with {ticker, metric, value, source: "fmp_direct", fmp_field} or error.
    """
    if metric not in FMP_FIELD_MAP:
        return json.dumps(
            {
                "error": f"Metric '{metric}' not available via FMP alt-source",
                "available_metrics": sorted(FMP_FIELD_MAP.keys()),
                "suggestion": "Use spot_check_metric (yfinance) for this metric instead",
            }
        )

    try:
        from src.data.fmp_fetcher import (
            FMPSubscriptionUnavailableError,
            get_fmp_fetcher,
        )

        fmp = get_fmp_fetcher()
        if not fmp.is_available():
            cooldown_until = getattr(fmp, "_cooldown_until", None)
            if not getattr(fmp, "api_key", None):
                return _build_fmp_access_failure(
                    ticker=ticker,
                    metric=metric,
                    error="FMP alt-source unavailable (no API key configured)",
                    suggestion="spot_check_metric uses yfinance as primary — same source as DATA_BLOCK pipeline",
                    retryable=False,
                )
            return _build_fmp_access_failure(
                ticker=ticker,
                metric=metric,
                error="FMP alt-source temporarily unavailable (cooldown active after quota/rate-limit response)",
                suggestion="Retry later or rely on official filings / primary data until FMP cooldown expires",
                retryable=True,
                cooldown_until=cooldown_until.isoformat() if cooldown_until else None,
            )

        endpoint, fmp_field = FMP_FIELD_MAP[metric]

        # FMP uses plain ticker symbols for most exchanges
        # Exchange suffix mapping for FMP compatibility
        fmp_ticker = ticker
        # FMP uses .T for Tokyo but some need no suffix changes
        # Most international tickers work as-is with FMP

        # FMPFetcher manages its own aiohttp sessions per _get() call —
        # no async context manager needed (or implemented) at the fetcher level.
        data = await fmp._get(endpoint, {"symbol": fmp_ticker, "limit": 1})

        if not data or not isinstance(data, list) or len(data) == 0:
            return json.dumps(
                {
                    "ticker": ticker,
                    "metric": metric,
                    "value": None,
                    "source": "fmp_direct",
                    "fmp_field": fmp_field,
                    "note": "No data returned by FMP for this ticker/metric",
                }
            )

        value = data[0].get(fmp_field)

        return json.dumps(
            {
                "ticker": ticker,
                "metric": metric,
                "value": value,
                "source": "fmp_direct",
                "fmp_field": fmp_field,
                "fmp_endpoint": endpoint,
                "period": data[0].get("period", "Annual"),
            }
        )

    except FMPSubscriptionUnavailableError as e:
        summary = summarize_exception(
            e,
            operation="spot_check_metric_alt",
            provider="unknown",
        )
        logger.debug(
            "spot_check_alt_subscription_unavailable",
            ticker=ticker,
            metric=metric,
            **summary,
        )
        return _build_fmp_access_failure(
            ticker=ticker,
            metric=metric,
            error="FMP alt-source unavailable for this ticker or endpoint",
            suggestion="The current FMP plan does not cover this ticker or endpoint. Use official filings or another primary source instead.",
            retryable=False,
        )
    except ValueError as e:
        # FMP API key validation error
        summary = summarize_exception(
            e,
            operation="spot_check_metric_alt",
            provider="unknown",
        )
        logger.warning(
            "spot_check_alt_key_error",
            ticker=ticker,
            metric=metric,
            **summary,
        )
        return _build_fmp_access_failure(
            ticker=ticker,
            metric=metric,
            error="FMP API key issue",
            suggestion="Check FMP API credentials or use official filings if independent cross-validation is still needed.",
            retryable=False,
        )
    except Exception as e:
        details = classify_failure(e, provider="unknown", model_name="fmp_alt_source")
        safe_payload = safe_error_payload(
            e,
            operation="spot_check_metric_alt",
            provider="unknown",
            extra={
                "ticker": ticker,
                "metric": metric,
                "provider": "fmp",
                "fmp_endpoint": FMP_FIELD_MAP[metric][0],
            },
        )
        logger.warning(
            "spot_check_alt_failed",
            ticker=ticker,
            metric=metric,
            failure_kind=details.kind,
            retryable=details.retryable,
            error_type=details.error_type,
            message_preview=safe_payload.get("message_preview"),
        )
        return json.dumps(safe_payload)


@tool("spot_check_metric_mcp_fmp")
async def spot_check_metric_mcp_fmp(
    ticker: Annotated[str, "Stock ticker (e.g., 7203.T, 0005.HK)"],
    metric: Annotated[str, "Metric name (e.g., trailingPE, debtToEquity)"],
) -> str:
    """Fetch a single financial metric from the **official FMP MCP** server.

    This is an independent, MCP‑based cross‑check that does not rely on
    the main pipeline's yfinance‑driven DATA_BLOCK.  Returns compact JSON.

    Use when you suspect a specific number in the analyst reports is wrong
    or when DATA_BLOCK and narrative claims diverge.
    """
    if metric not in _FMP_METRIC_DISPATCH:
        return json.dumps(
            {
                "error": f"Metric '{metric}' not available via FMP MCP spot-check",
                "available_metrics": sorted(_FMP_METRIC_DISPATCH.keys()),
                "provider": "fmp",
                "source": "fmp_mcp",
            }
        )

    services = get_current_runtime_services()
    if services is None or services.mcp_runtime is None:
        return _build_mcp_access_failure(
            ticker=ticker,
            key="metric",
            lookup=metric,
            provider="fmp",
            error="FMP MCP is not available in the current runtime",
            failure_kind="config_error",
            retryable=False,
            source="fmp_mcp",
        )

    tool_name, endpoint = _FMP_METRIC_DISPATCH[metric]
    metric_field = _fmp_mcp_field_for(metric)

    try:
        result = await _execute_mcp_via_tool_service(
            services,
            server_id="fmp_remote",
            tool_name=tool_name,
            arguments={"symbol": ticker, "endpoint": endpoint},
            agent_key="consultant",
            scope="consultant",
        )
    except MCPCallError as exc:
        return _build_mcp_access_failure(
            ticker=ticker,
            key="metric",
            lookup=metric,
            provider="fmp",
            error=exc.message,
            failure_kind=exc.category.value,
            retryable=exc.retryable,
            source="fmp_mcp",
        )
    except Exception as exc:
        return json.dumps(
            safe_error_payload(
                exc,
                operation="spot_check_metric_mcp_fmp",
                provider="unknown",
                extra={
                    "ticker": ticker,
                    "metric": metric,
                    "provider": "fmp",
                    "source": "fmp_mcp",
                },
            )
        )

    if result.blocked:
        failure_kind, error = _classify_mcp_blocked_result(result)
        return _build_mcp_access_failure(
            ticker=ticker,
            key="metric",
            lookup=metric,
            provider="fmp",
            error=error,
            failure_kind=failure_kind,
            retryable=False,
            source="fmp_mcp",
        )

    value = result.value
    if isinstance(value, str):
        return _build_mcp_text_payload(
            ticker=ticker,
            key="metric",
            lookup=metric,
            provider="fmp",
            source="fmp_mcp",
            text_payload=value,
        )
    if not isinstance(value, dict):
        return _build_mcp_access_failure(
            ticker=ticker,
            key="metric",
            lookup=metric,
            provider="fmp",
            error="unexpected_mcp_payload_shape",
            failure_kind="protocol",
            retryable=False,
            source="fmp_mcp",
        )

    # Vendor-side error (isError=true on CallToolResult): surface text_content
    # as the error rather than falling through to opaque shape-extraction failure.
    if value.get("is_error"):
        return _build_mcp_access_failure(
            ticker=ticker,
            key="metric",
            lookup=metric,
            provider="fmp",
            error=str(value.get("text_content") or "MCP tool returned isError=true"),
            failure_kind="tool_error",
            retryable=False,
            source="fmp_mcp",
        )

    normalized_payload = value
    for payload in _extract_candidate_payloads(normalized_payload):
        extracted = _find_nested_value(payload, metric_field)
        if extracted is not None:
            response = {
                "ticker": ticker,
                "metric": metric,
                "value": extracted,
                "provider": "fmp",
                "source": "fmp_mcp",
                "mcp_tool": tool_name,
                "mcp_endpoint": endpoint,
            }
            period = _find_period(payload)
            if period is not None:
                response["period"] = period
            return json.dumps(response)

    return json.dumps(
        {
            "error": "unexpected_mcp_payload_shape",
            "ticker": ticker,
            "metric": metric,
            "provider": "fmp",
            "source": "fmp_mcp",
            "mcp_tool": tool_name,
            "mcp_endpoint": endpoint,
        }
    )


def get_consultant_tools() -> list:
    """Get the list of tools available to the External Consultant.

    DELIBERATELY excludes spot_check_metric (yfinance) because the main
    pipeline already uses yfinance — verifying yfinance against yfinance is
    circular validation. The consultant gets only independent sources:
    - spot_check_metric_alt: FMP REST (independent of pipeline)
    - get_official_filings: Official filing APIs (EDINET/DART) for ground-truth
    - spot_check_metric_mcp_fmp: FMP via MCP (broader tool surface, same vendor)

    Twelve Data MCP is intentionally not exposed: their public MCP server only
    publishes ``u-tool`` (a free-form AI router) and ``doc-tool`` — neither
    fits the consultant's narrow-allowlist + structured-payload contract.
    """
    from src.tools.research import get_official_filings

    tools: list = [
        spot_check_metric_alt,
        get_official_filings,
    ]

    try:
        services = get_current_runtime_services()
    except Exception:
        services = None

    if (
        services is not None
        and services.mcp_runtime is not None
        and config.consultant_mcp_enabled
    ):
        runtime = services.mcp_runtime
        if _mcp_wrapper_available(
            runtime,
            server_id="fmp_remote",
            required_tools=FMP_MCP_REQUIRED_TOOLS,
        ):
            tools.append(spot_check_metric_mcp_fmp)

    return tools
