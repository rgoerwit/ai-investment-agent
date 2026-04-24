from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from langchain_core.tools import tool

from src.error_safety import safe_error_payload
from src.ibkr.account_service import AccountStatus, IbkrAccountService
from src.ibkr.portfolio_data_service import (
    CashSnapshot,
    IbkrPortfolioDataService,
    PortfolioSnapshot,
    WatchlistSnapshot,
)
from src.ibkr_config import ibkr_config


def _serialize_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value):
        return _serialize_value(asdict(value))
    if isinstance(value, set):
        return sorted(_serialize_value(item) for item in value)
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def _portfolio_service() -> IbkrPortfolioDataService:
    return IbkrPortfolioDataService(config=ibkr_config)


def _account_service() -> IbkrAccountService:
    return IbkrAccountService(config=ibkr_config)


def _ok_payload(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": _serialize_value(data)}


def _error_payload(exc: Exception, *, operation: str) -> dict[str, Any]:
    payload = safe_error_payload(exc, operation=operation, provider="unknown")
    payload["ok"] = False
    return payload


@tool
async def get_ibkr_holdings(account_id: str = "") -> dict[str, Any]:
    """Return normalized live IBKR holdings."""
    try:
        positions = await _portfolio_service().fetch_holdings(
            account_id=account_id or None
        )
        return _ok_payload(positions)
    except Exception as exc:
        return _error_payload(exc, operation="get_ibkr_holdings")


@tool
async def get_ibkr_watchlist(
    account_id: str = "",
    watchlist_name: str = "",
) -> dict[str, Any]:
    """Return the current IBKR watchlist as yfinance-format tickers."""
    try:
        snapshot: WatchlistSnapshot = await _portfolio_service().fetch_watchlist(
            watchlist_name=watchlist_name or None,
            explicitly_requested=bool(watchlist_name),
        )
        return _ok_payload(snapshot)
    except Exception as exc:
        return _error_payload(exc, operation="get_ibkr_watchlist")


@tool
async def get_ibkr_live_orders(account_id: str = "") -> dict[str, Any]:
    """Return current open or pending IBKR live orders."""
    try:
        orders = await _portfolio_service().fetch_live_orders(
            account_id=account_id or None
        )
        return _ok_payload(orders)
    except Exception as exc:
        return _error_payload(exc, operation="get_ibkr_live_orders")


@tool
async def get_ibkr_cash_summary(account_id: str = "") -> dict[str, Any]:
    """Return cash, settled cash, available cash, and portfolio value."""
    try:
        snapshot: CashSnapshot = await _portfolio_service().fetch_cash_snapshot(
            account_id=account_id or None
        )
        return _ok_payload(snapshot)
    except Exception as exc:
        return _error_payload(exc, operation="get_ibkr_cash_summary")


@tool
async def get_ibkr_account_status(account_id: str = "") -> dict[str, Any]:
    """Return account visibility plus a read-only ledger/portfolio summary."""
    try:
        status: AccountStatus = await _account_service().verify_connection(
            account_id=account_id or None,
            include_key_validation=False,
        )
        return _ok_payload(status)
    except Exception as exc:
        return _error_payload(exc, operation="get_ibkr_account_status")


@tool
async def get_ibkr_portfolio_snapshot(
    account_id: str = "",
    watchlist_name: str = "",
    include_live_orders: bool = False,
    cash_buffer_pct: float = 0.05,
) -> dict[str, Any]:
    """Return holdings, portfolio summary, watchlist, and optional live orders."""
    try:
        snapshot: PortfolioSnapshot = await _portfolio_service().fetch_snapshot(
            account_id=account_id or None,
            watchlist_name=watchlist_name or None,
            explicitly_requested=bool(watchlist_name),
            cash_buffer_pct=cash_buffer_pct,
            include_live_orders=include_live_orders,
        )
        return _ok_payload(snapshot)
    except Exception as exc:
        return _error_payload(exc, operation="get_ibkr_portfolio_snapshot")
