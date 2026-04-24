from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.ibkr.account_service import AccountStatus
from src.ibkr.portfolio_data_service import (
    CashSnapshot,
    PortfolioSnapshot,
    WatchlistSnapshot,
)
from src.tools.portfolio import (
    get_ibkr_account_status,
    get_ibkr_cash_summary,
    get_ibkr_holdings,
    get_ibkr_live_orders,
    get_ibkr_portfolio_snapshot,
    get_ibkr_watchlist,
)
from src.tools.registry import Toolkit


@dataclass
class FakePosition:
    ticker: str

    def model_dump(self) -> dict:
        return {"ticker": self.ticker}


class FakePortfolioService:
    async def fetch_holdings(self, *, account_id=None):
        return [FakePosition("7203.T")]

    async def fetch_watchlist(self, *, watchlist_name=None, explicitly_requested=False):
        return WatchlistSnapshot(
            tickers={"7203.T"},
            loaded_name="wl",
            total=1,
            found=True,
            explicitly_requested=explicitly_requested,
        )

    async def fetch_cash_snapshot(self, *, account_id=None):
        return CashSnapshot(
            cash_balance_usd=200.0,
            settled_cash_usd=150.0,
            available_cash_usd=100.0,
            portfolio_value_usd=1000.0,
        )

    async def fetch_snapshot(self, **kwargs):
        return PortfolioSnapshot(
            positions=[],
            watchlist=WatchlistSnapshot(
                tickers={"7203.T"},
                loaded_name="wl",
                total=1,
                found=True,
                explicitly_requested=True,
            ),
            live_orders=[],
            errors={},
        )

    async def fetch_live_orders(self, *, account_id=None):
        return [{"ticker": "7203", "side": "BUY"}]


class FakeAccountService:
    async def verify_connection(self, *, account_id=None, include_key_validation=False):
        return AccountStatus(
            account_id="U123456",
            visible_accounts=["U123456"],
            ledger={},
            key_info={},
            raw_position_count=2,
        )


@pytest.mark.asyncio
async def test_holdings_tool_returns_serializable_payload(monkeypatch):
    monkeypatch.setattr(
        "src.tools.portfolio._portfolio_service", lambda: FakePortfolioService()
    )
    payload = await get_ibkr_holdings.ainvoke({"account_id": ""})
    assert payload["ok"] is True
    assert payload["data"] == [{"ticker": "7203.T"}]


@pytest.mark.asyncio
async def test_watchlist_tool_returns_name_and_tickers(monkeypatch):
    monkeypatch.setattr(
        "src.tools.portfolio._portfolio_service", lambda: FakePortfolioService()
    )
    payload = await get_ibkr_watchlist.ainvoke({"watchlist_name": "wl"})
    assert payload["ok"] is True
    assert payload["data"]["loaded_name"] == "wl"
    assert set(payload["data"]["tickers"]) == {"7203.T"}


@pytest.mark.asyncio
async def test_cash_and_account_tools_return_structured_payload(monkeypatch):
    monkeypatch.setattr(
        "src.tools.portfolio._portfolio_service", lambda: FakePortfolioService()
    )
    monkeypatch.setattr(
        "src.tools.portfolio._account_service", lambda: FakeAccountService()
    )

    cash = await get_ibkr_cash_summary.ainvoke({"account_id": ""})
    account = await get_ibkr_account_status.ainvoke({"account_id": ""})

    assert cash["ok"] is True
    assert cash["data"]["settled_cash_usd"] == 150.0
    assert account["ok"] is True
    assert account["data"]["raw_position_count"] == 2


@pytest.mark.asyncio
async def test_live_orders_tool_returns_structured_payload(monkeypatch):
    monkeypatch.setattr(
        "src.tools.portfolio._portfolio_service", lambda: FakePortfolioService()
    )
    payload = await get_ibkr_live_orders.ainvoke({"account_id": ""})
    assert payload["ok"] is True
    assert payload["data"] == [{"ticker": "7203", "side": "BUY"}]


@pytest.mark.asyncio
async def test_snapshot_tool_returns_error_cleanly(monkeypatch):
    class FailingPortfolioService(FakePortfolioService):
        async def fetch_snapshot(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.tools.portfolio._portfolio_service", lambda: FailingPortfolioService()
    )
    payload = await get_ibkr_portfolio_snapshot.ainvoke({})
    assert payload["ok"] is False
    assert payload["error_type"] == "RuntimeError"
    assert payload["failure_kind"] == "unknown_provider_error"
    assert payload["retryable"] is False
    assert payload["message_preview"] == "boom"
    assert payload["error"].startswith(
        "Error in get_ibkr_portfolio_snapshot: RuntimeError"
    )


def test_registry_exposes_portfolio_tools():
    tools = Toolkit().get_portfolio_tools()
    names = {tool.name for tool in tools}
    assert names == {
        "get_ibkr_holdings",
        "get_ibkr_watchlist",
        "get_ibkr_live_orders",
        "get_ibkr_cash_summary",
        "get_ibkr_account_status",
        "get_ibkr_portfolio_snapshot",
    }
