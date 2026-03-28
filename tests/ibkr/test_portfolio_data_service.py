from __future__ import annotations

import pytest

from src.ibkr.models import PortfolioSummary
from src.ibkr.portfolio_data_service import IbkrPortfolioDataService


class FakeConfig:
    ibkr_account_id = "U123456"


class FakeClient:
    def __init__(self, config):
        self.config = config
        self.connected = False
        self.closed = False

    def connect(self, *, brokerage_session: bool) -> None:
        assert brokerage_session is False
        self.connected = True

    def close(self) -> None:
        self.closed = True

    def get_live_orders(self, account_id: str | None = None) -> list[dict]:
        return [{"ticker": "7203", "side": "BUY", "account_id": account_id}]


def _fake_read_portfolio(client, account_id, cash_buffer):
    assert client.connected is True
    assert account_id == "U123456"
    assert cash_buffer == 0.05
    return (
        [{"ticker": "7203.T", "quantity": 100}],
        PortfolioSummary(
            account_id="U123456",
            portfolio_value_usd=1000.0,
            cash_balance_usd=200.0,
            settled_cash_usd=150.0,
            available_cash_usd=100.0,
        ),
    )


def _fake_read_watchlist(client, watchlist_name):
    assert client.connected is True
    if watchlist_name == "missing":
        return None
    if watchlist_name == "empty":
        return set()
    return {"7203.T", "6758.T"}


@pytest.mark.asyncio
async def test_fetch_holdings_returns_positions_from_current_owner():
    service = IbkrPortfolioDataService(
        config=FakeConfig(),
        client_cls=FakeClient,
        read_portfolio_fn=_fake_read_portfolio,
    )
    positions = await service.fetch_holdings(account_id="U123456", cash_buffer_pct=0.05)
    assert positions == [{"ticker": "7203.T", "quantity": 100}]


@pytest.mark.asyncio
async def test_fetch_watchlist_preserves_not_found_vs_empty():
    service = IbkrPortfolioDataService(
        config=FakeConfig(),
        client_cls=FakeClient,
        read_portfolio_fn=_fake_read_portfolio,
        read_watchlist_fn=_fake_read_watchlist,
    )

    missing = await service.fetch_watchlist(
        watchlist_name="missing",
        explicitly_requested=True,
    )
    empty = await service.fetch_watchlist(
        watchlist_name="empty",
        explicitly_requested=True,
    )

    assert missing.found is False
    assert missing.tickers == set()
    assert empty.found is True
    assert empty.tickers == set()


@pytest.mark.asyncio
async def test_fetch_cash_snapshot_uses_portfolio_summary():
    service = IbkrPortfolioDataService(
        config=FakeConfig(),
        client_cls=FakeClient,
        read_portfolio_fn=_fake_read_portfolio,
    )
    cash = await service.fetch_cash_snapshot(account_id="U123456", cash_buffer_pct=0.05)
    assert cash.cash_balance_usd == 200.0
    assert cash.settled_cash_usd == 150.0
    assert cash.available_cash_usd == 100.0


@pytest.mark.asyncio
async def test_fetch_snapshot_preserves_name_total_and_progress():
    progress: list[str] = []
    service = IbkrPortfolioDataService(
        config=FakeConfig(),
        client_cls=FakeClient,
        read_portfolio_fn=_fake_read_portfolio,
        read_watchlist_fn=_fake_read_watchlist,
    )
    snapshot = await service.fetch_snapshot(
        account_id="U123456",
        watchlist_name="watchlist-2026",
        explicitly_requested=True,
        cash_buffer_pct=0.05,
        include_live_orders=True,
        progress=progress.append,
    )

    assert progress == [
        "Preparing IBKR client...",
        "Connecting to IBKR...",
        "Loading portfolio from IBKR...",
        "Loading watchlist from IBKR...",
        "Loading live orders from IBKR...",
    ]
    assert snapshot.positions == [{"ticker": "7203.T", "quantity": 100}]
    assert snapshot.portfolio.portfolio_value_usd == 1000.0
    assert snapshot.watchlist.tickers == {"7203.T", "6758.T"}
    assert snapshot.watchlist.loaded_name == "watchlist-2026"
    assert snapshot.watchlist.total == 2
    assert snapshot.live_orders == [
        {"ticker": "7203", "side": "BUY", "account_id": "U123456"}
    ]
