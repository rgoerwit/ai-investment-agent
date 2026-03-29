from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.ibkr.client import IbkrClient
from src.ibkr.models import NormalizedPosition, PortfolioSummary
from src.ibkr.portfolio import read_portfolio, read_watchlist
from src.ibkr.types import ProgressCallback


@dataclass
class WatchlistSnapshot:
    tickers: set[str] = field(default_factory=set)
    loaded_name: str | None = None
    total: int | None = None
    found: bool = True
    explicitly_requested: bool = False


@dataclass
class CashSnapshot:
    cash_balance_usd: float = 0.0
    settled_cash_usd: float = 0.0
    available_cash_usd: float = 0.0
    portfolio_value_usd: float = 0.0


@dataclass
class PortfolioSnapshot:
    positions: list[NormalizedPosition] = field(default_factory=list)
    portfolio: PortfolioSummary = field(default_factory=PortfolioSummary)
    watchlist: WatchlistSnapshot = field(default_factory=WatchlistSnapshot)
    live_orders: list[dict[str, Any]] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


class IbkrPortfolioDataService:
    """Async portfolio-data service over the sync IBKR client.

    Important: a single IbkrClient instance must not be shared across multiple
    concurrent threads. Integrated snapshot methods therefore run sequentially
    within one thread-confined client session.
    """

    def __init__(
        self,
        *,
        config=None,
        client_cls: type[IbkrClient] | None = None,
        read_portfolio_fn: Callable[
            ..., tuple[list[NormalizedPosition], PortfolioSummary]
        ]
        | None = None,
        read_watchlist_fn: Callable[..., set[str] | None] | None = None,
        prompt_for_missing_secret_fn: Callable[[Any], None] | None = None,
    ) -> None:
        self._config = config
        self._client_cls = client_cls or IbkrClient
        self._read_portfolio_fn = read_portfolio_fn or read_portfolio
        self._read_watchlist_fn = read_watchlist_fn or read_watchlist
        self._prompt_for_missing_secret_fn = prompt_for_missing_secret_fn

    async def fetch_holdings(
        self,
        *,
        account_id: str | None = None,
        cash_buffer_pct: float = 0.05,
    ) -> list[NormalizedPosition]:
        return await asyncio.to_thread(
            self._fetch_holdings_sync,
            account_id,
            cash_buffer_pct,
        )

    async def fetch_portfolio_summary(
        self,
        *,
        account_id: str | None = None,
        cash_buffer_pct: float = 0.05,
    ) -> PortfolioSummary:
        return await asyncio.to_thread(
            self._fetch_portfolio_summary_sync,
            account_id,
            cash_buffer_pct,
        )

    async def fetch_watchlist(
        self,
        *,
        watchlist_name: str | None,
        explicitly_requested: bool,
    ) -> WatchlistSnapshot:
        return await asyncio.to_thread(
            self._fetch_watchlist_sync,
            watchlist_name,
            explicitly_requested,
        )

    async def fetch_live_orders(
        self,
        *,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._fetch_live_orders_sync, account_id)

    async def fetch_cash_snapshot(
        self,
        *,
        account_id: str | None = None,
        cash_buffer_pct: float = 0.05,
    ) -> CashSnapshot:
        summary = await self.fetch_portfolio_summary(
            account_id=account_id,
            cash_buffer_pct=cash_buffer_pct,
        )
        return CashSnapshot(
            cash_balance_usd=summary.cash_balance_usd,
            settled_cash_usd=summary.settled_cash_usd,
            available_cash_usd=summary.available_cash_usd,
            portfolio_value_usd=summary.portfolio_value_usd,
        )

    async def fetch_snapshot(
        self,
        *,
        account_id: str | None,
        watchlist_name: str | None,
        explicitly_requested: bool,
        cash_buffer_pct: float,
        include_live_orders: bool,
        progress: ProgressCallback | None = None,
    ) -> PortfolioSnapshot:
        return await asyncio.to_thread(
            self._fetch_snapshot_sync,
            account_id,
            watchlist_name,
            explicitly_requested,
            cash_buffer_pct,
            include_live_orders,
            progress,
        )

    def _resolve_config(self):
        if self._config is not None:
            return self._config

        from src.ibkr_config import ibkr_config

        return ibkr_config

    def _build_client(self):
        config = self._resolve_config()
        if self._prompt_for_missing_secret_fn is not None:
            self._prompt_for_missing_secret_fn(config)
        return self._client_cls(config), config

    def _fetch_holdings_sync(
        self,
        account_id: str | None,
        cash_buffer_pct: float,
    ) -> list[NormalizedPosition]:
        client, config = self._build_client()
        acct = account_id or getattr(config, "ibkr_account_id", "")
        client.connect(brokerage_session=False)
        try:
            positions, _ = self._read_portfolio_fn(client, acct, cash_buffer_pct)
            return positions
        finally:
            client.close()

    def _fetch_portfolio_summary_sync(
        self,
        account_id: str | None,
        cash_buffer_pct: float,
    ) -> PortfolioSummary:
        client, config = self._build_client()
        acct = account_id or getattr(config, "ibkr_account_id", "")
        client.connect(brokerage_session=False)
        try:
            _, portfolio = self._read_portfolio_fn(client, acct, cash_buffer_pct)
            return portfolio
        finally:
            client.close()

    def _fetch_watchlist_sync(
        self,
        watchlist_name: str | None,
        explicitly_requested: bool,
    ) -> WatchlistSnapshot:
        client, _config = self._build_client()
        client.connect(brokerage_session=False)
        try:
            wl_name_hint = watchlist_name if explicitly_requested else ""
            result = self._read_watchlist_fn(client, wl_name_hint)
        finally:
            client.close()

        return self._build_watchlist_snapshot(
            result,
            watchlist_name=watchlist_name,
            explicitly_requested=explicitly_requested,
        )

    def _fetch_live_orders_sync(
        self,
        account_id: str | None,
    ) -> list[dict[str, Any]]:
        client, config = self._build_client()
        client.connect(brokerage_session=False)
        try:
            return self._get_live_orders(
                client,
                account_id or getattr(config, "ibkr_account_id", ""),
            )
        finally:
            client.close()

    def _fetch_snapshot_sync(
        self,
        account_id: str | None,
        watchlist_name: str | None,
        explicitly_requested: bool,
        cash_buffer_pct: float,
        include_live_orders: bool,
        progress: ProgressCallback | None,
    ) -> PortfolioSnapshot:
        client, config = self._build_client()
        acct = account_id or getattr(config, "ibkr_account_id", "")

        def emit(message: str) -> None:
            if progress is not None:
                progress(message)

        emit("Preparing IBKR client...")
        emit("Connecting to IBKR...")
        client.connect(brokerage_session=False)

        snapshot = PortfolioSnapshot()
        try:
            emit("Loading portfolio from IBKR...")
            positions, portfolio = self._read_portfolio_fn(
                client, acct, cash_buffer_pct
            )
            snapshot.positions = positions
            snapshot.portfolio = portfolio

            wl_name_hint = watchlist_name if explicitly_requested else ""
            emit("Loading watchlist from IBKR...")
            wl_result = self._read_watchlist_fn(client, wl_name_hint)
            snapshot.watchlist = self._build_watchlist_snapshot(
                wl_result,
                watchlist_name=watchlist_name,
                explicitly_requested=explicitly_requested,
            )

            if include_live_orders:
                emit("Loading live orders from IBKR...")
                try:
                    snapshot.live_orders = self._get_live_orders(client, acct)
                except Exception as exc:
                    snapshot.errors["live_orders"] = str(exc)
                    snapshot.live_orders = []
        finally:
            client.close()

        return snapshot

    @staticmethod
    def _get_live_orders(client: Any, account_id: str) -> list[dict[str, Any]]:
        return client.get_live_orders(account_id=account_id)

    @staticmethod
    def _build_watchlist_snapshot(
        result: set[str] | None,
        *,
        watchlist_name: str | None,
        explicitly_requested: bool,
    ) -> WatchlistSnapshot:
        if result is None:
            return WatchlistSnapshot(
                tickers=set(),
                loaded_name=watchlist_name if explicitly_requested else None,
                total=None,
                found=False,
                explicitly_requested=explicitly_requested,
            )

        if not result:
            return WatchlistSnapshot(
                tickers=set(),
                loaded_name=None,
                total=None,
                found=True,
                explicitly_requested=explicitly_requested,
            )

        return WatchlistSnapshot(
            tickers=result,
            loaded_name=watchlist_name if explicitly_requested else None,
            total=len(result),
            found=True,
            explicitly_requested=explicitly_requested,
        )
