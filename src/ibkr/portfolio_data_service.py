from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.error_safety import (
    format_error_message,
    safe_error_payload,
    summarize_exception,
)
from src.ibkr.client import IbkrClient
from src.ibkr.models import NormalizedPosition, PortfolioSummary
from src.ibkr.portfolio import read_portfolio, read_watchlist
from src.ibkr.portfolio_defaults import DEFAULT_CASH_BUFFER_PCT
from src.ibkr.session_manager import get_ibkr_session_manager
from src.ibkr.types import ProgressCallback

logger = structlog.get_logger(__name__)


def _account_id(value: object) -> str:
    return value if isinstance(value, str) else ""


@dataclass
class WatchlistSnapshot:
    tickers: set[str] = field(default_factory=set)
    loaded_name: str | None = None
    total: int | None = None
    found: bool = True
    explicitly_requested: bool = False
    # True when the watchlist could not be read at all (e.g. the /iserver
    # brokerage session was unavailable) — distinct from a genuinely empty or
    # not-found watchlist. Callers degrade (warn + continue) rather than abort.
    unavailable: bool = False


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

    Connections are not created per call: every method borrows the process-wide
    pooled connection from ``IbkrSessionManager`` (one OAuth session for the run,
    logged out once at teardown). IBKR allows a single brokerage session per
    username, so pooling — not per-thread client confinement — is the correct model;
    the pool serializes the lazy connect and brokerage-session init, and reads are
    safe to issue concurrently over the shared connection.
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
        cash_buffer_pct: float = DEFAULT_CASH_BUFFER_PCT,
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
        cash_buffer_pct: float = DEFAULT_CASH_BUFFER_PCT,
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
        cash_buffer_pct: float = DEFAULT_CASH_BUFFER_PCT,
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
        """Return ``(pooled_client, config)`` from the process-wide session pool.

        The client is shared (one OAuth session, logged out once at teardown) — the
        caller must NOT connect or close it. The prompt-for-missing-secret callback
        runs once, inside the pool's lazy connect.
        """
        config = self._resolve_config()
        manager = get_ibkr_session_manager()
        manager.configure(
            client_cls=self._client_cls,
            config=config,
            prompt_for_missing_secret_fn=self._prompt_for_missing_secret_fn,
        )
        return manager.acquire(), config

    def _fetch_holdings_sync(
        self,
        account_id: str | None,
        cash_buffer_pct: float,
    ) -> list[NormalizedPosition]:
        client, config = self._build_client()
        acct = account_id or _account_id(getattr(config, "ibkr_account_id", ""))
        positions, _ = self._read_portfolio_fn(client, acct, cash_buffer_pct)
        return positions

    def _fetch_portfolio_summary_sync(
        self,
        account_id: str | None,
        cash_buffer_pct: float,
    ) -> PortfolioSummary:
        client, config = self._build_client()
        acct = account_id or getattr(config, "ibkr_account_id", "")
        _, portfolio = self._read_portfolio_fn(client, acct, cash_buffer_pct)
        return portfolio

    def _fetch_watchlist_sync(
        self,
        watchlist_name: str | None,
        explicitly_requested: bool,
    ) -> WatchlistSnapshot:
        client, _config = self._build_client()
        wl_name_hint = (watchlist_name or "") if explicitly_requested else ""
        result = self._read_watchlist_fn(client, wl_name_hint)

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
        return self._get_live_orders(
            client,
            account_id or _account_id(getattr(config, "ibkr_account_id", "")),
        )

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
        acct = account_id or _account_id(getattr(config, "ibkr_account_id", ""))

        def emit(message: str) -> None:
            if progress is not None:
                progress(message)

        # Holdings come from the read-only Portal session (Tier 1) and are the
        # floor for a useful run. If THIS fails the OAuth session itself is dead,
        # so let it propagate (the caller aborts). The watchlist and live orders
        # need the /iserver brokerage session (Tier 2), which can be unavailable
        # (timed out / held by another login) while Tier 1 still works — those
        # degrade to a warning instead of aborting.
        snapshot = PortfolioSnapshot()
        emit("Loading holdings from IBKR...")
        positions, portfolio = self._read_portfolio_fn(client, acct, cash_buffer_pct)
        snapshot.positions = positions
        snapshot.portfolio = portfolio

        wl_name_hint = (watchlist_name or "") if explicitly_requested else ""
        emit("Loading watchlist from IBKR...")
        try:
            wl_result = self._read_watchlist_fn(client, wl_name_hint)
            snapshot.watchlist = self._build_watchlist_snapshot(
                wl_result,
                watchlist_name=watchlist_name,
                explicitly_requested=explicitly_requested,
            )
        except Exception as exc:
            error_payload = safe_error_payload(exc, operation="watchlist_fetch")
            snapshot.errors["watchlist"] = format_error_message(
                operation="watchlist_fetch",
                error_type=str(error_payload["error_type"]),
            )
            snapshot.watchlist = WatchlistSnapshot(
                found=False,
                unavailable=True,
                explicitly_requested=explicitly_requested,
                loaded_name=watchlist_name if explicitly_requested else None,
            )
            logger.warning(
                "ibkr_watchlist_unavailable",
                **summarize_exception(exc, operation="watchlist_fetch"),
            )
            emit("⚠ Watchlist unavailable — continuing with holdings only.")

        if include_live_orders:
            emit("Loading live orders from IBKR...")
            try:
                snapshot.live_orders = self._get_live_orders(client, acct)
            except Exception as exc:
                error_payload = safe_error_payload(exc, operation="live_orders")
                snapshot.errors["live_orders"] = format_error_message(
                    operation="live_orders",
                    error_type=str(error_payload["error_type"]),
                )
                snapshot.live_orders = []
                logger.warning(
                    "ibkr_live_orders_unavailable",
                    **summarize_exception(exc, operation="live_orders"),
                )
                emit("⚠ Live orders unavailable — open-order dedup disabled.")

        return snapshot

    @staticmethod
    def _get_live_orders(client: Any, account_id: str) -> list[dict[str, Any]]:
        orders = client.get_live_orders(account_id=account_id)
        return orders if isinstance(orders, list) else []

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
