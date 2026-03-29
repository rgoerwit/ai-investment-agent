from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.ibkr.client import IbkrClient
from src.ibkr.models import PortfolioSummary
from src.ibkr.portfolio import build_portfolio_summary


@dataclass
class AccountStatus:
    account_id: str
    visible_accounts: list[str] = field(default_factory=list)
    ledger: dict[str, Any] = field(default_factory=dict)
    key_info: dict[str, str] = field(default_factory=dict)
    portfolio_summary: PortfolioSummary = field(default_factory=PortfolioSummary)
    raw_position_count: int = 0


class IbkrAccountService:
    """Async wrappers over sync IBKR account/auth operations."""

    def __init__(
        self,
        *,
        config=None,
        client_cls: type[IbkrClient] | None = None,
        build_portfolio_summary_fn: Callable[..., PortfolioSummary] | None = None,
        check_config_fn: Callable[[Any], None] | None = None,
        validate_key_files_fn: Callable[[Any], dict[str, str]] | None = None,
        prompt_for_missing_secret_fn: Callable[[Any], None] | None = None,
    ) -> None:
        self._config = config
        self._client_cls = client_cls or IbkrClient
        self._build_portfolio_summary_fn = (
            build_portfolio_summary_fn or build_portfolio_summary
        )
        self._check_config_fn = check_config_fn
        self._validate_key_files_fn = validate_key_files_fn
        self._prompt_for_missing_secret_fn = prompt_for_missing_secret_fn

    async def fetch_account_ids(self) -> list[str]:
        return await asyncio.to_thread(self._fetch_account_ids_sync)

    async def fetch_ledger(self, *, account_id: str | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._fetch_ledger_sync, account_id)

    async def verify_connection(
        self,
        *,
        account_id: str | None = None,
        include_key_validation: bool = True,
    ) -> AccountStatus:
        return await asyncio.to_thread(
            self._verify_connection_sync, account_id, include_key_validation
        )

    def _resolve_config(self):
        if self._config is not None:
            return self._config

        from src.ibkr_config import ibkr_config

        return ibkr_config

    def _fetch_account_ids_sync(self) -> list[str]:
        config = self._resolve_config()
        if self._prompt_for_missing_secret_fn is not None:
            self._prompt_for_missing_secret_fn(config)
        client = self._client_cls(config)
        client.connect(brokerage_session=False)
        try:
            return client.get_accounts()
        finally:
            client.close()

    def _fetch_ledger_sync(self, account_id: str | None = None) -> dict[str, Any]:
        config = self._resolve_config()
        if self._prompt_for_missing_secret_fn is not None:
            self._prompt_for_missing_secret_fn(config)
        client = self._client_cls(config)
        client.connect(brokerage_session=False)
        acct = account_id or getattr(config, "ibkr_account_id", "")
        try:
            return client.get_ledger(acct)
        finally:
            client.close()

    def _verify_connection_sync(
        self,
        account_id: str | None,
        include_key_validation: bool,
    ) -> AccountStatus:
        config = self._resolve_config()

        if self._check_config_fn is not None:
            self._check_config_fn(config)

        key_info: dict[str, str] = {}
        if include_key_validation and self._validate_key_files_fn is not None:
            key_info = self._validate_key_files_fn(config)

        if self._prompt_for_missing_secret_fn is not None:
            self._prompt_for_missing_secret_fn(config)

        acct = account_id or getattr(config, "ibkr_account_id", "")
        client = self._client_cls(config)
        client.connect(brokerage_session=False)
        try:
            accounts = client.get_accounts()
            ledger = client.get_ledger(acct)
            raw_positions = client.get_positions(acct)
        finally:
            client.close()

        summary = self._build_portfolio_summary_fn(ledger, [], acct)
        return AccountStatus(
            account_id=acct,
            visible_accounts=accounts,
            ledger=ledger,
            key_info=key_info,
            portfolio_summary=summary,
            raw_position_count=len(raw_positions),
        )
