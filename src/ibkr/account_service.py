from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from src.ibkr.client import IbkrClient
from src.ibkr.models import PortfolioSummary
from src.ibkr.portfolio import build_portfolio_summary
from src.ibkr.session_manager import get_ibkr_session_manager


def _account_id(value: object) -> str:
    return value if isinstance(value, str) else ""


def _validated_account_ids(payload: object) -> list[str]:
    if not isinstance(payload, list):
        raise TypeError("IBKR get_accounts returned a non-list payload")
    if any(not isinstance(account, str) for account in payload):
        raise TypeError("IBKR get_accounts returned a non-string account ID")
    return cast(list[str], payload)


def _validated_ledger(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("IBKR get_ledger returned a non-mapping payload")
    return cast(dict[str, Any], payload)


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

    def _build_client(self):
        """Return ``(pooled_client, config)`` from the shared session pool.

        The client is reused process-wide and logged out once at teardown — never
        connect or close it here. The prompt-for-missing-secret callback runs once,
        inside the pool's lazy connect.
        """
        config = self._resolve_config()
        manager = get_ibkr_session_manager()
        manager.configure(
            client_cls=self._client_cls,
            config=config,
            prompt_for_missing_secret_fn=self._prompt_for_missing_secret_fn,
        )
        return manager.acquire(), config

    def _fetch_account_ids_sync(self) -> list[str]:
        client, _config = self._build_client()
        return _validated_account_ids(client.get_accounts())

    def _fetch_ledger_sync(self, account_id: str | None = None) -> dict[str, Any]:
        client, config = self._build_client()
        acct = account_id or _account_id(getattr(config, "ibkr_account_id", ""))
        return _validated_ledger(client.get_ledger(acct))

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

        acct = account_id or _account_id(getattr(config, "ibkr_account_id", ""))
        # Pool handles the prompt-for-missing-secret callback at connect time.
        client, _config = self._build_client()
        accounts = _validated_account_ids(client.get_accounts())
        ledger = _validated_ledger(client.get_ledger(acct))
        raw_positions = client.get_positions(acct)

        summary = self._build_portfolio_summary_fn(ledger, [], acct)
        return AccountStatus(
            account_id=acct,
            visible_accounts=accounts,
            ledger=ledger,
            key_info=key_info,
            portfolio_summary=summary,
            raw_position_count=len(raw_positions),
        )
