"""
IBKR API client wrapper around IBind.

Provides rate-limited access to IBKR REST API via OAuth 1.0a.
Two-tiered session: read-only (portfolio data) vs brokerage (orders).
"""

from __future__ import annotations

import time

import structlog

from src.ibkr.exceptions import (
    IBKRAPIError,
    IBKRAuthError,
    IBKRSessionConflictError,
)
from src.ibkr_config import IbkrSettings

logger = structlog.get_logger(__name__)


class IbkrClient:
    """
    Wrapper around IBind's IbkrClient with rate limiting and error handling.

    Usage:
        client = IbkrClient(settings)
        client.connect()
        positions = client.get_positions()
        client.close()

    Or as context manager:
        with IbkrClient(settings) as client:
            positions = client.get_positions()
    """

    def __init__(self, settings: IbkrSettings | None = None):
        self._settings = settings or IbkrSettings()
        self._ibind_client = None
        self._last_request_time: float = 0.0
        self._min_interval = 1.0 / self._settings.ibkr_rate_limit_per_sec

    def connect(self, brokerage_session: bool = False) -> None:
        """
        Establish connection to IBKR via IBind.

        Args:
            brokerage_session: If True, create a full brokerage session
                (needed for orders, only one per username).
                If False, read-only mode (portfolio data only).

        Raises:
            IBKRAuthError: If credentials are invalid or missing
            ImportError: If ibind is not installed
        """
        if not self._settings.is_configured():
            raise IBKRAuthError(
                "IBKR credentials not configured. "
                "Set IBKR_ACCOUNT_ID, IBKR_OAUTH_CONSUMER_KEY, "
                "IBKR_OAUTH_ENCRYPTION_KEY_FP, IBKR_OAUTH_SIGNATURE_KEY_FP in .env"
            )

        try:
            from ibind import IbkrClient as IBClient
        except ImportError as e:
            raise ImportError(
                "ibind package not installed. Run: poetry install -E ibkr"
            ) from e

        try:
            self._ibind_client = IBClient(
                account_id=self._settings.ibkr_account_id,
                oauth_consumer_key=self._settings.get_oauth_consumer_key(),
                access_token=self._settings.get_oauth_access_token(),
                access_token_secret=self._settings.get_oauth_access_token_secret(),
                encryption_key_fp=self._settings.ibkr_oauth_encryption_key_fp,
                signature_key_fp=self._settings.ibkr_oauth_signature_key_fp,
                dh_prime=self._settings.ibkr_oauth_dh_prime or None,
                start_brokerage_session=brokerage_session,
            )
            logger.info(
                "ibkr_connected",
                account=self._settings.ibkr_account_id,
                brokerage_session=brokerage_session,
            )
        except Exception as e:
            error_str = str(e).lower()
            if "auth" in error_str or "oauth" in error_str or "401" in error_str:
                raise IBKRAuthError(f"Authentication failed: {e}") from e
            if "session" in error_str and "conflict" in error_str:
                raise IBKRSessionConflictError(str(e)) from e
            raise IBKRAPIError(f"Connection failed: {e}") from e

    @property
    def account_id(self) -> str:
        """Return the configured IBKR account ID."""
        return self._settings.ibkr_account_id

    def close(self) -> None:
        """Close the IBKR connection."""
        self._ibind_client = None
        logger.debug("ibkr_disconnected")

    def __enter__(self) -> IbkrClient:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _rate_limit(self) -> None:
        """Enforce rate limit (10 req/sec by default)."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _ensure_connected(self) -> None:
        if self._ibind_client is None:
            raise IBKRAuthError("Not connected. Call connect() first.")

    # ── Portfolio Data (read-only) ──

    def get_accounts(self) -> list[str]:
        """Get list of account IDs."""
        self._ensure_connected()
        self._rate_limit()
        try:
            result = self._ibind_client.portfolio_accounts()
            data = result.data if hasattr(result, "data") else result
            if isinstance(data, list):
                return [
                    a.get("id", a.get("accountId", ""))
                    for a in data
                    if isinstance(a, dict)
                ]
            return (
                [self._settings.ibkr_account_id]
                if self._settings.ibkr_account_id
                else []
            )
        except Exception as e:
            raise IBKRAPIError(f"Failed to fetch accounts: {e}") from e

    def get_positions(self, account_id: str | None = None) -> list[dict]:
        """
        Get portfolio positions for an account.

        Returns list of raw IBKR position dicts.
        """
        self._ensure_connected()
        self._rate_limit()
        acct = account_id or self._settings.ibkr_account_id
        try:
            result = self._ibind_client.portfolio_positions(acct)
            data = result.data if hasattr(result, "data") else result
            return data if isinstance(data, list) else []
        except Exception as e:
            raise IBKRAPIError(f"Failed to fetch positions: {e}") from e

    def get_ledger(self, account_id: str | None = None) -> dict:
        """
        Get account ledger (cash balances, portfolio value).

        Returns raw IBKR ledger dict.
        """
        self._ensure_connected()
        self._rate_limit()
        acct = account_id or self._settings.ibkr_account_id
        try:
            result = self._ibind_client.portfolio_account_ledger(acct)
            data = result.data if hasattr(result, "data") else result
            return data if isinstance(data, dict) else {}
        except Exception as e:
            raise IBKRAPIError(f"Failed to fetch ledger: {e}") from e

    def stock_conid_by_symbol(self, symbol: str) -> dict:
        """
        Resolve stock conid from symbol.

        Returns dict of {symbol: [{conid, exchange, ...}]}.
        """
        self._ensure_connected()
        self._rate_limit()
        try:
            result = self._ibind_client.stock_conid_by_symbol(symbol)
            data = result.data if hasattr(result, "data") else result
            return data if isinstance(data, dict) else {}
        except Exception as e:
            raise IBKRAPIError(f"Failed to resolve conid for {symbol}: {e}") from e

    # ── Order Placement (brokerage session required) ──

    def place_order(self, account_id: str, order: dict) -> dict:
        """
        Place an order via IBKR.

        Requires a brokerage session (connect with brokerage_session=True).

        Args:
            account_id: IBKR account ID
            order: Order dict (from order_builder.build_order_dict)

        Returns:
            IBKR order response dict
        """
        self._ensure_connected()
        self._rate_limit()
        try:
            result = self._ibind_client.place_order(account_id, order)
            data = result.data if hasattr(result, "data") else result
            logger.info(
                "order_placed",
                account=account_id,
                conid=order.get("conid"),
                side=order.get("side"),
                quantity=order.get("quantity"),
            )
            return data if isinstance(data, dict) else {}
        except Exception as e:
            error_str = str(e).lower()
            if "session" in error_str:
                raise IBKRSessionConflictError(str(e)) from e
            raise IBKRAPIError(f"Order placement failed: {e}") from e
