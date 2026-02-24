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

# Known IBKR error payloads → human-readable hint
_IBKR_ERROR_HINTS: dict[str, str] = {
    "invalid consumer": (
        "Consumer key not recognized by IBKR. "
        "Verify IBKR_OAUTH_CONSUMER_KEY in .env exactly matches the 9-character key "
        "you entered in the IBKR OAuth portal. "
        "If the key was just configured, IBKR can take 24h+ (sometimes a weekend "
        "server restart) to activate it."
    ),
    "invalid token": ("Access token rejected. Check IBKR_OAUTH_ACCESS_TOKEN in .env."),
    "invalid signature": (
        "OAuth signature verification failed. "
        "The signature key file may be wrong or mismatched with the public key "
        "uploaded to IBKR."
    ),
    "token expired": (
        "Access token has expired. Generate a new token in the IBKR Client Portal."
    ),
    "session conflict": (
        "A brokerage session is already open under this account. "
        "Only one brokerage session is allowed at a time."
    ),
}


def _parse_ibkr_error(raw: str) -> str:
    """
    Convert ibind's verbose error string to a concise, actionable message.

    ibind formats errors as:
      "IbkrClient: response error Result(...) :: 401 :: Unauthorized :: {JSON}"
    We extract the JSON payload's 'error' field and map it to a hint.
    """
    import json
    import re

    # ibind puts the IBKR response JSON last: "... :: {\"error\":\"...\",\"statusCode\":N}"
    # Use findall with a non-nested pattern and take the last match.
    json_blobs = re.findall(r"\{[^{}]*\}", raw)
    ibkr_error: str | None = None
    for blob in reversed(json_blobs):
        try:
            payload = json.loads(blob)
            if "error" in payload:
                ibkr_error = payload["error"]
                break
        except json.JSONDecodeError:
            continue

    if ibkr_error:
        lower = ibkr_error.lower()
        for key, hint in _IBKR_ERROR_HINTS.items():
            if key in lower:
                return hint
        # Unknown JSON error — surface it cleanly without the ibind noise
        return f"IBKR rejected the request: {ibkr_error}"

    # No JSON — fall back to the raw string, stripped of ibind's boilerplate
    cleaned = re.sub(r"IbkrClient:\s*response error Result\([^)]*\)\s*::\s*", "", raw)
    return cleaned.strip() or raw


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
                "IBKR credentials not configured. Required in .env: "
                "IBKR_ACCOUNT_ID, IBKR_OAUTH_CONSUMER_KEY (9-char string you chose), "
                "IBKR_OAUTH_ACCESS_TOKEN, IBKR_OAUTH_ENCRYPTION_KEY_FP, "
                "IBKR_OAUTH_SIGNATURE_KEY_FP, IBKR_OAUTH_DH_PRIME_FP (or _DH_PRIME)"
            )

        try:
            from ibind import IbkrClient as IBClient
            from ibind.oauth.oauth1a import OAuth1aConfig
        except ImportError as e:
            raise ImportError(
                "ibind package not installed. Run: poetry install -E ibkr"
            ) from e

        try:
            # ibind requires credentials bundled into an OAuth1aConfig dataclass.
            # init_oauth=True triggers the live-session-token handshake inside
            # IBClient.__init__(), so the connection is live after this call.
            # maintain_oauth=False: no background tickler (we close after each use).
            oauth_kwargs: dict = {
                "access_token": self._settings.get_oauth_access_token(),
                "access_token_secret": self._settings.get_oauth_access_token_secret(),
                "consumer_key": self._settings.get_oauth_consumer_key(),
                "encryption_key_fp": self._settings.ibkr_oauth_encryption_key_fp
                or None,
                "signature_key_fp": self._settings.ibkr_oauth_signature_key_fp or None,
                "init_oauth": True,
                "init_brokerage_session": brokerage_session,
                "maintain_oauth": False,
                "shutdown_oauth": False,  # we call close() manually; skip atexit logout
            }
            # ibind requires dh_prime (no built-in default).
            # get_oauth_dh_prime_hex() normalises Base64 DER → hex (ibind requires hex).
            dh_prime_hex = self._settings.get_oauth_dh_prime_hex()
            if dh_prime_hex:
                oauth_kwargs["dh_prime"] = dh_prime_hex

            oauth_config = OAuth1aConfig(**oauth_kwargs)
            self._ibind_client = IBClient(
                account_id=self._settings.ibkr_account_id,
                use_oauth=True,
                oauth_config=oauth_config,
            )
            logger.info(
                "ibkr_connected",
                account=self._settings.ibkr_account_id,
                brokerage_session=brokerage_session,
            )
        except Exception as e:
            # ibind registers an atexit handler in __init__. If OAuth fails,
            # live_session_token is never set, so the atexit logout() call
            # crashes with AttributeError. Neutralise it here.
            if self._ibind_client is not None:
                try:
                    self._ibind_client.close = lambda *a, **kw: None
                except Exception:
                    pass
            self._ibind_client = None

            error_str = str(e)
            friendly = _parse_ibkr_error(error_str)
            lower = error_str.lower()
            if "auth" in lower or "oauth" in lower or "401" in lower:
                raise IBKRAuthError(friendly) from e
            if "session" in lower and "conflict" in lower:
                raise IBKRSessionConflictError(friendly) from e
            raise IBKRAPIError(friendly) from e

    @property
    def account_id(self) -> str:
        """Return the configured IBKR account ID."""
        return self._settings.ibkr_account_id

    def close(self) -> None:
        """Close the IBKR connection."""
        if self._ibind_client is not None:
            try:
                self._ibind_client.close()
            except Exception:
                pass
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
            result = self._ibind_client.positions(account_id=acct)
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
            result = self._ibind_client.get_ledger(account_id=acct)
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
            from ibind.client.ibkr_utils import OrderRequest, QuestionType

            order_request = OrderRequest(
                conid=order.get("conid"),
                side=order["side"],
                quantity=order["quantity"],
                order_type=order.get("orderType", "LMT"),
                acct_id=order.get("acctId", account_id),
                price=order.get("price"),
                tif=order.get("tif", "GTC"),
            )
            # Auto-confirm all IBKR pre-trade confirmation questions.
            answers = dict.fromkeys(QuestionType, True)
            result = self._ibind_client.place_order(
                order_request=order_request,
                answers=answers,
                account_id=account_id,
            )
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
