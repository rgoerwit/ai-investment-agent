"""
IBKR API client wrapper around IBind.

Provides rate-limited access to IBKR REST API via OAuth 1.0a.
Two-tiered session: read-only (portfolio data) vs brokerage (orders).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import structlog

from src.error_safety import summarize_exception
from src.ibkr.exceptions import (
    IBKRAPIError,
    IBKRAuthError,
    IBKRError,
    IBKRSessionConflictError,
)
from src.ibkr.throttle import IBKRThrottle
from src.ibkr_config import IbkrSettings

logger = structlog.get_logger(__name__)

# After ssodh/init the brokerage session often takes a beat to flip to
# authenticated (the "errored once, worked on re-run" symptom), so poll the
# status a few times before giving up. Status-only — never re-inits, so it can't
# bump a healthy or competing session.
# ssodh/init can answer {"wait": N} — the brokerage session is establishing
# asynchronously (often while bumping a competing session); the completed session
# only shows up on a subsequent init/status. So re-init (ssodh/init IS the
# documented re-auth) a few times, honoring the wait hint, rather than treating the
# transient not-authenticated/"wait" state as terminal. Empirically a second init
# ~1s after a "wait" authenticates where the first did not.
_BROKERAGE_INIT_ATTEMPTS = 4
_BROKERAGE_AUTH_POLL_INTERVAL_S = 1.0  # floor between re-inits
_BROKERAGE_INIT_WAIT_CAP_S = 3.0  # cap on honoring ssodh/init's "wait" hint

# IBKR permits a single brokerage session per username, so brokerage-session
# init (ssodh/init) must never run concurrently — two callers racing it produce
# the competing/not-authenticated churn. Serialize it process-wide; the second
# caller re-checks status inside the lock and returns without a redundant init.
_BROKERAGE_INIT_LOCK = threading.Lock()


def _brief_detail(value: Any) -> str | None:
    """Normalize an IBKR status ``message``/``prompts`` field (str|list|None) to a
    short string so the real ssodh/init reason can be logged/surfaced."""
    if not value:
        return None
    if isinstance(value, list | tuple):
        value = "; ".join(str(v) for v in value)
    text = str(value).strip()
    return text[:300] or None


def _brokerage_init_wait_seconds(detail: dict) -> float:
    """Seconds to wait before re-checking after an ssodh/init that returned a
    ``wait`` hint. IBKR reports it in milliseconds; bounded so a bad value can't
    stall the run. Falls back to the poll-interval floor when absent."""
    raw = detail.get("wait")
    if isinstance(raw, int | float) and raw > 0:
        seconds = raw / 1000.0 if raw >= 50 else float(raw)
        return max(
            _BROKERAGE_AUTH_POLL_INTERVAL_S, min(seconds, _BROKERAGE_INIT_WAIT_CAP_S)
        )
    return _BROKERAGE_AUTH_POLL_INTERVAL_S


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


def mask_account(account_id: str | None) -> str:
    """Mask an IBKR account ID for operator logs (e.g. 'U2***465').

    The full ID is identifying (not a credential); logs shared in issues or
    pasted for debugging shouldn't carry it verbatim.
    """
    if not account_id:
        return "?"
    if len(account_id) <= 5:
        return account_id[0] + "***"
    return f"{account_id[:2]}***{account_id[-3:]}"


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
        self._ibind_client: Any = None
        self._throttle = IBKRThrottle(
            rate_per_sec=self._settings.ibkr_rate_limit_per_sec,
        )

    def connect(
        self, brokerage_session: bool = False, *, maintain: bool = False
    ) -> None:
        """
        Establish connection to IBKR via IBind.

        Args:
            brokerage_session: If True, create a full brokerage session
                (needed for orders, only one per username).
                If False, read-only mode (portfolio data only).
            maintain: If True, start ibind's tickler so the OAuth live-session
                token is kept alive for the life of a pooled connection. Use this
                only for a managed, single-owner connection that is explicitly
                logged out at teardown (see IbkrSessionManager) — never for the
                per-call connect/close pattern, which would leak tickler threads.

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
            raise ImportError("ibind package not installed. Run: poetry install") from e

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
                "maintain_oauth": maintain,
                "shutdown_oauth": False,  # logout is driven explicitly via logout()
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
                account=mask_account(self._settings.ibkr_account_id),
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
        """Close the local connection WITHOUT logging out server-side.

        Use this for the short-lived per-call pattern where the OAuth session is
        allowed to expire on its own. For a pooled/managed connection, prefer
        logout() so the server-side session is terminated cleanly rather than
        orphaned until IBKR times it out.
        """
        if self._ibind_client is not None:
            try:
                self._ibind_client.close()
            except Exception:
                pass
            self._ibind_client = None
        logger.debug("ibkr_disconnected")

    def logout(self) -> None:
        """Terminate the OAuth/brokerage session server-side, then drop the client.

        Calls ibind's oauth_shutdown() (stops the tickler and POSTs /logout) so a
        pooled session is not left lingering server-side. Idempotent and
        best-effort — failures are logged, never raised, so teardown can't crash.
        """
        if self._ibind_client is not None:
            try:
                self._ibind_client.oauth_shutdown()  # stop_tickler() + logout()
            except Exception as e:
                logger.warning(
                    "ibkr_logout_failed",
                    **summarize_exception(e, operation="ibkr_logout"),
                )
            finally:
                self._ibind_client = None
        logger.debug("ibkr_logged_out")

    def __enter__(self) -> IbkrClient:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _ensure_connected(self) -> None:
        if self._ibind_client is None:
            raise IBKRAuthError("Not connected. Call connect() first.")

    # ── Portfolio Data (read-only) ──

    def get_accounts(self) -> list[str]:
        """Get list of account IDs."""
        self._ensure_connected()
        try:
            result = self._throttle.call(
                lambda: self._ibind_client.portfolio_accounts()
            )
            data = result.data if hasattr(result, "data") else result
            if isinstance(data, list):
                return [
                    account_id
                    for a in data
                    if isinstance(a, dict)
                    for account_id in (a.get("id") or a.get("accountId"),)
                    if isinstance(account_id, str) and account_id
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
        acct = account_id or self._settings.ibkr_account_id
        try:
            result = self._throttle.call(
                lambda: self._ibind_client.positions(account_id=acct)
            )
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
        acct = account_id or self._settings.ibkr_account_id
        try:
            result = self._throttle.call(
                lambda: self._ibind_client.get_ledger(account_id=acct)
            )
            data = result.data if hasattr(result, "data") else result
            return data if isinstance(data, dict) else {}
        except Exception as e:
            raise IBKRAPIError(f"Failed to fetch ledger: {e}") from e

    def stock_conid_by_symbol(
        self, symbol: str, default_filtering: bool = False
    ) -> dict:
        """
        Resolve stock conid from symbol.

        Returns dict of {symbol: [{conid, exchange, ...}]}.

        Note: default_filtering=False is the correct default for this system — ibind's
        built-in default applies {isUS: True} which silently drops all non-US contracts.
        """
        self._ensure_connected()
        try:
            result = self._throttle.call(
                lambda: self._ibind_client.stock_conid_by_symbol(
                    symbol, default_filtering=default_filtering
                )
            )
            data = result.data if hasattr(result, "data") else result
            return data if isinstance(data, dict) else {}
        except Exception as e:
            raise IBKRAPIError(f"Failed to resolve conid for {symbol}: {e}") from e

    def initialize_brokerage_session(self, compete: bool = True) -> bool:
        """
        Initialize the IBKR brokerage session (POST /iserver/auth/ssodh/init).

        Required before calling any /iserver/ endpoint (watchlists, market data,
        orders, contract info).  The portfolio /portfolio/ endpoints work without
        this; the brokerage session is an additional step on top of the OAuth
        live session token.

        Args:
            compete: When True (default), disconnects competing brokerage sessions.
                     When False, fails if another brokerage session is already active.

        Returns:
            True on success, False on failure (logs a warning on failure).
        """
        self._ensure_connected()
        self._last_brokerage_init_detail: dict = {}
        try:
            result = self._throttle.call(
                lambda: self._ibind_client.initialize_brokerage_session(compete=compete)
            )
            data = result.data if hasattr(result, "data") else result
            detail = data if isinstance(data, dict) else {}
            self._last_brokerage_init_detail = detail
            # ssodh/init returns authenticated/competing plus message/fail/prompts that
            # explain a non-authenticated result — surface them instead of discarding.
            logger.info(
                "brokerage_session_init_response",
                compete=compete,
                authenticated=detail.get("authenticated"),
                competing=detail.get("competing"),
                connected=detail.get("connected"),
                fail=detail.get("fail") or None,
                message=_brief_detail(detail.get("message")),
                prompts=_brief_detail(detail.get("prompts")),
            )
            return True
        except Exception as e:
            logger.warning(
                "brokerage_session_init_failed",
                **summarize_exception(e, operation="brokerage_session_init_failed"),
                compete=compete,
            )
            return False

    def _brokerage_auth_status(self) -> dict | None:
        """Return the parsed ``/iserver/auth/status`` dict, or None if it errored.

        ``/iserver/auth/status`` is IBKR's source of truth for whether the brokerage
        session is authenticated (``connected``/``authenticated``/``competing``).
        """
        try:
            result = self._throttle.call(
                lambda: self._ibind_client.authentication_status(log=False)
            )
        except Exception as e:
            logger.warning(
                "brokerage_session_status_check_failed",
                **summarize_exception(e, operation="authentication_status"),
            )
            return None
        data = result.data if hasattr(result, "data") else result
        return data if isinstance(data, dict) else None

    def _ensure_brokerage_session(self, *, operation: str) -> None:
        """Verify (and only if needed re-initialize) the /iserver brokerage session.

        Status-first: check ``/iserver/auth/status`` and only (re)init when the
        session is not authenticated — a status-authenticated session is never
        re-inited, so a healthy session is never bumped. When not authenticated,
        ssodh/init (the documented re-auth) is retried a bounded number of times
        because it can answer ``{"wait": N}`` while the session establishes
        asynchronously; ``authenticated=false`` does not raise, so the init response
        and ``/auth/status`` are both consulted before giving up.

        Raises:
            IBKRAuthError: with an actionable message naming ``operation`` when the
                brokerage session cannot be authenticated (e.g. the gateway needs
                re-login, or a competing session bumped this one).
        """
        self._ensure_connected()
        # Serialize so concurrent callers (e.g. pooled-session readers) can't race
        # ssodh/init; the loser re-checks status below and returns without re-init.
        with _BROKERAGE_INIT_LOCK:
            self._ensure_brokerage_session_locked(operation=operation)

    def _ensure_brokerage_session_locked(self, *, operation: str) -> None:
        status = self._brokerage_auth_status()
        if status is not None and status.get("authenticated") is True:
            return
        # Not authenticated (timed out / competing / unknown). Re-init: ssodh/init
        # is the documented re-auth, and it can answer {"wait": N} while the session
        # establishes asynchronously — the completed session only appears on a
        # subsequent init/status. So re-init a few times (honoring the wait hint)
        # rather than treating the first not-authenticated/"wait" as terminal. We
        # only reach here when no session is authenticated, so re-init can't bump a
        # healthy one.
        logger.info(
            "brokerage_session_reauth_attempt",
            operation=operation,
            connected=bool(status and status.get("connected")),
            competing=bool(status and status.get("competing")),
        )
        detail: dict = {}
        status = None
        for _attempt in range(_BROKERAGE_INIT_ATTEMPTS):
            self.initialize_brokerage_session()  # POST iserver/auth/ssodh/init
            detail = getattr(self, "_last_brokerage_init_detail", None) or {}
            if detail.get("authenticated") is True:
                return
            time.sleep(_brokerage_init_wait_seconds(detail))
            status = self._brokerage_auth_status()
            if status is not None and status.get("authenticated") is True:
                return
        connected = bool(status and status.get("connected"))
        competing = bool(status and status.get("competing"))
        init_detail = getattr(self, "_last_brokerage_init_detail", None) or {}
        ibkr_fail = init_detail.get("fail") or (status or {}).get("fail") or None
        ibkr_message = _brief_detail(init_detail.get("message")) or _brief_detail(
            (status or {}).get("message")
        )
        logger.warning(
            "brokerage_session_not_authenticated",
            operation=operation,
            connected=connected,
            competing=competing,
            ibkr_fail=ibkr_fail,
            ibkr_message=ibkr_message,
            ssodh_init_keys=sorted(init_detail.keys()) or None,
            auth_status_keys=sorted((status or {}).keys()) or None,
        )
        if competing:
            hint = (
                "another live session bumped yours — close the other session "
                "(IBKR Mobile / TWS / another API client), then re-run. To run the "
                "API alongside TWS/Mobile permanently, create a SECOND IBKR username "
                "dedicated to the API (IBKR allows only one brokerage session per "
                "username — its documented fix for concurrent use)"
            )
        else:
            # connected-but-unauthenticated with competing=False does NOT reliably
            # mean stale credentials. IBKR's competing flag is unreliable: the IBKR
            # Mobile app in particular silently reclaims the brokerage session while
            # /iserver/auth/status still reports competing=False, and ssodh/init's
            # compete=True does not always bump it. So lead with closing other
            # sessions (the common, verified cause); a stale OAuth key is the
            # fallback only after every other session is confirmed closed.
            hint = (
                "this is often transient — re-run first (the session usually "
                "authenticates on a fresh attempt). If it recurs, another live "
                "IBKR login is most likely still holding the brokerage session "
                "even though the API reports none competing — force-quit / log "
                "out of the IBKR Mobile app (it silently reclaims the session), "
                "then close TWS, Client Portal web, and any other API client. The "
                "durable fix for running the API while you also use TWS/Mobile is a "
                "SECOND IBKR username dedicated to the API (only one brokerage "
                "session is allowed per username). Only if it persists after every "
                "other session is closed is the OAuth key likely stale (regenerate "
                "it in IBKR Client Portal → Settings → API → OAuth and update .env)"
            )
        ibkr_detail = ""
        if ibkr_fail or ibkr_message:
            ibkr_detail = f" [IBKR reported: {ibkr_fail or ibkr_message}]"
        raise IBKRAuthError(
            f"IBKR brokerage session not authenticated for {operation} "
            f"(connected={connected}, competing={competing}){ibkr_detail}: {hint}."
        )

    def _call_iserver_accounts(self) -> bool:
        """Best-effort /iserver/accounts priming for market-data endpoints."""
        self._ensure_connected()
        for attr_name in (
            "receive_brokerage_accounts",
            "accounts",
            "iserver_accounts",
        ):
            method = getattr(self._ibind_client, attr_name, None)
            if callable(method):
                try:
                    self._throttle.call(method)
                    return True
                except Exception as exc:
                    logger.debug(
                        "iserver_accounts_prime_failed",
                        method=attr_name,
                        error=str(exc),
                    )
                    return False
        logger.debug("iserver_accounts_prime_unavailable")
        return False

    def _get_marketdata_snapshot_method(self):
        for attr_name in (
            "live_marketdata_snapshot",
            "marketdata_snapshot",
            "market_data_snapshot",
        ):
            method = getattr(self._ibind_client, attr_name, None)
            if callable(method):
                return method
        return None

    def get_marketdata_snapshot(
        self,
        conid: int,
        *,
        fields: str = "31,55,84,86,87,6004,6008,6509,7051",
        compete: bool = False,
    ) -> dict:
        """
        Fetch a single-contract market data snapshot.

        Uses the documented /iserver/accounts preflight plus the snapshot
        pre-flight pattern required by the IBKR Client Portal API.
        """
        self._ensure_connected()

        if not self.initialize_brokerage_session(compete=compete):
            logger.debug(
                "marketdata_snapshot_skipped_no_session",
                conid=conid,
                compete=compete,
            )
            return {}

        self._call_iserver_accounts()

        snapshot_method = self._get_marketdata_snapshot_method()
        if snapshot_method is None:
            logger.debug("marketdata_snapshot_method_unavailable", conid=conid)
            return {}

        field_ids = [field.strip() for field in fields.split(",") if field.strip()]

        def _request():
            return snapshot_method(conids=[str(conid)], fields=field_ids)

        try:
            result = self._throttle.call_with_warmup(
                preflight=_request,
                request=_request,
                warm_up_secs=0.5,
                label="marketdata_snapshot",
            )
            data = result.data if hasattr(result, "data") else result
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data[0]
            return {}
        except Exception as exc:
            logger.debug("marketdata_snapshot_failed", conid=conid, error=str(exc))
            return {}

    def get_watchlist(self, name_hint: str = "default watchlist") -> list[dict] | None:
        """
        Fetch watchlist rows from IBKR.

        Finds the watchlist whose name contains name_hint (case-insensitive),
        falling back to the first watchlist when hint is empty.

        Args:
            name_hint: Case-insensitive substring to match against watchlist name.
                       Empty string matches the first watchlist.

        Returns:
            List of raw watchlist row dicts (may be empty if watchlist exists but
            has no rows).  Returns None when the named watchlist was not found.

        Raises:
            IBKRAuthError: brokerage session could not be authenticated.
            IBKRAPIError: watchlist fetch failed (API/transport error) — callers
                decide whether to fail closed (explicit request) or soft-fail.
        """
        self._ensure_connected()
        # /iserver/ endpoints require an *authenticated* brokerage session on top of
        # the OAuth live session token.  Verify it (re-auth once) so a connected-but-
        # unauthenticated session surfaces as an actionable error rather than an
        # opaque downstream auth_error.
        self._ensure_brokerage_session(operation="watchlist_fetch")
        try:
            result = self._throttle.call(
                lambda: self._ibind_client.get_all_watchlists(sc="USER_WATCHLIST")
            )
            data = result.data if hasattr(result, "data") else result

            # Log raw shape for diagnostics
            if isinstance(data, dict):
                logger.debug("watchlists_raw_dict", keys=sorted(data.keys()))
            elif isinstance(data, list):
                logger.debug("watchlists_raw_list", length=len(data))
            else:
                logger.debug("watchlists_raw_other", data_type=type(data).__name__)

            def _extract_watchlist_list(payload, depth: int = 0) -> list:
                """Recursively find the list of watchlist entries in the response.

                IBKR response shapes observed:
                  [...]                                          — bare list
                  {"data": [...]}                                — one level of wrapping
                  {"MID":…, "action":…, "data": {"user_lists": [...]}}
                                                                 — two levels of wrapping
                """
                if isinstance(payload, list):
                    return payload
                if not isinstance(payload, dict) or depth > 3:
                    return []
                for key in ("data", "user_lists", "lists", "watchlists"):
                    val = payload.get(key)
                    if isinstance(val, list):
                        return val
                    if isinstance(val, dict):
                        nested = _extract_watchlist_list(val, depth + 1)
                        if nested:
                            return nested
                return []

            watchlists = _extract_watchlist_list(data)
            if not watchlists and isinstance(data, dict) and data:
                logger.warning("watchlists_unexpected_shape", keys=sorted(data.keys()))

            logger.info("watchlists_fetched", count=len(watchlists))
            if not watchlists:
                logger.debug("watchlist_none_found")
                return None if name_hint else []

            # Find by name_hint (case-insensitive substring).
            # Empty hint → fall back to the first watchlist.
            # Non-empty hint with no match → return None ("not found").
            matched = None
            hint_lower = name_hint.lower()
            for wl in watchlists:
                if hint_lower and hint_lower in (wl.get("name", "") or "").lower():
                    matched = wl
                    break
            if matched is None:
                if hint_lower:
                    available = [wl.get("name", wl.get("id", "?")) for wl in watchlists]
                    logger.warning(
                        "watchlist_not_found",
                        hint=name_hint,
                        available=available,
                    )
                    return None
                matched = watchlists[0]

            wl_id = matched.get("id", "")
            wl_name = matched.get("name", wl_id)
            info_result = self._throttle.call(
                lambda: self._ibind_client.get_watchlist_information(wl_id)
            )
            info_data = (
                info_result.data if hasattr(info_result, "data") else info_result
            )
            # Unwrap MID/action envelope: {"MID":…, "action":…, "data": {watchlist dict}}
            if isinstance(info_data, dict):
                inner = info_data.get("data")
                if isinstance(inner, dict):
                    info_data = inner
                elif isinstance(inner, list):
                    info_data = {"rows": inner}
            info = info_data if isinstance(info_data, dict) else {}
            logger.debug(
                "watchlist_info_keys",
                keys=sorted(info.keys()),
                rows_type=type(info.get("rows")).__name__,
            )
            rows = info.get("rows") or info.get("instruments") or []
            if not isinstance(rows, list):
                rows = []
            logger.info("watchlist_loaded", name=wl_name, count=len(rows))
            return rows
        except IBKRError:
            # Brokerage-session / typed IBKR failures (incl. the preflight's
            # IBKRAuthError) propagate unchanged so callers can distinguish a fetch
            # failure from a genuinely empty/not-found watchlist.
            raise
        except Exception as e:
            logger.warning(
                "watchlist_fetch_failed",
                **summarize_exception(e, operation="watchlist_fetch_failed"),
            )
            # API/transport error — raise (was: return []) so an explicitly requested
            # watchlist fails closed instead of silently degrading to "empty".
            raise IBKRAPIError("IBKR watchlist fetch failed") from e

    def get_live_orders(self, account_id: str | None = None) -> list[dict]:
        """
        Fetch open/pending orders from IBKR.

        Returns list of raw order dicts (may be empty when there are genuinely no
        open orders). Requires an authenticated brokerage session.

        IBKR's /iserver/account/orders endpoint requires a "pre-flight" call:
        the first request always returns an empty list while the server wakes
        up the orders engine.  A second request made shortly after returns the
        actual orders.  This method makes both calls automatically.

        Raises:
            IBKRAuthError: brokerage session could not be authenticated.
            IBKRAPIError: the orders fetch itself failed (API/transport error). The
                snapshot service catches this and records it as a non-fatal
                ``errors["live_orders"]`` so the report flags degraded order-dedup
                rather than silently treating it as "no open orders".
        """
        self._ensure_connected()
        # Verify the brokerage session (re-auth once); raises IBKRAuthError if it
        # can't be authenticated. Callers (snapshot service) catch and record this as
        # a non-fatal error so the report can flag that order-dedup is degraded.
        self._ensure_brokerage_session(operation="live_orders")
        acct = account_id or self._settings.ibkr_account_id

        def _extract(result: Any) -> list[dict[str, Any]]:
            data = result.data if hasattr(result, "data") else result
            if isinstance(data, dict):
                orders = data.get("orders", [])
                return orders if isinstance(orders, list) else []
            return data if isinstance(data, list) else []

        try:
            # Pre-flight wakes IBKR's orders engine; real call returns actual orders.
            # call_with_warmup encodes the engine-init pattern: pre-flight → sleep → call.
            raw = self._throttle.call_with_warmup(
                preflight=lambda: self._ibind_client.live_orders(
                    account_id=acct, force=True
                ),
                request=lambda: self._ibind_client.live_orders(account_id=acct),
                warm_up_secs=1.0,
                label="live_orders",
            )
            orders = _extract(raw)
            logger.info("live_orders_fetched", count=len(orders))
            return orders
        except IBKRError:
            raise
        except Exception as e:
            logger.warning(
                "live_orders_fetch_failed",
                **summarize_exception(e, operation="live_orders_fetch_failed"),
            )
            # Raise (was: return []) so a fetch failure is distinguishable from
            # "no open orders" and surfaces as a non-fatal degraded-dedup banner.
            raise IBKRAPIError("IBKR live orders fetch failed") from e

    def get_contract_info(self, conid: int, *, compete: bool = True) -> dict:
        """
        Get contract details (symbol, exchange, currency) for a given conid.

        Used to resolve watchlist conids to yfinance tickers.

        Args:
            conid: IBKR contract identifier.
            compete: Whether this call may displace another brokerage session.

        Returns raw contract info dict, or {} on failure.
        """
        self._ensure_connected()
        self.initialize_brokerage_session(compete=compete)
        try:
            result = self._throttle.call(
                lambda: self._ibind_client.contract_information_by_conid(str(conid))
            )
            data = result.data if hasattr(result, "data") else result
            return data if isinstance(data, dict) else {}
        except Exception as e:
            summary = summarize_exception(e, operation="contract_info_failed")
            summary.pop("message_preview", None)
            logger.debug("contract_info_failed", conid=conid, **summary)
            return {}

    def get_security_definition(self, conid: int) -> dict:
        """
        Get non-session security definition details for a conid.

        This endpoint does not require a brokerage-session handoff and can still
        expose listingExchange/allExchanges for held positions when
        /iserver/contract/{conid}/info is unavailable under compete=False.
        """
        self._ensure_connected()
        try:
            result = self._throttle.call(
                lambda: self._ibind_client.security_definition_by_conid([str(conid)])
            )
            data = result.data if hasattr(result, "data") else result
            if isinstance(data, dict):
                secdefs = data.get("secdef")
                if isinstance(secdefs, list):
                    for item in secdefs:
                        if isinstance(item, dict):
                            return item
            return {}
        except Exception as e:
            summary = summarize_exception(e, operation="security_definition_failed")
            summary.pop("message_preview", None)
            logger.debug(
                "security_definition_failed",
                conid=conid,
                **summary,
            )
            return {}

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
            result = self._throttle.call(
                lambda: self._ibind_client.place_order(
                    order_request=order_request,
                    answers=answers,
                    account_id=account_id,
                )
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
