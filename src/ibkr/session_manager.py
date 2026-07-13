"""Process-wide pooled IBKR connection.

IBKR permits a **single brokerage session per username**, so genuine concurrency
against multiple authenticated sessions is impossible — the prior connect-per-call
pattern just minted many competing, never-logged-out OAuth sessions that fought
each other (and the user's phone) for that one slot. The correct model is therefore
a single pooled session:

  - **one** client, connected lazily on first use and reused by every consumer
    (portfolio reads, security/conid probes, name resolution, data-vacuum rescue)
    for the life of the process;
  - the OAuth live-session token kept alive by ibind's tickler (``maintain=True``);
  - safe concurrent reads — ibind issues requests over a thread-safe
    ``requests.Session`` and signs each call from the read-only shared OAuth token,
    so multiple consumers may read on the one connection at once; only the lazy
    connect (guarded here) and brokerage-session init (guarded in the client via
    ``_BROKERAGE_INIT_LOCK``) are serialized, so neither is ever raced;
  - logged out **exactly once** at teardown (``atexit`` always; ``SIGINT``/``SIGTERM``
    when installed from the main thread) so sessions are terminated cleanly rather
    than orphaned until IBKR times them out.

Consumers should never construct/connect their own ``IbkrClient`` for shared reads;
they go through ``get_ibkr_session_manager().session()``.
"""

from __future__ import annotations

import atexit
import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import structlog

from src.ibkr.client import IbkrClient
from src.ibkr.exceptions import IBKRError

logger = structlog.get_logger(__name__)


class IbkrSessionManager:
    """Owns the single pooled IBKR connection for a process.

    Build inputs (``client_cls``/``config``/prompt) may be supplied at construction
    or via :meth:`configure` before the lazy first connect. Tests inject a fake
    ``client_cls`` and reset the global singleton between tests.
    """

    def __init__(
        self,
        *,
        client_cls: type[IbkrClient] = IbkrClient,
        config: Any = None,
        prompt_for_missing_secret_fn: Callable[[Any], None] | None = None,
        install_signal_handlers: bool = False,
    ) -> None:
        self._client_cls = client_cls
        self._config = config
        self._prompt_fn = prompt_for_missing_secret_fn
        self._lock = threading.RLock()
        self._client: IbkrClient | None = None
        self._closed = False
        self._atexit_registered = False
        self._signals_installed = False
        if install_signal_handlers:
            self.install_signal_handlers()

    def configure(
        self,
        *,
        client_cls: type[IbkrClient] | None = None,
        config: Any = None,
        prompt_for_missing_secret_fn: Callable[[Any], None] | None = None,
    ) -> None:
        """Set build inputs before the (lazy) first connect; ignored once connected."""
        with self._lock:
            if self._client is not None:
                return
            if client_cls is not None:
                self._client_cls = client_cls
            if config is not None:
                self._config = config
            if prompt_for_missing_secret_fn is not None:
                self._prompt_fn = prompt_for_missing_secret_fn

    def acquire(self, *, brokerage_session: bool = False) -> IbkrClient:
        """Return the shared connected client, connecting lazily on first use.

        The connection is reused by every consumer; reads may be issued
        concurrently. The client is NOT closed by callers — the pool owns its
        lifecycle and logs out once at :meth:`shutdown`.

        Args:
            brokerage_session: ensure the /iserver brokerage session before
                returning (otherwise it is ensured lazily by the read methods that
                need it, e.g. watchlist/orders).
        """
        with self._lock:
            if self._closed:
                raise IBKRError("IBKR session manager has been shut down")
            client = self._connect_if_needed()
        # Brokerage init has its own process-wide lock in the client; keep it out
        # of the manager lock so a slow ssodh/init can't block other readers.
        if brokerage_session:
            client._ensure_brokerage_session(operation="pooled_session")
        return client

    @contextmanager
    def session(self, *, brokerage_session: bool = False) -> Iterator[IbkrClient]:
        """`with` convenience around :meth:`acquire`; does NOT close (pool persists)."""
        yield self.acquire(brokerage_session=brokerage_session)

    def _connect_if_needed(self) -> IbkrClient:
        if self._client is None:
            if self._prompt_fn is not None and self._config is not None:
                self._prompt_fn(self._config)
            client = (
                self._client_cls(self._config)
                if self._config is not None
                else self._client_cls()
            )
            client.connect(maintain=True)
            self._client = client
            self._register_atexit()
            logger.info("ibkr_pool_connected")
        return self._client

    def shutdown(self, *_args: Any) -> None:
        """Log out the pooled session once and mark the manager closed. Idempotent."""
        with self._lock:
            client = self._client
            if client is not None:
                logger.info("ibkr_pool_shutdown")
                # Prefer logout() (terminates the server-side session); fall back to
                # close() for any client that only implements local teardown.
                teardown = getattr(client, "logout", None) or client.close
                try:
                    teardown()
                except Exception:
                    pass
                self._client = None
            self._closed = True

    def reopen(self) -> None:
        """Re-arm a shut-down manager (long-lived hosts that recycle, e.g. worker)."""
        with self._lock:
            self._closed = False

    def _register_atexit(self) -> None:
        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers that log the pool out before exit.

        Must run on the main thread (``signal.signal`` raises otherwise); off the
        main thread this is a no-op and ``atexit`` still covers normal exit.
        """
        with self._lock:
            if self._signals_installed:
                return
            installed_any = False
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    previous = signal.getsignal(sig)
                    signal.signal(sig, self._make_signal_handler(previous))
                    installed_any = True
                except (ValueError, OSError):
                    # Not main thread / unsupported platform — atexit still applies.
                    return
            self._signals_installed = installed_any

    def _make_signal_handler(self, previous: Any) -> Callable[[int, Any], None]:
        def _handler(signum: int, frame: Any) -> None:
            self.shutdown()
            if callable(previous):
                previous(signum, frame)  # e.g. default_int_handler → KeyboardInterrupt
            elif signum == signal.SIGINT:
                raise KeyboardInterrupt
            else:
                raise SystemExit(128 + signum)

        return _handler


_session_manager: IbkrSessionManager | None = None
_singleton_lock = threading.Lock()


def get_ibkr_session_manager() -> IbkrSessionManager:
    """Return the process-wide pooled IBKR session manager (lazily created)."""
    global _session_manager
    if _session_manager is None:
        with _singleton_lock:
            if _session_manager is None:
                _session_manager = IbkrSessionManager()
    return _session_manager


def set_ibkr_session_manager(manager: IbkrSessionManager | None) -> None:
    """Override the singleton — for tests / explicit process-isolated wiring."""
    global _session_manager
    _session_manager = manager


def reset_ibkr_session_manager() -> None:
    """Shut down and clear the singleton — call between tests."""
    global _session_manager
    with _singleton_lock:
        if _session_manager is not None:
            try:
                _session_manager.shutdown()
            except Exception:
                pass
        _session_manager = None
