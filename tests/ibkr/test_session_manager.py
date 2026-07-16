"""Tests for the pooled IBKR session manager."""

from __future__ import annotations

import signal
import threading
import time

import pytest

import src.ibkr.session_manager as sm
from src.ibkr.exceptions import IBKRAuthError, IBKRError
from src.ibkr.session_manager import (
    IbkrSessionManager,
    get_ibkr_session_manager,
    reset_ibkr_session_manager,
)


class _FakeClient:
    def __init__(self, config=None):
        self.config = config
        self.connect_calls: list[dict] = []
        self.logged_out = False
        self.closed = False
        self.brokerage_ops: list[str] = []

    def connect(self, brokerage_session: bool = False, *, maintain: bool = False):
        self.connect_calls.append(
            {"brokerage_session": brokerage_session, "maintain": maintain}
        )

    def logout(self):
        self.logged_out = True

    def close(self):
        self.closed = True

    def _ensure_brokerage_session(self, *, operation: str):
        self.brokerage_ops.append(operation)


class _CloseOnlyClient:
    """A client that only implements local close() (no logout)."""

    def __init__(self, config=None):
        self.closed = False

    def connect(self, brokerage_session: bool = False, *, maintain: bool = False):
        pass

    def close(self):
        self.closed = True


def test_acquire_pools_a_single_maintained_connection():
    mgr = IbkrSessionManager(client_cls=_FakeClient)
    c1 = mgr.acquire()
    c2 = mgr.acquire()
    assert c1 is c2  # reused, not reconnected
    assert len(c1.connect_calls) == 1
    assert c1.connect_calls[0]["maintain"] is True  # tickler keeps it alive


def test_configure_sets_client_cls_before_first_connect():
    mgr = IbkrSessionManager()
    mgr.configure(client_cls=_FakeClient, config={"k": "v"})
    client = mgr.acquire()
    assert isinstance(client, _FakeClient)
    assert client.config == {"k": "v"}


def test_configure_ignored_once_connected():
    mgr = IbkrSessionManager(client_cls=_FakeClient)
    first = mgr.acquire()
    mgr.configure(client_cls=_CloseOnlyClient)  # too late
    assert mgr.acquire() is first


def test_shutdown_logs_out_exactly_once_and_is_idempotent():
    mgr = IbkrSessionManager(client_cls=_FakeClient)
    client = mgr.acquire()
    mgr.shutdown()
    mgr.shutdown()  # no error, no double work
    assert client.logged_out is True


def test_acquire_after_shutdown_raises():
    mgr = IbkrSessionManager(client_cls=_FakeClient)
    mgr.acquire()
    mgr.shutdown()
    with pytest.raises(IBKRError):
        mgr.acquire()


def test_reopen_rearms_a_shut_down_manager():
    mgr = IbkrSessionManager(client_cls=_FakeClient)
    mgr.acquire()
    mgr.shutdown()
    mgr.reopen()
    assert mgr.acquire() is not None  # connects fresh


def test_brokerage_session_pre_ensured_when_requested():
    mgr = IbkrSessionManager(client_cls=_FakeClient)
    client = mgr.acquire(brokerage_session=True)
    assert client.brokerage_ops == ["pooled_session"]


def test_shutdown_falls_back_to_close_when_no_logout():
    mgr = IbkrSessionManager(client_cls=_CloseOnlyClient)
    client = mgr.acquire()
    mgr.shutdown()
    assert client.closed is True


def test_atexit_registered_on_first_connect(monkeypatch):
    registered: list = []
    monkeypatch.setattr(sm.atexit, "register", registered.append)
    mgr = IbkrSessionManager(client_cls=_FakeClient)
    mgr.acquire()
    assert mgr.shutdown in registered


def test_session_context_manager_yields_pool_without_closing():
    mgr = IbkrSessionManager(client_cls=_FakeClient)
    with mgr.session() as client:
        pass
    # pool persists — not closed/logged-out by the context manager
    assert client.logged_out is False
    assert mgr.acquire() is client


def test_signal_handler_shuts_down_then_chains_to_previous():
    mgr = IbkrSessionManager(client_cls=_FakeClient)
    client = mgr.acquire()
    chained: list[int] = []
    handler = mgr._make_signal_handler(lambda signum, frame: chained.append(signum))
    handler(signal.SIGTERM, None)
    assert client.logged_out is True
    assert chained == [signal.SIGTERM]


def test_install_signal_handlers_on_main_thread():
    originals = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        mgr = IbkrSessionManager(client_cls=_FakeClient)
        mgr.install_signal_handlers()
        assert mgr._signals_installed is True
        assert signal.getsignal(signal.SIGTERM) is not originals[signal.SIGTERM]
    finally:
        for s, h in originals.items():
            signal.signal(s, h)


def test_singleton_is_shared_and_resettable():
    a = get_ibkr_session_manager()
    assert get_ibkr_session_manager() is a
    reset_ibkr_session_manager()
    assert get_ibkr_session_manager() is not a


# --- Failure modes -----------------------------------------------------------


class _FailFirstConnect:
    """connect() raises on the first attempt, succeeds thereafter."""

    connect_attempts = 0

    def __init__(self, config=None):
        self.config = config

    def connect(self, brokerage_session: bool = False, *, maintain: bool = False):
        type(self).connect_attempts += 1
        if type(self).connect_attempts == 1:
            raise IBKRAuthError("transient connect failure")

    def logout(self):
        pass


def test_connect_failure_is_not_latched_and_retries():
    _FailFirstConnect.connect_attempts = 0
    mgr = IbkrSessionManager(client_cls=_FailFirstConnect)
    with pytest.raises(IBKRAuthError):
        mgr.acquire()
    # No permanent disable: a fresh process/run would re-probe; so must the pool.
    client = mgr.acquire()
    assert client is not None
    assert _FailFirstConnect.connect_attempts == 2


def test_shutdown_swallows_logout_errors():
    class _LogoutRaises:
        def __init__(self, config=None):
            pass

        def connect(self, brokerage_session=False, *, maintain=False):
            pass

        def logout(self):
            raise RuntimeError("logout failed")

        def close(self):
            pass

    mgr = IbkrSessionManager(client_cls=_LogoutRaises)
    mgr.acquire()
    mgr.shutdown()  # best-effort teardown must not raise
    assert mgr._closed is True
    with pytest.raises(IBKRError):
        mgr.acquire()


def test_concurrent_first_acquire_connects_once():
    class _SlowConnect:
        connects = 0
        _guard = threading.Lock()

        def __init__(self, config=None):
            pass

        def connect(self, brokerage_session=False, *, maintain=False):
            with _SlowConnect._guard:
                _SlowConnect.connects += 1
            time.sleep(0.02)  # widen the race window

        def logout(self):
            pass

    mgr = IbkrSessionManager(client_cls=_SlowConnect)
    results: list = []

    def worker():
        results.append(mgr.acquire())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _SlowConnect.connects == 1  # one connect despite 8 racers
    assert all(r is results[0] for r in results)


def test_brokerage_ensure_failure_does_not_poison_read_only_pool():
    class _BrokerageFails:
        def __init__(self, config=None):
            self.brokerage_calls = 0

        def connect(self, brokerage_session=False, *, maintain=False):
            pass

        def logout(self):
            pass

        def _ensure_brokerage_session(self, *, operation):
            self.brokerage_calls += 1
            raise IBKRAuthError("brokerage not authenticated")

    mgr = IbkrSessionManager(client_cls=_BrokerageFails)
    with pytest.raises(IBKRAuthError):
        mgr.acquire(brokerage_session=True)
    # The OAuth connection stays usable for read-only work; not torn down/reconnected.
    client = mgr.acquire()
    assert client.brokerage_calls == 1


def test_signal_handler_safe_when_nothing_connected():
    mgr = IbkrSessionManager(client_cls=_FakeClient)
    chained: list[int] = []
    handler = mgr._make_signal_handler(lambda signum, frame: chained.append(signum))
    handler(signal.SIGTERM, None)  # never acquired → shutdown is a no-op
    assert chained == [signal.SIGTERM]


def test_install_signal_handlers_off_main_thread_degrades(monkeypatch):
    def _raise(*_a, **_k):
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(sm.signal, "signal", _raise)
    mgr = IbkrSessionManager(client_cls=_FakeClient)
    mgr.install_signal_handlers()  # must not raise
    assert mgr._signals_installed is False
