"""Tests for IbkrClient methods (unit tests — no real IBKR connection)."""

from unittest.mock import MagicMock, call, patch

import pytest

from src.ibkr.client import IbkrClient
from src.ibkr.exceptions import IBKRAPIError, IBKRAuthError
from src.ibkr.throttle import IBKRThrottle
from src.ibkr_config import IbkrSettings


@pytest.fixture(autouse=True)
def _no_auth_poll_sleep():
    """The brokerage-session reauth poll sleeps between status checks; neutralize
    it so failure-path tests don't incur real wall-clock delays."""
    with patch("src.ibkr.client.time.sleep"):
        yield


def _make_client() -> IbkrClient:
    """Return an IbkrClient with mocked internals (bypasses __init__ / connect).

    The throttle is replaced with a passthrough mock so tests execute immediately
    with no rate-delay sleeps. Both call() and call_with_warmup() still invoke
    the supplied callables, so all ibind interactions remain exercisable.
    """
    settings = MagicMock(spec=IbkrSettings)
    settings.ibkr_account_id = "U1234567"
    settings.ibkr_rate_limit_per_sec = 5

    client = IbkrClient.__new__(IbkrClient)
    client._settings = settings
    client._ibind_client = MagicMock()
    # Default: brokerage session is authenticated so /iserver preflight passes.
    # Tests exercising the unauthenticated path override this return_value.
    client._ibind_client.authentication_status.return_value = _response(
        {"authenticated": True, "connected": True, "competing": False}
    )

    # Passthrough throttle: no rate delays, but still calls through correctly.
    mock_throttle = MagicMock(spec=IBKRThrottle)
    mock_throttle.call.side_effect = lambda fn: fn()

    def _passthrough_warmup(preflight, request, warm_up_secs=0.0, **kw):
        preflight()  # pre-flight is still invoked (tests assert force=True)
        return request()

    mock_throttle.call_with_warmup.side_effect = _passthrough_warmup
    client._throttle = mock_throttle
    return client


def _response(data) -> MagicMock:
    """Wrap data in a mock object that has a .data attribute (ibind response format)."""
    r = MagicMock()
    r.data = data
    return r


class TestGetLiveOrders:
    """Tests for IbkrClient.get_live_orders().

    The IBKR /iserver/account/orders endpoint requires two calls:
    1. Pre-flight (force=True) — wakes the orders engine; result discarded.
    2. Real call               — returns actual pending orders.
    A 1-second warm-up pause between the calls is delegated to IBKRThrottle.
    """

    _PATCH_ENSURE = "src.ibkr.client.IbkrClient._ensure_connected"
    _PATCH_SESSION = "src.ibkr.client.IbkrClient.initialize_brokerage_session"

    def _patches(self):
        """Context manager helpers: suppress connection side-effects."""
        return (
            patch(self._PATCH_ENSURE),
            patch(self._PATCH_SESSION),
        )

    # ------------------------------------------------------------------ #
    # Response format handling
    # ------------------------------------------------------------------ #

    def test_dict_response_orders_key_extracted(self):
        """ibind returns {'orders': [...], 'snapshot': True} — extract the list."""
        client = _make_client()
        client._ibind_client.live_orders.side_effect = [
            _response([]),  # pre-flight: empty
            _response({"orders": [{"orderId": 1, "symbol": "WDO"}], "snapshot": True}),
        ]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            result = client.get_live_orders()

        assert len(result) == 1
        assert result[0]["symbol"] == "WDO"

    def test_list_response_returned_directly(self):
        """ibind returns a plain list — pass through without modification."""
        client = _make_client()
        orders = [{"orderId": 2, "symbol": "AAPL"}, {"orderId": 3, "symbol": "7203"}]
        client._ibind_client.live_orders.side_effect = [
            _response([]),
            _response(orders),
        ]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            result = client.get_live_orders()

        assert result == orders

    def test_raw_dict_without_data_attr(self):
        """ibind returns a plain dict (no .data attr) — extract 'orders' key."""
        client = _make_client()
        client._ibind_client.live_orders.side_effect = [
            [],  # pre-flight: raw empty list (no .data)
            {"orders": [{"orderId": 7, "symbol": "TSM"}]},  # raw dict
        ]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            result = client.get_live_orders()

        assert len(result) == 1
        assert result[0]["symbol"] == "TSM"

    def test_empty_orders_returns_empty_list(self):
        """Empty orders list returned as [] (not None)."""
        client = _make_client()
        client._ibind_client.live_orders.side_effect = [
            _response([]),
            _response([]),
        ]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            result = client.get_live_orders()

        assert result == []


class TestGetMarketdataSnapshot:
    """Tests for IbkrClient.get_marketdata_snapshot()."""

    _PATCH_ENSURE = "src.ibkr.client.IbkrClient._ensure_connected"
    _PATCH_SESSION = "src.ibkr.client.IbkrClient.initialize_brokerage_session"

    def test_snapshot_uses_accounts_prime_and_two_call_warmup(self):
        client = _make_client()
        client._ibind_client.receive_brokerage_accounts.return_value = _response(
            [{"id": "U1234567"}]
        )
        client._ibind_client.live_marketdata_snapshot.side_effect = [
            _response([{}]),
            _response(
                [
                    {
                        "31": "123.45",
                        "55": "7203",
                        "7051": "Toyota Motor",
                    }
                ]
            ),
        ]

        with (
            patch(self._PATCH_ENSURE),
            patch(self._PATCH_SESSION, return_value=True) as mock_session,
        ):
            result = client.get_marketdata_snapshot(12345)

        mock_session.assert_called_once_with(compete=False)
        client._ibind_client.receive_brokerage_accounts.assert_called_once()
        assert client._ibind_client.live_marketdata_snapshot.call_count == 2
        assert result["31"] == "123.45"
        assert result["7051"] == "Toyota Motor"

    def test_snapshot_falls_back_to_alternate_method_name(self):
        client = _make_client()
        client._ibind_client.live_marketdata_snapshot = None
        client._ibind_client.marketdata_snapshot.side_effect = [
            _response([{}]),
            _response([{"31": "9.87"}]),
        ]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION, return_value=True):
            result = client.get_marketdata_snapshot(999)

        assert result["31"] == "9.87"

    def test_snapshot_returns_empty_when_session_init_fails(self):
        client = _make_client()

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION, return_value=False):
            result = client.get_marketdata_snapshot(12345)

        assert result == {}

    def test_snapshot_returns_empty_on_exception(self):
        client = _make_client()
        client._ibind_client.receive_brokerage_accounts.side_effect = RuntimeError(
            "boom"
        )

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION, return_value=True):
            result = client.get_marketdata_snapshot(12345)

        assert result == {}

    # ------------------------------------------------------------------ #
    # Two-call protocol (pre-flight + real)
    # ------------------------------------------------------------------ #

    def test_exactly_two_calls_to_ibind(self):
        """Exactly two ibind.live_orders calls: pre-flight then real."""
        client = _make_client()
        client._ibind_client.live_orders.side_effect = [_response([]), _response([])]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            client.get_live_orders()

        assert client._ibind_client.live_orders.call_count == 2

    def test_preflight_uses_force_true(self):
        """Pre-flight call passes force=True to wake the IBKR orders engine."""
        client = _make_client()
        client._ibind_client.live_orders.side_effect = [_response([]), _response([])]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            client.get_live_orders(account_id="U999")

        calls = client._ibind_client.live_orders.call_args_list
        assert calls[0] == call(account_id="U999", force=True)

    def test_real_call_does_not_use_force(self):
        """The real (second) call must NOT pass force=True."""
        client = _make_client()
        client._ibind_client.live_orders.side_effect = [_response([]), _response([])]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            client.get_live_orders(account_id="U999")

        real_call = client._ibind_client.live_orders.call_args_list[1]
        assert real_call == call(account_id="U999")
        assert "force" not in real_call.kwargs

    def test_warmup_1s_passed_to_throttle(self):
        """get_live_orders() delegates warm-up to IBKRThrottle with warm_up_secs=1.0."""
        client = _make_client()
        client._ibind_client.live_orders.side_effect = [_response([]), _response([])]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            client.get_live_orders()

        _, kwargs = client._throttle.call_with_warmup.call_args
        assert kwargs.get("warm_up_secs") == 1.0

    def test_preflight_result_discarded(self):
        """Pre-flight data is ignored; only the real call result is returned."""
        client = _make_client()
        client._ibind_client.live_orders.side_effect = [
            _response([{"orderId": 99, "symbol": "GHOST"}]),  # pre-flight "ghost" order
            _response([{"orderId": 42, "symbol": "REAL"}]),
        ]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            result = client.get_live_orders()

        assert len(result) == 1
        assert result[0]["symbol"] == "REAL"

    # ------------------------------------------------------------------ #
    # Account ID routing
    # ------------------------------------------------------------------ #

    def test_explicit_account_id_forwarded(self):
        """Explicit account_id is passed to both ibind calls."""
        client = _make_client()
        client._ibind_client.live_orders.side_effect = [_response([]), _response([])]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            client.get_live_orders(account_id="U9999999")

        for c in client._ibind_client.live_orders.call_args_list:
            assert c.kwargs.get("account_id") == "U9999999" or c.args[0] == "U9999999"

    def test_default_account_id_from_settings(self):
        """When account_id=None, falls back to settings.ibkr_account_id."""
        client = _make_client()
        client._ibind_client.live_orders.side_effect = [_response([]), _response([])]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            client.get_live_orders()  # no account_id

        calls = client._ibind_client.live_orders.call_args_list
        # Both calls should use the settings account ID
        assert all(
            c.kwargs.get("account_id") == "U1234567"
            or (c.args and c.args[0] == "U1234567")
            for c in calls
        )

    # ------------------------------------------------------------------ #
    # Error handling
    # ------------------------------------------------------------------ #

    def test_api_exception_raises_typed_error(self):
        """A fetch failure raises IBKRAPIError (was: silently returned []).

        Surfacing the failure lets the snapshot layer record a non-fatal
        ``errors["live_orders"]`` and flag degraded order-dedup, instead of
        masquerading as "no open orders".
        """
        client = _make_client()
        client._ibind_client.live_orders.side_effect = RuntimeError(
            "IBKR connection timeout"
        )

        with (
            patch(self._PATCH_ENSURE),
            patch(self._PATCH_SESSION),
            pytest.raises(IBKRAPIError),
        ):
            client.get_live_orders()

    def test_unauthenticated_session_raises(self):
        """Brokerage session not authenticated → IBKRAuthError before the orders call.

        The snapshot service catches this and records a non-fatal
        ``errors["live_orders"]`` so the report flags degraded order-dedup.
        """
        client = _make_client()
        client._ibind_client.authentication_status.return_value = _response(
            {"authenticated": False, "connected": True, "competing": False}
        )
        with patch(self._PATCH_ENSURE), pytest.raises(IBKRAuthError):
            client.get_live_orders()
        client._ibind_client.live_orders.assert_not_called()

    def test_unexpected_response_type_returns_empty_list(self):
        """Completely unexpected response type (not list/dict) is handled gracefully."""
        client = _make_client()
        weird = MagicMock()
        weird.data = 42  # integer — not list or dict
        client._ibind_client.live_orders.side_effect = [_response([]), weird]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            result = client.get_live_orders()

        assert result == []


class TestEnsureBrokerageSession:
    """Tests for the status-first /iserver brokerage-session preflight."""

    _PATCH_ENSURE = "src.ibkr.client.IbkrClient._ensure_connected"

    def test_authenticated_session_skips_init(self):
        """Already authenticated → no ssodh/init call (status-first, avoids churn)."""
        client = _make_client()  # fixture default: authenticated
        with patch(self._PATCH_ENSURE):
            client._ensure_brokerage_session(operation="watchlist_fetch")
        client._ibind_client.initialize_brokerage_session.assert_not_called()
        client._ibind_client.authentication_status.assert_called_once()

    def test_unauthenticated_then_reauth_succeeds(self):
        """connected-but-unauthenticated → ssodh/init once → re-check authenticated."""
        client = _make_client()
        client._ibind_client.authentication_status.side_effect = [
            _response({"authenticated": False, "connected": True, "competing": False}),
            _response({"authenticated": True, "connected": True, "competing": False}),
        ]
        with patch(self._PATCH_ENSURE):
            client._ensure_brokerage_session(operation="watchlist_fetch")
        client._ibind_client.initialize_brokerage_session.assert_called_once()
        assert client._ibind_client.authentication_status.call_count == 2

    def test_still_unauthenticated_raises(self):
        """Unauthenticated after re-auth + poll → actionable IBKRAuthError."""
        client = _make_client()
        client._ibind_client.authentication_status.return_value = _response(
            {"authenticated": False, "connected": True, "competing": True}
        )
        with patch(self._PATCH_ENSURE), pytest.raises(IBKRAuthError):
            client._ensure_brokerage_session(operation="watchlist_fetch")

    def test_delayed_auth_recovered_by_poll(self):
        """Transient: not-authenticated right after ssodh/init, then authenticates
        on a later poll → no error (auto-recovery, no manual re-run needed)."""
        client = _make_client()
        client._ibind_client.authentication_status.side_effect = [
            _response({"authenticated": False, "connected": True, "competing": False}),
            _response({"authenticated": False, "connected": True, "competing": False}),
            _response({"authenticated": True, "connected": True, "competing": False}),
        ]
        with patch(self._PATCH_ENSURE), patch("src.ibkr.client.time.sleep") as sleep:
            client._ensure_brokerage_session(operation="watchlist_fetch")
        client._ibind_client.initialize_brokerage_session.assert_called_once()
        # 1 pre-init check + 2 poll checks (2nd poll authenticates).
        assert client._ibind_client.authentication_status.call_count == 3
        sleep.assert_called_once()  # one wait between the two poll checks

    def test_status_check_error_treated_as_unauthenticated(self):
        """A raising authentication_status is treated as not-authenticated, not crash."""
        client = _make_client()
        client._ibind_client.authentication_status.side_effect = RuntimeError("boom")
        with patch(self._PATCH_ENSURE), pytest.raises(IBKRAuthError):
            client._ensure_brokerage_session(operation="live_orders")


class TestGetWatchlistFailClosed:
    """get_watchlist raises (rather than returning []) on API error."""

    _PATCH_ENSURE = "src.ibkr.client.IbkrClient._ensure_connected"

    def test_api_error_raises_typed_error(self):
        client = _make_client()  # authenticated
        client._ibind_client.get_all_watchlists.side_effect = RuntimeError("503")
        with patch(self._PATCH_ENSURE), pytest.raises(IBKRAPIError):
            client.get_watchlist("watchlist-2026")

    def test_unauthenticated_session_raises_before_fetch(self):
        client = _make_client()
        client._ibind_client.authentication_status.return_value = _response(
            {"authenticated": False, "connected": True, "competing": False}
        )
        with patch(self._PATCH_ENSURE), pytest.raises(IBKRAuthError):
            client.get_watchlist("watchlist-2026")
        client._ibind_client.get_all_watchlists.assert_not_called()


class TestMaskAccount:
    def test_masks_standard_account_id(self):
        from src.ibkr.client import mask_account

        assert mask_account("U1234567") == "U1***567"

    def test_short_and_empty_values(self):
        from src.ibkr.client import mask_account

        assert mask_account("U12") == "U***"
        assert mask_account("") == "?"
        assert mask_account(None) == "?"

    def test_full_id_never_in_mask(self):
        from src.ibkr.client import mask_account

        assert "1234567" not in mask_account("U1234567")
