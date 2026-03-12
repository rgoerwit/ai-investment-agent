"""Tests for IbkrClient methods (unit tests — no real IBKR connection)."""

from unittest.mock import MagicMock, call, patch

import pytest

from src.ibkr.client import IbkrClient
from src.ibkr.throttle import IBKRThrottle
from src.ibkr_config import IbkrSettings


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

    def test_api_exception_returns_empty_list(self):
        """Any exception from ibind is caught and returns [] (non-fatal)."""
        client = _make_client()
        client._ibind_client.live_orders.side_effect = RuntimeError(
            "IBKR connection timeout"
        )

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            result = client.get_live_orders()

        assert result == []

    def test_unexpected_response_type_returns_empty_list(self):
        """Completely unexpected response type (not list/dict) is handled gracefully."""
        client = _make_client()
        weird = MagicMock()
        weird.data = 42  # integer — not list or dict
        client._ibind_client.live_orders.side_effect = [_response([]), weird]

        with patch(self._PATCH_ENSURE), patch(self._PATCH_SESSION):
            result = client.get_live_orders()

        assert result == []
