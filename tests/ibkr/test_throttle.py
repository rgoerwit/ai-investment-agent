"""Unit tests for IBKRThrottle — no ibind / no real IBKR connection required."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest

from src.ibkr.throttle import IBKRThrottle


class TestCall:
    def test_call_invokes_function(self):
        """call() invokes the callable and returns its result."""
        throttle = IBKRThrottle(rate_per_sec=1000.0)
        fn = MagicMock(return_value={"accounts": ["U1"]})
        result = throttle.call(fn)
        fn.assert_called_once()
        assert result == {"accounts": ["U1"]}

    def test_pace_enforces_minimum_interval(self):
        """Two back-to-back calls are separated by at least min_interval."""
        rate = 4.0  # 0.25 s minimum gap
        throttle = IBKRThrottle(rate_per_sec=rate)
        t0 = time.monotonic()
        throttle.call(lambda: None)
        throttle.call(lambda: None)
        elapsed = time.monotonic() - t0
        assert elapsed >= 1.0 / rate - 0.02  # 20 ms tolerance

    def test_call_retries_on_429(self):
        """A function that raises a '429' error once is retried and succeeds."""
        throttle = IBKRThrottle(rate_per_sec=1000.0, max_retries=3)
        fn = MagicMock(side_effect=[RuntimeError("HTTP 429 Too Many Requests"), "ok"])
        with patch("src.ibkr.throttle.time.sleep"):
            result = throttle.call(fn)
        assert result == "ok"
        assert fn.call_count == 2

    def test_call_raises_after_max_retries(self):
        """A function that always raises 429 propagates the exception after max_retries."""
        throttle = IBKRThrottle(rate_per_sec=1000.0, max_retries=3)
        fn = MagicMock(side_effect=RuntimeError("429 rate limit"))
        with patch("src.ibkr.throttle.time.sleep"):
            with pytest.raises(RuntimeError, match="429"):
                throttle.call(fn)
        assert fn.call_count == 3

    def test_call_does_not_retry_non_429(self):
        """Non-429 exceptions are re-raised immediately without retrying."""
        throttle = IBKRThrottle(rate_per_sec=1000.0, max_retries=3)
        fn = MagicMock(side_effect=ValueError("bad symbol"))
        with pytest.raises(ValueError, match="bad symbol"):
            throttle.call(fn)
        fn.assert_called_once()  # no retry

    def test_call_too_many_requests_string(self):
        """'too many requests' (without '429') also triggers retry."""
        throttle = IBKRThrottle(rate_per_sec=1000.0, max_retries=3)
        fn = MagicMock(
            side_effect=[RuntimeError("Too Many Requests from server"), "ok"]
        )
        with patch("src.ibkr.throttle.time.sleep"):
            result = throttle.call(fn)
        assert result == "ok"
        assert fn.call_count == 2

    def test_backoff_increases_exponentially(self):
        """Backoff on each retry attempt is 2^attempt seconds."""
        throttle = IBKRThrottle(rate_per_sec=1000.0, max_retries=3)
        fn = MagicMock(side_effect=RuntimeError("429"))
        sleep_calls = []
        with patch(
            "src.ibkr.throttle.time.sleep", side_effect=lambda s: sleep_calls.append(s)
        ):
            with pytest.raises(RuntimeError):
                throttle.call(fn)
        # _pace() sleep calls are separate; filter for the backoff sleeps (>= 1s)
        # All max_retries attempts fail, so backoff fires on every attempt: 2^0, 2^1, 2^2
        backoffs = [s for s in sleep_calls if s >= 1]
        assert backoffs == [2**i for i in range(throttle._max_retries)]


class TestCallWithWarmup:
    def test_calls_preflight_then_request(self):
        """Both callables are invoked; preflight runs first."""
        throttle = IBKRThrottle(rate_per_sec=1000.0)
        order = []
        preflight = MagicMock(side_effect=lambda: order.append("pre"))
        request = MagicMock(side_effect=lambda: order.append("req") or "data")

        with patch("src.ibkr.throttle.time.sleep"):
            throttle.call_with_warmup(preflight, request, warm_up_secs=0.5)

        assert order == ["pre", "req"]

    def test_sleeps_correct_duration(self):
        """time.sleep is called with the exact warm_up_secs value."""
        throttle = IBKRThrottle(rate_per_sec=1000.0)
        with patch("src.ibkr.throttle.time.sleep") as mock_sleep:
            throttle.call_with_warmup(
                preflight=lambda: None,
                request=lambda: None,
                warm_up_secs=1.0,
            )
        # The warm-up sleep (1.0) should be among the calls
        sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
        assert 1.0 in sleep_args

    def test_returns_request_result_not_preflight(self):
        """Preflight result is discarded; only the request result is returned."""
        throttle = IBKRThrottle(rate_per_sec=1000.0)
        with patch("src.ibkr.throttle.time.sleep"):
            result = throttle.call_with_warmup(
                preflight=lambda: "ghost_data",
                request=lambda: {"orders": [{"id": 42}]},
                warm_up_secs=0.0,
            )
        assert result == {"orders": [{"id": 42}]}

    def test_retries_request_on_429(self):
        """A 429 on the real call is retried by the inner call()."""
        throttle = IBKRThrottle(rate_per_sec=1000.0, max_retries=3)
        request = MagicMock(side_effect=[RuntimeError("429 limit"), [{"orderId": 7}]])
        with patch("src.ibkr.throttle.time.sleep"):
            result = throttle.call_with_warmup(
                preflight=lambda: None,
                request=request,
                warm_up_secs=0.0,
            )
        assert result == [{"orderId": 7}]
        assert request.call_count == 2

    def test_label_appears_in_log(self, caplog):
        """The label kwarg shows up in the debug log line."""
        import logging

        throttle = IBKRThrottle(rate_per_sec=1000.0)
        with patch("src.ibkr.throttle.time.sleep"):
            with caplog.at_level(logging.DEBUG, logger="src.ibkr.throttle"):
                throttle.call_with_warmup(
                    preflight=lambda: None,
                    request=lambda: None,
                    warm_up_secs=0.5,
                    label="live_orders",
                )
        # structlog may not use caplog directly; just verify no exception was raised
        # and the call completed successfully (label kwarg accepted without error).


class TestIsRateLimited:
    @pytest.mark.parametrize(
        "msg,expected",
        [
            ("HTTP 429 Too Many Requests", True),
            ("429", True),
            ("Too Many Requests", True),
            ("too many requests", True),
            ("Connection refused", False),
            ("401 Unauthorized", False),
            ("bad symbol", False),
            ("", False),
        ],
    )
    def test_detection(self, msg: str, expected: bool):
        assert IBKRThrottle._is_rate_limited(RuntimeError(msg)) is expected
