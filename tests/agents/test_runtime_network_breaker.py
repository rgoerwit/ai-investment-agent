"""Integration tests: `invoke_with_rate_limit_handling` and NetworkBreaker.

Goal: prove that DNS / connect failures across multiple contexts trip the
shared breaker, and that subsequent calls raise immediately without sleeping.
This is the load-bearing fix for the May 2026 macOS DNS outage class.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.agents import invoke_with_rate_limit_handling
from src.agents.network_breaker import (
    NetworkBreakerOpenError,
    get_network_breaker,
    reset_network_breaker_singleton,
)
from src.config import config as settings_config
from src.token_tracker import get_tracker


@pytest.fixture(autouse=True)
def _reset_state():
    reset_network_breaker_singleton()
    get_tracker().reset()
    yield
    reset_network_breaker_singleton()


def _raise_dns_failure(*_args, **_kwargs):
    """Raise an exception that `classify_failure` will categorize as
    failure_kind='dns_resolution'. The classifier looks for gaierror in
    the cause chain and characteristic error messages."""
    import socket

    cause = socket.gaierror(8, "nodename nor servname provided, or not known")
    err = ConnectionError(
        "Cannot connect to host generativelanguage.googleapis.com:443 "
        "[nodename nor servname provided, or not known]"
    )
    err.__cause__ = cause
    raise err


class TestNetworkBreakerWireIn:
    @pytest.mark.asyncio
    async def test_repeated_dns_failures_trip_breaker(self):
        """Four DNS failures across distinct contexts → 5th call fast-fails."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=_raise_dns_failure)

        # Tight breaker to make the test fast and deterministic.
        with (
            patch.object(settings_config, "network_breaker_threshold", 4),
            patch.object(settings_config, "network_breaker_window_seconds", 60.0),
            patch.object(settings_config, "network_breaker_cool_off_seconds", 30.0),
            patch.object(settings_config, "llm_call_hard_timeout_seconds", 0.5),
        ):
            reset_network_breaker_singleton()

            # First 4 calls: each fails with a DNS error. Use max_attempts=1
            # to avoid backoff sleeps and keep this under a second.
            for i in range(4):
                with pytest.raises(ConnectionError):
                    await invoke_with_rate_limit_handling(
                        runnable,
                        {"input": "x"},
                        max_attempts=1,
                        context=f"AnalystAgent{i}",
                    )

            # Breaker should now be open.
            assert get_network_breaker().snapshot()["state"] == "open"

            # The 5th call: short-circuits via NetworkBreakerOpenError,
            # WITHOUT calling runnable.ainvoke again.
            calls_before = runnable.ainvoke.call_count
            with pytest.raises(NetworkBreakerOpenError):
                await invoke_with_rate_limit_handling(
                    runnable,
                    {"input": "x"},
                    max_attempts=1,
                    context="AnalystAgent5",
                )
            assert (
                runnable.ainvoke.call_count == calls_before
            ), "Network breaker must short-circuit before ainvoke is called"

    @pytest.mark.asyncio
    async def test_successful_call_closes_breaker(self):
        """An ok=True outcome immediately closes the breaker."""
        from langchain_core.messages import AIMessage

        # Pre-open the breaker.
        breaker = get_network_breaker()
        for _ in range(10):
            breaker.record_outcome(ok=False, failure_kind="dns_resolution")
        assert breaker.snapshot()["state"] == "open"

        # Force it to half_open by advancing past cool_off via a fresh
        # snapshot — easier path is to just inject a success directly.
        breaker.record_outcome(ok=True)
        assert breaker.snapshot()["state"] == "closed"

        # And the call surface still works.
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        result = await invoke_with_rate_limit_handling(
            runnable, {"input": "x"}, max_attempts=1, context="GoodAgent"
        )
        assert result.content == "ok"

    @pytest.mark.asyncio
    async def test_timeout_failures_do_not_trip_network_breaker(self):
        """Hard-timeout failures must not trip the network breaker —
        they're handled by the existing LLMCircuitBreaker (which is keyed
        per agent/provider/model). The two breakers must remain
        orthogonal."""

        async def hang(*_args, **_kwargs):
            await asyncio.get_event_loop().create_future()

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=hang)

        with (
            patch.object(settings_config, "network_breaker_threshold", 2),
            patch.object(settings_config, "llm_call_hard_timeout_seconds", 0.05),
        ):
            reset_network_breaker_singleton()

            for i in range(3):
                with pytest.raises(asyncio.TimeoutError):
                    await invoke_with_rate_limit_handling(
                        runnable,
                        {"input": "x"},
                        max_attempts=1,
                        context=f"HangAgent{i}",
                    )

            snap = get_network_breaker().snapshot()
            assert snap["state"] == "closed"
            assert snap["failures_in_window"] == 0

    @pytest.mark.asyncio
    async def test_disabled_breaker_does_not_intercept(self):
        """When config disables the breaker, it must never raise."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=_raise_dns_failure)

        with (
            patch.object(settings_config, "network_breaker_enabled", False),
            patch.object(settings_config, "llm_call_hard_timeout_seconds", 0.5),
        ):
            # 10 failures, breaker disabled → still ineligible to short-circuit
            for i in range(10):
                with pytest.raises(ConnectionError) as exc_info:
                    await invoke_with_rate_limit_handling(
                        runnable,
                        {"input": "x"},
                        max_attempts=1,
                        context=f"NoBreakerAgent{i}",
                    )
                # Should be the underlying DNS error, never the breaker error
                assert not isinstance(exc_info.value, NetworkBreakerOpenError)
