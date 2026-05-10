"""Regression tests for the hard wall-clock timeout wrap around LLM ainvoke.

Background
----------
A single ``runnable.ainvoke()`` against ``langchain_google_genai`` plumbs
``api_timeout`` and ``api_retry_attempts`` into the underlying
``HttpOptions(timeout=…, retry_options=HttpRetryOptions(attempts=…))``.  When
``api_retry_attempts`` was 10 and ``api_timeout`` was 300s, a stuck Gemini
upstream could legitimately consume up to 10 × 300s ≈ 50 minutes per call
before raising — multiplied by our outer ``max_attempts=3`` retry loop, that
produced the multi-hour ticker stalls observed on HK tickers in May 2026.

These tests pin two defenses against that pathology:

1. ``config.api_retry_attempts`` defaults to a small value so SDK-level retry
   storms can't park a worker for an hour.
2. ``invoke_with_rate_limit_handling`` wraps ``runnable.ainvoke`` in
   ``run_with_hard_timeout`` using ``config.llm_call_hard_timeout_seconds`` so a
   hung provider SDK is force-cancelled and the outer retry loop can either
   reattempt or surface the failure.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.agents import invoke_with_rate_limit_handling
from src.config import config as settings_config


class TestApiRetryAttemptsDefault:
    """The SDK retry budget must stay small to bound worst-case call duration.

    These tests check the *code default* on the Settings field, not the runtime
    resolved value (which may be overridden by the developer's local .env).
    """

    def _field_default(self, name: str):
        from src.config import Settings

        return Settings.model_fields[name].default

    def test_code_default_is_small_enough_to_bound_worst_case(self):
        # 10 retries × 300s timeout (api_timeout) = 50 min per single LLM call.
        # We cap retries low so the SDK can't hang on its own retries; outer
        # invoke_with_rate_limit_handling already retries 3× with backoff.
        default = self._field_default("api_retry_attempts")
        assert default <= 3, (
            f"api_retry_attempts default must stay small (≤3); got {default}. "
            "At higher values the Gemini SDK can park a single ainvoke() for "
            "tens of minutes via internal retries, defeating wall-clock "
            "timeouts. See May 2026 HK ticker slow-tail incident."
        )

    def test_hard_timeout_default_present_and_bounded(self):
        # The hard timeout is the load-bearing safety net; if it disappears or
        # is cranked unreasonably high the slow-tail returns.
        default = self._field_default("llm_call_hard_timeout_seconds")
        assert 60.0 <= default <= 1800.0, (
            f"llm_call_hard_timeout_seconds default {default} is out of the "
            "sane range [60s, 1800s]."
        )
        # Also verify the field exists on the runtime object (so the wrap can
        # actually look it up).
        assert hasattr(settings_config, "llm_call_hard_timeout_seconds")


class TestHardTimeoutWrap:
    """`invoke_with_rate_limit_handling` must enforce a wall-clock cap."""

    @pytest.mark.asyncio
    async def test_hung_ainvoke_is_force_cancelled_after_hard_timeout(self):
        """A coroutine that never resolves must not block longer than the cap."""

        async def never_resolves(*_args, **_kwargs):
            # Awaiting an unresolved Future never returns; this simulates a
            # hung HTTP read inside the provider SDK.
            await asyncio.get_event_loop().create_future()
            return AIMessage(content="never")

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=never_resolves)

        # Patch the cap to a tiny value so the test runs fast.
        with patch.object(settings_config, "llm_call_hard_timeout_seconds", 0.05):
            with pytest.raises(Exception) as exc_info:
                await invoke_with_rate_limit_handling(
                    runnable,
                    {"input": "x"},
                    max_attempts=1,  # don't burn time on retries
                    context="HardTimeoutTest",
                )

        # The hard-timeout path raises asyncio.TimeoutError; the outer wrapper
        # logs it as failure_kind='timeout' and re-raises.
        assert (
            isinstance(exc_info.value, asyncio.TimeoutError)
            or "timeout" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_fast_ainvoke_is_unaffected(self):
        """Calls that resolve quickly must succeed normally even with cap set."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

        with patch.object(settings_config, "llm_call_hard_timeout_seconds", 1.0):
            result = await invoke_with_rate_limit_handling(
                runnable,
                {"input": "x"},
                max_attempts=1,
                context="HardTimeoutTest",
            )

        assert getattr(result, "content", None) == "ok"

    @pytest.mark.asyncio
    async def test_wrap_is_present_in_invoke_path(self):
        """Guard against silent removal of run_with_hard_timeout from runtime.py.

        Without this test, a refactor that drops the wrap would still pass the
        other tests because AsyncMock returns instantly. Here we observe that
        the wrap is invoked exactly once per attempt with the configured cap.
        """
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

        seen_calls: list[dict] = []

        async def spy(coro, *, timeout, label):
            seen_calls.append({"timeout": timeout, "label": label})
            return await coro

        with patch.object(settings_config, "llm_call_hard_timeout_seconds", 12.5):
            with patch("src.agents.runtime.run_with_hard_timeout", new=spy):
                await invoke_with_rate_limit_handling(
                    runnable,
                    {"input": "x"},
                    max_attempts=1,
                    context="WrapPresenceTest",
                )

        assert len(seen_calls) == 1, (
            "run_with_hard_timeout must wrap every ainvoke; if this fails the "
            "wall-clock guarantee was silently removed."
        )
        assert seen_calls[0]["timeout"] == 12.5
        # Label should encode context/provider/model so hard_timeout_exceeded
        # warnings are attributable.
        assert seen_calls[0]["label"].startswith("llm:WrapPresenceTest:")

    @pytest.mark.asyncio
    async def test_hard_timeout_retries_via_outer_loop(self):
        """When attempt 1 hits the hard timeout, attempt 2 should run with backoff."""
        call_count = {"n": 0}

        async def first_hangs_then_succeeds(*_args, **_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                await asyncio.get_event_loop().create_future()
            return AIMessage(content="recovered")

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=first_hangs_then_succeeds)

        # Patch only the runtime module's asyncio.sleep so the outer backoff
        # is instant; the inner hang awaits an unresolved future, not sleep,
        # so it remains hung until the hard timeout fires.
        with patch.object(settings_config, "llm_call_hard_timeout_seconds", 0.05):
            with patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock):
                with patch("src.agents.runtime.random.uniform", return_value=2.0):
                    result = await invoke_with_rate_limit_handling(
                        runnable,
                        {"input": "x"},
                        max_attempts=2,
                        context="HardTimeoutTest",
                    )

        assert call_count["n"] == 2
        assert getattr(result, "content", None) == "recovered"
