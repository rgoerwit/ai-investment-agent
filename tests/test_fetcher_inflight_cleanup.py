"""Regression tests for the inflight-dedup leak in SmartMarketDataFetcher.

The fetcher dedups concurrent fetches for the same key by storing an
``asyncio.Task`` in ``_metrics_inflight`` / ``_history_inflight``. The
original implementation cleared the slot inside a try/finally around
``await task``; if the underlying task hung (e.g. a thread blocked on a
no-timeout socket read), the await never returned and the slot leaked,
poisoning every subsequent caller for the same ticker.

These tests verify the dedup slot is cleared via ``add_done_callback`` so
cleanup is independent of the awaiter's progress.
"""

from __future__ import annotations

import asyncio

import pytest

from src.data.fetcher import SmartMarketDataFetcher


@pytest.mark.asyncio
async def test_metrics_inflight_cleared_when_task_raises():
    fetcher = SmartMarketDataFetcher.__new__(SmartMarketDataFetcher)
    # Minimal init for the fields we touch.
    fetcher._metrics_inflight = {}
    fetcher._metrics_cache = {}
    fetcher._metrics_cache_expiry = {}

    async def boom(*_args, **_kwargs):
        raise RuntimeError("simulated provider failure")

    fetcher._get_financial_metrics_uncached = boom

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        await fetcher.get_financial_metrics("AAPL", timeout=1)

    # Allow the done-callback (scheduled on the loop) to run.
    await asyncio.sleep(0)
    assert (
        "AAPL" not in fetcher._metrics_inflight
    ), "inflight slot must be cleared even when the task raised"


@pytest.mark.asyncio
async def test_metrics_inflight_cleared_when_task_cancelled():
    fetcher = SmartMarketDataFetcher.__new__(SmartMarketDataFetcher)
    fetcher._metrics_inflight = {}
    fetcher._metrics_cache = {}
    fetcher._metrics_cache_expiry = {}

    started = asyncio.Event()

    async def hang(*_args, **_kwargs):
        started.set()
        await asyncio.sleep(60)
        return {}

    fetcher._get_financial_metrics_uncached = hang

    caller = asyncio.create_task(fetcher.get_financial_metrics("AAPL", timeout=1))
    await started.wait()
    assert "AAPL" in fetcher._metrics_inflight  # slot is registered

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    await asyncio.sleep(0)
    assert (
        "AAPL" not in fetcher._metrics_inflight
    ), "inflight slot must be cleared on cancellation"
