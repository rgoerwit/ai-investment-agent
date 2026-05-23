from __future__ import annotations

import time

import pytest

from src.blocking_io import BlockingCallPolicy, run_blocking_call


@pytest.mark.asyncio
async def test_run_blocking_call_returns_value() -> None:
    result = await run_blocking_call(
        BlockingCallPolicy("test_blocking_success", 1.0),
        lambda: "ok",
    )

    assert result == "ok"


@pytest.mark.asyncio
async def test_run_blocking_call_times_out() -> None:
    with pytest.raises(TimeoutError):
        await run_blocking_call(
            BlockingCallPolicy("test_blocking_timeout", 0.01),
            lambda: time.sleep(0.1),
        )


@pytest.mark.asyncio
async def test_run_blocking_call_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError):
        await run_blocking_call(
            BlockingCallPolicy("test_blocking_bad_timeout", 0),
            lambda: "never scheduled",
        )
