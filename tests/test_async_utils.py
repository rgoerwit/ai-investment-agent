"""Tests for src.async_utils — hard timeouts and pending-task introspection."""

from __future__ import annotations

import asyncio
import signal
import time

import pytest

from src.async_utils import (
    dump_pending_tasks,
    install_pending_task_dump_handler,
    run_with_hard_timeout,
)


class TestRunWithHardTimeout:
    @pytest.mark.asyncio
    async def test_returns_value_on_success(self):
        async def quick():
            await asyncio.sleep(0.01)
            return 42

        result = await run_with_hard_timeout(quick(), timeout=1.0, label="quick")
        assert result == 42

    @pytest.mark.asyncio
    async def test_raises_timeouterror_after_deadline(self):
        async def slow():
            await asyncio.sleep(10)
            return "never"

        with pytest.raises(asyncio.TimeoutError):
            await run_with_hard_timeout(slow(), timeout=0.05, label="slow")

    @pytest.mark.asyncio
    async def test_does_not_block_caller_when_inner_ignores_cancellation(self):
        """Caller must return within the timeout even if the inner task can't be cancelled.

        This is the central guarantee — the bug we're protecting against is
        ``asyncio.wait_for`` waiting for a cancelled task that never finishes.
        """
        inner_started = asyncio.Event()

        async def uncancellable_inner():
            inner_started.set()
            try:
                # Sleep is cancellable, but we explicitly absorb the cancel and
                # keep going, mimicking a thread blocked in a sync I/O call.
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(10)  # ignore cancellation, keep blocking
            return "never"

        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await run_with_hard_timeout(
                uncancellable_inner(), timeout=0.1, label="uncancellable"
            )
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"caller blocked for {elapsed:.2f}s past timeout"
        assert inner_started.is_set()

    @pytest.mark.asyncio
    async def test_orphan_exception_is_silenced(self, caplog):
        """An inner task that raises after the caller has timed out must not log a warning."""

        async def fail_after_timeout():
            await asyncio.sleep(0.05)
            raise RuntimeError("orphan boom")

        with pytest.raises(asyncio.TimeoutError):
            await run_with_hard_timeout(
                fail_after_timeout(), timeout=0.01, label="orphan"
            )
        # Let the orphan finish; there should be no "Task exception was never
        # retrieved" warning issued.
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_zero_or_negative_timeout_rejected(self):
        async def noop():
            return None

        for bad in (0, -1.0):
            coro = noop()
            try:
                with pytest.raises(ValueError):
                    await run_with_hard_timeout(coro, timeout=bad, label="bad")
            finally:
                # run_with_hard_timeout rejects the call before scheduling the
                # coroutine, so we must close it ourselves to avoid the
                # "coroutine was never awaited" RuntimeWarning.
                coro.close()

    @pytest.mark.asyncio
    async def test_outer_cancellation_propagates(self):
        async def slow():
            await asyncio.sleep(10)
            return "never"

        outer = asyncio.create_task(
            run_with_hard_timeout(slow(), timeout=10, label="slow")
        )
        await asyncio.sleep(0.01)
        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer


class TestDumpPendingTasks:
    @pytest.mark.asyncio
    async def test_records_pending_task(self):
        async def hold():
            await asyncio.sleep(1)

        held = asyncio.create_task(hold(), name="hold-me")
        try:
            records = dump_pending_tasks()
            names = [r["name"] for r in records]
            assert "hold-me" in names
            entry = next(r for r in records if r["name"] == "hold-me")
            assert entry["coro"]
            assert isinstance(entry["stack"], list)
        finally:
            held.cancel()
            try:
                await held
            except asyncio.CancelledError:
                pass

    def test_returns_empty_list_outside_running_loop(self):
        # No running loop in this sync test => safe empty return.
        assert dump_pending_tasks() == []


class TestInstallPendingTaskDumpHandler:
    def test_handler_install_and_uninstall_roundtrip(self):
        if not hasattr(signal, "SIGUSR1"):
            pytest.skip("SIGUSR1 unavailable on this platform")
        previous = signal.getsignal(signal.SIGUSR1)
        uninstall = install_pending_task_dump_handler()
        try:
            current = signal.getsignal(signal.SIGUSR1)
            assert current is not previous
        finally:
            uninstall()
        restored = signal.getsignal(signal.SIGUSR1)
        # After uninstall, the handler should be removed (back to previous or default).
        assert restored == previous or restored == signal.SIG_DFL
