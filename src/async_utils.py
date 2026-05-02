"""Concurrency utilities — hard timeouts, pending-task introspection.

Why this exists
---------------
``asyncio.wait_for`` cancels the inner task on timeout, then awaits the
cancelled task before raising ``TimeoutError``. When the inner task is
``await asyncio.to_thread(blocking_call)`` and the blocking call is stuck in
a socket read with no library-level timeout, the thread cannot be cancelled,
so ``wait_for`` waits forever for the cancelled task to finish — effectively
no timeout. Hangs in this codebase trace back to this pattern: yfinance,
yahooquery, Tavily, and DDG all run sync HTTP under ``to_thread`` and may not
honor cancellation promptly.

``run_with_hard_timeout`` is a *deadline-only* alternative: when the timeout
expires it raises ``TimeoutError`` immediately, leaving the inner task to
finish (or never finish) in the background. The caller proceeds. The orphan
is silenced so it doesn't trigger an "exception was never retrieved" warning.

This is paired with ``install_pending_task_dump_handler``, which on SIGUSR1
dumps every pending asyncio task and its current stack — invaluable for
diagnosing future hangs (``kill -USR1 <pid>``).
"""

from __future__ import annotations

import asyncio
import signal
import sys
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def _silence_orphan(task: asyncio.Task[Any], *, label: str) -> None:
    """Drain *task*'s exception when it eventually finishes, so no warning fires."""

    def _drain(t: asyncio.Task[Any]) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.debug(
                "hard_timeout_orphan_failed",
                label=label,
                error_type=type(exc).__name__,
            )

    task.add_done_callback(_drain)


async def run_with_hard_timeout(
    coro: Awaitable[T],
    *,
    timeout: float,
    label: str,
) -> T:
    """Await *coro* up to *timeout* seconds; raise TimeoutError without waiting on cleanup.

    Unlike ``asyncio.wait_for``, this does not block on the inner task after the
    timeout fires. The inner task is cancelled best-effort and orphaned: if the
    underlying code can honor cancellation it will, if not the thread/task
    lives on but no longer blocks the caller. Any exception the orphan raises
    later is silenced.

    *label* is logged on timeout to make orphan tasks attributable.
    """
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout!r}")

    task: asyncio.Task[T] = asyncio.ensure_future(coro)
    try:
        done, _pending = await asyncio.wait({task}, timeout=timeout)
    except BaseException:
        # Outer cancellation, KeyboardInterrupt, etc. — propagate after asking
        # the inner to stop. We do not wait on it.
        if not task.done():
            task.cancel()
            _silence_orphan(task, label=label)
        raise

    if task in done:
        return task.result()

    # Timeout: orphan the task.
    task.cancel()
    _silence_orphan(task, label=label)
    logger.warning(
        "hard_timeout_exceeded",
        label=label,
        timeout_seconds=timeout,
    )
    raise asyncio.TimeoutError(f"{label!r} exceeded hard timeout of {timeout:.1f}s")


def dump_pending_tasks(
    *,
    loop: asyncio.AbstractEventLoop | None = None,
    frame_limit: int = 8,
) -> list[dict[str, Any]]:
    """Return one record per pending asyncio task with name, coro, and stack.

    Safe to call from any context that has a running loop. Returns an empty
    list if no loop is available.
    """
    try:
        running_loop = loop or asyncio.get_running_loop()
    except RuntimeError:
        return []

    records: list[dict[str, Any]] = []
    for task in asyncio.all_tasks(running_loop):
        if task.done():
            continue
        coro = task.get_coro()
        stack_lines: list[str] = []
        try:
            for frame in task.get_stack(limit=frame_limit):
                stack_lines.append(
                    f"{frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}"
                )
        except Exception:  # pragma: no cover - defensive
            stack_lines = ["<stack unavailable>"]
        records.append(
            {
                "name": task.get_name(),
                "coro": getattr(coro, "__qualname__", repr(coro)),
                "stack": stack_lines,
            }
        )
    return records


def install_pending_task_dump_handler(
    *,
    sig: int = signal.SIGUSR1,
) -> Callable[[], None]:
    """Install a signal handler that logs all pending asyncio tasks.

    Trigger with ``kill -USR1 <pid>``. Returns a cleanup callable that removes
    the handler. No-op (and returns a no-op cleanup) on platforms without
    SIGUSR1, e.g. Windows.
    """
    if not hasattr(signal, "SIGUSR1") or sys.platform == "win32":
        return lambda: None

    previous = signal.getsignal(sig)

    def _handler(_signum: int, _frame: Any) -> None:
        records = dump_pending_tasks()
        logger.warning(
            "pending_tasks_dump",
            count=len(records),
            tasks=records,
        )
        # Also write a human-readable copy to stderr so it shows up in tee'd logs
        # even if structlog rendering is configured for a non-stderr sink.
        sys.stderr.write(f"[pending_tasks_dump] {len(records)} pending task(s)\n")
        for rec in records:
            sys.stderr.write(f"  - {rec['name']} ({rec['coro']})\n")
            for line in rec["stack"]:
                sys.stderr.write(f"      {line}\n")
        sys.stderr.flush()

    try:
        signal.signal(sig, _handler)
    except (ValueError, OSError):  # pragma: no cover - non-main-thread path
        return lambda: None

    def _uninstall() -> None:
        try:
            signal.signal(sig, previous if previous is not None else signal.SIG_DFL)
        except (ValueError, OSError):  # pragma: no cover
            pass

    return _uninstall


__all__ = [
    "dump_pending_tasks",
    "install_pending_task_dump_handler",
    "run_with_hard_timeout",
]
