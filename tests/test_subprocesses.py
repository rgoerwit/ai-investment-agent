"""Child processes: the posix_spawn configuration, and cleanup on every exit.

``src/subprocesses.py`` is where every child under ``src/`` starts. CPython uses
``posix_spawn`` only for an absolute executable with ``close_fds=False`` and no
``cwd``; otherwise it forks, and in the threaded analysis process ``fork()``
runs Network.framework's atfork handler, which can crash the child. A child is
killed and reaped on every abnormal exit -- its own timeout, an error, or the
caller's cancellation -- with the reap bounded, shielded, and its failure
logged, never leaked. Fakes with releasable stalls, plus a few harmless real
children (``/bin/sleep``, ``/bin/cat``, ``/usr/bin/false``).
"""

from __future__ import annotations

import asyncio
import gc
import os
import signal
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from src import subprocesses
from src.config import config

# A test-side guard so a broken bound fails instead of hanging the suite. It
# is never what a passing test waits for.
_GUARD_SECONDS = 5.0


class _LogRecorder:
    """Stands in for the module logger (capture_logs misses cached loggers)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str) -> Any:
        def record(event: str, **fields: Any) -> None:
            self.events.append((level, event, fields))

        return record


class _Child:
    """A fake child with releasable stalls in communicate() and wait()."""

    pid = 4242

    def __init__(
        self,
        *,
        stdout: bytes = b"out",
        stall: bool = False,
        stall_reap: bool = False,
        exits_on_its_own: bool = False,
        reap_error: BaseException | None = None,
    ) -> None:
        self.stdout = stdout
        self.stall = stall
        self.returncode: int | None = None
        self.in_communicate = asyncio.Event()
        self.reap_started = asyncio.Event()
        self.release_reap = asyncio.Event()
        if not stall_reap:
            self.release_reap.set()
        self.exits_on_its_own = exits_on_its_own
        self.reap_error = reap_error
        self.kill_calls = 0
        self.reaped = False

    async def communicate(self, payload: bytes) -> tuple[bytes, bytes]:
        del payload
        self.in_communicate.set()
        if self.stall:
            await asyncio.Event().wait()
        self.returncode = 0
        return self.stdout, b""

    def kill(self) -> None:
        self.kill_calls += 1
        if self.exits_on_its_own:
            # Exited between the returncode check and the kill.
            self.returncode = 0
            raise ProcessLookupError
        self.returncode = -9

    async def wait(self) -> int:
        self.reap_started.set()
        await self.release_reap.wait()
        if self.reap_error is not None:
            raise self.reap_error
        self.reaped = True
        assert self.returncode is not None
        return self.returncode


def _install_child(monkeypatch, child: _Child) -> list[dict[str, Any]]:
    """Hand ``child`` to run_program; return the spawn kwargs it was given."""
    spawned: list[dict[str, Any]] = []

    async def spawn(*args: Any, **kwargs: Any) -> _Child:
        spawned.append({"args": args, **kwargs})
        return child

    monkeypatch.setattr(subprocesses.asyncio, "create_subprocess_exec", spawn)
    return spawned


async def _run(timeout: float = _GUARD_SECONDS) -> subprocesses.ProgramOutput:
    return await subprocesses.run_program(
        "/usr/bin/pdftotext", "-", "-", input=b"%PDF", timeout=timeout
    )


@contextmanager
def _loop_errors() -> Iterator[list[dict[str, Any]]]:
    """Errors the running loop reports, e.g. "Task exception was never retrieved"."""
    errors: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    try:
        yield errors
    finally:
        loop.set_exception_handler(previous)


# --- The spawn configuration ---------------------------------------------------


@pytest.mark.asyncio
async def test_run_program_spawns_on_the_posix_spawn_path(monkeypatch):
    spawned = _install_child(monkeypatch, _Child())

    output = await _run()

    assert output == subprocesses.ProgramOutput(0, b"out")
    (spawn,) = spawned
    assert spawn["args"] == ("/usr/bin/pdftotext", "-", "-")
    assert spawn["close_fds"] is False
    assert "cwd" not in spawn
    assert "start_new_session" not in spawn
    assert "preexec_fn" not in spawn


@pytest.mark.parametrize("executable", ["pdftotext", "bin/pdftotext", ""])
def test_a_relative_executable_is_refused_before_spawning(monkeypatch, executable):
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not spawn")

    monkeypatch.setattr(subprocesses.subprocess, "run", forbidden)
    monkeypatch.setattr(subprocesses.asyncio, "create_subprocess_exec", forbidden)

    with pytest.raises(ValueError, match="absolute"):
        subprocesses.run_program_sync(executable, timeout=1.0)
    with pytest.raises(ValueError, match="absolute"):
        asyncio.run(subprocesses.run_program(executable, input=b"", timeout=1.0))


def test_resolve_program_returns_an_absolute_path_or_none(monkeypatch):
    monkeypatch.setattr(subprocesses.shutil, "which", lambda name: "./bin/" + name)
    assert subprocesses.resolve_program("git") == os.path.abspath("./bin/git")

    monkeypatch.setattr(subprocesses.shutil, "which", lambda name: None)
    assert subprocesses.resolve_program("git") is None


class _SpawnSpy:
    """Counts which branch CPython's Popen took for every child started."""

    def __init__(self, monkeypatch) -> None:
        self.posix_spawn = 0
        real = subprocess.Popen._posix_spawn  # type: ignore[attr-defined]

        def spy(popen: Any, *args: Any, **kwargs: Any) -> Any:
            self.posix_spawn += 1
            return real(popen, *args, **kwargs)

        monkeypatch.setattr(subprocess.Popen, "_posix_spawn", spy)
        self.forks = 0
        real_fork_exec = subprocess._fork_exec  # type: ignore[attr-defined]

        def fork_spy(*args: Any, **kwargs: Any) -> Any:
            self.forks += 1
            return real_fork_exec(*args, **kwargs)

        monkeypatch.setattr(subprocess, "_fork_exec", fork_spy)


@pytest.mark.skipif(not subprocess._USE_POSIX_SPAWN, reason="no posix_spawn here")  # type: ignore[attr-defined]
@pytest.mark.asyncio
async def test_a_real_async_child_is_started_by_posix_spawn(monkeypatch):
    spy = _SpawnSpy(monkeypatch)

    output = await subprocesses.run_program(
        "/bin/cat", input=b"echoed", timeout=_GUARD_SECONDS
    )

    assert output == subprocesses.ProgramOutput(0, b"echoed")
    assert (spy.posix_spawn, spy.forks) == (1, 0)


@pytest.mark.skipif(not subprocess._USE_POSIX_SPAWN, reason="no posix_spawn here")  # type: ignore[attr-defined]
def test_a_real_sync_child_is_started_by_posix_spawn(monkeypatch):
    spy = _SpawnSpy(monkeypatch)

    completed = subprocesses.run_program_sync("/bin/echo", "hi", timeout=_GUARD_SECONDS)

    assert completed.stdout == "hi\n"
    assert (spy.posix_spawn, spy.forks) == (1, 0)


def test_the_sync_runner_raises_on_a_non_zero_exit():
    with pytest.raises(subprocess.CalledProcessError):
        subprocesses.run_program_sync("/usr/bin/false", timeout=_GUARD_SECONDS)


def test_the_sync_runner_kills_a_child_that_overruns():
    with pytest.raises(subprocess.TimeoutExpired):
        subprocesses.run_program_sync("/bin/sleep", "30", timeout=0.05)


# --- Cleanup on every abnormal exit ---------------------------------------------


@pytest.mark.asyncio
async def test_normal_completion_neither_kills_nor_leaves_a_reaper(monkeypatch):
    child = _Child()
    _install_child(monkeypatch, child)

    await _run()

    assert child.kill_calls == 0
    assert subprocesses._PENDING_REAPS == set()


@pytest.mark.asyncio
async def test_a_non_zero_exit_is_returned_not_raised(monkeypatch):
    child = _Child()
    child.communicate = _exits_with(child, 1)  # type: ignore[method-assign]
    _install_child(monkeypatch, child)

    output = await _run()

    assert output.returncode == 1
    assert child.kill_calls == 0


def _exits_with(child: _Child, code: int) -> Any:
    async def communicate(payload: bytes) -> tuple[bytes, bytes]:
        del payload
        child.returncode = code
        return b"", b"error"

    return communicate


@pytest.mark.asyncio
async def test_cancel_during_communicate_kills_and_reaps(monkeypatch):
    child = _Child(stall=True)
    _install_child(monkeypatch, child)

    task = asyncio.ensure_future(_run())
    await child.in_communicate.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert child.kill_calls == 1
    assert child.reaped
    assert subprocesses._PENDING_REAPS == set()


@pytest.mark.asyncio
async def test_cancel_again_during_cleanup_still_reaps(monkeypatch):
    child = _Child(stall=True, stall_reap=True)
    _install_child(monkeypatch, child)

    with _loop_errors() as loop_errors:
        task = asyncio.ensure_future(_run())
        await child.in_communicate.wait()
        task.cancel()
        await child.reap_started.wait()
        task.cancel()  # Interrupts only the wait: the reap is shielded.

        with pytest.raises(asyncio.CancelledError):
            await task
        assert child.kill_calls == 1
        assert not child.reaped

        child.release_reap.set()
        while subprocesses._PENDING_REAPS:
            await asyncio.sleep(0)

        assert child.reaped
        assert loop_errors == []


@pytest.mark.asyncio
async def test_child_exiting_concurrently_is_still_reaped(monkeypatch):
    child = _Child(stall=True, exits_on_its_own=True)
    _install_child(monkeypatch, child)

    task = asyncio.ensure_future(_run())
    await child.in_communicate.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert child.kill_calls == 1
    assert child.reaped


@pytest.mark.asyncio
async def test_its_own_timeout_kills_reaps_and_raises_timeout(monkeypatch):
    child = _Child(stall=True)
    _install_child(monkeypatch, child)

    async with asyncio.timeout(_GUARD_SECONDS):
        with pytest.raises(TimeoutError):
            await _run(timeout=0.01)

    assert child.kill_calls == 1
    assert child.reaped


@pytest.mark.asyncio
async def test_an_error_in_communicate_kills_reaps_and_propagates(monkeypatch):
    child = _Child()

    async def broken(payload: bytes) -> tuple[bytes, bytes]:
        del payload
        raise BrokenPipeError("child closed stdin")

    child.communicate = broken  # type: ignore[method-assign]
    _install_child(monkeypatch, child)

    with pytest.raises(BrokenPipeError):
        await _run()

    assert child.kill_calls == 1
    assert child.reaped


@pytest.mark.asyncio
async def test_stalled_reap_is_bounded_and_its_late_failure_logged_not_leaked(
    monkeypatch,
):
    child = _Child(stall=True, stall_reap=True, reap_error=OSError("reap failed"))
    _install_child(monkeypatch, child)
    monkeypatch.setattr(config, "subprocess_cleanup_timeout_seconds", 0.01)

    task = asyncio.ensure_future(_run())
    await child.in_communicate.wait()
    logs = _LogRecorder()
    monkeypatch.setattr(subprocesses, "logger", logs)
    with _loop_errors() as loop_errors:
        task.cancel()
        async with asyncio.timeout(_GUARD_SECONDS):
            with pytest.raises(asyncio.CancelledError):
                await task
        # The caller was released with the reap still pending.
        assert len(subprocesses._PENDING_REAPS) == 1

        child.release_reap.set()
        while subprocesses._PENDING_REAPS:
            await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)

    events = {event: fields for _, event, fields in logs.events}
    assert events["subprocess_reap_pending"]["pid"] == _Child.pid
    assert events["subprocess_reap_failed"]["pid"] == _Child.pid
    assert loop_errors == []


@pytest.mark.asyncio
async def test_real_child_is_killed_and_reaped_on_cancellation(monkeypatch):
    """The same spawn and cleanup path, with a harmless real child."""
    started: list[asyncio.subprocess.Process] = []
    real_spawn = asyncio.create_subprocess_exec

    async def spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await real_spawn(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(subprocesses.asyncio, "create_subprocess_exec", spawn)
    task = asyncio.ensure_future(
        subprocesses.run_program("/bin/sleep", "30", input=b"", timeout=30.0)
    )
    try:
        while not started:
            await asyncio.sleep(0)
        await asyncio.sleep(0)  # Let it enter communicate().
        task.cancel()

        async with asyncio.timeout(_GUARD_SECONDS):
            with pytest.raises(asyncio.CancelledError):
                await task

        (process,) = started
        assert process.returncode == -signal.SIGKILL
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)
    finally:
        for process in started:
            if process.returncode is None:
                process.kill()
                await process.wait()
