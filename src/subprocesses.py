"""Child processes: spawned on the ``posix_spawn`` path, cleaned up on every exit.

Every subprocess under ``src/`` starts here (``tests/architecture/test_subprocess_spawns.py``
enforces it). CPython uses ``posix_spawn`` instead of ``fork()`` + ``exec()``
only for an absolute executable, ``close_fds=False`` and no ``cwd``,
``start_new_session`` or ``preexec_fn``. The analysis process is threaded and
has Apple's Network.framework loaded (gRPC, Gemini); ``fork()`` runs that
framework's atfork handler, which can crash the child before ``exec``.
``close_fds=False`` hands the child nothing it should not have: Python opens
descriptors non-inheritable (PEP 446).
"""

from __future__ import annotations

import asyncio
import functools
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import structlog

from src.config import config
from src.error_safety import summarize_exception

logger = structlog.get_logger(__name__)

# Detached reapers of killed children, held so one that outlives its bounded
# wait is neither garbage-collected nor left with an unobserved exception
# (_reap_finished logs it).
_PENDING_REAPS: set[asyncio.Future[Any]] = set()


@dataclass(frozen=True)
class ProgramOutput:
    """A finished child's exit status and standard output."""

    returncode: int
    stdout: bytes


def resolve_program(name: str) -> str | None:
    """The absolute path of ``name`` on ``PATH``, or None when it is not installed."""
    found = shutil.which(name)
    return None if found is None else os.path.abspath(found)


def _require_absolute(executable: str) -> None:
    # A bare name sends CPython down the fork path.
    if not os.path.isabs(executable):
        raise ValueError(f"executable must be an absolute path: {executable!r}")


async def run_program(
    executable: str, *args: str, input: bytes, timeout: float
) -> ProgramOutput:
    """Run a child to completion within ``timeout``, feeding it ``input``.

    Any abnormal exit -- ``TimeoutError``, an error, the caller's cancellation
    -- kills and reaps the child, then re-raises the original exception.
    """
    _require_absolute(executable)
    process = await asyncio.create_subprocess_exec(
        executable,
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        close_fds=False,
    )
    stdout = await _communicate_bounded(process, input, timeout=timeout)
    # Already exited (communicate() waited for it): this only reads the status.
    returncode = await process.wait()
    return ProgramOutput(returncode, stdout)


def run_program_sync(
    executable: str, *args: str, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run a child to completion within ``timeout``, capturing its text output.

    Raises ``CalledProcessError`` on a non-zero exit and ``TimeoutExpired``
    after killing and reaping a child that overran.
    """
    _require_absolute(executable)
    return subprocess.run(
        [executable, *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
        close_fds=False,
    )


def _reap_finished(pid: int, reap: asyncio.Future[Any]) -> None:
    """Done-callback of a reaper: forget it, and log (never leak) its failure."""
    _PENDING_REAPS.discard(reap)
    if reap.cancelled():
        logger.warning("subprocess_reap_cancelled", pid=pid)
        return
    exc = reap.exception()
    if exc is not None:
        logger.warning(
            "subprocess_reap_failed",
            pid=pid,
            **summarize_exception(exc, operation="subprocess_reap"),
        )


async def _kill_and_reap(process: asyncio.subprocess.Process) -> None:
    """Kill a child and reap it, the wait bounded and shielded.

    Runs from an abnormal exit (timeout, error, cancellation). The reap is a
    separate task behind ``asyncio.shield``, so a second cancellation of the
    caller interrupts only the wait: the reap still finishes and its outcome
    is logged by ``_reap_finished``. Past SUBPROCESS_CLEANUP_TIMEOUT_SECONDS
    the caller stops waiting and the reap continues detached.
    """
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass  # It exited between the check and the kill; reap it anyway.
    reap = asyncio.ensure_future(process.wait())
    _PENDING_REAPS.add(reap)
    reap.add_done_callback(functools.partial(_reap_finished, process.pid))
    try:
        async with asyncio.timeout(config.subprocess_cleanup_timeout_seconds):
            await asyncio.shield(reap)
    except TimeoutError:
        logger.warning(
            "subprocess_reap_pending",
            pid=process.pid,
            timeout_seconds=config.subprocess_cleanup_timeout_seconds,
        )
    except Exception:
        pass  # The reap failed; _reap_finished has logged it.


async def _communicate_bounded(
    process: asyncio.subprocess.Process, payload: bytes, *, timeout: float
) -> bytes:
    """``communicate()`` within ``timeout``; any abnormal exit kills and reaps."""
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(payload), timeout=timeout
        )
    except BaseException:
        await _kill_and_reap(process)
        raise
    return stdout
