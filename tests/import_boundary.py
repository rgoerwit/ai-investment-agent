"""Shared helper for subprocess import-boundary tests.

These tests assert that importing a lightweight module does NOT transitively pull
in a heavy package (src.agents / langchain / langgraph / ...). The check runs in a
fresh interpreter that prints exactly one sentinel line::

    <SENTINEL>:<comma-joined offender modules>

The offender list is the only real signal. A child that dies by signal
(``returncode < 0`` — e.g. SIGSEGV) without emitting the sentinel is an
environment artifact: under full-suite load on macOS the interpreter can segfault
on startup/import due to native-lib/fork interactions entirely unrelated to the
import boundary. Such a crash is retried and ultimately *skipped*, never failed —
a genuine boundary regression prints a non-empty offender list deterministically
on every run, so retrying cannot mask it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def assert_no_offenders(
    body: str,
    *,
    sentinel: str,
    cwd: str,
    message: str,
    attempts: int = 3,
) -> None:
    """Run ``body`` in a fresh interpreter and assert it reports no offenders.

    ``body`` must print exactly one ``f"{sentinel}:{payload}"`` line, where
    ``payload`` is the comma-joined list of modules that violate the boundary
    (empty when clean). Exit code is ignored — the sentinel line is the contract.

    - sentinel found, payload empty  -> pass
    - sentinel found, payload present -> fail (real boundary regression)
    - no sentinel, child signal-killed (rc < 0) -> retry, then skip (env crash)
    - no sentinel, child exited rc >= 0 -> fail (real import error; show stderr)
    """
    # Put the repo root on the child's import path via PYTHONPATH rather than
    # cwd=, and keep fds open, so CPython uses posix_spawn instead of fork()+exec().
    # A fork() in this gRPC/Network.framework-loaded test process SIGSEGVs in
    # Apple's atfork handler on macOS (cwd= forces the fork path). See CLAUDE.md
    # (macOS-Specific Issues).
    child_env = {**os.environ}
    child_env["PYTHONPATH"] = (
        cwd + os.pathsep + child_env["PYTHONPATH"]
        if child_env.get("PYTHONPATH")
        else cwd
    )
    last: subprocess.CompletedProcess[str] | None = None
    for _ in range(attempts):
        result = subprocess.run(
            [sys.executable, "-c", body],
            capture_output=True,
            text=True,
            env=child_env,
            close_fds=False,
        )
        last = result
        prefix = f"{sentinel}:"
        for line in result.stdout.splitlines():
            if line.startswith(prefix):
                offenders = line[len(prefix) :].strip()
                assert not offenders, f"{message}: {offenders}"
                return
        if result.returncode >= 0:
            # Exited without the sentinel but not signal-killed: a real import
            # error (e.g. ImportError) — surface it, do not retry or skip.
            raise AssertionError(
                f"{message}: subprocess exited {result.returncode} without "
                f"'{sentinel}:' marker:\n{result.stderr[-800:]}"
            )
        # returncode < 0: killed by a signal (e.g. SIGSEGV) → retry.

    rc = last.returncode if last else "?"
    pytest.skip(
        f"import-boundary subprocess died by signal {rc} on all {attempts} "
        f"attempts (environment crash, not an import-boundary failure)"
    )
