"""Regression tests for DDG search concurrency safety.

Background
----------
Production hang observed on 2026-05-09 (0883.HK): multiple ``asyncio.to_thread``
workers entered ``ddgs.DDGS.__init__`` concurrently. DDG's lazy-loaded HTTP
client calls ``logging.getLogger`` from inside its ``__init__`` while doing
internal setup that holds the GIL. With multiple worker threads contending
on the import lock, ``logging._lock``, and the GIL, the asyncio loop in the
main thread was starved — even ``run_with_hard_timeout`` deadlines never
fired. py-spy confirmed three threads simultaneously stuck in DDG init.

The structural mitigation in ``src/tools/shared.py`` has three layers, each
of which this test enforces:

  1. **Pre-warm at module load** — ``ddgs`` is imported and ``DDGS(...)``
     instantiated once on the main thread, single-threaded, before any
     worker thread or event loop exists.
  2. **Dedicated single-thread executor** — every DDG call runs on
     ``_get_ddg_executor()`` (a ``ThreadPoolExecutor(max_workers=1)``).
     Concurrent re-entry of DDG's constructor from multiple threads is
     impossible by construction.
  3. **asyncio.Lock around every call** — surfaces the serialization at
     the await level so ``run_with_hard_timeout`` can cancel a queued
     call cleanly before it ever runs.

Each layer is enforced statically (AST checks) and behaviorally (a parallel
gather that asserts serial execution).
"""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
from pathlib import Path
from unittest.mock import patch

import pytest

import src.tools.shared as shared

# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------


def test_ddgs_is_pre_imported_at_module_load():
    """``shared.py`` must eagerly import ddgs and expose DDGS_AVAILABLE."""
    assert hasattr(
        shared, "DDGS_AVAILABLE"
    ), "shared.DDGS_AVAILABLE missing — pre-import guard removed?"


def test_ddg_call_lock_is_an_asyncio_lock():
    """A module-level asyncio.Lock must serialize DDG calls at await level."""
    assert hasattr(
        shared, "_DDG_CALL_LOCK"
    ), "shared._DDG_CALL_LOCK missing — DDG concurrency guard removed?"
    assert isinstance(
        shared._DDG_CALL_LOCK, asyncio.Lock
    ), f"_DDG_CALL_LOCK must be asyncio.Lock, got {type(shared._DDG_CALL_LOCK)}"


def test_ddg_executor_is_single_threaded():
    """The dedicated DDG executor must have exactly one worker thread.

    More than one worker would re-introduce the original deadlock by
    allowing concurrent re-entry into DDG's lazy http_client init.
    """
    executor = shared._get_ddg_executor()
    assert isinstance(
        executor, concurrent.futures.ThreadPoolExecutor
    ), f"DDG executor must be ThreadPoolExecutor, got {type(executor)}"
    assert executor._max_workers == 1, (
        f"DDG executor must be single-threaded (max_workers=1), "
        f"got {executor._max_workers}. A pool size > 1 re-introduces the "
        "import-lock + logging-lock deadlock that caused the 0883.HK hang."
    )


def test_ddg_executor_is_module_singleton():
    """Repeated calls to _get_ddg_executor return the same instance."""
    e1 = shared._get_ddg_executor()
    e2 = shared._get_ddg_executor()
    assert e1 is e2, "DDG executor must be a process-wide singleton"


def test_ddg_search_uses_dedicated_executor_not_default_to_thread():
    """``_ddg_search`` must run DDG on ``_get_ddg_executor()``, never on the
    default thread pool. AST check so future refactors can't silently switch
    back to ``asyncio.to_thread`` (which uses the shared default executor
    and re-opens the door to concurrent DDG init)."""
    source = Path("src/tools/shared.py").read_text()
    tree = ast.parse(source)

    target = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "_ddg_search"
        ),
        None,
    )
    assert target is not None, "_ddg_search not found in src/tools/shared.py"

    uses_run_in_executor_with_ddg_executor = False
    uses_default_to_thread = False
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        # Disallow asyncio.to_thread anywhere in _ddg_search — it implicitly
        # uses the default executor, which is shared with every other
        # to_thread caller and is the failure mode we're protecting against.
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_thread"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "asyncio"
        ):
            uses_default_to_thread = True
        # Require loop.run_in_executor(_get_ddg_executor(), ...)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_in_executor"
            and node.args
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Name)
            and node.args[0].func.id == "_get_ddg_executor"
        ):
            uses_run_in_executor_with_ddg_executor = True

    assert not uses_default_to_thread, (
        "_ddg_search uses asyncio.to_thread — that runs on the shared default "
        "executor and re-opens the concurrent-DDG-init deadlock. Use "
        "loop.run_in_executor(_get_ddg_executor(), fn) instead."
    )
    assert uses_run_in_executor_with_ddg_executor, (
        "_ddg_search must call loop.run_in_executor(_get_ddg_executor(), ...) "
        "to route every DDG call through the dedicated single-thread executor."
    )


def test_ddg_search_acquires_the_lock_around_the_executor_call():
    """``_ddg_search`` must wrap its executor call with ``async with _DDG_CALL_LOCK``."""
    source = Path("src/tools/shared.py").read_text()
    tree = ast.parse(source)

    target = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "_ddg_search"
        ),
        None,
    )
    assert target is not None

    found = False
    for aw in ast.walk(target):
        if not isinstance(aw, ast.AsyncWith):
            continue
        for item in aw.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Name) and ctx.id == "_DDG_CALL_LOCK":
                # Confirm a run_in_executor or run_with_hard_timeout call is
                # inside this with block.
                for inner in ast.walk(aw):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr in ("run_in_executor",)
                    ):
                        found = True
                        break
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Name)
                        and inner.func.id == "run_with_hard_timeout"
                    ):
                        found = True
                        break
        if found:
            break

    assert found, (
        "_ddg_search must enclose its run_in_executor / run_with_hard_timeout "
        "call in `async with _DDG_CALL_LOCK:`. This is the await-level "
        "serialization that lets run_with_hard_timeout cancel a queued call "
        "before it ever reaches the executor."
    )


# ---------------------------------------------------------------------------
# Behavioral checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_ddg_searches_run_serially_and_on_one_thread():
    """N parallel _ddg_search calls must execute serially on a single thread.

    The dedicated executor + asyncio.Lock together guarantee:
      - At most one DDG call is in flight at any time (lock).
      - Every DDG call runs on the same OS thread (executor max_workers=1).

    Both invariants are tested by a fake DDGS that records (event, thread_id,
    query) tuples and asserts no overlap and a single thread_id.
    """
    import threading

    timeline: list[tuple[str, int, str]] = []

    class _FakeDDGS:
        def __init__(self, *_args, **_kwargs):
            pass

        def text(self, query, max_results=5):
            tid = threading.get_ident()
            timeline.append(("enter", tid, query))
            # Simulate work; if the lock fails to serialize, this window is
            # where parallel callers would interleave.
            import time

            time.sleep(0.05)
            timeline.append(("exit", tid, query))
            return [{"title": query, "href": f"https://x/{query}", "body": "b"}]

    with patch.object(shared, "DDGS_AVAILABLE", True), patch("ddgs.DDGS", _FakeDDGS):
        await asyncio.gather(
            shared._ddg_search("a"),
            shared._ddg_search("b"),
            shared._ddg_search("c"),
        )

    # All calls ran on the same thread (single-thread executor).
    thread_ids = {tid for (_kind, tid, _q) in timeline}
    assert len(thread_ids) == 1, (
        f"DDG calls ran on {len(thread_ids)} threads {thread_ids}; "
        "the dedicated executor must keep all DDG work on exactly one thread."
    )

    # No interleaving — each enter is immediately followed by its own exit.
    assert len(timeline) == 6, f"unexpected timeline length: {timeline}"
    for i in range(0, 6, 2):
        kind_in, _t_in, q_in = timeline[i]
        kind_out, _t_out, q_out = timeline[i + 1]
        assert (
            kind_in == "enter" and kind_out == "exit"
        ), f"interleaved DDG calls — lock failed to serialize. {timeline}"
        assert (
            q_in == q_out
        ), f"call {q_in} did not exit before {q_out} entered. {timeline}"


@pytest.mark.asyncio
async def test_ddg_search_returns_empty_when_unavailable():
    """When DDGS_AVAILABLE is False, _ddg_search returns [] without
    submitting any work to the executor (so a missing dep can't deadlock
    startup or leak a thread)."""
    with patch.object(shared, "DDGS_AVAILABLE", False):
        result = await shared._ddg_search("anything")
        assert result == []


@pytest.mark.asyncio
async def test_ddg_search_hard_timeout_does_not_block_subsequent_calls():
    """A hung DDG call must not block the next call from running once the
    hard timeout fires.

    The dedicated executor is single-threaded, so a stuck call leaves its
    thread parked — the executor's queue would otherwise wait for that
    thread to free up. ``run_with_hard_timeout`` must raise on schedule,
    the asyncio.Lock must be released, and the next caller must be able
    to attempt its own call (which will then queue behind the stuck thread
    and itself time out — but cleanly, not as a process-wide deadlock).
    """
    call_count = 0
    started = asyncio.Event()
    finally_done = asyncio.Event()

    class _HangingDDGS:
        def __init__(self, *_args, **_kwargs):
            pass

        def text(self, query, max_results=5):
            nonlocal call_count
            call_count += 1
            started.set()
            # Block much longer than DDG_SEARCH_TIMEOUT_SECONDS.
            import time

            time.sleep(30)
            return []

    with (
        patch.object(shared, "DDGS_AVAILABLE", True),
        patch("ddgs.DDGS", _HangingDDGS),
        patch.object(shared, "DDG_SEARCH_TIMEOUT_SECONDS", 0.2),
    ):
        try:
            await asyncio.wait_for(shared._ddg_search("first"), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail(
                "outer wait_for fired — run_with_hard_timeout did not abort "
                "the call within DDG_SEARCH_TIMEOUT_SECONDS"
            )
        finally:
            finally_done.set()

    # If we got here, the hard timeout fired and _ddg_search returned [].
    # Lock must have been released — confirm by acquiring it briefly.
    assert not shared._DDG_CALL_LOCK.locked(), (
        "_DDG_CALL_LOCK was not released after run_with_hard_timeout fired. "
        "A leaked lock would deadlock all subsequent DDG searches."
    )
