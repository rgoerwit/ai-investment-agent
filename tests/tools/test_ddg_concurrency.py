"""Regression tests for DDG search concurrency safety.

Background
----------
Production hang observed on 2026-05-09 (0883.HK): multiple worker threads
entered ``ddgs.DDGS.__init__`` concurrently. DDG's lazy-loaded HTTP client
calls ``logging.getLogger`` from inside its ``__init__`` while doing internal
setup that holds the GIL; with multiple threads contending on the import lock,
``logging._lock``, and the GIL, the asyncio loop starved and even
``run_with_hard_timeout`` deadlines never fired.

The *original* fix pinned the DDG executor to a single worker. That prevented
concurrent constructor re-entry but created a second, quieter failure (2026-07
pipeline ``child_timeout`` investigation, 214150.KQ): a DDG call that hangs on
an uncancellable OS socket read orphans the single worker thread forever, so
every subsequent DDG search queues behind it, times out at
``DDG_SEARCH_TIMEOUT_SECONDS``, and returns ``[]`` — DDG fallback silently dies
for the rest of the process.

The current design (``src/tools/shared.py``) decouples the two concerns:

  1. **Pre-warm at module load** — ``ddgs`` is imported and ``DDGS(...)``
     instantiated once on the main thread before any worker/event loop exists.
  2. **Constructor-only lock** — ``_DDG_INIT_LOCK`` (a ``threading.Lock``) is
     held ONLY around ``DDGS(...)`` construction, so no two threads re-enter
     DDG's lazy HTTP-client init at once. The network ``.text()`` call runs
     OUTSIDE the lock.
  3. **Small worker pool** — ``_get_ddg_executor()`` has
     ``_DDG_EXECUTOR_MAX_WORKERS`` (>1) workers, so one hung socket read orphans
     a single worker and the others stay free. Searches run in parallel.

Each layer is enforced statically (AST checks) and behaviorally (parallel
gathers that assert overlap + hang-resilience).
"""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

import src.tools.shared as shared


@pytest.fixture(autouse=True)
def _fresh_ddg_executor():
    """Give each test a fresh DDG pool so an orphaned (hung) worker from one
    test can't deplete the shared pool for the next. The old pool's orphaned
    thread lives on harmlessly (it is what we are modelling)."""
    shared._DDG_EXECUTOR = None
    yield
    shared._DDG_EXECUTOR = None


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------


def test_ddgs_is_pre_imported_at_module_load():
    """``shared.py`` must eagerly import ddgs and expose DDGS_AVAILABLE."""
    assert hasattr(shared, "DDGS_AVAILABLE"), (
        "shared.DDGS_AVAILABLE missing — pre-import guard removed?"
    )


def test_ddg_init_lock_is_a_threading_lock():
    """Constructor safety is a ``threading.Lock`` (held in the worker thread
    around ``DDGS(...)``), NOT an asyncio.Lock serializing whole calls."""
    assert hasattr(shared, "_DDG_INIT_LOCK"), (
        "shared._DDG_INIT_LOCK missing — DDG constructor guard removed?"
    )
    # threading.Lock() returns a _thread.lock instance; assert it quacks right.
    assert hasattr(shared._DDG_INIT_LOCK, "acquire") and hasattr(
        shared._DDG_INIT_LOCK, "release"
    ), f"_DDG_INIT_LOCK must be a threading lock, got {type(shared._DDG_INIT_LOCK)}"
    # And it must NOT be an asyncio.Lock (that would serialize network I/O too).
    assert not isinstance(shared._DDG_INIT_LOCK, asyncio.Lock), (
        "_DDG_INIT_LOCK must be a threading.Lock, not an asyncio.Lock"
    )


def test_ddg_call_lock_is_gone():
    """The old whole-call asyncio serialization must be removed — keeping it
    would defeat the multi-worker pool (calls would run one-at-a-time)."""
    assert not hasattr(shared, "_DDG_CALL_LOCK"), (
        "shared._DDG_CALL_LOCK still present — the whole-call asyncio lock "
        "serializes DDG searches and re-creates the head-of-line stall the "
        "multi-worker pool is meant to remove."
    )


def test_ddg_executor_is_multi_worker():
    """The DDG executor must have >1 worker so one hung (uncancellable) socket
    read orphans a single thread instead of saturating the pool."""
    executor = shared._get_ddg_executor()
    assert isinstance(executor, concurrent.futures.ThreadPoolExecutor), (
        f"DDG executor must be ThreadPoolExecutor, got {type(executor)}"
    )
    assert executor._max_workers == shared._DDG_EXECUTOR_MAX_WORKERS
    assert executor._max_workers > 1, (
        f"DDG executor must be multi-worker, got {executor._max_workers}. "
        "A single worker is permanently saturated by the first hung socket "
        "read, silently disabling DDG fallback for the rest of the process "
        "(2026-07 214150.KQ)."
    )


def test_ddg_executor_is_module_singleton():
    """Repeated calls to _get_ddg_executor return the same instance."""
    e1 = shared._get_ddg_executor()
    e2 = shared._get_ddg_executor()
    assert e1 is e2, "DDG executor must be a process-wide singleton"


def test_ddg_search_uses_dedicated_executor_not_default_to_thread():
    """``_ddg_search`` must run DDG on ``_get_ddg_executor()``, never on the
    default thread pool (which is shared with every other to_thread caller)."""
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
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_thread"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "asyncio"
        ):
            uses_default_to_thread = True
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
        "executor and re-opens the concurrent-DDG-init deadlock."
    )
    assert uses_run_in_executor_with_ddg_executor, (
        "_ddg_search must call loop.run_in_executor(_get_ddg_executor(), ...)."
    )


# ---------------------------------------------------------------------------
# Behavioral checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_ddg_searches_run_in_parallel_on_multiple_threads():
    """N parallel _ddg_search calls must overlap and use >1 worker thread.

    The multi-worker pool + constructor-only lock let the network ``.text()``
    calls run concurrently (the old single-worker + call-lock design forced
    strict serialization on one thread).
    """
    lock = threading.Lock()
    active = 0
    max_active = 0
    thread_ids: set[int] = set()

    class _FakeDDGS:
        def __init__(self, *_args, **_kwargs):
            pass

        def text(self, query, max_results=5):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                thread_ids.add(threading.get_ident())
            import time

            time.sleep(0.1)
            with lock:
                active -= 1
            return [{"title": query, "href": f"https://x/{query}", "body": "b"}]

    with patch.object(shared, "DDGS_AVAILABLE", True), patch("ddgs.DDGS", _FakeDDGS):
        results = await asyncio.gather(
            shared._ddg_search("a"),
            shared._ddg_search("b"),
            shared._ddg_search("c"),
        )

    assert all(r for r in results), "every search should have returned results"
    assert max_active >= 2, (
        f"DDG searches did not overlap (max concurrency {max_active}); the "
        "multi-worker pool must let network calls run in parallel."
    )
    assert len(thread_ids) >= 2, (
        f"DDG calls ran on {len(thread_ids)} thread(s); a multi-worker pool "
        "should spread parallel searches across threads."
    )


@pytest.mark.asyncio
async def test_hung_search_does_not_disable_subsequent_searches():
    """The core 2026-07 regression: a call hung on an uncancellable socket read
    must NOT prevent later searches from succeeding.

    With a single worker the hung thread saturated the pool and every later
    search returned []. With the multi-worker pool a second search runs on a
    free worker and returns real results while the first is still hung.
    """
    first_started = threading.Event()

    class _FirstHangsRestWork:
        def __init__(self, *_args, **_kwargs):
            pass

        def text(self, query, max_results=5):
            import time

            if query == "hung":
                first_started.set()
                time.sleep(2.0)  # >> DDG_SEARCH_TIMEOUT_SECONDS; orphans a worker
                return []
            return [{"title": query, "href": f"https://x/{query}", "body": "b"}]

    with (
        patch.object(shared, "DDGS_AVAILABLE", True),
        patch("ddgs.DDGS", _FirstHangsRestWork),
        patch.object(shared, "DDG_SEARCH_TIMEOUT_SECONDS", 0.3),
    ):
        # Launch the hung search; wait until its worker is actually blocked.
        hung = asyncio.ensure_future(shared._ddg_search("hung"))
        await asyncio.get_event_loop().run_in_executor(None, first_started.wait, 1.0)

        # A subsequent search must run on a different worker and succeed.
        result = await shared._ddg_search("healthy")
        assert result and result[0]["title"] == "healthy", (
            "a second search returned nothing while the first was hung — the "
            "worker pool was saturated by one orphaned socket read."
        )

        # The hung search itself still times out cleanly (returns []).
        assert await hung == []


@pytest.mark.asyncio
async def test_constructor_re_entry_is_serialized_by_init_lock():
    """Even with parallel workers, no two threads may be inside ``DDGS(...)``
    construction at once — that is the invariant the 0883.HK hang violated."""
    lock = threading.Lock()
    constructing = 0
    max_constructing = 0

    class _SlowCtorDDGS:
        def __init__(self, *_args, **_kwargs):
            nonlocal constructing, max_constructing
            with lock:
                constructing += 1
                max_constructing = max(max_constructing, constructing)
            import time

            time.sleep(0.05)  # widen the constructor window
            with lock:
                constructing -= 1

        def text(self, query, max_results=5):
            return [{"title": query, "href": "https://x", "body": "b"}]

    with (
        patch.object(shared, "DDGS_AVAILABLE", True),
        patch("ddgs.DDGS", _SlowCtorDDGS),
    ):
        await asyncio.gather(*(shared._ddg_search(q) for q in "abcd"))

    assert max_constructing == 1, (
        f"{max_constructing} threads constructed DDGS concurrently; "
        "_DDG_INIT_LOCK must serialize the constructor (0883.HK re-entry hang)."
    )


@pytest.mark.asyncio
async def test_ddg_search_materializes_results_to_list():
    """``_sync_search`` must return a concrete list (defensive against a future
    ddgs version returning a lazy iterator that would evaluate off-thread)."""

    class _IterDDGS:
        def __init__(self, *_args, **_kwargs):
            pass

        def text(self, query, max_results=5):
            # A generator: if _ddg_search forwarded it un-materialized the type
            # below would not be list.
            return iter([{"title": query, "href": "https://x", "body": "b"}])

    with patch.object(shared, "DDGS_AVAILABLE", True), patch("ddgs.DDGS", _IterDDGS):
        result = await shared._ddg_search("q")

    assert isinstance(result, list) and result[0]["title"] == "q"


@pytest.mark.asyncio
async def test_ddg_search_returns_empty_when_unavailable():
    """When DDGS_AVAILABLE is False, _ddg_search returns [] without submitting
    any work to the executor."""
    with patch.object(shared, "DDGS_AVAILABLE", False):
        result = await shared._ddg_search("anything")
        assert result == []
