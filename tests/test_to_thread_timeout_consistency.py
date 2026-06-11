"""
Enforce that every ``asyncio.to_thread(...)`` call in src/ is bounded by a
hard wall-clock timeout.

Why this exists
---------------
Most blocking I/O hangs in this codebase trace to ``await asyncio.to_thread(
blocking_call)`` where the inner C-level socket read cannot be cancelled.
``asyncio.wait_for`` does not solve this — see ``src/async_utils.py`` for the
detailed explanation. ``run_with_hard_timeout`` was written specifically for
this case and is the canonical bound.

The test fails when:
  * A new ``asyncio.to_thread(...)`` site is added without ``run_with_hard_timeout``
    or ``asyncio.wait_for`` directly enclosing it.
  * A site is moved off the canonical ``run_with_hard_timeout`` without an
    explicit allowlist entry justifying why.

When you legitimately need an exception (e.g. the call is protected by an
*outer* wrapper one or more frames away), add an entry to
``OUTER_WRAPPED_ALLOWLIST`` with a short justification.
"""

from __future__ import annotations

import ast
from pathlib import Path

# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------
#
# Each entry: "src/path/file.py:LINE" -> "why this is safe without an inline timeout"
#
# Use sparingly. The default expectation is that every ``asyncio.to_thread``
# call is enclosed in ``run_with_hard_timeout(...)`` (preferred) or
# ``asyncio.wait_for(...)`` (acceptable for awaiting a coroutine you control
# that honors cancellation, NOT for to_thread of blocking sync I/O).
#
# Entries here are checked at the listed line; if the line shifts, the test
# will surface the new line and you can update.
OUTER_WRAPPED_ALLOWLIST: dict[str, str] = {
    # source_fetchers.py: each builder is wrapped by run_with_hard_timeout in
    # fetch_all_sources_parallel (src/data/source_fetchers.py:263) under
    # PER_SOURCE_TIMEOUT=15. Inner to_thread calls inherit that bound.
    "src/data/source_fetchers.py:33": (
        "wrapped by run_with_hard_timeout in fetch_all_sources_parallel"
    ),
    "src/data/source_fetchers.py:54": (
        "wrapped by run_with_hard_timeout in fetch_all_sources_parallel"
    ),
    "src/data/source_fetchers.py:66": (
        "wrapped by run_with_hard_timeout in fetch_all_sources_parallel"
    ),
    # IBKR services: the ib_async client has its own per-request timeouts and
    # an outer connection-level timeout; sync wrappers are short and CPU-bound
    # rather than blocking on a remote socket read with no library timeout.
    "src/ibkr/account_service.py:50": "ib_async has its own request timeout",
    "src/ibkr/account_service.py:53": "ib_async has its own request timeout",
    "src/ibkr/account_service.py:61": "ib_async has its own request timeout",
    "src/ibkr/portfolio_data_service.py:76": "ib_async has its own request timeout",
    "src/ibkr/portfolio_data_service.py:88": "ib_async has its own request timeout",
    "src/ibkr/portfolio_data_service.py:100": "ib_async has its own request timeout",
    "src/ibkr/portfolio_data_service.py:111": "ib_async has its own request timeout",
    "src/ibkr/portfolio_data_service.py:140": "ib_async has its own request timeout",
    "src/ibkr/security_data_service.py:76": (
        "ib_async + yfinance probe; wrapped by caller-side bounds"
    ),
    # EDINET fetcher: wrapped by run_with_hard_timeout in
    # FilingRegistry.fetch (src/data/filings/registry.py); all internal
    # to_thread calls inherit the 30s outer timeout.
    "src/data/filings/edinet_fetcher.py:77": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
    "src/data/filings/edinet_fetcher.py:92": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
    "src/data/filings/edinet_fetcher.py:98": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
    "src/data/filings/edinet_fetcher.py:117": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
    "src/data/filings/edinet_fetcher.py:159": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
    "src/data/filings/edinet_fetcher.py:165": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
    "src/data/filings/edinet_fetcher.py:215": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
    "src/data/filings/edinet_fetcher.py:238": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
    "src/data/filings/edinet_fetcher.py:287": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
}


def _is_to_thread_call(node: ast.AST) -> bool:
    """Return True if *node* is the ``asyncio.to_thread(...)`` call expression."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to_thread"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "asyncio"
    )


def _is_run_with_hard_timeout_call(node: ast.AST) -> bool:
    """Return True if *node* is a ``run_with_hard_timeout(...)`` call."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "run_with_hard_timeout":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "run_with_hard_timeout":
        return True
    return False


def _is_wait_for_call(node: ast.AST) -> bool:
    """Return True if *node* is an ``asyncio.wait_for(...)`` call."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "wait_for"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "asyncio"
    )


def _attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]


def _enclosing_timeout_call(to_thread_node: ast.Call) -> str | None:
    """Walk up the parent chain. Return the timeout-call name, or None.

    A ``to_thread`` call is considered properly enclosed when the *first*
    positional argument of an ancestor ``run_with_hard_timeout(...)`` or
    ``asyncio.wait_for(...)`` call resolves to it. Any other ancestry
    (a ``gather``, an ``await`` alone, an assignment, etc.) does not count.
    """
    node: ast.AST = to_thread_node
    while True:
        parent = getattr(node, "parent", None)
        if parent is None:
            return None
        if isinstance(parent, ast.Call):
            if parent.args and parent.args[0] is node:
                if _is_run_with_hard_timeout_call(parent):
                    return "run_with_hard_timeout"
                if _is_wait_for_call(parent):
                    return "asyncio.wait_for"
            # Other surrounding calls (gather, ensure_future, etc.) do not
            # provide a wall-clock bound. Stop walking — wrapping happens at
            # the immediate enclosing position, not several frames up.
            return None
        if isinstance(
            parent,
            ast.Await
            | ast.Expr
            | ast.Assign
            | ast.AnnAssign
            | ast.AugAssign
            | ast.Return
            | ast.Lambda
            | ast.Yield
            | ast.YieldFrom,
        ):
            # Walk up: the to_thread node may be wrapped in await / assignment
            # / etc.; we still want to find an enclosing timeout call above.
            node = parent
            continue
        # Anything else (FunctionDef, ClassDef, Module, etc.): stop. The
        # to_thread is not in a recognized "passed-as-first-arg" shape.
        return None


def test_every_to_thread_has_a_hard_timeout():
    """Every asyncio.to_thread call must be enclosed by a timeout primitive.

    Canonical: ``run_with_hard_timeout(asyncio.to_thread(...), timeout=...,
    label=...)``. ``asyncio.wait_for`` is also accepted but is NOT a true
    timeout for blocking C-level socket reads — see src/async_utils.py.
    Prefer ``run_with_hard_timeout`` for any to_thread wrapping a sync HTTP
    or network call.
    """
    src_root = Path("src")
    violations: list[str] = []

    for py_file in sorted(src_root.rglob("*.py")):
        rel = py_file.as_posix()
        source = py_file.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        _attach_parents(tree)
        for node in ast.walk(tree):
            if not _is_to_thread_call(node):
                continue
            location = f"{rel}:{node.lineno}"
            if _enclosing_timeout_call(node) is not None:
                continue
            if location in OUTER_WRAPPED_ALLOWLIST:
                continue
            violations.append(location)

    assert not violations, (
        "Unbounded asyncio.to_thread calls found — these can hang the pipeline "
        "indefinitely if the underlying I/O blocks (yfinance, yahooquery, "
        "Tavily, etc. all do this). Wrap each call in run_with_hard_timeout(...)\n"
        "from src/async_utils.py, or add the line to OUTER_WRAPPED_ALLOWLIST "
        "in this test with a justification.\n\nOffending sites:\n  - "
        + "\n  - ".join(violations)
    )


def test_allowlist_entries_still_point_at_to_thread_calls():
    """Allowlist entries must reference real ``asyncio.to_thread`` lines.

    Without this check, file edits that shift line numbers leave stale
    allowlist entries that silently mask new violations on those lines.
    """
    stale: list[str] = []
    for location, _justification in OUTER_WRAPPED_ALLOWLIST.items():
        rel, _, lineno_str = location.rpartition(":")
        lineno = int(lineno_str)
        path = Path(rel)
        if not path.exists():
            stale.append(f"{location} (file does not exist)")
            continue
        source = path.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        found = False
        for node in ast.walk(tree):
            if _is_to_thread_call(node) and node.lineno == lineno:
                found = True
                break
        if not found:
            stale.append(f"{location} (no asyncio.to_thread call at this line)")

    assert not stale, (
        "OUTER_WRAPPED_ALLOWLIST entries do not match real to_thread call "
        "sites. Update line numbers or remove stale entries:\n  - "
        + "\n  - ".join(stale)
    )


def test_to_thread_blocking_io_prefers_hard_timeout():
    """``asyncio.to_thread(<blocking-network-call>)`` must use ``run_with_hard_timeout``.

    ``asyncio.wait_for`` may *appear* to bound a to_thread call, but per the
    docstring of src/async_utils.py it cannot interrupt a thread blocked in
    a C-level socket read with no library-level timeout — it waits forever
    for the cancelled task. ``run_with_hard_timeout`` is deadline-only and
    raises immediately on timeout, orphaning the unkillable thread.

    This is the test that prevents the regression that caused the 0883.HK
    overnight hang to recur. Any ``asyncio.wait_for(asyncio.to_thread(...))``
    should be re-written as ``run_with_hard_timeout(asyncio.to_thread(...))``.
    """
    src_root = Path("src")
    violations: list[str] = []

    for py_file in sorted(src_root.rglob("*.py")):
        rel = py_file.as_posix()
        source = py_file.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        _attach_parents(tree)
        for node in ast.walk(tree):
            if not _is_to_thread_call(node):
                continue
            wrapper = _enclosing_timeout_call(node)
            if wrapper == "asyncio.wait_for":
                violations.append(f"{rel}:{node.lineno}")

    assert not violations, (
        "asyncio.wait_for(asyncio.to_thread(...)) detected — this does NOT "
        "reliably bound blocking I/O (see src/async_utils.py docstring). "
        "Replace with run_with_hard_timeout. Sites:\n  - " + "\n  - ".join(violations)
    )
