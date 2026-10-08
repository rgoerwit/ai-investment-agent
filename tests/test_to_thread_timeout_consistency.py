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

Both checks resolve every name a module uses for asyncio — ``import asyncio as
aio``, ``from asyncio import to_thread``, and injected parameters such as
``asyncio_module: Any = asyncio`` — not just the literal ``asyncio.`` prefix.

When you legitimately need an exception (e.g. the call is protected by an
*outer* wrapper one or more frames away), add an entry to
``OUTER_WRAPPED_ALLOWLIST`` with a short justification.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

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
    # The yahooquery builder lambda (asyncio reached via the injectable
    # ``asyncio_module`` parameter) is awaited inside run_with_hard_timeout in
    # fetch_all_sources_parallel's _run_one.
    "src/data/source_fetchers.py:288": (
        "builder lambda awaited under run_with_hard_timeout in _run_one"
    ),
    # IBKR services: the ib_async client has its own per-request timeouts and
    # an outer connection-level timeout; sync wrappers are short and CPU-bound
    # rather than blocking on a remote socket read with no library timeout.
    "src/ibkr/account_service.py:65": "ib_async has its own request timeout",
    "src/ibkr/account_service.py:68": "ib_async has its own request timeout",
    "src/ibkr/account_service.py:76": "ib_async has its own request timeout",
    "src/ibkr/portfolio_data_service.py:94": "ib_async has its own request timeout",
    "src/ibkr/portfolio_data_service.py:106": "ib_async has its own request timeout",
    "src/ibkr/portfolio_data_service.py:118": "ib_async has its own request timeout",
    "src/ibkr/portfolio_data_service.py:129": "ib_async has its own request timeout",
    "src/ibkr/portfolio_data_service.py:158": "ib_async has its own request timeout",
    "src/ibkr/security_data_service.py:118": (
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
    "src/data/filings/edinet_fetcher.py:163": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
    "src/data/filings/edinet_fetcher.py:169": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
    "src/data/filings/edinet_fetcher.py:219": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
    "src/data/filings/edinet_fetcher.py:242": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
    "src/data/filings/edinet_fetcher.py:291": (
        "wrapped by run_with_hard_timeout in FilingRegistry.fetch"
    ),
}


@dataclass(frozen=True)
class _AsyncioNames:
    """Every spelling under which one module reaches ``asyncio`` primitives.

    Matching only the literal ``asyncio.to_thread`` misses the injectable
    ``asyncio_module: Any = asyncio`` parameter pattern used in src/data/ — which
    is exactly how ``gap_fill.py`` hid a ``wait_for(to_thread(...))`` from this
    guard. Resolution is module-wide rather than scope-precise: a name bound to
    asyncio anywhere in the module counts everywhere in it. That over-approximates
    only for a module that reuses one name for two different objects.
    """

    module_aliases: frozenset[str]
    to_thread_names: frozenset[str]
    wait_for_names: frozenset[str]

    def _is_primitive(self, node: ast.AST, attr: str, bare: frozenset[str]) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in bare
        return (
            isinstance(func, ast.Attribute)
            and func.attr == attr
            and isinstance(func.value, ast.Name)
            and func.value.id in self.module_aliases
        )

    def is_to_thread(self, node: ast.AST) -> bool:
        return self._is_primitive(node, "to_thread", self.to_thread_names)

    def is_wait_for(self, node: ast.AST) -> bool:
        return self._is_primitive(node, "wait_for", self.wait_for_names)


def _resolve_asyncio_names(tree: ast.AST) -> _AsyncioNames:
    modules = {"asyncio"}
    to_thread: set[str] = set()
    wait_for: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "asyncio":
                    modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "asyncio":
            for alias in node.names:
                if alias.name == "to_thread":
                    to_thread.add(alias.asname or alias.name)
                elif alias.name == "wait_for":
                    wait_for.add(alias.asname or alias.name)

    def _is_module_ref(value: ast.AST | None) -> bool:
        return isinstance(value, ast.Name) and value.id in modules

    # Second pass: names re-bound to the module (``aio = asyncio``) and
    # parameters defaulting to it (``asyncio_module: Any = asyncio``).
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_module_ref(node.value):
            modules.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and _is_module_ref(node.value)
        ):
            modules.add(node.target.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            args = node.args
            positional = [*args.posonlyargs, *args.args]
            for arg, default in zip(
                positional[len(positional) - len(args.defaults) :],
                args.defaults,
                strict=True,
            ):
                if _is_module_ref(default):
                    modules.add(arg.arg)
            for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                if _is_module_ref(kw_default):
                    modules.add(arg.arg)
    return _AsyncioNames(frozenset(modules), frozenset(to_thread), frozenset(wait_for))


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


def _attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]


def _enclosing_timeout_call(
    to_thread_node: ast.Call, names: _AsyncioNames
) -> str | None:
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
                if names.is_wait_for(parent):
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


def _scan_to_thread_sites(source: str) -> list[tuple[int, str | None]]:
    """Return ``(lineno, enclosing timeout primitive or None)`` per to_thread call."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names = _resolve_asyncio_names(tree)
    _attach_parents(tree)
    return [
        (node.lineno, _enclosing_timeout_call(node, names))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and names.is_to_thread(node)
    ]


def _unbounded_to_thread_lines(source: str) -> list[int]:
    return [line for line, wrapper in _scan_to_thread_sites(source) if wrapper is None]


def _wait_for_to_thread_lines(source: str) -> list[int]:
    return [
        line
        for line, wrapper in _scan_to_thread_sites(source)
        if wrapper == "asyncio.wait_for"
    ]


def _src_files() -> list[tuple[str, str]]:
    return [
        (py_file.as_posix(), py_file.read_text())
        for py_file in sorted(Path("src").rglob("*.py"))
    ]


def test_every_to_thread_has_a_hard_timeout():
    """Every asyncio.to_thread call must be enclosed by a timeout primitive.

    Canonical: ``run_with_hard_timeout(asyncio.to_thread(...), timeout=...,
    label=...)``. ``asyncio.wait_for`` is also accepted but is NOT a true
    timeout for blocking C-level socket reads — see src/async_utils.py.
    Prefer ``run_with_hard_timeout`` for any to_thread wrapping a sync HTTP
    or network call.
    """
    violations: list[str] = []

    for rel, source in _src_files():
        for lineno in _unbounded_to_thread_lines(source):
            location = f"{rel}:{lineno}"
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
        lines = {line for line, _wrapper in _scan_to_thread_sites(path.read_text())}
        if lineno not in lines:
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
    should be re-written as ``run_with_hard_timeout(asyncio.to_thread(...))``
    — whatever name asyncio is reached through.
    """
    violations = [
        f"{rel}:{lineno}"
        for rel, source in _src_files()
        for lineno in _wait_for_to_thread_lines(source)
    ]

    assert not violations, (
        "asyncio.wait_for(asyncio.to_thread(...)) detected — this does NOT "
        "reliably bound blocking I/O (see src/async_utils.py docstring). "
        "Replace with run_with_hard_timeout. Sites:\n  - " + "\n  - ".join(violations)
    )


# ---------------------------------------------------------------------------
# The scanner itself, proven against planted regressions
# ---------------------------------------------------------------------------

# The exact shape gap_fill.py carried until the fix: asyncio injected as a
# parameter default, so neither primitive was spelled ``asyncio.<name>``.
_INJECTED_ALIAS_FIXTURE = """
import asyncio
from typing import Any

async def lookup(ticker_obj, *, asyncio_module: Any = asyncio):
    info = await asyncio_module.wait_for(
        asyncio_module.to_thread(lambda: ticker_obj.info),
        timeout=5,
    )
    return info
"""


@pytest.mark.parametrize(
    ("label", "source", "expected_line"),
    [
        ("injected parameter default", _INJECTED_ALIAS_FIXTURE, 7),
        (
            "import asyncio as aio",
            "import asyncio as aio\n"
            "async def f(g):\n"
            "    return await aio.wait_for(aio.to_thread(g), timeout=1)\n",
            3,
        ),
        (
            "from asyncio import names",
            "from asyncio import to_thread, wait_for\n"
            "async def f(g):\n"
            "    return await wait_for(to_thread(g), timeout=1)\n",
            3,
        ),
        (
            "keyword-only default",
            "import asyncio\n"
            "async def f(g, *, aio=asyncio):\n"
            "    return await aio.wait_for(aio.to_thread(g), timeout=1)\n",
            3,
        ),
        (
            "module-level rebinding",
            "import asyncio\n"
            "aio = asyncio\n"
            "async def f(g):\n"
            "    return await aio.wait_for(aio.to_thread(g), timeout=1)\n",
            4,
        ),
    ],
)
def test_scanner_flags_aliased_wait_for_to_thread(
    label: str, source: str, expected_line: int
) -> None:
    assert _wait_for_to_thread_lines(source) == [expected_line], label


def test_scanner_accepts_aliased_hard_timeout_and_flags_bare_alias() -> None:
    bounded = (
        "import asyncio\n"
        "from src.async_utils import run_with_hard_timeout\n"
        "async def f(g, asyncio_module=asyncio):\n"
        "    return await run_with_hard_timeout(\n"
        "        asyncio_module.to_thread(g), timeout=1, label='x'\n"
        "    )\n"
    )
    unbounded = (
        "import asyncio\n"
        "async def f(g, asyncio_module=asyncio):\n"
        "    return await asyncio_module.to_thread(g)\n"
    )
    assert _wait_for_to_thread_lines(bounded) == []
    assert _unbounded_to_thread_lines(bounded) == []
    assert _unbounded_to_thread_lines(unbounded) == [3]


def test_scanner_ignores_unrelated_to_thread_receivers() -> None:
    """Only names resolved to asyncio count — no false positive on other objects."""
    source = (
        "import anyio\n"
        "async def f(g, pool):\n"
        "    await pool.wait_for(pool.to_thread(g))\n"
        "    await anyio.to_thread.run_sync(g)\n"
    )
    assert _scan_to_thread_sites(source) == []
